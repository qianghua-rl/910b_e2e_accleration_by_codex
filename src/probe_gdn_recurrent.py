"""Single-layer probe: npu_recurrent_gated_delta_rule vs HF torch_recurrent_gated_delta_rule.

Decode shape: T=1, B=1, H=48 value heads, K=V=128. Inputs are fixed-seed random tensors
(not captured activations), so this checks kernel semantics/layout, not end-to-end accuracy.
Reference: transformers torch_recurrent_gated_delta_rule (HF layout, state [B,H,K,V]).
Candidate: torch.ops.npu.npu_recurrent_gated_delta_rule, state assumed [B,H,V,K] (vLLM layout).
Usage: probe_gdn_recurrent.py [--device npu:0] [--iters 100]
"""
import argparse
import sys
import time
import traceback

import torch
import torch_npu  # noqa: F401
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_recurrent_gated_delta_rule

H, K, V = 48, 128, 128


def make_inputs(device, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(1, 1, H, K, generator=g).to(torch.bfloat16).to(device)
    k = torch.randn(1, 1, H, K, generator=g).to(torch.bfloat16).to(device)
    v = torch.randn(1, 1, H, V, generator=g).to(torch.bfloat16).to(device)
    gate = (-torch.rand(1, 1, H, generator=g).abs() * 0.5).to(device)   # g <= 0 (log decay)
    beta = torch.rand(1, 1, H, generator=g).to(torch.bfloat16).to(device)
    state = (torch.randn(1, H, K, V, generator=g) * 0.1).to(torch.bfloat16).to(device)  # HF [B,H,K,V]
    return q, k, v, gate, beta, state


def reference(q, k, v, gate, beta, state):
    out, new_state = torch_recurrent_gated_delta_rule(
        q, k, v, g=gate, beta=beta, initial_state=state.clone(),
        output_final_state=True, use_qk_l2norm_in_kernel=True)
    return out, new_state  # out [B,T,H,V], new_state [B,H,K,V]


def candidate(q, k, v, gate, beta, state_vk):
    # Pre-normalise q/k and scale q as the HF path does, then call the NPU op.
    qn = q.float() * torch.rsqrt((q.float() ** 2).sum(-1, keepdim=True) + 1e-6)
    kn = k.float() * torch.rsqrt((k.float() ** 2).sum(-1, keepdim=True) + 1e-6)
    qn = qn.to(torch.bfloat16)   # scale is applied inside the op, not here
    kn = kn.to(torch.bfloat16)
    # token-major [T,H,*] layout (vLLM convention)
    out = torch.ops.npu.npu_recurrent_gated_delta_rule(
        qn[0], kn[0], v[0], state_vk,
        beta=beta[0], scale=K ** -0.5, g=gate[0],
        actual_seq_lengths=torch.tensor([1], dtype=torch.int32, device=q.device),
        ssm_state_indices=torch.tensor([0], dtype=torch.int32, device=q.device))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--iters", type=int, default=100)
    a = ap.parse_args()
    dev = a.device
    q, k, v, gate, beta, state = make_inputs(dev)

    ref_out, ref_state = reference(q, k, v, gate, beta, state)
    print("reference out", tuple(ref_out.shape), ref_out.dtype,
          "state", tuple(ref_state.shape), flush=True)

    state_vk = state.transpose(-1, -2).contiguous().clone()  # HF [B,H,K,V] -> [B,H,V,K]
    try:
        out = candidate(q, k, v, gate, beta, state_vk)
        torch.npu.synchronize(dev)
        print("candidate out", tuple(out.shape), out.dtype, flush=True)
        ref = ref_out[0, 0].float()
        got = out.reshape(ref.shape).float()
        diff = (ref - got).abs()
        print(f"max_abs={diff.max().item():.4e} mean_abs={diff.mean().item():.4e} "
              f"ref_absmax={ref.abs().max().item():.4e}", flush=True)
        st_new = state_vk.transpose(-1, -2).float()
        sdiff = (st_new - ref_state.float()).abs()
        print(f"state max_abs={sdiff.max().item():.4e} (candidate state updated in place)",
              flush=True)
    except Exception:
        print("CANDIDATE FAILED:", flush=True)
        traceback.print_exc()
        sys.exit(2)

    # timing (eager, incl. host): reference vs candidate
    def bench(fn, n):
        for _ in range(5):
            fn()
        torch.npu.synchronize(dev)
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.npu.synchronize(dev)
        return (time.perf_counter() - t0) / n * 1e6

    t_ref = bench(lambda: reference(q, k, v, gate, beta, state), a.iters)
    t_cand = bench(lambda: candidate(q, k, v, gate, beta, state_vk.clone()), a.iters)
    print(f"reference {t_ref:.1f} us/call, candidate {t_cand:.1f} us/call", flush=True)


if __name__ == "__main__":
    main()
