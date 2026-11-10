"""Single-layer exact check for two GDN host-overhead candidates (decode, T=1).

  A. no zero-initialised core_attn_out buffer: the single step's output is written
     directly (same fp32 values, same dtype path as the reference).
  B. repeat_interleave(q,k) replaced by expand + reshape (identical values).

Reference: transformers torch_recurrent_gated_delta_rule (unchanged).
Acceptance here: bit-identical output and state on several fixed-seed inputs.
Timing is informational only (eager, includes host dispatch).
Usage: exact_gdn_candidates.py [--device npu:0] [--seeds 5] [--iters 200]
"""
import argparse
import sys
import time

import torch
import torch_npu  # noqa: F401
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    l2norm,
    torch_recurrent_gated_delta_rule as reference,
)

H_K, H_V, K, V = 16, 48, 128, 128  # Qwen3.5 linear attention: 16 q/k heads, 48 v heads


def candidate_recurrent(query, key, value, g, beta, initial_state, use_qk_l2norm_in_kernel=True):
    """T=1 specialisation of the reference recurrence with candidates A and B applied."""
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale
    state = initial_state.to(value)

    q_t = query[:, :, 0]
    k_t = key[:, :, 0]
    v_t = value[:, :, 0]
    g_t = g[:, :, 0].exp().unsqueeze(-1).unsqueeze(-1)
    beta_t = beta[:, :, 0].unsqueeze(-1)

    state = state * g_t
    kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
    delta = (v_t - kv_mem) * beta_t
    state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
    out_t = (state * q_t.unsqueeze(-1)).sum(dim=-2)           # [B,H,V], fp32
    core_attn_out = out_t.unsqueeze(2)                         # [B,H,1,V] — no zeros buffer
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, state


def expand_repeat(x, n):
    """Equivalent of repeat_interleave(n, dim=2) on [B,T,H,D] via expand+reshape."""
    b, t, h, d = x.shape
    return x.unsqueeze(3).expand(b, t, h, n, d).reshape(b, t, h * n, d)


def run_candidate_qk(query, key, value, gate, beta, state):
    """Full path including candidate B (repeat) and candidate A (output buffer)."""
    n = H_V // H_K
    query = expand_repeat(query, n)
    key = expand_repeat(key, n)
    return candidate_recurrent(query, key, value, gate, beta, state)


def make_inputs(device, seed):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(1, 1, H_K, K, generator=g).to(torch.bfloat16).to(device)
    k = torch.randn(1, 1, H_K, K, generator=g).to(torch.bfloat16).to(device)
    v = torch.randn(1, 1, H_V, V, generator=g).to(torch.bfloat16).to(device)
    gate = (-torch.rand(1, 1, H_V, generator=g).abs() * 0.5).to(device)
    beta = torch.rand(1, 1, H_V, generator=g).to(torch.bfloat16).to(device)
    state = (torch.randn(1, H_V, K, V, generator=g) * 0.1).to(torch.bfloat16).to(device)
    return q, k, v, gate, beta, state


def reference_full(q, k, v, gate, beta, state):
    # the reference receives repeated q/k exactly as the HF forward does
    n = H_V // H_K
    q = q.repeat_interleave(n, dim=2)
    k = k.repeat_interleave(n, dim=2)
    return reference(q, k, v, g=gate, beta=beta, initial_state=state.clone(),
                     output_final_state=True, use_qk_l2norm_in_kernel=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--iters", type=int, default=200)
    a = ap.parse_args()
    dev = a.device

    all_equal = True
    for seed in range(a.seeds):
        inp = make_inputs(dev, seed)
        ro, rs = reference_full(*inp)
        co, cs = run_candidate_qk(*inp[:5], inp[5].clone())
        torch.npu.synchronize(dev)
        eq_o = torch.equal(ro, co)
        eq_s = torch.equal(rs, cs)
        d_o = (ro.float() - co.float()).abs().max().item()
        d_s = (rs.float() - cs.float()).abs().max().item()
        print(f"seed {seed}: out_equal={eq_o} (max {d_o:.3e})  state_equal={eq_s} (max {d_s:.3e})",
              flush=True)
        all_equal &= eq_o and eq_s

    print("RESULT:", "BIT-EXACT on all seeds" if all_equal else "NOT exact", flush=True)

    inp = make_inputs(dev, 0)
    def bench(fn):
        for _ in range(10):
            fn()
        torch.npu.synchronize(dev)
        t0 = time.perf_counter()
        for _ in range(a.iters):
            fn()
        torch.npu.synchronize(dev)
        return (time.perf_counter() - t0) / a.iters * 1e6
    t_ref = bench(lambda: reference_full(*inp))
    t_cand = bench(lambda: run_candidate_qk(*inp[:5], inp[5].clone()))
    print(f"timing (info only): reference {t_ref:.1f} us, candidate {t_cand:.1f} us", flush=True)
    sys.exit(0 if all_equal else 1)


if __name__ == "__main__":
    main()
