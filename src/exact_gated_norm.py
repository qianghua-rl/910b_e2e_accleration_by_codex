"""Single-layer exact check for the GDN gated RMSNorm (decode shape, T=1).

Reference: transformers Qwen3_5RMSNormGated.forward, with the real layer-0 weight.
Candidate: same formula with fewer tensor casts/ops. Accepted only if the output is
bit-identical on fixed inputs; timing is reported for information only.
Usage: exact_gated_norm.py [--device npu:0] [--iters 200]
"""
import argparse
import json
import time

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401
from safetensors import safe_open

MODEL = "/remote-home/S45160/work/910b_e2e_accleration_by_codex/model/qwen3.5-27b"
HEADS, DIM = 48, 128  # linear_num_value_heads x head_v_dim


def load_weight(name_hint="layers.0.linear_attn.norm.weight"):
    idx = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]
    key = next(k for k in idx if k.endswith(name_hint))
    with safe_open(f"{MODEL}/{idx[key]}", framework="pt") as f:
        return key, f.get_tensor(key)


def reference(x, gate, weight, eps=1e-6):
    input_dtype = x.dtype
    h = x.to(torch.float32)
    var = h.pow(2).mean(-1, keepdim=True)
    h = h * torch.rsqrt(var + eps)
    h = weight * h.to(input_dtype)
    h = h * F.silu(gate.to(torch.float32))
    return h.to(input_dtype)


def candidate(x, gate, weight, eps=1e-6):
    # Same operations in the same order, but the gate is cast once and silu is applied
    # to it directly; the rsqrt-scale and weight multiply keep the reference dtype path.
    input_dtype = x.dtype
    h = x.float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    h = weight * h.to(input_dtype)
    g = F.silu(gate.float())
    return (h * g).to(input_dtype)


def bench(fn, args, device, iters):
    for _ in range(10):
        fn(*args)
    torch.npu.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.npu.synchronize(device)
    return (time.perf_counter() - t0) / iters * 1e6  # us per call


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--iters", type=int, default=200)
    a = ap.parse_args()

    key, w = load_weight()
    print("weight:", key, tuple(w.shape), w.dtype, flush=True)
    w = w.to(a.device).float()  # RMSNormGated keeps its weight in the checkpoint dtype
    g = torch.Generator().manual_seed(0)
    x = torch.randn(HEADS, DIM, generator=g).to(torch.bfloat16).to(a.device)
    gate = torch.randn(HEADS, DIM, generator=g).to(torch.bfloat16).to(a.device)
    w_bf = w.to(torch.bfloat16)

    ref = reference(x, gate, w_bf)
    cand = candidate(x, gate, w_bf)
    diff = (ref.float() - cand.float()).abs()
    bit_equal = torch.equal(ref, cand)
    print(f"bit_equal={bit_equal} max_abs={diff.max().item():.3e} "
          f"n_mismatch={(ref != cand).sum().item()}/{ref.numel()}", flush=True)

    t_ref = bench(reference, (x, gate, w_bf), a.device, a.iters)
    t_cand = bench(candidate, (x, gate, w_bf), a.device, a.iters)
    print(f"reference {t_ref:.1f} us/call, candidate {t_cand:.1f} us/call (eager, incl. host)",
          flush=True)


if __name__ == "__main__":
    main()
