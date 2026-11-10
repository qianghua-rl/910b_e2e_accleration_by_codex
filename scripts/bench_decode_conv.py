"""Micro-benchmark: decode causal depthwise conv (HF torch fallback) vs an elementwise
window formulation (mul + reduce-sum over K). No model load. Uses real GDN conv dims from config.json.
Run with ASCEND_RT_VISIBLE_DEVICES=1 and device npu:0."""
import json
import time

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

MODEL = "/remote-home/S45160/work/910b_e2e_accleration_by_codex/model/qwen3.5-27b"
DEV = "npu:0"
N = 16
ITERS = 200

tc = json.load(open(f"{MODEL}/config.json")).get("text_config", {})
C = tc["linear_num_key_heads"] * tc["linear_key_head_dim"] * 2 + tc["linear_num_value_heads"] * tc["linear_value_head_dim"]
K = tc.get("linear_conv_kernel_dim", 4)
print(f"conv_dim C={C} K={K} n={N}", flush=True)

g = torch.Generator().manual_seed(0)
w = torch.randn(C, K, generator=g).to(DEV, torch.bfloat16)  # depthwise weight [C,K]
b = torch.randn(C, generator=g).to(DEV, torch.bfloat16)
x0 = torch.randn(N, C, 1, generator=g).to(DEV, torch.bfloat16)
s0 = torch.randn(N, C, K, generator=g).to(DEV, torch.bfloat16)


def ref(x, state):
    # HF torch_causal_conv1d_update, state length = K (matches the engine's cache layout)
    hs = torch.cat([state, x], dim=-1).to(w.dtype)
    state.copy_(hs[:, :, -K:])
    out = F.conv1d(hs, w.unsqueeze(1), b, padding=0, groups=C)
    return F.silu(out[:, :, -1:])


def elem(x, state):
    hs = torch.cat([state, x], dim=-1).to(w.dtype)  # [n,C,K+1]
    state.copy_(hs[:, :, -K:])
    win = hs[:, :, -K:]                              # [n,C,K], last K inputs
    out = (win * w).sum(-1, keepdim=True) + b.view(1, C, 1)
    return F.silu(out)


def bench(fn):
    st = s0.clone()
    for _ in range(10):
        fn(x0, st)
    torch.npu.synchronize(DEV)
    t = time.perf_counter()
    for _ in range(ITERS):
        out = fn(x0, st)
    torch.npu.synchronize(DEV)
    return out, (time.perf_counter() - t) / ITERS * 1e3


with torch.inference_mode():
    o_r, t_r = bench(ref)
    o_e, t_e = bench(elem)
    s_r, s_e = s0.clone(), s0.clone()
    ref(x0, s_r)
    elem(x0, s_e)
    print(f"ref  (conv1d): {t_r:.4f} ms/call  x48 layers = {t_r*48:.2f} ms/step", flush=True)
    print(f"elem (mul+sum): {t_e:.4f} ms/call  x48 layers = {t_e*48:.2f} ms/step", flush=True)
    print(f"max|out diff| {(o_r.float()-o_e.float()).abs().max().item():.3e}  "
          f"max|state diff| {(s_r.float()-s_e.float()).abs().max().item():.3e}", flush=True)
