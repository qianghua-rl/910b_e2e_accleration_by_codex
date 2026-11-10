"""Correctness + timing: src/gdn_chunk_fast.py vs transformers' torch_chunk_gated_delta_rule.
Random inputs with the model's GDN head dims (from config.json). Run on one NPU only."""
import json
import sys
import time

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, "/remote-home/S45160/work/910b_fresh/src")
from gdn_chunk_fast import fast_chunk_gated_delta_rule  # noqa: E402
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule as ref_fn  # noqa: E402

MODEL = "/remote-home/S45160/work/910b_e2e_accleration_by_codex/model/qwen3.5-27b"
DEV = "npu:0"
cfg = json.load(open(f"{MODEL}/config.json"))
tc = cfg.get("text_config", cfg)
H = tc["linear_num_value_heads"]
Dk = tc["linear_key_head_dim"]
Dv = tc["linear_value_head_dim"]
print(f"heads={H} Dk={Dk} Dv={Dv}", flush=True)


def make(L, seed=0):
    g_ = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(1, L, H, Dk, generator=g_).to(DEV, torch.bfloat16)
    k = torch.randn(1, L, H, Dk, generator=g_).to(DEV, torch.bfloat16)
    v = torch.randn(1, L, H, Dv, generator=g_).to(DEV, torch.bfloat16)
    g = (-torch.rand(1, L, H, generator=g_) * 0.5).to(DEV, torch.float32)
    beta = torch.rand(1, L, H, generator=g_).to(DEV, torch.bfloat16)
    return q, k, v, g, beta


def timeit(fn, args, n=3):
    fn(*args, use_qk_l2norm_in_kernel=True, output_final_state=True)
    torch.npu.synchronize(DEV)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        out = fn(*args, use_qk_l2norm_in_kernel=True, output_final_state=True)
        torch.npu.synchronize(DEV)
        ts.append(time.perf_counter() - t0)
    return out, min(ts)


with torch.inference_mode():
    for L in [353, 1024, 2048]:
        args = make(L)
        (o_ref, s_ref), t_ref = timeit(ref_fn, args)
        (o_new, s_new), t_new = timeit(fast_chunk_gated_delta_rule, args)
        o_ref, o_new = o_ref.float(), o_new.float()
        diff = (o_ref - o_new).abs().max().item()
        scale = o_ref.abs().max().item()
        sdiff = (s_ref.float() - s_new.float()).abs().max().item()
        print(f"L={L:5d} out_maxdiff={diff:.3e} (ref max {scale:.3e}) state_maxdiff={sdiff:.3e} "
              f"ref={t_ref*1000:8.1f} ms  fast={t_new*1000:8.1f} ms  speedup={t_ref/t_new:.2f}x", flush=True)
