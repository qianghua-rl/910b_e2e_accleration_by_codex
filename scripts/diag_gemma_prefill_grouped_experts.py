"""Diagnostic: single-layer HF experts vs experts_grouped on REAL routed inputs at several T.

Why: scripts/probe_gemma_prefill_grouped.py showed grouped experts match HF at T=64 (exact) but
not at T=256+ (max abs diff ~7, argmax changes). Synthetic checks (E-G22, T<=16) cannot show this.
This captures (hidden, top_k_index, top_k_weights) from a real HF forward at each T and compares
the two expert implementations on those exact inputs, for the first and a middle layer.

Usage: diag_gemma_prefill_grouped_experts.py <out_json> --device npu:1
"""
import argparse
import json
import sys

sys.path.insert(0, "/remote-home/S45160/work/910b_fresh/src")

import torch
import torch_npu  # noqa: F401
from transformers import AutoModelForCausalLM, DynamicCache

from gemma_run import MODEL
from gemma_cb import experts_grouped

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--device", default="npu:1")
a = ap.parse_args()

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map=a.device).eval()
layers = model.model.language_model.layers
g = torch.Generator().manual_seed(0)
captured = {}


def make_hook(li):
    def hook(mod, args):
        captured[li] = [t.detach().clone() for t in args]
    return hook


handles = [layers[li].experts.register_forward_pre_hook(make_hook(li)) for li in (0, 15)]
res = {}
for T in (64, 128, 256, 512):
    ids = torch.randint(0, 200000, (1, T), generator=g).to(a.device)
    with torch.no_grad():
        model(input_ids=ids, past_key_values=DynamicCache(config=model.config), use_cache=True)
    for li in (0, 15):
        hs, ti, tw = captured[li]
        with torch.no_grad():
            ref = layers[li].experts(hs, ti, tw).float()
            got = experts_grouped(layers[li].experts, hs, ti, tw).float()
        d = (ref - got).abs()
        res[f"T{T}_L{li}"] = {
            "shape": list(hs.shape), "dtype": str(hs.dtype), "idx_dtype": str(ti.dtype),
            "max_abs_diff": round(d.max().item(), 5), "mean_abs_diff": round(d.mean().item(), 6),
            "ref_absmax": round(ref.abs().max().item(), 3),
            "rows_with_diff_gt_0.5": int((d.max(-1).values > 0.5).sum().item()),
        }
        print(f"T={T} L={li}: {res[f'T{T}_L{li}']}", flush=True)
for h in handles:
    h.remove()
with open(a.out, "w") as f:
    json.dump(res, f, indent=2)
print("DONE", a.out)
