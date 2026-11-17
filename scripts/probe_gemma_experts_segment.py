"""Segment timing of experts_grouped in the decode path, A (HF prefill) vs B (grouped prefill).

Run as separate processes (--prefill-grouped toggles B). Per decode step, times each
experts_grouped call (device-synced) and, on the same idx, the routing-prep part alone
(argsort + bincount + cumsum) so the cost splits into prep vs grouped matmul + scatter.
Also records the group sizes per layer (how tokens*top_k spread over the 128 experts).

Usage: probe_gemma_experts_segment.py <out_json> --device npu:1 [--prefill-grouped] [--steps 24]
Instrumentation only: the wrapper returns exactly experts_grouped's output.
"""
import argparse
import json
import statistics
import sys
import time

sys.path.insert(0, "/remote-home/S45160/work/910b_fresh/src")

import torch
import torch_npu  # noqa: F401
from transformers import AutoModelForCausalLM

import gemma_cb
from gemma_run import MODEL
from gemma_cb import GemmaCB
from gemma_fast_norm import patch_gemma_norms

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--device", default="npu:1")
ap.add_argument("--prefill-grouped", action="store_true")
ap.add_argument("--steps", type=int, default=24)
a = ap.parse_args()


def sync():
    torch.npu.synchronize()


model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map=a.device).eval()
patch_gemma_norms(model, noscale=True)
layers = model.model.language_model.layers
ex_to_li = {id(l.experts): li for li, l in enumerate(layers) if getattr(l, "enable_moe_block", False)}
eng = GemmaCB(model, None, 16, 2048, a.device, grouped=True, fused_rope=True,
              prefill_grouped=a.prefill_grouped)

E = layers[0].experts.num_experts
T_eg, T_prep = {}, {}          # li -> list of seconds
group_stats = {}               # li -> list of (nonzero_experts, max_group)
state = {"on": False}
orig_eg = gemma_cb.experts_grouped


def prep_only(ex, idx):
    flat = idx.reshape(-1)
    order = torch.argsort(flat, stable=True)
    gl = torch.cumsum(torch.bincount(flat, minlength=E), 0)
    return order, gl


def wrapped(ex, x, idx, w, geglu=False):
    if not state["on"]:
        return orig_eg(ex, x, idx, w, geglu=geglu)
    li = ex_to_li[id(ex)]
    sync(); t0 = time.perf_counter()
    out = orig_eg(ex, x, idx, w, geglu=geglu)
    sync(); T_eg.setdefault(li, []).append(time.perf_counter() - t0)
    sync(); t0 = time.perf_counter()
    prep_only(ex, idx)
    sync(); T_prep.setdefault(li, []).append(time.perf_counter() - t0)
    cnt = torch.bincount(idx.reshape(-1), minlength=E)
    group_stats.setdefault(li, []).append([int((cnt > 0).sum().item()), int(cnt.max().item())])
    return out


gemma_cb.experts_grouped = wrapped

g = torch.Generator().manual_seed(0)
for s in range(16):
    ids = torch.randint(0, 200000, (1, 140 + s), generator=g).to(a.device)
    eng.prefill_into_slot(s, ids)
sync()
tokens = torch.randint(0, 200000, (16,), generator=g).to(a.device)
pos = torch.tensor([140 + s for s in range(16)], device=a.device)

# warm up (not recorded)
eng.decode_step(tokens, pos)
sync()
state["on"] = True
step_t = []
for _ in range(a.steps):
    sync(); t0 = time.perf_counter()
    logits = eng.decode_step(tokens, pos)
    sync(); step_t.append(time.perf_counter() - t0)
    tokens = logits.argmax(-1)
    pos = pos + 1
state["on"] = False


def med(xs):
    return round(statistics.median(xs) * 1000, 3)


per_layer = {li: {"eg_ms": med(T_eg[li]), "prep_ms": med(T_prep[li]),
                  "nonzero_experts_med": statistics.median([x[0] for x in group_stats[li]]),
                  "max_group_med": statistics.median([x[1] for x in group_stats[li]])}
             for li in sorted(T_eg)}
out = {"prefill_grouped": a.prefill_grouped, "steps": a.steps,
       "step_ms_median": med(step_t),
       "experts_grouped_ms_sum_per_step_median":
           round(statistics.median([sum(T_eg[li][i] for li in T_eg) for i in range(a.steps)]) * 1000, 2),
       "prep_ms_sum_per_step_median":
           round(statistics.median([sum(T_prep[li][i] for li in T_prep) for i in range(a.steps)]) * 1000, 2),
       "per_layer": per_layer}
with open(a.out, "w") as f:
    json.dump(out, f, indent=1)
print("DONE", json.dumps({k: v for k, v in out.items() if k != "per_layer"}), flush=True)
