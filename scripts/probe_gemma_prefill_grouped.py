"""Micro-probe: does the HF batch-1 prefill get faster with grouped MoE experts? (plan phase 2)

Context: scripts/probe_gemma_prefill_reuse.py showed prefill time is nearly flat in prompt length
(64 tok 654 ms, 768 tok 1013 ms), i.e. a large fixed cost per forward. HF's experts forward is a
per-expert Python loop with host syncs, so it is the prime suspect. This probe swaps the experts
forward of the HF prefill path for experts_grouped (same function the CB decode path uses) and
compares time and last-position logits.

Usage: probe_gemma_prefill_grouped.py <out_json> --device npu:1
"""
import argparse
import json
import statistics
import sys
import time

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
dev = a.device
g = torch.Generator().manual_seed(0)
LENS = (64, 256, 512, 768)
inputs = {L: torch.randint(0, 200000, (1, L), generator=g).to(dev) for L in LENS}


def sync():
    torch.npu.synchronize()


def run(L):
    with torch.no_grad():
        return model(input_ids=inputs[L], past_key_values=DynamicCache(config=model.config),
                     use_cache=True).logits[0, -1].float()


def timed(L, reps=5):
    ts = []
    for _ in range(reps):
        sync()
        t0 = time.time()
        run(L)
        sync()
        ts.append(time.time() - t0)
    return round(statistics.median(ts) * 1000, 1)


res = {"hf": {}, "grouped": {}, "equiv": {}}
run(64)  # warmup
ref = {L: run(L) for L in LENS}
for L in LENS:
    res["hf"][L] = timed(L)
    print(f"hf L={L}: {res['hf'][L]} ms", flush=True)

# swap experts forward on every layer (HF path only; CB decode is unaffected)
orig = {}
for li, layer in enumerate(model.model.language_model.layers):
    ex = layer.experts
    orig[li] = ex.forward
    ex.forward = (lambda ex: (lambda hs, ti, tw: experts_grouped(ex, hs, ti, tw)))(ex)

run(64)
got = {L: run(L) for L in LENS}
for L in LENS:
    res["equiv"][L] = {"argmax_same": int(ref[L].argmax()) == int(got[L].argmax()),
                       "max_abs_diff": round((ref[L] - got[L]).abs().max().item(), 4)}
    res["grouped"][L] = timed(L)
    print(f"grouped L={L}: {res['grouped'][L]} ms  {res['equiv'][L]}", flush=True)

with open(a.out, "w") as f:
    json.dump(res, f, indent=2)
print("DONE", a.out)
