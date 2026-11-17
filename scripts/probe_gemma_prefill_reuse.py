"""Micro-probe for KV reuse across turns (plan phase 2). Not a benchmark run; no agent loop.

Questions:
  1. How does HF batch-1 prefill time scale with prompt length? (If flat, reusing KV saves
     little, because prefill is dispatch-bound rather than token-bound.)
  2. Is "prefill prefix P into a cache, then prefill suffix S with that cache" numerically close
     to "prefill P+S in one go"? (argmax agreement of last-position logits.)

Usage: probe_gemma_prefill_reuse.py <out_json> --device npu:1
"""
import argparse
import json
import statistics
import sys
import time

sys.path.insert(0, "/remote-home/S45160/work/910b_fresh/src")

import torch
import torch_npu  # noqa: F401
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from gemma_run import MODEL

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--device", default="npu:1")
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map=a.device).eval()
V = model.config.vocab_size if hasattr(model.config, "vocab_size") else model.config.text_config.vocab_size
dev = a.device
g = torch.Generator().manual_seed(0)


def ids(n):
    return torch.randint(0, 200000, (1, n), generator=g).to(dev)


def sync():
    torch.npu.synchronize()


res = {"timing": {}, "equivalence": []}

# warmup
with torch.no_grad():
    model(input_ids=ids(64), past_key_values=DynamicCache(config=model.config), use_cache=True)
sync()

# 1. prefill time vs length (median of 5)
for L in (64, 128, 256, 512, 768):
    ts = []
    for _ in range(5):
        x = ids(L)
        sync()
        t0 = time.time()
        with torch.no_grad():
            model(input_ids=x, past_key_values=DynamicCache(config=model.config), use_cache=True)
        sync()
        ts.append(time.time() - t0)
    res["timing"][L] = round(statistics.median(ts) * 1000, 1)
    print(f"prefill L={L}: {res['timing'][L]} ms", flush=True)

# 2. suffix reuse: prefix P + suffix S (S ~ tool output + turn header, 40-120 tokens)
for P_len, S_len in ((400, 60), (600, 100), (700, 120)):
    full = ids(P_len + S_len)
    P_ids, S_ids = full[:, :P_len], full[:, P_len:]
    with torch.no_grad():
        ref = model(input_ids=full, past_key_values=DynamicCache(config=model.config),
                    use_cache=True).logits[0, -1].float()
        cache = DynamicCache(config=model.config)
        model(input_ids=P_ids, past_key_values=cache, use_cache=True)
        sync()
        t0 = time.time()
        out = model(input_ids=S_ids, past_key_values=cache, use_cache=True)
        sync()
        t_suffix = time.time() - t0
        sync()
        t0 = time.time()
        model(input_ids=full, past_key_values=DynamicCache(config=model.config), use_cache=True)
        sync()
        t_full = time.time() - t0
    got = out.logits[0, -1].float()
    rec = {"P": P_len, "S": S_len, "argmax_same": int(ref.argmax()) == int(got.argmax()),
           "max_abs_diff": round((ref - got).abs().max().item(), 4),
           "t_full_ms": round(t_full * 1000, 1), "t_suffix_ms": round(t_suffix * 1000, 1)}
    res["equivalence"].append(rec)
    print(rec, flush=True)

with open(a.out, "w") as f:
    json.dump(res, f, indent=2)
print("DONE", a.out)
