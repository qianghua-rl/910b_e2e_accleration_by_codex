"""Calibrate Gemma B=1 decode step time WITHOUT profiling hooks.

Reconciles two numbers from earlier runs:
  - baseline average: ~175 ms/token (generate(), all 128 tasks)
  - breakdown run:    ~336 ms/step (hooks on, one 140-token prompt, 60 steps)

Two measurements on the same prompts, no hooks:
  1. generate():   same call as the baseline runner (per-token time = gen_s / tokens)
  2. manual loop:  the breakdown script's loop, with a device sync after each step,
                   so the step time is comparable to the breakdown's total_ms.
Also records step time vs KV length (decode steps 0..N), since attention cost grows with context.

Usage: calibrate_gemma_step.py <out.json> [--device npu:1] [--n-prompts 3] [--new 128]
"""
import argparse
import json
import sys
import time

import torch
import torch_npu  # noqa: F401
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

sys.path.insert(0, "/remote-home/S45160/work/910b_fresh/src")
import agent_protocol as P  # noqa: E402

MODEL = "/remote-home/S45160/work/910b_fresh/gemma-4-26B-A4B-it"
DATA = "/remote-home/S45160/work/910b_e2e_accleration_by_codex/data/gsm8k_128.jsonl"

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--device", default="npu:1")
ap.add_argument("--n-prompts", type=int, default=3)
ap.add_argument("--new", type=int, default=128)
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                             device_map=a.device).eval()
qs = [json.loads(l) for l in open(DATA)][:a.n_prompts]


def encode(q):
    msgs = [{"role": "system", "content": P.SYSTEM}, {"role": "user", "content": q["question"]}]
    p = tok.apply_chat_template(msgs, tools=P.TOOLS, add_generation_prompt=True,
                                tokenize=False, enable_thinking=False)
    return tok(p, return_tensors="pt", add_special_tokens=False).input_ids.to(a.device)


def gen_timed(ids):
    torch.npu.synchronize(a.device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=a.new,
                             do_sample=False, pad_token_id=tok.pad_token_id)
    torch.npu.synchronize(a.device)
    dt = time.perf_counter() - t0
    n = int(out.shape[1] - ids.shape[1])
    return dt, n


def manual_steps(ids, n_steps):
    """Same loop as profile_gemma_b1_breakdown.py, no hooks. Returns per-step ms and KV length."""
    rows = []
    with torch.no_grad():
        cache = DynamicCache()
        out = model(input_ids=ids, past_key_values=cache, use_cache=True)
        nxt = out.logits[0, -1].argmax().view(1, 1)
        kv = ids.shape[1]
        for _ in range(n_steps):
            torch.npu.synchronize(a.device)
            t0 = time.perf_counter()
            out = model(input_ids=nxt, past_key_values=cache, use_cache=True)
            nxt = out.logits[0, -1].argmax().view(1, 1)
            torch.npu.synchronize(a.device)
            rows.append({"kv_len": kv, "ms": (time.perf_counter() - t0) * 1000})
            kv += 1
    return rows


# Warm up both paths once so compile/alloc costs do not land in the measurements.
warm = encode(qs[0])
gen_timed(warm)
manual_steps(warm, 5)

res = {"model": MODEL, "device": a.device, "new_tokens": a.new, "prompts": []}
for q in qs:
    ids = encode(q)
    dt, n = gen_timed(ids)
    steps = manual_steps(ids, a.new)
    ms = [s["ms"] for s in steps]
    rec = {"id": q["id"], "prompt_tokens": int(ids.shape[1]),
           "generate_ms_per_token": dt / n * 1000, "generate_tokens": n,
           "manual_ms_per_step_mean": sum(ms) / len(ms),
           "manual_ms_per_step_median": sorted(ms)[len(ms) // 2],
           "steps": steps}
    res["prompts"].append(rec)
    print(f"id={q['id']} prompt={rec['prompt_tokens']} generate={rec['generate_ms_per_token']:.1f} "
          f"ms/tok  manual mean={rec['manual_ms_per_step_mean']:.1f} "
          f"median={rec['manual_ms_per_step_median']:.1f} ms/step", flush=True)

json.dump(res, open(a.out, "w"), indent=1)
print("WROTE", a.out, flush=True)
