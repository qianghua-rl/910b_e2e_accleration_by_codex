"""B=1 decode wall-clock breakdown for Gemma-4-26B-A4B-it (HF path, one NPU).

Mirrors the Qwen analysis (profile_decode_wall_breakdown.py) but for a plain HF model:
  - host enqueue time per module: forward hooks timed WITHOUT a device sync, so they measure
    Python + op-launch cost only.
  - total step time: measured WITH torch.npu.synchronize(), so the difference
    (total - host enqueue) is device work that was not overlapped with the host.
Modules timed: each decoder layer, its self_attn, and its MoE / MLP block (found by name).

Usage: profile_gemma_b1_breakdown.py <out.json> [--device npu:1] [--steps 60]
"""
import argparse
import json
import sys
import time

import torch
import torch_npu  # noqa: F401
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/remote-home/S45160/work/910b_fresh/src")
import agent_protocol as P  # noqa: E402

MODEL = "/remote-home/S45160/work/910b_fresh/gemma-4-26B-A4B-it"
DATA = "/remote-home/S45160/work/910b_e2e_accleration_by_codex/data/gsm8k_128.jsonl"

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--device", default="npu:1")
ap.add_argument("--steps", type=int, default=60)
ap.add_argument("--warmup", type=int, default=5)
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                             device_map=a.device).eval()
q = json.loads(open(DATA).readline())
msgs = [{"role": "system", "content": P.SYSTEM}, {"role": "user", "content": q["question"]}]
prompt = tok.apply_chat_template(msgs, tools=P.TOOLS, add_generation_prompt=True,
                                 tokenize=False, enable_thinking=False)
ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(a.device)

# Classify modules. Decoder layers are the children of the text model's `layers` ModuleList.
timers = {"layer": 0.0, "attn": 0.0, "moe_or_mlp": 0.0}
stack = {}


def pre(name):
    def f(mod, inp):
        stack[name] = time.perf_counter()
    return f


def post(name, key):
    def f(mod, inp, out):
        timers[key] += time.perf_counter() - stack.pop(name)
    return f


hooked = 0
for name, mod in model.named_modules():
    parts = name.split(".")
    if "layers" not in parts:
        continue
    idx = parts.index("layers")
    if len(parts) == idx + 2:  # decoder layer: ...layers.N
        mod.register_forward_pre_hook(pre(name))
        mod.register_forward_hook(post(name, "layer"))
        hooked += 1
    elif len(parts) == idx + 3 and parts[-1] == "self_attn":
        mod.register_forward_pre_hook(pre(name))
        mod.register_forward_hook(post(name, "attn"))
        hooked += 1
    elif len(parts) == idx + 3 and parts[-1] in ("mlp", "moe", "experts", "block_sparse_moe"):
        mod.register_forward_pre_hook(pre(name))
        mod.register_forward_hook(post(name, "moe_or_mlp"))
        hooked += 1

# Decode loop with KV cache, greedy, B=1. Each step: one forward of one new token.
from transformers import DynamicCache  # noqa: E402


def run(n_steps, sync):
    with torch.no_grad():
        cache = DynamicCache()
        out = model(input_ids=ids, past_key_values=cache, use_cache=True)
        nxt = out.logits[0, -1].argmax().view(1, 1)
        rows = []
        for _ in range(n_steps):
            for k in timers:
                timers[k] = 0.0
            if sync:
                torch.npu.synchronize(a.device)
            t0 = time.perf_counter()
            out = model(input_ids=nxt, past_key_values=cache, use_cache=True)
            nxt = out.logits[0, -1].argmax().view(1, 1)
            t_enq = time.perf_counter()  # host finished enqueuing ops (no device sync yet)
            torch.npu.synchronize(a.device)
            t1 = time.perf_counter()
            rows.append({"total_ms": (t1 - t0) * 1000, "enqueue_ms": (t_enq - t0) * 1000,
                         **{k + "_host_ms": v * 1000 for k, v in timers.items()}})
        return rows


run(a.warmup, sync=True)
rows = run(a.steps, sync=True)
mean = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
mean["device_wait_ms"] = mean["total_ms"] - mean["enqueue_ms"]
res = {"model": MODEL, "device": a.device, "steps": a.steps, "hooked_modules": hooked,
       "prompt_tokens": int(ids.shape[1]), "mean": mean, "rows": rows}
json.dump(res, open(a.out, "w"), indent=1)
print(json.dumps(mean, indent=1), flush=True)
