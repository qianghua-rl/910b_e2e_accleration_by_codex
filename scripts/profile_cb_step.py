"""Per-section timing of one CBEngine decode step with 16 active slots (real prompts from the dataset)."""
import json
import sys
import time

import torch
import torch_npu  # noqa: F401
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

sys.path.insert(0, "/remote-home/S45160/work/910b_fresh/src")
from agent_protocol import SYSTEM, TOOLS  # noqa: E402
from cb_engine import CBEngine, patch_model  # noqa: E402

MODEL = "/remote-home/S45160/work/910b_e2e_accleration_by_codex/model/qwen3.5-27b"
DATA = "/remote-home/S45160/work/910b_e2e_accleration_by_codex/data/gsm8k_128.jsonl"
DEV = "npu:0"
N = 16
STEPS = 30

model = Qwen3_5ForConditionalGeneration.from_pretrained(
    MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": DEV}).eval()
tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
patch_model(model)
eng = CBEngine(model, n_slots=N, t_max=4096, device=DEV)
tasks = [json.loads(l) for l in open(DATA)][:N]

acc = {}


def timed(name, fn):
    def wrap(*a, **k):
        torch.npu.synchronize(DEV)
        t = time.perf_counter()
        r = fn(*a, **k)
        torch.npu.synchronize(DEV)
        acc[name] = acc.get(name, 0.0) + time.perf_counter() - t
        return r
    return wrap


with torch.inference_mode():
    last = []
    for s, t in enumerate(tasks):
        p = tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                                     {"role": "user", "content": t["question"]}],
                                    tools=TOOLS, tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False)
        ids = tok(p, add_special_tokens=False, return_tensors="pt").input_ids.to(DEV)
        last.append(eng.prefill(s, ids))
    print("prompt lens", eng.lens, flush=True)
    # warmup
    for _ in range(3):
        last = eng.decode_step(list(range(N)), last)

    # whole-step timing without instrumentation
    torch.npu.synchronize(DEV)
    t0 = time.perf_counter()
    for _ in range(STEPS):
        last = eng.decode_step(list(range(N)), last)
    torch.npu.synchronize(DEV)
    full = (time.perf_counter() - t0) / STEPS
    print(f"full step: {full*1000:.1f} ms", flush=True)

    # instrumented timing (adds sync overhead; use for proportions)
    eng._attn = timed("attn", eng._attn)
    eng._gdn = timed("gdn", eng._gdn)
    for layer in eng.lm.layers:
        layer.mlp.forward = timed("mlp", layer.mlp.forward)
    acc.clear()
    for _ in range(STEPS):
        last = eng.decode_step(list(range(N)), last)
    for k, v in sorted(acc.items(), key=lambda x: -x[1]):
        print(f"{k:6s}: {v/STEPS*1000:7.1f} ms/step", flush=True)
    rest = full - sum(acc.values()) / STEPS
    print(f"other  : {rest*1000:7.1f} ms/step (embed, norms, rotary, lm_head, host work)", flush=True)
