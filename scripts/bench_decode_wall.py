"""Wall-clock decode step time (no profiler) for CBEngine with 16 active slots, conv1d (A) vs elem (B).
Same prefilled state for both modes; each mode measured twice in alternating order (ABBA) to check drift.
Run on one card with ELEM_CONV unset; the mode is switched in-process."""
import json
import sys
import time

import torch
import torch_npu  # noqa: F401
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

sys.path.insert(0, "/remote-home/S45160/work/910b_fresh/src")
import cb_engine  # noqa: E402
from agent_protocol import SYSTEM, TOOLS  # noqa: E402
from cb_engine import CBEngine, patch_model  # noqa: E402

MODEL = "/remote-home/S45160/work/910b_e2e_accleration_by_codex/model/qwen3.5-27b"
DATA = "/remote-home/S45160/work/910b_e2e_accleration_by_codex/data/gsm8k_128.jsonl"
DEV = "npu:0"
N = 16
STEPS = 60

model = Qwen3_5ForConditionalGeneration.from_pretrained(
    MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": DEV}).eval()
tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
patch_model(model, fast_gdn=True)
eng = CBEngine(model, n_slots=N, t_max=4096, device=DEV)
tasks = [json.loads(l) for l in open(DATA)][:N]

with torch.inference_mode():
    last = []
    for s, t in enumerate(tasks):
        p = tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                                     {"role": "user", "content": t["question"]}],
                                    tools=TOOLS, tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False)
        ids = tok(p, add_special_tokens=False, return_tensors="pt").input_ids.to(DEV)
        last.append(eng.prefill(s, ids))
    for _ in range(5):
        last = eng.decode_step(list(range(N)), last)

    def run(elem):
        global last
        cb_engine.ELEM_CONV = elem
        torch.npu.synchronize(DEV)
        t0 = time.perf_counter()
        for _ in range(STEPS):
            last = eng.decode_step(list(range(N)), last)
        torch.npu.synchronize(DEV)
        return (time.perf_counter() - t0) / STEPS * 1000

    res = {"A": [], "B": []}
    for mode in ["A", "B", "B", "A"]:
        ms = run(mode == "B")
        res[mode].append(ms)
        print(f"mode {mode}: {ms:.2f} ms/step", flush=True)
    for k, v in res.items():
        print(f"{k} mean {sum(v)/len(v):.2f} ms/step over {len(v)} runs", flush=True)
