"""Op-level profile of the batch-1 prefill path of the current best config (fast-gdn + gdn-v2).
Runs 8 real prompts (after warmup), profiles them, and writes the profiler output under OUT.
Then summarise with scripts/summarise_kernel_csv.py. Run on one card, nothing else on it."""
import json
import os
import sys
import time

import torch
import torch_npu  # noqa: F401
import torch_npu.profiler as tnp
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

sys.path.insert(0, "/remote-home/S45160/work/910b_fresh/src")
from agent_protocol import SYSTEM, TOOLS  # noqa: E402
from cb_engine import CBEngine, patch_model  # noqa: E402

MODEL = "/remote-home/S45160/work/910b_e2e_accleration_by_codex/model/qwen3.5-27b"
DATA = "/remote-home/S45160/work/910b_e2e_accleration_by_codex/data/gsm8k_128.jsonl"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/remote-home/S45160/work/910b_fresh/results/prefill_v2_ops"
DEV = "npu:0"
N_PROF = 8
os.makedirs(OUT, exist_ok=True)

model = Qwen3_5ForConditionalGeneration.from_pretrained(
    MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": DEV}).eval()
tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
patch_model(model, fast_gdn=True, gdn_v2=True)
eng = CBEngine(model, n_slots=16, t_max=4096, device=DEV)
tasks = [json.loads(l) for l in open(DATA)][:N_PROF + 2]


def ids_of(t):
    p = tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                                 {"role": "user", "content": t["question"]}],
                                tools=TOOLS, tokenize=False, add_generation_prompt=True,
                                enable_thinking=False)
    return tok(p, add_special_tokens=False, return_tensors="pt").input_ids.to(DEV)


with torch.inference_mode():
    all_ids = [ids_of(t) for t in tasks]
    for s in range(2):  # warmup (not profiled)
        eng.prefill(s, all_ids[s])
    torch.npu.synchronize(DEV)
    tokens = sum(x.shape[1] for x in all_ids[2:])
    prof = tnp.profile(
        activities=[tnp.ProfilerActivity.CPU, tnp.ProfilerActivity.NPU],
        on_trace_ready=tnp.tensorboard_trace_handler(OUT),
        record_shapes=False, profile_memory=False, with_stack=False)
    prof.start()
    t0 = time.perf_counter()
    for s, ids in enumerate(all_ids[2:]):
        eng.prefill(s % 16, ids)
    torch.npu.synchronize(DEV)
    wall = time.perf_counter() - t0
    prof.stop()

print(json.dumps({"prompts": N_PROF, "tokens": tokens, "wall_s": wall,
                  "wall_ms_per_prompt": wall * 1000 / N_PROF}), flush=True)
print("profile written to", OUT, flush=True)
