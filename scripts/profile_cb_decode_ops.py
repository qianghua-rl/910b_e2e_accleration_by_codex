"""Op-level profile of CBEngine.decode_step with 16 active slots (real prompts). Prints the top ops
by self device time and writes the per-op table to OUT/ops.txt. Run on one card, nothing else on it."""
import json
import os
import sys

import torch
import torch_npu  # noqa: F401
import torch_npu.profiler as tnp
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

sys.path.insert(0, "/remote-home/S45160/work/910b_fresh/src")
from agent_protocol import SYSTEM, TOOLS  # noqa: E402
from cb_engine import CBEngine, patch_model  # noqa: E402

MODEL = "/remote-home/S45160/work/910b_e2e_accleration_by_codex/model/qwen3.5-27b"
DATA = "/remote-home/S45160/work/910b_e2e_accleration_by_codex/data/gsm8k_128.jsonl"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/remote-home/S45160/work/910b_fresh/results/profile_cb_decode_ops"
ELEM = os.environ.get("ELEM_CONV") == "1"
DEV = "npu:0"
N = 16
STEPS = 20
os.makedirs(OUT, exist_ok=True)

model = Qwen3_5ForConditionalGeneration.from_pretrained(
    MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": DEV}).eval()
tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
patch_model(model)
import cb_engine
cb_engine.ELEM_CONV = ELEM
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
    for _ in range(3):
        last = eng.decode_step(list(range(N)), last)
    torch.npu.synchronize(DEV)
    prof = tnp.profile(
        activities=[tnp.ProfilerActivity.CPU, tnp.ProfilerActivity.NPU],
        on_trace_ready=tnp.tensorboard_trace_handler(OUT),
        record_shapes=False, profile_memory=False, with_stack=False)
    prof.start()
    for _ in range(STEPS):
        last = eng.decode_step(list(range(N)), last)
    torch.npu.synchronize(DEV)
    prof.stop()

print("profile written to", OUT, flush=True)
