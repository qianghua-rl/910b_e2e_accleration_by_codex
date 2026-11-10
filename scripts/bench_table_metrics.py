"""Table metrics for Qwen3.5-27B on one 910B3: prefill tok/s, decode ms/step and tok/s at a fixed batch.
Config = best self-written path: fast-gdn, gdn-v2, elem-conv, graph decode. Fixed batch = engine with
n_slots == B, all slots occupied, no admission during timing (so no scheduler effect).
Usage: bench_table_metrics.py <B> <out.json>"""
import json
import statistics
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
B = int(sys.argv[1]); OUT = sys.argv[2]
STEPS, WARM_STEPS = 100, 5
PREFILL_N = 8

torch.set_num_threads(8)
cb_engine.ELEM_CONV = True
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": DEV}).eval()
tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
patch_model(model, fast_gdn=True, gdn_v2=True)
eng = CBEngine(model, n_slots=B, t_max=8192, device=DEV, graph_decode=True)

tasks = [json.loads(l) for l in open(DATA)]
def ids_of(t):
    p = tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                                 {"role": "user", "content": t["question"]}],
                                tools=TOOLS, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tok(p, add_special_tokens=False, return_tensors="pt").input_ids.to(DEV)

res = {"B": B, "config": "fast-gdn gdn-v2 elem-conv graph-decode, single 910B3, bf16"}
with torch.inference_mode():
    # prefill tok/s: batch-1 prefill of real prompts, slot 0, after 2 warmups
    pf = [ids_of(t) for t in tasks[:PREFILL_N + 2]]
    for ids in pf[:2]:
        eng.prefill(0, ids)
    torch.npu.synchronize(DEV)
    tot_tok, tot_t = 0, 0.0
    for ids in pf[2:]:
        torch.npu.synchronize(DEV); t0 = time.perf_counter()
        eng.prefill(0, ids)
        torch.npu.synchronize(DEV)
        tot_t += time.perf_counter() - t0; tot_tok += ids.shape[1]
    res["prefill_tok_s"] = tot_tok / tot_t
    res["prefill_mean_prompt_tokens"] = tot_tok / PREFILL_N

    # decode: fill all B slots, then time steps (each step synchronized for per-step stats)
    last = []
    for s in range(B):
        last.append(eng.prefill(s, ids_of(tasks[s])))
    slots = list(range(B))
    for _ in range(WARM_STEPS):
        last = eng.decode_step(slots, last)
    per = []
    for _ in range(STEPS):
        torch.npu.synchronize(DEV); t0 = time.perf_counter()
        last = eng.decode_step(slots, last)
        torch.npu.synchronize(DEV)
        per.append((time.perf_counter() - t0) * 1000)
    mean_ms = statistics.mean(per); med_ms = statistics.median(per)
    res.update(decode_ms_step_mean=mean_ms, decode_ms_step_median=med_ms,
               decode_tok_s_mean=B / (mean_ms / 1000), decode_tok_s_median=B / (med_ms / 1000),
               steps_timed=STEPS)
print(json.dumps(res, indent=1), flush=True)
json.dump(res, open(OUT, "w"), indent=1)
