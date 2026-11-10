"""Table metrics for Qwen3.5-27B, TP=2 on two 910B3 (one rank per card). Same protocol as
bench_table_metrics.py (single card): prefill tok/s (batch-1), decode ms/step at fixed batch B,
decode tok/s = B / step. Config: fast-gdn, gdn-v2, elem-conv, graph decode.
Launch both ranks: RANK=r ASCEND_RT_VISIBLE_DEVICES=r python3 -I bench_table_metrics_tp2.py <B> <out.json>
Only rank 0 writes the JSON."""
import json
import os
import statistics
import sys
import time

import torch
import torch_npu  # noqa: F401
import torch.distributed as dist
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

sys.path.insert(0, "/remote-home/S45160/work/910b_fresh/src")
import cb_engine  # noqa: E402
import cb_tp  # noqa: E402
from agent_protocol import SYSTEM, TOOLS  # noqa: E402
from cb_engine import patch_model  # noqa: E402

MODEL = "/remote-home/S45160/work/910b_e2e_accleration_by_codex/model/qwen3.5-27b"
DATA = "/remote-home/S45160/work/910b_e2e_accleration_by_codex/data/gsm8k_128.jsonl"
DEV = "npu:0"
RANK = int(os.environ["RANK"]); TP = 2
B = int(sys.argv[1]); OUT = sys.argv[2]
STEPS, WARM_STEPS, PREFILL_N = 100, 5, 8

torch.set_num_threads(8)
cb_tp.init_tp(RANK, TP, os.environ.get("MASTER_PORT", "29561"))
cb_engine.ELEM_CONV = True
model = cb_tp.shard_model(
    Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", device_map=None).eval(),
    RANK, TP).to(DEV)
tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
patch_model(model, fast_gdn=True, gdn_v2=True)
eng = cb_tp.TPEngine(model, n_slots=B, t_max=8192, device=DEV, tp=TP, graph_decode=True)

tasks = [json.loads(l) for l in open(DATA)]
def ids_of(t):
    p = tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                                 {"role": "user", "content": t["question"]}],
                                tools=TOOLS, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tok(p, add_special_tokens=False, return_tensors="pt").input_ids.to(DEV)

res = {"B": B, "tp": TP, "config": "TP=2, fast-gdn gdn-v2 elem-conv graph-decode, 2x910B3, bf16"}
with torch.inference_mode():
    pf = [ids_of(t) for t in tasks[:PREFILL_N + 2]]
    for ids in pf[:2]:
        eng.prefill(0, ids)
    torch.npu.synchronize(DEV)
    tot_tok, tot_t = 0, 0.0
    for ids in pf[2:]:
        dist.barrier()
        torch.npu.synchronize(DEV); t0 = time.perf_counter()
        eng.prefill(0, ids)
        torch.npu.synchronize(DEV)
        tot_t += time.perf_counter() - t0; tot_tok += ids.shape[1]
    res["prefill_tok_s"] = tot_tok / tot_t
    res["prefill_mean_prompt_tokens"] = tot_tok / PREFILL_N

    last = [eng.prefill(s, ids_of(tasks[s])) for s in range(B)]
    slots = list(range(B))
    for _ in range(WARM_STEPS):
        last = eng.decode_step(slots, last)
    per = []
    for _ in range(STEPS):
        dist.barrier()
        torch.npu.synchronize(DEV); t0 = time.perf_counter()
        last = eng.decode_step(slots, last)
        torch.npu.synchronize(DEV)
        per.append((time.perf_counter() - t0) * 1000)
    mean_ms = statistics.mean(per); med_ms = statistics.median(per)
    res.update(decode_ms_step_mean=mean_ms, decode_ms_step_median=med_ms,
               decode_tok_s_mean=B / (mean_ms / 1000), decode_tok_s_median=B / (med_ms / 1000),
               steps_timed=STEPS)
if RANK == 0:
    print(json.dumps(res, indent=1), flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
dist.barrier()
