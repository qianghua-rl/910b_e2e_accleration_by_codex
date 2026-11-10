"""Correctness check: CBEngine (several slots decoded together) vs HF generate (B=1), greedy.
Run on a free card, e.g. ASCEND_RT_VISIBLE_DEVICES=1 ... --device npu:0 (inside that card's view).
"""
import argparse
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--batch-prefill", action="store_true")
    a = ap.parse_args()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": a.device}).eval()
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    patch_model(model)
    tasks = [json.loads(l) for l in open(DATA)][:a.n]
    prompts = [tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                                        {"role": "user", "content": t["question"]}],
                                       tools=TOOLS, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False) for t in tasks]
    ids_list = [tok(p, add_special_tokens=False, return_tensors="pt").input_ids.to(a.device) for p in prompts]

    # reference: HF generate, one row at a time
    ref = []
    with torch.inference_mode():
        for ids in ids_list:
            out = model.generate(input_ids=ids, max_new_tokens=a.steps, do_sample=False, use_cache=True)
            ref.append(out[0, ids.shape[1]:].tolist())

    # engine: all slots together, staggered by prefilling them one after another then decoding jointly
    eng = CBEngine(model, n_slots=a.n, t_max=4096, device=a.device)
    gen = [[] for _ in range(a.n)]
    last = []
    if a.batch_prefill:
        firsts = eng.prefill_batch(list(range(a.n)), ids_list)
        for s in range(a.n):
            gen[s].append(firsts[s])
        last = list(firsts)
    else:
        for s, ids in enumerate(ids_list):
            first = eng.prefill(s, ids)
            gen[s].append(first)
            last.append(first)
    t0 = time.perf_counter()
    for _ in range(a.steps - 1):
        nxt = eng.decode_step(list(range(a.n)), last)
        for s in range(a.n):
            gen[s].append(nxt[s])
        last = nxt
    torch.npu.synchronize(a.device)
    print(f"decode {a.steps-1} steps x {a.n} slots: {time.perf_counter()-t0:.2f}s", flush=True)

    for s in range(a.n):
        r, g = ref[s], gen[s]
        same = 0
        for x, y in zip(r, g):
            if x != y:
                break
            same += 1
        print(f"slot {s}: identical prefix {same}/{len(r)} tokens; ref[:8]={r[:8]} eng[:8]={g[:8]}", flush=True)


if __name__ == "__main__":
    main()
