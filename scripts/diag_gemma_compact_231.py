"""Locate the first divergence of task 231 between full-slot CB and compacted CB (first16, 16 slots).

Runs gemma_cb.run_cb twice in one process (same weights): compact=False, then compact=True,
each recording (task_id, gen_index, token, top2_gap) for every generated token.
For each task: first generated index where the two modes disagree, and the top-2 gap of the
full-mode run at that index (small gap = bf16 near-tie, large gap = real difference).

Usage: diag_gemma_compact_231.py <out_dir> --device npu:1
"""
import argparse
import json
import os
import sys

sys.path.insert(0, "/remote-home/S45160/work/910b_fresh/src")

import torch
import torch_npu  # noqa: F401
from transformers import AutoModelForCausalLM, AutoTokenizer

from gemma_run import DATA, MODEL
from gemma_cb import run_cb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--device", default="npu:1")
    ap.add_argument("--slots", type=int, default=16)
    ap.add_argument("--limit", type=int, default=16)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    qs = [json.loads(l) for l in open(DATA)][: a.limit]
    tok = AutoTokenizer.from_pretrained(MODEL)
    eos_ids = [1, 106, tok.convert_tokens_to_ids("<tool_call|>")]
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 device_map=a.device).eval()

    logs = {}
    for name, compact in [("full", False), ("compact", True)]:
        ml = []
        done, makespan = run_cb(model, tok, qs, eos_ids, a.slots, 2048, a.device,
                                log=lambda *x: None, compact=compact, margin_log=ml)
        logs[name] = {"margin_log": ml, "makespan": makespan,
                      "correct": {r["id"]: r["correct"] for r in done}}
        print(f"{name} done makespan={makespan:.1f}", flush=True)
        # run_cb allocates a fresh KV cache (16 slots x 30 layers) each call. Release the previous
        # one before the next call, otherwise two caches coexist and the NPU runs out of memory.
        del done
        import gc
        gc.collect()
        torch.npu.empty_cache()

    # Per-task alignment. Key = (turn, gen_index within turn), and keep the global order of
    # generated tokens per task. gen_index alone is NOT unique across turns (it resets each turn),
    # so the previous version overwrote entries and compared different turns.
    def per_task(ml):
        d = {}
        for tid, turn, gi, tk, gap in ml:
            d.setdefault(tid, {})[(turn, gi)] = (tk, gap)
        return d

    F, C = per_task(logs["full"]["margin_log"]), per_task(logs["compact"]["margin_log"])
    report = {}
    for tid in sorted(F):
        f, c = F[tid], C.get(tid, {})
        keys = sorted(set(f) | set(c))
        first = None
        for kk in keys:
            if f.get(kk, (None,))[0] != c.get(kk, (None,))[0]:
                first = kk
                break
        entry = {"n_full": len(f), "n_compact": len(c),
                 "correct_full": logs["full"]["correct"].get(tid),
                 "correct_compact": logs["compact"]["correct"].get(tid),
                 "first_divergence_turn_gen": list(first) if first else None}
        if first is not None:
            entry["full_token"] = f.get(first, (None, None))[0]
            entry["compact_token"] = c.get(first, (None, None))[0]
            entry["full_top2_gap_at_divergence"] = f.get(first, (None, None))[1]
            entry["compact_top2_gap_at_divergence"] = c.get(first, (None, None))[1]
        report[tid] = entry

    out = {"full_makespan": logs["full"]["makespan"],
           "compact_makespan": logs["compact"]["makespan"], "tasks": report,
           # raw logs, so future analysis does not need another 25-minute run
           "raw_margin_log": {n: logs[n]["margin_log"] for n in ["full", "compact"]}}
    with open(os.path.join(a.out_dir, "diverg.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("DIVERGENCE", json.dumps({k: v for k, v in report.items()
                                    if v["first_divergence_gen_index"] is not None},
                                   ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
