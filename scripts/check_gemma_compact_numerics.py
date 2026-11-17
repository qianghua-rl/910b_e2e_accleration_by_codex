"""Numerical check: active-row compaction vs full-slot decode, on the same state.

Question: the compaction ABBA showed a flip on task 231 (13/16 vs 14/16). Is it
  (a) a logic bug in the compacted path, or
  (b) a numerical effect of computing a different batch shape / attention length?

Method (16 slots, first16 prompts, 40 greedy steps along the full-decode trajectory):
  per step, from the same KV state:
    full    : decode_step(all 16 rows)                     -> logits_full
    control : decode_step(sl = all 16 rows, explicit)      -> must equal logits_full bit-for-bit
    subset  : decode_step(sl = random 8 rows)              -> compare with logits_full[sl]
  KV written by each call is restored from a snapshot of the touched positions, so the
  three calls see identical state. The trajectory is then advanced with one full call.

Reading the result:
  - control must be exact (diff 0). If not, the sl indexing path itself is wrong (bug).
  - subset differences at bf16 scale with rare argmax flips  -> numerical (shape / Tact effect).
  - large differences (>> bf16 ulp at logit scale ~26) or many flips -> logic bug.

Usage: check_gemma_compact_numerics.py <out_dir> --device npu:1
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, "/remote-home/S45160/work/910b_fresh/src")

import torch
import torch_npu  # noqa: F401
from transformers import AutoModelForCausalLM, AutoTokenizer

import agent_protocol as P
from gemma_run import DATA, MODEL
from gemma_cb import GemmaCB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--device", default="npu:1")
    ap.add_argument("--slots", type=int, default=16)
    ap.add_argument("--subset", type=int, default=8)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    N, dev = a.slots, a.device

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 device_map=dev).eval()
    eng = GemmaCB(model, tok, N, 2048, dev)
    ar = eng.ar

    qs = [json.loads(l) for l in open(DATA)][:N]
    toks, pos = [], []
    for s, q in enumerate(qs):
        msgs = [{"role": "system", "content": P.SYSTEM}, {"role": "user", "content": q["question"]}]
        p = tok.apply_chat_template(msgs, tools=P.TOOLS, add_generation_prompt=True,
                                    tokenize=False, enable_thinking=False)
        ids = tok(p, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
        logits = eng.prefill_into_slot(s, ids)
        toks.append(int(logits.argmax().item()))
        pos.append(int(ids.shape[1]))

    def snapshot(p):
        return [(eng.K[li][ar, :, p].clone(), eng.V[li][ar, :, p].clone())
                for li in range(len(eng.K))]

    def restore(p, snap):
        for li, (k, v) in enumerate(snap):
            eng.K[li][ar, :, p] = k
            eng.V[li][ar, :, p] = v

    rng = random.Random(a.seed)
    rows = []
    for step in range(a.steps):
        t = torch.tensor(toks, device=dev)
        p = torch.tensor(pos, device=dev)
        snap = snapshot(p)

        lf = eng.decode_step(t, p).float()                 # full
        restore(p, snap)

        ctrl = eng.decode_step(t, p, ar).float()           # control: explicit all-row sl
        restore(p, snap)
        ctrl_diff = (ctrl - lf).abs().max().item()

        sl_list = sorted(rng.sample(range(N), a.subset))
        sl = torch.tensor(sl_list, device=dev)
        ls = eng.decode_step(t[sl], p[sl], sl).float()     # subset
        restore(p, snap)

        lf_s = lf[sl]
        diff = (lf_s - ls).abs().max().item()
        am_f = lf_s.argmax(-1)
        am_s = ls.argmax(-1)
        flip_idx = (am_f != am_s).nonzero().flatten().tolist()
        flip_margins = []
        for i in flip_idx:
            top2 = torch.topk(lf_s[i], 2).values
            flip_margins.append(round((top2[0] - top2[1]).item(), 4))
        exact_rows = int(((lf_s == ls).all(-1)).sum().item())
        rows.append({"step": step, "pos_max": max(pos), "subset_rows": sl_list,
                     "control_max_diff": ctrl_diff, "subset_max_diff": round(diff, 6),
                     "subset_exact_rows": exact_rows, "argmax_flips": len(flip_idx),
                     "flip_full_top2_margin": flip_margins})

        # Advance the trajectory with the full decode (KV for this step is written here).
        lf_commit = eng.decode_step(t, p).float()
        toks = lf_commit.argmax(-1).tolist()
        pos = [x + 1 for x in pos]

    ctrl_max = max(r["control_max_diff"] for r in rows)
    sub_max = max(r["subset_max_diff"] for r in rows)
    flips = sum(r["argmax_flips"] for r in rows)
    margins = [m for r in rows for m in r["flip_full_top2_margin"]]
    summary = {
        "steps": a.steps, "subset": a.subset, "slots": N,
        "control_max_diff_all_steps": ctrl_max,
        "subset_max_diff_all_steps": sub_max,
        "subset_exact_rows_mean": round(sum(r["subset_exact_rows"] for r in rows) / len(rows), 2),
        "argmax_flips_total": flips,
        "flip_full_top2_margins": margins,
        "rows": rows,
    }
    with open(os.path.join(a.out_dir, "numerics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("NUMERICS", json.dumps({k: v for k, v in summary.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()
