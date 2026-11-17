"""Minimal reproduction of the task-231 divergence at its first decode step.

Setup: 16 slots, all first16 prompts prefilled (same as the CB run). Then, from this one state,
for the first decode step, compare the logits of each row computed three ways:
  full      : decode_step(all 16 rows, sl=None)           -> reference
  explicit  : decode_step(all 16 rows, sl=arange(16))     -> must equal full bit-for-bit
  single    : decode_step(only row k, sl=[k])             -> compact path for one active slot
Both `full` and `single` write the same KV values at the same positions (idempotent), so no
snapshot is needed. Also reports Tact each path uses, and the prompt length of each task.

Reading:
  - explicit == full for all rows, and single == full for row k -> compact path is numerically
    identical here, and the 231 flip comes from somewhere else in the run.
  - single != full for row k with a large gap -> bug in the sl path or in Tact/mask for sl.

Usage: diag_gemma_231_step1.py <out_dir> --device npu:1
"""
import argparse
import json
import os
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
    ap.add_argument("--target", type=int, default=231)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    dev, N = a.device, 16

    qs = [json.loads(l) for l in open(DATA)][:N]
    ids_of = [q["id"] for q in qs]
    k = ids_of.index(a.target)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 device_map=dev).eval()
    eng = GemmaCB(model, tok, N, 2048, dev)

    toks, pos, plen = [], [], []
    for s, q in enumerate(qs):
        msgs = [{"role": "system", "content": P.SYSTEM}, {"role": "user", "content": q["question"]}]
        p = tok.apply_chat_template(msgs, tools=P.TOOLS, add_generation_prompt=True,
                                    tokenize=False, enable_thinking=False)
        ids = tok(p, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
        logits = eng.prefill_into_slot(s, ids)
        toks.append(int(logits.argmax().item()))
        pos.append(int(ids.shape[1]))
        plen.append(int(ids.shape[1]))

    t_all = torch.tensor(toks, device=dev)
    p_all = torch.tensor(pos, device=dev)

    full = eng.decode_step(t_all, p_all).float()
    explicit = eng.decode_step(t_all, p_all, torch.arange(N, device=dev)).float()
    sl = torch.tensor([k], device=dev)
    single = eng.decode_step(t_all[sl], p_all[sl], sl).float()

    row_full = full[k]
    res = {
        "target": a.target, "row": k,
        "prompt_lengths": dict(zip(ids_of, plen)),
        "target_prompt_len": plen[k],
        "tact_full": min(eng.T, ((max(pos) + 1 + 63) // 64) * 64),
        "tact_single": min(eng.T, ((pos[k] + 1 + 63) // 64) * 64),
        "explicit_vs_full_max_diff_all_rows": (explicit - full).abs().max().item(),
        "single_vs_full_row_max_diff": (single[0] - row_full).abs().max().item(),
        "full_argmax": int(row_full.argmax().item()),
        "single_argmax": int(single[0].argmax().item()),
        "full_top2": torch.topk(row_full, 2).values.tolist(),
        "single_top2": torch.topk(single[0], 2).values.tolist(),
    }
    with open(os.path.join(a.out_dir, "step1.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("STEP1", json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
