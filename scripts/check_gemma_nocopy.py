"""Check GemmaCB slice_kv (K/V read as views, all slots active) vs the gather path, and time both.
Same engine, same cache state: each step runs both variants at the same position (both write
identical K/V for that position), compares logits, then advances with the expand-path argmax.
Usage: check_gemma_nocopy.py <out_dir> --device npu:1 [--steps 16]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch
import torch_npu  # noqa: F401
from transformers import AutoModelForCausalLM, AutoTokenizer

import agent_protocol as P
from gemma_cb import GemmaCB
from gemma_fast_norm import patch_gemma_norms
from gemma_run import DATA, MODEL


def prompt_ids(tok, device, q):
    msgs = [{"role": "system", "content": P.SYSTEM}, {"role": "user", "content": q["question"]}]
    p = tok.apply_chat_template(msgs, tools=P.TOOLS, add_generation_prompt=True,
                                tokenize=False, enable_thinking=False)
    return tok(p, return_tensors="pt", add_special_tokens=False).input_ids.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--device", default="npu:1")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--slots", type=int, default=16)
    ap.add_argument("--t-max", type=int, default=2048)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    dev, N = a.device, a.slots
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 device_map=dev).eval()
    patch_gemma_norms(model)
    qs = [json.loads(l) for l in open(DATA)][:N]
    eng = GemmaCB(model, tok, N, a.t_max, dev, grouped=True, nocopy=False)
    first, lens = [], []
    for s, q in enumerate(qs):
        ids = prompt_ids(tok, dev, q)
        first.append(int(eng.prefill_into_slot(s, ids).argmax().item()))
        lens.append(int(ids.shape[1]))
    sl = eng.ar  # identity object, as run_cb passes for all-active steps
    toks = torch.tensor(first, device=dev)
    pos = torch.tensor(lens, device=dev)

    def run(slice_kv):
        eng.slice_kv = slice_kv
        torch.npu.synchronize(dev)
        t = time.perf_counter()
        lg = eng.decode_step(toks, pos, sl)
        torch.npu.synchronize(dev)
        return lg, time.perf_counter() - t

    diffs, flips, t_exp, t_nc = [], 0, [], []
    for step in range(a.steps):
        lg_e, te = run(False)
        lg_n, tn = run(True)
        if step >= 2:  # skip warm-up steps for timing
            t_exp.append(te)
            t_nc.append(tn)
        d = (lg_e.float() - lg_n.float()).abs().max().item()
        diffs.append(d)
        flips += int((lg_e.argmax(-1) != lg_n.argmax(-1)).sum().item())
        toks = lg_e.argmax(-1)
        pos = pos + 1
    res = {"N": N, "steps": a.steps, "max_logit_absdiff": max(diffs),
           "argmax_flips_over_steps": flips,
           "gather_step_ms": round(1000 * sum(t_exp) / len(t_exp), 2),
           "slice_step_ms": round(1000 * sum(t_nc) / len(t_nc), 2)}
    with open(os.path.join(a.out_dir, "slice_check.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("SLICE_CHECK", json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
