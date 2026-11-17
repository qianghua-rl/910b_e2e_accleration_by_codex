"""Single-step feasibility test for NPUGraph capture of the Gemma CB decode step.

Config under test: --cb --compact --grouped-experts --fast-norm semantics, but with a FIXED shape:
all N=16 rows computed (idle rows are dummies, step cost is independent of row count per E-G28),
attention length Tact fixed for the whole test (one graph, no bucket switching).

Procedure (same pattern as scripts/test_acl_graph.py for Qwen, E18):
  1. Load model, patch fast-norm, build GemmaCB(grouped=True).
  2. Capture one graph of decode_step(static tok/pos buffers, sl=eng.ar, tact=TACT) BEFORE any prefill,
     after an eager warmup. Capture writes garbage into KV, so K/V are zeroed afterwards.
  3. Prefill the first N prompts into the slots (HF batch-1 prefill, as in run_cb).
  4. For STEPS greedy steps: run eager decode_step and graph replay on identical inputs, compare logits
     bit-for-bit. Both write the same KV values at the same positions, so repeated writes are idempotent.
     The next token is the eager argmax.
  5. Timing: 30 eager steps and 30 graph replays on a fixed input (positions not advanced).

Pass criteria are printed; this script does not decide acceptance (that needs ABBA, per FAIRNESS_AUDIT).

Usage: python -I scripts/test_gemma_graph_step.py [--device npu:1] [--steps 32]
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, "/remote-home/S45160/work/910b_fresh/src")

import torch
import torch_npu  # noqa: F401
from transformers import AutoModelForCausalLM, AutoTokenizer

import agent_protocol as P
from gemma_run import DATA, MODEL
from gemma_cb import GemmaCB
from gemma_fast_norm import patch_gemma_norms

N = 16
T_MAX = 2048
TACT_ALIGN = 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="npu:1")
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--timing-iters", type=int, default=30)
    ap.add_argument("--out", default="/remote-home/S45160/work/910b_fresh/results/gemma/graph_step/result.json")
    a = ap.parse_args()
    dev = a.device

    tok = AutoTokenizer.from_pretrained(MODEL)
    qs = [json.loads(l) for l in open(DATA)][:N]
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map=dev).eval()
    patch_gemma_norms(model)
    eng = GemmaCB(model, tok, N, T_MAX, dev, grouped=True)

    # Prompts (same format as run_cb's first turn)
    prompts = []
    for q in qs:
        msgs = [{"role": "system", "content": P.SYSTEM}, {"role": "user", "content": q["question"]}]
        p = tok.apply_chat_template(msgs, tools=P.TOOLS, add_generation_prompt=True,
                                    tokenize=False, enable_thinking=False)
        prompts.append(tok(p, return_tensors="pt", add_special_tokens=False).input_ids.to(dev))
    max_len = max(int(x.shape[1]) for x in prompts)
    TACT = min(T_MAX, math.ceil((max_len + a.steps + 1) / TACT_ALIGN) * TACT_ALIGN)
    assert max_len + a.steps < TACT + 1 and TACT <= T_MAX, (max_len, TACT)
    print(f"prompts: max_len={max_len}, fixed TACT={TACT}, steps={a.steps}", flush=True)

    tok_buf = torch.zeros(N, dtype=torch.long, device=dev)
    pos_buf = torch.zeros(N, dtype=torch.long, device=dev)
    result = {"device": dev, "N": N, "TACT": TACT, "steps": a.steps, "max_prompt_len": max_len}

    with torch.inference_mode():
        # ---- 2. capture (before any prefill) ----
        t0 = time.perf_counter()
        eng.decode_step(tok_buf, pos_buf, eng.ar, tact=TACT)     # eager warmup, creates lazy helpers
        torch.npu.synchronize(dev)
        g = torch.npu.NPUGraph()
        with torch.npu.graph(g):
            out_static = eng.decode_step(tok_buf, pos_buf, eng.ar, tact=TACT)
        torch.npu.synchronize(dev)
        result["capture_s"] = round(time.perf_counter() - t0, 2)
        print(f"captured graph in {result['capture_s']} s", flush=True)

        # diagnostic: does replay recompute the output? (sentinel test on the static output)
        out_static.fill_(7.0)
        tok_buf.copy_(torch.arange(N, device=dev) + 100)
        pos_buf.copy_(torch.full((N,), 5, dtype=torch.long, device=dev))
        g.replay()
        torch.npu.synchronize(dev)
        result["diag_replay_overwrote_sentinel"] = bool((out_static != 7.0).any().item())
        print(f"diag: replay overwrote sentinel = {result['diag_replay_overwrote_sentinel']}", flush=True)

        # clear garbage written during warmup/capture and diagnostic
        for buf in eng.K + eng.V:
            buf.zero_()

        # ---- 3. prefill ----
        lens, next_tok = [], []
        for s in range(N):
            logits = eng.prefill_into_slot(s, prompts[s])
            next_tok.append(int(logits.argmax().item()))
            lens.append(int(prompts[s].shape[1]))
        torch.npu.synchronize(dev)

        # ---- 4. step-by-step equality ----
        mismatches, argmax_mismatch = [], 0
        for step in range(a.steps):
            tok_buf.copy_(torch.tensor(next_tok, device=dev))
            pos_buf.copy_(torch.tensor(lens, device=dev))
            tok_e, pos_e = tok_buf.clone(), pos_buf.clone()
            logits_e = eng.decode_step(tok_e, pos_e, eng.ar, tact=TACT)
            g.replay()
            torch.npu.synchronize(dev)
            logits_g = out_static
            if not torch.equal(logits_e, logits_g):
                mismatches.append(step)
                d = (logits_e.float() - logits_g.float()).abs().max().item()
                if step == mismatches[0]:
                    result["first_mismatch_max_abs"] = d
            ae = logits_e.argmax(-1).tolist()
            ag = logits_g.argmax(-1).tolist()
            argmax_mismatch += sum(x != y for x, y in zip(ae, ag))
            next_tok = ae
            lens = [x + 1 for x in lens]
        result["bitwise_mismatch_steps"] = mismatches
        result["argmax_mismatches_total"] = argmax_mismatch
        print(f"equality: bitwise-mismatch steps={len(mismatches)}/{a.steps}, "
              f"argmax mismatches={argmax_mismatch}", flush=True)

        # ---- 5. timing on fixed inputs ----
        tok_buf.copy_(torch.tensor(next_tok, device=dev))
        pos_buf.copy_(torch.tensor(lens, device=dev))
        for _ in range(3):
            eng.decode_step(tok_buf, pos_buf, eng.ar, tact=TACT)
            g.replay()
        torch.npu.synchronize(dev)

        t = time.perf_counter()
        for _ in range(a.timing_iters):
            eng.decode_step(tok_buf, pos_buf, eng.ar, tact=TACT)
        torch.npu.synchronize(dev)
        eager_ms = 1000 * (time.perf_counter() - t) / a.timing_iters

        t = time.perf_counter()
        for _ in range(a.timing_iters):
            g.replay()
        torch.npu.synchronize(dev)
        graph_ms = 1000 * (time.perf_counter() - t) / a.timing_iters

        result["eager_ms_per_step"] = round(eager_ms, 2)
        result["graph_ms_per_step"] = round(graph_ms, 2)
        result["graph_vs_eager_pct"] = round(100 * (graph_ms - eager_ms) / eager_ms, 2)
        print(f"timing: eager {eager_ms:.1f} ms/step, graph {graph_ms:.1f} ms/step "
              f"({result['graph_vs_eager_pct']:+.1f}%)", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(result, f, indent=2)
    print("RESULT", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
