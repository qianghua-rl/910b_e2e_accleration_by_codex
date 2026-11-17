"""Segmented timing of one decode step on the CURRENT best config (E-G23):
  --cb --compact --grouped-experts --fast-norm, N=16 active slots (first 16 questions).

Question: where does the remaining per-step time go now (~246 s / first16)?

Measurements, same process/device:
  1. step_total (no hooks): mean decode_step time over `steps` steps, all 16 slots active.
  2. segments (sync timers, upper bounds): attn (GemmaCB._attn), experts (experts_grouped),
     router, dense mlp. remainder = total - those (norms, residuals, embed, lm_head, rope,
     mask build, argmax/host copies).
Segment *shares* are the useful output; use step_total for the absolute number.

Usage: probe_gemma_stack_segments.py <out_dir> --device npu:1 [--steps 32]
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
import gemma_cb
from gemma_cb import GemmaCB, experts_grouped
from gemma_fast_norm import patch_gemma_norms
from gemma_run import DATA, MODEL


def prompt_ids(tok, device, q):
    msgs = [{"role": "system", "content": P.SYSTEM}, {"role": "user", "content": q["question"]}]
    p = tok.apply_chat_template(msgs, tools=P.TOOLS, add_generation_prompt=True,
                                tokenize=False, enable_thinking=False)
    return tok(p, return_tensors="pt", add_special_tokens=False).input_ids.to(device)


def decode_loop(eng, first_tokens, lens, steps, device, sl):
    """All slots in `sl` active. Returns per-step seconds."""
    toks = torch.tensor(first_tokens, device=device)
    pos = torch.tensor(lens, device=device)
    slt = torch.tensor(sl, device=device)
    times = []
    for _ in range(steps):
        torch.npu.synchronize(device)
        t = time.perf_counter()
        logits = eng.decode_step(toks, pos, slt)
        toks = logits.argmax(-1)
        torch.npu.synchronize(device)
        times.append(time.perf_counter() - t)
        pos = pos + 1
    return times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--device", default="npu:1")
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--slots", type=int, default=16)
    ap.add_argument("--t-max", type=int, default=2048)
    ap.add_argument("--fast-router", action="store_true")
    ap.add_argument("--fast-norm-noscale", action="store_true")
    ap.add_argument("--fused-rope", action="store_true")
    ap.add_argument("--prefill-grouped", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    dev, N = a.device, a.slots

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 device_map=dev).eval()
    n_patched = patch_gemma_norms(model, noscale=a.fast_norm_noscale)
    qs = [json.loads(l) for l in open(DATA)][:N]
    eng = GemmaCB(model, tok, N, a.t_max, dev, grouped=True, fast_router=a.fast_router,
                  fused_rope=a.fused_rope, prefill_grouped=a.prefill_grouped)
    first, lens = [], []
    for s, q in enumerate(qs):
        ids = prompt_ids(tok, dev, q)
        logits = eng.prefill_into_slot(s, ids)
        first.append(int(logits.argmax().item()))
        lens.append(int(ids.shape[1]))
    sl = list(range(N))

    # 1. unhooked step time
    decode_loop(eng, first, lens, a.warmup, dev, sl)
    times = decode_loop(eng, first, lens, a.steps, dev, sl)
    total_ms = 1000 * sum(times) / len(times)

    # 2. segment timers (fresh counters; wrap module-level experts_grouped and engine methods)
    acc = {"attn": 0.0, "experts": 0.0, "router": 0.0, "mlp": 0.0}

    def timed(fn, key):
        def inner(*args, **kw):
            torch.npu.synchronize(dev)
            t = time.perf_counter()
            out = fn(*args, **kw)
            torch.npu.synchronize(dev)
            acc[key] += time.perf_counter() - t
            return out
        return inner

    gemma_cb.experts_grouped = timed(experts_grouped, "experts")
    orig_attn = eng._attn
    eng._attn = timed(orig_attn, "attn")
    for layer in model.model.language_model.layers:
        layer.mlp.forward = timed(layer.mlp.forward, "mlp")
        if getattr(layer, "enable_moe_block", False):
            if not a.fast_router:
                layer.router.forward = timed(layer.router.forward, "router")
            else:
                pass  # fused path calls router_fused directly; timed below via engine wrapper
    if a.fast_router:
        import gemma_cb as _g
        _orig_rf = _g.router_fused
        def _rf(r, x):
            return timed(_orig_rf, "router")(r, x)
        _g.router_fused = _rf

    decode_loop(eng, first, lens, a.warmup, dev, sl)
    for k in acc:
        acc[k] = 0.0
    times_seg = decode_loop(eng, first, lens, a.steps, dev, sl)
    total_seg = sum(times_seg)
    seg_ms = {k: round(1000 * acc[k] / a.steps, 2) for k in acc}
    rest_ms = round(1000 * (total_seg - sum(acc.values())) / a.steps, 2)

    result = {"config": "cb+grouped+fast_norm" + ("+noscale" if a.fast_norm_noscale else "") + ("+fused_rope" if a.fused_rope else "") + ("+prefill_grouped" if a.prefill_grouped else "") + ("+fast_router" if a.fast_router else ""), "N": N, "steps": a.steps,
              "patched_norms": n_patched, "step_total_ms": round(total_ms, 2),
              "step_segmented_total_ms": round(1000 * total_seg / a.steps, 2),
              "segments_ms_per_step": seg_ms, "remainder_ms_per_step": rest_ms,
              "segment_share": {k: round(acc[k] / total_seg, 4) for k in acc},
              "remainder_share": round((total_seg - sum(acc.values())) / total_seg, 4),
              "peak_allocated_bytes": torch.npu.max_memory_allocated(dev)}
    with open(os.path.join(a.out_dir, "segments.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("STACK_SEGMENTS", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
