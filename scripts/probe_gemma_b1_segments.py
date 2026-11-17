"""Segmented timing of one B=1 decode step: GemmaCB.decode_step vs HF generate().

Question (E-G19 follow-up): at batch 1 the CB decode step costs ~776 ms, while HF generate()
costs ~130-175 ms per token on the same model. Where does the CB step spend its time?

Two measurements on the same prompt, same process, same device:
  1. generate_per_token : HF generate(), (t(65 tokens) - t(1 token)) / 64, no hooks.
  2. cb_step_total      : GemmaCB.decode_step with N=1, no hooks (warm, 64 steps).
  3. cb_step_segments   : the same loop with synchronized timers around each block:
                          attention (GemmaCB._attn), router, experts, dense mlp.
                          remainder = total - attn - router - experts - mlp (norms, residuals,
                          embedding, lm_head, rope, mask build, host copies).
Timers add a sync per call, so segment times are upper bounds; compare segment *shares*, and
use cb_step_total (no hooks) for the absolute number.

Usage: probe_gemma_b1_segments.py <out_dir> --device npu:1 [--steps 64]
Output: <out_dir>/segments.json
"""
import argparse
import json
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


def build_prompt_ids(tok, device):
    q = [json.loads(l) for l in open(DATA)][0]
    msgs = [{"role": "system", "content": P.SYSTEM}, {"role": "user", "content": q["question"]}]
    p = tok.apply_chat_template(msgs, tools=P.TOOLS, add_generation_prompt=True,
                                tokenize=False, enable_thinking=False)
    return tok(p, return_tensors="pt", add_special_tokens=False).input_ids.to(device)


def time_generate(model, ids, n_new):
    torch.npu.synchronize(model.device)
    t = time.perf_counter()
    with torch.no_grad():
        model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=n_new,
                       do_sample=False, pad_token_id=0)
    torch.npu.synchronize(model.device)
    return time.perf_counter() - t


def cb_decode_loop(eng, ids, steps, device):
    """Prefill slot 0, then `steps` greedy decode steps. Returns per-step seconds."""
    logits = eng.prefill_into_slot(0, ids)
    tok_id = int(logits.argmax().item())
    pos = int(ids.shape[1])
    times = []
    for _ in range(steps):
        torch.npu.synchronize(device)
        t = time.perf_counter()
        logits = eng.decode_step(torch.tensor([tok_id], device=device),
                                 torch.tensor([pos], device=device))
        tok_id = int(logits.argmax(-1).item())
        torch.npu.synchronize(device)
        times.append(time.perf_counter() - t)
        pos += 1
    return times


def install_segment_timers(eng, model, acc, device):
    """Wrap attention, router, experts, dense mlp of every layer. Adds a sync per call."""
    def wrap(fn, key):
        def inner(*a, **k):
            torch.npu.synchronize(device)
            t = time.perf_counter()
            out = fn(*a, **k)
            torch.npu.synchronize(device)
            acc[key] += time.perf_counter() - t
            acc[key + "_calls"] += 1
            return out
        return inner

    for layer in model.model.language_model.layers:
        layer.self_attn.forward  # ensure exists
        layer_attn_fn = None  # attention is GemmaCB._attn, wrapped at the engine level below
        layer.mlp.forward = wrap(layer.mlp.forward, "mlp")
        if getattr(layer, "enable_moe_block", False):
            layer.router.forward = wrap(layer.router.forward, "router")
            layer.experts.forward = wrap(layer.experts.forward, "experts")
    orig_attn = eng._attn

    def attn_timed(li, layer, h, pos, cos, sin, mask):
        torch.npu.synchronize(device)
        t = time.perf_counter()
        out = orig_attn(li, layer, h, pos, cos, sin, mask)
        torch.npu.synchronize(device)
        acc["attn"] += time.perf_counter() - t
        acc["attn_calls"] += 1
        return out

    eng._attn = attn_timed


def mean_ms(xs):
    return round(1000 * sum(xs) / len(xs), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--device", default="npu:1")
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--t-max", type=int, default=2048)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 device_map=a.device).eval()
    ids = build_prompt_ids(tok, a.device)
    L = int(ids.shape[1])
    result = {"prompt_tokens": L, "steps": a.steps, "device": a.device}

    # 1. HF generate per token (no hooks). Warm up once, then difference of two lengths.
    time_generate(model, ids, 8)
    t1 = time_generate(model, ids, 1)
    t65 = time_generate(model, ids, 65)
    result["generate_per_token_ms"] = round(1000 * (t65 - t1) / 64, 2)

    # 2. CB decode, N=1, no hooks.
    eng = GemmaCB(model, tok, 1, a.t_max, a.device)
    cb_decode_loop(eng, ids, a.warmup, a.device)           # warm up
    times = cb_decode_loop(eng, ids, a.steps, a.device)
    result["cb_step_total_ms"] = mean_ms(times)
    result["cb_step_total_ms_median"] = round(1000 * sorted(times)[len(times) // 2], 2)

    # 3. CB decode, N=1, with segment timers (fresh engine: timers wrap methods in place).
    acc = {"attn": 0.0, "attn_calls": 0, "router": 0.0, "router_calls": 0,
           "experts": 0.0, "experts_calls": 0, "mlp": 0.0, "mlp_calls": 0}
    install_segment_timers(eng, model, acc, a.device)
    cb_decode_loop(eng, ids, a.warmup, a.device)
    for k in acc:
        acc[k] = 0.0 if not k.endswith("_calls") else 0
    times_seg = cb_decode_loop(eng, ids, a.steps, a.device)
    total = sum(times_seg)
    per_step = {k: round(1000 * acc[k] / a.steps, 2) for k in ["attn", "router", "experts", "mlp"]}
    calls_per_step = {k: round(acc[k + "_calls"] / a.steps, 2)
                      for k in ["attn", "router", "experts", "mlp"]}
    rest = 1000 * (total - acc["attn"] - acc["router"] - acc["experts"] - acc["mlp"]) / a.steps
    result["cb_step_segmented_total_ms"] = round(1000 * total / a.steps, 2)
    result["segments_ms_per_step"] = per_step
    result["segment_calls_per_step"] = calls_per_step
    result["remainder_ms_per_step"] = round(rest, 2)
    result["segment_share_of_step"] = {
        k: round(acc[k] / total, 4) for k in ["attn", "router", "experts", "mlp"]}
    result["remainder_share_of_step"] = round((total - sum(acc[k] for k in
                                                            ["attn", "router", "experts", "mlp"]))
                                               / total, 4)
    result["peak_allocated_bytes"] = torch.npu.max_memory_allocated(a.device)

    with open(os.path.join(a.out_dir, "segments.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("SEGMENTS", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
