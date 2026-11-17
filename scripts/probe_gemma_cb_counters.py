"""Counter probe for the Gemma CB decode loop (instrumentation only, numerics unchanged).

Re-runs the exact loop of gemma_cb.run_cb and adds counters that answer the open questions in
experiments/gemma_records.jsonl (E-G14, E-G15, E-G17):

  per decode step : wall time (device-synced), active slots, Tact (attention length)
  prefill         : count and time (HF batch-1 prefill into a slot, blocks all slots)
  tool execution  : count, time, timeouts (P.execute is synchronous and blocks all slots)
  occupancy       : idle slot-steps, mean active slots, tail steps (active <= N/4)
  step cost vs N  : mean step time grouped by active-slot count (tests E-G15)
  host overhead   : makespan - decode - prefill - tool
  expert hits     : optional (--expert-hits): distinct experts per MoE layer per decode step.
                    This adds a sync per layer; use only on short runs.

Self-check: the default first16 / slots 16 run must reproduce E-G16 (7806 generated tokens,
14/16 with fresh-v1 scoring). If the token count differs, the probe loop has drifted from run_cb.

Usage:
  probe_gemma_cb_counters.py <out_dir> --device npu:1 --slots 16 --limit 16 [--expert-hits] [--fast-norm]

Outputs: <out_dir>/probe.json (summary + per-step rows), <out_dir>/per_task.jsonl
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, "/remote-home/S45160/work/910b_fresh/src")  # same convention as gemma_run.py

import torch
import torch_npu  # noqa: F401
from transformers import AutoModelForCausalLM, AutoTokenizer

import agent_protocol as P
from gemma_run import DATA, MAX_NEW, MAX_TURNS, MODEL, parse_gemma_call, strip_specials
from gemma_cb import GemmaCB
from gemma_fast_norm import patch_gemma_norms


def install_expert_hook(model, S):
    """Record distinct experts hit per MoE layer, during decode steps only."""
    for layer in model.model.language_model.layers:
        if not getattr(layer, "enable_moe_block", False):
            continue

        def hook(mod, args):
            if S["in_decode"]:
                idx = args[1]                       # top_k_index [tokens, top_k]
                S["expert_hits"].append(int(torch.unique(idx).numel()))

        layer.experts.register_forward_pre_hook(hook)


def run_probe(eng, tok, qs, eos_ids, device, n_slots, log, S, compact=False):
    eos = set(eos_ids)
    queue = list(qs)
    slots = [None] * n_slots
    done = []
    S.update({"prefill_n": 0, "prefill_s": 0.0,
         "prefill_tokens": 0, "tool_n": 0, "tool_s": 0.0, "tool_timeouts": 0,
         "steps": 0, "decode_s": 0.0, "idle_slot_steps": 0, "active_slot_steps": 0,
         "step_rows": []})  # rows: [n_active, Tact, seconds]

    def timed_prefill(s, ids):
        torch.npu.synchronize(device)
        t = time.perf_counter()
        logits = eng.prefill_into_slot(s, ids)
        torch.npu.synchronize(device)
        S["prefill_s"] += time.perf_counter() - t
        S["prefill_n"] += 1
        S["prefill_tokens"] += int(ids.shape[1])
        return logits

    def timed_tool(code):
        t = time.perf_counter()
        out = P.execute(code)
        S["tool_s"] += time.perf_counter() - t
        S["tool_n"] += 1
        if out.startswith("Python execution timed out"):
            S["tool_timeouts"] += 1
        return out

    t_all = time.perf_counter()

    def start_turn(s, task):
        p = tok.apply_chat_template(task["msgs"], tools=P.TOOLS, add_generation_prompt=True,
                                    tokenize=False, enable_thinking=False)
        ids = tok(p, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        logits = timed_prefill(s, ids)
        first = int(logits.argmax().item())
        slots[s] = {"task": task, "gen": [first], "len": int(ids.shape[1]), "next": first,
                    "prompt_tokens": int(ids.shape[1])}
        if first in eos or len(slots[s]["gen"]) >= MAX_NEW:
            finish_turn(s)

    def finish_turn(s):
        st = slots[s]
        task = st["task"]
        raw = tok.decode(st["gen"], skip_special_tokens=False)
        if "<|tool_call>" in raw and "<tool_call|>" not in raw:
            raw = raw + "<tool_call|>"
        task["final_text"] = strip_specials(raw)
        rec = {"turn": len(task["turns"]), "prompt_tokens": st["prompt_tokens"],
               "gen_tokens": len(st["gen"])}
        task["n_tokens"] += len(st["gen"])
        code = parse_gemma_call(raw)
        if code is None:
            task["turns"].append(rec)
            return finish_task(s)
        out_text = timed_tool(code)          # blocks every slot, as in run_cb
        rec["tool_output"] = out_text
        task["turns"].append(rec)
        if len(task["turns"]) >= MAX_TURNS:
            return finish_task(s)
        task["msgs"].append({"role": "assistant", "content": "",
                             "tool_calls": [{"type": "function",
                                             "function": {"name": "python",
                                                          "arguments": {"code": code}}}]})
        task["msgs"].append({"role": "tool", "name": "python", "content": out_text})
        start_turn(s, task)

    def finish_task(s):
        task = slots[s]["task"]
        slots[s] = None
        pred = P.answer(task["final_text"])
        ok = P.correct(pred, task["q"]["answer"])
        done.append({"id": task["q"]["id"], "gold": task["q"]["answer"], "pred": pred,
                     "correct": ok, "turns": task["turns"], "n_tokens": task["n_tokens"],
                     "final_text": task["final_text"]})
        log(f"[{len(done)}/{len(qs)}] id={task['q']['id']} correct={ok} "
            f"turns={len(task['turns'])} tok={task['n_tokens']} "
            f"elapsed={time.perf_counter() - t_all:.0f}s")
        refill()

    def new_task(q):
        return {"q": q, "msgs": [{"role": "system", "content": P.SYSTEM},
                                 {"role": "user", "content": q["question"]}],
                "turns": [], "n_tokens": 0, "final_text": ""}

    def refill():
        for s in range(n_slots):
            if slots[s] is None and queue:
                start_turn(s, new_task(queue.pop(0)))

    refill()
    while any(st is not None for st in slots):
        active = [s for s in range(n_slots) if slots[s] is not None]
        pos_active = [slots[s]["len"] for s in active]
        # Tact mirrors GemmaCB.decode_step (round up to 64, capped at t_max), over the rows computed.
        tact = min(eng.T, ((max(pos_active) + 1 + 63) // 64) * 64)

        torch.npu.synchronize(device)
        t = time.perf_counter()
        S["in_decode"] = True
        if compact:
            # Mirrors run_cb(compact=True): only active rows are computed, in slot order.
            full = len(active) == n_slots
            sl_t = eng.ar if full else torch.tensor(active, device=device)
            logits = eng.decode_step(torch.tensor([slots[s]["next"] for s in active], device=device),
                                     torch.tensor(pos_active, device=device), sl_t)
            nxt = dict(zip(active, logits.argmax(-1).tolist()))   # host copy = sync point
        else:
            tokens = [slots[s]["next"] if slots[s] is not None else 0 for s in range(n_slots)]
            pos = [slots[s]["len"] if slots[s] is not None else 0 for s in range(n_slots)]
            logits = eng.decode_step(torch.tensor(tokens, device=device),
                                     torch.tensor(pos, device=device))
            nxt_all = logits.argmax(-1).tolist()
            nxt = {s: nxt_all[s] for s in active}
        S["in_decode"] = False
        dt = time.perf_counter() - t

        S["steps"] += 1
        S["decode_s"] += dt
        S["active_slot_steps"] += len(active)
        S["idle_slot_steps"] += n_slots - len(active)
        S["step_rows"].append([len(active), tact, round(dt, 6)])

        for s in active:
            st = slots[s]
            t_new = nxt[s]
            st["gen"].append(t_new)
            st["len"] += 1
            st["next"] = t_new
            if t_new in eos or len(st["gen"]) >= MAX_NEW:
                finish_turn(s)

    makespan = time.perf_counter() - t_all
    return done, makespan, S


def summarise(done, makespan, S, n_slots):
    rows = S["step_rows"]
    by_active = {}
    for n_act, _, dt in rows:
        by_active.setdefault(n_act, []).append(dt)
    step_cost_by_active = {k: {"steps": len(v), "mean_ms": round(1000 * sum(v) / len(v), 1)}
                           for k, v in sorted(by_active.items())}
    tail = [r for r in rows if r[0] <= n_slots // 4]
    decode_tokens = sum(r[0] for r in rows)   # one generated token per active slot per step
    total_tokens = sum(r["n_tokens"] for r in done)
    host_overhead = makespan - S["decode_s"] - S["prefill_s"] - S["tool_s"]
    out = {
        "n_slots": n_slots,
        "tasks": len(done),
        "correct_agent_protocol": sum(r["correct"] for r in done),
        "total_generated_tokens": total_tokens,
        "makespan_s": round(makespan, 2),
        "decode": {
            "steps": S["steps"],
            "decode_s": round(S["decode_s"], 2),
            "mean_step_ms": round(1000 * S["decode_s"] / max(1, S["steps"]), 1),
            "decode_tokens": decode_tokens,
            "mean_active_slots": round(S["active_slot_steps"] / max(1, S["steps"]), 2),
            "occupancy": round(S["active_slot_steps"] / max(1, S["steps"] * n_slots), 4),
            "idle_slot_steps": S["idle_slot_steps"],
            "tail_steps_active_le_quarter": len(tail),
            "tail_decode_s": round(sum(r[2] for r in tail), 2),
            "lower_bound_steps": math.ceil(decode_tokens / n_slots),
            "step_cost_by_active": step_cost_by_active,
        },
        "prefill": {"count": S["prefill_n"], "seconds": round(S["prefill_s"], 2),
                    "prompt_tokens": S["prefill_tokens"]},
        "tool": {"count": S["tool_n"], "seconds": round(S["tool_s"], 2),
                 "timeouts": S["tool_timeouts"]},
        "host_overhead_s": round(host_overhead, 2),
        "fractions_of_makespan": {
            "decode": round(S["decode_s"] / makespan, 4),
            "prefill": round(S["prefill_s"] / makespan, 4),
            "tool": round(S["tool_s"] / makespan, 4),
            "host_overhead": round(host_overhead / makespan, 4),
        },
    }
    if S.get("expert_hits"):
        hits = S["expert_hits"]
        out["experts"] = {"samples": len(hits),
                          "mean_distinct_per_layer_step": round(sum(hits) / len(hits), 2),
                          "max": max(hits)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--device", default="npu:1")
    ap.add_argument("--slots", type=int, default=16)
    ap.add_argument("--t-max", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=16)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--fast-norm", action="store_true")
    ap.add_argument("--compact", action="store_true")
    ap.add_argument("--grouped-experts", action="store_true")
    ap.add_argument("--expert-hits", action="store_true")
    ap.add_argument("--fast-norm-noscale", action="store_true")
    ap.add_argument("--fused-rope", action="store_true")
    ap.add_argument("--prefill-grouped", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    qs = [json.loads(l) for l in open(DATA)][: a.limit]
    tok = AutoTokenizer.from_pretrained(a.model)
    eos_ids = [1, 106, tok.convert_tokens_to_ids("<tool_call|>")]
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                                 device_map=a.device).eval()
    if a.fast_norm:
        patch_gemma_norms(model, noscale=a.fast_norm_noscale)
    eng = GemmaCB(model, tok, a.slots, a.t_max, a.device, grouped=a.grouped_experts,
                  fused_rope=a.fused_rope, prefill_grouped=a.prefill_grouped)

    # One shared state dict: the expert hook and run_probe must see the same in_decode flag.
    S = {"in_decode": False, "expert_hits": []}
    if a.expert_hits:
        install_expert_hook(model, S)

    done, makespan, S = run_probe(eng, tok, qs, eos_ids, a.device, a.slots, print, S,
                                  compact=a.compact)
    done.sort(key=lambda r: r["id"])
    with open(os.path.join(a.out_dir, "per_task.jsonl"), "w") as f:
        for r in done:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = summarise(done, makespan, S, a.slots)
    summary["args"] = {"limit": a.limit, "slots": a.slots, "t_max": a.t_max,
                       "fast_norm": a.fast_norm, "compact": a.compact,
                       "grouped_experts": a.grouped_experts, "expert_hits": a.expert_hits,
                       "fast_norm_noscale": a.fast_norm_noscale, "fused_rope": a.fused_rope,
                       "prefill_grouped": a.prefill_grouped}
    summary["peak_allocated_bytes"] = torch.npu.max_memory_allocated(a.device)
    summary["step_rows"] = S["step_rows"]
    with open(os.path.join(a.out_dir, "probe.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("PROBE", json.dumps({k: v for k, v in summary.items() if k != "step_rows"}), flush=True)


if __name__ == "__main__":
    main()
