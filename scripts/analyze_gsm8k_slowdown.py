"""Analyse why CB gsm8k_longctx_128 is slower than vLLM (and than 12-slot CB).

Inputs: per_task.jsonl files (CB or vLLM). Output: one JSON per run, plus a printed table.
Usage: python3 -I scripts/analyze_gsm8k_slowdown.py <out_dir> <name>=<per_task.jsonl> [...]
"""
import json, os, statistics as st, sys


def load(path):
    meta, summary, tasks = None, None, []
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        k = d.get("kind")
        if k == "metadata":
            meta = d
        elif k == "summary":
            summary = d
        elif k == "task":
            tasks.append(d)
    return meta, summary, tasks


def analyse(path):
    meta, summ, tasks = load(path)
    out = {"file": path, "n_tasks": len(tasks),
           "correct_inline": sum(t.get("correct") in (True, "True") for t in tasks)}
    rounds = []
    for t in tasks:
        R = t["rounds"]
        rounds.extend(eval(R) if isinstance(R, str) else R)
    gen_tokens = sum(len(r["tokens"]) for r in rounds)
    out["gen_tokens_total"] = gen_tokens
    out["rounds_total"] = len(rounds)
    out["makespan_max_completion_s"] = max(float(t["completion_seconds"]) for t in tasks)
    if summ and "decode_seconds" not in summ:  # eager CB runs: summary has no phase timing
        out["makespan_summary_s"] = summ["makespan_seconds"]
        out["prefill_seconds"] = summ.get("prefill_seconds")
        out["decode_steps"] = summ.get("decode_steps")
        out["note"] = "summary without decode_seconds/decode_log (eager run, older runner)"
    elif summ:  # CB runner summary
        slots = meta.get("slots") or meta.get("batch_size")
        out["slots"] = slots
        out["makespan_summary_s"] = summ["makespan_seconds"]
        out["prefill_seconds"] = summ["prefill_seconds"]
        out["decode_seconds"] = summ["decode_seconds"]
        out["decode_steps"] = summ["decode_steps"]
        out["prefill_calls"] = summ["prefill_calls"]
        out["other_seconds"] = round(summ["makespan_seconds"] - summ["prefill_seconds"] - summ["decode_seconds"], 1)
        out["peak_gib"] = round(summ["peak_bytes"] / 2**30, 2)
        # decode: tokens produced by decode steps = gen tokens minus one prefill-produced token per round
        dec_tok = gen_tokens - len(rounds)
        out["decode_tokens"] = dec_tok
        out["slot_occupancy"] = round(dec_tok / (summ["decode_steps"] * slots), 4)
        out["ms_per_step_mean"] = round(1000 * summ["decode_seconds"] / summ["decode_steps"], 1)
        pl = summ.get("prefill_log", [])
        lens = [p[1] for p in pl]
        secs = [p[2] for p in pl]
        out["prefill_prompt_tokens"] = sum(lens)
        out["prefill_tok_per_s"] = round(sum(lens) / sum(secs), 1)
        out["prefill_s_per_call_mean"] = round(st.mean(secs), 3)
        # linear fit seconds = a + b * len
        mx, my = st.mean(lens), st.mean(secs)
        b = sum((x - mx) * (y - my) for x, y in zip(lens, secs)) / sum((x - mx) ** 2 for x in lens)
        out["prefill_fit_ms_per_1k_tokens"] = round(1000 * b * 1000, 1)
        out["prefill_fit_intercept_s"] = round(my - b * mx, 3)
        dl = summ.get("decode_log", [])
        by_active = {}
        for act, _step, sec in dl:
            by_active.setdefault(act, []).append(sec)
        out["decode_ms_by_active"] = {str(k): {"steps": len(v), "mean_ms": round(1000 * st.mean(v), 1)}
                                      for k, v in sorted(by_active.items())}
    else:  # vLLM: no phase timing; per-round wall times overlap across concurrent requests
        out["prompt_tokens_total"] = sum(int(r["prompt_tokens"]) for r in rounds)
        out["note"] = "no prefill/decode split; generate_seconds are overlapping per-request wall times"
    return out


if __name__ == "__main__":
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    for arg in sys.argv[2:]:
        name, path = arg.split("=", 1)
        res = analyse(path)
        with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
            json.dump(res, f, indent=2)
        print(name, json.dumps({k: v for k, v in res.items() if k != "decode_ms_by_active"}, indent=None))
        if "decode_ms_by_active" in res:
            print("  decode ms by active:", res["decode_ms_by_active"])
