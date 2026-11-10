"""Serial GSM8K agent baseline on one NPU.

Protocol (self-defined for this workspace):
  - Qwen native tool-call format, non-thinking mode, greedy decoding
  - up to 512 new tokens per turn, up to 6 turns
  - python tool runs in a subprocess with a 10 s timeout
  - final answer read from the last \\boxed{N} in the assistant text
Usage: run_agent.py <out_dir> [--device npu:1] [--limit N]
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time

import torch
import torch_npu  # noqa: F401
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/remote-home/S45160/work/910b_e2e_accleration_by_codex/model/qwen3.5-27b"
DATA = "/remote-home/S45160/work/910b_e2e_accleration_by_codex/data/gsm8k_128.jsonl"
MAX_NEW = 512
MAX_TURNS = 6
TOOL_TIMEOUT = 10

TOOLS = [{"type": "function", "function": {
    "name": "python", "description": "Run Python code; print the result.",
    "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}}]
USER_SUFFIX = "\nSolve with Python, then give the final answer as \\boxed{N}."


def run_python(code):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, dir="/tmp") as f:
        f.write(code)
        path = f.name
    t0 = time.time()
    try:
        r = subprocess.run([sys.executable, "-I", path], capture_output=True, text=True,
                           timeout=TOOL_TIMEOUT)
        out = (r.stdout + r.stderr).strip()
        status = "ok" if r.returncode == 0 else "error"
    except subprocess.TimeoutExpired:
        out, status = "timeout", "timeout"
    finally:
        os.unlink(path)
    return out[:2000], status, time.time() - t0


def extract_tool_code(text):
    m = re.search(r"<parameter=code>\s*(.*?)\s*</parameter>", text, re.S)
    return m.group(1) if m else None


def extract_boxed(text):
    m = re.findall(r"\\boxed\{([^{}]*)\}", text)
    return m[-1].strip() if m else None


def to_number(s):
    if s is None:
        return None
    s = s.replace(",", "").replace("$", "").strip()
    try:
        return float(s)
    except ValueError:
        nums = re.findall(r"-?\d+(?:\.\d+)?", s)
        return float(nums[-1]) if nums else None


def solve(model, tok, q, device):
    msgs = [{"role": "user", "content": q["question"] + USER_SUFFIX}]
    turns = []
    tool_time = 0.0
    gen_time = 0.0
    n_tokens = 0
    final_text = ""
    t_start = time.time()
    for turn in range(MAX_TURNS):
        prompt = tok.apply_chat_template(msgs, tools=TOOLS, add_generation_prompt=True,
                                         tokenize=False, enable_thinking=False)
        ids = tok(prompt, return_tensors="pt").input_ids.to(device)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=MAX_NEW,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
        torch.npu.synchronize(device)
        g = time.time() - t0
        gen_time += g
        new_ids = out[0, ids.shape[1]:]
        n = int(new_ids.shape[0])
        n_tokens += n
        text = tok.decode(new_ids, skip_special_tokens=False)
        text = text.replace("<|im_end|>", "").strip()
        final_text = text
        rec = {"turn": turn, "prompt_tokens": int(ids.shape[1]), "gen_tokens": n, "gen_s": g}
        code = extract_tool_code(text)
        if code is None:
            turns.append(rec)
            break
        out_text, status, dt = run_python(code)
        tool_time += dt
        rec.update({"tool_status": status, "tool_s": dt, "tool_output": out_text})
        turns.append(rec)
        msgs.append({"role": "assistant", "content": text})
        msgs.append({"role": "tool", "content": out_text})
    wall = time.time() - t_start
    pred = extract_boxed(final_text)
    gold = to_number(str(q["answer"]))
    p = to_number(pred)
    correct = p is not None and gold is not None and abs(p - gold) < 1e-6
    return {"id": q["id"], "gold": q["answer"], "pred": pred, "correct": correct,
            "turns": turns, "n_tokens": n_tokens, "gen_s": gen_time, "tool_s": tool_time,
            "wall_s": wall, "final_text": final_text}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--device", default="npu:1")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    qs = [json.loads(l) for l in open(DATA)]
    if a.limit:
        qs = qs[:a.limit]

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 device_map=a.device).eval()
    torch.npu.reset_peak_memory_stats(a.device)

    results_path = os.path.join(a.out_dir, "per_task.jsonl")
    fout = open(results_path, "w")
    t_all = time.time()
    for i, q in enumerate(qs):
        r = solve(model, tok, q, a.device)
        fout.write(json.dumps(r, ensure_ascii=False) + "\n")
        fout.flush()
        print(f"[{i + 1}/{len(qs)}] id={q['id']} correct={r['correct']} turns={len(r['turns'])} "
              f"tok={r['n_tokens']} wall={r['wall_s']:.1f}s elapsed={time.time() - t_all:.0f}s",
              flush=True)
    makespan = time.time() - t_all
    peak = torch.npu.max_memory_allocated(a.device)
    reserved = torch.npu.max_memory_reserved(a.device)
    summary = {"n": len(qs), "correct": None, "makespan_s": makespan,
               "peak_allocated_bytes": peak, "peak_reserved_bytes": reserved,
               "device": a.device}
    rows = [json.loads(l) for l in open(results_path)]
    summary["correct"] = sum(r["correct"] for r in rows)
    summary["total_tokens"] = sum(r["n_tokens"] for r in rows)
    summary["total_tool_s"] = sum(r["tool_s"] for r in rows)
    summary["total_gen_s"] = sum(r["gen_s"] for r in rows)
    json.dump(summary, open(os.path.join(a.out_dir, "summary.json"), "w"), indent=2)
    print("SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
