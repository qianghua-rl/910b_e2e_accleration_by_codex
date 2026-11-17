"""Serial GSM8K agent baseline for Gemma-4-26B-A4B-it on one NPU.

Same workload as the Qwen baseline (src/run_agent.py / agent_protocol.py):
  - 128 GSM8K tasks, greedy, non-thinking, original SYSTEM prompt
  - up to 512 new tokens per turn, up to 6 turns
  - python tool runs in a subprocess with a 10 s timeout
  - scoring: fresh-v1 (src/scorer.py); final_text is written so scorer.py can re-score it

Gemma tool-call format (from chat_template.jinja):
  call:    <|tool_call>call:python{code:<|"|>...<|"|>}<tool_call|>
  result:  tool message -> <|tool_response>response:python{value:<|"|>...<|"|>}<tool_response|>

Usage: gemma_run.py <out_dir> [--device npu:0] [--limit N] [--model DIR]
"""
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, "/remote-home/S45160/work/910b_fresh/src")  # agent_protocol / scorer helpers

import torch
import torch_npu  # noqa: F401
from transformers import AutoModelForCausalLM, AutoTokenizer

import agent_protocol as P  # SYSTEM, TOOLS, execute, answer, correct, clip_tool_output
from gemma_fast_norm import patch_gemma_norms
from gemma_cb import run_cb

MODEL = "/remote-home/S45160/work/910b_fresh/gemma-4-26B-A4B-it"
DATA = "/remote-home/S45160/work/910b_e2e_accleration_by_codex/data/gsm8k_128.jsonl"
MAX_NEW = 512
MAX_TURNS = 6

CALL_RE = re.compile(r'<\|tool_call>call:python\{code:<\|"\|>(.*?)<\|"\|>\}', re.S)


def parse_gemma_call(text):
    m = CALL_RE.search(text)
    return m.group(1) if m else None


def strip_specials(text):
    for t in ["<turn|>", "<eos>", "<tool_call|>", "<|tool_response>"]:
        text = text.replace(t, "")
    return text.strip()


def solve(model, tok, q, device, eos_ids):
    msgs = [{"role": "system", "content": P.SYSTEM},
            {"role": "user", "content": q["question"]}]
    turns = []
    tool_time = 0.0
    gen_time = 0.0
    n_tokens = 0
    final_text = ""
    t_start = time.time()
    for turn in range(MAX_TURNS):
        prompt = tok.apply_chat_template(msgs, tools=P.TOOLS, add_generation_prompt=True,
                                         tokenize=False, enable_thinking=False)
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=MAX_NEW,
                                 do_sample=False, eos_token_id=eos_ids,
                                 pad_token_id=tok.pad_token_id)
        torch.npu.synchronize(device)
        g = time.time() - t0
        gen_time += g
        new_ids = out[0, ids.shape[1]:]
        n = int(new_ids.shape[0])
        n_tokens += n
        raw = tok.decode(new_ids, skip_special_tokens=False)
        # Re-add the closing tool-call token if generation stopped on it, so the regex matches.
        if "<|tool_call>" in raw and "<tool_call|>" not in raw:
            raw = raw + "<tool_call|>"
        text = strip_specials(raw)
        final_text = text
        rec = {"turn": turn, "prompt_tokens": int(ids.shape[1]), "gen_tokens": n, "gen_s": g}
        code = parse_gemma_call(raw)
        if code is None:
            turns.append(rec)
            break
        t_tool = time.time()
        out_text = P.execute(code)
        dt = time.time() - t_tool
        tool_time += dt
        rec.update({"tool_output": out_text})
        turns.append(rec)
        msgs.append({"role": "assistant", "content": "",
                     "tool_calls": [{"type": "function",
                                     "function": {"name": "python", "arguments": {"code": code}}}]})
        msgs.append({"role": "tool", "name": "python", "content": P.clip_tool_output(out_text, 0)})
    wall = time.time() - t_start
    pred = P.answer(final_text)
    ok = P.correct(pred, q["answer"])
    return {"id": q["id"], "gold": q["answer"], "pred": pred, "correct": ok,
            "turns": turns, "n_tokens": n_tokens, "gen_s": gen_time, "tool_s": tool_time,
            "wall_s": wall, "final_text": final_text}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--print-prompt", action="store_true")
    ap.add_argument("--fast-norm", action="store_true", help="fused RMSNorm (gemma_fast_norm.py)")
    ap.add_argument("--fused-rope", action="store_true",
                    help="with --cb: RoPE via torch_npu.npu_rotary_mul (gemma_cb._rope_fused)")
    ap.add_argument("--fast-norm-noscale", action="store_true",
                    help="with --fast-norm: also fuse the weight-free norms (v_norm, router norm)")
    ap.add_argument("--cb", action="store_true", help="continuous batching (gemma_cb.py)")
    ap.add_argument("--compact", action="store_true",
                    help="with --cb: decode only active slots (no idle dummy rows)")
    ap.add_argument("--host-tact", action="store_true",
                    help="with --cb --compact: compute attention length on host (no pos.max().item() sync)")
    ap.add_argument("--grouped-experts", action="store_true",
                    help="with --cb: MoE experts via npu_grouped_matmul (gemma_cb.experts_grouped)")
    ap.add_argument("--slice-kv", action="store_true",
                    help="with --cb: read K/V as views when all slots are active (no gather copy)")
    ap.add_argument("--fast-router", action="store_true",
                    help="with --cb: router with folded scale (gemma_cb.router_fused)")
    ap.add_argument("--nocopy", action="store_true",
                    help="with --cb: GQA attention without repeating K/V (gemma_cb nocopy path)")
    ap.add_argument("--fia", action="store_true",
                    help="with --cb: attention via torch_npu.npu_fused_infer_attention_score")
    ap.add_argument("--geglu", action="store_true",
                    help="with --cb --grouped-experts: MoE activation via npu_geglu (weights half-swapped at init)")
    ap.add_argument("--prefill-grouped", action="store_true",
                    help="with --cb --grouped-experts: HF batch-1 prefill also uses grouped experts (prefill only)")
    ap.add_argument("--slots", type=int, default=16)
    ap.add_argument("--t-max", type=int, default=2048)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    qs = [json.loads(l) for l in open(DATA)]
    if a.limit:
        qs = qs[:a.limit]

    tok = AutoTokenizer.from_pretrained(a.model)
    eos_ids = [1, 106, tok.convert_tokens_to_ids("<tool_call|>")]
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                                 device_map=a.device).eval()
    n_norm = patch_gemma_norms(model, noscale=a.fast_norm_noscale) if a.fast_norm else 0
    print(f"fast_norm={a.fast_norm} patched_norms={n_norm}", flush=True)
    torch.npu.reset_peak_memory_stats(a.device)

    if a.print_prompt:
        msgs = [{"role": "system", "content": P.SYSTEM}, {"role": "user", "content": qs[0]["question"]}]
        print(tok.apply_chat_template(msgs, tools=P.TOOLS, add_generation_prompt=True,
                                      tokenize=False, enable_thinking=False))
        return

    results_path = os.path.join(a.out_dir, "per_task.jsonl")
    if a.cb:
        done, makespan = run_cb(model, tok, qs, eos_ids, a.slots, a.t_max, a.device,
                                compact=a.compact, grouped=a.grouped_experts,
                                slice_kv=a.slice_kv, fast_router=a.fast_router,
                                host_tact=a.host_tact, fused_rope=a.fused_rope,
                                fia=a.fia, geglu=a.geglu, nocopy=a.nocopy,
                                prefill_grouped=a.prefill_grouped)
        done.sort(key=lambda r: r["id"])
        with open(results_path, "w") as f:
            for r in done:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary = {"model": a.model, "n": len(qs), "correct": sum(r["correct"] for r in done),
                   "makespan_s": makespan, "total_tokens": sum(r["n_tokens"] for r in done),
                   "peak_allocated_bytes": torch.npu.max_memory_allocated(a.device),
                   "peak_reserved_bytes": torch.npu.max_memory_reserved(a.device),
                   "device": a.device, "max_new": MAX_NEW, "max_turns": MAX_TURNS,
                   "greedy": True, "thinking": False, "fast_norm": a.fast_norm, "fast_norm_noscale": a.fast_norm_noscale,
                   "patched_norms": n_norm, "cb": True, "slots": a.slots, "t_max": a.t_max,
                   "compact": a.compact, "grouped_experts": a.grouped_experts, "slice_kv": a.slice_kv, "fast_router": a.fast_router, "host_tact": a.host_tact, "fused_rope": a.fused_rope, "prefill_grouped": a.prefill_grouped}
        json.dump(summary, open(os.path.join(a.out_dir, "summary.json"), "w"), indent=2)
        print("SUMMARY", json.dumps(summary), flush=True)
        return
    fout = open(results_path, "w")
    t_all = time.time()
    for i, q in enumerate(qs):
        r = solve(model, tok, q, a.device, eos_ids)
        fout.write(json.dumps(r, ensure_ascii=False) + "\n")
        fout.flush()
        print(f"[{i + 1}/{len(qs)}] id={q['id']} correct={r['correct']} turns={len(r['turns'])} "
              f"tok={r['n_tokens']} wall={r['wall_s']:.1f}s elapsed={time.time() - t_all:.0f}s",
              flush=True)
    makespan = time.time() - t_all
    rows = [json.loads(l) for l in open(results_path)]
    summary = {"model": a.model, "n": len(qs), "correct": sum(r["correct"] for r in rows),
               "makespan_s": makespan,
               "total_tokens": sum(r["n_tokens"] for r in rows),
               "total_gen_s": sum(r["gen_s"] for r in rows),
               "peak_allocated_bytes": torch.npu.max_memory_allocated(a.device),
               "peak_reserved_bytes": torch.npu.max_memory_reserved(a.device),
               "device": a.device, "max_new": MAX_NEW, "max_turns": MAX_TURNS,
               "greedy": True, "thinking": False, "fast_norm": a.fast_norm, "fast_norm_noscale": a.fast_norm_noscale,
               "patched_norms": n_norm}
    json.dump(summary, open(os.path.join(a.out_dir, "summary.json"), "w"), indent=2)
    print("SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
