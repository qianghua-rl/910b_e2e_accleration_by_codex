"""E100 configuration re-implemented in the fresh workspace.

Config (old E100 = batch16, generation_length order, final cap 12):
  HF Qwen3.5-27B BF16, SDPA, fused RMSNorm + fused gated RMSNorm,
  single-step recurrent (recurrent_single_step), synchronous batched turns,
  first-turn cap 512, later-turn cap --final-max-new-tokens, max 6 rounds, greedy.
Scheduling: 'generation_length' sorts tasks by the first-turn generation length recorded in
--order-file (an offline history run), 'fifo' keeps dataset order.
Output: one JSONL row per task (final_text, gold, rounds...) + a summary row with makespan.
"""
import argparse
import json
import sys
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5GatedDeltaNet, Qwen3_5RMSNorm, Qwen3_5RMSNormGated,
)

sys.path.insert(0, "/remote-home/S45160/work/910b_fresh/src")
from agent_protocol import SYSTEM, CONCISE_SYSTEM, TOOLS, parse_call, execute, answer, correct  # noqa: E402
from recurrent_single_step import recurrent_single_step  # noqa: E402
from shift_conv_update import shift_causal_conv1d_update  # noqa: E402

MODEL = "/remote-home/S45160/work/910b_e2e_accleration_by_codex/model/qwen3.5-27b"
DATA = "/remote-home/S45160/work/910b_e2e_accleration_by_codex/data/gsm8k_128.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--protocol", choices=["original", "concise"], default="concise")
    ap.add_argument("--final-max-new-tokens", type=int, default=512)
    ap.add_argument("--schedule", choices=["fifo", "generation_length"], default="generation_length")
    ap.add_argument("--order-file", default=None)
    ap.add_argument("--no-opt", action="store_true", help="disable fused/recurrent opts (control)")
    ap.add_argument("--shift-conv", action="store_true", help="decode conv as shifted multiply-add")
    ap.add_argument("--sync-free", action="store_true", help="device-tensor constants, no host syncs")
    ap.add_argument("--limit", type=int, default=None, help="only the first N tasks (dataset order)")
    a = ap.parse_args()

    out = Path(a.out)
    if out.exists():
        raise SystemExit("output exists")
    system_prompt = {"original": SYSTEM, "concise": CONCISE_SYSTEM}[a.protocol]
    torch.set_num_threads(8)
    torch.manual_seed(42)
    t0 = time.perf_counter()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": a.device}).eval()
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    if not a.no_opt:
        def fused_rmsnorm(self, x):
            return torch_npu.npu_rms_norm(x, (1.0 + self.weight).to(x.dtype), self.eps)[0]

        def fused_gated_rmsnorm(self, hidden_states, gate=None):
            normalized = torch_npu.npu_rms_norm(hidden_states, self.weight, self.variance_epsilon)[0]
            return torch_npu.npu_swiglu(torch.cat((gate, normalized), dim=-1))

        for m in model.modules():
            if isinstance(m, Qwen3_5RMSNorm):
                m.forward = fused_rmsnorm.__get__(m, Qwen3_5RMSNorm)
            elif isinstance(m, Qwen3_5RMSNormGated):
                m.forward = fused_gated_rmsnorm.__get__(m, Qwen3_5RMSNormGated)
            elif isinstance(m, Qwen3_5GatedDeltaNet):
                m.recurrent_gated_delta_rule = recurrent_single_step
                if a.shift_conv:
                    m.causal_conv1d_update = shift_causal_conv1d_update
    if a.sync_free:
        from sync_free import apply_sync_free
        print("SYNC_FREE", apply_sync_free(model), flush=True)
    torch.npu.synchronize(a.device)
    load_s = time.perf_counter() - t0

    data = Path(DATA).read_bytes()
    tasks = [json.loads(l) for l in data.splitlines()]
    if a.schedule == "generation_length":
        est = {}
        for line in Path(a.order_file).read_text().splitlines():
            r = json.loads(line)
            if r.get("kind") == "task" and r.get("rounds"):
                est[str(r["id"])] = len(r["rounds"][0].get("tokens", []))
        missing = [t["id"] for t in tasks if str(t["id"]) not in est]
        if missing:
            raise SystemExit(f"missing estimates for {len(missing)} tasks")
        tasks = sorted(tasks, key=lambda t: est[str(t["id"])], reverse=True)
    if a.limit is not None:
        tasks = tasks[:a.limit]

    def emit(row):
        with out.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps({k: v for k, v in row.items() if k != "rounds"}), flush=True)

    emit(dict(kind="metadata", load_seconds=load_s, protocol=a.protocol, system_prompt=system_prompt,
              schedule=a.schedule, order_file=a.order_file, batch_size=a.batch_size,
              final_max_new_tokens=a.final_max_new_tokens, optimizations=not a.no_opt,
              data_sha256=__import__("hashlib").sha256(data).hexdigest()))
    torch.npu.reset_peak_memory_stats(a.device)
    start = time.perf_counter()
    with torch.inference_mode():
        for base in range(0, len(tasks), a.batch_size):
            active = [dict(task=t, messages=[{"role": "system", "content": system_prompt},
                                             {"role": "user", "content": t["question"]}],
                           rounds=[], tool_count=0, prediction=None, text="")
                      for t in tasks[base:base + a.batch_size]]
            for turn in range(6):
                prompts = [tok.apply_chat_template(s["messages"], tools=TOOLS, tokenize=False,
                                                   add_generation_prompt=True, enable_thinking=False)
                           for s in active]
                inputs = tok(prompts, add_special_tokens=False, padding=True,
                             return_tensors="pt").to(a.device)
                cap = 512 if turn == 0 else a.final_max_new_tokens
                tg = time.perf_counter()
                gen = model.generate(**inputs, max_new_tokens=cap, do_sample=False, use_cache=True)
                rows = gen[:, inputs["input_ids"].shape[1]:].tolist()
                torch.npu.synchronize(a.device)
                gen_s = time.perf_counter() - tg
                survivors = []
                eos = model.generation_config.eos_token_id
                eos = set(eos if isinstance(eos, list) else [eos])
                for s, ids in zip(active, rows):
                    stop = next((i + 1 for i, v in enumerate(ids) if v in eos), len(ids))
                    ids = ids[:stop]
                    text = tok.decode(ids, skip_special_tokens=True)
                    code = parse_call(text)
                    rec = dict(turn=turn, tokens=ids, text=text, batch_generation_seconds=gen_s)
                    s["messages"].append({"role": "assistant", "content": text})
                    s["text"] = text
                    if code is not None:
                        ts = time.perf_counter()
                        result = execute(code)
                        s["tool_count"] += 1
                        rec.update(code=code, tool_output=result, tool_seconds=time.perf_counter() - ts)
                        s["messages"].append({"role": "tool", "name": "python", "content": result})
                    else:
                        s["prediction"] = answer(text)
                    s["rounds"].append(rec)
                    if code is not None and turn < 5:
                        survivors.append(s)
                    else:
                        t = s["task"]
                        emit(dict(kind="task", id=t["id"], gold=t["answer"], final_text=s["text"],
                                  prediction=s["prediction"], tool_count=s["tool_count"],
                                  correct=s["tool_count"] > 0 and correct(s["prediction"], t["answer"]),
                                  completion_seconds=time.perf_counter() - start, rounds=s["rounds"]))
                active = survivors
                if not active:
                    break
    emit(dict(kind="summary", makespan_seconds=time.perf_counter() - start,
              peak_bytes=int(torch.npu.max_memory_allocated(a.device))))


if __name__ == "__main__":
    main()
