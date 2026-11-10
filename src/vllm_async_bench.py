"""vLLM-Ascend baseline with task-level continuous batching (AsyncLLM).

Each task runs as its own coroutine: it submits one generation request per turn,
executes the python tool in a thread pool when the model asks for it, and submits
the next turn immediately. There is no global per-turn barrier, so a finished task
frees its engine slot right away, matching the CBEngine scheduling in cb_run.py.

Concurrency is bounded by vLLM's --max-num-seqs (default 16, same as CB slots=16).
Same agent protocol as vllm_agent_bench_fair.py / cb_run.py. Output rows match
per_task.jsonl of the CB runs (kind=metadata|task|summary), with gold/final_text
fields so scorer_math.py / scorer_mc.py can re-score offline.

A task whose prompt + generation budget does not fit max_model_len is recorded
as failed (error field) and the run continues; it does not abort the run.

Usage (do not use `python -I`; the EngineCore subprocess needs acl):
  python src/vllm_async_bench.py --output OUT/per_task.jsonl --data DATA.jsonl \
      --max-model-len 4096 --max-num-seqs 16 [--direct]
"""
import argparse
import asyncio
import hashlib
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from agent_protocol import (
    CONCISE_SYSTEM,
    MINIMAL_SYSTEM,
    SYSTEM,
    TOOLS,
    answer,
    clip_tool_output,
    correct,
    execute,
    parse_call,
)

MODEL_PATH = "/remote-home/S45160/work/910b_e2e_accleration_by_codex/model/qwen3.5-27b"
DEFAULT_DATA = "/remote-home/S45160/work/910b_e2e_accleration_by_codex/data/gsm8k_128.jsonl"
DIRECT_SYSTEM = ("Answer the question. Reason step by step, then give the final answer in \\boxed{}. "
                 "For multiple-choice questions, put only the option letter in \\boxed{}, e.g. \\boxed{B}.")
MAX_ROUNDS = 6


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--max-num-seqs", type=int, default=16)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.98)
    p.add_argument("--tensor-parallel-size", type=int, default=1,
                   help="shard the model across N cards (set ASCEND_RT_VISIBLE_DEVICES to match)")
    p.add_argument("--tool-workers", type=int, default=16)
    p.add_argument("--protocol", choices=["original", "concise", "minimal"], default="original")
    p.add_argument("--first-max-tokens", type=int, default=512)
    p.add_argument("--final-max-tokens", type=int, default=512)
    p.add_argument("--direct", action="store_true", help="no tools, single turn, \\boxed{} answer")
    p.add_argument("--tool-output-max-chars", type=int, default=0, help="keep last N chars of tool output (0=off)")
    p.add_argument("--direct-max-tokens", type=int, default=2048)
    return p.parse_args()


async def run_task(task, ctx):
    """Run one task to completion. Emits exactly one kind=task row."""
    args, engine, tokenizer, sampling_cls, pool, emit = (
        ctx["args"], ctx["engine"], ctx["tokenizer"], ctx["sampling_cls"],
        ctx["pool"], ctx["emit"])
    loop = asyncio.get_running_loop()
    started = time.perf_counter()
    messages = [{"role": "system", "content": ctx["system_prompt"]},
                {"role": "user", "content": task["question"]}]
    rounds = []
    tool_count = 0
    prediction = None
    try:
        for turn in range(MAX_ROUNDS):
            prompt = tokenizer.apply_chat_template(
                messages, tools=None if args.direct else TOOLS,
                add_generation_prompt=True, enable_thinking=False, tokenize=False)
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            if args.direct:
                max_new = args.direct_max_tokens
            else:
                max_new = args.first_max_tokens if turn == 0 else args.final_max_tokens
            if len(ids) + max_new > args.max_model_len:
                emit(dict(kind="task", id=task["id"], task=task.get("task"), gold=task["answer"],
                          expected=task["answer"], final_text="", prediction=None,
                          rounds=rounds, tool_count=tool_count, correct=False,
                          error=f"prompt {len(ids)} + gen {max_new} > max_model_len {args.max_model_len}",
                          service_seconds=time.perf_counter() - started,
                          completion_seconds=time.perf_counter() - ctx["start"]))
                return
            params = sampling_cls(temperature=0, max_tokens=max_new, detokenize=True)
            req_id = f"{task['id']}-{turn}-{uuid.uuid4().hex[:8]}"
            gen_start = time.perf_counter()
            final = None
            async for out in engine.generate({"prompt_token_ids": ids}, params, request_id=req_id):
                final = out
            gen_seconds = time.perf_counter() - gen_start
            comp = final.outputs[0]
            text, token_ids = comp.text, list(comp.token_ids)
            code = None if args.direct else parse_call(text)
            messages.append({"role": "assistant", "content": text})
            record = {"turn": turn, "tokens": token_ids, "text": text,
                      "prompt_tokens": len(ids), "generate_seconds": gen_seconds}
            if code is not None:
                tool_count += 1
                t0 = time.perf_counter()
                value = clip_tool_output(await loop.run_in_executor(pool, execute, code),
                                         args.tool_output_max_chars)
                record.update(code=code, tool_output=value, tool_seconds=time.perf_counter() - t0)
                messages.append({"role": "tool", "name": "python", "content": value})
            rounds.append(record)
            if code is None or turn == MAX_ROUNDS - 1:
                prediction = answer(text) if code is None else None
                final_text = text
                break
        emit(dict(kind="task", id=task["id"], task=task.get("task"), gold=task["answer"],
                  expected=task["answer"], final_text=final_text, prediction=prediction,
                  rounds=rounds, tool_count=tool_count,
                  correct=(tool_count > 0 or args.direct) and correct(prediction, task["answer"]),
                  service_seconds=time.perf_counter() - started,
                  completion_seconds=time.perf_counter() - ctx["start"]))
    except Exception as exc:  # record and continue; never abort the whole run
        emit(dict(kind="task", id=task["id"], task=task.get("task"), gold=task["answer"],
                  expected=task["answer"], final_text="", prediction=None, rounds=rounds,
                  tool_count=tool_count, correct=False, error=f"{type(exc).__name__}: {exc}",
                  service_seconds=time.perf_counter() - started,
                  completion_seconds=time.perf_counter() - ctx["start"]))


async def amain(args):
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    import vllm

    output = Path(args.output)
    if output.exists():
        raise SystemExit("output exists")
    output.parent.mkdir(parents=True, exist_ok=True)

    def emit(row):
        with output.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps({k: v for k, v in row.items() if k != "rounds"}, ensure_ascii=False), flush=True)

    data_bytes = Path(args.data).read_bytes()
    tasks = [json.loads(l) for l in data_bytes.splitlines()]
    if args.limit is not None:
        tasks = tasks[: args.limit]
    system_prompt = DIRECT_SYSTEM if args.direct else {
        "original": SYSTEM, "concise": CONCISE_SYSTEM, "minimal": MINIMAL_SYSTEM}[args.protocol]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="left")
    t_load = time.perf_counter()
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(
        model=MODEL_PATH, tensor_parallel_size=args.tensor_parallel_size, dtype="bfloat16",
        max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization, enable_prefix_caching=False))
    load_seconds = time.perf_counter() - t_load

    emit({"kind": "metadata", "framework": "vllm-ascend", "engine": "AsyncLLM",
          "vllm_version": vllm.__version__, "tasks": len(tasks),
          "batch_size": None, "max_num_seqs": args.max_num_seqs,
          "protocol": args.protocol, "direct": args.direct, "system_prompt": system_prompt,
          "data_path": args.data,
          "data_sha256": hashlib.sha256(data_bytes).hexdigest(),
          "max_model_len": args.max_model_len, "max_rounds": MAX_ROUNDS,
          "first_max_tokens": args.first_max_tokens, "tool_output_max_chars": args.tool_output_max_chars, "final_max_tokens": args.final_max_tokens,
          "direct_max_tokens": args.direct_max_tokens,
          "do_sample": False, "enable_thinking": False, "load_seconds": load_seconds})

    ctx = {"args": args, "engine": engine, "tokenizer": tokenizer, "sampling_cls": SamplingParams,
           "pool": ThreadPoolExecutor(max_workers=args.tool_workers), "emit": emit,
           "system_prompt": system_prompt, "start": time.perf_counter()}
    try:
        await asyncio.gather(*(run_task(t, ctx) for t in tasks))
        emit({"kind": "summary", "makespan_seconds": time.perf_counter() - ctx["start"],
              "tasks": len(tasks)})
    finally:
        ctx["pool"].shutdown(wait=False)
        engine.shutdown()


def main():
    asyncio.run(amain(parse_args()))


if __name__ == "__main__":
    main()
