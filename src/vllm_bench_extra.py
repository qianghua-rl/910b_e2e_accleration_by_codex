"""README-compatible batch agent benchmark using vLLM-Ascend."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import time
from pathlib import Path

from agent_protocol import (
    SYSTEM,
    CONCISE_SYSTEM,
    MINIMAL_SYSTEM,
    TOOLS,
    answer,
    correct,
    execute,
    parse_call,
)


DEFAULT_DATA = "/remote-home/S45160/work/910b_e2e_accleration_by_codex/data/gsm8k_128.jsonl"
DIRECT_SYSTEM = ("Answer the question. Reason step by step, then give the final answer in \\boxed{}. "
                 "For multiple-choice questions, put only the option letter in \\boxed{}, e.g. \\boxed{B}.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--task-order", help="Comma-separated task IDs in explicit execution order")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=16, help="vLLM scheduler concurrency (match CB slots=16)")
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--tool-workers", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.98)
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--protocol", choices=["original", "concise", "minimal"], default="original")
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--direct", action="store_true", help="no tools, single turn, \\boxed{} answer (GPQA/BBH)")
    parser.add_argument("--direct-max-tokens", type=int, default=2048)
    return parser.parse_args()


def main():
    args = parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit("output exists")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    import vllm

    system_prompt = DIRECT_SYSTEM if args.direct else {"original": SYSTEM, "concise": CONCISE_SYSTEM, "minimal": MINIMAL_SYSTEM}[args.protocol]
    model_path = "/remote-home/S45160/work/910b_e2e_accleration_by_codex/model/qwen3.5-27b"
    data_bytes = Path(args.data).read_bytes()
    tasks = [json.loads(line) for line in data_bytes.splitlines()][: args.limit]
    if args.task_order:
        requested_ids = [int(value) for value in args.task_order.split(",") if value]
        by_id = {task["id"]: task for task in tasks}
        if len(requested_ids) != len(set(requested_ids)) or any(task_id not in by_id for task_id in requested_ids):
            raise SystemExit("--task-order must contain unique task IDs from the selected dataset")
        if len(requested_ids) != args.limit:
            raise SystemExit("--task-order length must match --limit")
        tasks = [by_id[task_id] for task_id in requested_ids]
    if args.task_id is not None:
        tasks = [task for task in tasks if task["id"] == args.task_id]
    if not tasks:
        raise SystemExit("no tasks selected")
    if args.tool_workers < 1:
        raise SystemExit("--tool-workers must be positive")

    def emit(row):
        with output.open("a") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=args.enable_prefix_caching,
    )
    load_seconds = time.perf_counter() - started
    sampling = SamplingParams(temperature=0, max_tokens=args.direct_max_tokens if args.direct else 512, detokenize=True)
    emit(
        {
            "kind": "metadata",
            "framework": "vllm-ascend",
            "vllm_version": vllm.__version__,
            "tasks": len(tasks),
            "task_order": [task["id"] for task in tasks],
            "batch_size": args.batch_size,
            "protocol": args.protocol,
            "system_prompt": system_prompt,
            "data_sha256": hashlib.sha256(data_bytes).hexdigest(),
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "tool_workers": args.tool_workers,
            "max_new_tokens": args.direct_max_tokens if args.direct else 512,
            "direct": args.direct,
            "data_path": args.data,
            "max_rounds": 6,
            "do_sample": False,
            "enable_thinking": False,
            "enable_prefix_caching": args.enable_prefix_caching,
            "load_seconds": load_seconds,
        }
    )

    start = time.perf_counter()
    template_args = {
        "tools": None if args.direct else TOOLS,
        "add_generation_prompt": True,
        "enable_thinking": False,
        "tokenize": False,
    }
    def execute_timed(code):
        tool_start = time.perf_counter()
        value = execute(code)
        return value, time.perf_counter() - tool_start

    with ThreadPoolExecutor(max_workers=args.tool_workers) as tool_pool:
        for base in range(0, len(tasks), args.batch_size):
            active = [
                {
                    "task": task,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": task["question"]},
                    ],
                    "rounds": [],
                    "tool_count": 0,
                    "started": time.perf_counter(),
                }
                for task in tasks[base : base + args.batch_size]
            ]
            for turn in range(6):
                prompts = [
                    tokenizer.apply_chat_template(state["messages"], **template_args)
                    for state in active
                ]
                prompt_lengths = [
                    len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts
                ]
                too_long = [length for length in prompt_lengths if length > args.max_model_len]
                if too_long:
                    raise RuntimeError(
                        f"rendered prompt exceeds max_model_len={args.max_model_len}: {max(too_long)}"
                    )
                generate_start = time.perf_counter()
                results = llm.generate(prompts, sampling, use_tqdm=False)
                generate_seconds = time.perf_counter() - generate_start
                emit(
                    {
                        "kind": "generate_batch",
                        "batch_start": base,
                        "turn": turn,
                        "requests": len(active),
                        "max_prompt_tokens": max(prompt_lengths),
                        "generate_seconds": generate_seconds,
                    }
                )

                prepared = []
                tool_futures = {}
                for index, (state, result, prompt_tokens) in enumerate(
                    zip(active, results, prompt_lengths)
                ):
                    generated = result.outputs[0]
                    text = generated.text
                    token_ids = generated.token_ids
                    if prompt_tokens + len(token_ids) > args.max_model_len:
                        raise RuntimeError(
                            f"request exceeded max_model_len={args.max_model_len}: "
                            f"prompt={prompt_tokens}, generated={len(token_ids)}"
                        )
                    code = None if args.direct else parse_call(text)
                    state["messages"].append({"role": "assistant", "content": text})
                    prepared.append(
                        {
                            "state": state,
                            "text": text,
                            "code": code,
                            "record": {
                                "turn": turn,
                                "tokens": token_ids,
                                "text": text,
                                "prompt_tokens": prompt_tokens,
                                "generate_seconds": generate_seconds,
                            },
                        }
                    )
                    if code is not None:
                        tool_futures[index] = tool_pool.submit(execute_timed, code)

                tool_batch_start = time.perf_counter()
                tool_results = {
                    index: future.result() for index, future in tool_futures.items()
                }
                tool_batch_seconds = time.perf_counter() - tool_batch_start
                if tool_futures:
                    emit(
                        {
                            "kind": "tool_batch",
                            "batch_start": base,
                            "turn": turn,
                            "requests": len(tool_futures),
                            "workers": args.tool_workers,
                            "wall_seconds": tool_batch_seconds,
                            "sum_worker_seconds": sum(duration for _, duration in tool_results.values()),
                        }
                    )

                survivors = []
                for index, item in enumerate(prepared):
                    state = item["state"]
                    text = item["text"]
                    code = item["code"]
                    record = item["record"]
                    if code is not None:
                        value, tool_seconds = tool_results[index]
                        state["tool_count"] += 1
                        record.update(code=code, tool_output=value, tool_seconds=tool_seconds)
                        state["messages"].append(
                            {"role": "tool", "name": "python", "content": value}
                        )
                        if turn < 5:
                            survivors.append(state)
                    else:
                        state["prediction"] = answer(text)
                    state["rounds"].append(record)
                    if code is None or turn == 5:
                        emit(
                            {
                                "kind": "task",
                                "id": state["task"]["id"],
                                "rounds": state["rounds"],
                                "prediction": state.get("prediction"),
                                "expected": state["task"]["answer"],
                                "gold": state["task"]["answer"],
                                "task": state["task"].get("task"),
                                "final_text": state["rounds"][-1]["text"],
                                "tool_count": state["tool_count"],
                                "correct": state["tool_count"] > 0
                                and correct(state.get("prediction"), state["task"]["answer"]),
                                "service_seconds": time.perf_counter() - state["started"],
                                "completion_seconds": time.perf_counter() - start,
                            }
                        )
                active = survivors
                if not active:
                    break
    emit({"kind": "summary", "makespan_seconds": time.perf_counter() - start})


if __name__ == "__main__":
    main()
