"""Audit the E252 vLLM max-model-length screening run against E047."""

import argparse
import hashlib
import json
from pathlib import Path

from scoring_v2 import answer as answer_v2, correct as correct_v2
from scoring_v3 import answer as answer_v3, correct as correct_v3


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def task_map(rows):
    return {row["id"]: row for row in rows if row.get("kind") == "task"}


def failures(tasks, answer, correct):
    result = []
    for task_id, task in sorted(tasks.items()):
        final_text = task["rounds"][-1]["text"] if task.get("rounds") else ""
        if not correct(answer(final_text), task["expected"]):
            result.append(task_id)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        default="results/vllm_ascend_e252_maxlen1024_guarded_128.jsonl",
    )
    parser.add_argument(
        "--baseline",
        default="results/vllm_ascend_concise_batch128_128.scoring_v2.jsonl",
    )
    parser.add_argument(
        "--allow-slower",
        action="store_true",
        help="Audit quality and context without requiring a beat of historical E047.",
    )
    parser.add_argument(
        "--v3-primary",
        action="store_true",
        help="Use the repository's current quality-track scorer-v3 gate; retain v2 as diagnostic.",
    )
    args = parser.parse_args()
    candidate_rows = load_jsonl(args.candidate)
    baseline_rows = load_jsonl(args.baseline)
    dataset = [json.loads(line) for line in Path("data/gsm8k_128.jsonl").read_text().splitlines()]
    expected_ids = {row["id"] for row in dataset}
    data_hash = hashlib.sha256(Path("data/gsm8k_128.jsonl").read_bytes()).hexdigest()

    candidate_meta = [row for row in candidate_rows if row.get("kind") == "metadata"]
    baseline_meta = [row for row in baseline_rows if row.get("kind") == "metadata"]
    candidate = task_map(candidate_rows)
    baseline = task_map(baseline_rows)
    candidate_summaries = [row for row in candidate_rows if row.get("kind") == "summary"]
    baseline_summaries = [row for row in baseline_rows if row.get("kind") == "summary"]
    candidate_fail_v2 = failures(candidate, answer_v2, correct_v2)
    candidate_fail_v3 = failures(candidate, answer_v3, correct_v3)
    baseline_fail_v2 = failures(baseline, answer_v2, correct_v2)
    baseline_fail_v3 = failures(baseline, answer_v3, correct_v3)

    context_violations = []
    max_model_len = candidate_meta[0].get("max_model_len", 1024) if len(candidate_meta) == 1 else 1024
    malformed_tasks = []
    tool_calls = 0
    tool_exceptions = []
    token_count = 0
    trajectory_diff_ids = []
    tool_output_diff_ids = []
    for task_id in sorted(expected_ids & candidate.keys()):
        row = candidate[task_id]
        rounds = row.get("rounds", [])
        token_count += sum(len(turn.get("tokens", [])) for turn in rounds)
        seen_calls = 0
        if len(rounds) != 2:
            malformed_tasks.append(task_id)
        for index, turn in enumerate(rounds):
            prompt = turn.get("prompt_tokens", 0)
            generated = len(turn.get("tokens", []))
            if prompt + generated > max_model_len or prompt <= 0:
                context_violations.append(
                    {
                        "id": task_id,
                        "turn": index,
                        "prompt": prompt,
                        "generated": generated,
                        "max_model_len": max_model_len,
                    }
                )
            if "code" in turn:
                seen_calls += 1
                tool_calls += 1
                if "tool_output" not in turn:
                    malformed_tasks.append(task_id)
                elif any(
                    marker in turn["tool_output"]
                    for marker in (
                        "Python execution timed out",
                        "Traceback (most recent call last):",
                        "SyntaxError:",
                        "NameError:",
                        "ValueError:",
                        "ZeroDivisionError:",
                    )
                ):
                    tool_exceptions.append(task_id)
        if seen_calls != 1 or row.get("tool_count") != 1:
            malformed_tasks.append(task_id)
        if task_id in baseline:
            old_rounds = baseline[task_id].get("rounds", [])
            same_tokens = [turn.get("tokens") for turn in rounds] == [
                turn.get("tokens") for turn in old_rounds
            ]
            if not same_tokens:
                trajectory_diff_ids.append(task_id)
            old_tools = [turn.get("tool_output") for turn in old_rounds if "code" in turn]
            new_tools = [turn.get("tool_output") for turn in rounds if "code" in turn]
            if old_tools != new_tools:
                tool_output_diff_ids.append(task_id)

    generation_batches = [row for row in candidate_rows if row.get("kind") == "generate_batch"]
    tool_batches = [row for row in candidate_rows if row.get("kind") == "tool_batch"]
    tool_batches_valid = (
        sum(row.get("requests", 0) for row in tool_batches) == tool_calls
        and all(
            row.get("workers") == candidate_meta[0].get("tool_workers", 1)
            for row in tool_batches
        )
        if len(candidate_meta) == 1 and "tool_workers" in candidate_meta[0]
        else True
    )
    seconds = candidate_summaries[0].get("makespan_seconds") if len(candidate_summaries) == 1 else None
    baseline_seconds = baseline_summaries[0].get("makespan_seconds") if len(baseline_summaries) == 1 else None
    baseline_tokens = sum(
        len(turn.get("tokens", []))
        for row in baseline.values()
        for turn in row.get("rounds", [])
    )
    score_v2 = len(expected_ids) - len(candidate_fail_v2)
    score_v3 = len(expected_ids) - len(candidate_fail_v3)
    base_score_v2 = len(expected_ids) - len(baseline_fail_v2)
    base_score_v3 = len(expected_ids) - len(baseline_fail_v3)

    checks = {
        "candidate_one_metadata": len(candidate_meta) == 1,
        "baseline_one_metadata": len(baseline_meta) == 1,
        "dataset_hash_matches": len(candidate_meta) == 1 and candidate_meta[0].get("data_sha256") == data_hash,
        "all_128_unique_tasks": len(candidate) == 128 and set(candidate) == expected_ids,
        "all_tasks_two_rounds_one_tool": not malformed_tasks,
        "all_requests_within_configured_context": not context_violations,
        "no_tool_exceptions_or_timeouts": not tool_exceptions,
        "generation_batches_match_batch_size": len(generation_batches)
        == 2 * ((len(expected_ids) + candidate_meta[0].get("batch_size", 128) - 1)
                // candidate_meta[0].get("batch_size", 128))
        and all(row.get("requests", 0) <= candidate_meta[0].get("batch_size", 128)
                for row in generation_batches),
        "tool_batch_accounting_matches": tool_batches_valid,
        "summary_present": len(candidate_summaries) == 1,
        "v2_not_below_e047": score_v2 >= base_score_v2,
        "v2_no_new_failures": set(candidate_fail_v2) <= set(baseline_fail_v2),
        "v3_not_below_e047": score_v3 >= base_score_v3,
        "v3_no_new_failures": set(candidate_fail_v3) <= set(baseline_fail_v3),
        "faster_than_e047": seconds is not None and baseline_seconds is not None and seconds < baseline_seconds,
    }
    if args.allow_slower:
        checks.pop("faster_than_e047")
    if args.v3_primary:
        checks.pop("v2_not_below_e047")
        checks.pop("v2_no_new_failures")
    report = {
        "verified": all(checks.values()),
        "checks": checks,
        "quality_gate_scorer": "v3" if args.v3_primary else "v2_and_v3",
        "candidate": str(args.candidate),
        "baseline": str(args.baseline),
        "candidate_seconds": seconds,
        "baseline_seconds": baseline_seconds,
        "seconds_faster": baseline_seconds - seconds if seconds is not None and baseline_seconds is not None else None,
        "candidate_tokens": token_count,
        "baseline_tokens": baseline_tokens,
        "token_delta": token_count - baseline_tokens,
        "candidate_tool_calls": tool_calls,
        "candidate_v2": score_v2,
        "baseline_v2": base_score_v2,
        "candidate_v2_failures": candidate_fail_v2,
        "baseline_v2_failures": baseline_fail_v2,
        "candidate_v3": score_v3,
        "baseline_v3": base_score_v3,
        "candidate_v3_failures": candidate_fail_v3,
        "baseline_v3_failures": baseline_fail_v3,
        "token_trajectory_diff_count": len(trajectory_diff_ids),
        "token_trajectory_diff_ids": trajectory_diff_ids,
        "tool_output_diff_count": len(tool_output_diff_ids),
        "tool_output_diff_ids": tool_output_diff_ids,
        "malformed_task_ids": sorted(set(malformed_tasks)),
        "tool_exception_task_ids": sorted(set(tool_exceptions)),
        "context_violations": context_violations,
        "generation_batches": generation_batches,
        "tool_batches": tool_batches,
    }
    print(json.dumps(report, indent=2))
    raise SystemExit(not report["verified"])


if __name__ == "__main__":
    main()
