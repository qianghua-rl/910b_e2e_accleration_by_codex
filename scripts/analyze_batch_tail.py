"""Summarize synchronous generation batches from an agent benchmark JSONL."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


parser = argparse.ArgumentParser()
parser.add_argument("runs", nargs="+", help="One or more agent benchmark JSONL files")
args = parser.parse_args()

for run_path in args.runs:
    records = [json.loads(line) for line in Path(run_path).read_text().splitlines() if line]
    tasks = [row for row in records if row.get("kind") == "task"]
    summaries = [row for row in records if row.get("kind") == "summary"]
    if not tasks or len(summaries) != 1:
        raise SystemExit(f"{run_path}: expected task records and exactly one summary")

    calls = defaultdict(list)
    rounds_per_task = Counter()
    for task in tasks:
        rounds_per_task[len(task.get("rounds", []))] += 1
        for round_record in task.get("rounds", []):
            key = (round_record["turn"], round(round_record["batch_generation_seconds"], 6))
            calls[key].append(len(round_record.get("tokens", [])))

    by_turn = defaultdict(list)
    for (turn, batch_seconds), token_lengths in calls.items():
        by_turn[turn].append({
            "batch_generation_seconds": batch_seconds,
            "active_rows": len(token_lengths),
            "max_tokens": max(token_lengths),
            "median_tokens": median(token_lengths),
            "effective_tokens": sum(token_lengths),
            "padding_token_upper_bound": max(token_lengths) * len(token_lengths) - sum(token_lengths),
        })

    result = {
        "run": run_path,
        "tasks": len(tasks),
        "makespan_seconds": summaries[0]["makespan_seconds"],
        "rounds_per_task": dict(sorted(rounds_per_task.items())),
        "synchronous_calls": len(calls),
        "generation_seconds_sum": sum(batch["batch_generation_seconds"] for batches in by_turn.values() for batch in batches),
        "turns": {},
    }
    for turn, batches in sorted(by_turn.items()):
        active_rows = [batch["active_rows"] for batch in batches]
        result["turns"][str(turn)] = {
            "calls": len(batches),
            "active_rows_per_call": active_rows,
            "max_tokens_per_call": [batch["max_tokens"] for batch in batches],
            "median_tokens_per_call": [batch["median_tokens"] for batch in batches],
            "effective_tokens": sum(batch["effective_tokens"] for batch in batches),
            "padding_token_upper_bound": sum(batch["padding_token_upper_bound"] for batch in batches),
            "generation_seconds_sum": sum(batch["batch_generation_seconds"] for batch in batches),
            "longest_call_seconds": max(batch["batch_generation_seconds"] for batch in batches),
        }
    print(json.dumps(result, sort_keys=True))
