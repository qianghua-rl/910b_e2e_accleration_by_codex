"""Offline upper-bound model for continuous admission from a completed run."""
import argparse
import json
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("run")
parser.add_argument("--slots", type=int, default=16)
args = parser.parse_args()

rows = [json.loads(line) for line in Path(args.run).read_text().splitlines() if line]
tasks = [row for row in rows if row.get("kind") == "task"]
if not tasks:
    raise SystemExit("No task records")

lengths = [len(row["rounds"][0].get("tokens", [])) for row in tasks]
lengths += [len(row["rounds"][1].get("tokens", [])) for row in tasks if len(row.get("rounds", [])) > 1]
fixed_steps = 0
effective_steps = 0
batch_count = 0
for base in range(0, len(tasks), args.slots):
    batch = tasks[base:base + args.slots]
    for turn in range(max(len(row.get("rounds", [])) for row in batch)):
        values = [len(row["rounds"][turn].get("tokens", [])) for row in batch if turn < len(row.get("rounds", []))]
        if values:
            fixed_steps += max(values)
            effective_steps += sum(values) / args.slots
            batch_count += 1

print(json.dumps({
    "run": args.run,
    "tasks": len(tasks),
    "slots": args.slots,
    "synchronous_batches": batch_count,
    "effective_tokens": sum(lengths),
    "fixed_max_token_steps": fixed_steps,
    "ideal_continuous_token_steps": effective_steps,
    "ideal_step_reduction": 1.0 - effective_steps / fixed_steps,
    "note": "Upper bound only; excludes prompt prefill, cache rebase, tool dependencies, and smaller-batch efficiency.",
}, sort_keys=True))
