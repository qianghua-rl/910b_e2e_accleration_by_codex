"""Summarize completed and partial agent runs without conflating them."""
import argparse
import json
import statistics

p = argparse.ArgumentParser()
p.add_argument('file')
args = p.parse_args()
rows = [json.loads(line) for line in open(args.file)]
tasks = [r for r in rows if r['kind']=='task']
meta = next(r for r in rows if r['kind']=='metadata')
summary = next((r for r in rows if r['kind']=='summary'), None)
latencies = sorted(r['completion_seconds'] for r in tasks)
parts = {'prepare_seconds':0., 'generation_seconds':0., 'tool_seconds':0.}
complete = summary is not None and len(tasks)==meta['tasks']
tokens = 0
for task in tasks:
    for turn in task['rounds']:
        tokens += len(turn['tokens'])
        for key in parts:
            parts[key] += turn.get(key, 0.)
print(json.dumps(dict(expected_tasks=meta['tasks'], completed_tasks=len(tasks),
    complete=complete,
    correct=sum(r['correct'] for r in tasks),
    accuracy=sum(r['correct'] for r in tasks)/len(tasks) if tasks else None,
    accuracy_scope='complete_run' if complete else 'completed_subset_only',
    makespan_seconds=summary['makespan_seconds'] if summary else None,
    service_median=statistics.median(r['service_seconds'] for r in tasks) if tasks else None,
    completion_p50=statistics.median(latencies) if latencies else None,
    completion_p95=latencies[min(len(latencies)-1, int(.95*len(latencies)))] if latencies else None,
    generated_tokens=tokens, parts=parts), indent=2))
