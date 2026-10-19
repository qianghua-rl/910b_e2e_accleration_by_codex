"""Strict per-task/per-round greedy token equality and quality guardrail."""
import argparse
import json

p = argparse.ArgumentParser()
p.add_argument('reference')
p.add_argument('candidate')
args = p.parse_args()

def read(path):
    rows = [json.loads(x) for x in open(path)]
    return (next(x for x in rows if x['kind']=='metadata'),
            {x['id']:x for x in rows if x['kind']=='task'},
            next((x for x in rows if x['kind']=='summary'), None))

rm, ref, rs = read(args.reference)
cm, cand, cs = read(args.candidate)
settings = ['data_sha256', 'max_new_tokens', 'max_rounds', 'do_sample', 'enable_thinking']
protocol_equal = all(rm[k] == cm[k] for k in settings)
protocol_equal = protocol_equal and rm.get('scorer','v1')==cm.get('scorer','v1')
protocol_equal = protocol_equal and rm.get('protocol','original')==cm.get('protocol','original')
common = sorted(ref.keys() & cand.keys())
mismatch = []
for key in common:
    a, b = ref[key], cand[key]
    if ([r['tokens'] for r in a['rounds']] != [r['tokens'] for r in b['rounds']]
        or [r.get('tool_output') for r in a['rounds']] != [r.get('tool_output') for r in b['rounds']]):
        mismatch.append(key)
complete = (rs is not None and cs is not None and len(ref)==rm['tasks']
            and len(cand)==cm['tasks'] and ref.keys()==cand.keys())
quality = complete and sum(x['correct'] for x in cand.values()) >= sum(x['correct'] for x in ref.values())
result = dict(protocol_equal=protocol_equal, complete=complete, common_tasks=len(common),
              mismatching_task_ids=mismatch, greedy_bit_equal=complete and protocol_equal and not mismatch,
              accuracy_guardrail_pass=quality,
              speedup=rs['makespan_seconds']/cs['makespan_seconds'] if complete else None)
print(json.dumps(result, indent=2))
raise SystemExit(0 if result['greedy_bit_equal'] and quality else 1)
