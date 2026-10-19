"""Device busy union and idle gaps from an exported Ascend Chrome trace.

Only intervals on explicitly selected device tracks are used. Duration sums
alone double-count overlap; this script merges intervals. Dependency critical
path is not inferred from temporal proximity.
"""
import argparse
import collections
import json

p = argparse.ArgumentParser()
p.add_argument('trace')
p.add_argument('--pid', type=int, action='append', required=True,
               help='Device process IDs from trace metadata; repeat to select several')
args = p.parse_args()
raw = json.load(open(args.trace))
events = raw['traceEvents'] if isinstance(raw, dict) else raw
selected = [e for e in events if e.get('ph')=='X' and e.get('pid') in args.pid and float(e.get('dur',0))>0]
for e in selected:
    e['ts']=float(e['ts'])
    e['dur']=float(e['dur'])
if not selected:
    raise SystemExit('No duration events on selected device tracks')
spans = sorted((e['ts'], e['ts']+e['dur']) for e in selected)
merged = []
for a,b in spans:
    if merged and a<=merged[-1][1]:
        merged[-1][1] = max(merged[-1][1],b)
    else:
        merged.append([a,b])
busy = sum(b-a for a,b in merged)
window = merged[-1][1]-merged[0][0]
gaps = sorted((merged[i+1][0]-merged[i][1] for i in range(len(merged)-1)), reverse=True)
ops = collections.defaultdict(lambda:[0,0.])
for e in selected:
    ops[e['name']][0] += 1
    ops[e['name']][1] += e['dur']
print(json.dumps(dict(window_us=window, busy_union_us=busy, idle_us=window-busy,
    busy_fraction=busy/window, largest_idle_gaps_us=gaps[:20],
    operator_duration_sum_us=sum(e['dur'] for e in selected),
    top_operators=sorted(ops.items(),key=lambda kv:kv[1][1],reverse=True)[:30],
    limitation='Temporal union only; dependency critical path requires flow correlation'), indent=2))
