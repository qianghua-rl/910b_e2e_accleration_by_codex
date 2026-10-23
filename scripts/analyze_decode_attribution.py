"""Attribute device kernels via profiler flow to marked CPU module intervals."""
import argparse,bisect,collections,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('trace');p.add_argument('--output',required=True);a=p.parse_args()
r=json.loads(Path(a.trace).read_text());events=r if isinstance(r,list) else r['traceEvents']
labels=collections.defaultdict(list)
for e in events:
 if e.get('ph')=='X' and e.get('name','').startswith('decode_cost/'):
  labels[(e['pid'],e['tid'])].append((float(e['ts']),float(e['ts'])+float(e['dur']),e['name'].split('/')[1]))
for k in labels: labels[k].sort()
starts={k:[x[0] for x in v] for k,v in labels.items()}
def category(e):
 k=(e['pid'],e['tid']);v=labels.get(k,[]);i=bisect.bisect_right(starts.get(k,[]),float(e['ts']))-1
 return v[i][2] if i>=0 and float(e['ts'])<=v[i][1] else 'other'
flows={}
for e in events:
 if e.get('cat')=='async_npu' and e.get('ph')=='s': flows[e['id']]=(category(e),float(e['ts']))
ends={}
for e in events:
 if e.get('cat')=='async_npu' and e.get('ph')=='f': ends[(e['pid'],e['tid'],float(e['ts']))]=flows.get(e['id'],('unlinked',None))
pids={e['pid'] for e in events if e.get('name')=='process_name' and e.get('args',{}).get('name')=='Ascend Hardware'}
groups=collections.defaultdict(lambda:collections.defaultdict(lambda:{'count':0,'device_us':0.,'flow_delay_us':[]}))
spans=[]
for e in events:
 if e.get('ph')!='X' or e.get('pid') not in pids: continue
 t=float(e['ts']);dur=float(e.get('dur',0));cat,host=ends.get((e['pid'],e['tid'],t),('unlinked',None));name=e['name'].split('_')[0];g=groups[cat][name];g['count']+=1;g['device_us']+=dur
 if host is not None:g['flow_delay_us'].append(t-host)
 spans.append((t,t+dur))
for ops in groups.values():
 for g in ops.values():
  delays=sorted(g.pop('flow_delay_us'));g['flow_delay_median_us']=delays[len(delays)//2] if delays else None
merged=[]
for x,y in sorted(spans):
 if merged and x<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],y)
 else:merged.append([x,y])
result={'groups':groups,'device_busy_union_us':sum(y-x for x,y in merged),'device_window_us':merged[-1][1]-merged[0][0], 'cpu_marked_duration_us':{c:sum(y-x for v in labels.values() for x,y,z in v if z==c) for c in ['projection','mlp','lm_head','recurrent']},'limitations':['flow delay includes queueing and device backlog; not pure launch cost','device family sums can overlap','CPU marked intervals include nested dispatch; not additive to device time','four early decode steps only']}
Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
for c,ops in groups.items():print(c,sorted([(n,g['count'],round(g['device_us'],2)) for n,g in ops.items()],key=lambda x:-x[2])[:8])
