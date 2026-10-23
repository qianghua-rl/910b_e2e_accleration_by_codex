"""Disjoint host coverage of idle gaps; no causal/removable-time claim."""
import argparse,bisect,collections,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('trace');p.add_argument('--output',required=True);a=p.parse_args()
r=json.loads(Path(a.trace).read_text());e=r if isinstance(r,list) else r['traceEvents']
hw={x['pid'] for x in e if x.get('name')=='process_name' and x.get('args',{}).get('name')=='Ascend Hardware'}
main=next(x['pid'] for x in e if x.get('name')=='process_name' and x.get('args',{}).get('name')=='Python')
merged=[]
for s,t in sorted((float(x['ts']),float(x['ts'])+float(x['dur'])) for x in e if x.get('ph')=='X' and x.get('pid') in hw and not x.get('name','').startswith('EVENT')):
 if merged and s<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],t)
 else:merged.append([s,t])
gaps=[(merged[i][1],merged[i+1][0]) for i in range(len(merged)-1)];ends=[t for s,t in gaps];pieces=[[] for _ in gaps]
for x in e:
 if x.get('ph')!='X' or x.get('pid')!=main or x.get('tid')!=main:continue
 if x.get('cat') not in ('cpu_op','enqueue'):continue
 name=x['name']
 if name.startswith(('ProfilerStep','batch16_','decode_cost/')):continue
 s=float(x['ts']);t=s+float(x['dur']);i=bisect.bisect_right(ends,s)
 while i<len(gaps) and gaps[i][0]<t:
  pieces[i].append((max(s,gaps[i][0]),min(t,gaps[i][1]),float(x['dur']),name));i+=1
coverage=collections.defaultdict(float)
for (s,t),parts in zip(gaps,pieces):
 points=sorted({s,t,*[z for x,y,_,_ in parts for z in (x,y)]})
 for x,y in zip(points,points[1:]):
  hit=[(dur,n) for lo,hi,dur,n in parts if lo<=x and hi>=y]
  name=min(hit)[1] if hit else 'unrecorded_host'
  coverage[name]+=y-x
result={'trace':a.trace,'gap_us':sum(t-s for s,t in gaps),'coverage_us':dict(sorted(coverage.items(),key=lambda x:-x[1])),'limitation':'Shortest enclosing main-thread event identifies recorded host activity, not proven stall cause; unrecorded includes Python, profiler, scheduling and omitted calls'}
Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
