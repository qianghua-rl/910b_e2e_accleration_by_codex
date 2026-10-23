"""Flow-correlated host timing around gaps; not a causal launch estimator."""
import glob,json,collections
from pathlib import Path
p=glob.glob('results/profile_e100_decode_only/**/trace_view.json',recursive=True)[0]
r=json.load(open(p));e=r if isinstance(r,list) else r['traceEvents']
pids={x['pid'] for x in e if x.get('name')=='process_name' and x.get('args',{}).get('name')=='Ascend Hardware'}
s={x['id']:float(x['ts']) for x in e if x.get('cat')=='async_npu' and x.get('ph')=='s'}
f={(x['pid'],x['tid'],float(x['ts'])):s.get(x['id']) for x in e if x.get('cat')=='async_npu' and x.get('ph')=='f'}
k=sorted((float(x['ts']),float(x['ts'])+float(x.get('dur',0)),x) for x in e if x.get('ph')=='X' and x.get('pid') in pids and not x.get('name','').startswith('EVENT'))
end=None;d=collections.defaultdict(lambda:{'count':0,'gap_us':0.})
for start,stop,x in k:
 if end is not None and start>end:
  host=f.get((x['pid'],x['tid'],start));tag='unlinked' if host is None else ('host_flow_start_after_previous_device_end' if host>end else 'host_flow_started_before_previous_device_end')
  d[tag]['count']+=1;d[tag]['gap_us']+=start-end
 end=max(end or stop,stop)
r={'trace':p,'gap_groups':d,'limitations':['Host flow start marks dispatch origin, not exact device command availability','Host started earlier does not prove submitted/ready earlier','Gaps may include dependencies, device scheduling and profiler overhead','No pure launch or removable-idle estimate is derived']}
Path('results/e100_runtime_gap_diagnostic.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
