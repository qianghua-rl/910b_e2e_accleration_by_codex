"""Flow-based stage accounting of device-idle intervals, not speedup estimates."""
import argparse,bisect,json,collections
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('trace');p.add_argument('--output',required=True);a=p.parse_args()
r=json.loads(Path(a.trace).read_text());e=r if isinstance(r,list) else r['traceEvents']
pids={x['pid'] for x in e if x.get('name')=='process_name' and x.get('args',{}).get('name')=='Ascend Hardware'}
source={};dest={};launch={}
for x in e:
 cat=x.get('cat');ph=x.get('ph')
 if cat=='async_npu' and ph=='s':source[x['id']]=x
 if cat=='async_npu' and ph=='f':dest[(x['pid'],x['tid'],float(x['ts']))]=x['id']
 if cat=='HostToDevice' and ph=='s':launch[x['id']]=x
# HostToDevice destinations identify exact per-kernel command association.
command={}
for x in e:
 if x.get('cat')=='HostToDevice' and x.get('ph')=='f':
  s=launch.get(x['id'])
  if s:command[(x['pid'],x['tid'],float(x['ts']))]=s
tracks=collections.defaultdict(list)
for x in e:
 if x.get('ph')=='X' and x.get('pid') not in pids and x.get('cat') in ('cpu_op','dequeue'):
  tracks[(x['pid'],x['tid'])].append(x)
for v in tracks.values():v.sort(key=lambda x:float(x['ts']))
starts={k:[float(x['ts']) for x in v] for k,v in tracks.items()}
def parents(point):
 k=(point['pid'],point['tid']);t=float(point['ts']);v=tracks.get(k,[]);j=bisect.bisect_right(starts.get(k,[]),t)
 return [x for x in v[max(0,j-120):j] if float(x['ts'])<=t<=float(x['ts'])+float(x.get('dur',0))]
kernels=sorted((float(x['ts']),float(x['ts'])+float(x.get('dur',0)),x) for x in e if x.get('ph')=='X' and x.get('pid') in pids and not x.get('name','').startswith('EVENT'))
previous=None;gaps=[];groups=collections.defaultdict(lambda:{'count':0,'gap_us':0.,'before_host_us':0.,'host_to_command_us':0.,'after_command_us':0.})
for t,end,x in kernels:
 if previous is not None and t>previous:
  key=(x['pid'],x['tid'],t);host=source.get(dest.get(key));cmd=command.get(key)
  h=float(host['ts']) if host else None;c=float(cmd['ts']) if cmd else None
  ps=parents(host) if host else []; ds=parents(cmd) if cmd else []
  aten=[z for z in ps if z['name'].startswith('aten::')];op=min(aten,key=lambda z:float(z['dur']))['name'] if aten else 'unknown'
  # Partition the idle interval only; timestamps are observations, not critical-path proof.
  cut=max(previous,min(t,h)) if h is not None else previous
  cut2=max(cut,min(t,c)) if c is not None else cut
  row=dict(gap_us=t-previous,before_host_us=cut-previous,host_to_command_us=cut2-cut,after_command_us=t-cut2,
           next_kernel=x['name'],host_op=op,host_ts=h,command_ts=c,device_start=t,previous_device_end=previous,
           cpu_chain=[z['name'] for z in ps],dequeue_chain=[z['name'] for z in ds])
  gaps.append(row);g=groups[op];g['count']+=1
  for k in ('gap_us','before_host_us','host_to_command_us','after_command_us'):g[k]+=row[k]
 previous=max(previous or end,end)
out={'trace':a.trace,'totals':{k:sum(g[k] for g in gaps) for k in ('gap_us','before_host_us','host_to_command_us','after_command_us')},'by_next_host_op':dict(sorted(groups.items(),key=lambda x:-x[1]['gap_us'])),'largest_gaps':sorted(gaps,key=lambda x:-x['gap_us'])[:60],'limitations':['Partitions are clipped timestamp associations, not causal/removable costs','CPU flow source is not command-ready time','HostToDevice source is correlated runtime submission marker, not proven hardware acceptance','Unlinked timestamps retain unknown attribution']}
Path(a.output).write_text(json.dumps(out,indent=2)+'\n');print(out['totals']);print(list(out['by_next_host_op'].items())[:12]);print(out['largest_gaps'][:3])
