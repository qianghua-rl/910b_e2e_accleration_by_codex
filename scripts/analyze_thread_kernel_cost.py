"""Compare fixed-replay thread conditions using actual kernel and step data."""
import csv,glob,json,collections
from pathlib import Path
rows=[]
for tag in ('8a','1','4','8b'):
 root=f'results/profile_e118_threads_{tag}'
 kp=glob.glob(root+'/**/kernel_details.csv',recursive=True)
 sp=glob.glob(root+'/**/step_trace_time.csv',recursive=True)
 if not kp or not sp:continue
 kernels=list(csv.DictReader(open(kp[0])));steps=list(csv.DictReader(open(sp[0])))
 groups=collections.defaultdict(lambda:{'count':0,'sum_us':0.})
 for k in kernels:
  g=groups[k['Name'].split('_')[0]];g['count']+=1;g['sum_us']+=float(k['Duration(us)'])
 logs=[json.loads(s) for s in Path(root+'.log').read_text().splitlines() if s.startswith('{"kind": "batch16_profile"')]
 row=dict(tag=tag,threads=logs[-1]['threads'] if logs else None,steps=[s['Step'] for s in steps],kernel_count=len(kernels),
          computing_union_us=sum(float(s['Computing']) for s in steps),stage_us=sum(float(s['Stage']) for s in steps),
          free_us=sum(float(s['Free']) for s in steps),families=groups,
          fixed_replay_equal=logs[-1]['fixed_replay_equal'] if logs else None,
          affinity=logs[-1]['cpu_affinity'] if logs else None)
 rows.append(row)
 for group in ('aclnnMatmul','aclnnConvolution','aclnnMul','aclnnReduceSum','aclnnInplaceCopy'):
  print(tag,group,groups[group])
 print(tag,'compute/stage/free',row['computing_union_us'],row['stage_us'],row['free_us'])
Path('results/e118_thread_kernel_cost.json').write_text(json.dumps({'rows':rows,'limitations':['profiler perturbs host execution','kernel family sums may overlap; use computing union for total','free time is not all removable CPU overhead','no pure launch latency inferred']},indent=2)+'\n')
