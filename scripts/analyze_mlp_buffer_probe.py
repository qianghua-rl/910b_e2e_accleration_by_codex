"""Check complete MLP paired timing, actual allocation/layout changes."""
import csv,glob,json,statistics
from pathlib import Path
r=json.loads(Path('results/mlp_buffer_probe.json').read_text());summaries=[]
for row in r['rows']:
 comparisons=[]
 for queued in (False,True):
  t=[x for x in row['timings'] if x['queued']==queued]
  for i in (1,3):
   base=(t[i-1]['host_median_ms']+t[i+1]['host_median_ms'])/2
   comparisons.append(dict(queued=queued,mode=t[i]['mode'],baseline_host_ms=base,candidate_host_ms=t[i]['host_median_ms'],net_saving_ms=base-t[i]['host_median_ms']))
 layer=row['layer'].split('.')[-2];profiles={}
 for mode in ('baseline','reused','reused_owned_clone'):
  root=f'results/profile_mlp_buffer_ordered_{layer}_{mode}'
  op=glob.glob(root+'/**/operator_details.csv',recursive=True);kp=glob.glob(root+'/**/kernel_details.csv',recursive=True)
  if not op or not kp:continue
  operators=list(csv.DictReader(open(op[0])));kernels=list(csv.DictReader(open(kp[0])))
  profiles[mode]=dict(allocation_event_count=sum(x['Name'] in ('aten::empty','empty_tensor','aten::empty_strided') for x in operators),layout_events=[x['Name'] for x in operators if 'contiguous' in x['Name'] or 'format_cast' in x['Name']],gemm_kernel_count=sum('aclnnMatmul' in x['Name'] or 'aclnnMm_' in x['Name'] for x in kernels),device_kernel_sum_us=sum(float(x['Duration(us)']) for x in kernels),calls=4)
 summaries.append(dict(layer=row['layer'],all_real_inputs_exact=all(c['output_exact'] and c['finite'] for c in row['checks']),checks=row['checks'],paired_comparisons=comparisons,profiles=profiles,persistent_buffer_bytes=row['persistent_buffer_bytes']))
out=dict(layers=summaries,status='local_gate_only_not_deployed',limitations=r['limitations']+['F.linear lowers aclnnMatmul while mm(out=) lowers aclnnMm; this tests available buffer API, not allocator internals in isolation','kernel sums from short profiles not E100 fractions','timing is repeated within layer; full model weight residency differs'])
Path('results/mlp_buffer_probe_analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
