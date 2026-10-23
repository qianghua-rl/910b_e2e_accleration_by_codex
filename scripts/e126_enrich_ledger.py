import json,collections
from pathlib import Path
p=Path('results/e126_matmul_ledger.json');d=json.loads(p.read_text())
for mode in ('nd','nz'):
 rows=d[mode]['matmuls']
 for i,x in enumerate(rows):
  ann=x.get('cpu_annotation') or ''
  parts=ann.split('/')
  x['module_path']=ann[len('decode_cost/'): ] if ann.startswith('decode_cost/') else None
  x['stage']=parts[1] if len(parts)>1 else None
  x['layer']=next((int(z.split('.')[0]) for z in parts if z.startswith('layers.') and z.split('.')[0]=='layers'),None)
  x['input_output_shape']='unavailable_in_trace'
 # Summarize by module, retaining all call rows in the JSON.
 g=collections.defaultdict(list)
 for x in rows:g[x['module_path']].append(x)
 d[mode]['by_module']=[dict(module=k,calls=len(v),dispatches=collections.Counter(x['call'] for x in v),
                              kernel_us=sum(y['duration_us'] for x in v for y in x['kernel_matches']),
                              enqueue_us=sum(x['enqueue_duration_us'] for x in v),
                              dequeue_us=sum((x['dequeue_duration_us'] or 0) for x in v)) for k,v in g.items()]
p.write_text(json.dumps(d,indent=2)+'\n')
