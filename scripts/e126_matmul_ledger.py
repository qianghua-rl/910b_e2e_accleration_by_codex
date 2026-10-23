"""Build a conservative per-call Matmul ledger from Ascend trace_view.json.

The trace has timing/call ordering and CPU annotations, but this profiler
configuration does not retain tensor shapes. Missing fields stay explicit.
"""
import argparse, csv, json, glob, collections
import re
from pathlib import Path
from safetensors import safe_open

p=argparse.ArgumentParser(); p.add_argument('--nd',required=True); p.add_argument('--nz',required=True); p.add_argument('--output',default='results/e126_matmul_ledger.json'); a=p.parse_args()
root=Path('model/qwen3.5-27b')
weight_map=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
shape_cache={}
def weight_shape(module):
    key=module+'.weight'
    if key not in weight_map:return None
    if key not in shape_cache:
        with safe_open(str(root/weight_map[key]),framework='pt',device='cpu') as f:
            shape_cache[key]=list(f.get_slice(key).get_shape())
    return shape_cache[key]

def load(root):
    trace=glob.glob(str(Path(root)/'**/ASCEND_PROFILER_OUTPUT/trace_view.json'),recursive=True)[0]
    kernel=glob.glob(str(Path(root)/'**/ASCEND_PROFILER_OUTPUT/kernel_details.csv'),recursive=True)[0]
    data=json.loads(Path(trace).read_text())
    rows=list(csv.DictReader(open(kernel)))
    kernels=[]
    for r in rows:
        if 'Matmul' not in r['Name']: continue
        kernels.append(dict(name=r['Name'],start_us=float(r['Start Time(us)']),duration_us=float(r['Duration(us)'])))
    device_matmul_counts=collections.Counter(x['name'] for x in kernels if 'Matmul' in x['name'])
    enq={x['args']['correlation_id']:x for x in data if x.get('cat')=='enqueue' and x.get('ph')=='X' and x.get('name','') in ('Enqueue@aclnnMatmul','Enqueue@aclnnMatmulWeightNz')}
    deq={x['args']['correlation_id']:x for x in data if x.get('cat')=='dequeue' and x.get('ph')=='X' and x.get('name','') in ('Dequeue@aclnnMatmul','Dequeue@aclnnMatmulWeightNz')}
    annotations=[x for x in data if x.get('cat')=='cpu_op' and x.get('ph')=='X' and x.get('name','').startswith('decode_cost/')]
    conversions=[x for x in data if x.get('cat') in ('enqueue','dequeue') and any(k in x.get('name','').lower() for k in ('transdata','formatcast','cast','copy'))]
    out=[]
    available=sorted([x for x in kernels if 'Matmul' in x['name']],key=lambda x:x['start_us'])
    for order,(cid,e) in enumerate(sorted(enq.items(),key=lambda kv:float(kv[1]['ts']))):
        ts=float(e['ts']); d=deq.get(cid)
        # Associate by time containment/nearest CPU model annotation, never infer a layer from order.
        containing=[x for x in annotations if float(x['ts']) <= ts <= float(x['ts'])+float(x['dur'])*1000]
        ann=containing[-1] if containing else min(annotations,key=lambda x:abs(float(x['ts'])-ts)) if annotations else None
        # Device task starts are in the same timestamp domain as trace_view.
        anchor=float(d['ts']) if d else ts
        best=min(available,key=lambda x:abs(x['start_us']-anchor)) if available else None
        k=[dict(best,delta_us=best['start_us']-anchor)] if best and abs(best['start_us']-anchor)<300 else []
        if k: available.remove(best)
        annotation=ann['name'] if ann else None
        module=annotation.split('/',2)[2] if annotation and annotation.startswith('decode_cost/') else None
        ws=weight_shape(module) if module else None
        dims=dict(M=16,N=ws[0],K=ws[1]) if ws and len(ws)==2 else None
        layer_match=re.search(r'\.layers\.(\d+)\.',module or '')
        out.append(dict(order=order,correlation_id=cid,call=e['name'].split('@',1)[1],
                        enqueue_start_us=ts,enqueue_duration_us=float(e['dur']),
                        dequeue_start_us=float(d['ts']) if d else None,
                        dequeue_duration_us=float(d['dur']) if d else None,
                        cpu_annotation=annotation,module=module,
                        layer=int(layer_match.group(1)) if layer_match else None,
                        weight_shape=ws,matmul_dims=dims,dtype='bfloat16' if ws else None,
                        weight_layout='FRACTAL_NZ(format29)' if call_is_nz(e) else 'ND(original)',
                        kernel_matches=k,shape='unavailable_in_trace',
                        layout='unavailable_in_trace',weight_conversion='not_observed_in_matmul_call'))
    conversion_counts=collections.Counter(x['name'] for x in data if x.get('cat')=='enqueue' and x.get('ph')=='X' and any(k in x.get('name','').lower() for k in ('transdata','formatcast','cast','copy')))
    kernel_conversion=collections.Counter()
    for x in kernels:
        if any(k in x['name'].lower() for k in ('transdata','formatcast','copy','cast')):
            kernel_conversion[x['name']]+=x['duration_us']
    csv_path=str(Path(a.output).with_suffix('.csv'))
    fields=['order','layer','module','call','M','N','K','dtype','weight_layout','enqueue_duration_us','dequeue_duration_us','kernel_name','kernel_duration_us','cpu_annotation']
    with open(csv_path,'w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for x in out:
            dims=x['matmul_dims'] or {}; ker=x['kernel_matches'][0] if x['kernel_matches'] else {}
            w.writerow(dict(order=x['order'],layer=x['layer'],module=x['module'],call=x['call'],
                            M=dims.get('M'),N=dims.get('N'),K=dims.get('K'),dtype=x['dtype'],
                            weight_layout=x['weight_layout'],enqueue_duration_us=x['enqueue_duration_us'],
                            dequeue_duration_us=x['dequeue_duration_us'],kernel_name=ker.get('name'),
                            kernel_duration_us=ker.get('duration_us'),cpu_annotation=x['cpu_annotation']))
    return dict(trace=trace,kernel_csv=kernel,call_csv=csv_path,matmuls=out,conversion_events=conversions,
                counts={'matmul_calls':len(out),'regular':sum(x['call']=='aclnnMatmul' for x in out),
                        'weight_nz':sum(x['call']=='aclnnMatmulWeightNz' for x in out),
                        'device_matmul_kernel_counts':dict(device_matmul_counts),
                        'conversion_like_enqueues':sum(conversion_counts.values()),
                        'conversion_like_dispatch_counts':dict(conversion_counts),
                        'conversion_kernel_durations_us':dict(kernel_conversion)})

def call_is_nz(event):
    return event['name'].endswith('WeightNz')

result={'nd':load(a.nd),'nz':load(a.nz)}
Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v['counts'] for k,v in result.items()},indent=2))
