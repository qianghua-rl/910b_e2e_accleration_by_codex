"""Reproduce E121 and expand matmul boundaries with paired, fixed-input trials."""
import json, statistics, time
from pathlib import Path
import torch, torch_npu
import torch.nn.functional as F
from safetensors import safe_open

torch.set_num_threads(8)
root=Path('model/qwen3.5-27b')
index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
capture=torch.load('results/e100_mlp_activations.pt',map_location='cpu',weights_only=False)
all_rows=[]

def timed_block(fn):
    torch.npu.synchronize()
    start=torch.npu.Event(enable_timing=True); end=torch.npu.Event(enable_timing=True)
    t=time.perf_counter(); start.record()
    values=[fn(i) for i in range(8)]
    end.record(); torch.npu.synchronize()
    return (time.perf_counter()-t)*1000/8, start.elapsed_time(end)/8, values[-1]

with torch.inference_mode():
  for layer_no in (8,55):
    layer=f'model.language_model.layers.{layer_no}.mlp'
    xs=[z['input'].to('npu') for z in capture['snapshots'][layer]]
    nd=[]
    for op in ('gate_proj','up_proj','down_proj'):
      key=f'{layer}.{op}.weight'
      with safe_open(str(root/index[key]),framework='pt',device='cpu') as f:
        nd.append(f.get_tensor(key).to('npu'))
    torch.npu.config.allow_internal_format=True
    torch.npu.synchronize(); prep=time.perf_counter()
    nz=[torch_npu.npu_format_cast(w,29) for w in nd]
    torch.npu.synchronize(); prep_ms=(time.perf_counter()-prep)*1000
    assert [torch_npu.get_npu_format(w) for w in nz]==[29,29,29]
    def linear(x,w): return F.linear(x,w)
    def wrapper(x,ws):
      wg,wu,wd=ws
      return F.linear(F.silu(F.linear(x,wg))*F.linear(x,wu),wd)
    checks=[]
    for i,x in enumerate(xs):
      a=wrapper(x,nd);b=wrapper(x,nz);torch.npu.synchronize()
      checks.append(dict(input=i,bit_exact=torch.equal(a,b),finite=bool(b.isfinite().all()),
                         max_abs_error=float((a.float()-b.float()).abs().max()),
                         input_shape=list(x.shape),input_stride=list(x.stride()),
                         output_shape=list(b.shape),output_stride=list(b.stride())))
    timings=[]
    for boundary in ('matmul_gate','matmul_up','matmul_down','full_mlp'):
      for mode,ws in (('nd',nd),('nz',nz)):
        if boundary=='full_mlp': fn=lambda i,ws=ws:wrapper(xs[i%len(xs)],ws)
        else:
          idx={'matmul_gate':0,'matmul_up':1,'matmul_down':2}[boundary]
          if boundary=='matmul_down':
            acts=[F.silu(F.linear(x,nd[0]))*F.linear(x,nd[1]) for x in xs]
            torch.npu.synchronize()
            fn=lambda i,ws=ws,idx=idx:F.linear(acts[i%len(acts)],ws[idx])
          else: fn=lambda i,ws=ws,idx=idx:F.linear(xs[i%len(xs)],ws[idx])
        # Match E121: warm each variant four calls, then eight queued calls per sample,
        # with one sync at the measured window boundary. Alternate every paired window.
        for i in range(4): fn(i)
        torch.npu.synchronize()
        timings.append(dict(boundary=boundary,mode=mode,samples=[]))
      entries={x['mode']:x for x in timings if x['boundary']==boundary}
      fns={}
      for mode,ws in (('nd',nd),('nz',nz)):
        if boundary=='full_mlp': fns[mode]=lambda i,ws=ws:wrapper(xs[i%len(xs)],ws)
        elif boundary=='matmul_down': fns[mode]=lambda i,ws=ws:F.linear(acts[i%len(acts)],ws[2])
        else:
          idx=0 if boundary=='matmul_gate' else 1
          fns[mode]=lambda i,ws=ws,idx=idx:F.linear(xs[i%len(xs)],ws[idx])
      for pair in range(20):
        order=('nd','nz') if pair%2==0 else ('nz','nd')
        for mode in order:
          host,event,out=timed_block(fns[mode])
          entries[mode]['samples'].append(dict(pair=pair,order=list(order),host_wall_ms=host,
                                               device_event_ms=event,output_checksum=float(out.float().sum())))
      for mode in ('nd','nz'):
        entries[mode]['median_host_ms']=statistics.median(s['host_wall_ms'] for s in entries[mode]['samples'])
        entries[mode]['median_event_ms']=statistics.median(s['device_event_ms'] for s in entries[mode]['samples'])
      entries['nd']['paired_delta_pct']={m:100*statistics.median(
        (entries['nz']['samples'][i][m]-entries['nd']['samples'][i][m])/
         entries['nd']['samples'][i][m] for i in range(20)) for m in ('host_wall_ms','device_event_ms')}
    row=dict(layer=layer_no,weight_shapes=[list(w.shape) for w in nd],nd_strides=[list(w.stride()) for w in nd],
             nz_strides=[list(w.stride()) for w in nz],dtype=str(xs[0].dtype),nz_setup_ms=prep_ms,
             setup_in_timing=False,checks=checks,boundaries=timings,
             memory_allocated=torch.npu.memory_allocated(),peak_bytes=torch.npu.max_memory_allocated(),
             methodology='E121 F.linear/SiLU/mul ordering; 4 warmups per variant; 20 alternating pairs; 8 calls/window; synchronize at window boundaries; host wall and NPU event.')
    all_rows.append(row)
    Path('results/e125_nz_boundaries.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in all_rows))
    print(json.dumps(row),flush=True)
    del nd,nz,xs
