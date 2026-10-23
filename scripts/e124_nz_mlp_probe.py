"""Interleaved complete MLP wrapper timings on captured E100 activations."""
import json, statistics, time
from pathlib import Path
import torch, torch_npu
import torch.nn.functional as F
from safetensors import safe_open

torch.set_num_threads(8)
root=Path('model/qwen3.5-27b')
weight_map=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
artifact=torch.load('results/e100_mlp_activations.pt',map_location='cpu',weights_only=False)
rows=[]

def sample(fn):
    torch.npu.synchronize()
    start=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
    t=time.perf_counter();start.record();value=fn();end.record();torch.npu.synchronize()
    wall=(time.perf_counter()-t)*1000
    event=start.elapsed_time(end)
    return value,wall,event

with torch.inference_mode():
  for layer in ('model.language_model.layers.8.mlp','model.language_model.layers.55.mlp'):
    weights=[]
    for op in ('gate_proj','up_proj','down_proj'):
      key=f'{layer}.{op}.weight'
      with safe_open(str(root/weight_map[key]),framework='pt',device='cpu') as f:
        weights.append(f.get_tensor(key).to('npu'))
    nd=weights
    nz=[torch_npu.npu_format_cast(w,29) for w in nd]
    torch.npu.synchronize()
    xs=[s['input'].to('npu') for s in artifact['snapshots'][layer]]
    def wrapper(x,ws):
      wg,wu,wd=ws
      return F.linear(F.silu(F.linear(x,wg))*F.linear(x,wu),wd)
    checks=[]
    for x in xs:
      a=wrapper(x,nd);b=wrapper(x,nz);torch.npu.synchronize()
      err=(a.float()-b.float()).abs()
      checks.append(dict(equal=torch.equal(a,b),max_abs_error=float(err.max()),
                         finite=bool(b.isfinite().all()),shape=list(b.shape),
                         nd_stride=list(a.stride()),nz_stride=list(b.stride()),
                         input_stride=list(x.stride())))
    # Warm both variants, then alternate the same captured activations.
    for i in range(4):
      wrapper(xs[i%len(xs)],nd);wrapper(xs[i%len(xs)],nz)
    torch.npu.synchronize(); samples=[]
    for i in range(20):
      x=xs[i%len(xs)]
      order=('nd','nz') if i%2==0 else ('nz','nd')
      pair={'pair':i,'order':order}
      for kind in order:
        value,wall,event=sample(lambda kind=kind:wrapper(x,nd if kind=='nd' else nz))
        pair[kind]=dict(host_wall_ms=wall,device_event_ms=event,
                        output_checksum=float(value.float().sum()))
      samples.append(pair)
    summary={k:{metric:statistics.median(s[k][metric] for s in samples)
                for metric in ('host_wall_ms','device_event_ms')}
             for k in ('nd','nz')}
    rows.append(dict(layer=layer,input_shape=list(xs[0].shape),input_stride=list(xs[0].stride()),
                     dtype=str(xs[0].dtype),weight_shapes=[list(w.shape) for w in nd],
                     nd_weight_strides=[list(w.stride()) for w in nd],
                     nz_weight_strides=[list(w.stride()) for w in nz],checks=checks,
                     samples=samples,median=summary,
                     paired_delta_pct={m:100*statistics.median(
                         (s['nz'][m]-s['nd'][m])/s['nd'][m] for s in samples)
                         for m in ('host_wall_ms','device_event_ms')},
                     allocated_bytes=torch.npu.memory_allocated(),
                     peak_bytes=torch.npu.max_memory_allocated()))
    Path('results/e124_nz_mlp.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    print(layer,json.dumps(rows[-1]),flush=True)
    del weights,nd,nz,xs
