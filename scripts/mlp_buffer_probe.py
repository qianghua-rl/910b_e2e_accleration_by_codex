"""Complete BF16 MLP allocation vs explicit reusable output buffers."""
import json,time,statistics,argparse
from pathlib import Path
import torch,torch_npu
import torch.nn.functional as F
from safetensors import safe_open
p=argparse.ArgumentParser();p.add_argument('--profile',action='store_true');args=p.parse_args()
torch.set_num_threads(8)
root=Path('model/qwen3.5-27b');index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
r=torch.load('results/e100_mlp_activations.pt',map_location='cpu',weights_only=False)
rows=[]
def measure(fn,queued):
 for i in range(4):fn(i)
 torch.npu.synchronize();host=[];device=[]
 for _ in range(9):
  start=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
  t=time.perf_counter();start.record()
  for i in range(8):
   value=fn(i)
   if not queued:torch.npu.synchronize()
  end.record();torch.npu.synchronize();host.append((time.perf_counter()-t)*1000/8);device.append(start.elapsed_time(end)/8)
 return dict(host_median_ms=statistics.median(host),event_median_ms=statistics.median(device),host_samples_ms=host,event_samples_ms=device)
with torch.inference_mode():
 for layer,samples in r['snapshots'].items():
  weights=[]
  for op in ('gate_proj','up_proj','down_proj'):
   key=f'{layer}.{op}.weight'
   with safe_open(str(root/index[key]),framework='pt',device='cpu') as f:weights.append(f.get_tensor(key).to('npu'))
  wg,wu,wd=weights;xs=[s['input'].to('npu') for s in samples];flat=xs[0].reshape(-1,xs[0].shape[-1]);m=flat.shape[0];n=wg.shape[0]
  bufs=[torch.empty(m,n,device='npu',dtype=torch.bfloat16) for _ in range(4)];out=torch.empty(m,wd.shape[0],device='npu',dtype=torch.bfloat16)
  def baseline(i):
   x=xs[i%len(xs)];return F.linear(F.silu(F.linear(x,wg))*F.linear(x,wu),wd)
  def reused(i):
   x=xs[i%len(xs)];x2=x.reshape(-1,x.shape[-1])
   torch.mm(x2,wg.t(),out=bufs[0])
   torch.ops.aten.silu.out(bufs[0],out=bufs[2])
   torch.mm(x2,wu.t(),out=bufs[1]);torch.mul(bufs[2],bufs[1],out=bufs[3])
   torch.mm(bufs[3],wd.t(),out=out)
   return out.view(*x.shape[:-1],wd.shape[0])
  def owned(i):return reused(i).clone()
  checks=[]
  for i in range(len(xs)):
   a=baseline(i);b=reused(i);torch.npu.synchronize();delta=(a.float()-b.float()).abs()
   checks.append(dict(output_exact=bool(torch.equal(a,b)),max_abs_error=delta.max().item(),finite=bool(b.isfinite().all()),baseline_stride=list(a.stride()),candidate_stride=list(b.stride()),input_stride=list(xs[i].stride())))
  timings=[]
  # Interleaved complete-wrapper trials; all repeated calls read three real activations.
  for queued in (False,True):
   for name,fn in [('baseline',baseline),('reused',reused),('baseline',baseline),('reused_owned_clone',owned),('baseline',baseline)]:
    entry=dict(queued=queued,mode=name,**measure(fn,queued));timings.append(entry);print(layer,json.dumps(entry),flush=True)
  if args.profile:
   for name,fn in [('baseline',baseline),('reused',reused),('reused_owned_clone',owned)]:
    path='results/profile_mlp_buffer_ordered_'+layer.split('.')[-2]+'_'+name
    with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],record_shapes=True,on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(path)):
     for i in range(4):fn(i)
     torch.npu.synchronize()
  rows.append(dict(layer=layer,shape=list(xs[0].shape),dtype=str(xs[0].dtype),checks=checks,timings=timings,persistent_buffer_bytes=sum(b.numel()*b.element_size() for b in bufs+[out]),weight_stride=[list(w.stride()) for w in weights]))
  Path('results/mlp_buffer_probe.json').write_text(json.dumps(dict(rows=rows,activation_artifact='results/e100_mlp_activations.pt',scope='local complete MLP; no model patch',limitations=['weights repeated within layer, not full model residency','reused output aliases next call; owned_clone covers retention cost but not concurrency','event interval includes host gaps; no pure kernel claim','all weights remain BF16; mm out support validated by execution']),indent=2)+'\n')
  del weights,wg,wu,wd,xs,bufs,out
