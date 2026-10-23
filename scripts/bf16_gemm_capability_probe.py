"""Bounded ND/physical-transpose/NZ weight capability check; no model patch."""
import argparse,json,time,statistics
from pathlib import Path
import torch,torch_npu
import torch.nn.functional as F
from safetensors import safe_open
parser=argparse.ArgumentParser();parser.add_argument('--layer',type=int,default=8);parser.add_argument('--profile-nz',action='store_true');args=parser.parse_args()
torch.set_num_threads(8)
root=Path('model/qwen3.5-27b');idx=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
r=torch.load('results/e100_mlp_activations.pt',map_location='cpu',weights_only=False)
layer=f'model.language_model.layers.{args.layer}.mlp';xs=[s['input'].to('npu') for s in r['snapshots'][layer]]
weights=[]
for op in ('gate_proj','up_proj','down_proj'):
 key=layer+'.'+op+'.weight'
 with safe_open(str(root/idx[key]),framework='pt',device='cpu') as f:weights.append(f.get_tensor(key).to('npu'))
def baseline(i):
 x=xs[i%len(xs)];g,u,d=weights
 return F.linear(F.silu(F.linear(x,g))*F.linear(x,u),d)
def transpose_call(i):
 x=xs[i%len(xs)];g,u,d=transposed
 return (F.silu(x@g)*(x@u))@d
def nz_call(i):
 x=xs[i%len(xs)];g,u,d=nz
 return F.linear(F.silu(F.linear(x,g))*F.linear(x,u),d)
def measure(fn):
 for i in range(4):fn(i)
 torch.npu.synchronize();samples=[]
 for _ in range(7):
  t=time.perf_counter()
  for i in range(8):value=fn(i)
  torch.npu.synchronize();samples.append((time.perf_counter()-t)*1000/8)
 return {'median_ms':statistics.median(samples),'samples_ms':samples}
result={'layer':args.layer,'torch':torch.__version__,'torch_npu':torch_npu.__version__,'jit_compile_false':torch.npu.is_jit_compile_false(),'input':'E100 layer8 three real decode activations BF16 [16,1,5120]','checks':[],'timings':[]}
with torch.inference_mode():
 torch.npu.synchronize();t=time.perf_counter();transposed=[w.t().contiguous() for w in weights];torch.npu.synchronize();result['transpose_setup_ms']=(time.perf_counter()-t)*1000
 variants=[('baseline',baseline),('physical_transpose',transpose_call)]
 try:
  torch.npu.config.allow_internal_format=True
  torch.npu.synchronize();t=time.perf_counter();nz=[torch_npu.npu_format_cast(w,29) for w in weights];torch.npu.synchronize();result['nz_setup_ms']=(time.perf_counter()-t)*1000
  result['nz_actual_formats']=[torch_npu.get_npu_format(w) for w in nz]
  if all(x==29 for x in result['nz_actual_formats']):variants.append(('nz_weight',nz_call))
 except Exception as e:result['nz_error']=str(e)
 for name,fn in variants:
  try:
   checks=[]
   for i in range(len(xs)):
    a=baseline(i);b=fn(i);torch.npu.synchronize();checks.append(dict(exact=bool(torch.equal(a,b)),finite=bool(b.isfinite().all()),max_abs_error=(a.float()-b.float()).abs().max().item()))
   result['checks'].append(dict(mode=name,checks=checks))
   result['timings'].append(dict(mode='baseline_control',**measure(baseline)))
   result['timings'].append(dict(mode=name,**measure(fn)))
  except Exception as e:result['checks'].append(dict(mode=name,error=str(e)))
  Path(f'results/bf16_gemm_capability_layer{args.layer}.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if args.profile_nz and all(x==29 for x in result.get('nz_actual_formats',[])):
 with torch.inference_mode():
  for name,fn in [('baseline',baseline),('nz',nz_call),('baseline_repeat',baseline),('nz_repeat',nz_call)]:
   result['timings'].append(dict(mode=name,**measure(fn)))
  with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],record_shapes=True,on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(f'results/profile_bf16_nz_layer{args.layer}')):
   for i in range(4):nz_call(i)
   torch.npu.synchronize()
 Path(f'results/bf16_gemm_capability_layer{args.layer}.json').write_text(json.dumps(result,indent=2)+'\n')
