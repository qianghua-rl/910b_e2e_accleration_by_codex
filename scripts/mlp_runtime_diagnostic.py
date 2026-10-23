"""Diagnostic only: real MLP weights, M scaling, residency and submission."""
import json,time,statistics
from pathlib import Path
import torch,torch_npu
import torch.nn.functional as F
from safetensors import safe_open

torch.set_num_threads(8);torch.manual_seed(42)
root=Path('model/qwen3.5-27b');index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
def load(layer):
 out={}
 for op in ('gate_proj','up_proj','down_proj'):
  key=f'model.language_model.layers.{layer}.mlp.{op}.weight'
  with safe_open(str(root/index[key]),framework='pt',device='cpu') as f:out[op]=f.get_tensor(key).to('npu')
 return out
weights=[load(i) for i in (8,9,10,11)]
def mlp(x,w):return F.linear(F.silu(F.linear(x,w['gate_proj']))*F.linear(x,w['up_proj']),w['down_proj'])
def measure(fn,count,sync_each):
 for i in range(3):fn(i)
 torch.npu.synchronize();hosts=[];devices=[];enqueue=[]
 for _ in range(5):
  a=torch.npu.Event(enable_timing=True);b=torch.npu.Event(enable_timing=True)
  t=time.perf_counter();a.record()
  for i in range(count):
   y=fn(i)
   if sync_each:torch.npu.synchronize()
  b.record();submitted=time.perf_counter();torch.npu.synchronize()
  hosts.append((time.perf_counter()-t)*1000/count);enqueue.append((submitted-t)*1000/count);devices.append(a.elapsed_time(b)/count)
 return dict(host_ms=statistics.median(hosts),event_ms=statistics.median(devices),submission_ms=statistics.median(enqueue),host_samples_ms=hosts,event_samples_ms=devices)
rows=[]
with torch.inference_mode():
 for m in (1,16,64):
  x=torch.randn(m,5120,device='npu',dtype=torch.bfloat16)
  for op in ('gate_proj','up_proj','down_proj','mlp'):
   xx=x if op!='down_proj' else torch.randn(m,17408,device='npu',dtype=torch.bfloat16)
   def call(i,rotate=False):
    w=weights[i%len(weights)] if rotate else weights[0]
    return mlp(xx,w) if op=='mlp' else F.linear(xx,w[op])
   for mode in ('serial','queued','rotating'):
    result=measure(lambda i:call(i,mode=='rotating'),8,mode=='serial')
    row=dict(m=m,op=op,mode=mode,**result);rows.append(row);print(json.dumps(row),flush=True)
result={'rows':rows,'environment':{'torch':torch.__version__,'torch_npu':torch_npu.__version__,'device':torch.npu.get_device_name(0)},'weights_layers':[8,9,10,11],'activation':'synthetic standard normal BF16; not model quality evidence','limitations':['event window includes submission starvation/gaps, not pure kernel time','serial vs queued changes synchronization, not model semantics or a deployable runtime','rotating uses four checkpoint layers, not controlled cache flush; weight contents differ','microbenchmark does not model full-model cache residency or E100 throughput']}
Path('results/mlp_runtime_diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
