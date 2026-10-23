"""E130 cache control: evict L2 between two-projection sequences."""
import json,random,statistics,time
from pathlib import Path
import torch,torch_npu
import torch.nn.functional as F
from safetensors import safe_open

torch.set_num_threads(8); rng=random.Random(1302)
root=Path('model/qwen3.5-27b')
weight_map=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
capture=torch.load('results/e100_mlp_activations.pt',map_location='cpu',weights_only=False)
scenarios={'gate_up':('gate_proj','up_proj'),'up_gate':('up_proj','gate_proj'),
           'gate_gate':('gate_proj','gate_proj'),'up_up':('up_proj','up_proj')}
output=[]

def timed_gemm(x,w):
    start=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
    start.record();y=F.linear(x,w);end.record()
    return y,start,end

with torch.inference_mode():
  for layer in (8,55):
    prefix=f'model.language_model.layers.{layer}.mlp'
    xs=[s['input'].to('npu') for s in capture['snapshots'][prefix]]
    weights={}
    for op in ('gate_proj','up_proj'):
      key=f'{prefix}.{op}.weight'
      with safe_open(str(root/weight_map[key]),framework='pt',device='cpu') as f:
        weights[op]=f.get_tensor(key).to('npu')
    # 512 MiB exceeds the 192 MiB L2, forcing intervening traffic before each scenario.
    scratch=torch.empty(512*1024*1024//2,dtype=torch.bfloat16,device='npu')
    def evict():
      torch.npu.synchronize()
      scratch.add_(1)
      torch.npu.synchronize()
    # Warm every launch path and verify exact duplicate-weight outputs first.
    for x in xs:
      for seq in scenarios.values():
        ys=[F.linear(x,weights[name]) for name in seq]
        if seq[0]==seq[1]: assert torch.equal(*ys)
    torch.npu.synchronize()
    schedule=[name for name in scenarios for _ in range(30)]
    rng.shuffle(schedule)
    samples=[]
    for trial,scenario in enumerate(schedule):
      evict()
      x=xs[trial%len(xs)];seq=scenarios[scenario];events=[];ys=[]
      for name in seq:
        y,s,e=timed_gemm(x,weights[name]);ys.append(y);events.append((s,e))
      torch.npu.synchronize()
      vals=[s.elapsed_time(e) for s,e in events]
      samples.append(dict(trial=trial,scenario=scenario,sequence=list(seq),event_ms=vals,
                          duplicate_exact=bool(torch.equal(*ys)) if seq[0]==seq[1] else None,
                          checksums=[float(y.float().sum()) for y in ys]))
    summary={}
    for scenario in scenarios:
      rows=[s for s in samples if s['scenario']==scenario]
      summary[scenario]=dict(calls=len(rows),first_median_us=statistics.median(s['event_ms'][0] for s in rows)*1000,
                             second_median_us=statistics.median(s['event_ms'][1] for s in rows)*1000,
                             first_samples_us=[s['event_ms'][0]*1000 for s in rows],
                             second_samples_us=[s['event_ms'][1]*1000 for s in rows],
                             duplicate_exact=all(s['duplicate_exact'] for s in rows) if scenario in ('gate_gate','up_up') else None)
    output.append(dict(layer=layer,shape={'M':16,'K':5120,'N':17408},dtype=str(xs[0].dtype),
                       input_stride=list(xs[0].stride()),weight_strides={k:list(v.stride()) for k,v in weights.items()},
                       weight_formats={k:int(torch_npu.get_npu_format(v)) for k,v in weights.items()},
                       scratch_bytes=scratch.numel()*scratch.element_size(),scenario_summary=summary,samples=samples,
                       method='30 randomized trials/scenario; synchronize and read/write 512 MiB scratch (>192 MiB L2) before each pair, then two same-shape GEMMs with no intervening synchronization; final pair synchronization.'))
    Path('results/e130_weight_cache_control.json').write_text(json.dumps(dict(layers=output,limitations=[
      'This isolates cache residency in two calls but does not reproduce full-model contention or weight traversal.',
      'NPU event intervals include device scheduling; physical cache hit/miss counters are not available.',
      'No model implementation change or full Decode result.']),indent=2)+'\n')
    print(json.dumps({'layer':layer,'summary':summary}),flush=True)
    del scratch,weights,xs
