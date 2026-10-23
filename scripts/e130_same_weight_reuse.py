"""Distinguish call-position cost from same-weight reuse on real E100 activations."""
import json, random, statistics, time
from pathlib import Path
import torch, torch_npu
import torch.nn.functional as F
from safetensors import safe_open

torch.set_num_threads(8)
random.seed(1301)
root=Path('model/qwen3.5-27b')
weight_map=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
capture=torch.load('results/e100_mlp_activations.pt',map_location='cpu',weights_only=False)
results=[]

def call(x,w):
    start=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
    start.record();out=F.linear(x,w);end.record()
    return out,start,end

with torch.inference_mode():
  for layer in (8,55):
    prefix=f'model.language_model.layers.{layer}.mlp'
    xs=[s['input'].to('npu') for s in capture['snapshots'][prefix]]
    weights={}
    for op in ('gate_proj','up_proj'):
      key=f'{prefix}.{op}.weight'
      with safe_open(str(root/weight_map[key]),framework='pt',device='cpu') as f:
        weights[op]=f.get_tensor(key).to('npu')
    # Warm every scenario and activation, then one sync before the continuous sample stream.
    scenarios={'gate_up':('gate_proj','up_proj'),'up_gate':('up_proj','gate_proj'),
               'gate_gate':('gate_proj','gate_proj'),'up_up':('up_proj','up_proj')}
    for x in xs:
      for seq in scenarios.values():
        for name in seq:F.linear(x,weights[name])
    torch.npu.synchronize()
    schedule=[name for name in scenarios for _ in range(60)]
    random.shuffle(schedule)
    samples=[]
    for i,scenario in enumerate(schedule):
      x=xs[i%len(xs)]; seq=scenarios[scenario]; starts=[];ends=[];outputs=[]
      for name in seq:
        y,s,e=call(x,weights[name]);outputs.append(y);starts.append(s);ends.append(e)
      samples.append(dict(sample=i,scenario=scenario,first_weight=seq[0],second_weight=seq[1],
                          starts=starts,ends=ends,
                          output_exact=bool(torch.equal(outputs[0],outputs[1])) if seq[0]==seq[1] else None,
                          checksum=[float(y.float().sum()) for y in outputs]))
    torch.npu.synchronize()
    clean=[]
    for s in samples:
      ev=[a.elapsed_time(b) for a,b in zip(s.pop('starts'),s.pop('ends'))]
      s['event_ms']=ev;clean.append(s)
    summary={}
    for scenario in scenarios:
      arr=[s['event_ms'] for s in clean if s['scenario']==scenario]
      summary[scenario]=dict(calls=len(arr),first_position_median_ms=statistics.median(x[0] for x in arr),
                             second_position_median_ms=statistics.median(x[1] for x in arr),
                             first_position_samples_ms=[x[0] for x in arr],
                             second_position_samples_ms=[x[1] for x in arr],
                             duplicate_exact=all(s['output_exact'] for s in clean if s['scenario']==scenario) if scenario in ('gate_gate','up_up') else None)
    results.append(dict(layer=layer,shape={'M':16,'K':5120,'N':17408},dtype=str(xs[0].dtype),
                        input_stride=list(xs[0].stride()),
                        weight_strides={k:list(v.stride()) for k,v in weights.items()},
                        npu_formats={k:int(torch_npu.get_npu_format(v)) for k,v in weights.items()},
                        pointer_mod_512={k:int(v.data_ptr()%512) for k,v in weights.items()},
                        summary=summary,samples=clean,
                        method='60 samples/scenario, randomized scenario order, same captured activation cycling, continuous NPU stream, one final synchronize; each sequence has two Matmuls with no intervening synchronize.'))
    Path('results/e130_same_weight_reuse.json').write_text(json.dumps(dict(layers=results,limitations=[
      'Repeated identical weights test cache/reuse opportunity but are not the actual gate/up model sequence.',
      'NPU event intervals include stream/device state and do not reveal physical HBM bytes.',
      'No model patch or full Decode result.']),indent=2)+'\n')
    print(json.dumps({'layer':layer,'summary':summary}),flush=True)
    del xs,weights
