"""E130-A: separate gate/up weight identity from first/second call position."""
import json, random, statistics, time
from collections import defaultdict
from pathlib import Path
import torch, torch_npu
import torch.nn.functional as F
from safetensors import safe_open

torch.set_num_threads(8)
torch.manual_seed(19)
random.seed(19)
root=Path('model/qwen3.5-27b')
weight_map=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
capture=torch.load('results/e100_mlp_activations.pt',map_location='cpu',weights_only=False)
layers=(8,55)
repeats=100
results=[]

def event_call(fn):
    start=torch.npu.Event(enable_timing=True); end=torch.npu.Event(enable_timing=True)
    start.record(); host_start=time.perf_counter(); out=fn(); host_ms=(time.perf_counter()-host_start)*1000
    end.record()
    return out, start, end, host_ms

with torch.inference_mode():
  for layer_no in layers:
    prefix=f'model.language_model.layers.{layer_no}.mlp'
    tensors={}
    for op in ('gate_proj','up_proj','down_proj'):
      key=f'{prefix}.{op}.weight'
      with safe_open(str(root/weight_map[key]),framework='pt',device='cpu') as f:
        tensors[op]=f.get_tensor(key).to('npu')
    xs=[z['input'].to('npu') for z in capture['snapshots'][prefix]]
    wg,wu,wd=(tensors[x] for x in ('gate_proj','up_proj','down_proj'))
    def mlp(x,order):
      outputs={}; activated_gate=None
      for name in order:
        outputs[name]=F.linear(x,tensors[name])
        if name=='gate_proj': activated_gate=F.silu(outputs[name])
      return F.linear(activated_gate*outputs['up_proj'],wd),outputs
    checks=[]
    for x in xs:
      base,base_parts=mlp(x,('gate_proj','up_proj')); swapped,swap_parts=mlp(x,('up_proj','gate_proj'))
      torch.npu.synchronize()
      checks.append(dict(gate_exact=bool(torch.equal(base_parts['gate_proj'],swap_parts['gate_proj'])),
                         up_exact=bool(torch.equal(base_parts['up_proj'],swap_parts['up_proj'])),
                         mlp_exact=bool(torch.equal(base,swapped)),finite=bool(base.isfinite().all() and swapped.isfinite().all()),
                         output_shape=list(base.shape),output_stride=list(base.stride()),input_stride=list(x.stride())))
    # Compile/warm both invocation orders on every captured activation before sampling.
    for i in range(4):
      x=xs[i%len(xs)]; mlp(x,('gate_proj','up_proj')); mlp(x,('up_proj','gate_proj'))
    torch.npu.synchronize()
    samples=[]
    for pair in range(repeats):
      x=xs[pair%len(xs)]
      pair_orders=('gate_up','up_gate') if pair%2==0 else ('up_gate','gate_up')
      for order_name in pair_orders:
        order=('gate_proj','up_proj') if order_name=='gate_up' else ('up_proj','gate_proj')
        wrapper_start=torch.npu.Event(enable_timing=True); wrapper_start.record()
        pair_start=time.perf_counter()
        call_rows={}
        outputs={}
        activated_gate=None
        for pos,name in enumerate(order):
          value,start,end,host_ms=event_call(lambda name=name:F.linear(x,tensors[name]))
          outputs[name]=value
          call_rows[name]=dict(position=pos,host_submit_ms=host_ms,start=start,end=end)
          if name=='gate_proj': activated_gate=F.silu(value)
        # Preserve the model's gate -> SiLU dependency in both schedules.
        result=F.linear(activated_gate*outputs['up_proj'],wd)
        wrapper_end=torch.npu.Event(enable_timing=True);wrapper_end.record()
        pair_submit_ms=(time.perf_counter()-pair_start)*1000
        sample=dict(trial=pair,order=order_name,pair_submit_ms=pair_submit_ms,
                    wrapper_start=wrapper_start,wrapper_end=wrapper_end,projection={})
        for name,row in call_rows.items():
          sample['projection'][name]=dict(position=row['position'],host_submit_ms=row['host_submit_ms'],
                                           start=row['start'],end=row['end'],
                                           checksum=float(outputs[name].float().sum()))
        sample['wrapper_result_checksum']=float(result.float().sum())
        samples.append(sample)
    # Keep both schedules in one continuously submitted stream; synchronize only once.
    torch.npu.synchronize()
    for sample in samples:
      sample['pair_wrapper_ms']=sample['wrapper_start'].elapsed_time(sample['wrapper_end'])
      del sample['wrapper_start'],sample['wrapper_end']
      for projection in sample['projection'].values():
        projection['device_event_ms']=projection['start'].elapsed_time(projection['end'])
        del projection['start'],projection['end']
    # Paired per-weight contrasts across first/second positions; all samples are preserved.
    summary={}
    for name in ('gate_proj','up_proj'):
      first=[s['projection'][name]['device_event_ms'] for s in samples if s['projection'][name]['position']==0]
      second=[s['projection'][name]['device_event_ms'] for s in samples if s['projection'][name]['position']==1]
      host_first=[s['projection'][name]['host_submit_ms'] for s in samples if s['projection'][name]['position']==0]
      host_second=[s['projection'][name]['host_submit_ms'] for s in samples if s['projection'][name]['position']==1]
      summary[name]=dict(first_position_median_ms=statistics.median(first),
                         second_position_median_ms=statistics.median(second),
                         first_position_host_median_ms=statistics.median(host_first),
                         second_position_host_median_ms=statistics.median(host_second),
                         device_position_delta_pct=100*(statistics.median(second)-statistics.median(first))/statistics.median(first))
    pair_wall={order:statistics.median(s['pair_wrapper_ms'] for s in samples if s['order']==order)
               for order in ('gate_up','up_gate')}
    paired_order_delta=[(next(s['pair_wrapper_ms'] for s in samples if s['trial']==i and s['order']=='up_gate')-
                         next(s['pair_wrapper_ms'] for s in samples if s['trial']==i and s['order']=='gate_up'))
                        for i in range(repeats)]
    rng=random.Random(130+layer_no)
    bootstrap=sorted(statistics.median(rng.choices(paired_order_delta,k=len(paired_order_delta)))
                     for _ in range(5000))
    row=dict(layer=layer_no,shape={'M':16,'K':5120,'N':17408},dtype=str(xs[0].dtype),
             input_shape=list(xs[0].shape),input_stride=list(xs[0].stride()),
             gate_weight_shape=list(wg.shape),gate_weight_stride=list(wg.stride()),
             up_weight_shape=list(wu.shape),up_weight_stride=list(wu.stride()),
             gate_npu_format=int(torch_npu.get_npu_format(wg)),up_npu_format=int(torch_npu.get_npu_format(wu)),
             gate_data_ptr_mod_512=int(wg.data_ptr()%512),up_data_ptr_mod_512=int(wu.data_ptr()%512),
             repeats_per_order=sum(s['order']=='gate_up' for s in samples),checks=checks,
             projection_summary=summary,pair_wrapper_median_ms=pair_wall,
             paired_wrapper_delta_up_gate_minus_gate_up_ms=dict(median=statistics.median(paired_order_delta),
                 p10=sorted(paired_order_delta)[int(.1*(repeats-1))],p90=sorted(paired_order_delta)[int(.9*(repeats-1))],
                 bootstrap_95pct_ci_ms=[bootstrap[124],bootstrap[4874]],
                 up_gate_faster_pairs=sum(x<0 for x in paired_order_delta),
                 samples=paired_order_delta),samples=samples,
             methodology='100 same-activation paired order crossovers, order sequence alternates; continuous same-stream submissions with no synchronization between order trials and one final synchronization; 4 warmups/order; 3 captured real activations; identical per-call NPU events; exact original ND BF16 Linear weights and HF gate->SiLU semantics.')
    results.append(row)
    Path('results/e130_gate_up_crossover.json').write_text(json.dumps(dict(layers=results,limitations=[
      'Standalone pair and local complete MLP wrapper, not full model decode.',
      'Event time includes stream scheduling/queue effects; it is not isolated Cube execution time.',
      'Gate/up calls use distinct real weights. Crossing their order separates weight identity from position; it does not prove why a weight-specific difference occurs.',
      'No full Decode or end-to-end change is enabled in this experiment.']),indent=2)+'\n')
    print(json.dumps({k:row[k] for k in ('layer','projection_summary','pair_wrapper_median_ms','checks')}),flush=True)
    del tensors,xs
