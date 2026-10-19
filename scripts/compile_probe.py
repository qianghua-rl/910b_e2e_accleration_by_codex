"""Small generic recurrent-rule graph compatibility probe before full model use."""
import json
import time
import traceback
from pathlib import Path
import torch
import torch_npu
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_recurrent_gated_delta_rule

torch.set_num_threads(8)
torch.manual_seed(42)
args=[torch.randn(1,1,48,128,device='npu',dtype=torch.bfloat16) for _ in range(3)]
args += [-torch.rand(1,1,48,device='npu'),torch.rand(1,1,48,device='npu'),
         torch.randn(1,48,128,128,device='npu'),True,True]
def bench(fn):
    for _ in range(3): fn(*args)
    torch.npu.synchronize()
    samples=[]
    for _ in range(5):
        t=time.perf_counter()
        for _ in range(20): fn(*args)
        torch.npu.synchronize()
        samples.append((time.perf_counter()-t)/20)
    return samples

with torch.inference_mode():
    reference=torch_recurrent_gated_delta_rule(*args)
    row=dict(reference_samples=bench(torch_recurrent_gated_delta_rule),
             compiler='torchair',mode='reduce-overhead',shape=[1,1,48,128])
    try:
        import torchair
        config=torchair.CompilerConfig()
        config.mode='reduce-overhead'
        fn=torch.compile(torch_recurrent_gated_delta_rule,
            backend=torchair.get_npu_backend(compiler_config=config),fullgraph=True,dynamic=False)
        t=time.perf_counter()
        result=fn(*args)
        torch.npu.synchronize()
        row.update(compile_first_call_seconds=time.perf_counter()-t,
            output_bit_equal=torch.equal(reference[0],result[0]),
            state_bit_equal=torch.equal(reference[1],result[1]),
            output_max_error=float((reference[0]-result[0]).abs().max()),
            state_max_error=float((reference[1]-result[1]).abs().max()),samples=bench(fn))
    except Exception:
        row['error']=traceback.format_exc()
    Path('results/compile_probe.json').write_text(json.dumps(row,indent=2)+'\n')
    print(json.dumps(row),flush=True)
