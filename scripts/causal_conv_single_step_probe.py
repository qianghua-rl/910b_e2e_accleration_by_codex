"""Local correctness and wrapper timing on Qwen decode dimensions."""
import json
import statistics
import time
from pathlib import Path
import torch
import torch_npu
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_causal_conv1d_update
from causal_conv_single_step import causal_conv_single_step

torch.set_num_threads(8)
torch.manual_seed(42)
rows = []
for batch in (1, 16):
    for bias_enabled in (False, True):
        x = torch.randn(batch, 10240, 1, device='npu', dtype=torch.bfloat16)
        state = torch.randn(batch, 10240, 4, device='npu', dtype=torch.bfloat16)
        weight = torch.randn(10240, 4, device='npu', dtype=torch.bfloat16)
        bias = torch.randn(10240, device='npu', dtype=torch.bfloat16) if bias_enabled else None
        a, b = state.clone(), state.clone()
        expected = torch_causal_conv1d_update(x, a, weight, bias, 'silu')
        actual = causal_conv_single_step(x, b, weight, bias, 'silu')
        error = (actual.float()-expected.float()).abs()
        row = dict(batch=batch, bias=bias_enabled, max_abs_error=error.max().item(),
                   equal=bool(torch.equal(actual, expected)), cache_equal=bool(torch.equal(a,b)),
                   finite=bool(actual.isfinite().all()))
        for name, fn in [('reference',torch_causal_conv1d_update),('candidate',causal_conv_single_step)]:
            work = state.clone()
            for _ in range(4): fn(x, work, weight, bias, 'silu')
            torch.npu.synchronize()
            samples=[]
            for _ in range(12):
                start=time.perf_counter(); fn(x, work, weight, bias, 'silu'); torch.npu.synchronize()
                samples.append((time.perf_counter()-start)*1000)
            row[name+'_median_ms']=statistics.median(samples)
        rows.append(row)
        print(json.dumps(row),flush=True)
Path('results/causal_conv_single_step_probe.json').write_text(json.dumps(rows,indent=2)+'\n')
