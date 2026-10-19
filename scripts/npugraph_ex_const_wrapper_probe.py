"""Check whether fixing recurrent boolean arguments avoids graph host copies."""
import json
import time
import traceback
from pathlib import Path

import torch
import torch_npu
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_recurrent_gated_delta_rule

torch.set_num_threads(8)
torch.manual_seed(42)
args = [torch.randn(16, 1, 48, 128, device="npu", dtype=torch.bfloat16) for _ in range(3)]
args += [-torch.rand(16, 1, 48, device="npu"), torch.rand(16, 1, 48, device="npu"),
         torch.randn(16, 48, 128, 128, device="npu")]

def wrapped(query, key, value, g, beta, initial_state):
    return torch_recurrent_gated_delta_rule(query, key, value, g, beta, initial_state, True, True)

row = {"backend": "npugraph_ex", "shape": [16, 1, 48, 128], "bool_args": "closure"}
with torch.inference_mode():
    reference = wrapped(*args)
    try:
        fn = torch.compile(wrapped, backend="npugraph_ex", fullgraph=True, dynamic=False)
        started = time.perf_counter()
        result = fn(*args)
        torch.npu.synchronize()
        samples = []
        for _ in range(8):
            t = time.perf_counter()
            for _ in range(20):
                fn(*args)
            torch.npu.synchronize()
            samples.append((time.perf_counter() - t) / 20)
        row.update(compile_first_call_seconds=time.perf_counter()-started,
                   output_bit_equal=torch.equal(reference[0], result[0]),
                   state_bit_equal=torch.equal(reference[1], result[1]),
                   output_max_error=float((reference[0]-result[0]).abs().max()),
                   state_max_error=float((reference[1]-result[1]).abs().max()),
                   compiled_samples=samples)
    except Exception:
        row["error"] = traceback.format_exc()
Path("results/npugraph_ex_const_wrapper_probe.json").write_text(json.dumps(row, indent=2)+"\n")
print(json.dumps(row), flush=True)
