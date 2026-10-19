"""Distinguish npugraph_ex backend health from Qwen recurrent capture issues."""
import json
import time
import traceback
from pathlib import Path

import torch
import torch_npu


torch.set_num_threads(8)
torch.manual_seed(42)
x = torch.randn(16, 1024, device="npu", dtype=torch.bfloat16)
w = torch.randn(1024, 1024, device="npu", dtype=torch.bfloat16)


def simple(a, b):
    return torch.nn.functional.silu(a @ b)


with torch.inference_mode():
    reference = simple(x, w)
    torch.npu.synchronize()
    row = {"backend": "npugraph_ex", "shape": [16, 1024, 1024]}
    try:
        fn = torch.compile(simple, backend="npugraph_ex", fullgraph=True, dynamic=False)
        started = time.perf_counter()
        compiled = fn(x, w)
        torch.npu.synchronize()
        first_seconds = time.perf_counter() - started
        reference_samples = []
        compiled_samples = []
        for _ in range(8):
            started = time.perf_counter()
            simple(x, w)
            torch.npu.synchronize()
            reference_samples.append(time.perf_counter() - started)
            started = time.perf_counter()
            fn(x, w)
            torch.npu.synchronize()
            compiled_samples.append(time.perf_counter() - started)
        row.update(
            compile_first_call_seconds=first_seconds,
            output_bit_equal=torch.equal(reference, compiled),
            max_error=float((reference - compiled).abs().max()),
            reference_samples=reference_samples,
            compiled_samples=compiled_samples,
        )
    except Exception:
        row["error"] = traceback.format_exc()

Path("results/npugraph_ex_simple_probe.json").write_text(json.dumps(row, indent=2) + "\n")
print(json.dumps(row), flush=True)
