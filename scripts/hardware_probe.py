"""Synchronized single-device controlled experiments; no model optimization."""
import json
import time
from pathlib import Path
import torch
import torch_npu

torch.npu.set_device(0)
torch.set_num_threads(8)
out = Path('results/hardware.jsonl')

def emit(row):
    print(json.dumps(row), flush=True)
    with out.open('a') as f:
        f.write(json.dumps(row) + '\n')

def measure(fn, repeats=20):
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    samples = []
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(repeats):
            fn()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - start) / repeats)
    return sorted(samples)[2], samples

emit(dict(kind='device', name=torch.npu.get_device_name(0),
          properties=str(torch.npu.get_device_properties(0)), torch=torch.__version__,
          torch_npu=torch_npu.__version__))
with torch.inference_mode():
    for mb in [1, 16, 64, 256, 512]:
        x = torch.ones(mb * 1024 * 1024 // 2, dtype=torch.bfloat16, device='npu')
        y = torch.empty_like(x)
        dt, samples = measure(lambda: y.copy_(x))
        emit(dict(kind='hbm_copy', mib=mb, seconds=dt, samples=samples,
                  read_write_GBs=2 * x.numel() * 2 / dt / 1e9))
        del x, y
    # Decode to prefill shapes for the actual MLP up projection.
    w = torch.randn(5120, 17408, dtype=torch.bfloat16, device='npu')
    for batch in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
        x = torch.randn(batch, 5120, dtype=torch.bfloat16, device='npu')
        y = torch.empty(batch, 17408, dtype=torch.bfloat16, device='npu')
        dt, samples = measure(lambda: torch.mm(x, w, out=y))
        emit(dict(kind='matmul', m=batch, k=5120, n=17408, seconds=dt,
                  samples=samples, tflops=2*batch*5120*17408/dt/1e12,
                  allocated=torch.npu.memory_allocated(), peak=torch.npu.max_memory_allocated()))
    for mb in [1, 16, 64, 256]:
        host = torch.ones(mb*1024*1024//2, dtype=torch.bfloat16, pin_memory=True)
        dev = torch.empty_like(host, device='npu')
        dt, samples = measure(lambda: dev.copy_(host, non_blocking=True), 5)
        emit(dict(kind='h2d', mib=mb, seconds=dt, samples=samples, GBs=mb*1024*1024/dt/1e9))
