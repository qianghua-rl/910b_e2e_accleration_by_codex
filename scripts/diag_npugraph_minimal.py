"""Minimal NPUGraph capture check in the ascend-e2e env: does replay recompute a captured op?"""
import sys
import torch
import torch_npu  # noqa: F401

dev = sys.argv[1] if len(sys.argv) > 1 else "npu:1"
x = torch.zeros(16, 2816, dtype=torch.bfloat16, device=dev)
w = torch.randn(2816, 2816, dtype=torch.bfloat16, device=dev)
with torch.inference_mode():
    y = torch.matmul(x, w)                       # eager warmup
    torch.npu.synchronize(dev)
    g = torch.npu.NPUGraph()
    with torch.npu.graph(g):
        out = torch.matmul(x, w) * 2
    torch.npu.synchronize(dev)
    x.fill_(1.0)
    out.fill_(7.0)
    g.replay()
    torch.npu.synchronize(dev)
    ref = torch.matmul(x, w) * 2
    print("replay overwrote sentinel:", bool((out != 7.0).any().item()))
    print("replay matches eager:", torch.allclose(out.float(), ref.float(), atol=1e-2))
