"""Small probe: does NPU graph replay recompute captured work in this env?
A: torch.npu.graph, warmup on a side stream (torch.cuda-style recipe).
B: torch.compile(backend='npugraph_ex') on the same function, compared with eager.
Both check replay against a changed input (not just the sentinel)."""
import sys
import torch
import torch_npu  # noqa: F401

dev = sys.argv[1] if len(sys.argv) > 1 else "npu:1"
torch.manual_seed(0)
w = torch.randn(1024, 1024, dtype=torch.bfloat16, device=dev)
x = torch.zeros(16, 1024, dtype=torch.bfloat16, device=dev)


def fn(a):
    return torch.nn.functional.silu(torch.matmul(a, w)) * 2


with torch.inference_mode():
    # ---- A: side-stream warmup + capture ----
    s = torch.npu.Stream(device=dev)
    s.wait_stream(torch.npu.current_stream(dev))
    with torch.npu.stream(s):
        for _ in range(3):
            fn(x)
    torch.npu.current_stream(dev).wait_stream(s)
    torch.npu.synchronize(dev)
    g = torch.npu.NPUGraph()
    with torch.npu.graph(g):
        out = fn(x)
    torch.npu.synchronize(dev)
    x.copy_(torch.randn_like(x))                 # change input after capture
    ref = fn(x)
    torch.npu.synchronize(dev)
    g.replay()
    torch.npu.synchronize(dev)
    print("A side-stream graph: replay matches eager on new input:",
          torch.equal(out, ref), "max diff:", (out.float() - ref.float()).abs().max().item(), flush=True)

    # ---- B: npugraph_ex backend ----
    try:
        import torch._dynamo as dynamo
        print("B backends listed:", "npugraph_ex" in dynamo.list_backends(), flush=True)
        cf = torch.compile(fn, backend="npugraph_ex", fullgraph=True, dynamic=False)
        y0 = cf(x)
        torch.npu.synchronize(dev)
        x.copy_(torch.randn_like(x))
        ref2 = fn(x)
        y1 = cf(x)
        torch.npu.synchronize(dev)
        print("B npugraph_ex: matches eager on new input:", torch.equal(y1, ref2),
              "max diff:", (y1.float() - ref2.float()).abs().max().item(), flush=True)
    except Exception as e:
        print("B npugraph_ex failed:", type(e).__name__, str(e)[:300], flush=True)
