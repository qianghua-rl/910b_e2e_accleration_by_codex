"""Step 1: can an HCCL all_reduce be captured inside an NPUGraph (TP=2, one process per NPU)?

Each rank runs with ASCEND_RT_VISIBLE_DEVICES=<rank> and RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT set.
Graph body: x(static) @ W1 -> dist.all_reduce -> @ W2.
Checks:
  1. capture succeeds (or the error is printed)
  2. replay output == eager output on the same input (5 replays)
  3. after copy_() of a new input into the static buffer, replay == eager on the new input
     (proves the graph reads the static buffer and the all_reduce really runs on replay)
Run (both ranks at once):
  RANK=r ASCEND_RT_VISIBLE_DEVICES=r WORLD_SIZE=2 MASTER_ADDR=127.0.0.1 MASTER_PORT=29611 \
      /remote-home/S45160/miniforge3/envs/ascend-e2e/bin/python -I scripts/test_hccl_graph.py
"""
import os
import sys
import traceback

import torch
import torch_npu  # noqa: F401
import torch.distributed as dist

rank = int(os.environ["RANK"])
world = int(os.environ["WORLD_SIZE"])
dev = "npu:0"  # ASCEND_RT_VISIBLE_DEVICES=<rank> makes the physical card local index 0
torch.npu.set_device(dev)

dist.init_process_group("hccl", rank=rank, world_size=world)

N, D, H = 4, 64, 128
torch.manual_seed(0)  # identical weights on both ranks
W1 = torch.randn(D, H, dtype=torch.bfloat16, device=dev) * 0.05
W2 = torch.randn(H, D, dtype=torch.bfloat16, device=dev) * 0.05

x = torch.zeros(N, D, dtype=torch.bfloat16, device=dev)  # static input buffer


def core():
    y = x @ W1
    dist.all_reduce(y)
    return y @ W2


def eager(inp):
    x.copy_(inp)
    return core().clone()


def log(msg):
    print(f"[rank{rank}] {msg}", flush=True)


torch.manual_seed(100 + rank)  # different per-rank input so the all_reduce matters
in_a = torch.randn(N, D, dtype=torch.bfloat16, device=dev)
in_b = torch.randn(N, D, dtype=torch.bfloat16, device=dev)

ok = True
with torch.inference_mode():
    ref_a = eager(in_a)
    ref_b = eager(in_b)
    torch.npu.synchronize(dev)
    log("eager reference computed")

    try:
        # warmup on the capture stream so lazy HCCL/op init happens outside capture
        s = torch.npu.Stream(device=dev)
        s.wait_stream(torch.npu.current_stream(dev))
        with torch.npu.stream(s):
            x.copy_(in_a)
            core()
        torch.npu.current_stream(dev).wait_stream(s)
        torch.npu.synchronize(dev)

        g = torch.npu.NPUGraph()
        x.copy_(in_a)
        with torch.npu.graph(g):
            out = core()
        torch.npu.synchronize(dev)
        log("CAPTURE OK (all_reduce inside graph)")

        # 2. replay on input A
        x.copy_(in_a)
        for _ in range(5):
            g.replay()
        torch.npu.synchronize(dev)
        d = (out - ref_a).abs().max().item()
        same_a = torch.equal(out, ref_a)
        log(f"replay(A) vs eager(A): max|diff|={d} equal={same_a}")
        ok &= same_a

        # 3. update static buffer to input B, replay, compare with eager(B)
        x.copy_(in_b)
        g.replay()
        torch.npu.synchronize(dev)
        d = (out - ref_b).abs().max().item()
        same_b = torch.equal(out, ref_b)
        changed = not torch.equal(ref_a, ref_b)
        log(f"replay(B after copy_) vs eager(B): max|diff|={d} equal={same_b} (A!=B: {changed})")
        ok &= same_b and changed
    except Exception:
        ok = False
        log("CAPTURE/REPLAY FAILED:")
        traceback.print_exc()

dist.destroy_process_group()
if rank == 0:
    print("RESULT:", "PASS" if ok else "FAIL", flush=True)
sys.exit(0 if ok else 1)
