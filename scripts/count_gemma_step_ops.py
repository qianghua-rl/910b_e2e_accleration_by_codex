"""Count aten/NPU ops dispatched by one Gemma CB decode step (N=16, fixed Tact, grouped + fast-norm).
Uses TorchDispatchMode, so it counts every op the host dispatches (including views). Timing is not measured."""
import argparse, collections, json, sys
sys.path.insert(0, "/remote-home/S45160/work/910b_fresh/src")
import torch
import torch_npu  # noqa: F401
from torch.utils._python_dispatch import TorchDispatchMode
from transformers import AutoModelForCausalLM, AutoTokenizer
from gemma_run import MODEL
from gemma_cb import GemmaCB
from gemma_fast_norm import patch_gemma_norms

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="npu:1")
ap.add_argument("--noscale", action="store_true")
ap.add_argument("--fused-rope", action="store_true")
ap.add_argument("--tact", type=int, default=256)
ap.add_argument("--pos", type=int, default=200)
a = ap.parse_args()
dev, N = a.device, 16
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map=dev).eval()
patch_gemma_norms(model, noscale=a.noscale)
eng = GemmaCB(model, tok, N, 2048, dev, grouped=True, fused_rope=a.fused_rope)
toks = torch.zeros(N, dtype=torch.long, device=dev)
pos = torch.full((N,), a.pos, dtype=torch.long, device=dev)

class Counter(TorchDispatchMode):
    def __init__(self):
        super().__init__(); self.c = collections.Counter()
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.c[str(func)] += 1
        return func(*args, **(kwargs or {}))

with torch.inference_mode():
    eng.decode_step(toks, pos, eng.ar, tact=a.tact)   # warm
    torch.npu.synchronize(dev)
    m = Counter()
    with m:
        eng.decode_step(toks, pos, eng.ar, tact=a.tact)
    torch.npu.synchronize(dev)
total = sum(m.c.values())
print("TOTAL_OPS_PER_STEP", total, "per_layer_approx", round(total / 30, 1))
for name, n in m.c.most_common(30):
    print(f"{n:6d}  {name}")
