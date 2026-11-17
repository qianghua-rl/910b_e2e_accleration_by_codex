"""Enqueue vs. device-wait split for one Gemma CB decode step (N=16, fixed Tact, grouped + fast-norm).
t_enq : time for decode_step() to return (host work: Python + op dispatch), no sync
t_tot : time until device is idle (synchronize)
If t_enq ~ t_tot, the step is host/dispatch bound; if t_enq << t_tot, device bound.
Timing-only: K/V state is not prefilled; positions are fixed, so every step does identical work."""
import argparse, json, sys, time
sys.path.insert(0, "/remote-home/S45160/work/910b_fresh/src")
import torch
import torch_npu  # noqa: F401
from transformers import AutoModelForCausalLM, AutoTokenizer
from gemma_run import MODEL
from gemma_cb import GemmaCB
from gemma_fast_norm import patch_gemma_norms

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="npu:1")
ap.add_argument("--noscale", action="store_true")
ap.add_argument("--fused-rope", action="store_true")
ap.add_argument("--iters", type=int, default=30)
ap.add_argument("--pos", type=int, default=200)
ap.add_argument("--tact", type=int, default=256)
a = ap.parse_args()
dev = a.device
N = 16
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map=dev).eval()
patch_gemma_norms(model, noscale=a.noscale)
eng = GemmaCB(model, tok, N, 2048, dev, grouped=True, fused_rope=a.fused_rope)
toks = torch.zeros(N, dtype=torch.long, device=dev)
pos = torch.full((N,), a.pos, dtype=torch.long, device=dev)
with torch.inference_mode():
    for _ in range(3):
        eng.decode_step(toks, pos, eng.ar, tact=a.tact)
    torch.npu.synchronize(dev)
    enq, tot = [], []
    for _ in range(a.iters):
        t0 = time.perf_counter()
        eng.decode_step(toks, pos, eng.ar, tact=a.tact)
        t1 = time.perf_counter()
        torch.npu.synchronize(dev)
        t2 = time.perf_counter()
        enq.append(t1 - t0); tot.append(t2 - t0)
ms = lambda v: round(1000 * sorted(v)[len(v) // 2], 1)
res = {"median_enqueue_ms": ms(enq), "median_total_ms": ms(tot),
       "enqueue_fraction": round(sorted(enq)[len(enq)//2] / sorted(tot)[len(tot)//2], 3)}
print("RESULT", json.dumps(res), flush=True)
