"""Numerical check: experts_grouped (npu_grouped_matmul) vs HF Gemma4TextExperts.forward.
Synthetic bf16 weights at Gemma-4-26B-A4B expert dims; random top-8 routing.
Usage: python -I scripts/check_gemma_grouped_numerics.py [npu:1]
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import torch
import torch_npu  # noqa: F401
import torch.nn.functional as F
from types import SimpleNamespace

from gemma_cb import experts_grouped
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextExperts

dev = sys.argv[1] if len(sys.argv) > 1 else "npu:1"
torch.manual_seed(0)
cfg = SimpleNamespace(num_experts=128, hidden_size=2816, moe_intermediate_size=704,
                      hidden_activation="gelu_pytorch_tanh", _experts_implementation="eager")
ex = Gemma4TextExperts(cfg).to(torch.bfloat16)
with torch.no_grad():
    ex.gate_up_proj.normal_(0, 0.02)
    ex.down_proj.normal_(0, 0.02)
ex = ex.to(dev)
K = 8
worst = 0.0
for T in (1, 4, 16):
    for trial in range(5):
        x = torch.randn(T, cfg.hidden_size, dtype=torch.bfloat16, device=dev)
        probs = F.softmax(torch.randn(T, 128, device=dev), -1)
        w, idx = torch.topk(probs, K, dim=-1)
        w = (w / w.sum(-1, keepdim=True)).to(torch.bfloat16)
        with torch.no_grad():
            ref = ex(x, idx, w)
            out = experts_grouped(ex, x, idx, w)
        torch.npu.synchronize()
        d = (ref.float() - out.float()).abs()
        rel = d.max().item() / max(ref.float().abs().max().item(), 1e-9)
        worst = max(worst, rel)
        print(f"T={T} trial={trial} max_abs={d.max().item():.3e} "
              f"max_rel={rel:.3e} ref_absmax={ref.float().abs().max().item():.3e} "
              f"finite={bool(torch.isfinite(out).all())}", flush=True)
print(f"WORST_REL {worst:.3e}")
