"""Bitwise check: Gemma4RMSNorm(with_scale=False) original forward vs npu_rms_norm with gamma=1.
Dims cover v_norm (head_dim 256 and 512) and router norm (hidden 2816). bf16 inputs, shapes as used in decode."""
import sys
import torch
import torch.nn as nn
import torch_npu  # noqa: F401
from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm
sys.path.insert(0, "/remote-home/S45160/work/910b_fresh/src")
from gemma_fast_norm import patch_gemma_norms

dev = sys.argv[1] if len(sys.argv) > 1 else "npu:1"
torch.manual_seed(0)
for dim in (256, 512, 2816):
    ref = Gemma4RMSNorm(dim, eps=1e-6, with_scale=False).to(dev).eval()
    holder = nn.Sequential(Gemma4RMSNorm(dim, eps=1e-6, with_scale=False)).to(dev).eval()
    holder[0].load_state_dict({}, strict=False)
    patch_gemma_norms(holder, noscale=True)
    worst = 0.0; equal = True
    with torch.inference_mode():
        for shape in [(16, 1, 1, dim), (16, 1, dim), (1, 512, dim)]:
            for scale in (1.0, 30.0, 0.01):
                x = (torch.randn(*shape, dtype=torch.bfloat16, device=dev) * scale)
                y0 = ref(x)
                y1 = holder[0](x)
                equal &= torch.equal(y0, y1)
                worst = max(worst, (y0.float() - y1.float()).abs().max().item())
    print(f"dim={dim}: bitwise_equal={equal}, max_abs_diff={worst:.3g}", flush=True)
