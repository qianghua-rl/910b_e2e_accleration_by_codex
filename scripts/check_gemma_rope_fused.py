"""Compare HF apply_rotary_pos_emb (rotate_half, full head_dim) with torch_npu.npu_rotary_mul
on decode shapes: q [N,1,nh,hd], k [N,1,kvh,hd], cos/sin [N,1,hd] unsqueezed at dim 2 (as in the model)."""
import sys
import torch
import torch_npu  # noqa: F401
from transformers.models.gemma4.modeling_gemma4 import apply_rotary_pos_emb

dev = sys.argv[1] if len(sys.argv) > 1 else "npu:1"
torch.manual_seed(0)
N = 16
for nh, hd in [(16, 256), (8, 256), (16, 512), (8, 512), (2, 512)]:
    x = torch.randn(N, 1, nh, hd, dtype=torch.bfloat16, device=dev)
    cos = torch.randn(N, 1, hd, dtype=torch.bfloat16, device=dev)
    sin = torch.randn(N, 1, hd, dtype=torch.bfloat16, device=dev)
    with torch.inference_mode():
        ref = apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=2)
        c4, s4 = cos.unsqueeze(2), sin.unsqueeze(2)          # [N,1,1,hd]
        try:
            fused = torch_npu.npu_rotary_mul(x, c4, s4)
            eq = torch.equal(ref, fused)
            d = (ref.float() - fused.float()).abs().max().item()
            print(f"nh={nh} hd={hd}: shape_ok={tuple(fused.shape)==tuple(ref.shape)} bitwise_equal={eq} max_abs_diff={d:.3g}", flush=True)
        except Exception as e:
            print(f"nh={nh} hd={hd}: FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
