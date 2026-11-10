import sys
import torch, torch_npu  # noqa: F401
sys.path.insert(0, "/remote-home/S45160/work/910b_fresh/src")
from sync_free import l2norm_nosync, recurrent_nosync, fused_rmsnorm_nosync  # noqa: E402
from transformers.models.qwen3_5 import modeling_qwen3_5 as M  # noqa: E402
from recurrent_single_step import recurrent_single_step  # noqa: E402

dev = "npu:0"; torch.manual_seed(0)
B, H, D = 16, 16, 128
with torch.inference_mode():
    x = torch.randn(B, H, D, device=dev, dtype=torch.bfloat16)
    print("l2norm_diff", (M.l2norm(x) - l2norm_nosync(x)).abs().max().item())
    q, k = torch.randn(B, 1, H, D, device=dev, dtype=torch.bfloat16), torch.randn(B, 1, H, D, device=dev, dtype=torch.bfloat16)
    v = torch.randn(B, 1, H, D, device=dev, dtype=torch.bfloat16)
    g = torch.randn(B, 1, H, device=dev, dtype=torch.float32) * 0.1
    beta = torch.rand(B, 1, H, device=dev, dtype=torch.bfloat16)
    st = torch.randn(B, H, D, D, device=dev, dtype=torch.float32) * 0.01
    o1, s1 = recurrent_single_step(q, k, v, g, beta, st.clone(), True, True)
    o2, s2 = recurrent_nosync(q, k, v, g, beta, st.clone(), True, True)
    print("recurrent_out_diff", (o1.float() - o2.float()).abs().max().item(), "state_diff", (s1 - s2).abs().max().item(), "out_scale", o1.float().abs().mean().item())
    class N(torch.nn.Module):
        def __init__(s):
            super().__init__(); s.weight = torch.nn.Parameter(torch.randn(5120, device=dev, dtype=torch.bfloat16)); s.eps = 1e-6
    n = N(); n._w1 = (1.0 + n.weight).contiguous()
    xr = torch.randn(B, 1, 5120, device=dev, dtype=torch.bfloat16)
    ref = torch_npu.npu_rms_norm(xr, (1.0 + n.weight).to(xr.dtype), n.eps)[0]
    new = fused_rmsnorm_nosync(n, xr)
    print("rmsnorm_diff", (ref.float() - new.float()).abs().max().item())
