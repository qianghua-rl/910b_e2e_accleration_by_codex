"""Check shift_causal_conv1d_update against the reference on NPU: outputs and updated state."""
import sys
import torch, torch_npu  # noqa: F401
sys.path.insert(0, "/remote-home/S45160/work/910b_fresh/src")
from shift_conv_update import shift_causal_conv1d_update  # noqa: E402
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_causal_conv1d_update  # noqa: E402

dev = "npu:0"
torch.manual_seed(0)
B, C, K = 16, 10240, 4
w = torch.randn(C, K, device=dev, dtype=torch.bfloat16) * 0.2
b = torch.randn(C, device=dev, dtype=torch.bfloat16) * 0.1
s_ref = torch.randn(B, C, K, device=dev, dtype=torch.bfloat16)
s_new = s_ref.clone()
max_out, max_state = 0.0, 0.0
with torch.inference_mode():
    for step in range(20):
        x = torch.randn(B, C, 1, device=dev, dtype=torch.bfloat16)
        y_ref = torch_causal_conv1d_update(x.clone(), s_ref, w, b, "silu")
        y_new = shift_causal_conv1d_update(x.clone(), s_new, w, b, "silu")
        max_out = max(max_out, (y_ref.float() - y_new.float()).abs().max().item())
        max_state = max(max_state, (s_ref.float() - s_new.float()).abs().max().item())
print(f"max_abs_out_diff={max_out:.6f} max_abs_state_diff={max_state:.6f} out_scale={y_ref.float().abs().mean().item():.4f}")
