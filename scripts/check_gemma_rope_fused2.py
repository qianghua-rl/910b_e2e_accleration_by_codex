"""Test the rotate_half convention of npu_rotary_mul: compare against an explicit reference
x*cos + rotate_half(x)*sin, computed in fp32, for the same inputs. Also try the variant with
the rotation sign flipped, to see which convention the kernel uses."""
import sys
import torch
import torch_npu  # noqa: F401

dev = sys.argv[1] if len(sys.argv) > 1 else "npu:1"
torch.manual_seed(0)
N, nh, hd = 16, 8, 256
x = torch.randn(N, 1, nh, hd, dtype=torch.bfloat16, device=dev)
cos = torch.randn(N, 1, 1, hd, dtype=torch.bfloat16, device=dev)
sin = torch.randn(N, 1, 1, hd, dtype=torch.bfloat16, device=dev)
def rh(t):
    h = t.shape[-1] // 2
    return torch.cat((-t[..., h:], t[..., :h]), dim=-1)
with torch.inference_mode():
    xf, cf, sf = x.float(), cos.float(), sin.float()
    ref = xf * cf + rh(xf) * sf
    k = torch_npu.npu_rotary_mul(x, cos, sin).float()
    print("kernel vs fp32 rotate_half: max", (k - ref).abs().max().item(), flush=True)
    ref_neg = xf * cf - rh(xf) * sf
    print("kernel vs flipped-sign:     max", (k - ref_neg).abs().max().item(), flush=True)
    # same test with the halves swapped (interleaved convention)
    def rh_il(t):
        t1, t2 = t[..., 0::2], t[..., 1::2]
        return torch.stack((-t2, t1), dim=-1).flatten(-2)
    print("kernel vs interleaved:      max", (k - (xf * cf + rh_il(xf) * sf)).abs().max().item(), flush=True)
