"""Generic per-tensor FP8 reference for later Phase 3 kernel validation.

This is a numerical specification, not an accelerated NPU implementation.
No calibration on GSM8K labels or workload-dependent scales.
"""
import torch

def quantize_per_tensor(x):
    if not x.is_floating_point():
        raise TypeError('Expected floating input')
    maximum = torch.finfo(torch.float8_e4m3fn).max
    scale = x.float().abs().amax().clamp_min(torch.finfo(torch.float32).tiny) / maximum
    quantized = (x.float()/scale).clamp(-maximum,maximum).to(torch.float8_e4m3fn)
    return quantized, scale

def linear_reference(x, weight, bias=None):
    """FP8 E4M3FN activations/weights, one scale each; FP32 accumulation."""
    qx, sx = quantize_per_tensor(x)
    qw, sw = quantize_per_tensor(weight)
    out = (qx.float() @ qw.float().T) * (sx*sw)
    if bias is not None:
        out = out + bias.float()
    return out.to(x.dtype)

if __name__=='__main__':
    torch.manual_seed(42)
    x = torch.randn(4,128,dtype=torch.bfloat16)
    w = torch.randn(256,128,dtype=torch.bfloat16)
    y = linear_reference(x,w)
    assert y.shape==(4,256) and y.isfinite().all()
    q,s = quantize_per_tensor(torch.zeros(2,2))
    assert q.float().eq(0).all() and s>0
    print('FP8 reference smoke passed; no accuracy/performance conclusion')
