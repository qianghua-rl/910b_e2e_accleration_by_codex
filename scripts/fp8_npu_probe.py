"""Capability probe only: does this 910B3 expose native FP8 per-tensor GEMM?"""
import json, traceback
import torch
import torch_npu

torch.manual_seed(42)
row={"device":torch.npu.get_device_name(0),"shape":[4,128,256]}
with torch.inference_mode():
    x=torch.randn(4,128,device='npu',dtype=torch.bfloat16)
    w=torch.randn(256,128,device='npu',dtype=torch.bfloat16)
    finfo=torch.finfo(torch.float8_e4m3fn)
    sx=x.float().abs().amax().clamp_min(1e-12)/finfo.max
    sw=w.float().abs().amax().clamp_min(1e-12)/finfo.max
    try:
        qx=(x.cpu().float()/sx.cpu()).clamp(-finfo.max,finfo.max).to(torch.float8_e4m3fn).to('npu')
        qw=(w.cpu().float()/sw.cpu()).clamp(-finfo.max,finfo.max).to(torch.float8_e4m3fn).to('npu')
        torch.npu.synchronize()
        y=torch.mm(qx,qw.t())
        torch.npu.synchronize()
        row.update(torch_mm=True,output_dtype=str(y.dtype),output_shape=list(y.shape))
    except Exception:
        row.update(torch_mm=False,torch_mm_error=traceback.format_exc())
    try:
        row["npu_quant_matmul_float8"] = str(torch_npu.npu_quant_matmul(qx,qw,torch.tensor([sx*sw],device='npu'),output_dtype=torch.bfloat16).dtype)
    except Exception:
        row["npu_quant_matmul_float8_error"] = traceback.format_exc()
print(json.dumps(row,indent=2))
open('results/fp8_npu_probe.json','w').write(json.dumps(row,indent=2)+'\n')
