import json, time
from pathlib import Path
import torch, torch_npu

torch.manual_seed(7)
out=[]
for m,k,n in [(1,5120,17408),(8,5120,17408)]:
    x=torch.randn((m,k),device='npu',dtype=torch.bfloat16)
    w=torch.randn((k,n),device='cpu',dtype=torch.float32)
    yref=x@w.to(torch.bfloat16).npu()
    for group in [0,128,256]:
      try:
        if group == 0:
            scale=w.abs().amax(dim=0,keepdim=True).clamp_min(1e-8)/7
        else:
            scale=w.view(k//group,group,n).abs().amax(dim=1).clamp_min(1e-8)/7
            scale=scale.repeat_interleave(group,dim=0)
        wi=torch.round(w/scale).clamp(-8,7).to(torch.int32)
        packed=torch_npu.npu_convert_weight_to_int4pack(wi.npu())
        asc=scale[:1] if group == 0 else (w.view(k//group,group,n).abs().amax(dim=1).clamp_min(1e-8)/7)
        gsize=0 if group==0 else group
        def run():
            return torch_npu.npu_weight_quant_batchmatmul(x,packed,asc.to(torch.bfloat16).npu(),None,antiquant_group_size=gsize)
        for _ in range(2): y=run(); torch.npu.synchronize()
        ts=[]
        for _ in range(5):
            t=time.perf_counter(); y=run(); torch.npu.synchronize(); ts.append(time.perf_counter()-t)
        yf=y.float(); rf=yref.float()
        err=yf-rf
        out.append(dict(m=m,k=k,n=n,group=group,ok=True,median_s=sorted(ts)[len(ts)//2],relative_rms=float(err.square().mean().sqrt()/rf.square().mean().sqrt()),cosine=float(torch.nn.functional.cosine_similarity(yf.flatten(),rf.flatten(),dim=0))))
      except Exception as e:
        out.append(dict(m=m,k=k,n=n,group=group,ok=False,error=str(e)))
Path('results/int4_weight_probe.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
