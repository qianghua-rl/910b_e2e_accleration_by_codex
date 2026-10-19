"""Generic per-tensor INT8 linear probe, now allowed after FP8 constraint removal."""
import json,time,torch,torch_npu
torch.manual_seed(42); torch.npu.set_device(0)
def quant(x):
    s=x.float().abs().amax().clamp_min(1e-12)/127
    return (x.float()/s).round().clamp(-127,127).to(torch.int8),s
row={'device':torch.npu.get_device_name(0),'shapes':[]}
with torch.inference_mode():
  for m in [1,8,32,128]:
    x=torch.randn(m,5120,device='npu',dtype=torch.bfloat16); w=torch.randn(17408,5120,device='npu',dtype=torch.bfloat16)
    qx,sx=quant(x); qw,sw=quant(w); scale=(sx*sw).reshape(1)
    ref=(x.float()@w.float().t()).to(torch.bfloat16)
    try:
      t=time.perf_counter(); y=torch_npu.npu_quant_matmul(qx,qw,scale,output_dtype=torch.bfloat16); torch.npu.synchronize()
      first=time.perf_counter()-t; samples=[]
      for _ in range(10):
        t=time.perf_counter(); torch_npu.npu_quant_matmul(qx,qw,scale,output_dtype=torch.bfloat16); torch.npu.synchronize(); samples.append(time.perf_counter()-t)
      row['shapes'].append({'m':m,'seconds':sorted(samples)[len(samples)//2],'first_seconds':first,'max_abs_error':float((y.float()-ref.float()).abs().max()),'argmax_equal':bool(torch.equal(y.argmax(-1),ref.argmax(-1)))})
    except Exception as e: row['shapes'].append({'m':m,'error':repr(e)})
print(json.dumps(row,indent=2)); open('results/int8_npu_probe.json','w').write(json.dumps(row,indent=2)+'\n')
