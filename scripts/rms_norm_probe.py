"""Compare Ascend fused RMSNorm with the PyTorch reference on 910B."""
import json, time, torch, torch_npu
torch.manual_seed(0)
x = torch.randn(16, 5120, dtype=torch.bfloat16, device='npu')
w = torch.randn(5120, dtype=torch.bfloat16, device='npu')
eps = 1e-6
def ref():
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps).to(x.dtype) * w
def fused():
    return torch_npu.npu_rms_norm(x, w, eps)[0]
for fn in (ref, fused):
    for _ in range(10): fn()
torch.npu.synchronize()
times = []
for fn in (ref, fused):
    ss = []
    for _ in range(20):
        t = time.perf_counter(); y = fn(); torch.npu.synchronize(); ss.append(time.perf_counter() - t)
    times.append(sorted(ss)[len(ss) // 2])
a = ref().float(); b = fused().float(); d = (a - b).abs()
row = dict(shape=[16, 5120], ref_seconds=times[0], fused_seconds=times[1],
           speedup=times[0] / times[1], max_abs_error=float(d.max().cpu()),
           mean_abs_error=float(d.mean().cpu()),
           argmax_equal=bool(a.argmax(-1).eq(b.argmax(-1)).all().item()))
print(json.dumps(row), flush=True)
open('results/rms_norm_probe.json', 'w').write(json.dumps(row) + '\n')
