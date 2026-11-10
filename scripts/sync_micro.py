"""Which small op triggers aten::_local_scalar_dense on NPU? Count per op over 10 calls each."""
import sys, os, json, collections, re
import torch, torch_npu  # noqa: F401
import torch_npu.profiler as tnp
OUT = sys.argv[1]
dev = "npu:0"
B, H, D = 15, 16, 128
x = torch.randn(B, H, D, device=dev, dtype=torch.bfloat16)
w = torch.randn(D, device=dev, dtype=torch.bfloat16)
st = torch.randn(B, H, D, D, device=dev, dtype=torch.float32)
g = torch.randn(B, H, device=dev, dtype=torch.float32)
eps_t = torch.tensor(1e-6, device=dev, dtype=torch.bfloat16)
eps_f32 = torch.tensor(1e-6, device=dev, dtype=torch.float32)
w1 = (1.0 + w).contiguous()
scale_t = torch.tensor(1 / (D ** 0.5), device=dev, dtype=torch.bfloat16)
q_bf = torch.randn(B, H, D, device=dev, dtype=torch.bfloat16)
ops = {
    "npu_rms_norm_precomputed_w": lambda: torch_npu.npu_rms_norm(x, w1, 1e-6)[0],
    "mul_pyfloat_bf16": lambda: q_bf * (1 / (D ** 0.5)),
    "mul_devtensor_bf16": lambda: q_bf * scale_t,
    "mul_pyfloat_f32": lambda: st * (1 / (D ** 0.5)),
    "add_eps_devtensor_bf16": lambda: (x * x).sum(dim=-1, keepdim=True) + eps_t,
    "add_eps_f32_path": lambda: ((x * x).float().sum(dim=-1, keepdim=True) + eps_f32),
    "rsqrt_devtensor": lambda: torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps_t),
    "manual_rmsnorm_devtensor": lambda: x * torch.rsqrt((x * x).float().mean(dim=-1, keepdim=True) + eps_f32).to(x.dtype) * (1.0 + w),
    "npu_rms_norm": lambda: torch_npu.npu_rms_norm(x, (1.0 + w).to(x.dtype), 1e-6)[0],
    "sum_keepdim": lambda: (x * x).sum(dim=-1, keepdim=True),
    "add_eps": lambda: (x * x).sum(dim=-1, keepdim=True) + 1e-6,
    "rsqrt": lambda: torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + 1e-6),
    "contig_f32": lambda: x.contiguous().to(torch.float32),
    "exp_unsq": lambda: st * g.exp().unsqueeze(-1).unsqueeze(-1),
    "to_bf16": lambda: st.to(torch.bfloat16),
}
os.makedirs(OUT, exist_ok=True)
import glob
def count_sync(d):
    tr=[os.path.join(r,"trace_view.json") for r,_,f in os.walk(d) if "trace_view.json" in f]
    if not tr: return None
    ev=json.load(open(tr[0])); ev=ev["traceEvents"] if isinstance(ev,dict) else ev
    return sum(1 for e in ev if e.get("ph")=="X" and e.get("name")=="aten::_local_scalar_dense")
for name, fn in ops.items():
    fn(); torch.npu.synchronize(dev)
    d=os.path.join(OUT,name)
    prof = tnp.profile(activities=[tnp.ProfilerActivity.CPU, tnp.ProfilerActivity.NPU],
                       on_trace_ready=tnp.tensorboard_trace_handler(d), with_stack=False, record_shapes=False)
    prof.start()
    for _ in range(10):
        fn(); torch.npu.synchronize(dev)
    prof.stop()
    print("SYNC", name, count_sync(d), flush=True)
