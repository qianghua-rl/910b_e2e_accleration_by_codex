"""Probe Ascend's native recurrent gated-delta-rule operator layouts."""
import json, time, traceback
import torch, torch_npu

torch.manual_seed(0)
device = 'npu'
B, T, H, D = 1, 4, 2, 8
q = torch.randn(B, T, H * D, dtype=torch.bfloat16, device=device)
k = torch.randn_like(q)
v = torch.randn_like(q)
state = torch.zeros(B, H, D, D, dtype=torch.bfloat16, device=device)
beta = torch.sigmoid(torch.randn(B, T, H, dtype=torch.float32, device=device)).to(torch.bfloat16)
g = torch.randn(B, T, H, dtype=torch.float32, device=device)
rows = []
for name, fn in [('inplace', torch_npu.npu_recurrent_gated_delta_rule),
                 ('functional', torch_npu.npu_recurrent_gated_delta_rule_functional)]:
    try:
        s = state.clone()
        t = time.perf_counter()
        out = fn(q, k, v, s, beta=beta, scale=D ** -0.5, g=g)
        torch.npu.synchronize()
        y = out if torch.is_tensor(out) else out[0]
        rows.append(dict(op=name, ok=True, seconds=time.perf_counter() - t,
                         output_shape=str(tuple(y.shape)),
                         output_finite=bool(y.isfinite().all().item()),
                         state_shape=str(tuple(s.shape))))
    except Exception:
        rows.append(dict(op=name, ok=False, error=traceback.format_exc()))
print(json.dumps(rows), flush=True)
open('results/npu_recurrent_probe.json', 'w').write(json.dumps(rows) + '\n')
