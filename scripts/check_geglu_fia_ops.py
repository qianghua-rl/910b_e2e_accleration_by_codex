"""Isolate npu_geglu and npu_fused_infer_attention_score against fp32 references on random data."""
import torch, torch_npu
import torch.nn.functional as F
dev = "npu:1"
torch.manual_seed(0)
# geglu: which half is activated?
x = torch.randn(64, 1408, dtype=torch.float32, device=dev)
a, b = x[:, :704], x[:, 704:]
ref_g_first = F.gelu(a, approximate="tanh") * b
ref_g_second = F.gelu(b, approximate="tanh") * a
out = torch_npu.npu_geglu(x.to(torch.bfloat16), dim=-1, approximate=1)[0].float().cpu()
print("geglu vs gelu(first)*second:", (out - ref_g_first.cpu()).abs().max().item())
print("geglu vs gelu(second)*first:", (out - ref_g_second.cpu()).abs().max().item())

# FIA vs fp32 reference, GQA 16q/8kv and 16q/2kv (global, hd 512), boolean mask
for kvh, hd in [(8, 256), (2, 512)]:
    N, nh, T, t = 4, 16, 512, 300
    q = torch.randn(N, nh, 1, hd, device=dev, dtype=torch.bfloat16)
    K = torch.randn(N, kvh, T, hd, device=dev, dtype=torch.bfloat16)
    V = torch.randn(N, kvh, T, hd, device=dev, dtype=torch.bfloat16)
    scale = 1.0 / hd ** 0.5
    j = torch.arange(t, device=dev)[None, :]
    pos = torch.tensor([100, 200, 299, 50], device=dev)[:, None]
    ok = (j <= pos)
    neg = torch.finfo(torch.bfloat16).min
    mask = torch.where(ok, 0.0, neg).to(torch.bfloat16)[:, None, None, :]
    g = nh // kvh
    Kr = K[:, :, :t][:, :, None].expand(N, kvh, g, t, hd).reshape(N, nh, t, hd).float()
    Vr = V[:, :, :t][:, :, None].expand(N, kvh, g, t, hd).reshape(N, nh, t, hd).float()
    w = (q.float() @ Kr.transpose(2, 3)) * scale + mask.float()
    ref = (torch.softmax(w, -1) @ Vr).cpu()
    o = torch_npu.npu_fused_infer_attention_score(q.contiguous(), K[:, :, :t].contiguous(), V[:, :, :t].contiguous(),
        atten_mask=(mask < 0), num_heads=nh, num_key_value_heads=kvh, input_layout="BNSD", scale=scale)[0].float().cpu()
    print(f"FIA kvh={kvh} hd={hd}: out shape {tuple(o.shape)} max diff {(o - ref).abs().max().item():.4g} ref max {ref.abs().max().item():.3g}")
