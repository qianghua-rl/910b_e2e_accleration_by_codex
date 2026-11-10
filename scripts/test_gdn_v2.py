"""Compare gdn_chunk_fast2 against gdn_chunk_fast (v1, verified) on random inputs of prefill shape,
then time both. Qwen3.5-27B GDN: 48 value heads, dk=dv=128 (taken from the model config)."""
import sys, time, torch, torch_npu  # noqa: F401
sys.path.insert(0, "/remote-home/S45160/work/910b_fresh/src")
from gdn_chunk_fast import fast_chunk_gated_delta_rule as v1
from gdn_chunk_fast2 import fast_chunk_gated_delta_rule_v2 as v2
dev = "npu:0"; torch.manual_seed(0)
B, L, H, dk, dv = 1, 353, 48, 128, 128
def mk():
    q = torch.randn(B, L, H, dk, device=dev, dtype=torch.bfloat16)
    k = torch.randn(B, L, H, dk, device=dev, dtype=torch.bfloat16)
    v = torch.randn(B, L, H, dv, device=dev, dtype=torch.bfloat16)
    g = -torch.rand(B, L, H, device=dev, dtype=torch.float32) * 0.5
    b = torch.rand(B, L, H, device=dev, dtype=torch.bfloat16)
    return q, k, v, g, b
with torch.inference_mode():
    q, k, v, g, b = mk()
    o1, s1 = v1(q, k, v, g, b, use_qk_l2norm_in_kernel=True, output_final_state=True)
    o2, s2 = v2(q, k, v, g, b, use_qk_l2norm_in_kernel=True, output_final_state=True)
    print("out max|diff|", (o1.float() - o2.float()).abs().max().item(), "rel", ((o1.float()-o2.float()).norm()/o1.float().norm()).item())
    print("state max|diff|", (s1 - s2).abs().max().item(), "state scale", s1.abs().max().item())
    for name, fn in [("v1", v1), ("v2", v2)]:
        ts = []
        for _ in range(6):
            torch.npu.synchronize(dev); t0 = time.perf_counter()
            fn(q, k, v, g, b, use_qk_l2norm_in_kernel=True, output_final_state=True)
            torch.npu.synchronize(dev); ts.append(time.perf_counter() - t0)
        print(name, "mean ms", round(sum(ts[1:]) / 5 * 1000, 2))
