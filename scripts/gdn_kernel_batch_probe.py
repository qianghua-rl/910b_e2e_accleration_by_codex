"""Benchmark the available vLLM Ascend GDN kernel at decode batch size 16."""
import json
import time
from pathlib import Path

import torch
import torch_npu
from vllm.model_executor.layers.fla.ops import fused_recurrent_gated_delta_rule

torch.manual_seed(42)
batch, heads, key_dim, value_dim = 16, 48, 128, 128
q = torch.randn(batch, 1, heads, key_dim, device="npu", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn(batch, 1, heads, value_dim, device="npu", dtype=torch.bfloat16)
g = -torch.rand(batch, 1, heads, device="npu", dtype=torch.bfloat16)
beta = torch.rand(batch, 1, heads, device="npu", dtype=torch.bfloat16)
state = torch.randn(batch, heads, value_dim, key_dim, device="npu", dtype=torch.bfloat16)
state_indices = torch.arange(batch, device="npu", dtype=torch.int32)
state_before = state.clone()

def reference_step(q, k, v, g, beta, initial_state):
    qf, kf, vf = q[:, 0].float(), k[:, 0].float(), v[:, 0].float()
    gf, bf = g[:, 0].float(), beta[:, 0].float()
    qf = qf / (qf.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    kf = kf / (kf.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    st = initial_state.float()
    st = st * gf.exp().unsqueeze(-1).unsqueeze(-1)
    memory = (st * kf.unsqueeze(-1)).sum(-2)
    delta = (vf - memory) * bf.unsqueeze(-1)
    st = st + delta.unsqueeze(-1) * kf.unsqueeze(-2)
    out = (st * qf.unsqueeze(-1)).sum(-2).to(v.dtype).unsqueeze(1)
    return out, st.to(initial_state.dtype)

with torch.inference_mode():
    test_state = state_before.clone()
    test_out, test_final = fused_recurrent_gated_delta_rule(
        q, k, v, g, beta, initial_state=test_state,
        inplace_final_state=True, ssm_state_indices=state_indices,
        use_qk_l2norm_in_kernel=True)
    torch.npu.synchronize()
    ref_formula_out, ref_formula_state = reference_step(
        q, k, v, g, beta, state_before)
    torch.npu.synchronize()
    ref_out, ref_states = [], []
    for i in range(batch):
        one_state = state_before[i:i+1].clone()
        one_out, _ = fused_recurrent_gated_delta_rule(
            q[i:i+1], k[i:i+1], v[i:i+1], g[i:i+1], beta[i:i+1],
            initial_state=one_state, inplace_final_state=True,
            ssm_state_indices=torch.zeros(1, device="npu", dtype=torch.int32),
            use_qk_l2norm_in_kernel=True)
        ref_out.append(one_out)
        ref_states.append(one_state)
    ref_out = torch.cat(ref_out)
    ref_states = torch.cat(ref_states)
    bench_state = state_before.clone()
    samples = []
    for _ in range(8):
        started = time.perf_counter()
        for _ in range(20):
            out, final = fused_recurrent_gated_delta_rule(
                q, k, v, g, beta, initial_state=bench_state,
                inplace_final_state=True, ssm_state_indices=state_indices,
                use_qk_l2norm_in_kernel=True)
        torch.npu.synchronize()
        samples.append((time.perf_counter() - started) / 20)

row = {"batch": batch, "heads": heads, "key_dim": key_dim, "value_dim": value_dim,
       "samples_seconds": samples, "median_seconds": sorted(samples)[len(samples)//2],
       "finite": bool(torch.isfinite(out).all() and torch.isfinite(final).all()),
       "output_shape": list(out.shape), "state_shape": list(final.shape)}
row["state_changed_rows"] = int((test_state != state_before).flatten(1).any(1).sum().item())
row["state_distinct_rows"] = int(torch.unique(test_state.flatten(1), dim=0).shape[0])
row["batched_vs_single_rows_output_equal"] = bool(torch.equal(test_out, ref_out))
row["batched_vs_single_rows_state_equal"] = bool(torch.equal(test_state, ref_states))
row["batched_vs_single_rows_output_max_error"] = float((test_out - ref_out).abs().max().item())
row["batched_vs_single_rows_state_max_error"] = float((test_state - ref_states).abs().max().item())
row["kernel_vs_reference_output_max_error"] = float((test_out - ref_formula_out).abs().max().item())
row["kernel_vs_reference_state_max_error"] = float((test_state - ref_formula_state).abs().max().item())
Path("results/gdn_kernel_batch_probe.json").write_text(json.dumps(row, indent=2)+"\n")
print(json.dumps(row), flush=True)
