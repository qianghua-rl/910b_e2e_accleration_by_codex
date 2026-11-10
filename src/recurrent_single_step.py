"""Generic single-token gated delta rule, retaining reference FP32 operations."""
import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    l2norm, torch_recurrent_gated_delta_rule,
)


def recurrent_single_step(query, key, value, g, beta, initial_state,
                          output_final_state, use_qk_l2norm_in_kernel=False):
    if query.shape[1] != 1:
        return torch_recurrent_gated_delta_rule(
            query, key, value, g, beta, initial_state, output_final_state,
            use_qk_l2norm_in_kernel)
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    q, k, v, beta, g = [
        x[:, 0].contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]
    q = q * (1 / (query.shape[-1] ** 0.5))
    if initial_state is None:
        state = torch.zeros((*k.shape, v.shape[-1]), device=v.device, dtype=v.dtype)
    else:
        state = initial_state.to(v)
    state = state * g.exp().unsqueeze(-1).unsqueeze(-1)
    memory = (state * k.unsqueeze(-1)).sum(dim=-2)
    delta = (v - memory) * beta.unsqueeze(-1)
    state = state + k.unsqueeze(-1) * delta.unsqueeze(-2)
    output = (state * q.unsqueeze(-1)).sum(dim=-2)
    return output.unsqueeze(1).to(initial_dtype), state if output_final_state else None
