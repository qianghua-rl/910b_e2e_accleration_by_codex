"""KKT-only PTO adapter for vLLM's C=64 GDN pipeline."""
from __future__ import annotations

import torch

from megagdn_pto.kernel_libs import (
    chunk_gdn_causal_masks,
    run_scaled_dot_kkt,
    transpose_beta,
    transpose_gates,
)


def chunk_scaled_dot_kkt_fwd_pto(
    k, beta, g_cumsum=None, cu_seqlens=None, chunk_indices=None,
    chunk_size=64, output_dtype=torch.float32,
):
    if chunk_size != 64 or cu_seqlens is None or k.device.type != "npu":
        raise RuntimeError("KKT-only PTO adapter requires NPU varlen C=64 inputs")
    batch, length, key_heads, dim = k.shape
    heads = beta.shape[2]
    if g_cumsum is None or g_cumsum.shape != beta.shape:
        raise ValueError('Expected externally computed g_cumsum with beta shape')
    cu32 = cu_seqlens.to(torch.int32).contiguous()
    packed_k = k.reshape(1, -1, key_heads, dim).to(torch.float16).contiguous()
    packed_beta = beta.reshape(1, -1, heads).to(torch.float16).contiguous()
    packed_g = g_cumsum.reshape(1, -1, heads).contiguous()
    mask, _ = chunk_gdn_causal_masks("npu", k.device.index or 0, 64)
    g_t = transpose_gates(packed_g)
    beta_t = transpose_beta(packed_beta)
    out = torch.empty(1, packed_k.shape[1], heads, 64, device=k.device, dtype=torch.float16)
    run_scaled_dot_kkt(
        packed_k, packed_beta, packed_g, mask, out,
        g_t=g_t, beta_t=beta_t, chunk_size=64, cu_seqlens=cu32,
        batch_size_override=cu32.numel() - 1, key_heads=key_heads,
    )
    return out.view(batch, length, heads, 64).to(output_dtype)
