"""Chunked GDN prefill v2: same math as gdn_chunk_fast.fast_chunk_gated_delta_rule, but everything that
does not depend on the running state is computed once for all chunks (batched), so the sequential
loop only does the state-dependent products: vp=Kc@S, vn=V-vp, out=Qg@S + QK@vn, S=S*Gl + Kd^T@vn.
Reuses the (verified) intra-chunk inverse from gdn_chunk_fast."""
import torch
import torch.nn.functional as F

from gdn_chunk_fast import _l2norm, _unit_lower_inv


def fast_chunk_gated_delta_rule_v2(query, key, value, g, beta, chunk_size=64, initial_state=None,
                                   output_final_state=False, use_qk_l2norm_in_kernel=False):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]
    B, H, L, dk = key.shape
    dv = value.shape[-1]
    pad = (chunk_size - L % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad))
    key = F.pad(key, (0, 0, 0, pad))
    value = F.pad(value, (0, 0, 0, pad))
    beta = F.pad(beta, (0, pad))
    g = F.pad(g, (0, pad))
    Lp = L + pad
    query = query * (1 / (dk ** 0.5))

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    C = chunk_size
    n = Lp // C
    query, key, value, k_beta, v_beta = [
        x.reshape(B, H, n, C, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)]
    g = g.reshape(B, H, n, C).cumsum(dim=-1)
    tri_strict = torch.triu(torch.ones(C, C, dtype=torch.bool, device=query.device), diagonal=0)
    tri_incl = torch.triu(torch.ones(C, C, dtype=torch.bool, device=query.device), diagonal=1)

    decay = (g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float().tril()     # [B,H,n,C,C]
    X = -((k_beta @ key.transpose(-1, -2)) * decay).masked_fill(tri_strict, 0)
    attn = _unit_lower_inv(X)

    V = attn @ v_beta                                   # [B,H,n,C,dv]
    Kc = attn @ (k_beta * g.exp().unsqueeze(-1))        # [B,H,n,C,dk]
    QK = ((query @ key.transpose(-1, -2)) * decay).masked_fill(tri_incl, 0)   # intra-chunk, causal
    Qg = query * g.exp().unsqueeze(-1)                  # [B,H,n,C,dk]
    glast = g[..., -1]                                  # [B,H,n]
    Gl = glast.exp()
    Kd = (key * (glast.unsqueeze(-1) - g).exp().unsqueeze(-1)).transpose(-1, -2)  # [B,H,n,dk,C]

    S = (torch.zeros(B, H, dk, dv, dtype=torch.float32, device=query.device)
         if initial_state is None else initial_state.to(torch.float32))
    outs = []
    for i in range(n):
        vn = V[:, :, i] - Kc[:, :, i] @ S
        outs.append(Qg[:, :, i] @ S + QK[:, :, i] @ vn)
        S = S * Gl[:, :, i, None, None] + Kd[:, :, i] @ vn
    core = torch.stack(outs, dim=2).reshape(B, H, Lp, dv)[:, :, :L]
    last = S if output_final_state else None
    return core.transpose(1, 2).contiguous().to(initial_dtype), last
