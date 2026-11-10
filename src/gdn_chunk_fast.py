"""Self-written chunked gated-delta-rule prefill, drop-in for transformers'
torch_chunk_gated_delta_rule.

Only change vs the fallback: the intra-chunk triangular recurrence (a 63-step
Python loop with clones) is replaced by a doubling inverse.

For strictly lower-triangular X (chunk x chunk, nilpotent: X^chunk = 0),
    T = (I - X)^{-1} = sum_k X^k = prod_j (I + X^{2^j}),
computed as S_{j+1} = S_j (I + P_j), P_{j+1} = P_j @ P_j, with S_0 = I, P_0 = X.
After log2(chunk) steps S = T. The fallback's loop computes exactly this T
(attn + I), so results match up to fp32 rounding.
"""
import torch
import torch.nn.functional as F


def _l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def _small_unit_lower_inv(X):
    """(I - X)^{-1} for small strictly lower X, via the fallback's row recurrence
    R_i = X_i + sum_j X_ij R_j (T = I + R). Batched over leading dims."""
    n = X.shape[-1]
    R = X.clone()
    for i in range(1, n):
        row = R[..., i, :i].clone()
        sub = R[..., :i, :i].clone()
        R[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    return R + torch.eye(n, dtype=X.dtype, device=X.device)


def _unit_lower_inv(X, base=16):
    """(I - X)^{-1} for strictly lower X (..., n, n), n a multiple of base.

    All diagonal blocks are inverted together in one batched small recurrence
    (base-step loop instead of n-step), then merged pairwise with block products:
        (I - X)^{-1} over [lo,hi) = [[A, 0], [B X21 A, B]] with A, B the halves' inverses.
    No matrix powers are formed (powers overflow on real inputs even though X is nilpotent).
    """
    n = X.shape[-1]
    nb = n // base
    if nb == 1:
        return _small_unit_lower_inv(X)
    diag = torch.stack(
        [X[..., i * base:(i + 1) * base, i * base:(i + 1) * base] for i in range(nb)], dim=-3)
    Tdiag = _small_unit_lower_inv(diag)  # (..., nb, base, base)
    blocks = [Tdiag[..., i, :, :] for i in range(nb)]

    def merge(lo, hi):
        if hi - lo == 1:
            return blocks[lo]
        mid = (lo + hi) // 2
        A = merge(lo, mid)
        B = merge(mid, hi)
        X21 = X[..., mid * base:hi * base, lo * base:mid * base]
        T21 = B @ X21 @ A
        top = torch.cat([A, torch.zeros_like(T21.transpose(-1, -2))], dim=-1)
        bot = torch.cat([T21, B], dim=-1)
        return torch.cat([top, bot], dim=-2)

    return merge(0, nb)


def fast_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    strict_upper_or_diag = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    # chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    X = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(strict_upper_or_diag, 0)

    # T = (I - X)^{-1}, X strictly lower. Blocked inverse (see _unit_lower_inv).
    attn = _unit_lower_inv(X)

    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    n_chunks = total_sequence_length // chunk_size
    outs = []
    for i in range(n_chunks):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn_i = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        outs.append(attn_inter + attn_i @ v_new)
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    core_attn_out = torch.stack(outs, dim=2)
    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state
