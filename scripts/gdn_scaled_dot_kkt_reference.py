"""Standalone Ascend Triton KKT reference derived from FLA's Ascend backend.

    The kernel follows FLA's MIT-licensed
``fla/ops/common/backends/triton_ascend/chunk_scaled_dot_kkt.py`` algorithm,
    with its launcher made self-contained for this probe. It has no vLLM or FLA
    runtime imports. BK=64 matches FLA's UB-limited tile cap for this kernel.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _safe_exp(x):
    # The Qwen/vLLM GDN gate uses the natural exponential. Causal gate
    # differences should be nonpositive, so clamp positive values as in FLA's
    # vendored Ascend-safe reference to avoid overflow in masked lanes.
    return tl.exp(tl.where(x <= 0, x.to(tl.float32), float("-inf")))


@triton.jit(do_not_specialize=["T"])
def _chunk_scaled_dot_kkt_kernel(
    k,
    g,
    beta,
    A,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
):
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b, i_h = i_bh // HV, i_bh % HV
    if i_t * BT >= T:
        return

    k_base = k + (i_b * T * H + i_h // (HV // H)) * K
    A_base = A + (i_b * T * HV + i_h) * BT
    beta_base = beta + i_b * T * HV + i_h
    g_base = g + i_b * T * HV + i_h

    o_i = tl.arange(0, BC)
    n_sub = BT // BC
    for s in range(n_sub):
        i_tc_s = i_t * BT + s * BC
        m_s = (i_tc_s + o_i) < T
        p_bs = tl.make_block_ptr(beta_base, (T,), (HV,), (i_tc_s,), (BC,), (0,))
        b_bs = tl.load(p_bs, boundary_check=(0,))
        p_gs = tl.make_block_ptr(g_base, (T,), (HV,), (i_tc_s,), (BC,), (0,))
        b_gs = tl.load(p_gs, boundary_check=(0,))

        for c in range(s + 1):
            i_tc_c = i_t * BT + c * BC
            m_c = (i_tc_c + o_i) < T
            b_A = tl.zeros((BC, BC), dtype=tl.float32)
            for i_k in range(tl.cdiv(K, BK)):
                p_ks = tl.make_block_ptr(
                    k_base, (T, K), (H * K, 1), (i_tc_s, i_k * BK),
                    (BC, BK), (1, 0),
                )
                p_kc = tl.make_block_ptr(
                    k_base, (T, K), (H * K, 1), (i_tc_c, i_k * BK),
                    (BC, BK), (1, 0),
                )
                b_ks = tl.load(p_ks, boundary_check=(0, 1))
                b_kc = tl.load(p_kc, boundary_check=(0, 1))
                b_A += tl.dot(b_ks, tl.trans(b_kc), allow_tf32=False)

            p_gc = tl.make_block_ptr(g_base, (T,), (HV,), (i_tc_c,), (BC,), (0,))
            b_gc = tl.load(p_gc, boundary_check=(0,))
            b_A *= _safe_exp(b_gs[:, None] - b_gc[None, :])
            b_A *= b_bs[:, None]

            if s == c:
                m_blk = (o_i[:, None] > o_i[None, :]) & (m_s[:, None] & m_s)
            else:
                m_blk = m_s[:, None] & m_c
            b_A = tl.where(m_blk, b_A, 0)

            p_A = tl.make_block_ptr(
                A_base, (T, BT), (HV * BT, 1), (i_tc_s, c * BC),
                (BC, BC), (1, 0),
            )
            tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))


def chunk_scaled_dot_kkt_fwd(k, beta, g_cumsum, chunk_size=64, output_dtype=torch.float32):
    batch, length, heads, key_dim = k.shape
    value_heads = beta.shape[2]
    block_c = 16
    block_k = 64
    output = torch.empty(
        batch, length, value_heads, chunk_size,
        device=k.device, dtype=output_dtype,
    )
    grid = (triton.cdiv(length, chunk_size), batch * value_heads)
    _chunk_scaled_dot_kkt_kernel[grid](
        k=k,
        g=g_cumsum,
        beta=beta,
        A=output,
        T=length,
        H=heads,
        HV=value_heads,
        K=key_dim,
        BT=chunk_size,
        BC=block_c,
        BK=block_k,
        num_warps=4,
    )
    return output
