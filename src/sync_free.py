"""Sync-free variants of the decode-path ops (same math as the originals).

Host syncs on NPU come from Python float scalars in tensor ops (e.g. `x + 1e-6`, `q * 0.088`,
`(1.0 + w)` each call). Using device tensors for those constants avoids the sync.
"""
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401
from transformers.models.qwen3_5 import modeling_qwen3_5 as M


_CONST = {}


def _const(value, device, dtype):
    # Created once per (value, device, dtype): torch.tensor(..., device=) copies host->device each call.
    key = (value, str(device), dtype)
    t = _CONST.get(key)
    if t is None:
        t = torch.tensor(value, device=device, dtype=dtype)
        _CONST[key] = t
    return t


def l2norm_nosync(x, dim=-1, eps=1e-6):
    eps_t = _const(eps, x.device, x.dtype)
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps_t)
    return x * inv_norm


def recurrent_nosync(query, key, value, g, beta, initial_state, output_final_state,
                     use_qk_l2norm_in_kernel=False):
    if query.shape[1] != 1:
        return M.torch_recurrent_gated_delta_rule(
            query, key, value, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel)
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm_nosync(query, dim=-1, eps=1e-6)
        key = l2norm_nosync(key, dim=-1, eps=1e-6)
    q, k, v, beta, g = [x[:, 0].contiguous().to(torch.float32) for x in (query, key, value, beta, g)]
    scale_t = _const(1 / (query.shape[-1] ** 0.5), q.device, torch.float32)
    q = q * scale_t
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


def fused_rmsnorm_nosync(self, x):
    # (1 + weight) is cached once per module instead of recomputed as a Python-scalar op each call.
    return torch_npu.npu_rms_norm(x, self._w1.to(x.dtype), self.eps)[0]


def apply_sync_free(model):
    M.l2norm = l2norm_nosync  # GDN forward looks this up as a module global at call time
    n_norm = n_gdn = 0
    for m in model.modules():
        if isinstance(m, M.Qwen3_5RMSNorm):
            m._w1 = (1.0 + m.weight).contiguous()
            m.forward = fused_rmsnorm_nosync.__get__(m, M.Qwen3_5RMSNorm)
            n_norm += 1
        elif isinstance(m, M.Qwen3_5GatedDeltaNet):
            m.recurrent_gated_delta_rule = recurrent_nosync
            n_gdn += 1
    return {"rmsnorm": n_norm, "gdn": n_gdn}
