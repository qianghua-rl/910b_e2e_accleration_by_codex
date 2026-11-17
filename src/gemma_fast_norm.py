"""Fused RMSNorm for Gemma-4 on Ascend (opt-in via gemma_run.py --fast-norm).

Gemma4RMSNorm.forward is a Python chain (float cast, pow, mean, + eps as a Python float,
pow(-0.5), two muls, type_as): about 250 us per call on 910B3, and the model makes 210 calls
per decoded token (7 norms x 30 layers). torch_npu.npu_rms_norm computes the same function
in one kernel (about 49 us per call in microbench; output bit-identical on a random input).

Semantics matched to Gemma4RMSNorm, for norms with a weight (with_scale=True):
  y = x * rsqrt(mean(x^2) + eps) * weight
Norms with with_scale=False (no weight) are left unchanged.
Any leading shape is accepted as long as the last dim equals the weight length and the dtype
matches; otherwise the original forward is used, so the patch cannot silently change a shape
it does not understand.
"""
import types

import torch
import torch_npu  # noqa: F401
from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm


def _fused_forward(self, hidden_states):
    gamma = self._fast_gamma
    if hidden_states.dtype != gamma.dtype or hidden_states.shape[-1] != gamma.shape[0]:
        return self._orig_forward(hidden_states)
    return torch_npu.npu_rms_norm(hidden_states, gamma, self.eps)[0]


def _fused_forward_noscale(self, hidden_states):
    # with_scale=False: y = x * rsqrt(mean(x^2) + eps) == npu_rms_norm with gamma = 1.
    # gamma is cached per last-dim and dtype (ones are exact, so the multiply is a no-op).
    key = (hidden_states.shape[-1], hidden_states.dtype)
    cache = self._ones_cache
    if key not in cache:
        cache[key] = torch.ones(key[0], dtype=key[1], device=hidden_states.device)
    return torch_npu.npu_rms_norm(hidden_states, cache[key], self.eps)[0]


def patch_gemma_norms(model, noscale=False):
    """Replace Gemma4RMSNorm.forward in `model`. Returns the number patched.
    noscale=True also patches the norms without a weight (v_norm, router norm)."""
    n = 0
    for mod in model.modules():
        if not isinstance(mod, Gemma4RMSNorm):
            continue
        if mod.with_scale:
            mod._fast_gamma = mod.weight.detach().clone()
            mod._orig_forward = mod.forward
            mod.forward = types.MethodType(_fused_forward, mod)
            n += 1
        elif noscale:
            mod._ones_cache = {}
            mod._orig_forward = mod.forward
            mod.forward = types.MethodType(_fused_forward_noscale, mod)
            n += 1
    return n
