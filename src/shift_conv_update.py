"""Decode-step depthwise causal conv (kernel K=4) as shifted multiply-add, no Conv2D.

Same contract as transformers' torch_causal_conv1d_update:
  hidden_states: [B, C, T] (T == 1 in decode), conv_state: [B, C, K-1] (updated in place),
  weight: [C, K], bias: [C] or None. Returns silu(conv output) as hidden_states.dtype.
Only valid for T == 1; other shapes fall back to the reference function.
"""
import torch
import torch.nn.functional as F


def shift_causal_conv1d_update(hidden_states, conv_state, weight, bias=None, activation=None):
    _, _, seq_len = hidden_states.shape
    if seq_len != 1:
        from transformers.models.qwen3_5.modeling_qwen3_5 import torch_causal_conv1d_update
        return torch_causal_conv1d_update(hidden_states, conv_state, weight, bias, activation)
    k = weight.shape[-1]
    state_len = conv_state.shape[-1]          # HF cache stores K entries, not K-1
    x_new = hidden_states.to(weight.dtype)
    window = torch.cat([conv_state.to(weight.dtype), x_new], dim=-1)   # [B, C, state_len+1]
    conv_state.copy_(window[:, :, -state_len:])
    taps = window[:, :, -k:]                  # last K inputs produce the single output
    out = taps[:, :, 0:1] * weight[:, 0:1]
    for j in range(1, k):
        out = out + taps[:, :, j:j + 1] * weight[:, j:j + 1]
    if bias is not None:
        out = out + bias.view(1, -1, 1)
    out = F.silu(out)
    return out.to(hidden_states.dtype)
