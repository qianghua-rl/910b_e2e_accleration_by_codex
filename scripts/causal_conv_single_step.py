"""Single-token depthwise convolution; retain the original cache layout."""
import torch
import torch.nn.functional as F
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_causal_conv1d_update


def causal_conv_single_step(hidden_states, conv_state, weight, bias=None, activation=None):
    if hidden_states.shape[-1] != 1:
        return torch_causal_conv1d_update(hidden_states, conv_state, weight, bias, activation)
    # Retain the original convolution's accumulation and rounding. Only omit
    # outputs that HF immediately slices away during single-token decode.
    width = weight.shape[-1]
    history = torch.cat((conv_state, hidden_states), dim=-1).to(weight.dtype)
    window = history[:, :, -width:].contiguous()
    out = F.conv1d(window, weight.unsqueeze(1), bias, groups=weight.shape[0])
    conv_state.copy_(history[:, :, -conv_state.shape[-1]:])
    return F.silu(out.to(weight.dtype)).to(hidden_states.dtype)
