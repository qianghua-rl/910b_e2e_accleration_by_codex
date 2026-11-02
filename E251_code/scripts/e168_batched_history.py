"""Isolated HF ragged append: one model forward, explicit batched state steps."""
import copy
import inspect
import textwrap

import torch
import torch.nn.functional as F
from transformers.models.qwen3_5 import modeling_qwen3_5 as hf
from recurrent_single_step import recurrent_single_step


FIELDS = ('keys', 'values', 'conv_states', 'recurrent_states')


def clone_cache(cache, device='cpu'):
    result = copy.copy(cache)
    result.layers = []
    for layer in cache.layers:
        target = copy.copy(layer)
        for name in FIELDS:
            value = getattr(layer, name, None)
            if isinstance(value, torch.Tensor):
                setattr(target, name, value.detach().to(device).clone())
        if hasattr(target, 'device'):
            target.device = torch.device(device)
        result.layers.append(target)
    return result


def pack(snapshots, suffixes, device='npu:0', pad_token=0):
    if not snapshots or len(snapshots) != len(suffixes) or any(not s for s in suffixes):
        raise ValueError('nonempty aligned history/suffix lists required')
    widths = [len(s['mask']) for s in snapshots]
    p, t = max(widths), max(map(len, suffixes))
    cache = copy.copy(snapshots[0]['cache'])
    cache.layers = []
    for i, original in enumerate(snapshots[0]['cache'].layers):
        layer = copy.copy(original)
        for name in FIELDS:
            if not isinstance(getattr(original, name, None), torch.Tensor):
                continue
            parts = []
            for snap, width in zip(snapshots, widths):
                value = getattr(snap['cache'].layers[i], name)
                if name in ('keys', 'values'):
                    value = F.pad(value, (0, 0, p-width, 0))
                parts.append(value)
            setattr(layer, name, torch.cat(parts, dim=0).to(device))
        if hasattr(layer, 'device'): layer.device = torch.device(device)
        if hasattr(layer, 'max_batch_size'): layer.max_batch_size = len(snapshots)
        cache.layers.append(layer)
    ids = torch.full((len(snapshots), t), pad_token, dtype=torch.long, device=device)
    mask = torch.zeros((len(snapshots), p+t), dtype=torch.long, device=device)
    positions = torch.zeros_like(ids)
    for row, (snap, suffix, width) in enumerate(zip(snapshots, suffixes, widths)):
        mask[row, p-width:p] = mask.new_tensor(snap['mask'])
        mask[row, p:p+len(suffix)] = 1
        ids[row, :len(suffix)] = ids.new_tensor(suffix)
        positions[row] = torch.arange(t, device=device) + len(snap['consumed'])
    return cache, ids, mask, positions, p


def unpack(cache, snapshots, suffixes, p):
    results = []
    for row, (snap, suffix) in enumerate(zip(snapshots, suffixes)):
        result = copy.copy(cache)
        result.layers = []
        width, length = len(snap['mask']), len(suffix)
        for original in cache.layers:
            layer = copy.copy(original)
            for name in FIELDS:
                value = getattr(original, name, None)
                if not isinstance(value, torch.Tensor): continue
                value = value[row:row+1]
                if name in ('keys', 'values'):
                    value = torch.cat((value[..., p-width:p, :], value[..., p:p+length, :]), dim=-2)
                setattr(layer, name, value.detach().cpu().clone())
            if hasattr(layer, 'device'): layer.device = torch.device('cpu')
            if hasattr(layer, 'max_batch_size'): layer.max_batch_size = 1
            result.layers.append(layer)
        results.append(result)
    return results


class BatchedHistory:
    def __init__(self):
        self.lengths = None
        self.events = []
        self.capture_indices = set()
        self.captures = {}

    def conv(self, x, state, weight, bias=None, activation=None):
        index = len(self.events)//2
        if index in self.capture_indices:
            self.captures[index] = {'conv_inputs': [v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v
                for v in (x,state,weight,bias,activation)]}
        self.events.append({'op': 'conv', 'shape': list(x.shape), 'stride': list(x.stride()),
                            'dtype': str(x.dtype), 'state_dtype': str(state.dtype)})
        outputs = []
        for t in range(x.shape[-1]):
            active = (self.lengths > t).view(-1, 1, 1)
            working = state.clone()
            out = hf.torch_causal_conv1d_update(x[..., t:t+1], working, weight, bias, activation)
            state.copy_(torch.where(active, working, state))
            outputs.append(torch.where(active, out, 0))
        result = torch.cat(outputs, dim=-1)
        if index in self.capture_indices:
            self.captures[index].update(conv_output=result.detach().cpu(),conv_final=state.detach().cpu().clone())
        return result

    def recurrent(self, query, key, value, g, beta, initial_state,
                  output_final_state, use_qk_l2norm_in_kernel=False):
        index = len(self.events)//2
        if index in self.capture_indices:
            self.captures[index]['recurrent_inputs'] = [v.detach().cpu().clone()
                for v in (query,key,value,g,beta,initial_state)]
        self.events.append({'op': 'recurrent', 'shape': list(query.shape),
                            'stride': list(query.stride()), 'dtype': str(query.dtype),
                            'state_dtype': str(initial_state.dtype)})
        state = initial_state
        outputs = []
        for t in range(query.shape[1]):
            out, next_state = recurrent_single_step(
                query[:, t:t+1], key[:, t:t+1], value[:, t:t+1],
                g[:, t:t+1], beta[:, t:t+1], state, True, use_qk_l2norm_in_kernel)
            active = self.lengths > t
            # HF writes final state into its native cache dtype after every token.
            state = torch.where(active[:, None, None, None], next_state.to(initial_state.dtype), state)
            outputs.append(torch.where(active[:, None, None, None], out, 0))
        result = torch.cat(outputs, dim=1)
        if index in self.capture_indices:
            self.captures[index].update(recurrent_output=result.detach().cpu(),recurrent_final=state.detach().cpu().clone())
        return result, state if output_final_state else None

    def install(self, model):
        source = textwrap.dedent(inspect.getsource(hf.Qwen3_5GatedDeltaNet.forward))
        old = 'cache_params is not None and cache_params.has_previous_state(self.layer_idx) and seq_len == 1'
        assert source.count(old) == 1
        source = source.replace(old, 'cache_params is not None and cache_params.has_previous_state(self.layer_idx)')
        namespace = dict(vars(hf))
        exec(compile(source, '<e168_batched_history>', 'exec'), namespace)
        self.originals = []
        for module in model.modules():
            if isinstance(module, hf.Qwen3_5GatedDeltaNet):
                self.originals.append((module, module.forward, module.causal_conv1d_update,
                                       module.recurrent_gated_delta_rule))
                module.forward = namespace['forward'].__get__(module, type(module))
                module.causal_conv1d_update = self.conv
                module.recurrent_gated_delta_rule = self.recurrent
        return source

    def restore(self):
        for module, forward, conv, recurrent in self.originals:
            module.forward, module.causal_conv1d_update, module.recurrent_gated_delta_rule = forward, conv, recurrent
