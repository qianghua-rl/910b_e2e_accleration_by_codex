"""Validate row compaction for Qwen3.5 hybrid attention/recurrent caches."""
import json
import time
from pathlib import Path

import torch
import torch_npu
from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM


cfg = json.loads(Path('model/qwen3.5-27b/config.json').read_text())['text_config']
cfg.update(num_hidden_layers=4, layer_types=cfg['layer_types'][:4], vocab_size=1024)
config = Qwen3_5TextConfig(**cfg)
config._attn_implementation = 'sdpa'
torch.set_num_threads(8)
torch.manual_seed(42)
model = Qwen3_5ForCausalLM(config).to(dtype=torch.bfloat16, device='npu').eval()
prefix = torch.randint(0, 1024, (2, 48), device='npu')
next_tokens = torch.randint(0, 1024, (2, 1), device='npu')


def select_hybrid_rows(cache, indices):
    for layer in cache.layers:
        if getattr(layer, 'is_initialized', False):
            layer.keys = layer.keys.index_select(0, indices.to(layer.keys.device))
            layer.values = layer.values.index_select(0, indices.to(layer.values.device))
        if getattr(layer, 'is_conv_states_initialized', False):
            layer.conv_states = layer.conv_states.index_select(
                0, indices.to(layer.conv_states.device))
        if getattr(layer, 'is_recurrent_states_initialized', False):
            layer.recurrent_states = layer.recurrent_states.index_select(
                0, indices.to(layer.recurrent_states.device))


with torch.inference_mode():
    prefill = model(prefix, use_cache=True, logits_to_keep=1, return_dict=True)
    cache = prefill.past_key_values
    selected = torch.tensor([1], device='npu')
    expected = []
    for layer in cache.layers:
        for name, flag in [('keys', 'is_initialized'), ('values', 'is_initialized'),
                           ('conv_states', 'is_conv_states_initialized'),
                           ('recurrent_states', 'is_recurrent_states_initialized')]:
            value = getattr(layer, name, None)
            if value is not None and getattr(layer, flag, False):
                expected.append((layer, name, value[1:2].clone()))
    start = time.perf_counter()
    select_hybrid_rows(cache, selected)
    torch.npu.synchronize()
    compact_seconds = time.perf_counter() - start
    exact = all(torch.equal(getattr(layer, name), value)
                for layer, name, value in expected)
    decoded = model(input_ids=next_tokens[1:2], past_key_values=cache,
                    use_cache=True, logits_to_keep=1, return_dict=True)
    torch.npu.synchronize()
    finite = bool(torch.isfinite(decoded.logits).all())
    row = dict(
        kind='cache_compaction_probe', layers=len(cache.layers),
        selected_rows=[1], selected_state_tensors=len(expected),
        selected_state_exact=exact, compact_seconds=compact_seconds,
        output_finite=finite, output_shape=list(decoded.logits.shape),
        peak_bytes=int(torch.npu.max_memory_allocated()),
    )
Path('results/cache_compaction_probe.json').write_text(json.dumps(row) + '\n')
print(json.dumps(row), flush=True)
