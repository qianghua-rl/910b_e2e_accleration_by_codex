"""Validate Qwen3.5 Cache reuse with the real layer/state implementation.

Uses four real Qwen3.5 layers and random weights so it fits when the full
checkpoint cannot be loaded. It is an API/numerical probe, not an e2e score.
"""
import json
import time
from pathlib import Path
import torch
import torch_npu
from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM

cfg = json.load(open('model/qwen3.5-27b/config.json'))['text_config']
cfg.update(num_hidden_layers=4, layer_types=cfg['layer_types'][:4], vocab_size=1024)
config = Qwen3_5TextConfig(**cfg)
config._attn_implementation = 'sdpa'
torch.set_num_threads(8)
torch.manual_seed(42)
model = Qwen3_5ForCausalLM(config).to(dtype=torch.bfloat16, device='npu').eval()
prefix = torch.randint(0, 1024, (1, 32), device='npu')
append = torch.randint(0, 1024, (1, 11), device='npu')
full_ids = torch.cat([prefix, append], dim=1)

def call(ids, past=None):
    t = time.perf_counter()
    with torch.inference_mode():
        out = model(ids, past_key_values=past, use_cache=True, logits_to_keep=1)
    torch.npu.synchronize()
    return out, time.perf_counter() - t

full, full_s = call(full_ids)
base, base_s = call(prefix)
cached, append_s = call(append, base.past_key_values)
a = full.logits[:, -1, :].float()
b = cached.logits[:, -1, :].float()
err = (a - b).abs()
row = dict(kind='cache_api_tiny_probe', prefix_tokens=32, append_tokens=11,
           full_prefill_seconds=full_s, prefix_prefill_seconds=base_s,
           cached_append_seconds=append_s,
           speedup_vs_full=full_s/(base_s+append_s),
           logits_max_abs_error=float(err.max().cpu()),
           logits_mean_abs_error=float(err.mean().cpu()),
           argmax_equal=bool(a.argmax(-1).item() == b.argmax(-1).item()),
           finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
           peak_bytes=int(torch.npu.max_memory_allocated()))
Path('results/cache_api_tiny_probe.json').write_text(json.dumps(row)+'\n')
print(json.dumps(row), flush=True)
