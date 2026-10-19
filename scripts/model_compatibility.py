"""Random four-layer compatibility probe; never a quality/performance baseline."""
import json
import time
import traceback
import torch
import torch_npu
from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM

cfg = json.load(open('model/qwen3.5-27b/config.json'))['text_config']
cfg.update(num_hidden_layers=4, layer_types=cfg['layer_types'][:4], vocab_size=1024)
config = Qwen3_5TextConfig(**cfg)
config._attn_implementation = 'eager'
torch.set_num_threads(8)
torch.manual_seed(42)
model = Qwen3_5ForCausalLM(config).to(dtype=torch.bfloat16, device='npu').eval()
with torch.inference_mode():
    for batch, length in [(1, 16), (1, 128), (4, 128)]:
        row = dict(batch=batch, length=length, random_weights=True)
        try:
            ids = torch.randint(0, 1024, (batch, length), device='npu')
            torch.npu.synchronize()
            t = time.perf_counter()
            out = model(ids, use_cache=True, logits_to_keep=1)
            torch.npu.synchronize()
            row['prefill_seconds'] = time.perf_counter()-t
            row['finite'] = bool(out.logits.isfinite().all().item())
            t = time.perf_counter()
            nxt = model(out.logits.argmax(-1), past_key_values=out.past_key_values, use_cache=True, logits_to_keep=1)
            torch.npu.synchronize()
            row['decode_seconds'] = time.perf_counter()-t
            row['decode_finite'] = bool(nxt.logits.isfinite().all().item())
        except Exception:
            row['error'] = traceback.format_exc()
        print(json.dumps(row), flush=True)
        with open('results/model_compatibility.jsonl', 'a') as f:
            f.write(json.dumps(row)+'\n')
