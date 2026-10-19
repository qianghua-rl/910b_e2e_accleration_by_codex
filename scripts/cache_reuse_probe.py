"""Probe cross-call Qwen3.5 cache reuse for an agent tool turn.

Compares logits for a complete prompt against a prefilled prompt followed by
only the appended assistant/tool tokens. This is a capability/measurement
probe; it does not change the benchmark runner.
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch_npu
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration


parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True)
args = parser.parse_args()

torch.set_num_threads(8)
torch.manual_seed(42)
model_path = 'model/qwen3.5-27b'
tok = AutoTokenizer.from_pretrained(model_path)
t0 = time.perf_counter()
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    model_path, dtype=torch.bfloat16, attn_implementation='sdpa',
    device_map={'': 'npu:0'}).eval()
torch.npu.synchronize()
load_seconds = time.perf_counter() - t0

system = 'Use the python tool for arithmetic, then answer with a boxed number.'
question = 'A shop has 17 boxes with 24 items each. How many items are there?'
assistant = '<|im_start|>assistant\n<tool_call>python\nprint(17*24)\n</tool_call>\n'
tool = '<|im_start|>tool\n408<|im_end|>\n'
base = tok.apply_chat_template([
    {'role': 'system', 'content': system}, {'role': 'user', 'content': question}],
    tokenize=True, add_generation_prompt=True, enable_thinking=False,
    return_tensors='pt', return_dict=True)
base_ids = base['input_ids'].to('npu')
append_ids = tok(assistant + tool, add_special_tokens=False,
                  return_tensors='pt').input_ids.to('npu')
full_ids = torch.cat([base_ids, append_ids], dim=1)

def run_prefill(ids, past=None):
    start = time.perf_counter()
    with torch.inference_mode():
        out = model(input_ids=ids, past_key_values=past, use_cache=True,
                    logits_to_keep=1, return_dict=True)
    torch.npu.synchronize()
    return out, time.perf_counter() - start

with torch.inference_mode():
    full, full_seconds = run_prefill(full_ids)
    first, first_seconds = run_prefill(base_ids)
    cached, cached_seconds = run_prefill(append_ids, first.past_key_values)

full_logits = full.logits[:, -1, :].float()
cached_logits = cached.logits[:, -1, :].float()
diff = (full_logits - cached_logits).abs()
row = {
    'kind': 'cache_reuse_probe', 'load_seconds': load_seconds,
    'base_tokens': int(base_ids.shape[1]), 'append_tokens': int(append_ids.shape[1]),
    'full_tokens': int(full_ids.shape[1]), 'full_prefill_seconds': full_seconds,
    'base_prefill_seconds': first_seconds, 'append_cached_seconds': cached_seconds,
    'prefill_speedup_vs_full': full_seconds / (first_seconds + cached_seconds),
    'logits_max_abs_error': float(diff.max().cpu()),
    'logits_mean_abs_error': float(diff.mean().cpu()),
    'argmax_equal': bool(full_logits.argmax(-1).item() == cached_logits.argmax(-1).item()),
    'peak_bytes': int(torch.npu.max_memory_allocated()),
}
Path(args.output).write_text(json.dumps(row) + '\n')
print(json.dumps(row), flush=True)
