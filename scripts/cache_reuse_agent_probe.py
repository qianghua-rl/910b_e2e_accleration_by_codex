"""Measure tool-turn cache reuse on a real agent conversation."""
import argparse
import json
import time
from pathlib import Path

import torch
import torch_npu
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

from agent_protocol import CONCISE_SYSTEM, TOOLS, execute, parse_call


parser = argparse.ArgumentParser()
parser.add_argument('--task-id', type=int, default=640)
parser.add_argument('--output', required=True)
args = parser.parse_args()
output_path = Path(args.output)
if output_path.exists():
    raise SystemExit(f'Output exists: {output_path}')

torch.set_num_threads(8)
torch.manual_seed(42)
model_path = 'model/qwen3.5-27b'
tasks = [json.loads(line) for line in Path('data/gsm8k_128.jsonl').read_text().splitlines()]
task = next((row for row in tasks if row['id'] == args.task_id), None)
if task is None:
    raise SystemExit(f'Unknown task id: {args.task_id}')

start = time.perf_counter()
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    model_path, dtype=torch.bfloat16, attn_implementation='sdpa',
    device_map={'': 'npu:0'}).eval()
tok = AutoTokenizer.from_pretrained(model_path)
torch.npu.synchronize()
load_seconds = time.perf_counter() - start

messages = [
    {'role': 'system', 'content': CONCISE_SYSTEM},
    {'role': 'user', 'content': task['question']},
]
template = dict(tools=TOOLS, add_generation_prompt=True, enable_thinking=False,
                tokenize=True, return_dict=True, return_tensors='pt')
first_inputs = tok.apply_chat_template(messages, **template).to('npu')

with torch.inference_mode():
    start = time.perf_counter()
    generated = model.generate(**first_inputs, max_new_tokens=512, do_sample=False,
                               use_cache=True, return_dict_in_generate=True)
    torch.npu.synchronize()
    generation_seconds = time.perf_counter() - start

sequences = generated.sequences
first_length = first_inputs['input_ids'].shape[1]
generated_ids = sequences[:, first_length:]
text = tok.decode(generated_ids[0], skip_special_tokens=True)
code = parse_call(text)
if code is None:
    raise SystemExit('Selected task did not produce a parseable tool call')
tool_output = execute(code)
messages.extend([
    {'role': 'assistant', 'content': text},
    {'role': 'tool', 'name': 'python', 'content': tool_output},
])
full_inputs = tok.apply_chat_template(messages, **template).to('npu')
cache = generated.past_key_values
cache_length = cache.get_seq_length()
cache_matches_full_prefix = torch.equal(
    sequences[:, :cache_length], full_inputs['input_ids'][:, :cache_length])
append_ids = full_inputs['input_ids'][:, cache_length:]

with torch.inference_mode():
    torch.npu.synchronize()
    start = time.perf_counter()
    full = model(**full_inputs, use_cache=True, logits_to_keep=1, return_dict=True)
    torch.npu.synchronize()
    full_prefill_seconds = time.perf_counter() - start

    torch.npu.synchronize()
    start = time.perf_counter()
    cached = model(input_ids=append_ids, past_key_values=cache, use_cache=True,
                   logits_to_keep=1, return_dict=True)
    torch.npu.synchronize()
    cached_append_seconds = time.perf_counter() - start

full_logits = full.logits[:, -1, :].float()
cached_logits = cached.logits[:, -1, :].float()
diff = (full_logits - cached_logits).abs()
row = dict(
    kind='agent_cache_reuse_probe', task_id=args.task_id,
    load_seconds=load_seconds, first_round_generation_seconds=generation_seconds,
    first_prompt_tokens=int(first_inputs['input_ids'].shape[1]),
    first_generated_tokens=int(generated_ids.shape[1]),
    full_second_prompt_tokens=int(full_inputs['input_ids'].shape[1]),
    cache_length=int(cache_length), append_tokens=int(append_ids.shape[1]),
    cache_matches_full_prefix=bool(cache_matches_full_prefix),
    full_prefill_seconds=full_prefill_seconds,
    cached_append_seconds=cached_append_seconds,
    prefill_speedup=full_prefill_seconds / cached_append_seconds,
    logits_max_abs_error=float(diff.max().cpu()),
    logits_mean_abs_error=float(diff.mean().cpu()),
    argmax_equal=bool(full_logits.argmax(-1).item() == cached_logits.argmax(-1).item()),
    peak_bytes=int(torch.npu.max_memory_allocated()),
)
output_path.write_text(json.dumps(row) + '\n')
print(json.dumps(row), flush=True)
