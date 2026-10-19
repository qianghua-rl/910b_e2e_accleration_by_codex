"""Compare HF generate with an explicit cached greedy decode loop."""
import argparse
import json
import time
from pathlib import Path

import torch
import torch_npu
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
from agent_protocol import CONCISE_SYSTEM, TOOLS


parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True)
parser.add_argument('--max-new-tokens', type=int, default=64)
args = parser.parse_args()
output_path = Path(args.output)
if output_path.exists():
    raise SystemExit(f'Output already exists: {output_path}')

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

question = json.loads(Path('data/gsm8k_128.jsonl').read_text().splitlines()[0])['question']
inputs = tok.apply_chat_template(
    [{'role': 'system', 'content': CONCISE_SYSTEM}, {'role': 'user', 'content': question}],
    tools=TOOLS, tokenize=True, add_generation_prompt=True, enable_thinking=False,
    return_dict=True, return_tensors='pt').to('npu')
eos_ids = model.generation_config.eos_token_id
eos_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])

def sync():
    torch.npu.synchronize()

with torch.inference_mode():
    torch.npu.reset_peak_memory_stats()
    sync()
    start = time.perf_counter()
    reference = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                               do_sample=False, use_cache=True)
    sync()
    generate_seconds = time.perf_counter() - start
    reference_ids = reference[:, inputs['input_ids'].shape[1]:].squeeze(0).tolist()
    generate_peak = torch.npu.max_memory_allocated()
    del reference

    torch.npu.reset_peak_memory_stats()
    sync()
    start = time.perf_counter()
    out = model(**inputs, use_cache=True, logits_to_keep=1, return_dict=True)
    sync()
    prefill_seconds = time.perf_counter() - start
    cache = out.past_key_values
    next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    del out
    manual_ids = []
    step_seconds = []
    for _ in range(args.max_new_tokens):
        token = int(next_id.item())
        manual_ids.append(token)
        if token in eos_ids:
            break
        sync()
        step_start = time.perf_counter()
        out = model(input_ids=next_id, past_key_values=cache, use_cache=True,
                    logits_to_keep=1, return_dict=True)
        sync()
        step_seconds.append(time.perf_counter() - step_start)
        cache = out.past_key_values
        next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        del out
    manual_seconds = time.perf_counter() - start
    manual_peak = torch.npu.max_memory_allocated()

row = {
    'kind': 'decode_loop_probe', 'load_seconds': load_seconds,
    'prompt_tokens': int(inputs['input_ids'].shape[1]),
    'max_new_tokens': args.max_new_tokens,
    'generate_tokens': len(reference_ids), 'manual_tokens': len(manual_ids),
    'tokens_equal': reference_ids == manual_ids,
    'first_mismatch': next((i for i, pair in enumerate(zip(reference_ids, manual_ids))
                            if pair[0] != pair[1]), None),
    'generate_seconds': generate_seconds,
    'manual_prefill_seconds': prefill_seconds,
    'manual_decode_seconds': sum(step_seconds),
    'manual_total_seconds': manual_seconds,
    'manual_token_median_seconds': sorted(step_seconds)[len(step_seconds)//2] if step_seconds else None,
    'generate_peak_bytes': int(generate_peak), 'manual_peak_bytes': int(manual_peak),
    'generate_text': tok.decode(reference_ids, skip_special_tokens=True),
    'manual_text': tok.decode(manual_ids, skip_special_tokens=True),
}
output_path.write_text(json.dumps(row, ensure_ascii=False) + '\n')
print(json.dumps(row, ensure_ascii=False), flush=True)
