"""Probe the vLLM-Ascend GDN kernel through the HF Qwen3.5 model API."""
import json
import time
import argparse
from pathlib import Path

import torch
import torch_npu
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
import transformers.models.qwen3_5.modeling_qwen3_5 as qwen_mod
from vllm.model_executor.layers.fla.ops import fused_recurrent_gated_delta_rule as ascend_kernel

parser = argparse.ArgumentParser()
parser.add_argument('--kernel', action='store_true')
parser.add_argument('--batch-size', type=int, default=1)
args = parser.parse_args()

torch.set_num_threads(8)
torch.manual_seed(42)
model_path = 'model/qwen3.5-27b'

def kernel_adapter(query, key, value, g, beta, initial_state, output_final_state,
                   use_qk_l2norm_in_kernel=False):
    kernel_state = None if initial_state is None else initial_state.transpose(-1, -2).contiguous()
    out, final = ascend_kernel(query, key, value, g, beta, initial_state=kernel_state,
                               inplace_final_state=False,
                               use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel)
    if output_final_state and final is not None:
        final = final.transpose(-1, -2).contiguous()
    return out, final

if args.kernel:
    qwen_mod.fused_recurrent_gated_delta_rule = kernel_adapter
t0 = time.perf_counter()
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    model_path, dtype=torch.bfloat16, attn_implementation='sdpa',
    ).to('npu').eval()
tok = AutoTokenizer.from_pretrained(model_path)
question = json.loads(Path('data/gsm8k_128.jsonl').read_text().splitlines()[0])['question']
from agent_protocol import CONCISE_SYSTEM, TOOLS, parse_call, execute
messages = [{'role': 'system', 'content': CONCISE_SYSTEM}, {'role': 'user', 'content': question}]
prompt = tok.apply_chat_template(messages, tools=TOOLS, add_generation_prompt=True,
                                  enable_thinking=False, tokenize=False)
inputs = tok([prompt] * args.batch_size, return_tensors='pt', add_special_tokens=False,
             padding=True).to('npu')
with torch.inference_mode():
    start = time.perf_counter()
    generated = model.generate(**inputs, max_new_tokens=512, do_sample=False, use_cache=True)
    torch.npu.synchronize()
elapsed = time.perf_counter() - start
ids = generated[:, inputs['input_ids'].shape[1]:].tolist()
text = tok.decode(ids[0], skip_special_tokens=True)
code = parse_call(text)
tool_output = execute(code) if code is not None else None
row = {'kind': 'gdn_kernel_model_probe', 'kernel': args.kernel, 'batch_size': args.batch_size,
       'load_seconds': time.perf_counter() - t0,
       'generation_seconds': elapsed, 'tokens': len(ids[0]), 'tool_call': code is not None,
       'tool_output': tool_output, 'text': text, 'finite': bool(torch.isfinite(generated).all())}
Path('results/gdn_kernel_model_probe.json').write_text(json.dumps(row, ensure_ascii=False) + '\n')
print(json.dumps(row, ensure_ascii=False), flush=True)
