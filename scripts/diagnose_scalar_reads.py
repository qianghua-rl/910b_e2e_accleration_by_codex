"""Compare E100 first-batch logits and greedy tokens with a T=1 conv adapter."""
import argparse
import json
import time
from pathlib import Path

import torch
import torch_npu
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

from agent_protocol import CONCISE_SYSTEM, TOOLS


parser = argparse.ArgumentParser()
parser.add_argument('--run', required=True, help='Completed E056-style JSONL run')
parser.add_argument('--trace-dir', help='Unused compatibility argument')
parser.add_argument('--max-new-tokens', type=int, default=32)
parser.add_argument('--single-step-recurrent', action='store_true')
parser.add_argument('--fused-rmsnorm', action='store_true')
parser.add_argument('--fused-gated-rmsnorm', action='store_true')
parser.add_argument('--decode-only', action='store_true', help='Skip prefill and first decode; profile next four decode steps')
args = parser.parse_args()

torch.set_num_threads(8)
torch.manual_seed(42)
model_path = 'model/qwen3.5-27b'
data = {row['id']: row for row in
        (json.loads(line) for line in Path('data/gsm8k_128.jsonl').read_text().splitlines())}
records = [json.loads(line) for line in Path(args.run).read_text().splitlines()]
tasks = [row for row in records if row.get('kind') == 'task'][:16]
if len(tasks) != 16:
    raise SystemExit('Expected at least 16 task records in the reference run')

tok = AutoTokenizer.from_pretrained(model_path, padding_side='left')
prompts = []
for row in tasks:
    task = data[row['id']]
    messages = [
        {'role': 'system', 'content': CONCISE_SYSTEM},
        {'role': 'user', 'content': task['question']},
    ]
    prompts.append(tok.apply_chat_template(
        messages, tools=TOOLS, add_generation_prompt=True,
        enable_thinking=False, tokenize=False))
inputs = tok(prompts, add_special_tokens=False, padding=True,
             return_tensors='pt').to('npu')

model = Qwen3_5ForConditionalGeneration.from_pretrained(
    model_path, dtype=torch.bfloat16, attn_implementation='sdpa',
    device_map={'': 'npu:0'}).eval()
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5GatedDeltaNet, Qwen3_5RMSNorm, Qwen3_5RMSNormGated,
)
if args.single_step_recurrent:
    from recurrent_single_step import recurrent_single_step
    for module in model.modules():
        if isinstance(module, Qwen3_5GatedDeltaNet):
            module.recurrent_gated_delta_rule = recurrent_single_step
if args.fused_rmsnorm:
    def fused_rmsnorm(self, x):
        return torch_npu.npu_rms_norm(x, (1.0 + self.weight).to(x.dtype), self.eps)[0]
    for module in model.modules():
        if isinstance(module, Qwen3_5RMSNorm):
            module.forward = fused_rmsnorm.__get__(module, Qwen3_5RMSNorm)
if args.fused_gated_rmsnorm:
    def fused_gated_norm(self, hidden_states, gate=None):
        normalized = torch_npu.npu_rms_norm(hidden_states, self.weight,
                                             self.variance_epsilon)[0]
        return torch_npu.npu_swiglu(torch.cat((gate, normalized), dim=-1))
    for module in model.modules():
        if isinstance(module, Qwen3_5RMSNormGated):
            module.forward = fused_gated_norm.__get__(module, Qwen3_5RMSNormGated)
for module in model.modules():
    if isinstance(module, Qwen3_5RMSNorm):
        def fused_rmsnorm(self, x):
            return torch_npu.npu_rms_norm(x, (1.0 + self.weight).to(x.dtype), self.eps)[0]
        module.forward = fused_rmsnorm.__get__(module, Qwen3_5RMSNorm)
torch.npu.synchronize()



import traceback,collections
counts=collections.Counter();examples={};phase='warmup'
original_bool=torch.Tensor.__bool__
original_item=torch.Tensor.item
def note(x,kind):
    stack=traceback.extract_stack(limit=12)[:-2]
    key=(phase,kind,str(x.device),' | '.join(f'{z.filename}:{z.lineno}:{z.name}' for z in stack[-7:]))
    counts[key]+=1
    if key not in examples:examples[key]=dict(shape=list(x.shape),dtype=str(x.dtype),stack=[dict(file=z.filename,line=z.lineno,function=z.name,code=z.line) for z in stack])
def trace_bool(x):
    note(x,'bool');return original_bool(x)
def trace_item(x,*a,**kw):
    note(x,'item');return original_item(x,*a,**kw)
with torch.inference_mode():
    model.generate(**inputs,max_new_tokens=2,do_sample=False,use_cache=True)
    torch.npu.synchronize()
    phase='prefill'
    def pre(m,a,kw):
        global phase
        phase='decode' if kw.get('input_ids') is not None and kw['input_ids'].shape[1]==1 else 'prefill'
    hook=model.register_forward_pre_hook(pre,with_kwargs=True)
    torch.Tensor.__bool__=trace_bool;torch.Tensor.item=trace_item
    try:
        out=model.generate(**inputs,max_new_tokens=5,do_sample=False,use_cache=True)
        torch.npu.synchronize()
    finally:
        torch.Tensor.__bool__=original_bool;torch.Tensor.item=original_item;hook.remove()
rows=[dict(phase=k[0],kind=k[1],device=k[2],count=v,**examples[k]) for k,v in counts.items()]
Path('results/e119_scalar_stacks.json').write_text(json.dumps(rows,indent=2)+'\n')
for row in rows:print(row['phase'],row['kind'],row['device'],row['count'],row['stack'][-3:],flush=True)
