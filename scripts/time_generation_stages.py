"""Alternating original/timed HF generation on E100 first-batch prompts."""
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
parser.add_argument('--single-step-recurrent', action='store_true')
parser.add_argument('--fused-rmsnorm', action='store_true')
parser.add_argument('--fused-gated-rmsnorm', action='store_true')
parser.add_argument('--output', default='results/e123_generation_stages.json')
parser.add_argument('--steps', type=int, default=32)
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
torch.npu.synchronize()




from generation_stage_timing import StageTimer, install
import hashlib
results=[]
reference=None
source=None
original_sample=type(model)._sample
with torch.inference_mode():
    model.generate(**inputs,max_new_tokens=2,do_sample=False,use_cache=True)
    torch.npu.synchronize()
    for trial, instrument in enumerate([False,True,False,True,False,True,False]):
        timer=StageTimer()
        if instrument:
            source=install(model,timer)
        else:
            type(model)._sample=original_sample
        torch.npu.synchronize()
        torch.npu.reset_peak_memory_stats()
        started=time.perf_counter()
        output=model.generate(**inputs,max_new_tokens=args.steps,do_sample=False,use_cache=True)
        torch.npu.synchronize()
        wall=(time.perf_counter()-started)*1000
        stages=timer.finish() if instrument else []
        tokens=output[:,inputs['input_ids'].shape[1]:].cpu()
        if reference is None: reference=tokens.clone()
        exact=bool(torch.equal(tokens,reference))
        saved=torch.tensor([r['rounds'][0]['tokens'][:args.steps] for r in tasks])
        saved_exact=bool(torch.equal(tokens,saved))
        row=dict(trial=trial,instrumented=instrument,wall_ms=wall,
                 tokens_exact=exact,e100_tokens_exact=saved_exact,
                 peak_bytes=torch.npu.max_memory_allocated(),stages=stages)
        results.append(row)
        Path(args.output).write_text(json.dumps(dict(
            tasks=[r['id'] for r in tasks],steps=args.steps,results=results,
            source_sha256=hashlib.sha256(source.encode()).hexdigest() if source is not None else None,
            limitations=['Device event intervals include host submission gaps and queue effects; not kernel busy time.',
                         'scheduler means HF loop condition and input preparation, not Agent admission/tool scheduling.',
                         'postprocess includes model kwargs/cache metadata update, EOS masking, concatenation and stopping criteria.',
                         'sampling includes logits copy/processors and greedy argmax.',
                         'prefill includes HF prefill preparation and forward; excluded from decode comparison.',
                         'Terminal failed loop condition is recorded as scheduler step=steps.',
                         'Short first-round replay cannot establish full128 makespan or quality.']),indent=2)+'\n')
        print(json.dumps({k:v for k,v in row.items() if k!='stages'}),flush=True)
        if not exact or not saved_exact: raise SystemExit('Token drift: stop')
        del output
type(model)._sample=original_sample
