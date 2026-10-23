"""Thread sensitivity with fixed E100 prompts/tokens and original HF generation."""
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



from transformers import LogitsProcessor, LogitsProcessorList
import statistics
steps=12
forced_tokens=torch.tensor([row['rounds'][0]['tokens'][:steps] for row in tasks],device='npu',dtype=torch.long)
assert forced_tokens.shape==(16,steps)
class ForceReplay(LogitsProcessor):
    def __call__(self, ids, scores):
        pos=ids.shape[1]-inputs['input_ids'].shape[1]
        scores.fill_(float('-inf'))
        scores.scatter_(1,forced_tokens[:,pos:pos+1],0.)
        return scores
results=[]
reference=None
with torch.inference_mode():
    model.generate(**inputs,max_new_tokens=2,do_sample=False,use_cache=True)
    torch.npu.synchronize()
    for trial,threads in enumerate([8,1,8,4,8,4,8,1,8,4,8,1,8,1,8,4,8]):
        torch.set_num_threads(threads)
        logits=[];event_pairs=[];host=[]
        def before(module,a,kw):
            if kw.get('input_ids') is not None and kw['input_ids'].shape[1]==1:
                start=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
                host.append(time.perf_counter());start.record();event_pairs.append((start,end))
        def after(module,a,kw,out):
            if kw.get('input_ids') is not None and kw['input_ids'].shape[1]==1:
                event_pairs[-1][1].record();logits.append(out.logits.detach())
        pre=model.register_forward_pre_hook(before,with_kwargs=True)
        post=model.register_forward_hook(after,with_kwargs=True)
        torch.npu.synchronize();t=time.perf_counter()
        output=model.generate(**inputs,max_new_tokens=steps,do_sample=False,use_cache=True,
                              logits_processor=LogitsProcessorList([ForceReplay()]))
        torch.npu.synchronize();finished=time.perf_counter()
        pre.remove();post.remove()
        values=torch.stack([x.float().cpu() for x in logits])
        if reference is None:reference=values.clone()
        error=(values-reference).abs()
        row=dict(trial=trial,threads=threads,decode_calls=len(logits),
                 generate_wall_ms=(finished-t)*1000,
                 decode_span_wall_ms=(finished-host[0])*1000,
                 forward_event_ms=[a.elapsed_time(b) for a,b in event_pairs],
                 logits_exact=bool(torch.equal(values,reference)),
                 logits_max_abs_error=float(error.max()),finite=bool(values.isfinite().all()),
                 argmax_equal=bool(torch.equal(values.argmax(-1),reference.argmax(-1))),
                 replay_tokens_equal=bool(torch.equal(output[:,inputs['input_ids'].shape[1]:],forced_tokens)))
        row['forward_event_sum_ms']=sum(row['forward_event_ms'])
        results.append(row);print(json.dumps(row),flush=True)
        del output,logits,values,error
        Path('results/e100_thread_sensitivity.json').write_text(json.dumps(dict(
            tasks=[x['id'] for x in tasks],forced_tokens=forced_tokens.cpu().tolist(),results=results,
            limitations=['NPU events bracket forwards and include host submission gaps; not pure kernel time',
                         'wall decode span includes original HF generation preparation/processors and final synchronization',
                         'forced logits processor adds identical overhead to every condition',
                         'short first-batch replay; not E100 full quality or throughput validation']),indent=2)+'\n')
torch.set_num_threads(8)
