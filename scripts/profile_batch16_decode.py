"""Profile a short decode slice from the current sorted batch-16 workload."""
import argparse
import json
import time
import os
from pathlib import Path

import torch
import torch_npu
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

from agent_protocol import CONCISE_SYSTEM, TOOLS


parser = argparse.ArgumentParser()
parser.add_argument('--run', required=True, help='Completed E056-style JSONL run')
parser.add_argument('--trace-dir', required=True)
parser.add_argument('--max-new-tokens', type=int, default=32)
parser.add_argument('--single-step-recurrent', action='store_true')
parser.add_argument('--fused-rmsnorm', action='store_true')
parser.add_argument('--fused-gated-rmsnorm', action='store_true')
parser.add_argument('--decode-only', action='store_true', help='Skip prefill and first decode; profile next four decode steps')
parser.add_argument('--attribute-decode', action='store_true', help='Label linear modules and recurrent state operations')
parser.add_argument('--threads', type=int, default=8)
parser.add_argument('--fixed-replay', action='store_true')
parser.add_argument('--mlp-nz', action='store_true')
args = parser.parse_args()
trace_dir = Path(args.trace_dir)
if trace_dir.exists():
    raise SystemExit(f'Trace directory exists: {trace_dir}')

torch.set_num_threads(args.threads)
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
if args.mlp_nz:
    from mlp_nz import convert_mlp_nz
    nz_preparation = convert_mlp_nz(model)
else:
    nz_preparation = None
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

if args.attribute_decode:
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            original = module.forward
            category = 'mlp' if '.mlp.' in name else ('lm_head' if name.endswith('lm_head') else 'projection')
            def marked_linear(*a, _fn=original, _label=f'decode_cost/{category}/{name}', **kw):
                with torch.profiler.record_function(_label):
                    return _fn(*a, **kw)
            module.forward = marked_linear
    for module in model.modules():
        if isinstance(module, Qwen3_5GatedDeltaNet):
            original = module.recurrent_gated_delta_rule
            def marked_recurrent(*a, _fn=original, **kw):
                with torch.profiler.record_function('decode_cost/recurrent'):
                    return _fn(*a, **kw)
            module.recurrent_gated_delta_rule = marked_recurrent

with torch.inference_mode():
    model.generate(**inputs, max_new_tokens=2, do_sample=False, use_cache=True)
    torch.npu.synchronize()
    replay_options = {}
    if args.fixed_replay:
        from transformers import LogitsProcessor, LogitsProcessorList
        replay = torch.tensor([t['rounds'][0]['tokens'][:args.max_new_tokens] for t in tasks], device='npu')
        class ForceReplay(LogitsProcessor):
            def __call__(self, ids, scores):
                pos = ids.shape[1] - inputs['input_ids'].shape[1]
                scores.fill_(float('-inf'))
                return scores.scatter_(1, replay[:, pos:pos+1], 0.)
        replay_options['logits_processor'] = LogitsProcessorList([ForceReplay()])
    start = time.perf_counter()
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU],
        record_shapes=False, profile_memory=False, with_stack=False,
        schedule=torch_npu.profiler.schedule(wait=1, warmup=1, active=4, repeat=1) if args.decode_only else None,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(trace_dir)),
    ) as prof:
        hook = model.register_forward_hook(lambda *unused: prof.step()) if args.decode_only else None
        with torch.profiler.record_function('batch16_generate_prefill_decode'):
            output = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                    do_sample=False, use_cache=True, **replay_options)
            torch.npu.synchronize()
        if hook is not None:
            hook.remove()
    wall_seconds = time.perf_counter() - start

row = {
    'kind': 'batch16_profile', 'tasks': [task['id'] for task in tasks],
    'prompt_tokens': [int(v) for v in inputs['attention_mask'].sum(-1).cpu().tolist()],
    'profile_max_new_tokens': args.max_new_tokens,
    'decode_only': args.decode_only,
    'attribute_decode': args.attribute_decode,
    'threads': args.threads,
    'cpu_affinity': sorted(os.sched_getaffinity(0)),
    'fixed_replay_equal': bool(torch.equal(output[:, inputs['input_ids'].shape[1]:], replay)) if args.fixed_replay else None,
    'single_step_recurrent': args.single_step_recurrent,
    'fused_rmsnorm': args.fused_rmsnorm,
    'fused_gated_rmsnorm': args.fused_gated_rmsnorm,
    'profiled_wall_seconds': wall_seconds,
    'generated_tokens_per_row': int(output.shape[1] - inputs['input_ids'].shape[1]),
    'peak_bytes': int(torch.npu.max_memory_allocated()),
    'trace_dir': str(trace_dir),
    'mlp_nz': args.mlp_nz,
    'nz_preparation': nz_preparation,
}
print(json.dumps(row), flush=True)
