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
parser.add_argument('--mode', choices=('reference','candidate','both'), default='both')
parser.add_argument('--output', default='results/mlp_nz_model_gate.json')
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


from mlp_nz import convert_mlp_nz
modules = [m for m in model.modules() if isinstance(m, Qwen3_5GatedDeltaNet)]
originals = [m.causal_conv1d_update for m in modules]
results = {}
reference_scores = None
reference_tokens = None
with torch.inference_mode():
    modes = ('reference', 'candidate') if args.mode == 'both' else (args.mode,)
    for name in modes:
        if name == 'candidate':
            nz_preparation=convert_mlp_nz(model)
            print(json.dumps({'nz_preparation':nz_preparation}),flush=True)
        started = time.perf_counter()
        generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                   use_cache=True, return_dict_in_generate=True, output_scores=True)
        torch.npu.synchronize()
        seconds = time.perf_counter() - started
        tokens = generated.sequences[:, inputs['input_ids'].shape[1]:].cpu()
        scores = torch.stack([score.float().cpu() for score in generated.scores])
        del generated
        if name == 'reference':
            reference_scores, reference_tokens = scores, tokens
        results[name] = dict(seconds=seconds,
                             logits_max_abs_error=None, logits_mean_abs_error=None,
                             tokens_equal=None, logits_equal=None,
                             finite=bool(scores.isfinite().all()), tokens=tokens.tolist())
        if reference_scores is not None:
            delta = (scores-reference_scores).abs()
            results[name].update(logits_max_abs_error=float(delta.max()),
                                 logits_mean_abs_error=float(delta.mean()),
                                 tokens_equal=bool(torch.equal(tokens, reference_tokens)),
                                 logits_equal=bool(torch.equal(scores, reference_scores)))
        elif name == 'reference':
            results[name].update(logits_max_abs_error=0.0, logits_mean_abs_error=0.0,
                                 tokens_equal=True, logits_equal=True)
        print(json.dumps({name:results[name]}),flush=True)
        del scores
for m, original in zip(modules, originals): m.causal_conv1d_update = original
Path(args.output).write_text(json.dumps(results,indent=2)+'\n')
