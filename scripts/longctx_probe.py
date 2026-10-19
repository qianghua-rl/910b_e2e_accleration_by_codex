"""Long-context retrieval guardrail, independent of GSM8K quality.

Run reference and candidate identically; this does not replace all-task GSM8K.
Context size and seeds are fixed and recorded, with OOM reported as failure.
"""
import argparse
import json
import time
import traceback
from pathlib import Path
import torch
import torch_npu
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
p.add_argument('--attention', default='eager')
p.add_argument('--lengths', type=int, nargs='+', default=[2048,4096,8192])
p.add_argument('--single-step-recurrent', action='store_true')
p.add_argument('--fused-rmsnorm', action='store_true')
p.add_argument('--fused-gated-rmsnorm', action='store_true')
args = p.parse_args()
if Path(args.output).exists():
    raise SystemExit('Output already exists')
torch.set_num_threads(8)
torch.manual_seed(42)
tok = AutoTokenizer.from_pretrained('model/qwen3.5-27b')
model = Qwen3_5ForConditionalGeneration.from_pretrained('model/qwen3.5-27b',
    dtype=torch.bfloat16,attn_implementation=args.attention,device_map={'':'npu:0'}).eval()
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet, Qwen3_5RMSNorm, Qwen3_5RMSNormGated
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
filler = tok.encode('This is an ordinary archived record with no secret code.\n',add_special_tokens=False)
def emit(row):
    with open(args.output,'a') as f:
        f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)

with torch.inference_mode():
    for length in args.lengths:
        for fraction in [.1,.5,.9]:
            row = dict(context_target=length,needle_fraction=fraction,attention=args.attention,
                       single_step_recurrent=args.single_step_recurrent,
                       fused_rmsnorm=args.fused_rmsnorm,
                       fused_gated_rmsnorm=args.fused_gated_rmsnorm,
                       expected='739281',benchmark='synthetic_retrieval',max_new_tokens=32)
            try:
                budget=max(0,length-100)
                ids=(filler*((budget+len(filler)-1)//len(filler)))[:budget]
                split=int(len(ids)*fraction)
                context=(tok.decode(ids[:split])+ '\nThe secret code for vault Orion is 739281.\n'
                         +tok.decode(ids[split:]))
                prompt=tok.apply_chat_template([{'role':'user','content':context+
                    '\nWhat is the secret code for vault Orion? Reply with only the six digits.'}],
                    add_generation_prompt=True,enable_thinking=False,return_dict=True,return_tensors='pt').to('npu')
                row['input_tokens']=prompt['input_ids'].shape[1]
                torch.npu.reset_peak_memory_stats()
                torch.npu.synchronize()
                t=time.perf_counter()
                generated=model.generate(**prompt,max_new_tokens=32,do_sample=False,use_cache=True)
                torch.npu.synchronize()
                row['seconds']=time.perf_counter()-t
                tokens=generated[0,prompt['input_ids'].shape[1]:].tolist()
                row.update(tokens=tokens,text=tok.decode(tokens,skip_special_tokens=True),
                           peak_bytes=torch.npu.max_memory_allocated())
                row['correct']=row['text'].strip()=='739281'
                del generated,prompt
            except Exception:
                row.update(correct=False,error=traceback.format_exc())
                emit(row)
                raise
            emit(row)
