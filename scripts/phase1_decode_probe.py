"""Phase 1 launch parameter sweep; no custom kernel, fixed teacher-forced inputs.

Do not run concurrently with the e2e baseline. Numerical equality of logits and
argmax is recorded separately from speed, and requires e2e follow-up.
"""
import json
import time
from pathlib import Path
import torch
import torch_npu
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
from agent_protocol import SYSTEM, TOOLS

if Path('results/phase1_decode_probe.jsonl').exists():
    raise SystemExit('Output already exists')
torch.set_num_threads(8)
torch.manual_seed(42)
model = Qwen3_5ForConditionalGeneration.from_pretrained('model/qwen3.5-27b',
    dtype=torch.bfloat16,attn_implementation='eager',device_map={'':'npu:0'}).eval()
tok = AutoTokenizer.from_pretrained('model/qwen3.5-27b')
q=json.loads(open('data/gsm8k_128.jsonl').readline())['question']
inputs=tok.apply_chat_template([{'role':'system','content':SYSTEM},{'role':'user','content':q}],
    tools=TOOLS,add_generation_prompt=True,enable_thinking=False,return_dict=True,return_tensors='pt').to('npu')
reference=None
with torch.inference_mode():
    for threads in [8,1,4,16,32,8]:
        torch.set_num_threads(threads)
        out=model(**inputs,use_cache=True,logits_to_keep=1)
        # Same teacher-forced token sequence and fresh cache for every condition.
        ids=torch.full((1,1),100,device='npu',dtype=torch.long)
        samples=[]
        logits=[]
        for step in range(24):
            torch.npu.synchronize()
            t=time.perf_counter()
            out=model(input_ids=ids,past_key_values=out.past_key_values,use_cache=True,logits_to_keep=1)
            torch.npu.synchronize()
            if step>=4:
                samples.append(time.perf_counter()-t)
                logits.append(out.logits.cpu())
        values=torch.cat(logits)
        if reference is None:
            reference=values.clone()
        row=dict(threads=threads,samples=samples,median_seconds=sorted(samples)[len(samples)//2],
                 logits_bit_equal=bool(torch.equal(values,reference)),
                 argmax_equal=bool(torch.equal(values.argmax(-1),reference.argmax(-1))),
                 max_logit_error=float((values.float()-reference.float()).abs().max()))
        with open('results/phase1_decode_probe.jsonl','a') as f:
            f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        del out,values,logits
