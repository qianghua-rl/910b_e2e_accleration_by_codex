"""Sequential greedy reference runner. All 128 tasks are available at t=0."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import torch
import torch_npu
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm
from agent_protocol import SYSTEM, CONCISE_SYSTEM, TOOLS, parse_call, execute, answer, correct

parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True)
parser.add_argument('--limit', type=int, default=128)
parser.add_argument('--attention', default='eager')
parser.add_argument('--threads', type=int, default=8)
parser.add_argument('--protocol', choices=['original','concise'], default='original')
parser.add_argument('--max-new-tokens', type=int, default=512)
parser.add_argument('--fused-rmsnorm', action='store_true')
parser.add_argument('--fused-rmsnorm-prefill', action='store_true')
args = parser.parse_args()
system_prompt = SYSTEM if args.protocol=='original' else CONCISE_SYSTEM
output = Path(args.output)
if output.exists():
    raise SystemExit('Refusing to overwrite an existing experiment.')
torch.set_num_threads(args.threads)
torch.manual_seed(42)
path = 'model/qwen3.5-27b'
tload = time.perf_counter()
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    path, dtype=torch.bfloat16, attn_implementation=args.attention,
    device_map={'': 'npu:0'}).eval()
if args.fused_rmsnorm or args.fused_rmsnorm_prefill:
    def _fused_rmsnorm(self, x):
        if args.fused_rmsnorm_prefill and x.shape[-2] == 1:
            output = self._norm(x.float())
            return (output * (1.0 + self.weight.float())).type_as(x)
        return torch_npu.npu_rms_norm(x, (1.0 + self.weight).to(x.dtype), self.eps)[0]
    fused_count = 0
    for module in model.modules():
        if isinstance(module, Qwen3_5RMSNorm):
            module.forward = _fused_rmsnorm.__get__(module, Qwen3_5RMSNorm)
            fused_count += 1
else:
    fused_count = 0
tokenizer = AutoTokenizer.from_pretrained(path)
torch.npu.synchronize()
load_seconds = time.perf_counter()-tload
data_bytes = Path('data/gsm8k_128.jsonl').read_bytes()
tasks = [json.loads(line) for line in data_bytes.splitlines()][:args.limit]

def emit(row):
    with output.open('a') as f:
        f.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)

emit(dict(kind='metadata', load_seconds=load_seconds, tasks=len(tasks),
          threads=args.threads,
          protocol=args.protocol, system_prompt=system_prompt,
          tools=TOOLS,
          data_sha256=hashlib.sha256(data_bytes).hexdigest(), attention=args.attention,
          max_new_tokens=args.max_new_tokens, max_rounds=6, do_sample=False, enable_thinking=False,
          fused_rmsnorm=args.fused_rmsnorm, fused_rmsnorm_prefill=args.fused_rmsnorm_prefill,
          fused_rmsnorm_count=fused_count))
start = time.perf_counter()
with torch.inference_mode():
    for task in tasks:
        task_start = time.perf_counter()
        messages = [{'role':'system','content':system_prompt}, {'role':'user','content':task['question']}]
        rounds = []
        tool_count = 0
        final = None
        for turn in range(6):
            prep = time.perf_counter()
            inputs = tokenizer.apply_chat_template(messages, tools=TOOLS, add_generation_prompt=True,
                enable_thinking=False, tokenize=True, return_dict=True, return_tensors='pt').to('npu')
            torch.npu.synchronize()
            prep_seconds = time.perf_counter()-prep
            t = time.perf_counter()
            generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True)
            torch.npu.synchronize()
            generation_seconds = time.perf_counter()-t
            token_ids = generated[0, inputs['input_ids'].shape[1]:].tolist()
            text = tokenizer.decode(token_ids, skip_special_tokens=True)
            code = parse_call(text)
            record = dict(turn=turn, tokens=token_ids, text=text, prepare_seconds=prep_seconds,
                          generation_seconds=generation_seconds)
            messages.append({'role':'assistant','content':text})
            if code is not None:
                t = time.perf_counter()
                result = execute(code)
                tool_count += 1
                record.update(code=code, tool_output=result, tool_seconds=time.perf_counter()-t)
                messages.append({'role':'tool','name':'python','content':result})
            else:
                final = answer(text)
            rounds.append(record)
            del generated, inputs
            if code is None:
                break
        emit(dict(kind='task', id=task['id'], rounds=rounds, prediction=final,
                  expected=task['answer'], tool_count=tool_count,
                  correct=tool_count>0 and correct(final, task['answer']),
                  service_seconds=time.perf_counter()-task_start,
                  completion_seconds=time.perf_counter()-start))
emit(dict(kind='summary', makespan_seconds=time.perf_counter()-start,
          peak_bytes=torch.npu.max_memory_allocated()))
