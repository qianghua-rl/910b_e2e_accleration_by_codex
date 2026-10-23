"""Phase 2 candidate: synchronous batched turns, same tool/answer protocol.

Run only after Phase 1 selection. No custom kernels or speculative decoding.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import torch
import torch_npu
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm, Qwen3_5RMSNormGated
from agent_protocol import SYSTEM, CONCISE_SYSTEM, MINIMAL_SYSTEM, TOOLS, parse_call, execute, answer, correct

p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
p.add_argument('--batch-size', type=int, required=True)
p.add_argument('--limit', type=int, default=128)
p.add_argument('--task-ids', help='Optional comma-separated dataset task IDs for matched replay')
p.add_argument('--final-max-new-tokens', type=int, default=512,
               help='Generation cap after a tool result; the initial tool-call cap remains 512')
p.add_argument('--fused-rmsnorm', action='store_true')
p.add_argument('--fused-gated-rmsnorm', action='store_true')
p.add_argument('--single-step-recurrent', action='store_true')
p.add_argument('--single-step-conv', action='store_true')
p.add_argument('--mlp-nz', action='store_true')
p.add_argument('--attention', default='eager')
p.add_argument('--protocol', choices=['original','concise','minimal'], default='original')
p.add_argument('--schedule', choices=['fifo','length','decode_length','generation_length'], default='fifo')
p.add_argument('--order-file', default=None,
               help='Completed JSONL run used only to estimate per-task decode length')
args = p.parse_args()
system_prompt = {'original': SYSTEM, 'concise': CONCISE_SYSTEM,
                 'minimal': MINIMAL_SYSTEM}[args.protocol]
if args.batch_size < 1 or not 1 <= args.limit <= 128 or not 1 <= args.final_max_new_tokens <= 512:
    raise SystemExit('Invalid batch size or task count')
output = Path(args.output)
if output.exists():
    raise SystemExit('Output already exists')
torch.set_num_threads(8)
torch.manual_seed(42)
t = time.perf_counter()
model = Qwen3_5ForConditionalGeneration.from_pretrained('model/qwen3.5-27b',
    dtype=torch.bfloat16, attn_implementation=args.attention, device_map={'':'npu:0'}).eval()
if args.fused_rmsnorm:
    def fused_rmsnorm(self, x):
        return torch_npu.npu_rms_norm(x, (1.0 + self.weight).to(x.dtype), self.eps)[0]
    fused_rmsnorm_count = 0
    for module in model.modules():
        if isinstance(module, Qwen3_5RMSNorm):
            module.forward = fused_rmsnorm.__get__(module, Qwen3_5RMSNorm)
            fused_rmsnorm_count += 1
else:
    fused_rmsnorm_count = 0
if args.fused_gated_rmsnorm:
    def fused_gated_rmsnorm(self, hidden_states, gate=None):
        normalized = torch_npu.npu_rms_norm(hidden_states, self.weight, self.variance_epsilon)[0]
        return torch_npu.npu_swiglu(torch.cat((gate, normalized), dim=-1))
    fused_gated_rmsnorm_count = 0
    for module in model.modules():
        if isinstance(module, Qwen3_5RMSNormGated):
            module.forward = fused_gated_rmsnorm.__get__(module, Qwen3_5RMSNormGated)
            fused_gated_rmsnorm_count += 1
else:
    fused_gated_rmsnorm_count = 0
if args.single_step_recurrent:
    from recurrent_single_step import recurrent_single_step
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet
    for module in model.modules():
        if isinstance(module, Qwen3_5GatedDeltaNet):
            module.recurrent_gated_delta_rule = recurrent_single_step
nz_preparation=None
if args.mlp_nz:
    from mlp_nz import convert_mlp_nz
    nz_preparation=convert_mlp_nz(model)
    print(json.dumps({'kind':'nz_preparation',**nz_preparation}),flush=True)
tok = AutoTokenizer.from_pretrained('model/qwen3.5-27b', padding_side='left')
if args.single_step_conv:
    from causal_conv_single_step import causal_conv_single_step
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet
    for module in model.modules():
        if isinstance(module, Qwen3_5GatedDeltaNet):
            module.causal_conv1d_update = causal_conv_single_step
torch.npu.synchronize()
load_seconds = time.perf_counter()-t
data = Path('data/gsm8k_128.jsonl').read_bytes()
tasks = [json.loads(line) for line in data.splitlines()][:args.limit]
if args.task_ids:
    requested = set(args.task_ids.split(','))
    tasks = [task for task in tasks if str(task['id']) in requested]
    if not tasks or {str(task['id']) for task in tasks} != requested:
        raise SystemExit('Unknown task ID in --task-ids')
if args.schedule == 'length':
    # Static length-aware microbatches reduce padding while preserving stable ordering
    # within each nearby-length group. All tasks remain available at t=0.
    tasks = sorted(tasks, key=lambda task: len(tok.encode(task['question'])))
elif args.schedule == 'decode_length':
    if not args.order_file:
        raise SystemExit('--order-file is required for decode_length')
    estimates = {}
    for line in Path(args.order_file).read_text().splitlines():
        row = json.loads(line)
        if row.get('kind') == 'task':
            estimates[str(row['id'])] = sum(len(r.get('tokens', [])) for r in row.get('rounds', []))
    missing = [str(t['id']) for t in tasks if str(t['id']) not in estimates]
    if missing:
        raise SystemExit(f'missing decode estimates for {len(missing)} tasks')
    # Group similar observed decode work together to reduce synchronous barriers.
    tasks = sorted(tasks, key=lambda task: estimates[str(task['id'])])
elif args.schedule == 'generation_length':
    if not args.order_file:
        raise SystemExit('--order-file is required for generation_length')
    estimates = {}
    for line in Path(args.order_file).read_text().splitlines():
        row = json.loads(line)
        if row.get('kind') == 'task' and row.get('rounds'):
            estimates[str(row['id'])] = len(row['rounds'][0].get('tokens', []))
    missing = [str(task['id']) for task in tasks if str(task['id']) not in estimates]
    if missing:
        raise SystemExit(f'missing first-turn generation estimates for {len(missing)} tasks')
    # Put requests with similar observed first-turn lengths in the same microbatch.
    tasks = sorted(tasks, key=lambda task: estimates[str(task['id'])], reverse=True)
def emit(row):
    with output.open('a') as f:
        f.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)

emit(dict(kind='metadata', load_seconds=load_seconds, tasks=len(tasks), batch_size=args.batch_size,
    attention=args.attention, protocol=args.protocol, system_prompt=system_prompt, schedule=args.schedule,
    order_file=args.order_file,
    data_sha256=hashlib.sha256(data).hexdigest(),
    max_new_tokens=512,final_max_new_tokens=args.final_max_new_tokens,
    fused_rmsnorm=args.fused_rmsnorm,fused_rmsnorm_count=fused_rmsnorm_count,
    fused_gated_rmsnorm=args.fused_gated_rmsnorm,
    fused_gated_rmsnorm_count=fused_gated_rmsnorm_count,
    single_step_recurrent=args.single_step_recurrent,
    single_step_conv=args.single_step_conv,
    mlp_nz=args.mlp_nz,nz_preparation=nz_preparation,
    max_rounds=6,do_sample=False,enable_thinking=False))
start = time.perf_counter()
with torch.inference_mode():
    for base in range(0,len(tasks),args.batch_size):
        active = [dict(task=task,messages=[{'role':'system','content':system_prompt},
            {'role':'user','content':task['question']}],rounds=[],tool_count=0,
            prediction=None,started=time.perf_counter()) for task in tasks[base:base+args.batch_size]]
        for turn in range(6):
            prep = time.perf_counter()
            prompts = [tok.apply_chat_template(s['messages'],tools=TOOLS,tokenize=False,
                add_generation_prompt=True,enable_thinking=False) for s in active]
            inputs = tok(prompts,add_special_tokens=False,padding=True,return_tensors='pt').to('npu')
            torch.npu.synchronize()
            prepare_seconds = time.perf_counter()-prep
            t = time.perf_counter()
            max_new_tokens = 512 if turn == 0 else args.final_max_new_tokens
            generated = model.generate(**inputs,max_new_tokens=max_new_tokens,do_sample=False,use_cache=True)
            rows = generated[:,inputs['input_ids'].shape[1]:].tolist()
            torch.npu.synchronize()
            generation_seconds = time.perf_counter()-t
            survivors = []
            for state, ids in zip(active,rows):
                # HF pads finished rows. Retain first EOS, remove only trailing pads.
                eos = model.generation_config.eos_token_id
                eos = set(eos if isinstance(eos,list) else [eos])
                stop = next((i+1 for i,v in enumerate(ids) if v in eos),len(ids))
                ids = ids[:stop]
                text = tok.decode(ids,skip_special_tokens=True)
                code = parse_call(text)
                record = dict(turn=turn,tokens=ids,text=text,
                    prepare_seconds=prepare_seconds/len(active),
                    generation_seconds=generation_seconds/len(active),
                    batch_generation_seconds=generation_seconds)
                state['messages'].append({'role':'assistant','content':text})
                if code is not None:
                    t = time.perf_counter()
                    result = execute(code)
                    state['tool_count'] += 1
                    record.update(code=code,tool_output=result,tool_seconds=time.perf_counter()-t)
                    state['messages'].append({'role':'tool','name':'python','content':result})
                else:
                    state['prediction'] = answer(text)
                state['rounds'].append(record)
                if code is not None and turn<5:
                    survivors.append(state)
                else:
                    task = state['task']
                    emit(dict(kind='task',id=task['id'],rounds=state['rounds'],
                        prediction=state['prediction'],expected=task['answer'],
                        tool_count=state['tool_count'],correct=state['tool_count']>0 and correct(state['prediction'],task['answer']),
                        service_seconds=time.perf_counter()-state['started'],completion_seconds=time.perf_counter()-start))
            del inputs
            del generated
            active = survivors
            if not active:
                break
emit(dict(kind='summary',makespan_seconds=time.perf_counter()-start,peak_bytes=torch.npu.max_memory_allocated()))
