"""Capture per-row cache at EOS from real E100 batch16 generation, then continue privately."""
import argparse, copy, json, time
from pathlib import Path
import torch, torch_npu
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm, Qwen3_5RMSNormGated, Qwen3_5GatedDeltaNet
from recurrent_single_step import recurrent_single_step
from nonaligned_hybrid_cache import AgentHybridStateStore, TokenwiseHybridRunner, save_state
from agent_protocol import CONCISE_SYSTEM, TOOLS, parse_call, execute

IDS = [640, 1137, 214, 1030, 950, 428, 866, 175, 1130, 445, 1161, 611, 108, 1288, 554, 249]

def row_cache_to_cpu(cache, row):
    cloned = copy.copy(cache)
    layers = []
    for layer in cache.layers:
        target = copy.copy(layer)
        for name in ('keys', 'values', 'conv_states', 'recurrent_states'):
            value = getattr(layer, name, None)
            if isinstance(value, torch.Tensor):
                setattr(target, name, value[row:row + 1].detach().to('cpu').clone())
        if hasattr(target, 'max_batch_size'):
            target.max_batch_size = 1
        if hasattr(target, 'device'):
            target.device = torch.device('cpu')
        layers.append(target)
    cloned.layers = layers
    return cloned

def cache_to_device(cache, device):
    cloned = copy.copy(cache)
    layers = []
    for layer in cache.layers:
        target = copy.copy(layer)
        for name in ('keys', 'values', 'conv_states', 'recurrent_states'):
            value = getattr(layer, name, None)
            if isinstance(value, torch.Tensor): setattr(target, name, value.to(device))
        if hasattr(target, 'device'): target.device = torch.device(device)
        layers.append(target)
    cloned.layers = layers
    return cloned

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--limit', type=int, default=16); ap.add_argument('--output', default='results/e166_batch16_row_cache_workflow_gate.json'); ap.add_argument('--mode', choices=['candidate','baseline'], default='candidate'); args = ap.parse_args()
    if not 1 <= args.limit <= 16: raise SystemExit('--limit must be in [1,16]')
    baseline_path = Path('results/e140_full_ab/trial0_baseline.jsonl')
    base_rows = [json.loads(x) for x in baseline_path.read_text().splitlines()]
    base = {str(x['id']): x for x in base_rows if x.get('kind') == 'task'}
    data = {str(x['id']): x for x in map(json.loads, Path('data/gsm8k_128.jsonl').read_text().splitlines())}
    tokenizer = AutoTokenizer.from_pretrained('model/qwen3.5-27b', padding_side='left')
    model = Qwen3_5ForConditionalGeneration.from_pretrained('model/qwen3.5-27b', dtype=torch.bfloat16,
        attn_implementation='sdpa', device_map={'': 'npu:0'}).eval()
    def rms(self, x): return torch_npu.npu_rms_norm(x, (1.0 + self.weight).to(x.dtype), self.eps)[0]
    def gated(self, hidden_states, gate=None):
        normalized = torch_npu.npu_rms_norm(hidden_states, self.weight, self.variance_epsilon)[0]
        return torch_npu.npu_swiglu(torch.cat((gate, normalized), dim=-1))
    for m in model.modules():
        if isinstance(m, Qwen3_5RMSNorm): m.forward = rms.__get__(m, type(m))
        if isinstance(m, Qwen3_5RMSNormGated): m.forward = gated.__get__(m, type(m))
        if isinstance(m, Qwen3_5GatedDeltaNet): m.recurrent_gated_delta_rule = recurrent_single_step
    workflow_start = time.perf_counter()
    prompts = []
    for task_id in IDS[:args.limit]:
        messages = [{'role': 'system', 'content': CONCISE_SYSTEM}, {'role': 'user', 'content': data[str(task_id)]['question']}]
        prompts.append(tokenizer.apply_chat_template(messages, tools=TOOLS, tokenize=False,
            add_generation_prompt=True, enable_thinking=False))
    inputs = tokenizer(prompts, add_special_tokens=False, padding=True, return_tensors='pt').to('npu')
    eos = model.generation_config.eos_token_id; eos = eos if isinstance(eos, (list, tuple)) else [eos]
    eos_set = set(int(x) for x in eos)
    snapshots, call_index, seen = {}, 0, set()
    def capture(module, args, kwargs, output):
        nonlocal call_index
        cache = getattr(output, 'past_key_values', None)
        logits = getattr(output, 'logits', None)
        if cache is None or logits is None: return
        next_ids = logits[:, -1, :].argmax(-1).tolist()
        physical_len = int(cache.get_seq_length())
        for row, token in enumerate(next_ids):
            if row in seen or token not in eos_set: continue
            row_cache = row_cache_to_cpu(cache, row)
            prompt_mask = inputs['attention_mask'][row].tolist()
            logical_prompt = inputs['input_ids'][row][inputs['attention_mask'][row].bool()].tolist()
            generated_before_eos = call_index
            mask = prompt_mask + [1] * generated_before_eos
            assert len(mask) == physical_len, (len(mask), physical_len, call_index)
            snapshots[row] = {'cache': row_cache, 'mask': mask,
                              'logical_prompt': logical_prompt,
                              'generated_before_eos': generated_before_eos,
                              'physical_cache_length': physical_len}
            seen.add(row)
        call_index += 1
    torch.npu.reset_peak_memory_stats()
    hook = model.register_forward_hook(capture, with_kwargs=True)
    with torch.inference_mode():
        batch_out = model.generate(**inputs, max_new_tokens=512, do_sample=False, use_cache=True,
                                   return_dict_in_generate=True)
    hook.remove()
    gen_peak_allocated = int(torch.npu.max_memory_allocated())
    gen_peak_reserved = int(torch.npu.max_memory_reserved())
    generated = [x.tolist() for x in batch_out.sequences[:, inputs['input_ids'].shape[1]:]]
    trimmed=[]
    for tokens in generated:
        stop = next((i+1 for i,t in enumerate(tokens) if t in eos_set), len(tokens)); trimmed.append(tokens[:stop])
    assert len(snapshots) == args.limit, ('missing EOS cache snapshots', sorted(seen), args.limit)
    if args.mode == 'baseline':
        result={'mode':'baseline','batch_size':args.limit,
                'first_turn_tokens_exact':True,
                'workflow_seconds':time.perf_counter()-workflow_start,
                'peak_allocated_bytes':gen_peak_allocated,
                'peak_reserved_bytes':gen_peak_reserved,
                'performance_measured':False,'scope':'E100 batch first-turn peak allocation only'}
        Path(args.output).write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2)); return
    runner, store = TokenwiseHybridRunner(model), AgentHybridStateStore()
    results={}
    torch.npu.reset_peak_memory_stats()
    with torch.inference_mode():
        for row, task_id in enumerate(IDS[:args.limit]):
            task_key=str(task_id); expected=base[task_key]; snap=snapshots[row]
            first_tokens=trimmed[row]
            assert first_tokens == expected['rounds'][0]['tokens'], f'{task_key}: batch16 first-turn token drift'
            consumed=snap['logical_prompt'] + first_tokens[:-1]
            assert len(consumed) == expected['rounds'][0].get('prompt_tokens',len(snap['logical_prompt'])) + len(first_tokens)-1
            cache=cache_to_device(snap['cache'],'npu')
            store.put(task_key, save_state(consumed, cache, snap['mask']))
            messages=[{'role':'system','content':CONCISE_SYSTEM}, {'role':'user','content':data[task_key]['question']},
                      {'role':'assistant','content':tokenizer.decode(first_tokens,skip_special_tokens=True)}]
            code=parse_call(messages[-1]['content'])
            if code is None: raise RuntimeError(f'{task_key}: baseline tool call did not parse')
            tool_output=execute(code)
            if tool_output != expected['rounds'][0]['tool_output']:
                raise RuntimeError(f'{task_key}: tool output mismatch')
            messages.append({'role':'tool','name':'python','content':tool_output})
            rendered=tokenizer.apply_chat_template(messages,tools=TOOLS,tokenize=False,
                add_generation_prompt=True,enable_thinking=False)
            full=tokenizer(rendered,add_special_tokens=False,return_tensors='pt')['input_ids'].to('npu')
            decision,second,final_state=runner.generate_continuation(
                store,task_key,full[0].tolist(),len(consumed),
                attention_mask=torch.ones((1,full.shape[1]),dtype=torch.long,device='npu'),
                max_new_tokens=512,eos_token_ids=eos,device='npu')
            result={'first_exact':True,'first_tokens':len(first_tokens),
                    'snapshot_physical_length':snap['physical_cache_length'],
                    'logical_boundary':len(consumed),'mask_length':len(snap['mask']),
                    'route':decision,'second_tokens':len(second),
                    'second_exact':list(second)==expected['rounds'][1]['tokens'],
                    'final_boundary_exact':final_state.token_ids==tuple(full[0].tolist()+list(second[:-1]))}
            result['passed']=result['second_exact'] and result['route']['route']=='reuse' and result['final_boundary_exact']
            results[task_key]=result
            partial={'mode':'candidate','batch_size':args.limit,'tasks':results,'passed':all(x['passed'] for x in results.values()),
                     'generation_peak_allocated_bytes':gen_peak_allocated,'generation_peak_reserved_bytes':gen_peak_reserved,
                     'candidate_peak_allocated_bytes':int(torch.npu.max_memory_allocated()),
                     'candidate_peak_reserved_bytes':int(torch.npu.max_memory_reserved()),
                     'workflow_seconds':time.perf_counter()-workflow_start,
                     'peak_memory_measured':True,'performance_measured':True,
                     'scope':'E100 B16 first-turn generation; per-row EOS cache snapshot to private states; correctness only'}
            Path(args.output).write_text(json.dumps(partial,indent=2)+'\n')
            print(json.dumps({'task':task_key,**result}),flush=True)
            if not result['passed']: break
    torch.npu.synchronize()
    final = json.loads(Path(args.output).read_text())
    final['workflow_seconds'] = time.perf_counter()-workflow_start
    final['candidate_peak_allocated_bytes'] = int(torch.npu.max_memory_allocated())
    final['candidate_peak_reserved_bytes'] = int(torch.npu.max_memory_reserved())
    final['passed'] = len(results) == args.limit and all(x['passed'] for x in results.values())
    Path(args.output).write_text(json.dumps(final,indent=2)+'\n')
    print(json.dumps({'passed':final['passed'],'tasks_checked':len(results),
                      'workflow_seconds':final['workflow_seconds'],
                      'peak_allocated_bytes':final['candidate_peak_allocated_bytes']},indent=2))

if __name__=='__main__': main()
