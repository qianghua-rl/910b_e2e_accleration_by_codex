"""First16 complete real workflow: native first turn, batched historical second."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import torch_npu
from transformers import AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

from agent_protocol import CONCISE_SYSTEM,TOOLS,parse_call,execute,answer,correct
from e166_batch16_row_cache_workflow_gate import IDS,row_cache_to_cpu
from e168_batched_history import pack
from e168_ragged_history_gate import load_model,write
from e176_history_runtime import HistoryRuntime,continue_batched


MEMORY_BUDGET=62094008832
CACHE_FIELDS=('keys','values','conv_states','recurrent_states')


def cache_digest(cache):
    digest=hashlib.sha256()
    for index,layer in enumerate(cache.layers):
        for name in CACHE_FIELDS:
            value=getattr(layer,name,None)
            if not isinstance(value,torch.Tensor):continue
            host=value.detach().to('cpu').contiguous()
            digest.update(f'{index}:{name}:{tuple(host.shape)}:{host.dtype}'.encode())
            digest.update(host.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def row_cache_to_npu(cache,row):
    import copy
    cloned=copy.copy(cache)
    layers=[]
    for layer in cache.layers:
        target=copy.copy(layer)
        for name in ('keys','values','conv_states','recurrent_states'):
            value=getattr(layer,name,None)
            if isinstance(value,torch.Tensor):setattr(target,name,value[row:row+1].detach().to('npu:0').clone())
        if hasattr(target,'max_batch_size'):target.max_batch_size=1
        if hasattr(target,'device'):target.device=torch.device('npu:0')
        layers.append(target)
    cloned.layers=layers
    return cloned


def preflight():
    torch.npu.set_device(0)
    free,total=torch.npu.mem_get_info()
    # Model load needs about56GB; retain4GB headroom. The measured E100 peak
    # remains a separate hard candidate gate after the workflow.
    required=60000000000
    return {'device':'npu:0','physical_device':__import__('os').environ.get('ASCEND_RT_VISIBLE_DEVICES','unknown'),
        'free_bytes':int(free),'total_bytes':int(total),
        'required_free_bytes':required,'resident_model_benchmark_ready':free>=required,
        'checked_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}


def run(argv=None,shared_model=None,shared_tokenizer=None):
    global IDS
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',required=True)
    ap.add_argument('--preflight-only',action='store_true')
    ap.add_argument('--snapshot-device',choices=['cpu','npu'],default='cpu')
    ap.add_argument('--correctness-only',action='store_true')
    ap.add_argument('--continue-on-token-cap',action='store_true',help='Record truncated rows and continue eligible rows via fresh batched second round; report failures without aborting workload')
    ap.add_argument('--task-ids',help='Exactly16 IDs in frozen E100 order for one batch; default first16')
    ap.add_argument('--fresh-batch-on-mismatch',action='store_true')
    ap.add_argument('--defer-reference-check',action='store_true',help='Persistent timed parent audits after total timing')
    ap.add_argument('--recurrent-graph-state-dtype',choices=['float32','bfloat16'],
        help='Lazily capture each layer only when its cached T=1 recurrent callback first runs')
    ap.add_argument('--recurrent-graph-layer-stride',type=int,default=1,
        help='Capture every Nth recurrent layer; other layers keep the native recurrent path')
    args=ap.parse_args(argv)
    if args.recurrent_graph_layer_stride < 1:raise SystemExit('graph layer stride must be >=1')
    if args.task_ids:
        IDS=[int(x) for x in args.task_ids.split(',')]
        if not 1<=len(IDS)<=16 or len(set(IDS))!=len(IDS):raise SystemExit('batch must contain 1..16 unique rows')
    path=Path(args.output)
    transcript=Path(args.output+'.workflow.jsonl')
    if path.exists() or transcript.exists():raise SystemExit('output exists; preserve prior trial')
    resource=preflight() if shared_model is None else {'physical_device':__import__('os').environ.get('ASCEND_RT_VISIBLE_DEVICES'),
        'resident_model_benchmark_ready':True,'shared_resident_model':True}
    if args.preflight_only or not resource['resident_model_benchmark_ready']:
        write(path,{'kind':'preflight','resource':resource,'performance_measured':False,
            'status':'ready' if resource['resident_model_benchmark_ready'] else 'resource_blocked_before_load'})
        print(json.dumps(resource),flush=True)
        return 0 if resource['resident_model_benchmark_ready'] else 75
    torch.set_num_threads(8);torch.manual_seed(42)
    load_start=time.perf_counter()
    model=load_model() if shared_model is None else shared_model
    tok=AutoTokenizer.from_pretrained('model/qwen3.5-27b',padding_side='left') if shared_tokenizer is None else shared_tokenizer
    graph_wrappers=[]
    graph_pool=None
    if args.recurrent_graph_state_dtype:
        from e069_hf_recurrent_call_graph import RecurrentGraph
        graph_dtype={'float32':torch.float32,'bfloat16':torch.bfloat16}[args.recurrent_graph_state_dtype]
        recurrent_modules=[module for module in model.modules() if isinstance(module,Qwen3_5GatedDeltaNet)]
        selected=[(index,module) for index,module in enumerate(recurrent_modules)
                  if index % args.recurrent_graph_layer_stride == 0]
        graph_wrappers=[RecurrentGraph(index,state_dtype=graph_dtype,verify_on_capture=False)
                        for index,_ in selected]
        for (_,module),wrapper in zip(selected,graph_wrappers):
            module.recurrent_gated_delta_rule=wrapper
    torch.npu.synchronize()
    load_seconds=time.perf_counter()-load_start
    source=Path('data/gsm8k_128.jsonl').read_bytes()
    data={r['id']:r for r in map(json.loads,source.splitlines())}
    eos=model.generation_config.eos_token_id
    eos=set(eos if isinstance(eos,list) else [eos])
    path.parent.mkdir(parents=True,exist_ok=True)
    graph_event_path=Path(str(path)+'.graph-events.jsonl')
    if graph_wrappers and graph_event_path.exists():
        raise SystemExit('graph event log exists; preserve prior trial')
    def graph_event(row):
        if not graph_wrappers:return
        row={**row,
            'allocated_bytes':int(torch.npu.memory_allocated()),
            'reserved_bytes':int(torch.npu.memory_reserved()),
            'peak_allocated_bytes':int(torch.npu.max_memory_allocated()),
            'peak_reserved_bytes':int(torch.npu.max_memory_reserved())}
        with graph_event_path.open('a') as stream:
            stream.write(json.dumps(row)+'\n')
            stream.flush()
    for wrapper in graph_wrappers:
        wrapper.event_callback=graph_event
    def emit(row):
        with transcript.open('a') as stream:stream.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    emit({'kind':'metadata','load_seconds':load_seconds,'task_order':IDS,'batch_size':len(IDS),
        'protocol':'concise','attention':'sdpa','dtype':'bfloat16','max_new_tokens':512,'final_max_new_tokens':12,
        'recurrent_graph_state_dtype':args.recurrent_graph_state_dtype,
        'recurrent_graph_layer_stride':args.recurrent_graph_layer_stride,
        'recurrent_graph_pool_policy':'per_graph_default_pool' if graph_wrappers else None,
        'graph_capture_policy':'first cached T=1 callback; prompt prefill stays on chunk_gated_delta_rule'})
    snapshots={};calls=0;graph_forward_events=[]
    # Keep row-level evidence when a batch cannot produce all EOS snapshots.
    # This makes the fail-closed boundary actionable without changing capture
    # semantics or synthesizing a cache snapshot for a non-EOS row.
    row_ledger=[{'row': row, 'task_id': IDS[row], 'last_token': None,
                 'eos_forward': None, 'tokens_seen': 0} for row in range(len(IDS))]
    def capture(module,hook_args,kwargs,out):
        nonlocal calls
        tokens=out.logits[:,-1].argmax(-1).tolist()
        for row,token in enumerate(tokens):
            row_ledger[row]['last_token']=int(token)
            row_ledger[row]['tokens_seen'] += 1
            if token in eos and row_ledger[row]['eos_forward'] is None:
                row_ledger[row]['eos_forward']=calls
            if row in snapshots or token not in eos:continue
            snapshots[row]={'cache':row_cache_to_cpu(out.past_key_values,row) if args.snapshot_device=='cpu'
                else row_cache_to_npu(out.past_key_values,row),
                'mask':inputs['attention_mask'][row].tolist()+[1]*calls,
                'logical_prompt':inputs['input_ids'][row][inputs['attention_mask'][row].bool()].tolist()}
        calls+=1
        if graph_wrappers:
            event={'kind':'model_forward_end','forward_index':forward_index-1,
                'captures':sum(wrapper.capture_count for wrapper in graph_wrappers),
                'replays':sum(wrapper.replay_count for wrapper in graph_wrappers)}
            graph_forward_events.append(event)
            graph_event(event)
    start=time.perf_counter()
    states=[{'task':data[i],'messages':[{'role':'system','content':CONCISE_SYSTEM},
        {'role':'user','content':data[i]['question']}],'rounds':[]} for i in IDS]
    prompts=[tok.apply_chat_template(s['messages'],tools=TOOLS,tokenize=False,
        add_generation_prompt=True,enable_thinking=False) for s in states]
    inputs=tok(prompts,add_special_tokens=False,padding=True,return_tensors='pt').to('npu:0')
    forward_index=0
    def forward_start(module,hook_args):
        nonlocal forward_index
        if graph_wrappers:
            graph_event({'kind':'model_forward_start','forward_index':forward_index,
                'captures':sum(wrapper.capture_count for wrapper in graph_wrappers),
                'replays':sum(wrapper.replay_count for wrapper in graph_wrappers)})
            forward_index+=1
    pre_handle=model.register_forward_pre_hook(forward_start) if graph_wrappers else None
    handle=model.register_forward_hook(capture,with_kwargs=True)
    first_start=time.perf_counter()
    with torch.inference_mode():
        try:
            generated=model.generate(**inputs,max_new_tokens=512,do_sample=False,use_cache=True)
        finally:
            handle.remove()
            if pre_handle is not None:pre_handle.remove()
        first=[]
        for ids in generated[:,inputs['input_ids'].shape[1]:].tolist():
            stop=next((i+1 for i,t in enumerate(ids) if t in eos),len(ids))
            first.append(ids[:stop])
        torch.npu.synchronize()
        first_seconds=time.perf_counter()-first_start
        first_peak={'allocated':int(torch.npu.max_memory_allocated()),'reserved':int(torch.npu.max_memory_reserved())}
        if len(snapshots)!=len(IDS):
            failure={'status':'missing_real_eos_boundary','batch_task_ids':IDS,
                'captured_rows':sorted(snapshots),
                'missing_rows':[x for x in row_ledger if x['row'] not in snapshots],
                'forward_calls':calls,'generated_width':int(generated.shape[1]-inputs['input_ids'].shape[1]),
                'eos_token_ids':sorted(eos)}
            Path(str(path)+'.eos_failure.json').write_text(json.dumps(failure,indent=2)+'\n')
            if args.continue_on_token_cap:
                # An EOS in the returned sequence without its snapshot is still
                # a capture bug, not a length-budget stop eligible for fallback.
                if any(first[row] and first[row][-1] in eos for row in range(len(IDS)) if row not in snapshots):
                    raise RuntimeError('EOS returned but snapshot missing; see '+str(path)+'.eos_failure.json')
                from capped_workflow import finish_batch
                snapshots.clear()
                del generated,inputs
                transition=time.perf_counter()
                tasks=finish_batch(model,tok,states,first,eos,emit,execute,parse_call,answer,correct,TOOLS)
                torch.npu.synchronize()
                elapsed=time.perf_counter()-start
                peak=int(torch.npu.max_memory_allocated());reserved=int(torch.npu.max_memory_reserved())
                emit(dict(kind='summary',makespan_seconds=elapsed,peak_bytes=peak,peak_reserved_bytes=reserved))
                result=dict(kind='candidate_workflow',policy='continue-on-token-cap-v1',
                    workflow_seconds=elapsed,first_generation_seconds=first_seconds,
                    tool_prepare_pack_seconds=None,second_generation_seconds=None,
                    first_stage_peak=first_peak,peak_allocated_bytes=peak,peak_reserved_bytes=reserved,
                    resource_before=resource,task_order=IDS,tasks=tasks,batch_size=len(IDS),
                    route='token_cap_fresh_eligible_batch',mismatch_ids=[],snapshot_device=args.snapshot_device,
                    correctness_only=args.correctness_only,performance_measured=not args.correctness_only,
                    trajectory_passed=False,memory_passed=peak<=MEMORY_BUDGET,
                    truncated_ids=[t['id'] for t in tasks if t['first_truncated']],
                    incomplete_ids=[t['id'] for t in tasks if t['completion_status']!='completed'],
                    completed_task_count=sum(t['completion_status']=='completed' for t in tasks),
                    score_correct=sum(t['correct'] for t in tasks),
                    tool_count=sum(t['tool_count'] for t in tasks),data_sha256=hashlib.sha256(source).hexdigest())
                write(path,result)
                # Quality failure is data, not a process failure; memory remains
                # a hard runtime stop. Parent must audit all128 denominators.
                return 0 if result['memory_passed'] else 1
            raise RuntimeError('not all real EOS boundaries captured; see '+str(path)+'.eos_failure.json')
        first_cache_digests={str(IDS[row]):cache_digest(snapshots[row]['cache']) for row in range(len(IDS))}
        suffixes=[];ordered=[];mismatches=[];next_prompts=[];final_cache=None
        transition=time.perf_counter()
        for row,(state,tokens) in enumerate(zip(states,first)):
            text=tok.decode(tokens,skip_special_tokens=True)
            code=parse_call(text)
            if code is None:raise RuntimeError('initial first16 contract expects actual tool call')
            tool_output=execute(code)
            state['rounds'].append({'turn':0,'tokens':tokens,'text':text,'code':code,'tool_output':tool_output})
            state['messages'].extend([{'role':'assistant','content':text},
                {'role':'tool','name':'python','content':tool_output}])
            rendered=tok.apply_chat_template(state['messages'],tools=TOOLS,tokenize=False,
                add_generation_prompt=True,enable_thinking=False)
            next_prompts.append(rendered)
            full=tok(rendered,add_special_tokens=False)['input_ids']
            snap=snapshots[row]
            consumed=snap.pop('logical_prompt')+tokens[:-1]
            if full[:len(consumed)]!=consumed:
                mismatches.append(state['task']['id'])
                if not args.fresh_batch_on_mismatch:raise RuntimeError('prefix mismatch: fresh group not implemented; stop')
            if len(snap['mask'])!=snap['cache'].get_seq_length() or sum(snap['mask'])!=len(consumed):
                raise RuntimeError('invalid snapshot boundary')
            snap['consumed']=consumed
            ordered.append(snap);suffixes.append(full[len(consumed):])
        del generated,inputs
        if mismatches:
            del snapshots,ordered,snap
            fresh_inputs=tok(next_prompts,add_special_tokens=False,padding=True,return_tensors='pt').to('npu:0')
            torch.npu.synchronize()
            transition_seconds=time.perf_counter()-transition
            second_start=time.perf_counter()
            generated_second=model.generate(**fresh_inputs,max_new_tokens=12,do_sample=False,use_cache=True)
            second=[]
            for tokens in generated_second[:,fresh_inputs['input_ids'].shape[1]:].tolist():
                stop=next((i+1 for i,t in enumerate(tokens) if t in eos),len(tokens))
                second.append(tokens[:stop])
            torch.npu.synchronize()
            second_seconds=time.perf_counter()-second_start
            model_calls=None
            del fresh_inputs,generated_second
        else:
            second,model_calls,transition_seconds,second_seconds,final_cache=run_reuse(
                model,ordered,snapshots,suffixes,eos,transition)
            del ordered,snapshots,snap
        for state,tokens in zip(states,second):
            text=tok.decode(tokens,skip_special_tokens=True)
            if parse_call(text) is not None:raise RuntimeError('third tool round outside initial first16 gate; stop')
            state['rounds'].append({'turn':1,'tokens':tokens,'text':text})
            state['prediction']=answer(text)
            task=state['task']
            # Match baseline task serialization and scoring inside workflow time.
            emit({'kind':'task','id':task['id'],'rounds':state['rounds'],
                'prediction':state['prediction'],'expected':task['answer'],'tool_count':1,
                'correct':correct(state['prediction'],task['answer'])})
        torch.npu.synchronize()
        workflow_seconds=time.perf_counter()-start
        final_cache_digest=cache_digest(final_cache) if final_cache is not None else None
    emit({'kind':'summary','makespan_seconds':workflow_seconds,
          'peak_bytes':int(torch.npu.max_memory_allocated()),
          'peak_reserved_bytes':int(torch.npu.max_memory_reserved())})
    # All baseline/reference comparison and result serialization occurs after timing.
    baseline=None if args.defer_reference_check else {r['id']:r for r in map(json.loads,Path('results/phase2_concise_sdpa_batch16_final12_length128.jsonl').read_text().splitlines())
        if r.get('kind')=='task'}
    tasks=[]
    for state in states:
        task=state['task']
        trajectory_exact=None
        if baseline is not None:
            ref=baseline[task['id']]
            trajectory_exact=all(r['tokens']==expected['tokens'] and r.get('tool_output')==expected.get('tool_output')
                for r,expected in zip(state['rounds'],ref['rounds'])) and len(state['rounds'])==len(ref['rounds'])
        tasks.append({'id':task['id'],'rounds':state['rounds'],'prediction':state['prediction'],
            'expected':task['answer'],'correct':correct(state['prediction'],task['answer']),
            'trajectory_exact':trajectory_exact})
    peak=int(torch.npu.max_memory_allocated())
    result={'kind':'candidate_workflow','policy':'OPTIMIZATION_ACCEPTANCE.md','load_seconds':load_seconds,
        'workflow_seconds':workflow_seconds,'first_generation_seconds':first_seconds,
        'tool_prepare_pack_seconds':transition_seconds,'second_generation_seconds':second_seconds,
        'first_stage_peak':first_peak,'peak_allocated_bytes':peak,
        'peak_reserved_bytes':int(torch.npu.max_memory_reserved()),'resource_before':resource,
        'first_eos_cache_sha256':first_cache_digests,'final_cache_sha256':final_cache_digest,
        'recurrent_graph_state_dtype':args.recurrent_graph_state_dtype,
        'recurrent_graph_layer_stride':args.recurrent_graph_layer_stride,
        'recurrent_graphs_captured':sum(wrapper.capture_count for wrapper in graph_wrappers),
        'recurrent_graph_replays':sum(wrapper.replay_count for wrapper in graph_wrappers),
        'recurrent_capture_checks':[wrapper.initial_check for wrapper in graph_wrappers],
        'recurrent_graph_forward_events':graph_forward_events,
        'task_order':IDS,'tasks':tasks,'second_turn_model_forwards':model_calls,
        'suffix_lengths':list(map(len,suffixes)),'batch_size':len(IDS),
        'snapshot_device':args.snapshot_device,'correctness_only':args.correctness_only,
        'route':'fresh_batch' if mismatches else 'reuse_batch','mismatch_ids':mismatches,
        'trajectory_passed':None if args.defer_reference_check else all(t['trajectory_exact'] for t in tasks),'memory_passed':peak<=MEMORY_BUDGET,
        'performance_measured':not args.correctness_only,'b1_tensor_difference_veto':False,
        'data_sha256':hashlib.sha256(source).hexdigest()}
    write(path,result)
    print(json.dumps({k:v for k,v in result.items() if k!='tasks'},indent=2),flush=True)
    return 0 if (args.defer_reference_check or result['trajectory_passed']) and result['memory_passed'] else 1


def run_reuse(model,ordered,snapshots,suffixes,eos,transition):
    runtime=HistoryRuntime();runtime.install(model)
    try:
        cache,ids,mask,positions,p=pack(ordered,suffixes)
        # Clear caller-owned snapshot containers after packing to avoid retaining
        # all historical row copies through the second-turn forward.
        snapshots.clear();ordered.clear()
        torch.npu.synchronize()
        transition_seconds=time.perf_counter()-transition
        second_start=time.perf_counter()
        second,cache,mask,positions,calls=continue_batched(model,runtime,cache,ids,mask,positions,suffixes,eos)
        torch.npu.synchronize()
        second_seconds=time.perf_counter()-second_start
        return second,calls,transition_seconds,second_seconds,cache
    finally:runtime.restore()


if __name__=='__main__':raise SystemExit(run())
