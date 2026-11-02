"""Real E166 boundaries -> single-forward B2/B16 history append gates."""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import torch
import torch_npu
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5 import modeling_qwen3_5 as hf

from agent_protocol import CONCISE_SYSTEM, TOOLS, parse_call, execute
from e166_batch16_row_cache_workflow_gate import IDS, row_cache_to_cpu
from e168_batched_history import BatchedHistory, FIELDS, clone_cache, pack, unpack
from recurrent_single_step import recurrent_single_step

ROOT = Path('results/e168')
BASE = Path('results/e140_full_ab/trial0_baseline.jsonl')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def require_stage(stage):
    previous={'b2_one':'capture','b2_suffix':'b2_one','b16':'b2_suffix','trajectory':'b16'}
    if stage not in previous: return
    path=ROOT/(previous[stage]+'.json')
    if not path.exists():
        raise RuntimeError(f'{stage}: prerequisite {previous[stage]} has not been run')
    prior=json.loads(path.read_text())
    passed=prior.get('semantic_passed',prior.get('passed',False))
    if not passed:
        raise RuntimeError(f'{stage}: prior numerical/semantic gate failed; stop expansion')


def digest(cache):
    h = hashlib.sha256()
    for layer in cache.layers:
        for name in FIELDS:
            value = getattr(layer, name, None)
            if isinstance(value, torch.Tensor):
                h.update(value.contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def metrics(actual, expected):
    actual, expected = actual.float(), expected.float()
    diff = actual - expected
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    max_abs = float(diff.abs().max())
    relative_l2 = float(torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(expected).clamp_min(1e-12))
    scaled_max = max_abs / max(1., float(expected.abs().max()))
    return {'exact': torch.equal(actual, expected), 'finite': finite, 'max_abs': max_abs,
            'relative_l2': relative_l2, 'scaled_max': scaled_max,
            'passed': finite and relative_l2 <= .02 and scaled_max <= .10}


def compare_cache(actual, expected, snapshot=None):
    rows = []
    for index, (a, b) in enumerate(zip(actual.layers, expected.layers)):
        for name in FIELDS:
            x, y = getattr(a, name, None), getattr(b, name, None)
            if not isinstance(x, torch.Tensor): continue
            item = {'layer': index, 'name': name, **metrics(x, y)}
            if snapshot is not None and name in ('keys', 'values'):
                old = getattr(snapshot['cache'].layers[index], name)
                item['old_history_exact'] = torch.equal(x[..., :old.shape[-2], :], old)
                item['new_suffix'] = metrics(x[..., old.shape[-2]:, :], y[..., old.shape[-2]:, :])
                item['passed'] &= item['old_history_exact'] and item['new_suffix']['passed']
            rows.append(item)
    return {'passed': all(x['passed'] for x in rows), 'exact': all(x['exact'] for x in rows), 'tensors': rows}


def load_model():
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        'model/qwen3.5-27b', dtype=torch.bfloat16, attn_implementation='sdpa',
        device_map={'': 'npu:0'}).eval()
    def rms(self, x):
        return torch_npu.npu_rms_norm(x, (1.0+self.weight).to(x.dtype), self.eps)[0]
    def gated(self, hidden_states, gate=None):
        normalized = torch_npu.npu_rms_norm(hidden_states, self.weight, self.variance_epsilon)[0]
        return torch_npu.npu_swiglu(torch.cat((gate, normalized), dim=-1))
    for module in model.modules():
        if isinstance(module, hf.Qwen3_5RMSNorm): module.forward = rms.__get__(module, type(module))
        if isinstance(module, hf.Qwen3_5RMSNormGated): module.forward = gated.__get__(module, type(module))
        if isinstance(module, hf.Qwen3_5GatedDeltaNet): module.recurrent_gated_delta_rule = recurrent_single_step
    return model


def capture(model):
    if (ROOT/'snapshots.pt').exists():
        raise RuntimeError('capture already exists; preserve it and use the following stages')
    base = {str(x['id']): x for x in map(json.loads, BASE.read_text().splitlines()) if x.get('kind') == 'task'}
    data = {str(x['id']): x for x in map(json.loads, Path('data/gsm8k_128.jsonl').read_text().splitlines())}
    tok = AutoTokenizer.from_pretrained('model/qwen3.5-27b', padding_side='left')
    prompts, messages = [], []
    for task in IDS:
        msg = [{'role':'system','content':CONCISE_SYSTEM}, {'role':'user','content':data[str(task)]['question']}]
        messages.append(msg)
        prompts.append(tok.apply_chat_template(msg, tools=TOOLS, tokenize=False, add_generation_prompt=True, enable_thinking=False))
    inputs = tok(prompts, add_special_tokens=False, padding=True, return_tensors='pt').to('npu')
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    snapshots, call = {}, 0
    def hook(module, args, kwargs, output):
        nonlocal call
        cache = output.past_key_values
        for row, token in enumerate(output.logits[:, -1].argmax(-1).tolist()):
            if row in snapshots or token not in eos: continue
            mask = inputs['attention_mask'][row].tolist() + [1]*call
            assert len(mask) == cache.get_seq_length()
            snapshots[row] = {'cache': row_cache_to_cpu(cache, row), 'mask': mask,
                              'logical_prompt': inputs['input_ids'][row][inputs['attention_mask'][row].bool()].tolist()}
        call += 1
    handle = model.register_forward_hook(hook, with_kwargs=True)
    out = model.generate(**inputs, max_new_tokens=512, do_sample=False, use_cache=True, return_dict_in_generate=True)
    handle.remove()
    assert len(snapshots) == 16
    result = {'stage':'capture', 'first_turn':[], 'baseline_sha256':hashlib.sha256(BASE.read_bytes()).hexdigest()}
    ordered = []
    for row, task in enumerate(IDS):
        tokens = out.sequences[row, inputs['input_ids'].shape[1]:].tolist()
        stop = next((i+1 for i,t in enumerate(tokens) if t in eos), len(tokens))
        tokens = tokens[:stop]
        assert tokens == base[str(task)]['rounds'][0]['tokens'], f'first-turn drift {task}'
        snap = snapshots[row]
        snap['consumed'] = snap.pop('logical_prompt') + tokens[:-1]
        msg = messages[row] + [{'role':'assistant','content':tok.decode(tokens, skip_special_tokens=True)}]
        tool = execute(parse_call(msg[-1]['content']))
        assert tool == base[str(task)]['rounds'][0]['tool_output']
        msg.append({'role':'tool','name':'python','content':tool})
        full = tok(tok.apply_chat_template(msg, tools=TOOLS, tokenize=False,
                    add_generation_prompt=True, enable_thinking=False), add_special_tokens=False)['input_ids']
        assert full[:len(snap['consumed'])] == snap['consumed']
        snap.update(task=task, suffix=full[len(snap['consumed']):],
                    expected_second=base[str(task)]['rounds'][1]['tokens'], first_tokens=tokens,
                    tool_output=tool)
        assert sum(snap['mask']) == len(snap['consumed'])
        ordered.append(snap)
        result['first_turn'].append({'task':task, 'exact':True, 'logical_boundary':len(snap['consumed']),
                                     'physical_boundary':len(snap['mask']), 'suffix_length':len(snap['suffix'])})
    ROOT.mkdir(parents=True, exist_ok=True)
    torch.save(ordered, ROOT/'snapshots.pt')
    result.update(passed=True, snapshot_bytes=(ROOT/'snapshots.pt').stat().st_size,
                  snapshot_sha256=hashlib.sha256((ROOT/'snapshots.pt').read_bytes()).hexdigest())
    write(ROOT/'capture.json', result)
    return result


def operator_gate(device):
    torch.manual_seed(168)
    dtype = torch.float32 if device == 'cpu' else torch.bfloat16
    impl = BatchedHistory()
    impl.lengths = torch.tensor([5,2], device=device)
    q,k,v = [torch.randn(2,5,3,8, device=device, dtype=dtype) for _ in range(3)]
    g = -torch.rand(2,5,3, device=device)
    beta = torch.rand(2,5,3, device=device, dtype=dtype)
    initial = torch.randn(2,3,8,8, device=device, dtype=dtype)
    output, state = impl.recurrent(q,k,v,g,beta,initial,True,True)
    rows = []
    for row, length in enumerate([5,2]):
        reference = initial[row:row+1].clone()
        outs = []
        for t in range(length):
            y, reference = hf.torch_recurrent_gated_delta_rule(
                q[row:row+1,t:t+1],k[row:row+1,t:t+1],v[row:row+1,t:t+1],
                g[row:row+1,t:t+1],beta[row:row+1,t:t+1],reference,True,True)
            reference = reference.to(dtype)
            outs.append(y)
        rows.append({'row':row, 'output_close':torch.allclose(output[row:row+1,:length],torch.cat(outs,1),atol=1e-6,rtol=1e-5),
                     'state_exact':torch.equal(state[row:row+1],reference)})
    x = torch.randn(2,6,5,device=device,dtype=dtype)
    conv_initial = torch.randn(2,6,4,device=device,dtype=dtype)
    weight = torch.randn(6,4,device=device,dtype=dtype)
    conv_state = conv_initial.clone()
    conv_out = impl.conv(x,conv_state,weight)
    for row,length in enumerate([5,2]):
        ref = conv_initial[row:row+1].clone()
        outs=[]
        for t in range(length):
            outs.append(hf.torch_causal_conv1d_update(x[row:row+1,:,t:t+1],ref,weight))
        rows[row].update(conv_exact=torch.equal(conv_out[row:row+1,:,:length],torch.cat(outs,-1)),
                         conv_state_exact=torch.equal(conv_state[row:row+1],ref))
    return {'device':device,'dtype':str(dtype),'rows':rows,
            'passed':all(all(v for k,v in row.items() if k!='row') for row in rows)}


def audited_forward(model, **kwargs):
    events=[]
    def hook(module, args, kw):
        value = kw.get('input_ids', kw.get('hidden_states', args[0] if args else None))
        events.append({'module':type(module).__name__, 'shape':list(value.shape)})
    handles=[model.register_forward_pre_hook(hook, with_kwargs=True)]
    for module in model.modules():
        if isinstance(module, hf.Qwen3_5DecoderLayer):
            handles.append(module.register_forward_pre_hook(hook, with_kwargs=True))
    start=time.perf_counter()
    try:
        output = model(**kwargs)
        torch.npu.synchronize()
    finally:
        for handle in handles: handle.remove()
    return output, {'wall_seconds_instrumented':time.perf_counter()-start,'events':events,
                    'model_forwards':sum(x['module']=='Qwen3_5ForConditionalGeneration' for x in events),
                    'decoder_calls':sum(x['module']=='Qwen3_5DecoderLayer' for x in events)}


def run_append(model, impl, snapshots, suffixes, pad_token=0, cache_observer=None):
    cache, ids, mask, pos, p = pack(snapshots, suffixes, pad_token=pad_token)
    lengths = list(map(len,suffixes))
    impl.lengths = ids.new_tensor(lengths)
    impl.events = []
    # Validate packing before any operation can mutate the packed cache.
    transport=[]
    for row,(snap,length) in enumerate(zip(snapshots,lengths)):
        width=len(snap['mask'])
        exact=True
        for index,layer in enumerate(cache.layers):
            for name in FIELDS:
                value=getattr(layer,name,None)
                if not isinstance(value,torch.Tensor): continue
                got=value[row:row+1].cpu()
                if name in ('keys','values'): got=got[...,p-width:p,:]
                exact &= torch.equal(got,getattr(snap['cache'].layers[index],name))
        transport.append({'task':snap['task'],'packed_exact':exact,'position_start':int(pos[row,0]),
                          'position_end':int(pos[row,length-1]),'valid_history':int(mask[row,:p].sum()),
                          'valid_final':int(mask[row].sum()), 'physical_history':width,
                          'logical_final':len(snap['consumed'])+length,
                          'mask_exact':mask[row,p-width:p].tolist()==snap['mask']})
    cleanup = cache_observer(cache) if cache_observer is not None else None
    try:
        out,audit = audited_forward(model,input_ids=ids,attention_mask=mask,position_ids=pos,
            past_key_values=cache,use_cache=True,return_dict=True,logits_to_keep=0)
    finally:
        if cleanup is not None:
            cleanup()
    logits=[out.logits[row,:length].detach().cpu() for row,length in enumerate(lengths)]
    caches=unpack(out.past_key_values,snapshots,suffixes,p)
    audit['gdn_events']=list(impl.events)
    audit['single_batch_call']=(audit['model_forwards']==1 and audit['decoder_calls']==64
                               and all(x['shape'][0]==len(snapshots) for x in audit['events']))
    del out,cache
    return logits,caches,transport,audit


def reference(model,snap,suffix):
    cache=clone_cache(snap['cache'],'npu:0')
    logits=[]
    for t,token in enumerate(suffix):
        ids=torch.tensor([[token]],device='npu:0')
        mask=ids.new_tensor([snap['mask']+[1]*(t+1)])
        out=model(input_ids=ids,attention_mask=mask,
                  position_ids=ids.new_tensor([[len(snap['consumed'])+t]]),
                  past_key_values=cache,use_cache=True,return_dict=True,logits_to_keep=1)
        cache=out.past_key_values
        logits.append(out.logits[0,0].detach().cpu())
    return torch.stack(logits),clone_cache(cache)


def gate(model,stage):
    require_stage(stage)
    snapshots=torch.load(ROOT/'snapshots.pt',map_location='cpu',weights_only=False)
    if stage!='b16': snapshots=snapshots[:2]
    suffixes=[s['suffix'][:1] if stage=='b2_one' else s['suffix'] for s in snapshots]
    if stage=='b2_suffix': assert list(map(len,suffixes))==[50,22]
    before=[digest(s['cache']) for s in snapshots]
    result={'stage':stage,'contract_sha256':hashlib.sha256(Path('results/e168_contract.md').read_bytes()).hexdigest(),
            'batch':len(snapshots),'suffix_lengths':list(map(len,suffixes)),
            'operator_gates':[operator_gate('cpu'),operator_gate('npu:0')], 'tasks':[]}
    if not all(x['passed'] for x in result['operator_gates']):
        result.update(semantic_passed=False,stopped='operator_gate')
        write(ROOT/(stage+'.json'),result)
        return result
    impl=BatchedHistory()
    source=impl.install(model)
    result['candidate_forward_sha256']=hashlib.sha256(source.encode()).hexdigest()
    logits,caches,transport,audit=run_append(model,impl,snapshots,suffixes)
    result.update(transport=transport,forward_audit=audit)
    # Change row 0 only, maintaining all tensor shapes. All other rows must stay exact.
    altered=copy.deepcopy(suffixes)
    altered[0][0]=(altered[0][0]+1)%model.config.text_config.vocab_size
    other_logits,other_caches,_,isolation_audit=run_append(model,impl,snapshots,altered)
    result['isolation']={'audit':isolation_audit, 'unaffected_rows_exact':all(
        torch.equal(logits[row],other_logits[row]) and compare_cache(caches[row],other_caches[row])['exact']
        for row in range(1,len(snapshots))),
        'changed_row_observed':not torch.equal(logits[0],other_logits[0])}
    del other_logits,other_caches
    if len(set(map(len,suffixes)))>1:
        pad_logits,pad_caches,_,pad_audit=run_append(model,impl,snapshots,suffixes,pad_token=123)
        result['inactive_pad']={'audit':pad_audit, 'valid_outputs_and_final_states_exact':all(
            torch.equal(a,b) and compare_cache(c,d)['exact']
            for a,b,c,d in zip(logits,pad_logits,caches,pad_caches))}
        del pad_logits,pad_caches
    impl.restore()
    for snap,suffix,actual,cache in zip(snapshots,suffixes,logits,caches):
        ref,ref_cache=reference(model,snap,suffix)
        logit_metric=metrics(actual,ref)
        states=compare_cache(cache,ref_cache,snap)
        item={'task':snap['task'],'logits':logit_metric,'cache':states,
              'teacher_argmax_exact':torch.equal(actual.argmax(-1),ref.argmax(-1)),
              'candidate_argmax':actual.argmax(-1).tolist(),'reference_argmax':ref.argmax(-1).tolist(),
              'physical_final':int(cache.get_seq_length()),
              'length_exact':int(cache.get_seq_length())==len(snap['mask'])+len(suffix),
              'passed':logit_metric['passed'] and states['passed']}
        result['tasks'].append(item)
        write(ROOT/(stage+'.partial.json'),result)
        print(json.dumps({'stage':stage,'task':snap['task'],'logits':logit_metric,
                          'state_pass':states['passed'],'teacher_argmax_exact':item['teacher_argmax_exact']}),flush=True)
        del ref,ref_cache
    result['source_snapshots_unchanged']=[digest(s['cache']) for s in snapshots]==before
    result['semantic_passed']=(all(x['passed'] and x['length_exact'] for x in result['tasks'])
        and audit['single_batch_call'] and result['isolation']['unaffected_rows_exact']
        and result['isolation']['changed_row_observed'] and result['source_snapshots_unchanged']
        and result.get('inactive_pad',{}).get('valid_outputs_and_final_states_exact',True)
        and all(x['packed_exact'] and x['mask_exact'] and x['valid_final']==x['logical_final']
                and x['position_start']==len(s['consumed']) and x['position_end']==x['logical_final']-1
                for x,s in zip(transport,snapshots)))
    result['trajectory_gate']='not_run: capability gate only; teacher-forced argmax is not a generated trajectory'
    result['performance_measured']=False
    torch.save({'logits':logits,'caches':caches},ROOT/(stage+'_outputs.pt'))
    return result


def trajectory(model):
    require_stage('trajectory')
    snapshots=torch.load(ROOT/'snapshots.pt',map_location='cpu',weights_only=False)
    suffixes=[s['suffix'] for s in snapshots]
    cache,ids,mask,pos,p=pack(snapshots,suffixes)
    impl=BatchedHistory()
    impl.install(model)
    impl.lengths=ids.new_tensor(list(map(len,suffixes)))
    out,audit=audited_forward(model,input_ids=ids,attention_mask=mask,position_ids=pos,
        past_key_values=cache,use_cache=True,return_dict=True,logits_to_keep=0)
    next_ids=torch.stack([out.logits[row,len(suffix)-1].argmax(-1) for row,suffix in enumerate(suffixes)])
    cache=out.past_key_values
    del out
    eos=model.generation_config.eos_token_id
    eos=set(eos if isinstance(eos,list) else [eos])
    active=[True]*len(snapshots)
    generated=[[] for _ in snapshots]
    result={'stage':'trajectory','sampling':'greedy; max_new_tokens=12; no teacher forcing',
            'first_turn_exact':json.loads((ROOT/'capture.json').read_text())['passed'],
            'forward_audits':[audit],'first_difference':None}
    for step in range(12):
        chosen=next_ids.tolist()
        for row,(snap,token) in enumerate(zip(snapshots,chosen)):
            if not active[row]: continue
            generated[row].append(token)
            expected=snap['expected_second']
            if step>=len(expected) or token!=expected[step]:
                if result['first_difference'] is None:
                    result['first_difference']={'task':snap['task'],'offset':step,'actual':token,
                        'expected':expected[step] if step<len(expected) else None}
            if token in eos: active[row]=False
        if result['first_difference'] is not None or not any(active) or step==11: break
        append_ids=next_ids[:,None]
        live=ids.new_tensor(active,dtype=torch.long)
        mask=torch.cat((mask,live[:,None]),dim=-1)
        positions=ids.new_tensor([[len(s['consumed'])+len(s['suffix'])+step] for s in snapshots])
        impl.lengths=live
        out,audit=audited_forward(model,input_ids=append_ids,attention_mask=mask,position_ids=positions,
            past_key_values=cache,use_cache=True,return_dict=True,logits_to_keep=1)
        cache=out.past_key_values
        next_ids=out.logits[:,-1].argmax(-1)
        del out
        result['forward_audits'].append(audit)
    impl.restore()
    result['tasks']=[{'task':s['task'],'generated':tokens,'expected':s['expected_second'],
                      'exact':tokens==s['expected_second'],'tool_output':s['tool_output']}
                     for s,tokens in zip(snapshots,generated)]
    result['passed']=result['first_turn_exact'] and all(x['exact'] for x in result['tasks'])
    result['all_calls_batched']=all(a['model_forwards']==1 and a['decoder_calls']==64
        and all(e['shape'][0]==16 for e in a['events']) for a in result['forward_audits'])
    result['passed'] &= result['all_calls_batched']
    result['performance_measured']=False
    return result


def diagnose(model):
    snapshots=torch.load(ROOT/'snapshots.pt',map_location='cpu',weights_only=False)[:2]
    suffixes=[s['suffix'] for s in snapshots]
    impl=BatchedHistory()
    impl.install(model)
    impl.capture_indices={0,47}
    first=next(m for m in model.modules() if isinstance(m,hf.Qwen3_5GatedDeltaNet))
    projection={}
    def capture_projection(module,args,output):
        projection['input']=args[0].detach().cpu()
        projection['output']=output.detach().cpu()
    handle=first.in_proj_qkv.register_forward_hook(capture_projection)
    logits,caches,transport,audit=run_append(model,impl,snapshots,suffixes)
    handle.remove()
    result={'stage':'diagnose','scope':'same failing B2 input, no expansion',
            'operator_rows':[],'model_audit':audit}
    for index,c in impl.captures.items():
        for row,length in enumerate(map(len,suffixes)):
            x,initial,weight,bias,activation=c['conv_inputs']
            state=initial[row:row+1].to('npu:0').clone()
            weight=weight.to('npu:0')
            bias=None if bias is None else bias.to('npu:0')
            outputs=[]
            for t in range(length):
                outputs.append(hf.torch_causal_conv1d_update(x[row:row+1,:,t:t+1].to('npu:0'),state,weight,bias,activation))
            conv_output=torch.cat(outputs,-1).cpu()
            conv_final=state.cpu()
            q,k,v,g,beta,initial=c['recurrent_inputs']
            state=initial[row:row+1].to('npu:0').clone()
            outputs=[]
            for t in range(length):
                inputs=[a[row:row+1,t:t+1].to('npu:0') for a in (q,k,v,g,beta)]
                y,state=hf.torch_recurrent_gated_delta_rule(*inputs,state,True,True)
                state=state.to(initial.dtype)
                outputs.append(y)
            recurrent_output=torch.cat(outputs,1).cpu()
            item={'gdn_ordinal':index,'task':snapshots[row]['task'],
                  'conv_output_exact':torch.equal(conv_output,c['conv_output'][row:row+1,:,:length]),
                  'conv_state_exact':torch.equal(conv_final,c['conv_final'][row:row+1]),
                  'recurrent_output_exact':torch.equal(recurrent_output,c['recurrent_output'][row:row+1,:length]),
                  'recurrent_state_exact':torch.equal(state.cpu(),c['recurrent_final'][row:row+1])}
            result['operator_rows'].append(item)
    # Hold the first layer's normalized input fixed: change only Linear's M shape.
    token_outputs=[]
    for row,length in enumerate(map(len,suffixes)):
        outputs=[]
        for t in range(length):
            outputs.append(first.in_proj_qkv(projection['input'][row:row+1,t:t+1].to('npu:0')).cpu())
        token_outputs.append(torch.cat(outputs,1))
    result['first_qkv_projection_same_input']=[{'task':s['task'],
        **metrics(projection['output'][row:row+1,:len(suffix)],token_outputs[row])}
        for row,(s,suffix) in enumerate(zip(snapshots,suffixes))]
    result['operator_same_input_exact']=all(all(v for k,v in x.items() if k.endswith('_exact'))
        for x in result['operator_rows'])
    torch.save({'gdn':impl.captures,'first_qkv_projection':projection},ROOT/'diagnostic_tensors.pt')
    impl.restore()
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--stage',choices=['capture','b2_one','b2_suffix','b16','trajectory','diagnose'],required=True)
    args=ap.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    require_stage(args.stage)
    torch.set_num_threads(8)
    start=time.perf_counter()
    model=load_model()
    torch.npu.reset_peak_memory_stats()
    with torch.inference_mode():
        if args.stage=='capture': result=capture(model)
        elif args.stage=='trajectory': result=trajectory(model)
        elif args.stage=='diagnose': result=diagnose(model)
        else: result=gate(model,args.stage)
    result.update(wall_seconds_instrumented=time.perf_counter()-start,
                  peak_allocated_bytes=int(torch.npu.max_memory_allocated()),
                  peak_reserved_bytes=int(torch.npu.max_memory_reserved()))
    result['source_hashes']={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
        ['scripts/e168_batched_history.py','scripts/e168_ragged_history_gate.py']}
    write(ROOT/(args.stage+'.json'),result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('tasks','forward_audit','isolation','inactive_pad')},indent=2),flush=True)


if __name__=='__main__': main()
