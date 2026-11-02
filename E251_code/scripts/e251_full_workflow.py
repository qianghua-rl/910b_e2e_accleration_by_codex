"""One resident model, eight original B16 batches, explicit whole-batch fresh."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import torch
from transformers import AutoTokenizer
import e176_workflow_bench as workflow
from e250_chunk_history import ChunkHistory
from e168_ragged_history_gate import load_model,write


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True)
    p.add_argument('--performance-run',action='store_true')
    p.add_argument('--smoke32',action='store_true')
    p.add_argument('--baseline',action='store_true')
    args=p.parse_args()
    if not args.baseline:workflow.HistoryRuntime=ChunkHistory
    root=Path(args.output_dir);root.mkdir(parents=True,exist_ok=True)
    if (root/'gate.json').exists():raise SystemExit('preserve prior run')
    if __import__('os').environ.get('ASCEND_RT_VISIBLE_DEVICES')!='0':raise SystemExit('physical0 required')
    resource=workflow.preflight()
    if not resource['resident_model_benchmark_ready']:raise SystemExit('insufficient resident memory')
    first=json.loads(Path('results/e250/abba/audit.json').read_text())
    if not first['verified'] or not first['performance_passed']:raise SystemExit('first16 performance prerequisite failed')
    if not args.smoke32:
        smoke=json.loads(Path('results/e251/smoke32/audit.json').read_text())
        if not smoke['verified']:raise SystemExit('32 task prerequisite failed')
    if args.performance_run:
        full_gate=Path('results/e251/gate_verified/audit.json')
        if not full_gate.exists() or not json.loads(full_gate.read_text())['verified']:
            raise SystemExit('full128 correctness prerequisite failed')
    torch.set_num_threads(8);torch.manual_seed(42)
    start=time.perf_counter()
    model=load_model();tok=AutoTokenizer.from_pretrained('model/qwen3.5-27b',padding_side='left')
    torch.npu.synchronize();load_seconds=time.perf_counter()-start
    rows=[json.loads(s) for s in Path('results/phase2_concise_sdpa_batch16_final12_length128.jsonl').read_text().splitlines()]
    tasks=[r for r in rows if r.get('kind')=='task']
    assert len(tasks)==128
    if args.smoke32:
        if args.performance_run:raise SystemExit('smoke32 is correctness only')
        tasks=tasks[:32]
    count=len(tasks)
    result={'physical_device':resource['physical_device'],'load_seconds':load_seconds,'resource':resource,
        'baseline':args.baseline,'batches':[],'source_hashes':{name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in ('scripts/e251_full_workflow.py','scripts/audit_e251.py','scripts/e250_chunk_history.py','scripts/e248_vector_history_conv.py','scripts/e176_workflow_bench.py','scripts/e176_history_runtime.py','results/e251_contract.md')},'performance_measured':args.performance_run,'contract_sha256':hashlib.sha256(Path('results/e251_contract.md').read_bytes()).hexdigest()}
    tool_times=[]
    original_execute=workflow.execute
    def timed_execute(code):
        tool_start=time.perf_counter()
        try:return original_execute(code)
        finally:tool_times.append(time.perf_counter()-tool_start)
    workflow.execute=timed_execute
    torch.npu.reset_peak_memory_stats();start=time.perf_counter()
    for index in range(count//16):
        output=root/f'batch{index}.json'
        command=['--output',str(output),'--snapshot-device','npu','--correctness-only','--fresh-batch-on-mismatch','--defer-reference-check',
            '--task-ids',','.join(str(r['id']) for r in tasks[index*16:index*16+16])]
        if args.performance_run:
            command.remove('--correctness-only')
        code=workflow.run(command,shared_model=model,shared_tokenizer=tok)
        data=json.loads(output.read_text())
        result['batches'].append({'index':index,'output':str(output),'route':data['route'],
            'mismatch_ids':data['mismatch_ids'],'trajectory_passed':data['trajectory_passed'],
            'memory_passed':data['memory_passed'],'exit_code':code})
        if not args.performance_run:write(root/'progress.json',result)
        if code!=0:
            result.update(passed=False,status='stopped_first_trajectory_or_memory_failure')
            write(root/'gate.json',result);return code
    torch.npu.synchronize()
    total_seconds=time.perf_counter()-start
    workflow.execute=original_execute
    result['tool_seconds']=sum(tool_times)
    result['per_task_tool_seconds']=[{'id':task['id'],'seconds':elapsed} for task,elapsed in zip(tasks,tool_times)]
    # Reference checks and v3 scoring after the full timed interval.
    from scoring_v3 import answer,correct
    observed=[]
    for batch in result['batches']:
        data=json.loads(Path(batch['output']).read_text());observed.extend(data['tasks'])
    exact=all(a['id']==b['id'] and len(a['rounds'])==len(b['rounds']) and all(
        x['tokens']==y['tokens'] and x.get('tool_output')==y.get('tool_output')
        for x,y in zip(a['rounds'],b['rounds'])) for a,b in zip(observed,tasks)) and len(observed)==count
    score=sum(correct(answer(t['rounds'][-1]['text']),t['expected']) for t in observed)
    first_exact=len(observed)==count and all(a['id']==b['id'] and a['rounds'][0]['tokens']==b['rounds'][0]['tokens'] and a['rounds'][0].get('tool_output')==b['rounds'][0].get('tool_output') and len(a['rounds'])==len(b['rounds']) for a,b in zip(observed,tasks))
    failures=[t['id'] for t in observed if not correct(answer(t['rounds'][-1]['text']),t['expected'])]
    reference_failures=[t['id'] for t in tasks if not correct(answer(t['rounds'][-1]['text']),t['expected'])]
    quality=first_exact and set(failures)<=set(reference_failures) and score>=count-len(reference_failures)
    result.update(passed=quality and 0<int(torch.npu.max_memory_allocated())<=62094008832,status='smoke32_completed' if args.smoke32 else 'full128_completed',task_count=count,trajectory_passed=exact,
        scorer_v3_correct=score,failed_ids=[t['id'] for t in observed if not correct(answer(t['rounds'][-1]['text']),t['expected'])],
        makespan_seconds=total_seconds if args.performance_run else None,
        wall_seconds_correctness_only=total_seconds if not args.performance_run else None,
        peak_allocated_bytes=int(torch.npu.max_memory_allocated()),
        peak_reserved_bytes=int(torch.npu.max_memory_reserved()),deployment_accepted=False)
    write(root/'gate.json',result);print(json.dumps(result,indent=2));return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
