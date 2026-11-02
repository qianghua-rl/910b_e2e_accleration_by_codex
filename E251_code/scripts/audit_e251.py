"""Independent raw-workflow audit for E250 expansion, not a performance decision."""
import argparse
import hashlib
import json
from pathlib import Path
from scoring_v3 import answer, correct


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--count', type=int, choices=[32, 128], default=128)
    parser.add_argument('--performance-run', action='store_true')
    parser.add_argument('--prompt-lengths', action='store_true')
    args = parser.parse_args()
    root = Path(args.output_dir)
    manifest = json.loads((root/'gate.json').read_text())
    source = Path('results/phase2_concise_sdpa_batch16_final12_length128.jsonl')
    reference = [r for r in map(json.loads, source.read_text().splitlines()) if r.get('kind') == 'task'][:args.count]
    checks = {'batch_count': len(manifest['batches']) == args.count//16,
              'device': manifest['physical_device'] == '0',
              'timing_mode': manifest['performance_measured'] == args.performance_run,
              'sources': bool(manifest.get('source_hashes')) and all(digest(p) == h for p, h in manifest.get('source_hashes', {}).items()),
              'memory': 0 < manifest['peak_allocated_bytes'] <= 62094008832,
              'reserved': manifest['peak_reserved_bytes'] >= manifest['peak_allocated_bytes']}
    actual, hashes, routes = [], {}, []
    for i, batch in enumerate(manifest['batches']):
        path = Path(batch['output'])
        data = json.loads(path.read_text())
        transcript = Path(str(path)+'.workflow.jsonl')
        raw = [json.loads(line) for line in transcript.read_text().splitlines()]
        tasks = [r for r in raw if r.get('kind') == 'task']
        actual.extend(tasks)
        hashes[str(path)], hashes[str(transcript)] = digest(path), digest(transcript)
        checks[f'batch{i}/raw_matches_summary'] = len(tasks) == len(data['tasks']) == 16 and all(
            a['id'] == b['id'] and a['rounds'] == b['rounds'] and a['prediction'] == b['prediction'] for a, b in zip(tasks, data['tasks']))
        checks[f'batch{i}/timer'] = next(r for r in raw if r.get('kind') == 'summary')['makespan_seconds'] == data['workflow_seconds']
        checks[f'batch{i}/memory'] = 0 < data['peak_allocated_bytes'] <= 62094008832 and data['peak_reserved_bytes'] >= data['peak_allocated_bytes']
        checks[f'batch{i}/device_mode'] = data['resource_before']['physical_device'] == '0' and data['performance_measured'] == args.performance_run
        checks[f'batch{i}/complete'] = batch['exit_code'] == 0 and batch['index'] == i
        routes.append({'index': i, 'route': data['route'], 'mismatch_ids': data['mismatch_ids']})
    checks['order_count'] = len(actual) == args.count and [r['id'] for r in actual] == [r['id'] for r in reference]
    strict = len(actual) == len(reference) and all(
        len(a['rounds']) == len(b['rounds']) and a['prediction'] == b['prediction'] and all(
            all(x.get(k) == y.get(k) for k in ('turn','tokens','text','code','tool_output'))
            for x,y in zip(a['rounds'],b['rounds'])) for a,b in zip(actual,reference))
    score = sum(correct(answer(t['rounds'][-1]['text']),t['expected']) for t in actual)
    failures = [t['id'] for t in actual if not correct(answer(t['rounds'][-1]['text']),t['expected'])]
    expected_failures = [t['id'] for t in reference if not correct(answer(t['rounds'][-1]['text']),t['expected'])]
    checks['score_failure_set'] = score >= args.count-len(expected_failures) and set(failures) <= set(expected_failures) and manifest['scorer_v3_correct'] == score and manifest['failed_ids'] == failures
    checks['first_round_tools_exact'] = len(actual)==len(reference) and all(a['id']==b['id'] and all(a['rounds'][0].get(k)==b['rounds'][0].get(k) for k in ('tokens','text','code','tool_output')) for a,b in zip(actual,reference))
    checks['completed_rounds'] = len(actual)==len(reference) and all(len(a['rounds'])==len(b['rounds']) and 0<len(a['rounds'][-1]['tokens'])<=12 for a,b in zip(actual,reference))
    checks['predictions_match_text'] = all(t['prediction']==answer(t['rounds'][-1]['text']) for t in actual)
    checks['baseline_exact'] = not manifest['baseline'] or strict
    checks['candidate_script_identity'] = (manifest['baseline'] or all(
        manifest.get('source_hashes',{}).get(name)==digest(name) for name in ('scripts/e250_chunk_history.py','scripts/e248_vector_history_conv.py')))
    checks['runner_script_identity'] = manifest.get('source_hashes',{}).get('scripts/e251_full_workflow.py')==digest('scripts/e251_full_workflow.py')
    tokens = sum(len(r['tokens']) for t in actual for r in t['rounds'])
    reference_tokens = sum(len(r['tokens']) for t in reference for r in t['rounds'])
    tools = sum('tool_output' in r for t in actual for r in t['rounds'])
    checks['tools'] = tools == args.count
    checks['tool_timing'] = ([r['id'] for r in manifest['per_task_tool_seconds']]==[r['id'] for r in reference] and all(r['seconds']>=0 for r in manifest['per_task_tool_seconds']) and abs(sum(r['seconds'] for r in manifest['per_task_tool_seconds'])-manifest['tool_seconds'])<1e-9)
    checks['fallbacks'] = ([r for r in routes if r['route'] == 'fresh_batch'] == ([{'index':2,'route':'fresh_batch','mismatch_ids':[99]}, {'index':3,'route':'fresh_batch','mismatch_ids':[785]}] if args.count==128 else []))
    result = {'verified': all(checks.values()), 'checks': checks, 'strict_exact':strict, 'reference_token_count':reference_tokens, 'baseline':manifest['baseline'], 'task_count': len(actual), 'scorer_v3_correct':score,
              'failed_ids':failures, 'token_count':tokens, 'tool_count':tools, 'tool_seconds':manifest['tool_seconds'], 'routes':routes,
              'artifact_hashes':hashes, 'reference_sha256':digest(source), 'manifest_sha256':digest(root/'gate.json'), 'peak_allocated_bytes':manifest['peak_allocated_bytes'], 'peak_reserved_bytes':manifest['peak_reserved_bytes'],
              'performance_measured':args.performance_run, 'performance_accepted':False}
    if args.prompt_lengths:
        from transformers import AutoTokenizer
        from agent_protocol import CONCISE_SYSTEM,TOOLS
        tok=AutoTokenizer.from_pretrained('model/qwen3.5-27b')
        data={r['id']:r for r in map(json.loads,Path('data/gsm8k_128.jsonl').read_text().splitlines())}
        prompts=[]
        for task in actual:
            messages=[{'role':'system','content':CONCISE_SYSTEM},{'role':'user','content':data[task['id']]['question']}]
            def count():
                text=tok.apply_chat_template(messages,tools=TOOLS,tokenize=False,add_generation_prompt=True,enable_thinking=False)
                return len(tok(text,add_special_tokens=False)['input_ids'])
            first=count()
            messages.extend([{'role':'assistant','content':task['rounds'][0]['text']},{'role':'tool','name':'python','content':task['rounds'][0]['tool_output']}])
            prompts.append({'id':task['id'],'first':first,'second':count()})
        result['prompt_lengths']=prompts
    (root/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return 0 if result['verified'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
