"""Isolated, alternating E124-A ND/NZ endpoint runs."""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--run', default='results/phase2_concise_sdpa_batch16_final12_length128.jsonl')
p.add_argument('--repeats', type=int, default=4)
p.add_argument('--output', default='results/e124_nz_e2e.jsonl')
p.add_argument('--max-new-tokens', type=int, default=16)
p.add_argument('--order-file', default='results/phase2_concise_sdpa_batch16_final32_128.jsonl')
p.add_argument('--reference-run', default='results/mlp_nz_baseline16.jsonl')
a = p.parse_args()
order = ['nd','nz','nz','nd','nd','nz','nz','nd'][:2*a.repeats]
out = Path(a.output); out.unlink(missing_ok=True)
baseline_tasks=[json.loads(x) for x in Path(a.reference_run).read_text().splitlines()]
task_ids=[str(x['id']) for x in baseline_tasks if x.get('kind')=='task'][:16]
if len(task_ids)!=16: raise SystemExit('E122 reference run must contain first-batch16 task records')
for trial, mode in enumerate(order):
    detail = f'results/e124_nz_trial_{trial}.json'
    detail = f'results/e124_nz_trial_{trial}.jsonl'
    cmd = [sys.executable,'scripts/agent_batch_bench.py','--output',detail,
           '--batch-size','16','--limit','128','--task-ids',','.join(task_ids),
           '--final-max-new-tokens','12','--fused-rmsnorm','--fused-gated-rmsnorm',
           '--single-step-recurrent','--attention','sdpa','--protocol','concise',
           '--schedule','generation_length','--order-file',a.order_file]
    if mode=='nz': cmd.append('--mlp-nz')
    started = time.perf_counter()
    env = dict(os.environ); env['OMP_NUM_THREADS'] = '8'
    if mode == 'nz':
        # Existing gate always runs reference then candidate; use candidate timing.
        pass
    proc = subprocess.run(cmd, text=True, capture_output=True, env=env)
    elapsed = time.perf_counter() - started
    records=[json.loads(x) for x in Path(detail).read_text().splitlines()] if Path(detail).exists() else []
    metadata=next((x for x in records if x.get('kind')=='metadata'),{})
    summary=next((x for x in records if x.get('kind')=='summary'),{})
    tasks=[x for x in records if x.get('kind')=='task']
    trajectory=[(x.get('id'),[(z.get('tokens'),z.get('code'),z.get('tool_output')) for z in x.get('rounds',[])]) for x in tasks]
    digest=hashlib.sha256(json.dumps(trajectory,sort_keys=True).encode()).hexdigest()
    selected=dict(makespan_seconds=summary.get('makespan_seconds'),peak_bytes=summary.get('peak_bytes'),
                  load_seconds=metadata.get('load_seconds'),nz_preparation=metadata.get('nz_preparation'),
                  task_count=len(tasks),trajectory_sha256=digest,
                  task_ids=[x.get('id') for x in tasks],correct_count=sum(bool(x.get('correct')) for x in tasks))
    row = dict(trial=trial, mode=mode, process_wall_seconds=elapsed,
               returncode=proc.returncode, selected=selected, detail=detail)
    with out.open('a') as f: f.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)
    if proc.returncode != 0: raise SystemExit(proc.stderr[-4000:])
