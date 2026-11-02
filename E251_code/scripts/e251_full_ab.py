"""Incremental full128 ABBA: same resident workflow, E180 vs E250 history."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


SOURCES=('scripts/e251_full_workflow.py','scripts/e251_full_ab.py','scripts/audit_e251.py','scripts/audit_e251_ab.py',
         'scripts/e250_chunk_history.py','scripts/e248_vector_history_conv.py',
         'scripts/e176_workflow_bench.py','scripts/e176_history_runtime.py','results/e251_contract.md')


def main():
    root=Path('results/e251/abba_verified');root.mkdir(parents=True,exist_ok=True)
    if list(root.iterdir()):raise SystemExit('preserve prior trials')
    if os.environ.get('ASCEND_RT_VISIBLE_DEVICES')!='0':raise SystemExit('physical0 required')
    gate=json.loads(Path('results/e251/gate_verified/audit.json').read_text())
    if not gate['verified']:raise SystemExit('full128 quality/memory gate failed')
    result={'order':['baseline','candidate','candidate','baseline'],'physical_device':'0','trials':[],
        'reference':'matched E180 snapshot workflow','historical_e100_seconds':265.030128,
        'source_hashes':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in SOURCES}}
    def save(name):
        (root/name).write_text(json.dumps(result,indent=2)+'\n')
    for i,kind in enumerate(result['order']):
        destination=root/f'trial{i}_{kind}'
        command=[sys.executable,'-u','scripts/e251_full_workflow.py','--output-dir',str(destination),'--performance-run']
        if kind=='baseline':command.append('--baseline')
        with (root/f'trial{i}_{kind}.log').open('w') as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            result['live']={'index':i,'pid':process.pid,'command':command};save('live.json')
            code=process.wait()
        trial={'index':i,'kind':kind,'exit_code':code,'command':command};result['trials'].append(trial)
        if code or not (destination/'gate.json').exists():
            result.update(passed=False,status='stopped_trial_failure');save('decision.json');return 1
        audit_command=[sys.executable,'scripts/audit_e251.py','--output-dir',str(destination),'--performance-run','--prompt-lengths']
        with (root/f'trial{i}_{kind}.audit.log').open('w') as log:
            reviewed=subprocess.run(audit_command,stdout=log,stderr=subprocess.STDOUT)
        data=json.loads((destination/'gate.json').read_text())
        audit=json.loads((destination/'audit.json').read_text()) if (destination/'audit.json').exists() else {}
        trial.update(seconds=data['makespan_seconds'],peak_allocated_bytes=data['peak_allocated_bytes'],
            peak_reserved_bytes=data['peak_reserved_bytes'],quality_passed=audit.get('verified',False),
            strict_exact=audit.get('strict_exact',False),token_count=audit.get('token_count'),
            tool_count=audit.get('tool_count'),score_v3=audit.get('scorer_v3_correct'),failed_ids=audit.get('failed_ids'),
            audit_exit_code=reviewed.returncode,audit_command=audit_command,
            manifest_sha256=hashlib.sha256((destination/'gate.json').read_bytes()).hexdigest())
        save('progress.json');print(json.dumps(trial),flush=True)
        if reviewed.returncode or not audit.get('verified') or not trial['seconds']:
            result.update(passed=False,status='stopped_raw_quality_memory_audit');save('decision.json');return 1
    a,b,c,d=[t['seconds'] for t in result['trials']]
    gain=(a+d-b-c)/(a+d);pairs=[(a-b)/a,(d-c)/d];ranges=[abs(a-d)/min(a,d),abs(b-c)/min(b,c)]
    passed=gain>=.02 and min(pairs)>0 and gain>max(ranges)
    result.update(passed=passed,status='full128_incremental_performance_passed' if passed else 'full128_performance_not_proven',
        mean_gain_fraction=gain,paired_gains=pairs,range_fractions=ranges,
        baseline_mean_seconds=(a+d)/2,candidate_mean_seconds=(b+c)/2,
        historical_e100_gain_fraction=1-(b+c)/2/265.030128,
        workload_identical=all(t['strict_exact'] for t in result['trials']),deployment_accepted=False)
    save('decision.json');print(json.dumps(result,indent=2));return 0 if passed else 1


if __name__=='__main__':raise SystemExit(main())
