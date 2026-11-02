"""Independently reopen E251 full128 raw trials and recompute acceptance."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    root=Path('results/e251/abba_verified');decision=json.loads((root/'decision.json').read_text())
    checks={'four_trials':len(decision['trials'])==4,'device0':decision['physical_device']=='0',
        'source_hashes':all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in decision['source_hashes'].items()),
        'first16_gate':json.loads(Path('results/e250/abba/audit.json').read_text())['performance_passed']}
    trials=[]
    for i,kind in enumerate(('baseline','candidate','candidate','baseline')):
        path=root/f'trial{i}_{kind}'
        if not (path/'gate.json').exists():checks[f'trial{i}/exists']=False;continue
        command=[sys.executable,'scripts/audit_e251.py','--output-dir',str(path),'--performance-run','--prompt-lengths']
        with (root/f'trial{i}_{kind}.independent_audit.log').open('w') as log:
            code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT).returncode
        data=json.loads((path/'gate.json').read_text());audit=json.loads((path/'audit.json').read_text())
        record=decision['trials'][i]
        checks[f'trial{i}/raw_quality_memory']=code==0 and audit['verified']
        checks[f'trial{i}/baseline_selection']=data['baseline']==(kind=='baseline') and ('--baseline' in record['command'])==(kind=='baseline')
        checks[f'trial{i}/driver_matches']=record['kind']==kind and record['index']==i and record['exit_code']==0 and record['audit_exit_code']==0 and record['seconds']==data['makespan_seconds'] and '--performance-run' in record['command'] and 'scripts/e251_full_workflow.py' in record['command'] and record['manifest_sha256']==hashlib.sha256((path/'gate.json').read_bytes()).hexdigest()
        checks[f'trial{i}/workload_matches']=record['token_count']==audit['token_count'] and record['tool_count']==audit['tool_count'] and record['strict_exact']==audit['strict_exact'] and record['score_v3']==audit['scorer_v3_correct'] and record['failed_ids']==audit['failed_ids']
        checks[f'trial{i}/tool_time_scope']=0<=data['tool_seconds']<data['makespan_seconds']
        trials.append({'kind':kind,'seconds':data['makespan_seconds'],'load_seconds':data['load_seconds'],
            'score_v3':audit['scorer_v3_correct'],'failed_ids':audit['failed_ids'],'token_count':audit['token_count'],
            'tool_count':audit['tool_count'],'tool_seconds':audit['tool_seconds'],'strict_exact':audit['strict_exact'],
            'allocated':data['peak_allocated_bytes'],'reserved':data['peak_reserved_bytes'],
            'manifest_sha256':audit['manifest_sha256'],'raw_artifact_hashes':audit['artifact_hashes'],
            'prompt_lengths':audit['prompt_lengths']})
    gain=None;pairs=[];ranges=[];passed=False
    if len(trials)==4:
        a,b,c,d=[t['seconds'] for t in trials]
        gain=(a+d-b-c)/(a+d);pairs=[(a-b)/a,(d-c)/d];ranges=[abs(a-d)/min(a,d),abs(b-c)/min(b,c)]
        passed=gain>=.02 and min(pairs)>0 and gain>max(ranges)
        checks['decision_matches']=decision['passed']==passed and decision['mean_gain_fraction']==gain and decision['paired_gains']==pairs and decision['range_fractions']==ranges
        checks['workload_flag']=decision['workload_identical']==all(t['strict_exact'] for t in trials)
        checks['means']=decision['baseline_mean_seconds']==(a+d)/2 and decision['candidate_mean_seconds']==(b+c)/2
    result={'verified':all(checks.values()),'checks':checks,'performance_passed':passed and all(checks.values()),
        'mean_gain_fraction':gain,'paired_gains':pairs,'range_fractions':ranges,'trials':trials,
        'deployment_accepted':False,'scorer':'v3'}
    (root/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='trials'},indent=2))
    return 0 if result['verified'] else 1


if __name__=='__main__':raise SystemExit(main())
