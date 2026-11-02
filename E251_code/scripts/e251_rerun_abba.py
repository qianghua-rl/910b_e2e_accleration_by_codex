"""Repeat the frozen E251 full128 ABBA into a fresh evidence directory."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path('results/e251/abba_rerun_20261102')
SOURCES = (
    'scripts/e251_full_workflow.py', 'scripts/e251_full_ab.py',
    'scripts/audit_e251.py', 'scripts/audit_e251_ab.py',
    'scripts/e250_chunk_history.py', 'scripts/e248_vector_history_conv.py',
    'scripts/e176_workflow_bench.py', 'scripts/e176_history_runtime.py',
    'results/e251_contract.md',
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, data):
    (ROOT / name).write_text(json.dumps(data, indent=2) + '\n')


def main():
    if ROOT.exists() and any(ROOT.iterdir()):
        raise SystemExit(f'preserve prior rerun artifacts: {ROOT}')
    ROOT.mkdir(parents=True, exist_ok=True)
    if os.environ.get('ASCEND_RT_VISIBLE_DEVICES') != '0':
        raise SystemExit('physical NPU 0 required')
    gate = json.loads(Path('results/e251/gate_verified/audit.json').read_text())
    if not gate.get('verified'):
        raise SystemExit('frozen full128 correctness gate is not verified')
    data = {
        'order': ['baseline', 'candidate', 'candidate', 'baseline'],
        'physical_device': '0', 'trials': [],
        'reference': 'matched E180 snapshot workflow',
        'source_hashes': {path: digest(path) for path in SOURCES},
    }
    save('live.json', data)
    for index, kind in enumerate(data['order']):
        trial_root = ROOT / f'trial{index}_{kind}'
        command = [sys.executable, '-u', 'scripts/e251_full_workflow.py',
                   '--output-dir', str(trial_root), '--performance-run']
        if kind == 'baseline':
            command.append('--baseline')
        with (ROOT / f'trial{index}_{kind}.log').open('w') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            data['live'] = {'index': index, 'pid': process.pid, 'command': command}
            save('live.json', data)
            code = process.wait()
        record = {'index': index, 'kind': kind, 'exit_code': code,
                  'command': command}
        data['trials'].append(record)
        if code or not (trial_root / 'gate.json').exists():
            data.update(passed=False, status='stopped_trial_failure')
            save('decision.json', data)
            return 1
        audit_command = [sys.executable, 'scripts/audit_e251.py',
                         '--output-dir', str(trial_root), '--performance-run',
                         '--prompt-lengths']
        with (ROOT / f'trial{index}_{kind}.audit.log').open('w') as log:
            audit_code = subprocess.run(audit_command, stdout=log,
                                        stderr=subprocess.STDOUT).returncode
        manifest = json.loads((trial_root / 'gate.json').read_text())
        audit = json.loads((trial_root / 'audit.json').read_text())
        record.update(
            seconds=manifest['makespan_seconds'],
            load_seconds=manifest['load_seconds'],
            peak_allocated_bytes=manifest['peak_allocated_bytes'],
            peak_reserved_bytes=manifest['peak_reserved_bytes'],
            quality_passed=audit.get('verified', False),
            strict_exact=audit.get('strict_exact', False),
            token_count=audit.get('token_count'), tool_count=audit.get('tool_count'),
            tool_seconds=audit.get('tool_seconds'),
            score_v3=audit.get('scorer_v3_correct'),
            failed_ids=audit.get('failed_ids'), audit_exit_code=audit_code,
            audit_command=audit_command,
            manifest_sha256=digest(trial_root / 'gate.json'),
        )
        save('progress.json', data)
        print(json.dumps(record), flush=True)
        if audit_code or not audit.get('verified'):
            data.update(passed=False, status='stopped_raw_audit_failure')
            save('decision.json', data)
            return 1

    a, b, c, d = [trial['seconds'] for trial in data['trials']]
    gain = (a + d - b - c) / (a + d)
    pairs = [(a - b) / a, (d - c) / d]
    ranges = [abs(a - d) / min(a, d), abs(b - c) / min(b, c)]
    passed = gain >= .02 and min(pairs) > 0 and gain > max(ranges)
    data.update(
        passed=passed,
        status='full128_incremental_performance_passed' if passed else 'full128_performance_not_proven',
        mean_gain_fraction=gain, paired_gains=pairs, range_fractions=ranges,
        baseline_mean_seconds=(a + d) / 2,
        candidate_mean_seconds=(b + c) / 2,
        workload_identical=all(trial['strict_exact'] for trial in data['trials']),
        deployment_accepted=False,
    )
    save('decision.json', data)

    checks = {
        'four_trials': len(data['trials']) == 4,
        'device0': data['physical_device'] == '0',
        'source_hashes': all(digest(path) == value
                             for path, value in data['source_hashes'].items()),
    }
    audits = []
    for index, kind in enumerate(data['order']):
        trial_root = ROOT / f'trial{index}_{kind}'
        manifest = json.loads((trial_root / 'gate.json').read_text())
        audit = json.loads((trial_root / 'audit.json').read_text())
        record = data['trials'][index]
        checks[f'trial{index}/audit'] = audit['verified'] and record['audit_exit_code'] == 0
        checks[f'trial{index}/selection'] = manifest['baseline'] == (kind == 'baseline')
        checks[f'trial{index}/time'] = record['seconds'] == manifest['makespan_seconds']
        checks[f'trial{index}/memory'] = manifest['peak_allocated_bytes'] <= 62094008832
        audits.append({'kind': kind, 'seconds': record['seconds'],
                       'strict_exact': audit['strict_exact'],
                       'score_v3': audit['scorer_v3_correct'],
                       'failed_ids': audit['failed_ids'],
                       'token_count': audit['token_count'],
                       'tool_count': audit['tool_count'],
                       'allocated': manifest['peak_allocated_bytes'],
                       'reserved': manifest['peak_reserved_bytes']})
    independent = {
        'verified': all(checks.values()), 'checks': checks,
        'performance_passed': passed and all(checks.values()),
        'mean_gain_fraction': gain, 'paired_gains': pairs,
        'range_fractions': ranges, 'trials': audits,
    }
    save('independent_audit.json', independent)
    print(json.dumps({'decision': data, 'independent_audit': independent}, indent=2),
          flush=True)
    return 0 if independent['verified'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
