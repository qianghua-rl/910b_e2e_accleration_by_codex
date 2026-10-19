"""Wait for an identified baseline process, validate completion, then profile.

Never restarts the baseline or accepts a partial result. Run from repo root.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--pid', required=True, type=int)
p.add_argument('--baseline', required=True)
args = p.parse_args()

def identity():
    try:
        # Field 22 = starttime; parse after parenthesized comm to tolerate spaces.
        fields = Path(f'/proc/{args.pid}/stat').read_text().rsplit(')',1)[1].split()
        return fields[19] if fields[0] != 'Z' else None
    except FileNotFoundError:
        return None

initial = identity()
if initial is None:
    raise SystemExit('Expected baseline process is not live; inspect manually')
print(f'Waiting for pid={args.pid}, start_ticks={initial}', flush=True)
while identity() == initial:
    time.sleep(10)
rows = [json.loads(x) for x in Path(args.baseline).read_text().splitlines()]
meta = next(x for x in rows if x['kind']=='metadata')
tasks = [x for x in rows if x['kind']=='task']
if (meta['tasks'] != 128 or len(tasks)!=128 or len({x['id'] for x in tasks})!=128
        or not any(x['kind']=='summary' for x in rows)):
    raise SystemExit('Baseline terminated without complete 128-task result; profile not started')
print('Baseline complete; starting isolated profile', flush=True)
with open('results/profile_baseline.stdout.log','x') as log:
    result = subprocess.run([sys.executable,'scripts/profile_inference.py'], stdout=log, stderr=subprocess.STDOUT)
print(f'Profiler exit={result.returncode}', flush=True)
raise SystemExit(result.returncode)
