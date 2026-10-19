"""Audit a run against the fixed dataset and regenerate scores from saved text."""
import argparse
import hashlib
import json
import sys
sys.path.insert(0, 'scripts')
from agent_protocol import answer, correct, parse_call

p = argparse.ArgumentParser()
p.add_argument('run')
p.add_argument('--allow-partial', action='store_true')
args = p.parse_args()
data = open('data/gsm8k_128.jsonl','rb').read()
expected = {r['id']:r for r in map(json.loads,data.splitlines())}
rows = [json.loads(x) for x in open(args.run)]
meta = [r for r in rows if r['kind']=='metadata']
if len(meta)==1 and meta[0].get('scorer')=='balanced_box_numeric_time_v2':
    from scoring_v2 import answer, correct
elif len(meta)==1 and meta[0].get('scorer')=='balanced_box_numeric_time_v3':
    from scoring_v3 import answer, correct
tasks = [r for r in rows if r['kind']=='task']
errors=[]
if len(meta)!=1:
    errors.append('Expected exactly one metadata record')
else:
    m=meta[0]
    if m['data_sha256'] != hashlib.sha256(data).hexdigest(): errors.append('Dataset hash mismatch')
    if (m['max_rounds'],m['max_new_tokens'],m['do_sample'],m['enable_thinking'])!=(6,512,False,False):
        errors.append('Protocol setting mismatch')
seen=set()
for r in tasks:
    key=r['id']
    if key in seen or key not in expected:
        errors.append(f'Duplicate/unknown id {key}')
        continue
    seen.add(key)
    rounds=r['rounds']
    if not 1<=len(rounds)<=6: errors.append(f'{key}: invalid round count')
    calls=0
    pred=None
    for i,turn in enumerate(rounds):
        if turn['turn']!=i or not 1<=len(turn['tokens'])<=512:
            errors.append(f'{key}: invalid turn index/token count')
        code=parse_call(turn['text'])
        if code is not None:
            calls+=1
            if turn.get('code')!=code or 'tool_output' not in turn:
                errors.append(f'{key}: missing/mismatched tool execution')
        else:
            pred=answer(turn['text'])
            if i!=len(rounds)-1: errors.append(f'{key}: continuation after final turn')
    score=calls>0 and correct(pred,expected[key]['answer'])
    if (r['prediction'],r['tool_count'],r['correct'],r['expected'])!=(pred,calls,score,expected[key]['answer']):
        errors.append(f'{key}: saved score/protocol inconsistency')
complete=(len(seen)==128 and len(tasks)==128 and len(meta)==1
          and meta[0]['tasks']==128 and sum(r['kind']=='summary' for r in rows)==1)
if not args.allow_partial and not complete: errors.append('Full 128-task completion not established')
print(json.dumps(dict(complete=complete,audited_tasks=len(tasks),errors=errors),indent=2))
raise SystemExit(bool(errors))
