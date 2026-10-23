"""Pair isolated E124-A results and classify stability."""
import argparse, json, statistics as st
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--input',default='results/e124_nz_e2e.jsonl'); p.add_argument('--output',default='results/e124_nz_gain_gap_report.md'); a=p.parse_args()
r=[json.loads(x) for x in Path(a.input).read_text().splitlines()]; pairs=[]
for i in range(0,len(r),2):
    if i+1>=len(r): break
    nd=next(x for x in r[i:i+2] if x['mode']=='nd'); nz=next(x for x in r[i:i+2] if x['mode']=='nz')
    nt=nd['selected'].get('makespan_seconds'); zt=nz['selected'].get('makespan_seconds')
    pairs.append(dict(nd_trial=nd['trial'],nz_trial=nz['trial'],nd_s=nt,nz_s=zt,delta_pct=100*(zt-nt)/nt,
                      tokens_equal=nd['selected'].get('trajectory_sha256') == nz['selected'].get('trajectory_sha256'),
                      logits_equal=None))
d=[x['delta_pct'] for x in pairs]; stable=sum(x>0 for x in d)>=len(d)*.75 if d else False
md=['# E124 NZ gain gap report','','## E124-A','',f'Pairs: {len(pairs)}; NZ slower pairs: {sum(x>0 for x in d)}/{len(d)}; median delta: {st.median(d) if d else None:.3f}%.', '', '|pair|ND s|NZ s|delta|tokens|logits|','|---:|---:|---:|---:|---|---|']
for i,x in enumerate(pairs): md.append(f"|{i}|{x['nd_s']:.4f}|{x['nz_s']:.4f}|{x['delta_pct']:+.3f}%|{x['tokens_equal']}|{x['logits_equal']}|")
md += ['',f'Decision: {"E124-A supports entering E124-B" if stable else "E124-A does not yet establish stable NZ regression"}.', 'All runs are fresh processes; selected generation time excludes model loading and NZ conversion.']
Path(a.output).write_text('\n'.join(md)+'\n'); Path(a.output+'.json').write_text(json.dumps(dict(pairs=pairs,median_delta_pct=st.median(d) if d else None,stable_regression=stable),indent=2)+'\n')
print('\n'.join(md))
