"""Summarize stage samples without interpreting stream elapsed as kernel busy."""
import argparse
import json
import statistics as st
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--input',default='results/e123_generation_stages.json')
p.add_argument('--output',default='results/e123_generation_stages_analysis.json')
a=p.parse_args()
data=json.loads(Path(a.input).read_text())
trials=data['results']
summary=[]
for i,r in enumerate(trials):
    if not r['instrumented']: continue
    by_step={}
    for sample in r['stages']:
        if not 1 <= sample['step'] < data['steps']: continue
        entry=by_step.setdefault(sample['step'],{})
        item=entry.setdefault(sample['stage'],dict(host_ms=0.,device_ms=0.))
        for metric in item: item[metric]+=sample[metric]
    assert len(by_step)==data['steps']-1
    assert all(set(v)=={'scheduler','model_forward','sampling','postprocess'} for v in by_step.values())
    totals={stage:{metric:sum(step[stage][metric] for step in by_step.values())
                   for metric in ['host_ms','device_ms']}
            for stage in ['scheduler','model_forward','sampling','postprocess']}
    total_host=sum(v['host_ms'] for v in totals.values())
    for stage,v in totals.items():
        v['host_fraction']=v['host_ms']/total_host
        v['host_step_median_ms']=st.median(s[stage]['host_ms'] for s in by_step.values())
        v['device_step_median_ms']=st.median(s[stage]['device_ms'] for s in by_step.values())
    left,right=trials[i-1],trials[i+1]
    control=(left['wall_ms']+right['wall_ms'])/2
    summary.append(dict(trial=i,decode_steps=len(by_step),stages=totals,
                        adjacent_control_drift_pct=100*abs(left['wall_ms']-right['wall_ms'])/control,
                        instrumentation_delta_pct=100*(r['wall_ms']/control-1),
                        steps=by_step))
result=dict(all_tokens_exact=all(r['tokens_exact'] and r['e100_tokens_exact'] for r in trials),
            peak_bytes=max(r['peak_bytes'] for r in trials),trials=summary,
            interpretation='Stage fractions are diagnostic upper bounds. Event intervals include host idle and queue effects. No proven removable/overlap budget; no128-task claim.')
Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
for r in summary:
    print('trial',r['trial'],'control drift %',round(r['adjacent_control_drift_pct'],3),
          'instrument delta %',round(r['instrumentation_delta_pct'],3))
    for k,v in r['stages'].items(): print(k,{m:round(x,4) for m,x in v.items()})
