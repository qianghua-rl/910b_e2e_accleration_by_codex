"""Adjacent eight-thread controls; preserve drift instead of cherry-picking."""
import json,statistics
from pathlib import Path
r=json.loads(Path('results/e100_thread_sensitivity.json').read_text());rows=r['results'];pairs=[]
for i,x in enumerate(rows):
 if x['threads']==8 or i==0 or i==len(rows)-1:continue
 a,b=rows[i-1],rows[i+1]
 if a['threads']!=8 or b['threads']!=8:continue
 base=(a['decode_span_wall_ms']+b['decode_span_wall_ms'])/2
 drift=abs(a['decode_span_wall_ms']-b['decode_span_wall_ms'])/base
 pairs.append(dict(trial=x['trial'],threads=x['threads'],control_drift_fraction=drift,
                   stable_controls_under_2percent=drift<.02,
                   relative_wall_saving=1-x['decode_span_wall_ms']/base,
                   relative_event_saving=1-x['forward_event_sum_ms']/((a['forward_event_sum_ms']+b['forward_event_sum_ms'])/2)))
summary={}
for threads in (1,4):
 p=[x for x in pairs if x['threads']==threads];stable=[x for x in p if x['stable_controls_under_2percent']]
 summary[threads]={'all_pairs':len(p),'stable_pairs':len(stable),'all_relative_savings':[x['relative_wall_saving'] for x in p],
                   'stable_median_relative_wall_saving':statistics.median(x['relative_wall_saving'] for x in stable) if stable else None,
                   'stable_median_relative_event_saving':statistics.median(x['relative_event_saving'] for x in stable) if stable else None}
out={'rows':rows,'pairs':pairs,'summary':summary,'all_logits_exact':all(x['logits_exact'] for x in rows),'all_finite':all(x['finite'] for x in rows),'all_replay_equal':all(x['replay_tokens_equal'] for x in rows),'limitations':r['limitations']+['adjacent-control stability threshold is diagnostic, not a significance test','event time includes host submission gaps; cannot prove device-only bound']}
Path('results/e100_thread_sensitivity_analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(summary,indent=2))
