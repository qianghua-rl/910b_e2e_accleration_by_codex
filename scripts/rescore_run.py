"""Write a scored derivative; original tokens, text, timing and file stay intact."""
import argparse
import json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('source')
p.add_argument('output')
p.add_argument('--scorer',choices=['v2','v3'],default='v2')
args=p.parse_args()
scoring=__import__(f'scoring_{args.scorer}',fromlist=['answer','correct'])
answer,correct=scoring.answer,scoring.correct
if Path(args.output).exists():
    raise SystemExit('Output exists')
rows=[json.loads(x) for x in open(args.source)]
changes=[]
for row in rows:
    if row['kind']=='metadata':
        row.update(scorer=f'balanced_box_numeric_time_{args.scorer}',source_run=args.source)
    elif row['kind']=='task':
        pred=answer(row['rounds'][-1]['text'])
        score=row['tool_count']>0 and correct(pred,row['expected'])
        row['original_prediction']=row['prediction']
        row['original_correct']=row['correct']
        if (row['prediction'],row['correct'])!=(pred,score):
            changes.append(dict(id=row['id'],before=row['prediction'],after=pred,
                                old_correct=row['correct'],new_correct=score))
        row.update(prediction=pred,correct=score)
Path(args.output).write_text(''.join(json.dumps(row)+'\n' for row in rows))
print(json.dumps(changes,indent=2))
