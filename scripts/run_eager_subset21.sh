#!/bin/bash
# Eager (no --graph-decode) self-written TP=2, 8 slots, on the 21 tasks where self and vLLM disagree (matched r1).
# Purpose: test whether the graph decode path causes the accuracy gap on these tasks.
cd /remote-home/S45160/work/910b_fresh
PY=/remote-home/S45160/miniforge3/envs/ascend-e2e/bin/python
D=results/official/continuous_batching_tp2/eager_subset21
for r in 0 1; do
  RANK=$r ASCEND_RT_VISIBLE_DEVICES=$r $PY -I src/cb_run.py --out $D/per_task.jsonl \
    --data $D/data.jsonl --tp 2 --fast-gdn --gdn-v2 --elem-conv --prefill-group 1 --slots 8 --t-max 8192 \
    --direct --direct-max-new-tokens 6144 --first-max-new-tokens 6144 --final-max-new-tokens 2048 \
    --tool-output-max-chars 2000 > logs/eager_subset21_rank$r.log 2>&1 &
done
wait
echo "eager subset finished $(date +%F_%T)"
