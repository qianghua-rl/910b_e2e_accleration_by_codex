#!/bin/bash
# Diagnostic: unanswered AIME 2024 tasks (0,1,5,7,21,28) with higher caps (first 7168, final 4096), graph decode, TP=2.
cd /remote-home/S45160/work/910b_fresh
PY=/remote-home/S45160/miniforge3/envs/ascend-e2e/bin/python
OUT=results/ablations/continuous_batching_tp2/graph_trunc_cap_aime2024_subset
for r in 0 1; do
  RANK=$r ASCEND_RT_VISIBLE_DEVICES=$r $PY -I src/cb_run.py --out $OUT/per_task.jsonl \
    --data results/ablations/hf_reference/aime2024_unanswered_subset.jsonl --tp 2 --fast-gdn --gdn-v2 --elem-conv \
    --prefill-group 1 --slots 12 --t-max 8192 --direct-max-new-tokens 6144 --first-max-new-tokens 7168 \
    --final-max-new-tokens 4096 --tool-output-max-chars 2000 --graph-decode --truncate-to-fit \
    > logs/cb_tp2_cap_subset_rank$r.log 2>&1 &
done
wait
echo "cap subset done $(date +%H:%M:%S)"
