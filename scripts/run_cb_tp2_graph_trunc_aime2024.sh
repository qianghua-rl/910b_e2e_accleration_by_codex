#!/bin/bash
# AIME 2024, TP=2 graph decode, with --truncate-to-fit (cap final turn at t_max-prompt-1 instead of rejecting).
cd /remote-home/S45160/work/910b_fresh
PY=/remote-home/S45160/miniforge3/envs/ascend-e2e/bin/python
OUT=results/ablations/continuous_batching_tp2/graph_trunc_aime2024
for r in 0 1; do
  RANK=$r ASCEND_RT_VISIBLE_DEVICES=$r $PY -I src/cb_run.py --out $OUT/per_task.jsonl \
    --data data/benchmarks/aime2024.jsonl --tp 2 --fast-gdn --gdn-v2 --elem-conv --prefill-group 1 --slots 12 --t-max 8192 --direct-max-new-tokens 6144 --first-max-new-tokens 6144 --final-max-new-tokens 2048 --tool-output-max-chars 2000 --graph-decode --truncate-to-fit > logs/cb_tp2_graph_trunc_aime2024_rank$r.log 2>&1 &
done
wait
echo "trunc aime2024 done $(date +%H:%M:%S)"
