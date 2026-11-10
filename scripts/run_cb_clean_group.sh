#!/bin/bash
# Sequential, unshared CB prefill-group runs on npu:0 (one job at a time).
PY=/remote-home/S45160/miniforge3/envs/ascend-e2e/bin/python
cd /remote-home/S45160/work/910b_fresh
for G in 4 8; do
  $PY -I src/cb_run.py --out results/ablations/continuous_batching/clean_group$G --device npu:0 --prefill-group $G \
    --protocol original > logs/cb_clean_group$G.log 2>&1
  echo "group$G EXIT $?" >> logs/cb_clean_status.txt
done
echo DONE >> logs/cb_clean_status.txt
