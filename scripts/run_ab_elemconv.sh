#!/bin/bash
# first16 A/B on one card, sequential: fast-gdn fixed, decode conv torch (A) vs elem (B).
PY=/remote-home/S45160/miniforge3/envs/ascend-e2e/bin/python
cd /remote-home/S45160/work/910b_fresh
export ASCEND_RT_VISIBLE_DEVICES=1
$PY -I src/cb_run.py --out results/ablations/continuous_batching/ab_elemconv_A16/per_task.jsonl --device npu:0 --fast-gdn --limit 16 > logs/ab_elemconv_A16.log 2>&1
$PY -I src/cb_run.py --out results/ablations/continuous_batching/ab_elemconv_B16/per_task.jsonl --device npu:0 --fast-gdn --elem-conv --limit 16 > logs/ab_elemconv_B16.log 2>&1
echo DONE > logs/ab_elemconv_status.txt
