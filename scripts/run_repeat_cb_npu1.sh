#!/bin/bash
# Sequential repeat of CB full128 base vs fast-gdn on one card (NPU 1), no concurrency.
cd /remote-home/S45160/work/910b_fresh
PY=/remote-home/S45160/miniforge3/envs/ascend-e2e/bin/python
export ASCEND_RT_VISIBLE_DEVICES=1
$PY -I src/cb_run.py --out results/ablations/continuous_batching/repeat_base_r2/per_task.jsonl --device npu:0 > logs/cb_repeat_base_r2.log 2>&1
$PY -I src/cb_run.py --out results/ablations/continuous_batching/repeat_fast_r2/per_task.jsonl --device npu:0 --fast-gdn > logs/cb_repeat_fast_r2.log 2>&1
echo DONE > logs/cb_repeat_status.txt
