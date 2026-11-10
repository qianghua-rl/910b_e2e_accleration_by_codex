#!/bin/bash
# full128, --prefill-group 1, --fast-gdn: eager then graph-decode, sequential on NPU 0.
cd /remote-home/S45160/work/910b_fresh
PY=/remote-home/S45160/miniforge3/envs/ascend-e2e/bin/python
export ASCEND_RT_VISIBLE_DEVICES=0
$PY -I src/cb_run.py --out results/ablations/continuous_batching/full128_pg1_eager/per_task.jsonl --device npu:0 --fast-gdn --prefill-group 1 > logs/cb_full128_pg1_eager.log 2>&1
$PY -I src/cb_run.py --out results/ablations/continuous_batching/full128_pg1_graph/per_task.jsonl --device npu:0 --fast-gdn --prefill-group 1 --graph-decode > logs/cb_full128_pg1_graph.log 2>&1
echo DONE > logs/cb_full128_pg1_status.txt
