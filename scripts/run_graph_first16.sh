#!/bin/bash
# first16 correctness: eager vs --graph-decode, sequential on NPU 0 (same code, same schedule).
cd /remote-home/S45160/work/910b_fresh
PY=/remote-home/S45160/miniforge3/envs/ascend-e2e/bin/python
export ASCEND_RT_VISIBLE_DEVICES=0
$PY -I src/cb_run.py --out results/ablations/continuous_batching/graph_first16_eager/per_task.jsonl --device npu:0 --fast-gdn --limit 16 > logs/cb_graph_first16_eager.log 2>&1
$PY -I src/cb_run.py --out results/ablations/continuous_batching/graph_first16_graph/per_task.jsonl --device npu:0 --fast-gdn --limit 16 --graph-decode > logs/cb_graph_first16_graph.log 2>&1
echo DONE > logs/cb_graph_first16_status.txt
