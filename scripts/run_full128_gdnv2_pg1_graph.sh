#!/bin/bash
# full128: --fast-gdn --gdn-v2 --prefill-group 1 --graph-decode (NPU 0). Compare with full128_pg1_graph (499.7 s).
cd /remote-home/S45160/work/910b_fresh
PY=/remote-home/S45160/miniforge3/envs/ascend-e2e/bin/python
export ASCEND_RT_VISIBLE_DEVICES=0
$PY -I src/cb_run.py --out results/ablations/continuous_batching/full128_pg1_gdnv2_graph/per_task.jsonl --device npu:0 --fast-gdn --gdn-v2 --prefill-group 1 --graph-decode > logs/cb_full128_pg1_gdnv2_graph.log 2>&1
echo DONE > logs/cb_full128_pg1_gdnv2_graph_status.txt
