#!/bin/bash
# full128: --fast-gdn --prefill-group 4 --graph-decode with expandable_segments allocator (NPU 0).
cd /remote-home/S45160/work/910b_fresh
PY=/remote-home/S45160/miniforge3/envs/ascend-e2e/bin/python
export ASCEND_RT_VISIBLE_DEVICES=0 PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
$PY -I src/cb_run.py --out results/ablations/continuous_batching/full128_pg4_graph_expseg/per_task.jsonl --device npu:0 --fast-gdn --prefill-group 4 --graph-decode > logs/cb_full128_pg4_graph_expseg.log 2>&1
echo DONE > logs/cb_full128_pg4_graph_expseg_status.txt
