#!/bin/bash
# ABBA timing: eager vs --graph-decode, first16, --prefill-group 1, --fast-gdn, sequential on NPU 0.
cd /remote-home/S45160/work/910b_fresh
PY=/remote-home/S45160/miniforge3/envs/ascend-e2e/bin/python
export ASCEND_RT_VISIBLE_DEVICES=0
run() { # $1 label $2 extra-flags
  $PY -I src/cb_run.py --out results/ablations/continuous_batching/abba_graph_pg1_$1/per_task.jsonl --device npu:0 --fast-gdn --limit 16 --prefill-group 1 $2 > logs/cb_abba_graph_pg1_$1.log 2>&1
}
run A1 ""
run B1 "--graph-decode"
run B2 "--graph-decode"
run A2 ""
echo DONE > logs/cb_abba_graph_pg1_status.txt
