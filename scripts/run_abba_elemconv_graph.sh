#!/bin/bash
# ABBA timing: best config (A) vs best + --elem-conv (B), first16, graph decode, sequential on NPU 0.
cd /remote-home/S45160/work/910b_fresh
PY=/remote-home/S45160/miniforge3/envs/ascend-e2e/bin/python
export ASCEND_RT_VISIBLE_DEVICES=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
run() { # $1 label $2 extra-flags
  $PY -I src/cb_run.py --out results/ablations/continuous_batching/abba_elemconv_graph_$1/per_task.jsonl --device npu:0 --fast-gdn --gdn-v2 --graph-decode --prefill-group 1 --limit 16 $2 > logs/cb_abba_elemconv_graph_$1.log 2>&1
}
run A1 ""
run B1 "--elem-conv"
run B2 "--elem-conv"
run A2 ""
echo DONE > logs/cb_abba_elemconv_graph_status.txt
