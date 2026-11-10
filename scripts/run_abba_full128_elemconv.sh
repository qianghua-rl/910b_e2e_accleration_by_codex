#!/bin/bash
# ABBA on full128, --fast-gdn --gdn-v2 --prefill-group 1 --graph-decode, NPU 0, sequential.
# A = best config, B = + --elem-conv.
cd /remote-home/S45160/work/910b_fresh
PY=/remote-home/S45160/miniforge3/envs/ascend-e2e/bin/python
export ASCEND_RT_VISIBLE_DEVICES=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
run() { # $1 label, $2 extra flags
  $PY -I src/cb_run.py --out results/ablations/continuous_batching/abba_full128_elemconv_$1/per_task.jsonl --device npu:0 --fast-gdn --gdn-v2 --prefill-group 1 --graph-decode $2 > logs/cb_abba_full128_elemconv_$1.log 2>&1
}
run A1 ""
run B1 "--elem-conv"
run B2 "--elem-conv"
run A2 ""
echo DONE > logs/cb_abba_full128_elemconv_status.txt
