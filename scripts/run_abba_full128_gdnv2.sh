#!/bin/bash
# ABBA on full128, --fast-gdn --prefill-group 1 --graph-decode, NPU 0, sequential.
# A = baseline (gdn_chunk_fast), B = --gdn-v2 (gdn_chunk_fast2).
cd /remote-home/S45160/work/910b_fresh
PY=/remote-home/S45160/miniforge3/envs/ascend-e2e/bin/python
export ASCEND_RT_VISIBLE_DEVICES=0
run() { # $1 label, $2 extra flags
  $PY -I src/cb_run.py --out results/ablations/continuous_batching/abba_full128_$1/per_task.jsonl --device npu:0 --fast-gdn --prefill-group 1 --graph-decode $2 > logs/cb_abba_full128_$1.log 2>&1
}
run A1 ""
run B1 "--gdn-v2"
run B2 "--gdn-v2"
run A2 ""
echo DONE > logs/cb_abba_full128_status.txt
