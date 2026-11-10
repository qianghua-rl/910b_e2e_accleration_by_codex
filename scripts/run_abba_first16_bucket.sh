#!/bin/bash
# ABBA on first16, --fast-gdn --gdn-v2 --prefill-group 1 --graph-decode --elem-conv, NPU 0, sequential.
# A = CB_GRAPH_BUCKET=512 (current), B = CB_GRAPH_BUCKET=128 (candidate, untested before).
cd /remote-home/S45160/work/910b_fresh
PY=/remote-home/S45160/miniforge3/envs/ascend-e2e/bin/python
export ASCEND_RT_VISIBLE_DEVICES=0
run() { # $1 label, $2 bucket
  CB_GRAPH_BUCKET=$2 $PY -I src/cb_run.py --out results/ablations/continuous_batching/abba_first16_bucket_$1/per_task.jsonl --device npu:0 \
    --fast-gdn --gdn-v2 --prefill-group 1 --graph-decode --elem-conv --limit 16 > logs/cb_abba_first16_bucket_$1.log 2>&1
}
run A1 512
run B1 128
run B2 128
run A2 512
echo DONE > logs/cb_abba_first16_bucket_status.txt
