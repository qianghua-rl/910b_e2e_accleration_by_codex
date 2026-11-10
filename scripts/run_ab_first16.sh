#!/bin/bash
# Sequential A/B on npu:0: A = current optimised HF, B = + shift decode conv. Same tasks, fifo order.
set -u
cd /remote-home/S45160/work/910b_fresh
PY=/remote-home/S45160/miniforge3/envs/ascend-e2e/bin/python
run() { # name, extra args
  local name=$1; shift
  mkdir -p results/shift_conv/$name; rm -f results/shift_conv/$name/per_task.jsonl
  ASCEND_RT_VISIBLE_DEVICES=0 $PY -I src/e100_batch.py --out results/shift_conv/$name/per_task.jsonl \
     --device npu:0 --protocol original --schedule fifo --limit 16 "$@" > logs/shift_conv_$name.log 2>&1
  echo "$name exit=$?" >> logs/shift_conv_ab_status.txt
}
run A
run B --shift-conv
echo DONE >> logs/shift_conv_ab_status.txt
