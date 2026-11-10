#!/bin/bash
# ABBA confirmation on npu:0: C3 (sync-free), A2 (baseline), A2 then C3 order kept sequential.
set -u
cd /remote-home/S45160/work/910b_fresh
PY=/remote-home/S45160/miniforge3/envs/ascend-e2e/bin/python
run() { local name=$1; shift
  mkdir -p results/shift_conv/$name; rm -f results/shift_conv/$name/per_task.jsonl
  ASCEND_RT_VISIBLE_DEVICES=0 $PY -I src/e100_batch.py --out results/shift_conv/$name/per_task.jsonl \
     --device npu:0 --protocol original --schedule fifo --limit 16 "$@" > logs/abba_$name.log 2>&1
  echo "$name exit=$?" >> logs/abba_status.txt; }
rm -f logs/abba_status.txt
run C3 --sync-free
run A2
run A3
run C4 --sync-free
echo DONE >> logs/abba_status.txt
