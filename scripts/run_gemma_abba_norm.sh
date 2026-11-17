#!/bin/bash
# ABBA on first16 GSM8K, Gemma serial runner, NPU 0, strictly sequential.
# A = original RMSNorm, B = --fast-norm (gemma_fast_norm.py). Order A1 B1 B2 A2.
cd /remote-home/S45160/work/910b_fresh
PY=/remote-home/S45160/miniforge3/envs/ascend-e2e/bin/python
run() { # name flags...
  name=$1; shift
  mkdir -p results/gemma/abba_norm/$name
  $PY -I src/gemma_run.py results/gemma/abba_norm/$name --device npu:0 --limit 16 "$@" \
     > logs/gemma_abba_norm_$name.log 2>&1
  echo "$name exit=$? $(date)" >> logs/gemma_abba_norm_driver.log
}
run A1
run B1 --fast-norm
run B2 --fast-norm
run A2
echo DONE >> logs/gemma_abba_norm_driver.log
