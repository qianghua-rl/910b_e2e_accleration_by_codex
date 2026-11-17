#!/bin/bash
# ABBA: grouped experts in the HF batch-1 prefill (phase 2 probe), full128, npu:1.
# A = current best (E-G35 config). B = A + --prefill-grouped. Order A1 -> B1 -> B2 -> A2.
set -u
cd /remote-home/S45160/work/910b_fresh
PY=/remote-home/S45160/miniforge3/envs/ascend-e2e/bin/python
OUT=results/gemma/abba_prefill_full128
mkdir -p logs "$OUT"
BASE="--cb --slots 16 --compact --grouped-experts --fast-norm --fast-norm-noscale --fused-rope"
run() {
  local name=$1; shift
  echo "[$(date +%T)] START $name $*"
  $PY src/gemma_run.py "$OUT/$name" --device npu:1 $BASE "$@" > "logs/gemma_abba_prefill_full128_$name.log" 2>&1
  local rc=$?
  echo "[$(date +%T)] END $name rc=$rc"
  if [ $rc -ne 0 ] || [ ! -f "$OUT/$name/summary.json" ]; then echo "FAILED $name, stopping"; exit 1; fi
}
run A1
run B1 --prefill-grouped
run B2 --prefill-grouped
run A2
echo "ALL_DONE"
