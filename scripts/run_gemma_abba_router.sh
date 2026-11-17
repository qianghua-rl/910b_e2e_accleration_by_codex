#!/bin/bash
# ABBA: fused router on top of the accepted stack (compact + grouped-experts + fast-norm), first16, npu:1.
# Order A1 -> B1 -> B2 -> A2. A = stack config (E-G23 C), B = A + --fast-router. Stops at first failure.
set -u
cd /remote-home/S45160/work/910b_fresh
PY=/remote-home/S45160/miniforge3/envs/ascend-e2e/bin/python
OUT=results/gemma/abba_router
mkdir -p logs "$OUT"
run() {
  local name=$1; shift
  echo "[$(date +%T)] START $name $*"
  $PY src/gemma_run.py "$OUT/$name" --device npu:1 --cb --slots 16 --limit 16 --compact \
     --grouped-experts --fast-norm "$@" > "logs/gemma_abba_router_$name.log" 2>&1
  local rc=$?
  echo "[$(date +%T)] END $name rc=$rc"
  if [ $rc -ne 0 ] || [ ! -f "$OUT/$name/summary.json" ]; then echo "FAILED $name, stopping"; exit 1; fi
}
run A1
run B1 --fast-router
run B2 --fast-router
run A2
echo "ALL_DONE"
