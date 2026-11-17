#!/bin/bash
# ABBA: operator-count reduction (no-scale norms fused + RoPE via npu_rotary_mul), Gemma CB, first16, npu:1.
# A = current best (--cb --compact --grouped-experts --fast-norm). B = A + --fast-norm-noscale + --fused-rope.
# Order A1 -> B1 -> B2 -> A2. Stops at first failed run.
set -u
cd /remote-home/S45160/work/910b_fresh
PY=/remote-home/S45160/miniforge3/envs/ascend-e2e/bin/python
OUT=results/gemma/abba_ops
mkdir -p logs "$OUT"
run() {
  local name=$1; shift
  echo "[$(date +%T)] START $name $*"
  $PY src/gemma_run.py "$OUT/$name" --device npu:1 --cb --slots 16 --limit 16 --compact --grouped-experts --fast-norm "$@" \
     > "logs/gemma_abba_ops_$name.log" 2>&1
  local rc=$?
  echo "[$(date +%T)] END $name rc=$rc"
  if [ $rc -ne 0 ] || [ ! -f "$OUT/$name/summary.json" ]; then
    echo "FAILED $name, stopping"; exit 1
  fi
}
run A1
run B1 --fast-norm-noscale --fused-rope
run B2 --fast-norm-noscale --fused-rope
run A2
echo "ALL_DONE"
