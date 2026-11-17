#!/bin/bash
# ABBA: fused attention (FIA) + fused MoE activation (geglu), Gemma CB, first16, npu:1.
# A = current best (E-G35 B config: --fast-norm-noscale --fused-rope). B = A + --fia --geglu.
# Order A1 -> B1 -> B2 -> A2. Stops at first failed run.
set -u
cd /remote-home/S45160/work/910b_fresh
PY=/remote-home/S45160/miniforge3/envs/ascend-e2e/bin/python
OUT=results/gemma/abba_fia_geglu
mkdir -p logs "$OUT"
run() {
  local name=$1; shift
  echo "[$(date +%T)] START $name $*"
  $PY src/gemma_run.py "$OUT/$name" --device npu:1 --cb --slots 16 --limit 16 --compact --grouped-experts --fast-norm \
     --fast-norm-noscale --fused-rope "$@" > "logs/gemma_abba_fia_geglu_$name.log" 2>&1
  local rc=$?
  echo "[$(date +%T)] END $name rc=$rc"
  if [ $rc -ne 0 ] || [ ! -f "$OUT/$name/summary.json" ]; then
    echo "FAILED $name, stopping"; exit 1
  fi
}
run A1
run B1 --fia --geglu
run B2 --fia --geglu
run A2
echo "ALL_DONE"
