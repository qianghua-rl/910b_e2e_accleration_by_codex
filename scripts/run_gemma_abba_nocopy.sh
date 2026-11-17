#!/bin/bash
# ABBA: nocopy GQA attention on top of the E-G35 config, Gemma CB, first16, npu:1.
# A = --fast-norm-noscale --fused-rope (E-G35 B). B = A + --nocopy. Order A1 -> B1 -> B2 -> A2.
set -u
cd /remote-home/S45160/work/910b_fresh
PY=/remote-home/S45160/miniforge3/envs/ascend-e2e/bin/python
OUT=results/gemma/abba_nocopy
mkdir -p logs "$OUT"
run() {
  local name=$1; shift
  echo "[$(date +%T)] START $name $*"
  $PY src/gemma_run.py "$OUT/$name" --device npu:1 --cb --slots 16 --limit 16 --compact --grouped-experts --fast-norm \
     --fast-norm-noscale --fused-rope "$@" > "logs/gemma_abba_nocopy_$name.log" 2>&1
  local rc=$?
  echo "[$(date +%T)] END $name rc=$rc"
  if [ $rc -ne 0 ] || [ ! -f "$OUT/$name/summary.json" ]; then
    echo "FAILED $name, stopping"; exit 1
  fi
}
run A1
run B1 --nocopy
run B2 --nocopy
run A2
echo "ALL_DONE"
