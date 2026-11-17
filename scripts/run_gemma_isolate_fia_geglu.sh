#!/bin/bash
# Isolate which fusion breaks outputs (E-G36). Each arm alone on top of E-G35 config, first16, npu:1.
# Arms: F = --fia only, G = --geglu only. Order F -> G. Stops at first failed run.
set -u
cd /remote-home/S45160/work/910b_fresh
PY=/remote-home/S45160/miniforge3/envs/ascend-e2e/bin/python
OUT=results/gemma/isolate_fia_geglu
mkdir -p logs "$OUT"
run() {
  local name=$1; shift
  echo "[$(date +%T)] START $name $*"
  $PY src/gemma_run.py "$OUT/$name" --device npu:1 --cb --slots 16 --limit 16 --compact --grouped-experts --fast-norm \
     --fast-norm-noscale --fused-rope "$@" > "logs/gemma_isolate_$name.log" 2>&1
  local rc=$?
  echo "[$(date +%T)] END $name rc=$rc"
  if [ $rc -ne 0 ] || [ ! -f "$OUT/$name/summary.json" ]; then
    echo "FAILED $name, stopping"; exit 1
  fi
}
run F --fia
run G --geglu
echo "ALL_DONE"
