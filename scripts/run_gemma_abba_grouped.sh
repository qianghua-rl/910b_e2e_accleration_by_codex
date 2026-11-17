#!/bin/bash
# ABBA for grouped MoE experts on Gemma CB, first16, 16 slots, npu:1 (sequential).
# Order A1 -> B1 -> B2 -> A2. Both arms use --compact (accepted E-G21 config).
# A = --cb --compact (HF per-expert loop), B = --cb --compact --grouped-experts.
# Stops at the first failed run so a broken B does not waste the remaining runs.
set -u
cd /remote-home/S45160/work/910b_fresh
PY=/remote-home/S45160/miniforge3/envs/ascend-e2e/bin/python
OUT=results/gemma/abba_grouped
mkdir -p logs "$OUT"
run() {  # name, extra flags...
  local name=$1; shift
  echo "[$(date +%T)] START $name $*"
  $PY src/gemma_run.py "$OUT/$name" --device npu:1 --cb --slots 16 --limit 16 --compact "$@" \
     > "logs/gemma_abba_grouped_$name.log" 2>&1
  local rc=$?
  echo "[$(date +%T)] END $name rc=$rc"
  if [ $rc -ne 0 ] || [ ! -f "$OUT/$name/summary.json" ]; then
    echo "FAILED $name, stopping"; exit 1
  fi
}
run A1
run B1 --grouped-experts
run B2 --grouped-experts
run A2
echo "ALL_DONE"
