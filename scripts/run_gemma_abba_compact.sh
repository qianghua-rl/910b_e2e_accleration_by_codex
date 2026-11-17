#!/bin/bash
# ABBA for active-row compaction on Gemma CB, first16, 16 slots, npu:1 (sequential).
# Order A1 -> B1 -> B2 -> A2. A = --cb (all slots computed), B = --cb --compact.
# Stops at the first failed run so a broken B does not waste the remaining runs.
set -u
cd /remote-home/S45160/work/910b_fresh
PY=/remote-home/S45160/miniforge3/envs/ascend-e2e/bin/python
OUT=results/gemma/abba_compact
mkdir -p logs "$OUT"
run() {  # name, extra flags...
  local name=$1; shift
  echo "[$(date +%T)] START $name $*"
  $PY src/gemma_run.py "$OUT/$name" --device npu:1 --cb --slots 16 --limit 16 "$@" \
     > "logs/gemma_abba_compact_$name.log" 2>&1
  local rc=$?
  echo "[$(date +%T)] END $name rc=$rc"
  if [ $rc -ne 0 ] || [ ! -f "$OUT/$name/summary.json" ]; then
    echo "FAILED $name, stopping"; exit 1
  fi
}
run A1
run B1 --compact
run B2 --compact
run A2
echo "ALL_DONE"
