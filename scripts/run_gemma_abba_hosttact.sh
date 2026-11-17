#!/bin/bash
# ABBA: host-computed attention length (no pos.max().item() sync), Gemma CB, first16, 16 slots, npu:1.
# Order A1 -> B1 -> B2 -> A2. A = --cb --compact --grouped-experts --fast-norm (E-G23 C config).
# B = A + --host-tact. Stops at first failed run.
set -u
cd /remote-home/S45160/work/910b_fresh
PY=/remote-home/S45160/miniforge3/envs/ascend-e2e/bin/python
OUT=results/gemma/abba_hosttact
mkdir -p logs "$OUT"
run() {  # name, extra flags...
  local name=$1; shift
  echo "[$(date +%T)] START $name $*"
  $PY src/gemma_run.py "$OUT/$name" --device npu:1 --cb --slots 16 --limit 16 --compact --grouped-experts --fast-norm "$@" \
     > "logs/gemma_abba_hosttact_$name.log" 2>&1
  local rc=$?
  echo "[$(date +%T)] END $name rc=$rc"
  if [ $rc -ne 0 ] || [ ! -f "$OUT/$name/summary.json" ]; then
    echo "FAILED $name, stopping"; exit 1
  fi
}
run A1
run B1 --host-tact
run B2 --host-tact
run A2
echo "ALL_DONE"
