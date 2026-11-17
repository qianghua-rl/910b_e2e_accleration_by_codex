#!/bin/bash
# ABBA: stacking fast-norm on top of grouped experts + compact, Gemma CB, first16, 16 slots, npu:1.
# Order B1 -> C1 -> C2 -> B2. B = --cb --compact --grouped-experts (E-G22 accepted config).
# C = B + --fast-norm (E-G06 accepted). Stops at first failed run.
set -u
cd /remote-home/S45160/work/910b_fresh
PY=/remote-home/S45160/miniforge3/envs/ascend-e2e/bin/python
OUT=results/gemma/abba_stack
mkdir -p logs "$OUT"
run() {  # name, extra flags...
  local name=$1; shift
  echo "[$(date +%T)] START $name $*"
  $PY src/gemma_run.py "$OUT/$name" --device npu:1 --cb --slots 16 --limit 16 --compact --grouped-experts "$@" \
     > "logs/gemma_abba_stack_$name.log" 2>&1
  local rc=$?
  echo "[$(date +%T)] END $name rc=$rc"
  if [ $rc -ne 0 ] || [ ! -f "$OUT/$name/summary.json" ]; then
    echo "FAILED $name, stopping"; exit 1
  fi
}
run B1
run C1 --fast-norm
run C2 --fast-norm
run B2
echo "ALL_DONE"
