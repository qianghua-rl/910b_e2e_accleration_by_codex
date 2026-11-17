#!/bin/bash
# Timed counter-probe ABBA for grouped prefill (E-G40 check): order B1 -> A1 -> B2 -> A2, first16, npu:1.
set -u
cd /remote-home/S45160/work/910b_fresh
PY=/remote-home/S45160/miniforge3/envs/ascend-e2e/bin/python
OUT=results/gemma/probe_prefill_abba
mkdir -p logs "$OUT"
BASE="--slots 16 --limit 16 --compact --grouped-experts --fast-norm --fast-norm-noscale --fused-rope"
run() {
  local name=$1; shift
  echo "[$(date +%T)] START $name $*"
  $PY -I scripts/probe_gemma_cb_counters.py "$OUT/$name" --device npu:1 $BASE "$@" > "logs/gemma_probe_abba_$name.log" 2>&1
  local rc=$?
  echo "[$(date +%T)] END $name rc=$rc"
  if [ $rc -ne 0 ] || [ ! -f "$OUT/$name/probe.json" ]; then echo "FAILED $name, stopping"; exit 1; fi
}
run B1 --prefill-grouped
run A1
run B2 --prefill-grouped
run A2
echo "ALL_DONE"
