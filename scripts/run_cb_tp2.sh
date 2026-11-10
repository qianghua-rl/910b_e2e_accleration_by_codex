#!/bin/bash
# Self-written CB engine, tensor parallel over NPU 0 and 1 (src/cb_tp.py). Two processes, one card each.
# Settings match the vLLM TP=2 run (results/bench_vllm_tp2): t_max 8192, direct/first-turn cap 6144,
# final-turn cap 2048, tool output clipped to 2000 chars, 12 slots, prefill one request at a time.
# Output: results/ablations/continuous_batching_tp2/<name>/per_task.jsonl (rank 0 writes; rank 1 computes too).
# Usage: bash scripts/run_cb_tp2.sh <smoke|full>
cd /remote-home/S45160/work/910b_fresh
PY=/remote-home/S45160/miniforge3/envs/ascend-e2e/bin/python
MODE=${1:-smoke}
OUT=results/ablations/continuous_batching_tp2
mkdir -p $OUT logs

COMMON="--tp 2 --fast-gdn --gdn-v2 --elem-conv --prefill-group 1 --slots 12 --t-max 8192 --direct-max-new-tokens 6144 --first-max-new-tokens 6144 --final-max-new-tokens 2048 --tool-output-max-chars 2000"

run() { # $1 name, $2 data, extra flags...
  local name=$1 data=$2; shift 2
  mkdir -p $OUT/$name
  for r in 0 1; do
    RANK=$r ASCEND_RT_VISIBLE_DEVICES=$r $PY -I src/cb_run.py --out $OUT/$name/per_task.jsonl \
      --data "$data" $COMMON "$@" > logs/cb_tp2_${name}_rank$r.log 2>&1 &
  done
  wait
}

case "$MODE" in
  smoke)
    rm -rf $OUT/smoke; mkdir -p $OUT/smoke
    run smoke data/benchmarks/gpqa_diamond.jsonl --direct --limit 2
    echo "smoke done" ;;
  full)
    run gpqa_diamond data/benchmarks/gpqa_diamond.jsonl --direct
    run aime2024     data/benchmarks/aime2024.jsonl
    run aime2025     data/benchmarks/aime2025.jsonl
    run math500      data/benchmarks/math500.jsonl
    run bbh          data/benchmarks/bbh.jsonl --direct
    echo "full done" ;;
  *) echo "unknown mode: $MODE"; exit 1 ;;
esac
