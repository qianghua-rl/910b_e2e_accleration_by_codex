#!/bin/bash
# Self-written CB engine across two cards (NPU 0 + NPU 1) via --split-devices (layer-wise pipeline split).
# Matches the long-generation settings of the vLLM TP=2 run (results/bench_vllm_tp2) so the two are comparable:
#   t_max 8192, direct/first-turn cap 6144, final-turn cap 2048, tool output clipped to 2000 chars, 12 slots.
# Caveats:
#   - --split-devices is pipeline (layer) split, not tensor parallel: it frees memory per card but does not speed up.
#   - It is incompatible with --graph-decode, so this runs eager decode. Expect slower wall time than the
#     single-card graph-decode best (410-500 s on full128); compare per-token throughput, not makespan.
# Output: results/ablations/continuous_batching_split/<name>/per_task.jsonl (new dir; earlier results untouched).
# Usage: bash scripts/run_cb_split_long.sh <smoke|rest|full>
#   smoke: 2 GPQA tasks, to check it starts and fits in memory
#   rest:  math500 + bbh (the two datasets not yet run in this config)
#   full:  gpqa_diamond, aime2024, aime2025, math500, bbh
cd /remote-home/S45160/work/910b_fresh
PY=/remote-home/S45160/miniforge3/envs/ascend-e2e/bin/python
export ASCEND_RT_VISIBLE_DEVICES=0,1
MODE=${1:-smoke}
OUT=results/ablations/continuous_batching_split
mkdir -p $OUT logs

COMMON="--split-devices npu:0,npu:1 --fast-gdn --gdn-v2 --elem-conv --prefill-group 1 --slots 12 --t-max 8192 --direct-max-new-tokens 6144 --first-max-new-tokens 6144 --final-max-new-tokens 2048 --tool-output-max-chars 2000"

run() { # $1 name, $2 data, extra flags...
  local name=$1 data=$2; shift 2
  mkdir -p $OUT/$name
  $PY -I src/cb_run.py --out $OUT/$name/per_task.jsonl --data "$data" $COMMON "$@" \
    > logs/cb_split_$name.log 2>&1
}

case "$MODE" in
  smoke)
    rm -rf $OUT/smoke; mkdir -p $OUT/smoke
    $PY -I src/cb_run.py --out $OUT/smoke/per_task.jsonl --data data/benchmarks/gpqa_diamond.jsonl \
      $COMMON --direct --limit 2 > logs/cb_split_smoke.log 2>&1
    echo "smoke done" ;;
  rest)
    run math500 data/benchmarks/math500.jsonl
    run bbh     data/benchmarks/bbh.jsonl --direct
    echo "rest done" ;;
  full)
    run gpqa_diamond data/benchmarks/gpqa_diamond.jsonl --direct
    run aime2024     data/benchmarks/aime2024.jsonl
    run aime2025     data/benchmarks/aime2025.jsonl
    run math500      data/benchmarks/math500.jsonl
    run bbh          data/benchmarks/bbh.jsonl --direct
    echo "full done" ;;
  *) echo "unknown mode: $MODE"; exit 1 ;;
esac
