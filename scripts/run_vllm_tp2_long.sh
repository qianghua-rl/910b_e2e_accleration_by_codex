#!/bin/bash
# vLLM-Ascend, tensor parallel 2 over NPU 0 and 1, larger generation caps.
# Purpose: test whether truncation (no \boxed{}) was the main cause of low GPQA/AIME scores.
# Settings: max_model_len 8192, direct/first-turn cap 6144, final-turn cap 2048.
# Output: results/bench_vllm_tp2/<name>/per_task.jsonl (new dir; earlier results untouched).
# Usage: bash scripts/run_vllm_tp2_long.sh <smoke|full>
# Do NOT run with `python -I` (vLLM EngineCore subprocess needs acl).
cd /remote-home/S45160/work/910b_fresh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
export ASCEND_RT_VISIBLE_DEVICES=0,1
# Worker processes crashed at startup with "pool INTERNAL ASSERT ... Invalid thread pool"
# (torch OpenMP state after fork). spawn + single OMP thread avoids it.
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=1
PY=/remote-home/S45160/miniforge3/envs/vllm-ascend-py310/bin/python
MODE=${1:-smoke}
OUT=results/bench_vllm_tp2
mkdir -p $OUT logs

COMMON="--tensor-parallel-size 2 --max-num-seqs 12 --max-model-len 8192 --first-max-tokens 6144 --final-max-tokens 2048 --direct-max-tokens 6144 --tool-output-max-chars 2000"

run() { # $1 name, $2 data, extra flags...
  local name=$1 data=$2; shift 2
  mkdir -p $OUT/$name
  $PY src/vllm_async_bench.py --output $OUT/$name/per_task.jsonl --data "$data" $COMMON "$@" \
    > logs/vllm_tp2_$name.log 2>&1
}

if [ "$MODE" = smoke ]; then
  rm -rf $OUT/smoke; mkdir -p $OUT/smoke
  $PY src/vllm_async_bench.py --output $OUT/smoke/per_task.jsonl --data data/benchmarks/gpqa_diamond.jsonl \
    $COMMON --direct --limit 2 > logs/vllm_tp2_smoke.log 2>&1
  echo "smoke done"
elif [ "$MODE" = rest ]; then
  # Remaining datasets, same flags as the original vLLM baseline (math500 uses tools, bbh is direct).
  run math500 data/benchmarks/math500.jsonl
  run bbh     data/benchmarks/bbh.jsonl --direct
  echo "rest done"
else
  run gpqa_diamond data/benchmarks/gpqa_diamond.jsonl --direct
  run aime2024     data/benchmarks/aime2024.jsonl
  run aime2025     data/benchmarks/aime2025.jsonl
  echo "full done"
fi
