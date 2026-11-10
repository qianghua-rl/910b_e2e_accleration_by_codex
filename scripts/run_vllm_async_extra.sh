#!/bin/bash
# vLLM-Ascend baseline, AsyncLLM task-level continuous batching, NPU 1. max_num_seqs=12 (matches CB), max_model_len 4096, first-turn cap 2048, tool output clipped to 2000 chars.
# Replaces run_vllm_baseline_extra.sh (static / per-turn batching). Same protocol as run_bench_extra.sh.
# GPQA/BBH use --direct (no tools). Scoring: scorer_math.py (AIME, MATH-500), scorer_mc.py (GPQA, BBH).
# Do NOT run with `python -I` (vLLM EngineCore subprocess needs acl).
cd /remote-home/S45160/work/910b_fresh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
export ASCEND_RT_VISIBLE_DEVICES=1
PY=/remote-home/S45160/miniforge3/envs/vllm-ascend-py310/bin/python
mkdir -p results/bench_vllm logs

run() { # $1 name, $2 jsonl, $3 max-model-len, extra flags...
  local name=$1 data=$2 mml=$3; shift 3
  mkdir -p results/bench_vllm/$name
  $PY src/vllm_async_bench.py --output results/bench_vllm/$name/per_task.jsonl \
    --data "$data" --max-num-seqs 12 --max-model-len "$mml" --first-max-tokens 2048 --tool-output-max-chars 2000 "$@" \
    > logs/vllm_async_$name.log 2>&1
}
run aime2024     data/benchmarks/aime2024.jsonl 4096
run aime2025     data/benchmarks/aime2025.jsonl 4096
run math500      data/benchmarks/math500.jsonl 4096
run gpqa_diamond data/benchmarks/gpqa_diamond.jsonl 4096 --direct
run bbh          data/benchmarks/bbh.jsonl 4096 --direct
