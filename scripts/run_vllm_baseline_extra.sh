#!/bin/bash
# vLLM-Ascend baseline (continuous batching via vLLM scheduler: all tasks submitted per turn, max_num_seqs=16), NPU 1.
# Same agent protocol as run_bench_extra.sh; GPQA/BBH use --direct (no tools).
# Scoring: offline with scorer_math.py (AIME, MATH-500) and scorer_mc.py (GPQA, BBH), reading final_text/gold.
# Do NOT run with `python -I` (vLLM EngineCore subprocess needs acl).
cd /remote-home/S45160/work/910b_fresh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
export ASCEND_RT_VISIBLE_DEVICES=1
PY=/remote-home/S45160/miniforge3/envs/vllm-ascend-py310/bin/python
mkdir -p results/bench_vllm logs

run() { # $1 name, $2 jsonl, $3 total tasks, $4 max-model-len, extra flags...
  local name=$1 data=$2 n=$3 mml=$4; shift 4
  mkdir -p results/bench_vllm/$name
  $PY src/vllm_bench_extra.py --output results/bench_vllm/$name/per_task.jsonl \
    --data "$data" --limit "$n" --batch-size 100000 --max-num-seqs 16 --max-model-len "$mml" "$@" \
    > logs/vllm_bench_$name.log 2>&1
}
run aime2024 data/benchmarks/aime2024.jsonl 30 2048
run aime2025 data/benchmarks/aime2025.jsonl 30 2048
run math500  data/benchmarks/math500.jsonl 500 2048
run gpqa_diamond data/benchmarks/gpqa_diamond.jsonl 198 4096 --direct
run bbh      data/benchmarks/bbh.jsonl 1350 4096 --direct
