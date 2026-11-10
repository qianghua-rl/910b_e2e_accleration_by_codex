#!/bin/bash
# Extra benchmarks (MATH-500, AIME 2024, AIME 2025) with the current best CB config.
# Same agent protocol as gsm8k_128. Sequential on NPU 0.
# Scoring: MATH-500 -> src/scorer_math.py (fresh-math-v1, offline re-score of per_task.jsonl).
#          AIME     -> src/scorer.py     (fresh-v1, numeric answers only).
cd /remote-home/S45160/work/910b_fresh
PY=/remote-home/S45160/miniforge3/envs/ascend-e2e/bin/python
export ASCEND_RT_VISIBLE_DEVICES=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
run() { # $1 dataset name, $2 jsonl path
  $PY -I src/cb_run.py --data "$2" --out results/bench/$1/per_task.jsonl --device npu:0 \
    --slots 16 --fast-gdn --gdn-v2 --prefill-group 1 --graph-decode --elem-conv \
    > logs/bench_$1.log 2>&1
}
run aime2024 data/benchmarks/aime2024.jsonl
run aime2025 data/benchmarks/aime2025.jsonl
run math500  data/benchmarks/math500.jsonl

# Non-math benchmarks: direct answering (no tools), scored offline with scorer_mc.py.
run_direct() { # $1 dataset name, $2 jsonl path
  $PY -I src/cb_run.py --direct --data "$2" --out results/bench/$1/per_task.jsonl --device npu:0 \
    --slots 16 --fast-gdn --gdn-v2 --prefill-group 1 --graph-decode --elem-conv \
    > logs/bench_$1.log 2>&1
}
run_direct gpqa_diamond data/benchmarks/gpqa_diamond.jsonl
run_direct bbh          data/benchmarks/bbh.jsonl
