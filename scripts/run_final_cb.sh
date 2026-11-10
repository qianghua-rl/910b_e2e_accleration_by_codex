#!/bin/bash
# CB (our engine), NPU 0. 12 slots, t_max 4096, first-turn cap 2048, tool output clipped to 2000 chars.
# BBH is not rerun here: its earlier CB run is direct-mode (no tools) and unaffected by these flags.
cd /remote-home/S45160/work/910b_fresh
PY=/remote-home/S45160/miniforge3/envs/ascend-e2e/bin/python
export ASCEND_RT_VISIBLE_DEVICES=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
COMMON="--slots 12 --t-max 4096 --first-max-new-tokens 2048 --tool-output-max-chars 2000 --fast-gdn --gdn-v2 --prefill-group 1 --graph-decode --elem-conv"
run() { local name=$1 data=$2; shift 2
  $PY -I src/cb_run.py --data "$data" --out results/bench/$name/per_task.jsonl --device npu:0 $COMMON "$@" > logs/bench_$name.log 2>&1
}
run aime2024 data/benchmarks/aime2024.jsonl
run aime2025 data/benchmarks/aime2025.jsonl
run math500 data/benchmarks/math500.jsonl
run gpqa_diamond data/benchmarks/gpqa_diamond.jsonl --direct
