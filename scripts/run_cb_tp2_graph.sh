#!/bin/bash
# Same settings as scripts/run_cb_tp2.sh full, plus --graph-decode (TP=2 NPUGraph decode, verified in
# scripts/test_hccl_graph.py and the GSM8K longctx run). Runs the five datasets one after another.
# Output: results/ablations/continuous_batching_tp2/graph_<name>/per_task.jsonl (new dirs; existing
# eager baselines in results/continuous_batching_tp2/<name>/ are NOT touched).
# Usage: bash scripts/run_cb_tp2_graph.sh [dataset ...]   (default: all five)
cd /remote-home/S45160/work/910b_fresh
PY=/remote-home/S45160/miniforge3/envs/ascend-e2e/bin/python
OUT=results/ablations/continuous_batching_tp2
mkdir -p $OUT logs

COMMON="--tp 2 --fast-gdn --gdn-v2 --elem-conv --prefill-group 1 --slots 12 --t-max 8192 --direct-max-new-tokens 6144 --first-max-new-tokens 6144 --final-max-new-tokens 2048 --tool-output-max-chars 2000 --graph-decode"

run() { # $1 name, $2 data, extra flags...
  local name=$1 data=$2; shift 2
  if [ -e $OUT/graph_$name/per_task.jsonl ]; then echo "exists, skip: $name"; return 1; fi
  mkdir -p $OUT/graph_$name
  for r in 0 1; do
    RANK=$r ASCEND_RT_VISIBLE_DEVICES=$r $PY -I src/cb_run.py --out $OUT/graph_$name/per_task.jsonl \
      --data "$data" $COMMON "$@" > logs/cb_tp2_graph_${name}_rank$r.log 2>&1 &
  done
  wait
  echo "$name finished $(date +%H:%M:%S)"
}

WANT=("$@")
want() { # $1 dataset name: run it if no names were given on the command line, or if it is listed
  [ ${#WANT[@]} -eq 0 ] || [[ " ${WANT[*]} " == *" $1 "* ]]; }
want gpqa_diamond && run gpqa_diamond data/benchmarks/gpqa_diamond.jsonl --direct
want aime2024     && run aime2024     data/benchmarks/aime2024.jsonl
want aime2025     && run aime2025     data/benchmarks/aime2025.jsonl
want math500      && run math500      data/benchmarks/math500.jsonl
want bbh          && run bbh          data/benchmarks/bbh.jsonl --direct
echo "graph queue done"
