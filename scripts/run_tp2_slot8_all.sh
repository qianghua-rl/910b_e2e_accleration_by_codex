#!/bin/bash
# First full pass of every benchmark on the self-written CB engine, TP=2, 8 slots, CUDA-graph-style decode
# (--graph-decode). One shared setting for all datasets so the numbers are comparable:
#   slots 8, t_max 8192, first-turn cap 6144, final-turn cap 2048, tool output 2000 chars, prefill group 1.
# Exception: gsm8k_longctx_128 needs t_max 16384 (at 8192, 114 tasks are rejected for context), so it is run
# at 16384 with the same caps; this is the only deviation and is recorded in the metadata.
# Output: results/official/tp2_slot8/<name>/per_task.jsonl ; logs: logs/tp2/tp2_slot8_<name>_rank{0,1}.log
# Usage: bash scripts/run_tp2_slot8_all.sh
cd /remote-home/S45160/work/910b_fresh
PY=/remote-home/S45160/miniforge3/envs/ascend-e2e/bin/python
OUT=results/official/tp2_slot8
mkdir -p $OUT logs/tp2

BASE="--tp 2 --fast-gdn --gdn-v2 --elem-conv --prefill-group 1 --slots 8 --direct-max-new-tokens 6144 --first-max-new-tokens 6144 --final-max-new-tokens 2048 --tool-output-max-chars 2000 --graph-decode"

run() { # $1 name, $2 t_max, $3 data, extra flags...
  local name=$1 tmax=$2 data=$3; shift 3
  if [ -s $OUT/$name/per_task.jsonl ] && grep -q '"kind": "summary"' $OUT/$name/per_task.jsonl; then
    echo "already complete, skip: $name"; return 0
  fi
  mkdir -p $OUT/$name
  for r in 0 1; do
    RANK=$r ASCEND_RT_VISIBLE_DEVICES=$r $PY -I src/cb_run.py --out $OUT/$name/per_task.jsonl \
      --data "$data" $BASE --t-max $tmax "$@" > logs/tp2/tp2_slot8_${name}_rank$r.log 2>&1 &
  done
  wait
  echo "$name finished $(date +%F_%T)"
}

run gpqa_diamond 8192 data/benchmarks/gpqa_diamond.jsonl --direct
run aime2024     8192 data/benchmarks/aime2024.jsonl
run aime2025     8192 data/benchmarks/aime2025.jsonl
run math500      8192 data/benchmarks/math500.jsonl
run bbh          8192 data/benchmarks/bbh.jsonl --direct
run gsm8k_longctx_128 16384 data/benchmarks/gsm8k_longctx_128.jsonl
echo "tp2_slot8 full pass done $(date +%F_%T)"
