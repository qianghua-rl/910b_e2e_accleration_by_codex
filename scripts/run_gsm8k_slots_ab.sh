#!/bin/bash
# Controlled A/B for the gsm8k_longctx_128 slowdown: same code, same flags, only --slots differs.
# Runs sequentially (both TP=2 ranks per run, one run at a time) so the NPUs are never shared.
# Outputs: results/investigations/gsm8k_slowdown/runs/<tag>/per_task.jsonl
# Logs:    logs/investigations/gsm8k_slowdown/<tag>_rank{0,1}.log
# Usage:   nohup bash scripts/run_gsm8k_slots_ab.sh > logs/investigations/gsm8k_slowdown/driver.out 2>&1 &
cd /remote-home/S45160/work/910b_fresh
PY=/remote-home/S45160/miniforge3/envs/ascend-e2e/bin/python
RUNS=results/investigations/gsm8k_slowdown/runs
DATA=data/benchmarks/gsm8k_longctx_128.jsonl
BASE="--tp 2 --fast-gdn --gdn-v2 --elem-conv --prefill-group 1 --direct-max-new-tokens 6144 --first-max-new-tokens 6144 --final-max-new-tokens 2048 --tool-output-max-chars 2000 --graph-decode --t-max 16384"

run() { # $1 tag, $2 slots
  local tag=$1 slots=$2
  if [ -e $RUNS/$tag/per_task.jsonl ]; then echo "exists, skip: $tag"; return 0; fi
  mkdir -p $RUNS/$tag
  echo "start $tag slots=$slots $(date +%F_%T)"
  for r in 0 1; do
    RANK=$r ASCEND_RT_VISIBLE_DEVICES=$r $PY -I src/cb_run.py --out $RUNS/$tag/per_task.jsonl \
      --data $DATA $BASE --slots $slots > logs/investigations/gsm8k_slowdown/${tag}_rank$r.log 2>&1 &
  done
  wait
  echo "done $tag $(date +%F_%T)"
}

run cb_slots8_today 8
run cb_slots12_today 12
echo "ab done $(date +%F_%T)"
