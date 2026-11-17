#!/bin/bash
# vLLM-Ascend counterpart of scripts/run_tp2_slot8_all.sh. Same settings for every dataset:
#   TP=2 on NPU 0,1; 8 concurrent sequences (--max-num-seqs 8); max_model_len 8192;
#   direct/first-turn cap 6144; final-turn cap 2048; tool output clipped to 2000 chars.
# Exception: gsm8k_longctx_128 uses max_model_len 16384 (same as the self-written run; 8192 rejects tasks).
# Output: results/official/vllm_tp2_slot8/<name>/per_task.jsonl ; logs: logs/vllm/vllm_tp2_slot8_<name>.log
#
# Gate: the self-written pass (run_tp2_slot8_all.sh) uses the same two NPUs. This script waits until that
# pass has printed its done line before starting, so the two never share a card.
# Usage: bash scripts/run_vllm_tp2_slot8_all.sh          (waits, then runs)
#        bash scripts/run_vllm_tp2_slot8_all.sh --no-wait
# Do NOT run with `python -I` (vLLM EngineCore subprocess needs acl).
cd /remote-home/S45160/work/910b_fresh
DRIVER=logs/status_and_drivers/tp2_slot8_all_driver.out
if [ "$1" != "--no-wait" ]; then
  until grep -q "tp2_slot8 full pass done" "$DRIVER" 2>/dev/null; do sleep 120; done
  echo "self-written pass finished, starting vLLM pass $(date +%F_%T)"
fi

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
export ASCEND_RT_VISIBLE_DEVICES=0,1
# spawn + single OMP thread: avoids "Invalid thread pool" worker crashes (see run_vllm_tp2_long.sh)
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=1
PY=/remote-home/S45160/miniforge3/envs/vllm-ascend-py310/bin/python
OUT=results/official/vllm_tp2_slot8
mkdir -p $OUT logs/vllm

BASE="--tensor-parallel-size 2 --max-num-seqs 8 --first-max-tokens 6144 --final-max-tokens 2048 --direct-max-tokens 6144 --tool-output-max-chars 2000"

run() { # $1 name, $2 max_model_len, $3 data, extra flags...
  local name=$1 mml=$2 data=$3; shift 3
  if [ -s $OUT/$name/per_task.jsonl ] && grep -q '"kind": "summary"' $OUT/$name/per_task.jsonl; then
    echo "already complete, skip: $name"; return 0
  fi
  mkdir -p $OUT/$name
  $PY src/vllm_async_bench.py --output $OUT/$name/per_task.jsonl --data "$data" $BASE \
    --max-model-len $mml "$@" > logs/vllm/vllm_tp2_slot8_$name.log 2>&1
  echo "$name finished $(date +%F_%T)"
}

run gpqa_diamond      8192  data/benchmarks/gpqa_diamond.jsonl --direct
run aime2024          8192  data/benchmarks/aime2024.jsonl
run aime2025          8192  data/benchmarks/aime2025.jsonl
run math500           8192  data/benchmarks/math500.jsonl
run bbh               8192  data/benchmarks/bbh.jsonl --direct
run gsm8k_longctx_128 16384 data/benchmarks/gsm8k_longctx_128.jsonl
echo "vllm_tp2_slot8 full pass done $(date +%F_%T)"
