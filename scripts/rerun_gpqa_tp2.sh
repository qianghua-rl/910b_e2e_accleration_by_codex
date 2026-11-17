#!/bin/bash
# Re-run GPQA Diamond, TP=2 on NPU 0+1, self-written CB (graph decode) vs vLLM-Ascend, strictly sequential
# (never two jobs on the same cards). Order: self r1 -> vLLM r1 -> self r2 (self repeated to bracket drift).
# Settings are taken from scripts/run_cb_tp2_graph.sh (self) and scripts/run_vllm_tp2_long.sh (vLLM);
# both use data/benchmarks/gpqa_diamond.jsonl (sha256 4a13446b...), greedy, enable_thinking=False,
# direct mode, DIRECT system prompt, t_max/max_model_len 8192, cap 6144 (direct), 12 concurrent slots.
# Outputs are NEW directories; existing results are not touched.
# Usage: nohup bash scripts/rerun_gpqa_tp2.sh > logs/rerun_gpqa_tp2_driver.out 2>&1 &
cd /remote-home/S45160/work/910b_fresh
DATA=data/benchmarks/gpqa_diamond.jsonl
SELF_PY=/remote-home/S45160/miniforge3/envs/ascend-e2e/bin/python
VLLM_PY=/remote-home/S45160/miniforge3/envs/vllm-ascend-py310/bin/python
SELF_OUT=results/official/continuous_batching_tp2
VLLM_OUT=results/official/bench_vllm_tp2
mkdir -p logs

self_run() { # $1 tag
  local dir=$SELF_OUT/rerun_gpqa_$1
  if [ -e $dir/per_task.jsonl ]; then echo "exists, skip self $1"; return; fi
  mkdir -p $dir
  for r in 0 1; do
    RANK=$r ASCEND_RT_VISIBLE_DEVICES=$r $SELF_PY -I src/cb_run.py --out $dir/per_task.jsonl \
      --data $DATA --tp 2 --fast-gdn --gdn-v2 --elem-conv --prefill-group 1 --slots 12 --t-max 8192 \
      --direct --direct-max-new-tokens 6144 --first-max-new-tokens 6144 --final-max-new-tokens 2048 \
      --tool-output-max-chars 2000 --graph-decode > logs/rerun_self_${1}_rank$r.log 2>&1 &
  done
  wait
  echo "self $1 finished $(date +%F_%T)"
}

vllm_run() { # $1 tag
  local dir=$VLLM_OUT/rerun_gpqa_$1
  if [ -e $dir/per_task.jsonl ]; then echo "exists, skip vllm $1"; return; fi
  mkdir -p $dir
  (
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
    export ASCEND_RT_VISIBLE_DEVICES=0,1
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    export OMP_NUM_THREADS=1
    $VLLM_PY src/vllm_async_bench.py --output $dir/per_task.jsonl --data $DATA \
      --tensor-parallel-size 2 --max-num-seqs 12 --max-model-len 8192 --first-max-tokens 6144 \
      --final-max-tokens 2048 --direct-max-tokens 6144 --tool-output-max-chars 2000 --direct \
      > logs/rerun_vllm_${1}.log 2>&1
  )
  echo "vllm $1 finished $(date +%F_%T)"
}

echo "start $(date +%F_%T)"
self_run r1
vllm_run r1
self_run r2
echo "all done $(date +%F_%T)"
