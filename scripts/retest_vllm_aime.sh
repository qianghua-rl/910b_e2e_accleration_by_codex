#!/bin/bash
# Re-run vLLM-Ascend AIME 2024/2025 with the same settings as our cb_run results:
#   max_model_len 4096, first-turn cap 2048, final cap 512, tool output clipped to 2000 chars.
# Earlier results in results/bench_vllm/aime* were produced with first-turn cap 512 and
# max_model_len 2048, so every task was truncated before its first tool call.
# Output goes to a new directory; the old results are kept untouched.
# Do NOT run with `python -I` (vLLM EngineCore subprocess needs acl).
cd /remote-home/S45160/work/910b_fresh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
export ASCEND_RT_VISIBLE_DEVICES=1
PY=/remote-home/S45160/miniforge3/envs/vllm-ascend-py310/bin/python
OUT=results/bench_vllm_retest
mkdir -p $OUT logs

for name in aime2024 aime2025; do
  mkdir -p $OUT/$name
  $PY src/vllm_async_bench.py --output $OUT/$name/per_task.jsonl \
    --data data/benchmarks/$name.jsonl --max-num-seqs 12 --max-model-len 4096 \
    --first-max-tokens 2048 --final-max-tokens 512 --tool-output-max-chars 2000 \
    > logs/vllm_retest_$name.log 2>&1
done
echo "retest done"
