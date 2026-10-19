#!/usr/bin/env bash
# Source this file from the repository root. Restrict all experiments to one NPU.
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:${LD_LIBRARY_PATH:-}
export ASCEND_RT_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PATH=/remote-home/S45160/miniforge3/envs/ascend-e2e/bin:$PATH
