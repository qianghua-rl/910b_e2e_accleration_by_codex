#!/bin/bash
# Sequential decode op profiles on NPU 1: conv1d (A) vs elem conv (B).
PY=/remote-home/S45160/miniforge3/envs/ascend-e2e/bin/python
cd /remote-home/S45160/work/910b_fresh
export ASCEND_RT_VISIBLE_DEVICES=1
rm -rf results/profile_decode_A results/profile_decode_B
$PY -I scripts/profile_cb_decode_ops.py results/profile_decode_A > logs/prof_decode_A.log 2>&1
ELEM_CONV=1 $PY -I scripts/profile_cb_decode_ops.py results/profile_decode_B > logs/prof_decode_B.log 2>&1
echo DONE > logs/prof_decode_status.txt
