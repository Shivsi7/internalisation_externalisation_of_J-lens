#!/usr/bin/env bash
# Figure 24 mini-replication grid, v2.
#
# v1 (run_fig24_grid.sh, results 20260805_1405..1433) was compromised by the
# CoT generation cap: 47% of clean-CoT chains hit max_new=1024 (length-stop),
# 37.5% under medium ablation, so the CoT accuracies partly measured
# truncation, not ablation. v2 raises CoT max_new to 2048 and direct to 32
# (90/200 ablated-direct runs hit the 16-token cap: ablated direct mode stops
# emitting EOS). v1 results are preserved, not overwritten.
#
# Usage: CUDA_VISIBLE_DEVICES=3 bash scripts/run_fig24_grid_v2.sh
set -u
PY=/home_nfs/shivam/harvard/cs_2881_hw0/.venv/bin/python
RUN=/home_nfs/shivam/harvard/cs_2881_hw0/harvard-cs2881-hw0/scripts/run_gsm8k.py
N=200

cot()    { echo "=== START cot $* $(date -u +%H:%M:%S)";    $PY "$RUN" --n $N --seed 0 --batch-size 16 --max-new 2048 --mode cot    "$@"; echo "=== END cot $* rc=$?"; }
direct() { echo "=== START direct $* $(date -u +%H:%M:%S)"; $PY "$RUN" --n $N --seed 0 --batch-size 16 --max-new 32   --mode direct "$@"; echo "=== END direct $* rc=$?"; }

direct --ablation none
direct --ablation medium
direct --ablation light
direct --ablation heavy
direct --ablation medium --direction-source random --norm-match
cot    --ablation none
cot    --ablation medium
cot    --ablation light
cot    --ablation heavy
cot    --ablation medium --direction-source random --norm-match

echo "GRID_COMPLETE $(date -u +%H:%M:%S)"
