#!/usr/bin/env bash
# Figure 24 mini-replication grid: GSM8K CoT vs direct under J-space ablation.
# 10 runs, n=200, seed 0. Baselines first, then the headline medium pair, then
# light/heavy, then the norm-matched random control at medium (paper's control).
# Usage: CUDA_VISIBLE_DEVICES=3 bash scripts/run_fig24_grid.sh
set -u
PY=/home_nfs/shivam/harvard/cs_2881_hw0/.venv/bin/python
RUN=/home_nfs/shivam/harvard/cs_2881_hw0/harvard-cs2881-hw0/scripts/run_gsm8k.py
N=200

run() {
  echo "=== START $* $(date -u +%H:%M:%S)"
  $PY "$RUN" --n $N --seed 0 --batch-size 16 "$@"
  echo "=== END $* rc=$? $(date -u +%H:%M:%S)"
}

run --mode cot    --ablation none
run --mode direct --ablation none
run --mode cot    --ablation medium
run --mode direct --ablation medium
run --mode cot    --ablation light
run --mode direct --ablation light
run --mode cot    --ablation heavy
run --mode direct --ablation heavy
run --mode cot    --ablation medium --direction-source random --norm-match
run --mode direct --ablation medium --direction-source random --norm-match

echo "GRID_COMPLETE $(date -u +%H:%M:%S)"
