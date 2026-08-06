#!/usr/bin/env bash
# Deprioritized tail (aion 3): after the seed replication completes, run the
# requeued GSM8K CoT random-control cell (nice-to-have).
# Usage: CUDA_VISIBLE_DEVICES=3 bash scripts/run_tail_aion3.sh
set -u
PY=/home_nfs/shivam/harvard/cs_2881_hw0/.venv/bin/python
RUN=/home_nfs/shivam/harvard/cs_2881_hw0/harvard-cs2881-hw0/scripts/run_gsm8k.py
until grep -qa SEED_REPLICATION_COMPLETE /var/tmp/hw0_seedrep.log; do sleep 120; done
echo "seed replication done, running requeued CoT random cell $(date -u +%H:%M:%S)"
$PY "$RUN" --n 200 --seed 0 --batch-size 16 --max-new 2048 --mode cot --ablation medium --direction-source random --norm-match
echo "TAIL_A3_COMPLETE $(date -u +%H:%M:%S)"
