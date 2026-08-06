#!/usr/bin/env bash
# Deprioritized tail (aion 0): after the dual-task run completes, run the
# requeued AIME random-control pair (nice-to-have; slot in if it lands).
# Usage: CUDA_VISIBLE_DEVICES=0 bash scripts/run_tail_aion0.sh
set -u
PY=/home_nfs/shivam/harvard/cs_2881_hw0/.venv/bin/python
RUN=/home_nfs/shivam/harvard/cs_2881_hw0/harvard-cs2881-hw0/scripts/run_eval.py
until grep -qa DUALTASK_COMPLETE /var/tmp/hw0_dualtask_a0.log; do sleep 120; done
echo "dual-task done, running requeued AIME randoms $(date -u +%H:%M:%S)"
$PY "$RUN" --task aime --seed 0 --batch-size 12 --mode cot --ablation medium --direction-source random --norm-match
$PY "$RUN" --task aime --seed 0 --batch-size 12 --mode direct --ablation medium --direction-source random --norm-match
echo "TAIL_A0_COMPLETE $(date -u +%H:%M:%S)"
