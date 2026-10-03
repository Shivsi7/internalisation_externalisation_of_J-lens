#!/usr/bin/env bash
# Waits for the extension-1 seed campaign to finish on this card, then runs
# the full task-balance grid (extension 3) on it. Keeps substantial GPU work
# to one lab card at a time.
# Usage: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
#          bash scripts/queue_task_balance_after_seeds.sh
set -u
PY=/home_nfs/shivam/harvard/cs_2881_hw0/.venv/bin/python
SCRIPT=/home_nfs/shivam/harvard/cs_2881_hw0/harvard-cs2881-hw0/scripts/run_task_balance.py
until grep -qa SEED_CAMPAIGN_EXT1_COMPLETE /var/tmp/hw0_ext1_seeds.log; do
  if grep -qaE 'Traceback|CUDA out of memory' /var/tmp/hw0_ext1_seeds.log; then
    echo "seed campaign failed, aborting task-balance queue"; exit 1
  fi
  sleep 300
done
echo "seed campaign done, starting task balance $(date -u +%H:%M:%S)"
$PY -u "$SCRIPT" --n 80 --cot-max-new 2048 --batch-size 16
echo "TASK_BALANCE_COMPLETE $(date -u +%H:%M:%S)"
