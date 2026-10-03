#!/usr/bin/env bash
# Waits for the task-balance run to finish on this card, then runs the A6.5
# prefix-rescue battery (full-set cells + filler variants).
set -u
PY=/home_nfs/shivam/harvard/cs_2881_hw0/.venv/bin/python
S=/home_nfs/shivam/harvard/cs_2881_hw0/harvard-cs2881-hw0/scripts
until grep -qa TASK_BALANCE_COMPLETE /var/tmp/hw0_taskbal.log; do
  if grep -qaE 'aborting|Traceback|CUDA out of memory' /var/tmp/hw0_taskbal.log; then
    echo "task balance failed, aborting battery queue"; exit 1
  fi
  sleep 300
done
echo "task balance done, starting rescue battery $(date -u +%H:%M:%S)"
$PY -u "$S/run_prefix_rescue.py" --battery --n 200
echo "RESCUE_BATTERY_COMPLETE $(date -u +%H:%M:%S)"
