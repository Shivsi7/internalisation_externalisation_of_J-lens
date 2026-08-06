#!/usr/bin/env bash
# Ungated variant of run_post_grid_chain.sh: the grid's final run (CoT
# random control) was deprioritized to the tail by Shivam, so GRID_COMPLETE
# will never appear; this starts immediately. Appends to the ORIGINAL
# postgrid log so the seed-replication gate still fires on its marker.
# Usage: CUDA_VISIBLE_DEVICES=3 bash scripts/run_post_grid_now.sh
set -u
PY=/home_nfs/shivam/harvard/cs_2881_hw0/.venv/bin/python
SCRIPTS=/home_nfs/shivam/harvard/cs_2881_hw0/harvard-cs2881-hw0/scripts

echo "=== START fluency $(date -u +%H:%M:%S)"
MPLCONFIGDIR=/var/tmp/mpl_shivam $PY "$SCRIPTS/run_fluency_control.py"
echo "=== END fluency rc=$?"

echo "=== START prefix_rescue $(date -u +%H:%M:%S)"
$PY "$SCRIPTS/run_prefix_rescue.py" --n 150 --seed 0 --batch-size 16
echo "=== END prefix_rescue rc=$?"

echo "POST_GRID_CHAIN_COMPLETE $(date -u +%H:%M:%S)"
