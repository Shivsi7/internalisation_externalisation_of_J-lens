#!/usr/bin/env bash
# Post-grid chain (aion GPU 3): fluency control (C5, paper's own top-1
# agreement metric) then the prefix-rescue/filler experiment. Gated on the
# Fig-24 grid completing on the same card.
# Usage: CUDA_VISIBLE_DEVICES=3 bash scripts/run_post_grid_chain.sh
set -u
PY=/home_nfs/shivam/harvard/cs_2881_hw0/.venv/bin/python
SCRIPTS=/home_nfs/shivam/harvard/cs_2881_hw0/harvard-cs2881-hw0/scripts
GRID_LOG=/var/tmp/hw0_fig24_grid_v2.log

until grep -qa GRID_COMPLETE "$GRID_LOG"; do sleep 120; done
echo "grid complete, starting post-grid chain $(date -u +%H:%M:%S)"

echo "=== START fluency $(date -u +%H:%M:%S)"
MPLCONFIGDIR=/var/tmp/mpl_shivam $PY "$SCRIPTS/run_fluency_control.py"
echo "=== END fluency rc=$?"

echo "=== START prefix_rescue $(date -u +%H:%M:%S)"
$PY "$SCRIPTS/run_prefix_rescue.py" --n 150 --seed 0 --batch-size 16
echo "=== END prefix_rescue rc=$?"

echo "POST_GRID_CHAIN_COMPLETE $(date -u +%H:%M:%S)"
