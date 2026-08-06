#!/usr/bin/env bash
# AIME chain: all 60 problems (2024 I+II + 2025 I+II), years split in
# summaries. Gated on the Figure-24 grid finishing (same GPU): waits for
# GRID_COMPLETE in the grid log before claiming the card.
# CoT capped at 4096 new tokens — AIME chains will truncate; the length-stop
# rate is reported and all conditions share the cap (matched budget).
# Usage: CUDA_VISIBLE_DEVICES=3 bash scripts/run_aime_chain.sh
set -u
PY=/home_nfs/shivam/harvard/cs_2881_hw0/.venv/bin/python
RUN=/home_nfs/shivam/harvard/cs_2881_hw0/harvard-cs2881-hw0/scripts/run_eval.py
GRID_LOG=/var/tmp/hw0_fig24_grid_v2.log

until grep -qa GRID_COMPLETE "$GRID_LOG"; do sleep 120; done
echo "grid complete, starting AIME chain $(date -u +%H:%M:%S)"

go() {
  echo "=== START $* $(date -u +%H:%M:%S)"
  $PY "$RUN" --task aime --seed 0 --batch-size 12 "$@"
  echo "=== END $* rc=$? $(date -u +%H:%M:%S)"
}

go --mode cot    --ablation none
go --mode direct --ablation none
go --mode cot    --ablation medium
go --mode direct --ablation medium
go --mode cot    --ablation medium --direction-source random --norm-match
go --mode direct --ablation medium --direction-source random --norm-match

echo "AIME_CHAIN_COMPLETE $(date -u +%H:%M:%S)"
