#!/usr/bin/env bash
# Controls chain (cedar GPU1, 48GB): the three zero-new-code controls that
# close the report's open interpretive holes, priority-ordered.
#   B8  exclusion filter OFF   - is the headline partly the filter's asymmetry?
#   B3  rank-shifted J-vectors - same dictionary, ranks 51-60: "the TOP of the
#                                J-space specifically" (strongest control)
#   B5  logit-lens directions  - does the Jacobian correction matter at all?
# All at GSM8K medium, n=200, seed 0 — directly comparable to grid v2.
# Usage: CUDA_VISIBLE_DEVICES=1 bash scripts/run_controls_chain.sh
set -u
PY=/home_nfs/shivam/harvard/cs_2881_hw0/.venv/bin/python
RUN=/home_nfs/shivam/harvard/cs_2881_hw0/harvard-cs2881-hw0/scripts/run_eval.py

go() {
  echo "=== START $* $(date -u +%H:%M:%S)"
  $PY "$RUN" --task gsm8k --n 200 --seed 0 --batch-size 16 "$@"
  echo "=== END $* rc=$? $(date -u +%H:%M:%S)"
}

go --mode cot    --ablation medium --no-filter
go --mode direct --ablation medium --no-filter
go --mode cot    --ablation medium --direction-source jlens_rankshift
go --mode direct --ablation medium --direction-source jlens_rankshift
go --mode cot    --ablation medium --direction-source logitlens
go --mode direct --ablation medium --direction-source logitlens

echo "CONTROLS_CHAIN_COMPLETE $(date -u +%H:%M:%S)"
