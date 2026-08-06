#!/usr/bin/env bash
# MATH-500 chain: n=250 stratified by level (compute-limited; documented),
# priority-ordered so partial completion is still informative.
# Usage: CUDA_VISIBLE_DEVICES=2 bash scripts/run_math500_chain.sh
set -u
PY=/home_nfs/shivam/harvard/cs_2881_hw0/.venv/bin/python
RUN=/home_nfs/shivam/harvard/cs_2881_hw0/harvard-cs2881-hw0/scripts/run_eval.py

go() {
  echo "=== START $* $(date -u +%H:%M:%S)"
  $PY "$RUN" --task math500 --n 250 --seed 0 --batch-size 16 "$@"
  echo "=== END $* rc=$? $(date -u +%H:%M:%S)"
}

go --mode cot    --ablation none
go --mode direct --ablation none
go --mode cot    --ablation medium
go --mode direct --ablation medium
go --mode cot    --ablation medium --direction-source random --norm-match
go --mode direct --ablation medium --direction-source random --norm-match

echo "MATH500_CHAIN_COMPLETE $(date -u +%H:%M:%S)"
