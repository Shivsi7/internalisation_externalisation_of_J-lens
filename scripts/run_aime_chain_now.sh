#!/usr/bin/env bash
# AIME chain, ungated variant: aion 0's thesis chain was verified complete
# (all 6 ladder stages DONE, final.pkl present, card idle), so this runs
# immediately there instead of queuing behind the Fig-24 grid on GPU 3.
# Identical run list to run_aime_chain.sh.
# Usage: CUDA_VISIBLE_DEVICES=0 bash scripts/run_aime_chain_now.sh
set -u
PY=/home_nfs/shivam/harvard/cs_2881_hw0/.venv/bin/python
RUN=/home_nfs/shivam/harvard/cs_2881_hw0/harvard-cs2881-hw0/scripts/run_eval.py

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
