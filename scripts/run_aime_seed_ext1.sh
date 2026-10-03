#!/usr/bin/env bash
# Extension 1 part 2a: AIME headline cells at one additional seed, on the
# second cedar card overnight (batch 8 for the 3090's 24 GB). MATH-500 seeds
# are too large for a lab night and go to the HPC.
# Usage: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 SEED=1 \
#          bash scripts/run_aime_seed_ext1.sh
set -u
PY=/home_nfs/shivam/harvard/cs_2881_hw0/.venv/bin/python
RUN=/home_nfs/shivam/harvard/cs_2881_hw0/harvard-cs2881-hw0/scripts/run_eval.py
SEED=${SEED:-1}
go() {
  echo "=== START seed=$SEED $* $(date -u +%H:%M:%S)"
  $PY "$RUN" --task aime --seed "$SEED" --batch-size 8 "$@"
  echo "=== END seed=$SEED $* rc=$? $(date -u +%H:%M:%S)"
}
go --mode cot    --ablation medium
go --mode cot    --ablation none
go --mode direct --ablation medium
go --mode direct --ablation none
echo "AIME_SEED_${SEED}_COMPLETE $(date -u +%H:%M:%S)"
