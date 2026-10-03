#!/usr/bin/env bash
# Extension 1, part 1: seed replication of the four headline GSM8K cells
# (cot/direct x clean/medium) at seeds 2, 3, 4, bringing the load-bearing
# cells to five independent sampling seeds (0 and 1 already exist). One GPU,
# sequential; measured cost ~72 min/seed on an RTX 6000 Ada, so ~4 h total.
# MATH-500 and AIME seed replication is deferred to the HPC.
# Usage: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
#          bash scripts/run_seed_campaign_ext1.sh
set -u
PY=/home_nfs/shivam/harvard/cs_2881_hw0/.venv/bin/python
RUN=/home_nfs/shivam/harvard/cs_2881_hw0/harvard-cs2881-hw0/scripts/run_eval.py

go() {
  local seed=$1; shift
  echo "=== START seed=$seed $* $(date -u +%H:%M:%S)"
  $PY "$RUN" --task gsm8k --n 200 --seed "$seed" --batch-size 16 "$@"
  echo "=== END seed=$seed $* rc=$?"
}

for SEED in 2 3 4; do
  go "$SEED" --mode cot    --ablation medium --max-new 2048
  go "$SEED" --mode cot    --ablation none   --max-new 2048
  go "$SEED" --mode direct --ablation medium
  go "$SEED" --mode direct --ablation none
done

echo "SEED_CAMPAIGN_EXT1_COMPLETE $(date -u +%H:%M:%S)"
