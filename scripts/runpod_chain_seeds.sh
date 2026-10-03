#!/usr/bin/env bash
# MATH-500 headline cells at the seeds given in $SEEDS (space-separated).
set -u
cd /workspace/hw0
for SEED in $SEEDS; do
  for args in "--mode cot --ablation medium" "--mode cot --ablation none" \
              "--mode direct --ablation medium" "--mode direct --ablation none"; do
    echo "=== START math500 seed=$SEED $args $(date -u +%H:%M:%S)"
    python -u scripts/run_eval.py --task math500 --n 250 --seed "$SEED" \
      --batch-size 16 $args
    echo "=== END math500 seed=$SEED $args rc=$?"
  done
done
echo "MATH500_SEEDS_COMPLETE ($SEEDS) $(date -u +%H:%M:%S)"
