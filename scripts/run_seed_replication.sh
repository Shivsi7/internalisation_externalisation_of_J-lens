#!/usr/bin/env bash
# Seed replication of the four headline GSM8K cells (cot/direct x
# clean/medium) at seed 1: the house n>=3 rule cannot be met tonight, but a
# second independent sampling seed on the load-bearing cells separates
# "sampling noise" from "effect". Gated on the post-grid chain (same card).
# Usage: CUDA_VISIBLE_DEVICES=3 bash scripts/run_seed_replication.sh
set -u
PY=/home_nfs/shivam/harvard/cs_2881_hw0/.venv/bin/python
RUN=/home_nfs/shivam/harvard/cs_2881_hw0/harvard-cs2881-hw0/scripts/run_eval.py
until grep -qa POST_GRID_CHAIN_COMPLETE /var/tmp/hw0_postgrid_chain.log; do sleep 120; done
echo "post-grid done, starting seed replication $(date -u +%H:%M:%S)"

go() {
  echo "=== START $* $(date -u +%H:%M:%S)"
  $PY "$RUN" --task gsm8k --n 200 --seed 1 --batch-size 16 "$@"
  echo "=== END $* rc=$?"
}

go --mode direct --ablation none
go --mode direct --ablation medium
go --mode cot    --ablation none   --max-new 2048
go --mode cot    --ablation medium --max-new 2048

echo "SEED_REPLICATION_COMPLETE $(date -u +%H:%M:%S)"
