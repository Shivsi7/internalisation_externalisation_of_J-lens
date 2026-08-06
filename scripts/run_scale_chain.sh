#!/usr/bin/env bash
# Scale sweep C7 (cedar GPU0, RTX 3090 24GB): Qwen3-1.7B and Qwen3-8B on
# GSM8K {cot,direct} x {clean, medium}, using their own pre-fitted lenses.
# Answers the paper's §9.1 open question (do smaller models have a workspace
# at all?) and converts the H3 risk into a trend measurement.
# 8B runs use batch 6 (16GB weights + paired KV caches on a 24GB card).
# Usage: CUDA_VISIBLE_DEVICES=0 bash scripts/run_scale_chain.sh
set -u
PY=/home_nfs/shivam/harvard/cs_2881_hw0/.venv/bin/python
RUN=/home_nfs/shivam/harvard/cs_2881_hw0/harvard-cs2881-hw0/scripts/run_eval.py
LENSES=/home_nfs/shivam/harvard/cs_2881_hw0/harvard-cs2881-hw0/lenses

go() {
  local model=$1 lens=$2 bs=$3; shift 3
  echo "=== START $model $* $(date -u +%H:%M:%S)"
  $PY "$RUN" --task gsm8k --n 200 --seed 0 --batch-size "$bs" \
      --model "$model" --lens-path "$lens" "$@"
  echo "=== END $model $* rc=$? $(date -u +%H:%M:%S)"
}

L17=$LENSES/qwen3-1.7b/qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt
L8=$LENSES/qwen3-8b/qwen3-8b/jlens/Salesforce-wikitext/Qwen3-8B_jacobian_lens.pt

go Qwen/Qwen3-1.7B "$L17" 16 --mode cot    --ablation none
go Qwen/Qwen3-1.7B "$L17" 16 --mode direct --ablation none
go Qwen/Qwen3-1.7B "$L17" 16 --mode cot    --ablation medium
go Qwen/Qwen3-1.7B "$L17" 16 --mode direct --ablation medium
go Qwen/Qwen3-8B  "$L8"   6 --mode direct --ablation none
go Qwen/Qwen3-8B  "$L8"   6 --mode direct --ablation medium
go Qwen/Qwen3-8B  "$L8"   6 --mode cot    --ablation none
go Qwen/Qwen3-8B  "$L8"   6 --mode cot    --ablation medium

echo "SCALE_CHAIN_COMPLETE $(date -u +%H:%M:%S)"
