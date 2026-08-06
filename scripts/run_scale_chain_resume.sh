#!/usr/bin/env bash
# Resume of run_scale_chain.sh after the cedar device-order swap: the first
# two 1.7B runs (cot none, direct none) completed before the restart and are
# skipped here. Run with CUDA_DEVICE_ORDER=PCI_BUS_ID so device 0 is the
# 3090 as nvidia-smi reports it.
# Usage: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 bash run_scale_chain_resume.sh
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

go Qwen/Qwen3-1.7B "$L17" 16 --mode cot    --ablation medium
go Qwen/Qwen3-1.7B "$L17" 16 --mode direct --ablation medium
go Qwen/Qwen3-8B  "$L8"   6 --mode direct --ablation none
go Qwen/Qwen3-8B  "$L8"   6 --mode direct --ablation medium
go Qwen/Qwen3-8B  "$L8"   6 --mode cot    --ablation none
go Qwen/Qwen3-8B  "$L8"   6 --mode cot    --ablation medium

echo "SCALE_CHAIN_COMPLETE $(date -u +%H:%M:%S)"
