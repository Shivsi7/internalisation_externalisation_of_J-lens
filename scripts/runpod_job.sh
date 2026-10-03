#!/usr/bin/env bash
# Argument-driven job dispatcher for RunPod pods. No environment variables:
# the job name is the single argument, so nothing can be word-split away.
# Usage: bash scripts/runpod_job.sh <seed1|seed2|seed3|seed4|ksweep|14b_a|14b_b>
set -u
cd /workspace/hw0
JOB="${1:?job name required}"
run() { echo "=== START $* $(date -u +%H:%M:%S)"; python -u scripts/run_eval.py "$@"; echo "=== END $* rc=$?"; }
M14="Qwen/Qwen3-14B"
L14="lenses/qwen3-14b/qwen3-14b/jlens/Salesforce-wikitext/Qwen3-14B_jacobian_lens.pt"
case "$JOB" in
  seed1|seed2|seed3|seed4)
    S="${JOB#seed}"
    run --task math500 --n 250 --seed "$S" --batch-size 16 --mode cot --ablation medium
    run --task math500 --n 250 --seed "$S" --batch-size 16 --mode cot --ablation none
    run --task math500 --n 250 --seed "$S" --batch-size 16 --mode direct --ablation medium
    run --task math500 --n 250 --seed "$S" --batch-size 16 --mode direct --ablation none
    ;;
  ksweep)
    for K in 5 20; do
      run --task gsm8k --n 200 --seed 0 --batch-size 16 --ablation medium --k "$K" --mode cot --max-new 2048
      run --task gsm8k --n 200 --seed 0 --batch-size 16 --ablation medium --k "$K" --mode direct
    done
    ;;
  14b_a)
    python scripts/fetch_lens.py --model qwen3-14b
    run --task gsm8k --n 200 --seed 0 --batch-size 8 --model "$M14" --lens-path "$L14" --mode cot --ablation medium --max-new 2048
    run --task gsm8k --n 200 --seed 0 --batch-size 8 --model "$M14" --lens-path "$L14" --mode direct --ablation medium
    run --task gsm8k --n 200 --seed 0 --batch-size 8 --model "$M14" --lens-path "$L14" --mode direct --ablation none
    ;;
  14b_b)
    python scripts/fetch_lens.py --model qwen3-14b
    run --task gsm8k --n 200 --seed 0 --batch-size 8 --model "$M14" --lens-path "$L14" --mode cot --ablation none --max-new 2048
    export HW0_MODEL="$M14" HW0_LENS="/workspace/hw0/$L14"
    for probe in run_masked_maintenance run_capacity_maintenance run_cot_correspondence; do
      echo "=== START 14b $probe $(date -u +%H:%M:%S)"
      python -u "scripts/$probe.py"; echo "=== END 14b $probe rc=$?"
    done
    ;;
  *) echo "unknown job $JOB"; exit 1;;
esac
echo "JOB_${JOB}_COMPLETE $(date -u +%H:%M:%S)"
