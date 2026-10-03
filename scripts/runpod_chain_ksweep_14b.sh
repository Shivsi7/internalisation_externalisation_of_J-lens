#!/usr/bin/env bash
# k-sweep on GSM8K, then the Qwen3-14B replication suite.
set -u
cd /workspace/hw0
for K in 5 20; do
  for args in "--mode cot --max-new 2048" "--mode direct"; do
    echo "=== START gsm8k k=$K $args $(date -u +%H:%M:%S)"
    python -u scripts/run_eval.py --task gsm8k --n 200 --seed 0 \
      --batch-size 16 --ablation medium --k "$K" $args
    echo "=== END gsm8k k=$K $args rc=$?"
  done
done
echo "KSWEEP_COMPLETE $(date -u +%H:%M:%S)"
python scripts/fetch_lens.py --model qwen3-14b
M="Qwen/Qwen3-14B"
L="lenses/qwen3-14b/qwen3-14b/jlens/Salesforce-wikitext/Qwen3-14B_jacobian_lens.pt"
for args in "--mode cot --ablation none --max-new 2048" \
            "--mode cot --ablation medium --max-new 2048" \
            "--mode direct --ablation none" "--mode direct --ablation medium"; do
  echo "=== START 14b gsm8k $args $(date -u +%H:%M:%S)"
  python -u scripts/run_eval.py --task gsm8k --n 200 --seed 0 \
    --batch-size 8 --model "$M" --lens-path "$L" $args
  echo "=== END 14b gsm8k $args rc=$?"
done
export HW0_MODEL="$M" HW0_LENS="/workspace/hw0/$L"
for probe in run_masked_maintenance run_capacity_maintenance run_cot_correspondence; do
  echo "=== START 14b $probe $(date -u +%H:%M:%S)"
  python -u "scripts/$probe.py"
  echo "=== END 14b $probe rc=$?"
done
echo "POD_C_COMPLETE $(date -u +%H:%M:%S)"
