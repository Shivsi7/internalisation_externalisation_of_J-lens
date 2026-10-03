#!/usr/bin/env bash
# Llama-3.1-8B-Instruct replication: headline GSM8K grid + mechanism probes.
# Ungated mirror of the Meta weights; lens from Neuronpedia (fitted on the
# same weights). Band ports by depth fraction automatically (32 layers).
set -u
cd /workspace/hw0
python scripts/fetch_lens.py --model llama3.1-8b-it
M="unsloth/Meta-Llama-3.1-8B-Instruct"
L="lenses/llama3.1-8b-it/llama3.1-8b-it/jlens/Salesforce-wikitext/Llama-3.1-8B-Instruct_jacobian_lens.pt"
for args in "--mode cot --ablation none --max-new 2048" \
            "--mode cot --ablation medium --max-new 2048" \
            "--mode direct --ablation none" "--mode direct --ablation medium"; do
  echo "=== START llama gsm8k $args $(date -u +%H:%M:%S)"
  python -u scripts/run_eval.py --task gsm8k --n 200 --seed 0 \
    --batch-size 8 --model "$M" --lens-path "$L" $args
  echo "=== END llama gsm8k $args rc=$?"
done
export HW0_MODEL="$M" HW0_LENS="/workspace/hw0/$L"
for probe in run_masked_maintenance run_capacity_maintenance run_cot_correspondence; do
  echo "=== START llama $probe $(date -u +%H:%M:%S)"
  python -u "scripts/$probe.py"
  echo "=== END llama $probe rc=$?"
done
echo "POD_D_COMPLETE $(date -u +%H:%M:%S)"
