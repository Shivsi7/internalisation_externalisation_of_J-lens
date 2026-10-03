# Is Chain of Thought an Interchangeable Scratchpad?

Harvard CS 2881 Homework Zero. Investigates how the internal/external reasoning
trade-off reported in [*Verbalizable Representations Form a Global Workspace in
Language Models*](https://transformer-circuits.pub/2026/workspace/index.html)
(Anthropic, July 2026) behaves as problems get harder — and whether
"interchangeable scratchpads" is the right reading of it.

Model: [`Qwen/Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B) (mandated).
Datasets: GSM8K, MATH-500, AIME 2024+2025 (exact sources and revisions below).
The report is **`report.pdf`** at the repository root; its LaTeX source is
`reports/report.tex`.

## Headline results

- The paper's Figure-24 contrast replicates at 4B with controls the paper did
  not run: medium J-space ablation costs chain-of-thought GSM8K nothing
  (91.0% → 92.0%) while removing ~40% of direct-answering capability
  (21.0% → 13.0%); norm-matched random and rank-shifted controls both retain
  0.88, so the damage is specific to the *top* J-lens directions. Protection
  ratio ρ = 1.63, bootstrap 95% CI [1.05, 2.73].
- On our probes, each of the four mechanism hypotheses (exchange, duplication,
  complementary, epiphenomenal) either met evidence against it or lost its
  premise. An attention-masked maintenance test shows the model does not carry
  even a single literal digit across a ~30-token gap (13.5% vs a 12.5% severed
  floor; a written copy restores 100%), so storage lives in the context; a
  prefix-rescue test shows the chain of thought's benefit is its content, not
  its extra forward passes (own reasoning 98% vs token-matched filler 23%).
- Chain-of-thought protection did **not** measurably weaken with difficulty
  (CoT retention 1.01 on GSM8K, 1.04 on MATH-500, with only the uncontrolled
  AIME cell dipping to 0.79); the falling ρ decomposes into direct-arm
  changes. A rival decline hypothesis recorded in advance was not supported.
- The workspace's serial depth budget is directly measurable: **D = 2 chained
  operations**, falsifying the pre-registered D ∈ [3,7] while confirming the
  registered coupling between readout collapse and the behavioural cliff.
- Qwen3 tokenizes numbers digit-by-digit, which places every value ≥ 10 in the
  J-lens's known-bad multi-token regime — a tokenizer dependence the paper's
  much larger models could not expose, and a constraint that shaped every
  mechanism probe here (single-token content only).

## Repository structure

| path | contents |
|---|---|
| `report.pdf` | the original submitted report (root, per the assignment; unchanged since submission) |
| `extended_report.pdf` | the report extended with the post-submission campaign (seeds, k-sweep, Qwen3-14B, Llama-3.1-8B, capacity/page-masking/battery/task-balance instruments); source `reports/extended_report.tex` |
| `reports/report.tex`, `reports/figures/`, `reports/tables/` | LaTeX source, all figures (PDF+PNG), generated tables |
| `hypotheses.md` | **pre-registration**, committed before any run; post-hoc changes confined to dated append-only addenda (four addenda record every late instrument, correction, and falsified prediction) |
| `design.md` | experiment design rationale: conditions, controls, metrics, staging |
| `experiments.md` | the executable plan (the four-rung decision procedure) |
| `src/ablation/` | **our intervention**: J-space ablation, occupancy measurement, batched ablated generation with the paired clean pass (the reference repo contains readout only) |
| `src/eval/` | dataset adapters, prompt construction, answer extraction, scoring |
| `prompts/` | every prompt template, verbatim |
| `scripts/` | entry points for every experiment and figure (see Reproduction) |
| `results/` | per-run manifests, raw generations, scored summaries (`results/README.md` documents the layout) |
| `tests/` | CPU test suite for the ablation/generation code |
| `notes/experiment_log.md` | narrative log of every run, including failures |
| `notes/design_review.md` | the design's self-review (claims, controls, weak joints) |
| `spec.md` | the assignment text |

## Setup

Requires Python ≥ 3.10 and one CUDA GPU with ~20 GB memory (bf16;
ablated chain-of-thought generation with the paired clean pass peaks near
24 GB at batch 16 — reduce `--batch-size` on smaller cards).

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # match your driver
pip install "transformers>=5.5" huggingface_hub numpy datasets math-verify \
            sympy pandas scipy matplotlib pyyaml accelerate pytest
pip install "jlens @ git+https://github.com/anthropics/jacobian-lens@581d398613e5602a5af361e1c34d3a92ea82ba8e"
python scripts/fetch_lens.py        # pre-fitted Qwen3-4B lens -> lenses/ (458 MB)
```

Artifacts too large for the repository, fetched at run time:

- **Pre-fitted Jacobian lenses** (458 MB for 4B; also 1.7B/8B for the scale
  sweep): hosted at
  [`neuronpedia/jacobian-lens`](https://huggingface.co/neuronpedia/jacobian-lens),
  fetched and verified by `scripts/fetch_lens.py` (`--model qwen3-1.7b` etc.).
  Provenance: fitted on Salesforce/wikitext-103, 479 prompts, bf16,
  final-layer target (a documented deviation from the paper's penultimate
  default — see `design.md` §6.1).
- **Model weights**: `Qwen/Qwen3-4B` (rev. `1cfa9a72`), `Qwen3-1.7B`,
  `Qwen3-8B`, downloaded via `transformers` on first use.
- **Datasets** (exact sources, pinned in `src/eval/tasks.py`):
  [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k) (test split);
  [`HuggingFaceH4/MATH-500`](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)
  rev. `6e4ed1a2`;
  AIME 2024 I+II = [`HuggingFaceH4/aime_2024`](https://huggingface.co/datasets/HuggingFaceH4/aime_2024)
  rev. `2fe88a2f1091` (30 problems);
  AIME 2025 I+II = [`MathArena/aime_2025`](https://huggingface.co/datasets/MathArena/aime_2025)
  rev. `c94da77eb22b` (30 problems).

The reference implementation
[`anthropics/jacobian-lens`](https://github.com/anthropics/jacobian-lens) is a
*reading* tool (fit + readout); it is pinned to commit `581d3986` because
upstream is explicitly unmaintained. **It contains no ablation code** — the
intervention in `src/ablation/` is this project's own work.

## Verification (run these first)

```bash
pytest tests/ -q                     # 13 CPU tests; includes token-for-token
                                     # equality of our generation loop vs HF generate()
python scripts/verify_occupancy_is_jlens.py   # our occupancy measure ==
                                     # reference JacobianLens.apply() readout
                                     # ordering at every band layer (GPU)
```

## Reproduction

Every run writes a self-documenting directory under `results/` (complete
parameter manifest, raw generations, scored summary — see
`results/README.md`). The principal experiments:

```bash
# Main grid: one (dataset, mode, intervention) cell per invocation
python scripts/run_eval.py --task gsm8k --mode cot    --ablation none   --n 200 --max-new 2048
python scripts/run_eval.py --task gsm8k --mode cot    --ablation medium --n 200 --max-new 2048
python scripts/run_eval.py --task gsm8k --mode direct --ablation medium --n 200
python scripts/run_eval.py --task gsm8k --mode direct --ablation medium --direction-source random --norm-match --n 200
python scripts/run_eval.py --task gsm8k --mode direct --ablation medium --direction-source jlens_rankshift --n 200
python scripts/run_eval.py --task gsm8k --mode cot    --ablation medium --no-filter --n 200 --max-new 2048
python scripts/run_eval.py --task math500 --mode cot  --ablation medium --n 250
python scripts/run_eval.py --task aime  --mode cot    --ablation medium
# Scale sweep: same recipe, other checkpoints + their own lenses
python scripts/run_eval.py --task gsm8k --mode direct --ablation medium \
    --model Qwen/Qwen3-8B --lens-path lenses/qwen3-8b/qwen3-8b/jlens/Salesforce-wikitext/Qwen3-8B_jacobian_lens.pt --batch-size 6

# Chain scripts used for the actual campaign (same commands, sequenced):
#   run_fig24_grid_v2.sh, run_math500_chain.sh, run_aime_chain_now.sh,
#   run_controls_chain.sh, run_scale_chain.sh, run_seed_replication.sh

# Mechanism probes
python scripts/probe_lens.py              # validation-gate readout probes
python scripts/probe_numeric.py           # digit-occupancy gate + ordering signature
python scripts/probe_gsm8k_presurface.py  # answer pre-surfacing on GSM8K
python scripts/run_depth_ladder.py        # serial depth budget D
python scripts/run_dualtask.py            # held-word dual-task probe
python scripts/run_prefix_rescue.py       # content-vs-filler prefix comparison
python scripts/run_fluency_control.py     # clean-vs-ablated top-1 agreement
python scripts/run_masked_maintenance.py  # attention-masked maintenance test (report §2, §4)
python scripts/run_cot_correspondence.py  # correspondence, CoT arm: model's own reasoning
python scripts/run_cot_persistence.py     # teacher-forced persistence probe (kept as a
                                          # record; superseded in the report by the
                                          # CoT-arm correspondence run)
```

## Regenerating the report's figures and tables

```bash
python scripts/make_figures.py
```

reads everything under `results/` and rewrites `reports/figures/*.{pdf,png}`
and `reports/tables/all_runs.tex` (the complete per-run results table). The
mapping to the report: `fig_gsm8k_retention` is Figure 1 (left half of the
merged float; Table 1 on its right takes its numbers from the `results/`
summaries), `fig_numeric_ordering` is Figure 2a, `fig_cot_correspondence` is Figure 2b
(the CoT arm of the correspondence test), `fig_prefix_rescue` is Figure 2c
(clean and ablated arms grouped side by side), `fig_depth_ladder` is
Figure 3a, and `fig_scale_retention` is Figure 3b. `fig_datasets_retention`
and `fig_maintenance` are generated but not used in the final report. The
masked-maintenance and CoT-correspondence numbers quoted in the report come
directly from their run directories' `summary.json`. Compile
`reports/report.tex` (standard LaTeX, no exotic packages) and place the PDF
at the root as `report.pdf`.

## External links

- Paper: https://transformer-circuits.pub/2026/workspace/index.html
- Reference implementation: https://github.com/anthropics/jacobian-lens
- Pre-fitted lenses: https://huggingface.co/neuronpedia/jacobian-lens
- Model: https://huggingface.co/Qwen/Qwen3-4B
- Datasets: linked with revisions under Setup above.

## Attribution

`jlens` is © Anthropic PBC, Apache-2.0. Pre-fitted lenses are published by
Neuronpedia. Model weights and datasets download at run time under their own
licenses. The full text of the paper is deliberately not redistributed here.
