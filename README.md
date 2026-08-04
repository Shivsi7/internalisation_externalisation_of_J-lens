# J-space ablation vs. external chain of thought, across problem difficulty

Homework Zero for Harvard CS 2881 (AI Safety). Investigates how the internal/external
reasoning trade-off reported in [*Verbalizable Representations Form a Global Workspace in
Language Models*](https://transformer-circuits.pub/2026/workspace/index.html) (Anthropic,
July 2026) behaves as problems get harder.

Model: `Qwen/Qwen3-4B`. Datasets: GSM8K, MATH-500, AIME 2024+2025.

> **Status: design stage.** Hypotheses are pre-registered; the experiments have not been run.

## Where everything is

| Path | What |
|---|---|
| `report.pdf` | The two-page report *(not yet written)* |
| `hypotheses.md` | **Pre-registration.** Written and committed before any experiment was run. |
| `design.md` | Experiment design: conditions, controls, metrics, staging, compute plan |
| `src/ablation/` | The J-space ablation and ablated-generation hooks — our code |
| `src/eval/` | Datasets, generation, answer extraction, scoring |
| `configs/` | One YAML per experimental condition; every run reproducible from a config |
| `prompts/` | All prompt templates, verbatim |
| `results/` | Raw generations (gzipped jsonl) and scored CSVs |
| `notebooks/` | Figure generation |
| `scripts/` | Entry points — `fetch_lens.py`, and the end-to-end reproduction scripts |

Read `hypotheses.md` before `design.md`; the design exists to make the hypotheses falsifiable
in both directions, and several of its more unusual choices only make sense in that light.

## The Jacobian lens is a dependency, not our work

Anthropic released a reference implementation at
[`anthropics/jacobian-lens`](https://github.com/anthropics/jacobian-lens) (Apache-2.0). It is
pinned in `pyproject.toml` to commit `581d398613e5602a5af361e1c34d3a92ea82ba8e` — upstream is
explicitly unmaintained, so the pin matters. We do not reimplement it.

Neuronpedia hosts **pre-fitted lenses** for the whole Qwen3 family, including the exact
required checkpoint, at [`neuronpedia/jacobian-lens`](https://huggingface.co/neuronpedia/jacobian-lens):

```
qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt   # 458 MB
```

So fitting the lens is a download rather than a GPU job. `scripts/fetch_lens.py` pulls it and
checks that it loads, matches the intended checkpoint, and covers the layers we intend to ablate.

**What the reference implementation does *not* provide is the ablation.** It is a reading tool:
fit `J`, read out tokens, visualise. `jlens.hooks.ActivationRecorder` captures residuals and
modifies nothing, and `JacobianLens.apply` is `@torch.no_grad()` and does not generate. The
intervention this project is built on, and batched ablated generation on top of it, live in
`src/ablation/` and are ours. See `design.md` §6.1–6.3.

## Setup

Requires Python ≥3.10 and a CUDA GPU (Qwen3-4B is ~8 GB in bf16; there is no local GPU on the
development machine, so everything runs on rented compute — see `design.md` §6.4).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python scripts/fetch_lens.py            # Qwen3-4B lens -> lenses/qwen3-4b/
```

## Reproducing the results

*To be written as the experiments are run.* Each figure and table in the report will name the
script that regenerates it from `results/`.

## External links

- Paper: https://transformer-circuits.pub/2026/workspace/index.html
- Reference implementation: https://github.com/anthropics/jacobian-lens
- Pre-fitted lenses: https://huggingface.co/neuronpedia/jacobian-lens
- AIME problem set: *exact source and revision to be recorded here* (`design.md` §3)

## Attribution

`jlens` is © Anthropic PBC, Apache-2.0. The pre-fitted lenses are published by Neuronpedia.
Model weights and datasets are downloaded at run time under their own licenses.
