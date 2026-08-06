# results/

One directory per run: `results/<hostname>/<UTC timestamp>_<condition tag>/`.
Nothing is ever overwritten; the timestamp is the run's identity, and when a
condition was rerun (e.g. the truncation-invalidated v1 GSM8K grid), both
runs remain on disk and the figure scripts select the latest valid one.

Each evaluation run contains:

| file | contents |
|---|---|
| `config.json` | the complete parameter manifest: argv, task, mode, sampling settings, ablation config, model, lens path, dataset revisions, git commit, GPU, hostname |
| `generations.jsonl` | one line per problem: question, gold answer, full model output, extracted prediction, correctness, token count, stop reason |
| `summary.json` | accuracy with Wilson 95% CI, per-group breakdown (MATH level / AIME year), token statistics, length-stop counts, per-layer ablation statistics |

Probe runs (`*_probes`, `*_numeric_probe`, `*_presurface`, `*_rung15`,
`*_depth_ladder`, `*_dualtask`, `*_fluency`, `*_prefix_rescue`,
`*_masked_maintenance`, `*_cot_persistence`, `*_cot_correspondence`) write
`summary.json` plus, where applicable, `detail.jsonl` with per-item
measurements.

`hostname` is `aion` or `cedar` (the two lab machines used); the mapping of
runs to figures is in `scripts/make_figures.py`, which reads everything under
this directory afresh on each invocation.
