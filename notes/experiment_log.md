# Experiment log

One narrative entry per run or run-group, appended chronologically. Smokes
included; raw outputs under `results/<host>/<timestamp>_<tag>/`.

---

## 2026-08-05 — Validation-gate probes (first GPU contact)

**Hypothesis/purpose.** design.md §4 gate, criteria 1 (lens interpretability):
do the paper's qualitative readout signatures exist on Qwen3-4B with the
pre-fitted Neuronpedia lens (target_layer=final, 479 prompts)?

**Config.** `scripts/probe_lens.py`, aion GPU 3, bf16, heavy band (L15–32),
readout at final prompt position, top-8 per layer.
Output: `results/aion/20260805_140144_probes/`.

**Result.**
- Multi-hop factual probes PASS qualitatively: "tower in the capital of
  France" surfaces `Paris`/`巴黎` at L27–31 before the model outputs " E";
  planets probe surfaces colour words (green/blue/yellow/red) at L30–32.
  Intermediate concepts visible mid-band = workspace signature present.
- Arithmetic chain probe FAILS: `calc: (4+17)*2+7 = ` shows no 21→42→49
  anywhere in L15–32 top-8; readouts are punctuation/underscore noise even
  though the model's actual next token is correct. Either numeric
  intermediates are not carried in lens-readable form at 4B, or the readout
  position/prompt format needs work. Bears directly on rung 1/1.5, which
  read *numeric* occupancy.
- Ablation smoke: heavy ablation leaves a 2-step word problem intact
  (correct answer, arguably more coherent than clean). Per-layer relative
  norm removed is 0.4–3.4% — the intervention is small in norm terms, as
  expected for k=10 directions out of d=2560.

**Verdict.** Gate criterion 1 half-passes: concept-like content readable,
numeric intermediates not (so far). Do not interpret occupancy-of-numbers
results until this is resolved or worked around. Gate criteria 2–4 (band
re-derivation, multi-hop ablation positive control at n, selectivity floor)
still to run.

## 2026-08-05 — Runner smokes (n=8)

`smoke_cot_medium`: CoT + medium ablation + filter, 8 problems, 30 s,
acc 4/8. `smoke_direct_clean`: direct arm (non-thinking, "The answer is"
prefill), acc 3/8, extraction handles markdown ("**120**."). Both code paths
green. Results kept under `results/aion/20260805_1404*`.

## 2026-08-05 — Figure 24 mini-replication grid LAUNCHED

**Hypothesis.** The paper's headline contrast (Fig 24: GSM8K CoT more robust
to J-ablation than GSM8K direct) replicates qualitatively on Qwen3-4B.
Pre-registered guesses in hypotheses.md §5 (e.g. clean CoT ~90%, clean
direct ~25%, medium-ablated direct ~5–10%, medium-ablated CoT ~60–75%).

**Config.** `scripts/run_fig24_grid.sh` on aion GPU 3: n=200, seed 0,
temp 0.6/top-p 0.95/top-k 20, CoT max_new 1024 / direct 16, batch 16.
10 runs: {cot,direct} × {none, light, medium, heavy, random-medium
(norm-matched, paper's control)}. Exclusion filter ON everywhere (B8
filter-off contrast deferred to the full grid). Log:
aion:/var/tmp/hw0_fig24_grid.log.

**Result.** _pending_
