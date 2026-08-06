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

**Result (v1 — INVALIDATED for CoT, kept for the record).** Four runs
completed before the grid was killed:
cot/clean 75.5% [69.1, 80.9], cot/medium **87.0%** [81.6, 91.0],
direct/clean 20.5% [15.5, 26.6], direct/medium 13.5% [9.4, 18.9].
The apparent *improvement* of CoT under medium ablation is confounded:
94/200 clean-CoT chains hit the max_new=1024 cap (length-stop) vs 75/200
under ablation, so CoT accuracy partly measured truncation. Mean chain
length 854 tokens — the cap was far too tight for Qwen3-4B's thinking mode.
Also: 90/200 medium-ablated *direct* runs hit even the 16-token cap
(ablation suppresses EOS in the direct arm — itself a real observation).
Direct-arm accuracies are trustworthy; CoT ones are not.

**Lesson pre-registered into v2:** generation caps are part of the
measurement. CoT cap raised to 2048, direct to 32; v1 outputs preserved
under `results/aion/20260805_140557..143307_*`.

## 2026-08-05 — Numeric-readout gate probe (`scripts/probe_numeric.py`)

**Tokenization discovery that reframes the port.** Qwen3 tokenizes numbers
digit-by-digit — there are no multi-digit tokens. The morning calc-probe
"failure" was partly structural (a "42" readout token cannot exist), and the
paper's own finding that the J-lens degrades sharply on multi-token
intermediates applies to *every number ≥ 10* on this model family.

**Design.** Battery A: 40 three-step chains with single-digit operands and
intermediates, all 7 values distinct; occupancy of each digit token z-scored
across the ten digit classes per (layer, position). Battery B: paper-style
2-digit chains scored by leading digit. Null = digits with no role in the
prompt, identical statistics.

**Results** (`results/aion/20260805_144827_numeric_probe`, plus rerun with
final-position statistic `20260805_1452*`):
- All-position max-z: target-vs-none AUC 0.61 [0.53, 0.69]; operand control
  at chance — the max-over-cells statistic inflates the null (median "none"
  z = 1.86 against a ~2.8 ceiling). Statistic, not lens, at fault.
- **Final ("=") position: target-vs-none AUC 0.67 [0.59, 0.74]** — real
  signal, just under the pre-registered 0.70 gate. Structured ordering:
  median z targets 1.58 > none 1.14 > operands 0.80. Operands being
  *suppressed* below no-role digits at the answer position is itself
  workspace-like (inputs consumed, results held) — though it means "operand
  occupancy" is the wrong positive control at this position; it should be
  measured at the operand's own position instead (fix in next iteration).
- **Ordering signature reproduces**: first layer at which an intermediate
  surfaces correlates with its computation index — Spearman ρ = 0.70,
  p = 2e-8 (all positions); ρ = 0.42, p = .017 (final position only).
  This is the paper's §3.3 21→42→49 signature, in digit-token form, on a
  model 100x smaller.
- Battery B (leading digit of 2-digit values): AUC 0.46 — nothing, as the
  paper's multi-token result predicts.

**Verdict.** Gate formally FAILS (0.67 < 0.70) but with CI excluding
chance and the ordering signature strongly present. Remediation ladder
continues: better positive control, more prompts, penultimate-layer refit
(iii), format variants (iv). Design implication live: rung 1.5's synthetic
chains should use single-digit intermediates.

## 2026-08-05 — Figure 24 grid v2 (in progress)

Same design as v1, max_new 2048 (cot) / 32 (direct), direct runs first.
`scripts/run_fig24_grid_v2.sh`, log aion:/var/tmp/hw0_fig24_grid_v2.log.

**Direct arm complete (n=200 each):**
| condition | acc | 95% CI |
|---|---|---|
| clean | 21.0% | [15.9, 27.2] |
| light | 10.0% | [6.6, 14.9] |
| medium | 13.0% | [9.0, 18.4] |
| heavy | 15.5% | [11.1, 21.2] |
| random medium, norm-matched | 18.5% | [13.7, 24.5] |

J-ablation at medium cuts direct accuracy to 0.62 of clean while the
norm-matched random control sits at 0.88 of clean — directionally the
paper's selectivity pattern. Oddity to watch: non-monotonicity across
strengths (light < medium < heavy, i.e. the *narrowest* band hurt most);
CIs overlap, so noise is possible, but if it survives larger n it needs an
explanation (filter interaction? band placement?). CoT arm still running.

**Result (CoT arm).** _pending_

## 2026-08-05 — Rung 1.5 pilot (maintenance signature), iterated live

`scripts/rung15_maintenance.py`, cedar Ada, three iterations in ~40 min
(each run 6–23 s wall — generate, teacher-forced re-forward, occupancy).
Design: same single-digit variable program presented two ways — Written
(model rewrites each line with its value; the page holds the carried value)
vs Unwritten (model answers directly; the value is never written). Carried
value defined at line 1, reused after {1,2,4} filler lines. Occupancy of
the carried value's digit token during the gap lines, z-scored across the
ten digit classes (medium band).

**Final run (n=34/gap, gaps 1/2/4; parse 88%):**

Behavioural — the page is load-bearing, and increasingly so with gap:
| gap | Written acc | Unwritten (direct) acc |
|---|---|---|
| 1 | 44% (see anomaly) | 29% |
| 2 | 82% | 35% |
| 4 | **100%** | **6%** |

Occupancy — NO detectable carry of the value in either mode during the
gap: gap-region max-z for the carried digit ≈ decoy max-z everywhere
(e.g. gap 2: 1.80 vs 1.79 written, 1.46 vs 1.39 unwritten); mean-z at or
below decoy level; layer profile flat-negative in Unwritten mode,
slightly positive early-band in Written mode (consistent with
re-ingestion of the written token, not holding).

**Registered gate fires:** the exchange-vs-duplication verdict requires
visible internal carry in the Unwritten mode; there is none, so the
lens-level verdict is *uninformative* — but the overall pattern supports a
sharper reading: at 4B the J-space does not do idle-value maintenance in
either mode (consistent with the paper's token-exclusion result, Fig 73);
context is the only effective store, and the workspace acts as a compute
buffer. Behaviourally there is almost nothing internal to "duplicate":
without the page, multi-step variable tracking collapses (6% at gap 4).

**Anomaly, inspected:** gap-1 Written accuracy (44%) is an in-context
format effect, not arithmetic failure — with only two solved lines above,
the model substitutes *operands* instead of computed *values* into the
final line (e.g. a=2+6=8, b=9-4=5, then "c = a - b = -3" = 2-5). The
error mode disappears as more solved lines accumulate (82% at 3 lines,
100% at 5). Reported as a caveat on the Written gap-1 cell.

**Caveat:** the numeric readout is marginal (gate AUC 0.67), so "no carry
signal" is bounded by measurement sensitivity; the behavioural
dissociation carries the claim, occupancy nulls are supporting evidence.

## 2026-08-05 — GSM8K answer pre-surfacing probe (`probe_gsm8k_presurface.py`)

we's correction applied: the paper's §3.3 signature is per-operation,
across LAYERS, at the computation site — not persistence across tokens.
This probe reads digit occupancy at the positions just before the model
emits its direct-mode GSM8K answer (existing 200 clean rollouts,
teacher-forced re-forward; n=199 usable).

| bucket (layers, position) | emitted | gold | decoys |
|---|---|---|---|
| late (29-32), imminent | **+1.06** | +1.06 | −0.03 |
| late, imminent, WRONG answers | **+0.70** | +0.34 | +0.10 |
| mid (15-28), imminent | −0.24 | −0.24 | +0.27 |
| mid, early (2-6 tok before) | −0.18 | −0.18 | +0.35 |

Three findings:
1. **Pipeline validated**: the to-be-emitted digit reads strongly at the
   motor end (z=+1.06), as it must.
2. **Paper's report-tracking signature reproduced on real GSM8K**: on
   incorrect rollouts the readout shows the WRONG answer the model is
   about to give (0.70) far more than the correct one (0.34) — the
   workspace determines the report, not the truth (paper Fig 6 logic,
   observational form).
3. **No mid-band pre-surfacing of multi-digit GSM8K answers** — in fact
   in-play digits read as *suppressed* mid-band relative to absent digits
   (z is relative across the ten digit classes). Contrast with battery A:
   single-digit synthetic intermediates DO surface across layers in
   computation order. Reading: on a digit-by-digit tokenizer, mid-band
   numeric state is not digit-token-shaped; numbers crystallize into
   vocabulary form only at the motor stage. The J-lens (a vocabulary
   readout) can therefore see concepts mid-band but multi-digit numbers
   only at emission — which connects directly to the headline: writing a
   number into tokens (CoT) is what makes numeric state durable and
   re-readable on this model family.

## 2026-08-05 — Single-digit GSM8K pre-surfacing (we's design)

Question: was the missing mid-band pre-surfacing on GSM8K about numbers in
general, or about MULTI-DIGIT numbers (tokenization story)? Test: the 252
GSM8K test problems whose gold answer is a single digit (natural problems,
lens-readable answers). Direct clean on the subset: 24.6% (like the full
set). Presurface probe on all 252 rollouts:

| bucket | emitted | gold | decoy |
|---|---|---|---|
| late imminent, correct | +1.29 | +1.30 | −0.02 |
| late imminent, incorrect | +0.73 | +0.41 | +0.07 |
| **mid imminent, correct** | **+0.36** | +0.35 | +0.21 |
| mid imminent, incorrect | −0.15 | +0.27 | +0.27 |
| mid early (2-6 tok before) | ~0.12 | ~0.13 | ~0.31 |

Findings: (1) **single-digit answers DO pre-surface in the mid-band at the
imminent position** (+0.36 vs +0.21 decoys) where multi-digit answers did
not (−0.24 vs +0.27) — the tokenization account is confirmed on natural
GSM8K, not just synthetic chains. Effect is modest and confined to the
imminent position; nothing at earlier positions. (2) On incorrect answers
the mid-band does NOT carry the wrong answer the model is about to emit
(−0.15, below decoys) even though the late layers commit to it (+0.73) —
weak evidence that these errors arise late (motor-stage), not from wrong
mid-band content; small effects, flagged as suggestive only.

## 2026-08-05 (late evening) — Addendum-2 experiments, executed same night

**Depth ladder (A2.4), `run_depth_ladder.py`, n=30/length.** Sequential
single-digit chains, direct mode. Accuracy: L=2 **100%** → L=3 **30%** →
L=8 10%. Readout surfacing of the emitted answer (z≥2, any band layer,
answer-adjacent position): 63% at L=2 → 30% at L=3 → ≈0-20% beyond.
**D = 2** — the registered prediction D∈[3,7] is FALSIFIED (budget is even
shallower at 4B than the conservative guess); the registered *coupling*
prediction (behavioural cliff coincides with readout-D within ±2) is
CONFIRMED — both collapse at exactly L=2→3.

**Rung-1 problem-shuffled null (re-analysis of battery-A detail).** A
prompt's true intermediates vs the same digits scored in prompts where they
have no role: AUC **0.659 [0.588, 0.726]** — correspondence survives its
carrying null (modest, consistent with the 0.61-0.67 target-vs-none AUCs).

**Dual-task word probe (A2.3) smoke (n=8, cedar).** All four conditions
run; recall ~ceiling everywhere as predicted; held-word occupancy ALIVE
(direct arm mean z +0.38 vs decoys ~0, max-z ~3.2) — words are in the
readable regime as designed. Full n=120 (4 conditions + occupancy) queued
on aion 0 behind the AIME chain (log: /var/tmp/hw0_dualtask_a0.log).

## 2026-08-05 (late evening) — B8 VERDICT: the exclusion filter is innocent

Controls chain (cedar): cot medium **no-filter** 95.5% [91.7, 97.6] vs
92.0% with filter; direct medium no-filter 14.0% [9.9, 19.5] vs 13.0%.
ρ_nofilter ≈ 1.57 vs ρ_filter ≈ 1.63 — the CoT/direct gap is stable under
the toggle, so the paper's filter does NOT manufacture the headline
contrast (pre-registered as the #1 methodological threat, A1/§6.2(4)).
Rank-shift (B3) and logit-lens (B5) pairs still running.

## 2026-08-05 (late evening) — Scale sweep first numbers (8B)

Qwen3-8B direct: clean 24.0% [18.6, 30.4], medium-ablated 25.0%
[19.5, 31.4] — **no direct-arm ablation damage at 8B with the naive band
port**, in sharp contrast to 4B (21→13%). Either the 38-92% depth-fraction
port misplaces the 8B band, or 8B carries more redundancy. Flagged for the
report as a scale/band-calibration observation, not yet interpreted.
Grid: cot light 94.0% — CoT arm flat under light too. 1.7B numbers pending
collection.

## 2026-08-05 — AIME clean results

cot clean **23.3%** [14.4, 35.4] (pre-registered guess: 20-25%);
direct clean **0/60** (floor rule fires, as registered). Medium runs pending.

## 2026-08-05 — MATH-500 and AIME chains launched

MATH-500 (`scripts/run_math500_chain.sh`, aion GPU 2, n=243 stratified by
level): {cot,direct} × {clean, medium, random-medium norm-matched}, cot cap
3072. AIME (`scripts/run_aime_chain_now.sh`, aion GPU 0 — moved from the
gated GPU-3 queue after verifying the thesis chain there had completed):
same 6 conditions, all 60 problems, cot cap 4096 (truncation expected and
reported; all conditions share the cap). Logs:
aion:/var/tmp/hw0_math500_chain.log, /var/tmp/hw0_aime_chain_a0.log.
Early real datapoint from the adapter smoke: **AIME direct clean = 0/60** —
the pre-registered floor rule (hypotheses.md §4) fires on AIME exactly as
predicted; \rho is uninformative there and the informative AIME quantity is
the CoT arm's own retention.

Incidental infra note (thesis-relevant, diagnosed on request): the aion-0
ladder chain did NOT fail its science — all 6 stages completed with DONE
markers and final.pkl. The chain process died after the last stage because
`container/run_ladder3.sh` was edited in place at 10:29 while the 10:24
bash was executing it (stale-offset re-read → phantom syntax error at
line 41 of a 40-line file), so the tail `LADDER3_GPU0_COMPLETE` echo never
ran. Nothing currently gates on that marker; anything that does should gate
on the per-stage DONE lines instead. day_a0.log left untouched.

## 2026-08-05 (evening) — final results batch

**Prefix-rescue (epiphenomenal test), n=150 clean-CoT-solved problems, medium
ablation:** own correct CoT prefix 98.0%; token-matched filler 23.3%
(≈ no-prefix 19.3%); wrong-problem CoT 0%. Clean: 99.3/22.0/0/13.3%.
Epiphenomenal reading REJECTED — protection is content, not forward passes.
Caveat: prefilled chains derive their answer, so this measures
rescue-by-reading (sufficiency), not generation.

**Dual-task full (n=120):** recall at ceiling in all 4 conditions (≥117/120),
as the diagnosticity condition predicted; held-word occupancy ≈ decoy level
both modes (mean z 0.07/0.03) → uninformative under the pre-registered
positive-asymmetry rule. Reported as such.

**Fluency (paper's Fig-22 metric):** top-1 agreement 77.5/70.7/62.5%
light/medium/heavy (paper Sonnet: 87/75/65); norm-matched random 80.0% with
near-clean task accuracy — token-level disruption without task damage.

**Controls chain complete.** Rank-shift: direct 18.5% (= random control),
cot 93.0%. Logit-lens: direct 17.5% [12.9,23.4] — near control level; the
Jacobian transport matters for the intervention's potency.

**MATH-500 medium:** cot 67.9% [61.8,73.5] (ret 1.04), direct 19.8%
[15.2,25.2] (ret 0.83) → ρ(MATH)=1.26. Ordering ρ 1.63→1.26 holds; CoT
retention 1.01/1.04→0.79 (AIME).

**GSM8K cot heavy:** 92.0% [87.4,95.0] — CoT flat under the widest band.

**Seed-1 replication (headline cells):** cot 93.0→94.0%, direct 25.0→18.0%;
ρ(s1)=1.40 [0.95,2.09]; pooled both seeds ρ=1.50 [1.13,2.03]. Direction and
magnitudes replicate across sampling seeds.

**8B cot medium + requeued nice-to-haves (GSM8K cot-random, AIME randoms):**
still running at log time; report complete without them.

## 2026-08-05 (late) — requeued AIME random control: an honest complication

aime cot medium random norm-matched: 15.0% [8.1, 26.1] (n=60) vs real
ablation 18.3% [10.6, 29.9] — indistinguishable. The AIME reasoning-arm
retention drop (0.79) is consistent with H2's direction but NOT attributable
specifically to the J-space at this n; generic perturbation damages AIME CoT
similarly. H2's direction-specific ordering evidence therefore rests on the
ρ axis (1.63 → 1.26, both datasets with innocuous controls at n=200).
Report updated accordingly. aime direct random: 0/60 (floor, as expected).

## 2026-08-05 (night) — campaign complete; final tail results

All queued and requeued runs finished. gsm8k cot medium random: 88.5%
[83.3,92.2] (ret 0.97, flat as predicted). 8B cot medium: 91.5% [86.8,94.6]
(ret 1.02 — CoT flat at all three scales). math500 cot random: 64.2%
(ret 0.99); **math500 direct random: 18.5% [14.1,23.9] vs real ablation
19.8% — indistinguishable**, so MATH-500's mode contrast, like AIME's, is
not attributable specifically to the J-space at this power; random-ρ(MATH)
= 1.28 ≈ real 1.26. Direction-specific protection is established on GSM8K
only (random-ρ 1.10 vs real 1.63). Report updated accordingly. Figures
regenerated in the paper-website style; all GPUs idle.

## 2026-08-06 (final review) — complementarity verdict corrected

During we's final read, he identified that the complementary hypothesis,
properly read, is a claim about workspace content DURING CoT (when
externalisation has occurred), making it the exact mirror image of
duplication over the same measurement. The direct-mode correspondence test
therefore only rules out the stronger by-content-kind form (the workspace
demonstrably holds CoT-type values when nothing is externalised); the
during-CoT form falls to the same maintenance-design confound as
duplication. Report verdicts updated in all four places: mechanism
scoreboard is now epiphenomenal REJECTED, complementary-by-content ruled
out, complementary-during-CoT / duplication / exchange OPEN. Logit-lens
discussion also removed from the report on review (addressed
by the original paper); runs remain in the repo.

## 2026-08-06 (pre-dawn) — masked-maintenance deconfound (EXP: masked_maintenance)

**Why.** On review of the maintenance verdict we decided that if the test is
confounded, the fix should be demonstrated, not deferred to future work.
Budget: ~30 minutes, cedar GPUs idle.

**Hypothesis.** The maintenance confound (defining expression re-readable in
the frozen context) can be removed causally with attention masking. If the
model holds values internally, accuracy survives when re-reading is blocked;
if it re-reads, accuracy falls to the severed floor. A2.5's registered
"context is the only store" reading predicts the floor.

**Design.** scripts/run_masked_maintenance.py, single forward pass per item,
no sampling (argmax over digit tokens at the final position). Segments:
demo (w=3, own filler), definition `{name} = {value}.`, digit-free filler
gap (~30 tokens), question. 4D additive attention mask, sdpa. Conditions:
baseline / masked_use (question cannot see definition) / severed (gap and
question cannot see definition; floor control) / page (masked_use +
restatement inside the gap). n=96 (6 names x 8 values x 2 fillers; value 3
reserved for the demo). Occupancy: medium band (L19-28), z of the correct
digit against the other eight at gap positions, from the baseline pass.

**Runs.** cedar GPU 1 (Ada, PCI order), venv python. First launch crashed
(KeyError: ablator built without band= so J only covered the default band);
fixed by passing AblationConfig(band=band, exclusion_filter=False) and
matching run_dualtask's hidden_states[layer+1] convention. Second launch
clean, ~3 min end to end. results/cedar/20260806T010855Z_masked_maintenance.

**Result.** baseline 100% (96/96), masked_use 13.5% [8.1, 21.8], severed
12.5% [7.3, 20.6], page 100%. masked_use is statistically indistinguishable
from the severed floor (chance over the digit candidates is ~11-12.5%). Gap
occupancy z ~= 0.03; the 13 masked-correct items had LOWER mean occupancy
(-0.62) than the misses (+0.13), i.e. the survivors are luck, not holding.

**Verdict.** Decisive within its scope. The model does not maintain even one
literal digit across a ~30-token gap in the workspace; storage is the
context, and a later visible copy (the page analog) restores ceiling. This
confirms A2.5's registered reading for literal values and converts the
report's "storage lives on the page" inference into a causal result. Scope
limits: literal single-digit values, one gap length, one band, argmax
readout, n=96. Report updated (design instrument, results verdict, analysis,
future work); WikiText fluency sentence cut from the report to hold three
pages (the run and numbers remain in results/ and this log).

## 2026-08-06 (pre-dawn) — during-CoT persistence probe (EXP: cot_persistence)

**Why.** The gap, identified precisely on review: the correspondence test
measured workspace content in DIRECT mode only, but duplication and exchange
are claims about the CoT arm, what the workspace holds while the values sit
on the page. That was never measured.

**Design.** scripts/run_cot_persistence.py. Teacher-forced worked
transcripts (the page), single forward pass, no generation. Programs of
three worked lines plus a use line (d = a op c). Occupancy z of v1 (value of
line 1, against all nine digits, medium band, hidden_states[layer+1]) in
three windows: write (line 1), post (lines 2-3, the discriminator), use
(line 4). Direct arm: same program unworked. Constraint: v1 distinct from
v2, v3 and every literal digit in lines 2-3, so post-window signal cannot be
v1's digit appearing in the text. n=80 programs, seed 0. Answer accuracy
read as argmax digit at the final position.

**Result (results/cedar/20260806T012501Z_cot_persistence).**
cot: z_write 0.47 (se 0.08), z_post 0.08 (se 0.09), z_use 0.42 (se 0.09),
accuracy 100%. direct: 0.24 / 0.18 / 0.08, accuracy 13.75%.

**Verdict.** Duplication rejected with a built-in positive control: the
value is strongly present exactly when written and when re-read, and absent
in between (write-vs-post drop ~0.38, ~4 se). Exchange loses its premise:
the direct arm shows no persistent holding even without externalisation, so
writing removes nothing. Together with masked maintenance, prefix rescue and
correspondence, all four mechanism hypotheses now have verdicts, converging
on A2.5's registered reading: workspace = per-step computation, page = all
storage. Scope: single-digit values, teacher-forced transcripts, one band,
z-based occupancy (AUC for digits is modest, 0.67). Report updated; to hold
three pages the wrong-answer late-layer tracking clause, the end-of-sequence
suppression observation, and the AIME-split contamination sentence moved out
of the report (all remain in results/ and this log).

## 2026-08-06 (pre-dawn, later) — persistence probe removed from the report

We judged the teacher-forced persistence design unconvincing and cut it
from the report (design, results, analysis). The duplication/exchange verdict
now rests on the masked maintenance test alone: nothing is held in either
mode, so there is nothing for the page to duplicate and nothing for writing
to remove. The run, its results directory, Addendum 4, and this log entry
remain the record of the experiment.

## 2026-08-06 (pre-dawn, later still) — CoT-arm correspondence (EXP: cot_correspondence)

**Why.** The criticism, correct: with the persistence probe cut, the
report's only content measurement ran in direct mode, but hypothesis (3)
concerns the CoT scenario. We also rejected teacher-forcing, so this arm
uses the model's OWN chain of thought.

**Design.** scripts/run_cot_correspondence.py. Same program generator and
v1-exclusion constraints as the persistence probe. The model solves each
program via chat template, greedy, max_new=256. Windows located by first
occurrence of each value's digit token in the generated ids (v1 cannot
appear earlier by construction). Items with out-of-order appearance
dropped. Occupancy z of v1 vs the nine digits, medium band.

**Runs.** cedar GPU 1. First launch crashed (transformers 5 returns a dict
from apply_chat_template, fixed). Second run clean.
results/cedar/20260806T024836Z_cot_correspondence.

**Result.** n=69 scored, 11 dropped. z_write 1.20 (se 0.05), z_post 0.16
(se 0.06), z_use 0.42 (se 0.06). The 'acc' field (29%) is NOT a valid CoT
accuracy, the 256-token cap truncates think blocks so the last-digit
readout is meaningless. Occupancy windows are unaffected (they lie inside
the generated text).

**Verdict.** In the model's own CoT the workspace strongly represents the
value being written, mostly releases it over the following lines, and
re-represents it at use. By-content complementarity fails in the CoT
setting itself, and the transient-touch storage picture replicates on the
model's own page (the cut persistence probe's conclusion, now in the
scenario we endorses). Report updated (design, results, analysis).
