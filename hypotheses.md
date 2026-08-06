# Pre-registration

**Written before any experiment was run.** Commit this file (and note its commit hash in the
report) before executing anything in `design.md`. Nothing below may be edited after the first
result is produced; corrections go in a dated addendum at the bottom.

Model: `Qwen/Qwen3-4B`. Datasets: GSM8K, MATH-500, AIME 2024+2025.

---

## 1. The mechanism I think is at work

The paper gives two resources a model can use to carry a multi-step computation, and they have
different shapes:

- **The J-space** is high-bandwidth but shallow. It holds "on the order of tens of concepts at a
  time" (§4.2) at *every* token position, but only for the duration of one forward pass, so the
  serial depth it can support is bounded by the number of layers in the workspace band.
- **Written chain of thought** is deep but narrow. §9.3 states this directly: externalising
  intermediate results into the context makes the model's "workspace-like processing unbounded in
  serial depth, but punctuated at regularly spaced intervals by a significant bandwidth
  constraint." Every hand-off between steps has to squeeze through a discrete token.

The paper's GSM8K result — CoT-solved problems are far more robust to J-space ablation than
directly-answered ones — is usually read as "CoT and the J-space are interchangeable scratchpads."
I think that reading is only half right. CoT can substitute for the *serial depth* the J-space
provides. It cannot substitute for the *per-step bandwidth*, because anything that goes onto the
page has to be verbalisable, and the J-space is explicitly the set of verbalisable representations
only up to a point — §4.2 finds the J-space component carries under 10% of activation variance,
so most of what a reasoning step conditions on was never in a serialisable format to begin with.

GSM8K is close to the best case for substitution: each step is a single arithmetic operation on
named quantities, the state is a handful of running totals, and every intermediate has a short
verbal name. The whole burden is depth. That is exactly the burden CoT removes.

Competition-level problems are not like that. The hard part of an AIME problem is usually not
executing a known chain but deciding which chain to execute — holding several candidate
approaches partly-evaluated, noticing that a substitution will collapse a term, recognising a
configuration. That work is high-bandwidth and largely not serialisable; writing "let me try
substituting" does not externalise the comparison that produced the decision.

Two results in the paper support this reading, though neither was framed this way:

- **§A.17 / §4.2 (`paper.md:3084`, `7088`) — load falls on computation, not storage.** Under a
  dual-task load, a *held* concept's presence in the workspace barely moves, but a *computed*
  answer's presence drops from 95% to 72%. The paper's own summary: "it is the **computation**
  that bears the cost of sharing the workspace." If workspace pressure specifically degrades
  computation rather than storage, then externalising storage onto the page — which is what CoT
  does well — should not relieve it.
- **§3.3 (`paper.md:1942`) — each serial step costs layer depth.** On `calc: (4+17)*2+7`, the
  intermediates 21 → 42 → 49 reach top-1 in computation order, separated by roughly eight
  reindexed layers each, inside a workspace band about 54 units wide. That implies a hard *serial
  depth budget* of roughly six or seven chained steps per forward pass.

## 2. Hypotheses

**H1 — Substitution (the paper's surface reading).** CoT and the J-space are interchangeable.
The protection CoT confers against J-space ablation is roughly constant across difficulty.

**H2 — Bandwidth-limited substitution (my primary hypothesis).** CoT substitutes only for the
serial-depth component of the reasoning load. Protection therefore *shrinks* as problems get
harder, because the harder problem's load is increasingly per-step bandwidth rather than depth.

**H3 — Null / artefact.** There is no interpretable trade-off at this model scale. Either the
ablation degrades Qwen3-4B broadly rather than selectively, or the apparent CoT advantage is a
floor artefact (see §4).

I consider H3 a live possibility rather than a formality. The paper reports that on **Haiku 4.5**,
"J-space ablation degrades coherence before yielding any qualitative change in responses" (§3.5.3).
Qwen3-4B is substantially smaller than Haiku 4.5. It is entirely possible that a 4B model does not
have a workspace clean enough to ablate selectively, and if so the honest result is a
characterisation of *that*, supported by the control battery, not a forced trade-off story.

**Ordering prediction under H2:**

    rho(GSM8K)  >  rho(MATH-500)  >  rho(AIME)  ~=  1

where `rho` is the protection ratio defined in `design.md` §5.

## 3. The prediction that actually discriminates

The ordering above is weak evidence on its own, because difficulty is confounded with everything
else that differs between these three datasets — answer format, topic, solution length, and
training-set contamination. Difficulty alone would produce the H2 ordering under several
mechanisms I do not believe.

So the discriminating prediction is about **serialisability, not difficulty**:

> Protection tracks how serialisable a problem's intermediate state is, and only incidentally
> tracks how hard the problem is.

This is tested by the synthetic depth ladder (`design.md` §3, condition C3): chains of elementary
arithmetic, arbitrarily long. These are *depth-hard and bandwidth-trivial*. H2 predicts they stay
well protected by CoT no matter how long the chain gets — even at chain lengths where clean
accuracy has fallen to AIME levels. H1 predicts nothing in particular here. A "difficulty is all
that matters" account predicts protection collapses on long chains just as it does on AIME.

If long arithmetic chains are protected and AIME is not, *at matched clean accuracy*, that
separates serialisability from difficulty and is the result I would most want to report.

**A sharper, numerical version of the same prediction.** The serial depth budget above is
measurable, not just rhetorical. Using the J-lens on the depth ladder, I can ask at what chain
length the final answer stops appearing in the readout before the answer token — that is the
number of steps Qwen3-4B can chain inside one forward pass. Call it `D`. H2 then predicts that
CoT protection *switches on* at chain length ≈ `D`: below `D` the model does not need the page and
ablation hurts CoT and direct alike; above `D` the page becomes load-bearing and CoT pulls away.

I record the prediction now: scaling the paper's ~8-layers-per-step against Qwen3-4B's narrower
band, I expect **`D` between 3 and 7**, and I expect the protection crossover on the depth ladder
to land within ±2 steps of whatever `D` measures. If the crossover is nowhere near `D`, the
depth-budget mechanism is wrong even if the ordering prediction survives.

## 4. What would falsify H2, stated in advance

| Observation | Reading |
|---|---|
| `rho` flat or rising across GSM8K → MATH-500 → AIME | H1; H2 wrong |
| Long arithmetic chains lose protection as fast as AIME at matched clean accuracy | The effect is difficulty, not serialisability; H2's mechanism wrong even if its ordering holds |
| Norm-matched and rank-shifted controls damage as much as the J-ablation | H3; nothing here is about the J-space |
| Automatic-task battery (MMLU, CoLA, SQuAD, sentiment) degrades in step with the math tasks | H3; broad degradation |
| Direct-answer clean accuracy < 10% on a dataset | That dataset's CoT-vs-direct contrast is **uninformative**, not evidence for H1 or H2 |

That last row is a pre-committed stopping rule and it matters more than it looks. Qwen3-4B
answering AIME with no chain of thought will score near zero *before* any ablation. A
near-zero score cannot fall, so "CoT is more robust than direct" is guaranteed on AIME for
arithmetic reasons that have nothing to do with the workspace. Any analysis that reports the
CoT/direct gap on AIME without handling this is measuring a floor. This is the single largest
threat to the whole experiment and it is why the design replaces the binary CoT/no-CoT contrast
with a continuous CoT-budget sweep.

## 5. Quantitative guesses

Recorded so I cannot retro-fit my sense of what counts as a large effect. These are guesses, and
I expect several to be wrong.

- Clean, thinking mode: GSM8K ~90%, MATH-500 ~70%, AIME ~20-25% (avg@8).
- Clean, direct answer: GSM8K ~25%, MATH-500 ~10%, AIME ~0-3%. **AIME direct is at floor.**
- Medium J-ablation, direct answer, GSM8K: falls to ~5-10%.
- Medium J-ablation, CoT, GSM8K: falls to ~60-75%, i.e. clearly protected.
- Medium J-ablation, CoT, AIME: falls to near zero — protection largely gone.
- Norm-matched random control at medium strength: within a few points of clean everywhere.
- Probability the ablation is *not* selective enough at 4B to interpret any of this (H3): **~35%.**

## 6. Things I expect to go wrong

- The J-lens readouts on a 4B model may simply not be interpretable. Validation gate in
  `design.md` §4 must pass before any ablation result is reported as meaningful.
- Ablation is applied at every generated token, so a longer chain of thought receives *more*
  total intervention. Harder problems produce longer chains. That confound points the same way
  as H2 and would fake it. Token-matched conditions are mandatory, not optional.
- Compute. There is no local GPU. Long-CoT generation with forward hooks and a paired clean pass
  is roughly 4x the cost of ordinary generation and cannot use vLLM.
- **Automaticity from RL training.** The paper notes (`paper.md:5011`) that a behaviour reinforced
  enough to become a fixed circuit can bypass the J-space entirely. Qwen3-4B is heavily
  RL-trained on exactly this kind of maths. GSM8K-style reasoning may have been compiled into
  J-space-independent circuitry, which would make GSM8K look *robust* to ablation for a reason
  that has nothing to do with CoT. The direct-answer arm at low budget is the diagnostic: if
  GSM8K survives ablation even with no chain of thought at all, automaticity is the explanation,
  not externalisation.
- **Contamination.** GSM8K and AIME 2024 are almost certainly in Qwen3's training data; AIME 2025
  is likelier not. Reporting the two AIME years separately is the check; a large gap means the
  difficulty axis is partly a memorisation axis.

---

*Addenda (dated, append-only) go below this line.*

---

## Addendum 1 — 2026-08-05: rung 1.5 taxonomy, falsifiable metrics, and the numeric-readout gate

**Timing disclosure.** Written after the validation-gate probes and after four runs of the
(subsequently invalidated) Figure-24 grid v1, but before grid v2 completed and before any rung
1/1.5/2/3/4 experiment. What had been observed when this was written: (a) multi-hop *concept*
readouts work on Qwen3-4B (Paris/colour surface mid-band); (b) the `calc:` arithmetic-chain
probe showed **no numeric intermediates** in the raw top-8 readout; (c) v1 partial results
cot/clean 75.5%, cot/medium 87.0% (confounded by the 1024-token cap; 47% length-stops),
direct/clean 20.5%, direct/medium 13.5%; (d) ablation suppresses EOS in the direct arm.
Everything below is registered *before* the corresponding experiments run.

### A1.1 Stage-1 replication criterion (Figure 24, GSM8K)

The replication claim is: at medium strength, `rho(GSM8K) > 1` with bootstrapped 95% CI
excluding 1, **and** the norm-matched random control gives `rho ≈ 1` (CI containing 1).
Anything else is a failed or confounded replication and will be reported as such.
Length-stop rates are part of the measurement and are reported per condition; any condition
with >10% length-stops is flagged as budget-confounded.

### A1.2 The four-way taxonomy is the object of study

The "interchangeable scratchpads" sentence is treated as four competing relationships between
page and workspace: (1) exchange/offload, (2) duplication, (3) complementary, (4) epiphenomenal.
**Primary expectation:** (1) holds for serialisable content (consistent with H2), with (2) as
the live alternative; I assign them roughly 45/30, with ~15 on (4) and ~10 on (3).

### A1.3 Directly falsifiable metrics for rung 1.5

**M-1.5a — maintenance signature (primary, lens-based).** On synthetic chains where value `v`
is computed at step `i` and next used at step `i+d`: let `z(v, l, t)` be occupancy z-scored
against magnitude-matched decoys at the same (layer, position), and `Z_gap(arm)` its mean over
the gap positions and band layers. The registered statistic is
`Delta_gap = Z_gap(direct) − Z_gap(CoT)`, bootstrapped over problems.

- Exchange: `Delta_gap > 0` (CI excludes 0), CoT-arm `Z_gap` at decoy level, and a re-read
  spike at `i+d` in the CoT arm.
- Duplication: both arms elevated over decoys and CI on `Delta_gap` contains 0.
- Uninterpretable unless both positive controls pass: pre-write occupancy rise for `v`
  (z ≥ 2 somewhere in the band) and operand occupancy above decoy level.

**M-1.5b — dual-task probe (convergent, behavioural + lens).**
`Delta_recall = recall(CoT) − recall(direct)` for a held secondary item, CI over items.
Exchange predicts > 0; duplication ≈ 0. **Pre-committed diagnosticity check:** the probe is
marked *uninformative* (not evidence for duplication) unless the load manipulation itself
bites, i.e. recall-under-math < recall-no-math baseline by ≥ 10 points. (The paper's §A.17
shows held items barely degrade under load; if holding is free here too, a null Delta_recall
is predicted by *both* hypotheses.)

**M-1.5c — behavioural gap-length effect (lens-free).** Direct-arm accuracy declines with gap
`d` faster than CoT-arm (negative interaction, CI excluding 0), and ablation hurts the direct
arm disproportionately at large `d` (three-way interaction, same sign requirement).

### A1.4 The numeric-readout gate (forced by the dataset mandate)

All three mandated datasets (GSM8K, MATH-500, AIME) have numeric intermediate state, and the
`calc:` probe suggests numeric readout may fail at 4B. Rungs 1 and 1.5a are conditional on this
gate:

**Gate metric.** On ≥ 40 synthetic arithmetic prompts: AUC separating true intermediates from
magnitude-matched decoys, using best (max) z-scored occupancy over band layers × positions, and
separately best rank *within the numeric token sub-vocabulary* (raw top-k is dominated by
punctuation/motor tokens; that is a readout-presentation artifact, not absence).
**Pass: AUC ≥ 0.70 with bootstrap CI excluding 0.5** on either statistic.

**Remediation ladder, in order, before declaring failure:** (i) all-positions readout +
numeric-subvocabulary ranking; (ii) multi-token number handling (max over constituent-token
variants, with and without leading space); (iii) refit the lens to the penultimate layer (the
public lens reads to the final layer — a documented deviation); (iv) prompt-format variants
closer to the fitting distribution.

**If the gate fails after remediation:** the registered interpretation is "numeric intermediates
are not J-lens-readable at 4B scale" — an H3-flavoured scale finding, reported as a result.
Rung 1.5 then rests on M-1.5b and M-1.5c (which do not need numeric readout; the lens
demonstrably reads word-like content, and M-1.5c needs no lens at all), and the maintenance
signature may be run on word-valued chains **as mechanism evidence only** — the headline stays
on the mandated numeric datasets.

---

## Addendum 2 — 2026-08-05 (evening): design corrections forced by the day's measurements

**Timing disclosure.** Written after: the digit-by-digit tokenization discovery; the numeric
gate landing at AUC 0.67 (marginal); the rung-1.5 maintenance pilot (no idle-value carry in
either presentation mode; direct-mode accuracy collapsing with reuse gap 29→35→6%); the
single-digit GSM8K pre-surfacing result (mid-band presence for single-token answers only);
and the GSM8K headline (ρ=1.63, CoT retention 1.01). Written **before** the runs it registers.

**A2.1 Content constraint (applies to rungs 1, 1.5, 3-targeted, 4).** All lens-read or
lens-ablated values must be single-token (digits 1–9 or single-token words). Multi-digit
numbers are structurally unreadable on this tokenizer (battery B: AUC 0.46) and any
correspondence/occupancy null on them is uninformative about the workspace.

**A2.2 Rung 1 status.** Correspondence on synthetic single-digit chains is treated as
established by the day's probes (intermediates surface at the computation site during direct
answering, target-vs-none AUC 0.67; ordering ρ=0.70; single-digit GSM8K answers pre-surface
mid-band). Remaining registered check: the problem-shuffled null (a prompt's targets scored
against a different prompt's readout) — re-analysis of existing probe data, no new runs.

**A2.3 Rung 1.5 primary instrument = dual-task word probe (registered before running).**
Design: single-digit GSM8K problems with a held single-token word ("Remember this word: X"),
answered in both arms (CoT / direct), clean and medium-ablated; measure (i) recall accuracy,
(ii) J-lens occupancy of the held word during the solving span, z-scored against unused
candidate words. Registered outcomes:
- *Held-item persistence* (paper §A.17 analogue): word occupancy above decoy level during
  solving, in whichever arm — the workspace holds idle verbal content even mid-computation.
- *Token-exclusion*: word occupancy at decoy level during computation-heavy spans in both
  arms — extends the pilot's radical-externalization reading from digits to words, and the
  workspace is then a pure compute buffer at 4B.
- *Arm asymmetry* (exchange signature): occupancy reliably lower in the CoT arm than direct.
Behavioural prediction: recall stays near ceiling in all conditions (the word remains in
context and re-reading suffices); a recall DROP under ablation in the direct arm specifically
would instead indicate workspace-dependent retrieval. No binary exchange/duplication verdict
is claimed from this alone; it discriminates the three outcomes above.

**A2.4 Depth ladder operationalized (tests the §3 registered prediction D ∈ [3,7]).**
Fully sequential single-digit chains (each line consumes the previous line's value), length
L ∈ {2..8}, direct mode. D := the largest L at which the final answer's digit still surfaces
in the band readout before emission (z ≥ 2 at the answer-adjacent position). Registered
companions: direct-mode accuracy collapse point vs L, and (if grid time allows) CoT-vs-direct
protection onset vs L. The §3 prediction stands as written: D ∈ [3,7], crossover within ±2 of D.

**A2.5 Third-outcome registration for the taxonomy.** The four-way table gains an explicit
fifth possibility, suggested by the pilot: *radical externalization* — the workspace maintains
no idle values in any mode; context is the only store; "interchangeable scratchpads" is then
wrong not because the page fails to substitute for the workspace but because they were never
doing the same job (workspace = per-step computation, page = all storage). This is the reading
the pilot's evidence currently favours and it is registered here before the confirmatory runs.

### A1.5 Dataset commitments

Mandated: GSM8K, MATH-500, AIME. All three are reported for every headline analysis. The
difficulty *inference* is carried primarily by the MATH-500 level 1–5 gradient (topic, format,
answer type held constant); the GSM8K → MATH-500 → AIME ordering is the mandated presentation
and robustness check; where they disagree, MATH-500 levels are believed. AIME sets, exactly:
**2024 I+II = `HuggingFaceH4/aime_2024` @ `2fe88a2f1091` (30 problems); 2025 I+II =
`MathArena/aime_2025` @ `c94da77eb22b` (30 problems)**; years reported separately as the
contamination check (§6). `rho` is suppressed as *uninformative* wherever
`acc_clean(direct) < 10%` (§4), which AIME direct will almost certainly trigger.

## Addendum 3 — 2026-08-06 (pre-dawn): masked-maintenance deconfound, run post-hoc

Addendum 2's A2.5 registered a fifth taxonomy outcome ("radical externalization":
the workspace maintains no idle values in any mode; context is the only store) as
the reading the pilot favoured but the confounded maintenance design could not
confirm. This addendum records the deconfounding instrument, designed and run
AFTER the main campaign (post-hoc; no outcome was registered in advance beyond
A2.5 itself).

**Design (scripts/run_masked_maintenance.py).** A 4D attention mask makes the
definition span (`x = 7.`) unreadable from the question span onward while gap
(filler) positions retain normal access, so re-reading at the point of use is
impossible but internal carrying through gap-position residuals is not. Four
conditions: baseline (causal only), masked_use (question blocked from the
definition), severed (gap AND question blocked — the floor control), page
(masked_use plus a restatement inside the gap — the written-page analog).
Answer read as argmax over digit tokens at the final position; n = 96 items
(6 names x 8 values x 2 fillers; demo value 3 excluded from the value set).

**Result (results/cedar/20260806T010855Z_masked_maintenance).** baseline
100%, masked_use 13.5% [8.1, 21.8], severed 12.5% [7.3, 20.6], page 100%.
Gap occupancy of the held digit z ~= 0.03. The model does not carry a single
literal digit across a ~30-token gap; retrieval is re-reading, and a visible
later copy fully substitutes. A2.5's registered reading is confirmed for
literal single-digit values at this gap length.

## Addendum 4 — 2026-08-06 (pre-dawn): during-CoT persistence probe, run post-hoc

Same session as Addendum 3. The correspondence test (A2.x taxonomy) measured
the workspace in direct mode only; duplication/exchange concern the CoT arm.
scripts/run_cot_persistence.py teacher-forces worked transcripts and reads
v1's occupancy in write / post / use windows against the unworked direct arm
(n=80, constraints exclude v1's digit from lines 2-3). Registered post-hoc;
predictions stated in the script header before the run: duplication =
persistence after the write, exchange = write-only appearance.

Result: cot z 0.47 / 0.08 / 0.42 (accuracy 100%); direct 0.24 / 0.18 / 0.08
(accuracy 13.75%). Written values vanish from the workspace between writing
and re-reading. Duplication rejected; exchange emptied of its premise (no
holding in either arm for writing to remove). A2.5's reading is now
supported by four convergent instruments.
