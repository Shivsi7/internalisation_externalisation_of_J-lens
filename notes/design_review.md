# Design review — every experiment, its logic, and where it is weak

Written 2026-08-05 evening, at the user's request, while the three chains were
running. Mirrors the review delivered in-session; kept in-repo for revision
while writing the report.

---

## A. The claim ladder

Everything hangs off one sentence in the paper: *"written chain of thought and
the internal J-space appear to be partially interchangeable scratchpads."*
The design decomposes it into an ordered ladder of questions, each answerable
only if the one below it holds:

1. **Does the lens read anything at 4B?** (validation gate) — if not, nothing
   above is interpretable and the honest report is about scale.
2. **Does ablation damage selectively?** (random control) — if norm-matched
   noise does the same damage, nothing is about the J-space.
3. **Does CoT protect against that damage?** (Fig-24 replication) — the
   paper's phenomenon itself.
4. **Does protection track difficulty/serialisability?** (MATH levels, AIME,
   depth ladder) — H1 vs H2.
5. **What KIND of relationship is it?** (exchange/duplication/complementary/
   epiphenomenal — rungs 1–4) — the interpretation.

Tonight's runs answer 1–4. Rung 5 is designed, pre-registered, and not run —
the report must claim accordingly.

## B. Tonight's experiments

### B1. Validation gate (probes + numeric gate)

**Measures.** Whether J-lens readouts on Qwen3-4B carry interpretable content:
concept probes (qualitative), digit-chain occupancy AUC (quantitative,
pre-registered pass ≥ 0.70), computation-order signature (Spearman).

**Status.** Concepts: pass (Paris at L27–31). Numeric: AUC 0.67 [0.59, 0.74]
— above chance, formally under the gate. Ordering: ρ=0.70, p=2e-8, n=49.
Battery B (leading digit of 2-digit values): chance, as the paper's
multi-token result predicts — every number ≥ 10 is multi-token on Qwen3.

**Weak joints.** (i) The z-statistic normalises across only 10 digit classes →
ceiling ≈ 2.85 and an inflated null under max-over-cells; the AUC is therefore
conservative but coarse. (ii) The "operand" positive control was mis-specified
at the answer position (operands are consumed there — their suppression is
itself structure, but it means the gate currently lacks a clean positive
control; the right one is operand occupancy at the operand's own position).
(iii) Public lens reads to the final layer, paper's default is penultimate;
refit not done. The gate verdict "marginal, not absent" is the honest one.

### B2. Figure-24 replication grid (GSM8K)

**Measures.** Score retention (ablated/clean) per arm per strength, and
ρ = retention(CoT)/retention(direct).

| | Control | Tests |
|---|---|---|
| Pos | clean baselines per arm | anchors; floor rule needs them |
| Pos | ablation strengths ordered light→heavy | dose sanity |
| **Neg** | **norm-matched random directions (medium)** | **the carrying null: equal norm removed, wrong directions — kills "any perturbation does this"** |
| Neg | pre-registered floor rule (clean direct <10% → ρ suppressed) | kills the floor artifact |

**Inference logic.** Selectivity = J-retention ≪ random-retention (already:
0.62 vs 0.88, direct arm). Protection = retention(CoT) > retention(direct)
with CI; replication criterion registered as ρ>1 with CI excluding 1 AND
random-ρ ≈ 1.

**Weak joints, ranked.**
1. **B8 (exclusion filter) not run tonight.** The filter spares the clean
   model's top-10 next tokens; in direct mode the imminent token IS the
   answer, so the filter differentially shields the direct arm. Direction of
   bias: protects direct → pushes ρ toward 1 → a positive ρ result is
   conservative w.r.t. this confound, but a null result would be ambiguous.
   One flag away (`--no-filter`); first thing to run with extra time.
2. **avg@1.** Single sample at T=0.6; n=200 gives ±5–6pt Wilson CIs per cell,
   and ρ's CI (bootstrap ratio-of-ratios) will be wide. Registered as
   compute-limited; greedy-degeneration on Qwen3 rules out the cheap greedy
   alternative.
3. **Caps are load-bearing.** v1 (cap 1024) inverted the CoT effect via
   truncation — caught, discarded, logged. v2 cap 2048 still truncates 19.5%
   of clean-CoT chains (39/200, mean 1176 tok) → above the registered 10%
   flag; the clean-CoT 91.0% is a floor on the true ceiling. All conditions
   share the cap (matched budget), so retention ratios are defensible;
   absolute accuracies are not ceilings.
4. **Token-count asymmetry runs the right way.** CoT receives ablation over
   ~10× more positions than direct yet is predicted (and observed in v1) to
   survive better — the confound is anti-conservative for the null, i.e.
   safe for a positive result.
5. **Band port is naive** (depth fractions → L22–26/19–28/15–32); the gate's
   band re-derivation criterion was not run. The direct arm's observed
   non-monotonicity (light 10.0% < medium 13.0% < heavy 15.5%) is within CI
   overlap but may be a band-placement artifact — flagged, unexplained.
6. **Automaticity.** Qwen3-4B is RL-trained on grade-school math; GSM8K
   circuits may partially bypass the J-space. Diagnostic (registered): if
   direct-arm GSM8K had survived ablation, automaticity — it did not
   (21→13%), so the workspace is doing work even in the direct arm.

### B3. MATH-500 chain (n=243, stratified by level)

**Measures.** The difficulty axis with topic/format/answer-type held constant
— the within-dataset level 1–5 gradient is the primary difficulty inference
(cross-dataset ordering is presentation + robustness; if they disagree,
believe the levels).

**Weak joints.** Clean direct accuracy at levels 4–5 will likely trigger the
floor rule per-level (report per-level ρ only where licensed); 3072 cap will
truncate hard-level chains (report per-level length-stops); n≈49/level makes
per-level CIs ±13pts — the gradient claim rests on the trend across five
levels, not any single level.

### B4. AIME chain (60 problems, years split)

**Measures.** The hard end. Direct clean = 0/60 (already measured): the floor
rule fires, ρ is uninformative — by design, and reporting that the rule fired
is itself a result (the assignment's "design so the opposite result is
visible" requirement, exercised). The informative quantities: CoT-arm
retention under ablation, and the 2024-vs-2025 split as the contamination
check (2024 is pre-cutoff, 2025 largely post-).

**Weak joints.** n=60 → ±10pt CIs at best; 4096 cap truncates most AIME
thinking chains (Qwen3-4B typically wants ≫4k tokens) so absolute accuracy is
strongly budget-limited — retention at matched cap is the only defensible
statistic; year subsets are n=30 each, directional only.

## C. Designed, pre-registered, not run tonight

- **Rung 1 (correspondence)** — do direct-mode workspaces hold the values CoT
  writes? Rank-based AUC vs the problem-shuffled null. Now constrained by the
  digit-tokenization finding: single-digit construction required.
- **Rung 1.5 (exchange vs duplication)** — the maintenance signature
  (occupancy across the computed→reused gap; dip+spike = exchange, sustained
  = duplication) + dual-task capacity probe. The addendum's M-1.5a/b/c carry
  the falsifiable statistics. The whole rung sits on the marginal numeric
  gate — its riskiest dependency.
- **Rung 2 (substitution + filler control)** — iso-accuracy contours over
  (k × CoT budget); the filler control is the ONLY thing in the design that
  can kill the epiphenomenal reading (CoT = more forward passes, not
  storage). Its absence tonight means the epiphenomenal alternative is
  UNADDRESSED in the report — must be stated.
- **Rung 3 (compensation; prefix crossover; B7 read/write split)** — does the
  ablated model externalize *because* damaged; does its own ablated CoT serve
  it better than the clean model's (crossover = active compensation). B7
  targets the paper's §9.3 re-ingestion seam.
- **Rung 4 (targeted causal)** — ablate exactly the direction rung 1 found;
  rescue by injecting it. Strongest test; farthest away.
- **C4/C5 (task battery + fluency)** — capability-selectivity. Tonight's
  selectivity evidence is direction-specificity (random control) ONLY; the
  claim "most capabilities intact" is NOT tested and must not be asserted.
- **C7 (scale sweep 1.7B–14B)** — pre-fitted lenses exist for all; converts
  "4B is too small" from a limitation into a measurement. Pure inference
  cost; first thing to add with another day.

## D. What tonight licenses, and what it forbids

**Licensed, with CIs:** the workspace exists and is structured at 4B (concept
readouts, ordering signature, direction-specific damage); the Fig-24 CoT/
direct contrast at matched budgets on GSM8K; the difficulty trend across
MATH-500 levels; AIME CoT retention at matched budget; the floor rule firing
on AIME as predicted.

**Forbidden without the unrun controls:** "interchangeable scratchpads"
(rungs 1.5–4 unrun); "storage, not compute" (filler unrun); "most
capabilities intact" (battery unrun); any absolute-accuracy claim where the
length-stop flag fired; per-level ρ where the per-level floor rule fires.

**The report's honest shape:** a replication-with-controls of the phenomenon
on a 100×-smaller model, a difficulty-axis extension with the floor artifact
handled by design, a tokenizer-dependence discovery the paper could not have
seen, and a pre-registered decision procedure for the interpretation —
executed up to rung 1's doorstep, with the remaining rungs as designed,
falsifiable next steps.
