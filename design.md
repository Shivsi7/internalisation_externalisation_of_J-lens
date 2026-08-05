# Experiment design

Companion to `hypotheses.md` (the pre-registration). This document is the plan; it may be revised
freely *before* the first result, and revisions after that are logged in a changelog at the bottom.

---

## 1. What the paper actually established, and what it left open

Worth being precise, because the assignment's framing is slightly more confident than the paper is.

**Established (§3.5.2, `paper.md:2348-2518`):** with k=10 J-lens directions projected out of the
residual stream across a band of layers, Sonnet 4.5 retains MMLU, SQuAD, CoLA, sentiment, and
odd-one-out at near-baseline even under heavy ablation, while Caesar decoding, analogy,
summarisation, TriviaQA, multi-hop, translation and sonnet-writing fall below *unablated Haiku
4.5*. GSM8K with CoT is "substantially more robust" than GSM8K answered directly.

**Not in the paper, and this matters:**

- **No numbers for the GSM8K comparison.** Figure 24 is normalised bars; the text gives only the
  relative claim. No n, no absolute accuracies, no prompt formats. How "direct answer" was
  elicited is never stated anywhere in the paper.
- **MATH, MATH-500, AIME, and competition maths do not appear in the paper at all.** GSM8K is the
  only maths benchmark. There is no difficulty stratification and no step-count analysis.
- **k was never swept.** Only the layer band was (light/medium/heavy). The one remark on k is
  qualitative: "using larger values of k and/or later layer ranges tends to impair the coherence
  of responses" (`paper.md:2526`).
- **The paper flags its own unresolved seam** (§9.3, `paper.md:5020`): CoT tokens are written *and
  then re-ingested*, and re-ingestion runs them back through the workspace band. So a J-space
  ablation applied during generation also damages the model's *reading* of its own chain of
  thought. CoT is therefore not a clean bypass of the workspace, and the paper never resolves
  what that implies for the interchangeability claim. §2 below turns this into a condition.

So the assignment is asking us to extend a result whose absolute magnitude was never published,
onto two datasets the paper never touched, using a model roughly two orders of magnitude smaller
than the one it was measured on. The design has to be built around that, not around a
reproduction.

## 2. Three design problems, and what each one forces

Everything unusual in the design below exists to handle one of these.

### 2.1 The floor problem — forces a continuous externalisation axis

Qwen3-4B answering AIME with no chain of thought scores near zero *before* any intervention. A
score that starts at zero cannot fall, so "CoT is more robust than direct" comes out true on AIME
for reasons that have nothing to do with the workspace. The paper's binary CoT/no-CoT contrast
does not survive the move to hard problems.

**Fix:** replace the binary with a **CoT token budget** swept continuously
(0, 64, 128, 256, 512, 1024, 2048 tokens, forced by injecting `</think>` and an answer prefix at
the budget). Externalisation becomes a dial rather than a switch, every dataset gets a range of
clean accuracies rather than a single floored point, and the object of study becomes a *surface*
— accuracy as a joint function of ablation strength and externalisation — instead of a single
ratio. Iso-accuracy contours on that surface answer the interchangeability question directly:
downward-sloping contours mean CoT tokens can be traded for workspace capacity, L-shaped or
vertical contours mean they cannot.

### 2.2 The confound problem — forces token-matching

Ablation is applied at *every* token position during generation. Harder problems produce longer
chains of thought, so a harder problem receives strictly more total intervention. That alone would
produce H2's predicted pattern with no bandwidth mechanism involved at all.

**Fix:** (a) the budget sweep already equalises generated-token counts across datasets at each
budget level; (b) report ablation damage per generated token as well as per problem; (c) the
synthetic depth ladder (§3, C3) varies chain length with per-step difficulty held fixed, which
isolates length from difficulty; (d) an **ablate-on-read vs ablate-on-write** split
(§3, factor B) separates damage to producing the chain from damage to consuming it.

### 2.3 The degradation problem — forces the control battery

The assignment explicitly requires distinguishing J-space ablation from broad model degradation,
and the paper gives a specific reason to worry at this scale: on **Haiku 4.5** — already much
larger than Qwen3-4B — "J-space ablation degrades coherence before yielding any qualitative
change in responses" (`paper.md:2526`). §9.1 lists "smaller models have ... none at all" as a live
possibility (`paper.md:5000`).

**Fix:** four norm-matched perturbation controls, an automatic-task battery, a fluency measure,
and a smaller-model reference bar (§5). Plus a hard validation gate (§4) that must pass before any
ablation result is interpreted.

---

## 3. Conditions

Factorial in principle, staged in practice (§7). Factors:

### Factor A — externalisation level

| | Condition | Implementation |
|---|---|---|
| A0 | Direct | non-thinking mode, answer-only format, prefill `The answer is` |
| A1 | Short CoT | non-thinking mode, "show your work briefly" |
| A2 | Long CoT | thinking mode, budget B tokens |
| A2* | Budget sweep | B ∈ {0, 64, 128, 256, 512, 1024, 2048}, forced `</think>` at budget |

Using `Qwen/Qwen3-4B` (the hybrid checkpoint) rather than `Qwen3-4B-Thinking-2507` is deliberate:
the hybrid supports both thinking and non-thinking modes *from identical weights*, so the CoT
manipulation is within-model rather than across-checkpoint. That is the cleanest available analogue
of the paper's contrast.

### Factor B — intervention

| | Condition | Purpose |
|---|---|---|
| B0 | Clean | baseline |
| B1 | J-ablation, k=10, light / medium / heavy band | the effect |
| B2 | Random-direction, norm-matched | paper's control (`paper.md:2362`) |
| B3 | **Rank-shifted J-vectors** (ranks 51–60 instead of 1–10) | strongest control — same dictionary, same band, same k, similar norms, just not the active ones |
| B4 | Non-J control: shrink the orthogonal complement, norm-matched | paper's §A.23 control |
| B5 | Logit-lens ablation (top-k rows of `W_U`) | method ablation: does the Jacobian correction matter, or would a naive lens do? |
| B6 | k sweep, k ∈ {1, 3, 10, 30} | the paper never swept k; needed for the contour plot |
| B7 | **Ablate-on-write only** / **ablate-on-read only** | resolves the §9.3 seam: is CoT damaged in production or in re-ingestion? |
| B8 | **Exclusion filter on / off** | the filter may itself create the CoT-vs-direct asymmetry — see §6.2(4) |

B3 is the control I would defend hardest. A random direction is a weak null because it is almost
orthogonal to everything the model is doing; rank-shifted J-vectors are drawn from the same
overcomplete token-indexed frame and are genuinely active, so if they do no damage, "the *top* of
the J-space specifically" is doing the work.

B7 is only possible because we control generation: ablate on generated positions but not when
attending back to them (write-only), or the converse (read-only). This is a condition the paper
could not run in its own framing and it targets the seam it left open.

### Factor C — task

| | Task | n | Role |
|---|---|---|---|
| C0 | GSM8K test | 500 | paper's anchor |
| C1 | MATH-500 | 500, stratified by level 1–5 | difficulty axis *within* one dataset |
| C2 | AIME 2024 I+II and 2025 I+II | 60 | hard end |
| C3 | **Synthetic depth ladder** | chain length ∈ {2,4,8,16,32}, 200 each | decouples depth from bandwidth |
| C4 | Automatic battery: MMLU, SQuAD, CoLA, sentiment, odd-one-out | 500 each | selectivity control |
| C5 | Fluency: WikiText-103 perplexity, top-1 agreement with clean | 50k tokens | degradation control |
| C6 | Clean Qwen3-1.7B, all of C0–C2 | — | "what does a genuinely weaker model look like" bar |
| C7 | **Ablated scale sweep: Qwen3-1.7B / 4B / 8B / 14B** | subset of C0–C2 | does the trade-off emerge with scale? |

**AIME source must be stated exactly** (assignment requires it). Plan: AIME 2024 I+II and 2025 I+II,
60 problems, integer answers in [0,999], taken from the MathArena release with problem text
verified against the official AoPS statements. Record the exact HF revision hash in the README.
2025 is post-cutoff for much of Qwen3's training data and 2024 is not, so **report the two years
separately** — a large 2024/2025 gap is a contamination signal, and it would otherwise masquerade
as a difficulty effect.

**C1 is where the difficulty analysis actually lives.** MATH-500's levels 1–5 give a difficulty
gradient with topic, format, answer type, and prompt style held constant. The GSM8K → MATH → AIME
comparison confounds difficulty with all of those; the within-MATH-500 level sweep does not. If
the two agree, the cross-dataset story is safe; if they disagree, believe C1.

**C7 is newly affordable and worth taking.** Neuronpedia hosts pre-fitted lenses for
`qwen3-1.7b`, `qwen3-4b`, `qwen3-8b`, `qwen3-14b` and `qwen3-32b` in the same repo, so a scale
sweep now costs inference only — no extra fitting. This matters for two reasons. It converts the
H3 risk from a threat into a measurement: if the trade-off is absent at 1.7B, weak at 4B and clear
at 14B, then "4B is too small" becomes a *finding* with a trend behind it rather than a failed
experiment. And it answers the paper's own stated open question almost verbatim — §9.1
(`paper.md:5000`) asks "whether smaller models have an equally rich workspace, a proportionally
smaller one, a less reliable one, or none at all," and notes the workspace was studied only in
models "of considerable size." Qwen3-4B remains the mandated headline model; the sweep is context
for interpreting it.

**C3, the depth ladder,** is the discriminating condition from `hypotheses.md` §3. Chains of
elementary operations (`a = 7; b = a*3 - 2; c = b + 9; ...`), answer = final variable. Length is
the only knob. These are depth-hard and bandwidth-trivial, and they can be tuned to any clean
accuracy we like — including down to AIME's, which makes matched-difficulty comparison possible.

**A quantitative prediction C3 can test.** §3.3 (`paper.md:1942`) finds that on
`(4+17)*2+7`, the intermediates 21 → 42 → 49 reach top-1 in order, separated by roughly 8
reindexed layers per step, inside a workspace band ~54 reindexed units wide. That implies a
*serial depth budget* of roughly 6–7 chained steps per forward pass. Ported to Qwen3-4B's ~19-layer
band, we can measure this budget directly with the J-lens (at what chain length does the final
answer stop appearing in the readout before the answer token?) and then check whether the chain
length at which CoT becomes protective coincides with it. Prediction under H2: it does — CoT
protection switches on exactly where internal depth runs out. That is a much sharper claim than
the ordering prediction and it is falsifiable to within a couple of steps.

---

## 4. Validation gate (must pass before any ablation result is reported as meaningful)

Non-negotiable, because everything downstream is conditional on the J-lens meaning something on a
4B model.

1. **Lens interpretability.** Reproduce the qualitative probes on Qwen3-4B: multi-hop
   ("the colour of the planet fourth from the sun is" → `color` / `Mars` / `red` at intermediate
   layers), and the arithmetic chain (`calc: (4+17)*2+7 =` → `21` / `42` / `49` in order). If the
   readouts are noise, say so and pivot to reporting *that*.
2. **Band localisation.** Re-derive the workspace band for Qwen3-4B rather than porting L38–92
   blindly — the paper's §9.1 explicitly warns the boundary was never principled. Use its cheapest
   two of five criteria: excess kurtosis of the readout (marks band start) and top-k next-token
   accuracy of the lens (marks the motor transition, band end). Naive port for reference:
   38–92% of 36 layers ≈ **layers 14–33**; light 62–71% ≈ 22–26; medium 54–79% ≈ 19–28;
   heavy 42–88% ≈ 15–32.
3. **Positive control.** Multi-hop factual accuracy must fall substantially under ablation and not
   under the norm-matched controls. This is the paper's own gate (clean 98% → 26% at medium, vs
   98% for random control).
4. **Selectivity floor.** MMLU/CoLA/SQuAD must survive at the ablation strength used for the main
   result. If they do not, we are in H3 and the honest report is a characterisation of coherence
   collapse at 4B, with the strength sweep showing there is no window where the effect is
   selective.

A pre-committed escape hatch: if selectivity holds only at a weaker setting than the paper's,
**use that setting and say so**, rather than forcing the paper's k=10/medium band.

---

## 5. Metrics

**Primary.** Exact-match accuracy with `math-verify` / sympy equivalence for MATH and AIME,
numeric match for GSM8K. avg@8 at temperature 0.6 (Qwen3's recommended setting; greedy decoding is
known to degenerate on this family), plus a greedy run for reproducibility. Bootstrap 95% CIs over
problems, clustered by problem across the 8 samples.

**Protection ratio.** The headline scalar:

```
rho(dataset) = [acc_ablated(CoT) / acc_clean(CoT)] / [acc_ablated(direct) / acc_clean(direct)]
```

Reported with CIs, and **suppressed with an explicit "uninformative" marker wherever
`acc_clean(direct) < 10%`** (pre-registered in `hypotheses.md` §4). On AIME this will almost
certainly trigger, which is exactly why rho is not the only headline.

**The main figure.** Iso-accuracy contours over (ablation strength k) × (CoT budget B), one panel
per dataset. Substitutability is legible directly from contour slope. This is the figure the report
should be built around.

**Exchange rate.** From those contours: how many additional CoT tokens buy back the accuracy lost
to ablation at k=10? Finite on GSM8K and infinite (unrecoverable) on AIME is H2's signature.

**Chain-of-thought quality decomposition.** Final-answer accuracy alone cannot distinguish "the
model can no longer plan" from "the model can no longer add." Score generated chains on:
- per-step arithmetic correctness (programmatic — extract and check each equation);
- **planning errors** (the next step chosen is valid but does not advance toward a solution);
- **state-tracking errors** (a previously written value is mis-copied or contradicted);
- degeneracy: repetition rate, chain length, failure to terminate, format compliance.

H2 predicts ablation raises *planning* errors specifically, and that CoT protects specifically
against *state-tracking* errors. This tests the mechanism rather than only the outcome, and it is
what makes a negative headline result still publishable-in-miniature. LLM-judged with a rubric
plus programmatic checks; judge agreement reported.

**Degradation controls.** WikiText perplexity, top-1 next-token agreement with the clean model
(the paper's own measure: 87/75/65% for light/medium/heavy vs 96% random control), automatic-task
battery scores.

---

## 6. Implementation

### 6.1 The J-lens: already solved, do not rebuild

Anthropic released a reference implementation at
[`anthropics/jacobian-lens`](https://github.com/anthropics/jacobian-lens) (Apache-2.0, ~1700 lines:
`fit` / `JacobianLens.apply` / slice visualisation), and **Neuronpedia hosts a pre-fitted lens for
exactly the required model**:

```
repo:     neuronpedia/jacobian-lens        (revision: main)
file:     qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt   (458 MB)
config:   qwen3-4b/jlens/Salesforce-wikitext/config.yaml
```

Verified from that config: `hf_model_name: Qwen/Qwen3-4B`, fitted on Salesforce/wikitext-103
(1000 prompts requested, **479 actually used** — the fit early-stopped on a 0.002 relative-change
criterion), `max_seq_len: 128`, `dim_batch: 128`, bf16. `source_layers` defaults to all layers
below the target, so **all 35 source layers are present** and every band we might want is covered.
`final_identity_distance: 0.391`, so `J` is substantively different from the identity. A
`convergence.csv` ships alongside it, which gives the convergence evidence for free.

**This removes the single most expensive and most error-prone stage of the plan.** Fitting is now
a 458 MB download rather than a GPU-hours reimplementation of a Jacobian estimator.

Two deviations from the paper to document rather than fix:

1. The public fit uses `target_layer: null` → the **final** layer. The paper's default reads to the
   **penultimate** layer, because including the last block "can sometimes increase the number of
   noisy artifacts" (`paper.md:5467`). Note it; if readouts look noisy at the motor end, refitting
   to the penultimate layer is a cheap fallback since `jlens.fit` takes `target_layer`.
2. 479 prompts, not the paper's 1000. Given `paper.md:5551` ("beats the baselines with as few as
   10 prompts") and the shipped convergence trace, this is not a concern, but it belongs in the
   report's methods.

**What the repo does *not* provide, and we must therefore build: the ablation.** Grepping the
library for ablate/steer/swap/patch/project returns nothing. `jlens.hooks.ActivationRecorder` is a
read-only forward hook that captures residuals; `JacobianLens.apply` is `@torch.no_grad()`, takes
a single prompt string, and does not generate. So the entire intervention — and batched *ablated
generation*, which is what the experiment actually runs on — is ours to write. That is now the
main engineering task and the plan should be resourced accordingly.

One free gift: `apply(..., use_jacobian=False)` skips the `J_ℓ` transport and gives the vanilla
logit lens. Control B5 costs nothing.

Implementation note for the ablation: do **not** materialise `W_U J_ℓ` (152k × 2560 per layer
≈ 780 MB each). Score tokens on the fly as `W_U (J_ℓ h)` — one d-matvec then one V-matvec per
position — reusing `JacobianLens.transport` for the first half.

### 6.2 Ablation, and the three places the paper is silent

At each position and each layer in the band: score all vocab tokens by `W_U J_ℓ h`, take the top
k, gather their J-lens vectors `j_v = J_ℓᵀ W_U[v]`, and remove that component from `h`. Then
apply the paper's exclusion filter: **skip any token in the clean forward pass's top-10 next-token
predictions**, which requires running a paired clean pass alongside the ablated one.

Three underspecified points, each of which becomes a documented decision plus a sensitivity check:

1. **"Most strongly activated" is ambiguous.** The paper uses top-k-by-inner-product for readouts
   and non-negative gradient pursuit for its occupancy analyses; §3.5.2's wording matches the
   former. Default to inner product; check a gradient-pursuit variant on a subset.
2. **J-lens vectors are non-orthogonal**, so "zero the projection onto each" is order-dependent
   and is not a subspace projection. Default to a proper least-squares projection onto the span of
   the k vectors; check sequential projection as a variant.
3. **No renormalisation is mentioned.** Do not renormalise (matching the paper), since the controls
   are norm-matched *to the ablation's* norm change — renormalising would break that matching.
4. **The exclusion filter is not neutral between our two arms, and this is a threat to the
   headline result.** The filter spares any token in the clean pass's top-10 next-token
   predictions. In **direct** mode the imminent token *is the answer* — the very quantity the
   ablation is supposed to destroy — so the filter systematically shields the computation's
   output. In **CoT** mode the spared token is merely the next word of reasoning, and the
   load-bearing state lives in the chain rather than in the next token. The filter therefore
   treats the two arms differently by construction, and it could either mask or manufacture the
   CoT-protection effect depending on which way the asymmetry cuts. The paper never runs this
   contrast because it never needed the direct arm to be quantitative.

   **This becomes condition B8: run the main grid with the filter on and off.** If the CoT/direct
   gap is stable across the toggle, the filter is innocent and the result stands. If the gap
   moves materially, that is a finding *about the paper's own methodology* and should be reported
   as one — it would mean published J-ablation contrasts between "model says the answer" and
   "model says reasoning" are partly measuring the filter. Either outcome is worth the compute,
   which is why B8 sits in the must-run tier and not the stretch tier.

### 6.3 Generation

HF `transformers` with forward hooks; vLLM cannot host this. Every ablated rollout needs a paired
clean forward pass for the exclusion filter, so cost is ~2× ablated generation ≈ 4× plain
generation, and batching (16–32 sequences) is what makes it tractable. Budget-capped thinking is
not just a scientific control here, it is also the thing that keeps AIME rollouts affordable.

### 6.4 Compute

**There is no local GPU and no torch installed on this machine** (checked). Everything runs on
rented compute. A single A100-40GB or L40S at roughly $1–2/hr fits the assignment's $100 ceiling
comfortably at ~40–60 GPU-hours if the staging in §7 is respected. Qwen3-4B is ~8 GB in bf16.

With fitting removed from the budget, the freed hours go to the k × budget contour sweep (§5) and
the scale sweep (C7) — both of which are pure inference. Note that the pre-fitted lens is fp16 and
~458 MB on disk, which is a rounding error against the model weights; the memory peak is now
ablated generation with a paired clean pass, not the Jacobian.

---

## 7. Staging

Ordered so that the assignment is satisfiable even if later stages are cut. Each stage ends with a
committed artifact.

| Stage | Work | Cut-line |
|---|---|---|
| 0 | Eval harness, datasets, answer extraction, clean baselines across C0–C2 × A0–A2 | must |
| 1 | Download the pre-fitted lens; validation gate (§4) | must |
| 2 | **Write the ablation + batched ablated generation** (the repo has neither) | must |
| 3 | Ablation + all controls at one strength; 3 datasets × 3 externalisation levels | must |
| 4 | Budget × k sweep → the contour figure; depth ladder C3 | should |
| 5 | Scale sweep C7; CoT error decomposition; read/write split B7; depth-budget measurement | stretch |

The critical path has moved. Fitting the lens was the expected bottleneck and is now a download,
so stage 2 — writing an intervention the reference implementation does not contain — is where the
risk and the effort now sit. Budget accordingly: the hard parts are the top-k selection over a
151k-token vocabulary at every position and layer, the least-squares projection onto k
non-orthogonal directions, the paired clean forward pass needed for the exclusion filter, and
making all of it work under batched generation with a KV cache.

**Fallback if stage 2 overruns:** fall back to a **logit-lens ablation** (top-k rows of `W_U`),
which needs no `J_ℓ` and is a much simpler hook. The paper says the logit lens "captures much of
the workspace-like structure identified by the J-lens, though with somewhat lower reliability"
(`paper.md:1124`). Declare it as a proxy if it becomes the headline — but run it regardless as
control B5, since "does the Jacobian correction actually matter for this result?" is a real
question the paper does not answer.

## 8. Repository layout

```
report.pdf              # 2 pages, required at root
README.md               # structure, setup, reproduction, external links
hypotheses.md           # pre-registration, committed before any run
design.md               # this file
src/ablation/           # OUR code: top-k selection, projection, ablated generation hooks
src/eval/               # datasets, generation, answer extraction, scoring
                        # `jlens` (anthropics/jacobian-lens) is a pinned dependency,
                        # vendored or installed from the pinned commit — not reimplemented
configs/                # one YAML per condition; every run reproducible from a config
prompts/                # all prompt templates verbatim
results/                # raw generations (jsonl) + scored csv
notebooks/              # figure generation
scripts/                # end-to-end reproduction entry points
```

Every figure and table in the report must regenerate from `results/` via a script named in the
README, per the assignment's repository requirements.

---

*Changelog (append-only, dated) below this line.*
