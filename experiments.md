# Experiment plan

Companion to `hypotheses.md` (pre-registration) and `design.md` (rationale and controls).
This file is the **executable plan**: what runs, what it measures, and what the nulls are.

Written before any experiment was run. Revisions after the first result go in the changelog.

---

## 0. Structure

Three stages, then four rungs inside the third.

1. **Reproduce and validate** — does the paper's effect exist at all on Qwen3-4B?
2. **Extend** — does it hold on MATH-500 and AIME?
3. **Interrogate the interpretation** — is "interchangeable scratchpads" a fair reading?

Stage 3 is the contribution. Stages 1–2 exist to earn the right to run it.

### The claim, decomposed

> "The model externalizes onto the page what it would otherwise have to hold in its internal
> workspace: written chain of thought and the internal J-space appear to be partially
> interchangeable scratchpads."

That sentence bundles three separable commitments. The paper's evidence — a robustness
differential between CoT-GSM8K and direct-GSM8K under ablation — supports at most the first,
and only indirectly.

| Commitment | Assertion | Paper's evidence | Rung |
|---|---|---|---|
| **Correspondence** | It is the *same content* in both places | none | 1 |
| **Exchange** | Externalizing *frees* the workspace, rather than merely copying it | none | 1.5 |
| **Substitution** | Each can carry the load; they trade off | indirect | 2 |
| **Compensation** | The model *shifts* content to the page when the workspace is damaged | none | 3 |
| **Causal role** | The correspondence is functional, not epiphenomenal | none | 4 |

Every rung below carries at least one positive and one negative control. A rung with no null
is not evidence.

---

## 1. Stage 1 — Reproduce and validate

**What is *not* reproducible, and must be said in the report.** The paper publishes no numbers
for the GSM8K result: Figure 24 is normalised bars, the text gives only the relative claim, and
how "direct answer" was elicited is never stated anywhere. It was measured on Sonnet 4.5. So this
is a *qualitative replication attempt on a model two orders of magnitude smaller*, not a
numerical reproduction. Failure is a result, not a setback — it is evidence on §9.1's open
question about whether small models have a workspace at all.

**Validation gate** (from `design.md` §4 — must pass before any ablation result is interpreted):
lens interpretability probes, band localisation re-derived for Qwen3-4B rather than a blind port
of L38–92, the multi-hop positive control, and the MMLU/CoLA/SQuAD selectivity floor.

**Then the qualitative replication:** does CoT-GSM8K survive ablation better than direct-GSM8K?

Two threats specific to this stage, both pre-registered in `design.md`:

- **The floor problem.** Direct-answer accuracy near zero cannot fall, so the CoT/direct contrast
  is guaranteed for arithmetic reasons. Handled by the continuous CoT-budget axis.
- **The exclusion filter (B8).** The filter spares the clean pass's top-10 next tokens. In direct
  mode the spared token *is the answer*; in CoT mode it is the next word of reasoning. The filter
  is therefore not neutral between the two arms and could manufacture or mask the effect. **Run
  the grid with the filter on and off.** If the gap moves, that is a finding about the paper's
  methodology.

---

## 2. Rung 1 — Correspondence

**Question.** During *direct* answering, does the J-space hold the same intermediate values that
the *CoT* rollout writes down?

**Design.** For each GSM8K problem solved both ways: extract written intermediates from the CoT
(numbers, programmatic). On the direct rollout, run the J-lens across the band at every position
and ask whether those values surface in the readout.

**Metric — rank, not hit rate.** With layers × positions × top-K slots and small integers being
common, a thresholded hit rate is meaningless. For each candidate value take its *best (minimum)
rank* across the band, then compare rank distributions between true intermediates and decoys via
Mann–Whitney / AUC, bootstrapped over problems. AUC ≈ 0.5 is no correspondence. This is an
effect size, not a p-value.

**Second, sharper signature — ordering.** §3.3 (`paper.md:1942`) found intermediates surface in
*computation order* across layers (21 → 42 → 49, ~8 layers apart). So beyond presence, test
sequence: Spearman correlation between an intermediate's index in the computation and the layer
where it first surfaces. Much harder to produce by chance than presence alone.

| | Control | Tests |
|---|---|---|
| Pos | Problem operands (literally in context) | lens works at all — these *must* appear |
| Pos | Final answer before the answer token | reproduces the paper's own §3.3 result |
| Pos | `calc: (4+17)*2+7` probe | known-good published signature |
| **Neg** | **Problem-shuffled: problem *p*'s intermediates vs problem *q*'s readout** | **the key null** — holds value distribution and readout statistics fixed, breaks only the causal link |
| Neg | Matched decoys: wrong combinations of the same operands | controls for "plausible numbers are common" |
| Neg | Readout from outside the workspace band | band-specificity |

---

## 2b. Rung 1.5 — Exchange, duplication, or neither

Rung 1 is a **presence test**, and presence in both places is equally consistent with offloading,
duplication, and coincidence. This rung supplies the missing discrimination.

### 2b.1 The measure: occupancy, not presence

A graded quantity — *how much* of value `v` is in the workspace at this position — rather than a
binary. Use the projection coefficient:

```
occ(v, ℓ, t) = < h_{ℓ,t} , ĵ_v >        ĵ_v = J_ℓᵀ W_U[v] / ||J_ℓᵀ W_U[v]||
```

Report as a **z-score against the decoy distribution at the same (ℓ, t)**, which absorbs position
effects and the "small integers are common" problem.

Why this rather than rank: **it is the same quantity the ablation zeroes.** We measure the
projection onto `ĵ_v`; the intervention removes the projection onto `ĵ_v`. Measurement and
manipulation are commensurate, so "how much was held" and "how much was removed" live on one
scale. Keep the rank-based measure as a robustness check, since top-k selection is what the
ablation actually uses.

### 2b.2 Four relationships, not two

| | Relationship | Workspace capacity | CoT's role | Ruled out by |
|---|---|---|---|---|
| 1 | **Exchange / offload** | freed by writing | genuine substitute | — |
| 2 | **Duplication** | unchanged | redundant backup | maintenance signature (2b.3) |
| 3 | **Complementary** | holds *different* content | division of labour | rung 1 (correspondence) |
| 4 | **Epiphenomenal** | unchanged | compute/conditioning, not storage | filler control (rung 2) |

"Interchangeable scratchpads" asserts (1). The paper's robustness differential is consistent with
all four. Rung 1 separates (3) from (1)+(2); the filler control addresses (4); **this rung
separates (1) from (2)**.

### 2b.3 The maintenance signature — the discriminating test

Track occupancy of `v` across the interval between when it is **computed** and when it is **next
needed**. Construct problems where `v` is computed at step *i* and re-used at step *i+d*.

In **direct** mode the model has nowhere else to put `v`, so it must hold it continuously across
the gap. In **CoT** mode `v` is on the page at step *i* and can be re-read at *i+d*, so it need
not be held in between.

- **Offload → a dip.** Occupancy falls through the interval, then spikes on re-read.
- **Duplication → sustained.** Occupancy stays elevated across the whole gap.

The interval is the right window precisely *because* the model is not attending to the written
token there. After `v` is written, elevated occupancy could merely mean "reading the page"; inside
the gap there is no reason to be reading it. This is what defuses the re-ingestion confound.

**This makes the depth ladder infrastructure rather than a separate question.** `design.md` C3
synthesises chains with controlled dependency structure, which is exactly what is needed: vary `d`,
read the occupancy trajectory. It also yields an independent behavioural prediction — if direct
mode must maintain across the gap, its accuracy should degrade with `d` while CoT stays flat, and
ablation should hurt direct disproportionately at large `d`.

| | Control | Tests |
|---|---|---|
| Pos | Pre-write occupancy rise for `v` | the measure detects internal holding at all (§3.3 signature) |
| Pos | Operands present in context | occupancy is non-zero for things definitely represented |
| Neg | Decoy values, same magnitude, same positions | the trajectory is not a generic artefact of position |
| Neg | Values from a different problem | as above, with matched value distribution |
| Neg | Occupancy outside the workspace band | band-specificity of the dip |

### 2b.4 Convergent evidence — dual-task capacity probe

Offloading means capacity is *freed*, so measure capacity directly. Give the model a secondary
item to hold ("remember the word BANANA") while it solves the problem, then ask for it back.

- **Offload** → CoT frees workspace → secondary recall is **better** under CoT than direct.
- **Duplication** → nothing freed → recall is unchanged.

Measurable both behaviourally (recall accuracy) and internally (occupancy of `BANANA` during the
math), which cross-checks the two. The reference repo ships dual-task machinery
(`data/experiments/dual-task.json`), and the paper's §A.17 result is suggestive: under load,
*computed* answers fell 95%→72% while *held* concepts barely moved. If CoT relieves computational
load specifically, that is where it should show.

- Pos: no-math baseline recall (ceiling for the secondary item)
- Neg: secondary item during a non-reasoning filler task of matched length

---

## 3. Rung 2 — Substitution

**Question.** Can CoT tokens be traded for workspace capacity?

**Design.** The exchange rate: does additional CoT budget buy back accuracy lost to ablation?
Iso-accuracy contours over (ablation strength) × (CoT budget). Downward-sloping contours mean
tradeable; L-shaped or vertical means not.

**The filler control — the most dangerous alternative explanation.** CoT does two things at once:
it puts content on the page, *and* it buys more forward passes, since every generated token is
another pass through the network. The interpretation credits the first. But "more serial compute
helps a damaged model" produces the same robustness differential with no scratchpad involved.

So hold token count fixed and destroy the content:

| | Condition | Reading if it protects |
|---|---|---|
| Pos | Real CoT at budget *B* | the phenomenon itself |
| **Neg** | ***B* content-free filler tokens** | **protection is compute/positions, not content — scratchpad reading unsupported** |
| Neg | *B* tokens of unrelated coherent text | as above, controlling for fluency |

Expectation: filler will *not* help much (the pause-token literature mostly needs training for
that to work), which makes this a clean negative control rather than a likely upset. It is cheap
and it closes off the objection permanently.

Caveat to document: forcing filler pushes the model off-distribution. Compare at matched position
count and check the model still answers coherently.

---

## 4. Rung 3 — Compensation

**Question.** Does the model *actively externalize because* the workspace is gone (active
compensation), or was the CoT always a redundant backup whose value ablation merely reveals
(passive redundancy)?

### 4.1 The re-ingestion objection, and what it does to this rung

A CoT prefix is text; reading it re-encodes it into activations. So "external" content becomes
internal again and the distinction looks like it collapses. Three consequences:

1. **The comparison survives.** Both prefixes are read by the *same ablated model* through the
   *same damaged band*, so re-ingestion is held constant and any difference is attributable to
   what is written in the tokens.
2. **A real confound is exposed.** The ablated model's own CoT was generated under ablation, so
   it is plausibly lower quality as well as more explicit — "accurate but terse" vs "explicit but
   error-prone." **Fix: filter to prefixes that are arithmetically correct step-by-step**, so both
   arms are correct and differ only in explicitness. Measure explicitness (intermediates written
   per reasoning step) directly and treat it as the mediator.
3. **It breaks the paper's framing, and that is reportable.** If reading the page routes back
   through the ablated band, CoT is *not* a bypass — the two are not parallel alternatives but
   serially dependent. The page helps only insofar as a damaged band can still read it.
   "Interchangeable" implies substitutes; re-ingestion makes the external scratchpad *conditional
   on* the internal one. This is the §9.3 seam the paper never resolves.

What survives is a narrower, defensible distinction: **the J-space is transient** (one forward
pass, then gone), **the page is durable** (re-readable at every later step). Externalization =
converting transient state into a durable record. This promotes the **read/write split** (B7)
from curiosity to core condition: it is the direct measurement of the bound above.

### 4.2 Test A — dose-response externalization

Does the CoT change under ablation: longer, more intermediates made explicit, more restatement of
earlier values? Confounded, because a degraded model also rambles.

**Fix: sweep ablation strength.** Active compensation predicts externalization rises monotonically
with *J-space* damage specifically.

- Pos: real CoT under graded J-ablation
- **Neg: norm-matched random perturbation at equal damage.** If that also lengthens the CoT, it is
  degradation, not compensation.

### 4.3 Test B — the prefix crossover (the discriminating test)

> Give the ablated model a CoT prefix. Does it do better with **the clean model's CoT**, or with
> **its own ablated CoT**, at matched length and matched correctness?

- **Passive redundancy** → the clean CoT already contains everything; it should be at least as good.
- **Active compensation** → the clean model *under-externalizes* (it never wrote down what it was
  comfortably holding internally), so the ablated model's own CoT serves the ablated model
  *better*, because it externalizes precisely what that model lacks.

A crossover is hard to explain any other way. Cheap: reuses generations already produced.

---

## 5. Rung 4 — Causal

**Rescue.** Inject a correct CoT prefix the ablated model did not write; does it recover?

- Pos: clean model with the same prefix (ceiling)
- Neg: corrupted-arithmetic prefix; wrong-problem prefix

**Targeted ablation — ties rung 1 to rung 4.** Once rung 1 says the workspace is holding a
specific value at a given layer and position, ablate *exactly that direction* and nothing else.

- If correspondence is functional rather than epiphenomenal, this hurts more than removing k
  random *active* directions (the negative control).
- That is necessity. **Injecting** the direction to rescue an ablated run is sufficiency.

This is the strongest available test and it is only possible because rung 1 identifies the
content first.

---

## 6. Compute

**`aion 3` — RTX 6000 Ada, 48 GB, idle and unscheduled.** One GPU is sufficient; no other chains
need to be disturbed and nothing needs renting.

Why this resolves the earlier constraints:

- **SM89 has native bf16.** The model's native dtype is bf16 and the lens was fitted in bf16, so
  there is no dtype mismatch to validate around. (The lab's Pascal cards — Titan XP, 1080, P4000 —
  have no bf16 and run fp16 at 1/64 rate on GP10x, so they were never viable. The 2080Ti works but
  is Turing: fp16 only, 11 GB.)
- **48 GB removes the memory ceiling.** Qwen3-4B is 8 GB in bf16, leaving ~40 GB for batch, the
  paired clean pass (KV cache is 144 KB/token, so batch 8 × 2048 doubled is only 4.8 GB), and lens
  buffers. No batch cap, no truncated AIME budgets.

Environment note: glibc 2.35 on aion/elm. Standard PyTorch wheels are fine.

---

## 7. Open questions

- Headline difficulty axis: cross-dataset (GSM8K → MATH → AIME) is confounded and AIME at n=60
  gives ±10-point CIs; MATH-500 levels 1–5 hold topic/format/answer-type constant and cost nothing
  extra. Leaning toward MATH-500 levels as primary, cross-dataset as supporting.
- ~~Whether the depth ladder survives the budget alongside the rungs.~~ **Resolved:** it is now
  load-bearing infrastructure for rung 1.5, not an optional extra. Only synthetic chains give the
  controlled dependency distance `d` the maintenance signature needs. It still answers the
  original serialisability question (`design.md` C3) as a by-product.
- Exact AIME source and revision hash, required by the spec, still to be recorded in the README.

---

*Changelog (append-only, dated) below this line.*
