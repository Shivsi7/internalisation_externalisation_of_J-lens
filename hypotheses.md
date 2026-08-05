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
