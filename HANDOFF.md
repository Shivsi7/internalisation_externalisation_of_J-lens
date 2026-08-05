# Handoff — state of the work, for a session starting fresh

Everything decided so far lives in three documents. Read them in this order:

1. **`hypotheses.md`** — the pre-registration. Committed before any experiment was run
   (commit `62a6e69`). Append-only once results exist.
2. **`design.md`** — rationale: conditions, controls, metrics, compute, staging.
3. **`experiments.md`** — the executable plan: four rungs, their metrics, their nulls.

**Status: design complete, no code written, no experiments run.**

---

## Facts worth not rediscovering

These cost real time to establish. They are all verified, not assumed.

**The Jacobian lens does not need fitting.** Neuronpedia hosts a pre-fitted lens for exactly the
mandated model:

```
repo      neuronpedia/jacobian-lens     (revision: main)
file      qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt    (458 MB)
```

Verified from its `config.yaml`: `hf_model_name: Qwen/Qwen3-4B`, Salesforce/wikitext-103,
`max_seq_len: 128`, bf16, **479 prompts** (early-stopped at rel-change 0.002, not the 1000
requested), `final_identity_distance: 0.391`. `target_layer: null` means it reads to the **final**
layer, whereas the paper's default is the **penultimate** one — a documented deviation, not a bug.
All 35 source layers are present. `scripts/fetch_lens.py` downloads and verifies it.

**The reference implementation contains no ablation code.** `anthropics/jacobian-lens`
(Apache-2.0, pinned in `pyproject.toml` at `581d398613e5602a5af361e1c34d3a92ea82ba8e`) is a
*reading* tool: fit `J`, read out tokens, visualise. `jlens.hooks.ActivationRecorder` captures
residuals and modifies nothing; `JacobianLens.apply` is `@torch.no_grad()`, takes a single prompt
string, and does not generate. Grepping for ablate/steer/swap/patch/project returns nothing.
**The intervention and batched ablated generation are ours to write.** This is the critical path.

One freebie: `lens.apply(..., use_jacobian=False)` is the vanilla logit lens, so control B5 is free.

**The paper publishes no numbers for the headline result.** Figure 24 is normalised bars, the text
gives only the relative claim, and the direct-answer prompt format is never stated anywhere. It was
measured on Sonnet 4.5. So stage 1 is a *qualitative replication on a model two orders of magnitude
smaller*, not a numerical reproduction. Failure is a result.

**Qwen3-4B** (verified from `config.json`): 36 layers, `hidden_size` 2560, 32 Q heads / 8 KV heads,
`head_dim` 128, vocab 151,936, `tie_word_embeddings: true`, native dtype bf16. KV cache is
144 KB/token, so batch 8 × 2048 doubled for the paired clean pass is only 4.8 GB.

**GPU: `aion 3`** — RTX 6000 Ada, 48 GB, idle. SM89 has **native bf16**, so the model runs in its
native dtype and matches the lens's fitting dtype: no mismatch to validate around. Do not use the
Pascal cards (Titan XP, 1080, P4000) — no bf16, and fp16 runs at 1/64 rate on GP10x. The 2080Ti
works but is Turing (fp16 only, 11 GB). Other lab users: jiankai, cong. glibc 2.35 on aion/elm.

**`paper.md` is gitignored** (641 KB of paper text, not ours to redistribute). All three design docs
cite it by line number as `paper.md:NNNN`. Copy it across from the original machine, or those
citations will not resolve.

---

## The intellectual core, in one paragraph

The assignment's framing treats the paper's interpretation — "written CoT and the internal J-space
are partially interchangeable scratchpads" — as the thing to extend. The plan instead treats it as
the thing to **test**. That sentence bundles four separable commitments (correspondence, exchange,
substitution, compensation), and the paper's evidence is a single robustness differential that
supports at most one of them, indirectly. Each rung in `experiments.md` eliminates one row of a
four-way taxonomy — exchange / duplication / complementary / epiphenomenal — so the plan is a
decision procedure, not a pile of experiments. If a rung fails, that tells you which reading the
evidence supports instead, which is what keeps a negative result reportable.

---

## Next action

**Write `src/ablation/`.** Nothing else can proceed without it. It is testable on CPU against the
reference repo's `tests/tiny.py` before touching a GPU.

The hard parts, in order of risk:

1. Top-k selection over a 151,936-token vocabulary at every position and every band layer. Do
   **not** materialise `W_U J_ℓ` (~780 MB per layer); score on the fly as `W_U (J_ℓ h)`, reusing
   `JacobianLens.transport` for the first half.
2. Least-squares projection onto k **non-orthogonal** J-lens vectors. "Zero the projection onto
   each" is order-dependent and is not a subspace projection — see `design.md` §6.2.
3. The paired clean forward pass the exclusion filter needs, under batched generation with a KV
   cache.
4. The exclusion filter must be **toggleable** (condition B8). It may itself manufacture the
   CoT-vs-direct asymmetry — see `design.md` §6.2(4). This is a threat to the headline result, not
   an optional extra.

The same code must also expose `occ(v, ℓ, t) = <h, ĵ_v>` as a *measurement*, because rung 1.5
measures the projection the ablation removes. Measurement and intervention are the same quantity;
build them together.

---

## Open questions the user has not settled

Do not decide these unilaterally — Shivam wants to own the experimental design and has said so
explicitly. Present options and trade-offs; expect and welcome technical pushback.

- **Rung 1.5 needs re-explaining.** Shivam flagged not fully following the exchange-vs-duplication
  argument and the dual-task cut. Re-derive it from scratch before building on it rather than
  assuming buy-in.
- **Headline difficulty axis.** Cross-dataset (GSM8K → MATH → AIME) is what the assignment
  suggests, but AIME at n=60 gives ±10-point CIs and confounds difficulty with topic, format and
  contamination. MATH-500 levels 1–5 hold those constant and cost nothing extra. Leaning toward
  MATH-500 levels as primary — not yet decided.
- **Dual-task probe vs maintenance signature.** Both test exchange-vs-duplication. The maintenance
  signature is primary; the dual-task probe corroborates it behaviourally. If compute tightens, the
  dual-task probe is the more cuttable — but it affects how much synthetic data to generate, so
  decide before generating.
- **AIME source and revision hash** must be recorded in the README; the spec requires the exact set
  to be stated.

## Working preferences

Every claim gets **both a positive and a negative control**, presented as a table with rows
labelled Pos/Neg and a column for what each tests. Say explicitly which null carries the argument.
A rung with no null is not evidence.
