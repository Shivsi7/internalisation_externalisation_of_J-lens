# Q&A preparation

Each answer is written to be spoken. Numbers are the ones in the figures and
the extended report. The hardest questions here are ones I asked myself while
building the deck, so expect them.

## Validity of the core result

**Q. Isn't "no storage across tokens" trivial? Transformers recompute
everything each token.**
Nothing resets. Every processed position's keys and values persist for the
whole context, and a two-hop relay through gap positions (definition to gap
column at an early layer, gap column to question at a later layer) is an
ordinary attention pattern. My mask leaves the gap columns fully readable, so
a parked copy anywhere in the gap would survive it. And injection proves the
channel functions, planting the value in the gap rescues to 47 percent, so
the absence is learned behaviour, not architecture.

**Q. Maybe your attention mask just breaks the model.**
The written-copy condition uses the identical mask and scores 100 percent, in
all three models. Random-position masks of matched size in the knockout
experiment also stay at 100 percent. The mask only hurts when it takes away
the one thing being tested.

**Q. Could the value be stored in a form your lens can't see?**
For the observational probes, yes, and I say so, the lens has modest
sensitivity (AUC 0.67). But the masked-maintenance test does not use the lens
at all. It blocks information flow and reads accuracy. Any stored copy, in
any encoding the model can use, would have rescued performance. It did not.

**Q. On Llama the masked vs severed sign test is p=0.06. Isn't that holding?**
Direction matters, masked sits about five points above severed, and masked is
the condition where gap positions can still read the definition. So the edge,
if real, is a weak relay through the gap, context absorption, not workspace
storage. Against that five point whisper, the collapse is 82 points from
baseline. And a p above 0.05 was never my claim, the claim is the effect
size, masked lands next to the floor, not next to baseline.

## Statistics

**Q. Why McNemar and sign tests instead of a t-test or Mann-Whitney?**
The data are the same problems under two conditions, paired, with a binary
outcome each. Rank tests and t-tests assume independent groups and numeric
values. The exact sign test is the right tool and it is simple, discard
agreements, and ask whether the disagreements could be fair coin flips. Own
reasoning versus the empty scratchpad disagreed on 403 problem-instances and
won all 403.

**Q. You show several pairwise p values. Multiple comparisons?**
The comparisons were registered before the runs, a Cochran's Q omnibus across
all six battery conditions rejects globally at p below 1e-100, and the
pairwise values survive any correction, a Bonferroni factor of five does not
dent an exponent of minus 124. The one data-selected comparison, best filler
versus empty, is biased toward finding an effect and still comes back null,
which is conservative for my claim.

**Q. "Best filler vs empty p=0.47" accepts the null. That's not equivalence.**
Correct, and I don't rest on it. The claim rests on the token-matched
contrast, own reasoning beats every filler on every single disagreeing
problem, 415 to 0, 411 to 0, 429 to 0. Whatever tiny effect a filler might
have, it is bounded far below anything that matters.

**Q. Most cells are one run at one seed.**
The headline contrast is five seeds per dataset, the battery is three seed
runs, the deterministic probes replicate across gap lengths, value types, and
three models from two families. Where a cell is single-seed, the report says
so, and the two anomalies I found are logged, not hidden.

## Interpretation

**Q. The paper says the workspace holds information. Are you contradicting it?**
Sharpening it. The J-space holds information in use, during computation, and
it is readable and steerable there, I replicate that. What I add is that it
holds nothing across a gap between uses. A global workspace in the broadcast
sense, not a memory. The paper never tested the between-use case.

**Q. Doesn't the null just mean the model had no reason to hold the value?**
Partly, and the capacity variant addresses it, telling the model in advance
that the definition will disappear and it must memorise changes nothing. So
no incentive I could construct induces holding. I claim no default holding
and no induceable holding under my conditions, not impossibility, and
injection shows the capacity is there.

**Q. Why does the model never use a channel that works?**
Labeled speculation. Training never demands it. During pretraining and RL the
context is always present, so re-reading is always available and cheaper than
maintaining a representation against thousands of intervening updates.
There is no gradient pressure to build a habit of internal carrying. That is
also why I find the trainability question interesting as future work.

**Q. Why does the injection overdose fail? 47 percent at scale 0.3 but 5 at 1.0.**
The injection adds the value's direction scaled by the local residual norm.
At full norm it dominates the residual stream and pushes the state off
distribution, so the model degrades generically. The inverted U is what you
expect from writing into a live representation, there is a dose that speaks
and a dose that shouts everything else down.

**Q. Is the injection rescue just generic steering toward digits?**
The wrong-digit control answers that. Plant a different digit and the model
reports that digit half the time. The effect is content-specific transport
through the workspace, not a generic accuracy boost.

**Q. The 14B model shows the biggest direct damage but 8B shows none. What's
the scale story?**
There isn't one, and I say so. Each model is ablated through its own
pre-fitted lens, so susceptibility differences confound lens quality with
scale, the 8B zero is most plausibly a lens that aims less well. The
direction-specificity controls that would decide it exist only at 4B. The
only scale claim I make is the lens-invariant one, chain of thought is
protected everywhere.

**Q. Clean model scoring below the ablated one in the battery's no-prefix
cell. Explain.**
A diagnosed measurement artifact. The healthy model declines the empty-think
format and keeps reasoning in prose, and the extractor scores the first
number in that prose, usually an intermediate. So that cell measures format
compliance, not ability, it is excluded as a baseline, and the within-format
filler comparisons are unaffected.

## Scope

**Q. Everything is single-digit arithmetic. Does this generalise?**
The maintenance result replicates with word values, across gap lengths, and
in three models from two families. But the probes do read literal
single-token values through a lens of modest sensitivity, so the honest scope
is, no holding of simple verbalizable values under any incentive I built.
Richer content is exactly where I would probe next.

**Q. GSM8K, MATH-500 and AIME differ in more than difficulty.**
Yes, topic, format and contamination all move together, which is why I
registered in advance that the cross-dataset comparison only weakly tests the
difficulty hypothesis. Within what it can test, protection replicates at
every seed.

**Q. AIME CoT retention is 0.79. Is protection failing on hard problems?**
Possibly, and it is the one cell that could say so, but the AIME CoT cells
are budget-limited, 59 of 60 generations hit the token ceiling, so the dip is
confounded with truncation. It repeats at a second seed, so it is stable
enough to flag, and not clean enough to interpret.

**Q. On Llama your ablation produces gibberish. Doesn't that undermine the
cross-family claim?**
It bounds it. For Llama the ablation recipe is not selective, which the paper
also reports for one of their models, so on that family I only claim the
masking results, which use no ablation. Cross-family storage claims, not
cross-family protection claims.

## Safety framing

**Q. Why should a safety audience care?**
Two sides of one coin. Today, if the model cannot carry values silently
across positions, what it needs to remember must pass through the transcript
we can read, which is good news for chain of thought monitoring. But
injection shows the silent channel physically works. Whether training can
recruit it, and what that would cost transcript faithfulness, seems worth
knowing before models are optimised against monitors.

**Q. What would change your mind about the buffer picture?**
A condition where the value's occupancy persists across a gap without the
page, or a masked-maintenance cell that stays high with the definition
unreadable, or the direct-arm correspondence coming back with a plateau
instead of a fade. All three are concrete and one is already registered.
