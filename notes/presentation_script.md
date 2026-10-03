# Presentation script (15-20 min) — working draft

Style note to self: speak plainly, first person. Slides carry the figures,
this script carries what I say.

## Slide 1 — Title (0:30)

[FIGURE: none. Title, name, course.]

Hi everyone, I'm Shivam and I'm a visiting student here at Harvard from the UK. Boaz asked me to share some of the experiments I ran as part of my Homework 0 submission, which is what I'll be telling you a little bit about today.

## Slide 2 — The paper and the J-space (1:30)

[FIGURE: screenshot of the paper's two equations (J_l definition and lens(h_l) = softmax(W_U norm(J_l h_l))), from the paper website. Optionally a simple grid diagram: token positions across, layers up.]
Just as a very quick refresher, as I believe everyone here has already read the paper, this summer Anthropic published a paper called Verbalizable Representations Form
a Global Workspace in Language Models. They took this idea from neuroscience, called the 'Global
workspace'  is the idea that the brain has many specialised processors, and
cognition happens when information gets broadcast into a shared workspace that
they can all read. The paper's claim is that LLMs have something like this, and
it introduces a new interpretability technique to access it.

[show the two equations]

To do this, they introduced something they called the 'J-lens'. For each layer, they average the Jacobian, so how the
final layer's state moves when this layer's state moves, over many positions and
many prompts. Averaging washes out what the model is doing right now and keeps
its general disposition. You get a single matrix per layer that maps the state
here to what it will tend to become at the output. Push any internal state
through that matrix and the unembedding, and you get a readout over the
vocabulary. So for every token in the vocabulary, the Jacobian lens identifies a
direction in the middle of the network that encodes the potential for the model
to verbalise that token in the future, before the model has said anything. That
readable subspace is the J-space, the paper's candidate for the model's global
workspace.

## Slide 3 — The paper validated this causally (1:00)

[FIGURE: recreate the spider/ant example as a text mock on the slide: prompt + '8', then swap arrow spider->ant, + '6'. No repo figure; screenshot from the paper if allowed.]

And they did lots of cool things to show this isn't just a correlate. First,
they showed the J-space supports verbal report. They injected a concept's lens
direction at the token position just before an output, and the output changed
appropriately. Write into the workspace, and the model says what you wrote.
Second, they showed it mediates internal reasoning. Take the prompt "the number
of legs on the animal that spins webs is". The model says 8, because internally,
at the intermediate layers, the lens can read the intermediate concept spider
before any answer token exists. But if you swap the spider direction for the ant
direction at those layers, touching nothing else, the model actually says 6! So
the J-space is readable and writable, and the reasoning downstream follows what
is in the workspace.

## Slide 4 — Ablation and the headline result (1:30)

[FIGURE: the paper's Figure 24 screenshot for their result, or hold and reuse my replication figure on slide 6. Plus a small ablation schematic: grid with top-10 directions crossed out.]

Because the workspace is a set of directions, you can also damage it
surgically. At every position, find the ten directions the lens currently reads
most strongly, and delete them from the state. That is ablation, erasing
whatever the workspace is most clearly representing right now.

The paper's headline result is that this damage destroys silent, direct
answering on GSM8K, but leaves chain of thought answering basically untouched.
They interpret this as the page and the workspace being partially
interchangeable scratchpads. Externalise onto the page and you no longer need
the internal one.

The homework asked us to investigate this internal-external tradeoff and
whether it continues as problems get harder. And preparing that, I got stuck on
one word. What does externalisation actually mean, mechanically? That became my
main investigation.

## Slide 5 — Research questions and the four hypotheses (1:30)

[FIGURE: reports/figures/fig_hypotheses.png (the four cards, verdict-free, for predict-then-reveal). reports/figures/fig_verdicts.png is the same layout with the final verdict banners stamped in, for the synthesis slide or the reveal.]

To interrogate this, we ask two questions. First, and as our main
investigation, what is the relationship between the internal and external
scratchpads? Second, how much protection does the CoT provide as problems
harden?

For the first question, the authors suggest that CoT survives because the model
externalises onto the page, yet they do not investigate what this
externalisation means. We consider four mechanism hypotheses.

1. (exchange) externalisation actually removes the representation from the
   J-space, which is no longer needed to hold it.
2. (duplication) externalisation merely duplicates information that continues
   to be held in the J-space.
3. (complementary) the externalised and internal contents are complementary, a
   division of labour.
4. (epiphenomenal) the correlation is unrelated to storage altogether. Every
   generated token buys another forward pass, so the benefit could be extra
   serial computation rather than anything written down.

And my aim was that each hypothesis should get a test of necessity, a test of
sufficiency, and a direct observation where one applies.

For the second question, we hypothesise (H1) that the protection offered by CoT
is roughly constant across difficulty. We note in advance that the
cross-dataset comparison can only weakly test this, since GSM8K, MATH-500 and
AIME differ in topic, format and contamination as well as difficulty.

Two metrics to keep in mind. Retention is ablated accuracy divided by clean
accuracy, so 1.0 means ablation cost nothing. And occupancy is how strongly the
lens reads a given token at a given position.

## Slide 6 — Replication with extra controls (1:30)

[FIGURE: reports/figures/fig_gsm8k_retention.png (grouped by arm: CoT cluster flat across all six conditions, direct cluster dips only under the true ablations; tint = clean, solid = ablation, gray = the two wrong-direction controls). For the difficulty axis, reports/figures/fig_datasets_accuracy.png (Table 1 as bars, same tint/solid encoding, cells verified against the table's seed-0 runs).]

The first thing I wanted to do was to replicate that headline result, but add
some controls that I thought the paper lacked and that would be helpful in
increasing our confidence in interpreting the results. I add three.

1. A norm-matched random control, so delete an equally large component in
   random directions, to check the damage is not just any big perturbation.
2. A rank-shifted control, so delete the 51st to 60th strongest lens directions
   instead of the top ten, to check it is really about the currently most
   active directions.
3. The paper's method quietly includes a filter that protects the model's
   imminent top predictions from ablation, and in direct answering the imminent
   token is the answer. So I made it toggleable, to check the contrast is not
   an artifact of the filter.

As you can see, the replication holds. Direct answering drops from 21 to 13
percent while chain of thought is untouched, and the two wrong-direction
controls in the figure both come back clean. The filter check is not in the
figure, but with the filter off entirely the numbers barely move, 96 and 14
percent, so the contrast is not the filter's doing either. The damage genuinely
requires the top of the J-space.

## Slide 7 — Hypothesis 4, epiphenomenal (1:30)

[FIGURE: reports/figures/fig_filler_battery.png (full set, 3 fillers, per-seed dots). Backup: reports/figures/fig_prefix_rescue.png (clean vs ablated clusters, matched subset).]

I started with the hypothesis that is cheapest to test. If chain of thought
helps just because generating tokens buys extra forward passes, then any tokens
should help. So I pre-wrote the model's scratchpad before letting it answer,
and compared four versions of the handed-in page. Its own correct reasoning.
Meaningless filler of exactly the same length, and I tried three kinds, a
repeated sentence, the model's own reasoning tokens shuffled, and just dots.
Nothing. Or another problem's reasoning.

The result is very clean. Own reasoning takes the ablated model to 100 percent.
Every filler type performs no better than a blank page, across three seeds. And
another problem's reasoning drops it to zero, the model trusts whatever page it
is given. So the benefit is the content, not the token count. Epiphenomenal is
rejected.

## Slide 8 — Hypothesis 3, complementary (1:45)

[FIGURE: reports/figures/fig_numeric_ordering.png (values appear in computation order, pooled 3 seeds), then reports/figures/fig_cot_correspondence.png (write / fade / return in the model's own transcript, three seeds). NEW OPTION: reports/figures/fig_correspondence_contrast.png supersedes the second figure if you want the stronger claim, same three windows in both arms on budget-matched programs, the fade is not caused by writing and the write peak is emission-locked.]

Next, does the workspace hold a different kind of content than the page? If
the two stores divide the work by content type, then page-type values should
not show up in the workspace at all.

I tested this in two steps. First with the page removed entirely. I gave the
model small programs, like a = 9 - 5, b = 2 + 1, c = 8 - 6, d = a - c, and made
it answer silently with no writing allowed, and just watched with the lens.
The direct arm is the clean one for this measurement, there is no written copy
in the context, so anything the lens sees must be internal rather than an echo
of the text.

What you see is that the intermediate values, 4 then 3 then 2, appear inside
the model one after another, in exactly the order a written solution would
produce them. So the workspace demonstrably handles the very content a chain of
thought writes.

On its own that only shows capability. The model might still split the work by
content when it actually writes. So the second step ran the same measurement on
the model's own chain of thought. As each value is written it also occupies the
workspace, it fades while later lines are written, and it comes back where it
is used. Same values, both stores, in both modes. The by-content version of
complementary is rejected.

And when I later looked at what the workspace does hold while values sit on
the page, about 99 percent of the top readouts were not program content of any
kind, so there is no hidden parallel program in there either. What separates
the two stores is not what they hold but how long they hold it, and that is
exactly where the next experiment goes.

## Slide 9 — The memory test (1:30)

[FIGURE: reports/figures/fig_masking_schematic.png (the three-box design, four rows, row colours match the result figure's bar colours). Result figure: reports/figures/fig_model_probes.png (masked vs severed vs page, three models, doubles as the generality preview).]

Now the storage question, which is where exchange and duplication live. Both
assume the workspace holds values over time. So I built a direct memory test.

The model reads a definition, x = 7. Then about thirty tokens of completely
unrelated text. Then the question, what is the value of x? And the trick is
attention masking, which is an established interpretability tool, I am
repurposing attention knockout from Geva et al. 2023. At the question, the
model is forbidden from looking back at x = 7. During the middle text it could
still look freely. So if the model carried the 7 forward internally at any
point, the mask costs it nothing. If its habit is to look the answer up at the
last moment, it now cannot.

The result. With the mask, accuracy crashes to pure guessing, 13.5 percent
against a 12.5 percent floor. Telling the model in advance to memorise the
value because the definition will disappear changes nothing. A written copy in
the middle text restores 100 percent. And this replicates at every gap length I
tried, with words instead of digits, and in three different models from two
families. The model carries nothing. It re-reads.

## Slide 10 — The twist (1:30)

[FIGURE: the model's own transcript with written values crossed out, plus a big '76%?'. Build on-slide; no repo figure.]

But then I ran a version that seemed to contradict this. I took the model's own
finished reasoning, and at the final line I hid every written copy of the
intermediate values from it. If the model carries nothing internally, hiding
the page copies should hurt a lot. Yet the model still answered correctly 76
percent of the time.

So which is it? Either some internal memory exists after all, or something else
is going on with the page.

## Slide 11 — The knockout curve (1:30)

[FIGURE: reports/figures/fig_knockout.png (the centerpiece curve).]

To decide, I widened the hiding step by step. Just the written values. The
values plus three neighbouring words. Plus six. Then the entire written
reasoning, with only the original question left visible. And as a control, hide
the same number of randomly chosen words instead.

Accuracy slides down smoothly, 100, 71, 64, 56, and with the whole page hidden
it collapses to 10 percent, which is the re-derivation floor. Random hiding
costs nothing at all. So the 76 percent was not internal memory. As the model
writes, each value soaks into the neighbouring words of its own page. Hide the
whole page, and there is nothing left inside. The information lives in the
written context and nowhere else. Duplication is rejected.

## Slide 12 — The injection test (1:30)

[FIGURE: reports/figures/fig_injection.png (floor vs three injection doses, wrong-digit annotation). Callback icon to slide 3's injection demo.]

One question remained for me. We know the model never stores a value
internally. But could it, if one were there? Remember the paper's own injection
trick from slide 3. I used exactly that validated write-channel, and planted
the value's lens direction into the workspace during the memory gap myself.

With the right strength, accuracy jumps from the 12.5 percent floor to 47
percent. And when I deliberately plant the wrong digit, the model reports the
planted digit half the time. So the internal memory channel works, and it
carries real value identity. The model could remember. It just never does. It
prefers to re-read its page. So duplication and exchange do not fail because
the workspace cannot store, they fail because the model never uses it as
storage.

## Slide 13 — The difficulty question, H1 (1:00)

[FIGURE: reports/figures/fig_seed_rho.png (per-seed protection ratios, both datasets).]

Briefly, the second research question. I repeated the headline comparison at
five independent seeds on GSM8K and MATH-500. The protection ratio sits above
1.0 at every single seed, and CoT retention never measurably drops. The falling
average ratio between datasets comes from the direct arm recovering, not from
CoT protection weakening. On AIME the clean direct score is at zero, so by our
own floor rule that cell cannot inform the comparison, and it stays that way at
a second seed. So within what we can test, H1 stands.

## Slide 14 — Generality, and being honest (1:30)

[FIGURE: reports/figures/fig_scale_retention.png (four model sizes). Llama caveat and honesty items as text; no figure needed.]

The storage picture replicates on Qwen3-14B and on Llama-3.1-8B, a different
family with a different tokenizer. One honest caveat. On Llama, the ablation
recipe itself breaks the model into gibberish before doing anything selective,
which is actually the same thing the paper reports for Haiku 4.5. So for Llama
we only trust the masking experiments, which use no ablation, and we say so.

I also want to be upfront about how this was run. Every experiment had its
predictions registered before running, and one of mine was falsified, I
predicted an internal step budget of 3 to 7 operations and measured 2. And two
of my own measurement artifacts were caught and logged along the way, including
one my figures inherited from an answer parser misreading chatty outputs.

## Slide 15 — The picture, and what is next (1:00)

[FIGURE: reports/figures/fig_verdicts.png (the four cards with final verdict banners, same layout as slide 5's fig_hypotheses so the verdicts read as the reveal). Or draw the synthesis diagram: workspace as a small 2-op buffer, page as the store, arrows for write/re-read. Optional inset: reports/figures/fig_depth_ladder.png (D=2).]

So here is the picture I ended with, and it fits on one line. The internal
scratchpad is an input output buffer for simple intermediate reasoning, not a
store of it. It computes about two operations deep. It holds values only while
producing or consuming them. The page is the only durable memory, and writing
does not move or copy a representation, it creates the only lasting one. And
the model has a working memory channel that provably works, injection drives
it, yet it never uses it.

I read this as sharpening the paper rather than contradicting it. The J-space
really is a global workspace in the broadcast sense, whatever is in use is
readable and steerable there. It is just not a memory. And for a safety course
there is one consequence worth saying out loud. If the model cannot carry
values silently across positions, then what it needs to remember has to pass
through the transcript we can read.

If it really is a short-term buffer, it should behave like one, and that is
testable in two ways. First, a buffer has a decay curve. Instead of my three
coarse windows, measure the value's occupancy token by token after the moment
it is computed, in both arms. A buffer fades within a few tokens regardless of
whether a page exists, a store plateaus until the page appears. I have
registered the first half of this already, the same correspondence windows on
the direct arm, where there is no page at all. Second, a buffer has eviction.
Working memory is displaced by new content, not just by time. So compute a
value, then force an unrelated computation through, and compare the value's
trace against a time-matched idle gap. If new work pushes the old value out
faster than time alone, that is buffer behaviour, capacity being reclaimed.

And there is a way to give exchange a real test after all. Exchange dissolved
because the workspace never holds anything, so writing never has anything to
free. But injection can create the missing precondition. Plant a value in the
workspace, let the model write it onto the page, and compare the planted
copy's occupancy after the write against a no-write control at the same
distance. If the act of writing evicts the planted copy, exchange is a real
mechanism the model just never gets to use on its own. If the copy persists,
writing frees nothing even when something is there to free.

And the channel question stays on my list. Injection shows the hardware
works, so could a model be trained to actually use its internal memory, and
what would that do to how much of its reasoning still reaches the page.
Thank you.


## Backup slides (for questions)

[FIGURE: reports/figures/fig_task_balance.png — ablation costs nothing on lookup tasks]
[FIGURE: reports/figures/fig_depth_ladder.png — the D=2 serial budget]
[FIGURE: reports/figures/fig_prefix_rescue.png — clean vs ablated, matched subset]
[FIGURE: reports/figures/fig_datasets_retention.png — retention across datasets]
[TABLE: the verdict scoreboard from extended_report.pdf — hypothesis / instrument / statistic / verdict]
