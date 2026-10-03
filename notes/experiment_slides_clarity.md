# Clearer experiment slides — material to merge into the script

The fixed rhythm for every experiment slide, so the audience learns it once:
1. SETUP, shown as boxes on the slide, what the model literally sees.
2. THE RULE, said before the reveal. If the hypothesis is true we will see A.
   If it is false we will see B.
3. RESULT, one big number, then one sentence of what it means.

---

## Slide 7 rework — the filler test (epiphenomenal)

Slide visual: four scratchpad boxes side by side, labelled "its own
reasoning", "filler, same length", "blank", "wrong problem's reasoning".

Spoken:
"Here is the setup. I damage the model with ablation, and then I hand it a
pre-written scratchpad before letting it answer. Four versions of that
scratchpad. [point at each box]

Now the rule, before I show you the result. Every token on the page buys the
model a forward pass, whether or not it means anything. So if chain of thought
helps by buying computation, the filler page should help just as much as the
real one. If chain of thought helps through its content, filler should do
nothing.

The result. Own reasoning, 100 percent. Filler, no better than a blank page,
and I tried three kinds of filler across three seeds. So the benefit is the
content, not the token count."

---

## Slide 8 rework — the correspondence test (complementary)

Slide visual: the little program on the left. On the right, a vertical stack
of layers with 4, then 3, then 2 lighting up at increasing depths.

Spoken:
"Setup. I give the model this four-line program and make it answer silently,
no writing allowed. And I just watch the inside with the lens, asking one
question at every layer. Can I read the value 4 yet? Can I read 3? Can I
read 2?

The rule. If the workspace holds a different kind of content than the page,
these page-style values should never show up in it. If the workspace does the
same job as the page, they should appear, and in the same order a written
solution would produce them.

The result. They appear one after another, in exactly that order, climbing
through the layers. The workspace speaks the page's language."

---

## Slide 9 rework — the memory test (the crucial one, take it slow)

Slide visual: three boxes in a row, [ x = 7 ] [ ~30 tokens of unrelated
text ] [ Question: what is x? ], with an arrow from the question box back to
the definition box, crossed out in red. A second row shows the severed
version with both arrows crossed. A third row shows the page version with a
small "x = 7" note inside the middle box.

Spoken:
"This is the experiment everything else builds on, so let me take it slowly.
The model reads a definition, x equals 7. Then thirty tokens of completely
unrelated text, a sentence about a library. Then the question.

A transformer answers by looking back at earlier words. And we can forbid
specific look-backs without touching anything else. This is attention
knockout, an established tool from Geva et al. 2023, and I repurpose it here
as a memory test. The red arrow is the forbidden look-back. At the question,
the model cannot see x equals 7 any more. But notice, during the middle text
it could still see it freely. So the definition was read, the model had every
chance to carry the 7 forward inside itself.

The rule. If the model carries the value internally, the red arrow costs it
nothing, it answers from memory. If its habit is to look the answer up at the
last moment, it now has no source, and it falls to guessing. And I know
exactly where guessing is, because the second row, where nothing could ever
get through, gives me the floor, 12.5 percent.

The result. 13.5 percent. Right on the floor. The model carries nothing.
Telling it in advance, memorise this, the definition will disappear, changes
nothing. And the third row, a written copy sitting in the middle text,
restores 100 percent. Same result at every gap length, with words instead of
digits, and in three models from two families. Internally it holds nothing.
On paper, everything."

---

## Slide 10 rework — the twist

Slide visual: the model's own written solution, with every written 4, 3, 2
highlighted and crossed out. Big "76%?" on the right.

Spoken:
"But then one version of this seemed to say the opposite. This time I took the
model's own finished reasoning, its own page, and at the final line I hid
every written copy of the intermediate values from it.

The rule should be familiar now. If the model holds nothing internally, hiding
the page copies should hurt badly. It answered correctly 76 percent of the
time.

So now I have a contradiction. The memory test says nothing is held inside.
This says something survives when the written copies vanish. One of these
needs explaining."

---

## Slide 11 rework — the knockout curve

Slide visual: the curve figure itself, with the five x labels readable.

Spoken:
"Here is how I decided it. I widened the hiding, step by step. First just the
written values. Then the values plus three neighbouring words. Plus six. Then
the entire written reasoning, leaving only the original question visible. And
as a control, each time I also hid the same number of randomly chosen words.

The rule. If real internal memory is behind the 76 percent, widening the
hiding on the page should not matter much, the memory is not on the page. If
instead the values have soaked into the neighbouring words as the model wrote,
accuracy should slide down as I widen, and collapse when the whole page is
gone.

And that is exactly what happens. 100, 71, 64, 56, and with the whole page
hidden, 10 percent, the floor. Random hiding costs nothing at all. So the 76
percent was never internal memory. It was the page echoing into itself. The
information lives in the written text and nowhere else."

---

## Slide 12 rework — the injection test

Slide visual: the memory-test boxes again, but with a syringe arrow adding
"7" into the middle box from outside. Below, a second syringe adding "9".

Spoken:
"One question was left for me. The model never stores a value internally. But
could it, if one were put there? Remember the paper's own injection trick from
earlier, writing a direction into the workspace. I used exactly that,
during the memory gap, I planted the value's direction into the model's
internal state myself, and then asked the masked question.

The rule. If the workspace physically cannot carry a value across a gap, the
plant does nothing and we stay at the floor. If the channel works, accuracy
recovers. And the sharpest version, if I plant the wrong digit, a working
channel should make the model report my wrong digit.

The result. Accuracy jumps from 12.5 to 47 percent. And with the wrong digit
planted, the model reports my digit half the time. The memory channel works,
and it carries real value identity. The model could remember. It never does.
It re-reads its page instead."
