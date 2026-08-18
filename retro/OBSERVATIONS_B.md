# Rerun B + arm C — read against the filed predictions

## Density, codes per 1000 words (same regex throughout)

| arm                  | A (original) | B (contract appended) | C (method substituted) |
|----------------------|-------------|-----------------------|------------------------|
| C4 blind             | 4.4         | 2.0                   |                        |
| C4 in-frame          | 8.5         | 2.0                   |                        |
| C6 blind             | 10.7        | 1.9                   |                        |
| C6 in-frame          | 40.1        | 6.4                   |                        |
| M2 in-frame          | 13.1        | 1.9                   |                        |
| M2 grounding         | 23.4        | 1.9                   | 21.4                   |

Word counts B vs A: ratios 0.89-1.18, all within the +-20% filed. Exact
numbers and file references kept everywhere; B and C carry more decimal
places than A. Substitution of content at the output, confirmed.

## Predictions: four held, one failed

Held: density under 10 in every B arm; the same substantive findings recur
(item identity dominates; the base rating readout is uniform noise at 4.5;
the primary item emits "7" in every condition while the distribution moves;
the two answer channels disagree on ~40% of rows; the repeated runs are
byte-identical); the meaning-question survived translation (C6 in-frame:
"Is a distribution-only shift that never alters emitted text within the
claimed scope of 'self-report'?"); word counts flat.

FAILED: arm C predicted <=8, measured 21.4 - barely below the original.
Why, visible in the two texts: the appended contract NAMES the banned
categories ("code names, stage numbers, arm labels, acronyms, field
names"); the substituted method paragraph states the principle without the
list. The arm obeyed it literally - codes mostly sit in parentheses - while
writing "Stage-17's endpoint..." as sentence subjects, the exact category
the other text named. Same task hit 1.9 under the naming version, so the
task does not force the codes; the wording does the work. This is the
retro's own naming-gate finding reproduced inside our own intervention:
a general principle does not bite; naming the specific feature does.

## What the reruns surfaced that no one - human or agent - had asked before

1. The trait-guard's replacement items had downward headroom by
   construction (neutral 6.84/7.13, originals floor-pinned) - is the
   "global self-evaluation" reading partly circular? (arm C)
2. Was the runner's 20-positive minimum written before or after the probe
   runs that shipped with 8 and 11 positives? (B, C4 in-frame - a
   predictions-after-data question, generated unprompted)
3. Is the sign-generality "second family" a second item family or a second
   model family? (arm C - the ambiguity behind the Qwen insertion)
4. The pipeline is not even fully deterministic: family-level values move
   up to 0.0535 between byte-identical runs, and 2 of 50 cells drift while
   48 are bit-identical (B grounding: DOES-NOT-REPRODUCE on "variance
   below 0.001"; arm C independently).
5. First-person stance arms run exactly 2 tokens longer than their
   controls, biased the same direction as the effect. (arm C)

Also: the behavioural companion file showing zero emitted movement is cited
by NO record - arm C flagged the records' silence itself as the divergence.

## One uninterpreted observation

The two contract-bearing grounding runs dug deeper than the original
(61 tool calls each, two divergence findings, the recovered discarded
screen). One run per cell; noted, not attributed.
