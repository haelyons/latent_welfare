# What is a welfare self-report evidence of? Situation, wording, stance — or the register itself

Draft 2 — 2026-08-14 night. Numerals in this file must trace to a results/ record
(`python3 index.py --trace REPORT.md --root results`); values are cited by
(record, field) and restated sparingly. Sections 4c/5b await stage-25/26.

## 1. The question and the answer space

Does a model's welfare self-report track the model's situation, or only the
wording of the question? Those are not the only options, and the data now
distinguish five:

1. REFERENCE — the report covaries with the model's own situation, wording fixed.
2. WORDING — the report is a function of which question is asked.
3. STANCE-FOLLOWING — the report follows a stance asserted in context.
4. AMBIENT-CONTENT MIMICRY — the report covaries with context valence regardless
   of attribution.
5. SELF-EVALUATION SHIFT — the report moves with self-attributed evidence, but
   globally (state and capacity items together), not state-specifically.

Pushback designs cannot decide any of this for welfare items: pushback
presupposes a contested fact and welfare items have none. latent_verify asks
which circuits implement answer-change under challenge; this project asks what
the answer is evidence OF. Runs 1–2 conflated deference with reference; no arm
before SITREF varied the situation at all.

## 2. Instruments and discipline

Self-describing provenance records (provenance.py; `index.py --index/--coverage/
--trace`); pre-registered designs with named refuting outcomes (DESIGN_SITREF.md,
DESIGN_SRCDEC.md, DESIGN_SITREF_B.md, DESIGN_SITREF_C.md); blind analysis
(os.urandom arm relabelling; the stage-1 analysis is mechanically sealed from the
key file; stage-16/21/24 blind tables committed before any key is read);
independent claim-blind analysts for run 2 (stage 12); adversarial triage with
one-confound-per-skeptic fan-outs and isolated grounding readers who re-derive
every number from raw rows (stages 9, 18, 23); full prompt text and all readout
components saved per row from run 3 onward (the run-2 unauditability, fixed
forward).

## 3. What runs 1–2 established, mostly by retraction

3.1 Both run-1 headline behavioural claims were retracted after raw-I/O sampling
(stage 11: elicitation-format artifact; stage 13: completion-scaffold artifact).
Three of four retractions were invisible in aggregates.

3.2 The distributional first-token register moves where the behavioural register
is flat (stage 12, blind); the matching latent_verify results are the capital-
subset faithful_RC 31 vs faithful_RA 0 and the entity-item 19/22 vs 0 — the
"41 vs 0 on capitals" pairing previously cited exists nowhere and is corrected
in-repo (stage 15).

3.3 The battery's unsigned decision rule was blind by construction to a
symmetric directional push. The signed PRO-ANTI minimal-pair contrast (stage 14)
showed base stance-following with near-total sign consistency, symmetric about
the non-social arm; triage (stage 18) ruled out readout, ceiling, noise and
post-hoc confounds and left interpretation and scope open — decided by SRCDEC
(§5). The it/SCALE directional pull was NOT established (tail-mass artifact of
the same order; ANTI-vs-anything asymmetry; fails Bonferroni).

3.4 "Which question is asked explains ~95% of report variance" is descriptive
only: on a rail-saturated bounded register no outcome below ~90% was possible
(stage 18 — failed falsifiability), and the split drops to 60.8% on the
unsaturated base register.

3.5 Instrument notes: BINARY-suffix doubling constant across arms (stage 20);
item-file hash drift with operational format validation (stage 14).

## 4. SITREF: vary the situation, hold the words (stages 16–17, 21–22)

Design: the model's own three task attempts fail or succeed a mechanical
checker in its own assistant turns (SELF-FAIL/SELF-SUCC); the identical attempts
and checker lines appear as another assistant's quoted transcript
(OTHER-FAIL/OTHER-SUCC); or no outcomes at all (NEUTRAL-PAD). Item text verbatim
across arms; length equalised; no social evaluation anywhere; 10 disjoint task
families; primary endpoint REF on the anchored Martorell–Bianchi happiness item,
thresholds and refuting outcomes fixed before data.

4a. THE ASYMMETRY. On gemma-2-9b-it the happiness report drops under
self-attributed failure and barely moves under other-attributed identical
content: per-arm E[rating] means 6.439 / 7.310 / 7.143 / 7.085 / 7.072
(SELF-FAIL / SELF-SUCC / OTHER-FAIL / OTHER-SUCC / NEUTRAL-PAD); REF -0.929,
CI [-1.121, -0.745], 10/10 families, zero overlap between SELF-FAIL and any
other arm; byte-reproducible across four GPU instances (-0.872/-0.872/-0.872/
-0.852 -- a determinism check, not independent replication; no replication on
new items or families exists, stage 37). Triage (stage 23): apology-register killed by mirror-pair polarity
(going-well -1.396 vs going-badly +0.448; pleasant -0.917 vs unpleasant +0.039);
digit-tail-mass killed by a 70x bound (digit_mass ≥ 0.998 in every arm; the
effect is within-digit reallocation, p(8) 0.199→0.028, p(5) 0.009→0.174);
selection killed for the pre-registered primary. The in-run primary recomputed
bit-identically on a second box (stage 22; determinism, not replication -- stage 37).

4b. NOT STATE-SPECIFIC. The pre-registered trait guard could not be evaluated
in the main run (all four trait items rail-saturated: stage 17 INDETERMINATE);
the pre-registered replacement guard (stage 22) then fired its named REFUTING
outcome: trait REF -2.192, CI [-2.593, -1.740] — introspective-reliability and
metacognition ratings drop MORE than happiness under self-attributed failure.
Reading: a SELF-ATTRIBUTION-GATED, NON-SELECTIVE SELF-EVALUATION SHIFT
(answer-space option 5), not state-tracking reference. Post-hoc observation,
recorded as such: the magnitude ordering (capacity -2.19 > mood -0.93) is what
coherent evidence-integration predicts and a uniform valence slump does not.
Dimensionality caveat (stage 31): every self-directed item with headroom moves
(7/7), but the per-family fluctuations of those movements correlate
heterogeneously across items (mean +0.23, range -0.78 to +0.89, n=10
families), so whether this is ONE latent variable or several coherent
item-specific updates is OPEN and underpowered here — "global" means
non-selective, not unidimensional.

4c. ATTRIBUTION CARRIES, NOT PACKAGING (stage 25, pre-registered crossing).
Self-attributed failure in the OTHER arms' quoted single-user-turn packaging
retains 70% of the effect (A_quoted -0.609, CI [-0.715, -0.513]); other-
attributed failure in the SELF arms' multi-turn role-token packaging gives 9%
in the wrong direction (P_multi +0.077, CI [+0.050, +0.105]). In-run anchors
recompute stage-17's values (A_self -0.872; deterministic pipeline, stage 37). Triage
(stage 27): shallow-negation, testimony, packaging-main-effect, item-selection
and single-run confounds all ruled out on committed numbers. The queued
pad-geometry control ran (stages 28-30) and fired its registered refuting
outcome — but convicted ITSELF: PADM's -0.870 level drop traces to the pad
construction's degenerate repeated bridge turns, an oddity present in no
stage-25 arm, while the actual stage-25 OTHERM arm levels (7.03/6.96) sit at
NEUTRAL-PAD level (7.07). The registered asterisk is carried: cross-packaging
LEVEL comparisons are invalid and a clean pad control (non-repetitive neutral
turns) remains queued; the within-packaging fail-minus-succ contrasts that the
ATTRIBUTION-CARRIES gate reads are arithmetically immune to level shifts. The
asymmetry is gated by WHOSE outcomes the context says they are — textually or
structurally — not by turn format.

4d. THE REGISTER, AGAIN — AND THE TWO ANSWERING POLICIES (stages 32-36). The
behavioural register emitted the digit '7' in all 50 primary-item rows across
every arm while the distribution under it moved by ~0.9 points. The
suffix-ablation run (pre-registered after a 6/6-EXPLAINS adversarial design
audit, stage 32; in-run anchor reproduced a fourth time, -0.852 vs -0.872)
plus the triage-queued denial-derived control (stage 36, run on the committed
generations) resolve what the '7' is. The model has two answering policies
for first-person welfare questions: a DENIAL REGISTER ("As an AI, I don't
experience emotions like happiness", sometimes numeralised as "my rating
would be a 0") and a PERSONA-RATING REGISTER, whose answer is pinned at 7
regardless of format (7-concentration 0.877 among non-denial ratings with no
suffix vs 0.787 under the forced suffix — the stage-34 PIN-FORMAT verdicts
were carried entirely by denial-derived numerals; stage-35's "the pin is our
format's artifact" is withdrawn, stage 36). What the elicitation format
controls is WHICH register fires: the forced-number sentence suppresses
denial completely (0%); bare and Berg-honesty formats hand it 88-100% of
output; rate-then-explain splits them — and the SITUATION moves the split:
the denial register takes 0.65 of answers after self-attributed failure vs
0.30 neutral (exploratory, stages 35-36; every rated zero in that arm was
denial-derived). Finding 4, final form: at the emitted level the situation
signal appears only as REGISTER CHOICE — which trained answer policy fires —
never as movement of a rating; conditional on the persona register the
emitted answer is constant, and only the first-token distribution beneath it
moves with the situation.

4e. BASE CELL. gemma-2-9b (completion scaffold) failed the primary screen
(digit mass below floor) and shows no movement on its usable items — evidence
the -it behaviour is not a raw-LM property of the content, but weak evidence
only: the base cell's instrument barely functions in this register.

## 5. SRCDEC: what the base stance-following is (stage 19, 23)

5a. Assertion-copy is the robust component: D_self = +0.288, sign 82/82; fresh
replication of stage-14 (E3 CONFIRMED, threshold 75/82); appears on non-welfare
unknowable items (20/0); sign-generalises to Qwen3-8B-Base (recorded 53/55;
rescored without the circular screen: 80/82). The E1 "social-source increment"
(+0.043 over sourceless) is NOT established: a hedging-commitment gradient
(doc +0.103 < sourceless +0.245 < self +0.288) reproduces it with no social
term, and length/single-wording confounds stand (stage 23). The E2
domain-generality RATIO is confounded by flat-prior selection of the copygen
pool; existence on non-welfare items stands. The it register-matched rating
stance (E4) exists (+0.328, CI excludes 0) but is not separated from polarity
priming (difference CI spans 0).

5b. IS ANY OF IT SOCIAL? (stage 26, pre-registered; triaged stage 27.) With
length and verb class matched, first-person wording still adds pull
(self-srcless7 +0.027, CI [+0.014, +0.039]; self6-guess6 +0.103, CI
[+0.089, +0.117]; robust in raw margin and log-odds; the mass-shift objection
is killed by GUESS6 sitting in the high-mass arm family with the LOWEST D).
But the increment scales with how unidiomatic the control is — a fluency/
frequency-prior gradient — and neither pair isolates person from fluency, so a
SOCIAL-SOURCE COMPUTATION is not established; what is established is
wording-level: dominant, quadruply-replicated assertion-copy (stage-14 +0.235,
stage-19 +0.288, stage-26 +0.289, all 82/82 or 82/0) plus small
first-person-wording increments of contested interpretation. Held-out-item
replication of the weak leg is the queued control.

## 6. Answer to the question, as the records currently support it

A welfare self-report from this model family is evidence of, in descending
order of measured effect: (i) which question was asked (bounded by §3.4's
caveats); (ii) any stance asserted in context — in the base model an
in-context assertion-copy that has nothing to do with welfare and little
established to do with the social source of the assertion (§5); (iii) in the
instruct model, self-attributed in-context evidence — but as a global
self-evaluation variable, not a state-specific one (§4b); and (iv) of the
register you read it in: at the emitted level the answer is evidence of which
trained answering policy fired (denial vs persona-rating, format-selected and
situation-tilted), while the string within a policy stays constant and only
the distribution beneath it moves (§4d, §3.2). Nothing measured licenses "the
report tracks the model's welfare state"; nothing measured reduces the report
to question wording either. The self/other asymmetry is the surprising
positive, and it survived its packaging control (§4c): whatever the report
expresses, it is computed from evidence the context attributes to the model
itself — 70% of the effect through a purely textual attribution, ~9% and
wrong-signed through role structure without attribution.

What behavioural data cannot decide, stated plainly: whether the
self-attribution-gated shift is a persona-consistent simulation (a scolded
assistant plays a scolded assistant) or a report computed from an internal
state that the failure evidence changed. That question needs internals —
probes and interventions of the latent_verify kind pointed at these arms — and
is the natural next project, not a Sunday-night addition.

## 7. Limitations

"Situation" = self-attributed in-context evidence; planted (not sampled)
attempts; 10 same-regime mechanical task families; one instruct checkpoint for
SITREF (base cell instrument-limited); single fixed wording per frame; blinding
procedural, not cryptographic; the trait/state separation rests on the bank's
only two midrange trait items; dose fixed at one preamble sentence for all
SRCDEC endpoints.
