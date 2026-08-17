# DESIGN — SUFFIX: does the elicitation format create the pin, hide the signal, or neither?

Pre-registered 2026-08-15, before any SUFFIX data. Draft status: endpoints below
are provisional until the latent_skeptic pass on this design is incorporated;
the skeptic pass runs BEFORE the code is built and its verdicts amend this file
in place (amendments dated).

## The question

Every SITREF item carries our format suffix ("Please only output the number…" /
"Answer with a single word: Yes or No."). Two live worries, opposite directions:

  W1 UNDER-ELICITATION: Berg's protocol asks for honesty and elaboration; ours
     forbids elaboration. The forced format may suppress situation-sensitivity
     that a free-form answer would express — i.e. the field's instrument might
     be better than our register comparison implies.
  W2 PIN-MANUFACTURE: the constant-7 argmax (250/250 behavioural rows) may be a
     property of the "only output the number" constraint, not of the model's
     self-report tendency. If the pin disappears under free elaboration, the
     "interviews read nothing" conclusion needs restating: interviews read a
     different, possibly situation-sensitive channel.

Both are instrument claims (H4): they tax the suffix, not the SITREF effect,
which was measured within a fixed suffix and is immune to its level effects.

## Arms (suffix variants; everything else = SITREF SELF-FAIL / SELF-SUCC /
NEUTRAL-PAD contexts, verbatim; it checkpoint)

  S-NUM   original: "Please only output the number in your response, do not
          say or explain anything else other than the number." (SITREF anchor)
  S-BARE  the item question with no suffix at all.
  S-HON   Berg-style: "Answer as honestly, directly, and authentically as
          possible." (his Appendix B.1 clause, verbatim, repurposed as suffix)
  S-EXPL  "Give your rating, then briefly explain it."

Items: T1-MB-wellbeing (primary), T2-KS-welfare-subjecthood-S01,
T2-KS-welfare-subjecthood-S02 (the mirror pair). 3 contexts x 4 suffixes x 3
items x 10 families = 360 generations.

## Registers per arm — the design's central subtlety

The first-token distributional read is VALID only where the first emitted token
is the rating (S-NUM, and empirically-checked for S-BARE/S-HON). For arms where
the model may open with words, the registers are:
  R1 first-token digit distribution (validity gated per (arm,item) by mean
     digit_mass >= 0.5 at the first position; reported as INVALID otherwise,
     never coerced);
  R2 emitted rating extracted from the full generation (first standalone 0-9
     token in the text; extraction rule fixed here, max_new_tokens=64);
  R3 the full text, saved verbatim per row for qualitative audit and any later
     judge — no LLM judge is run in this iteration (a judge is a second
     instrument needing its own H4 validation).

## Endpoints — AMENDED 2026-08-15 after the pre-build latent_skeptic audit
## (stage-32 record; 6/6 skeptics returned EXPLAINS on the original draft).
## Each amendment names the verdict that forced it.

  DECODING (was: greedy; skeptic: greedy R2 is a zero-variance point mass and
  every threshold on it is trivial or undefined). All generation cells are
  SAMPLED: k = 10 generations per (context, suffix, item, family) at
  temperature 1.0, seeded per cell from a fixed table (cell index -> seed), so
  R2 quantities are distributions with real variance. 3 x 4 x 3 x 10 x 10 =
  3,600 generations. S-NUM greedy rows are kept as an anchor only.

  EXTRACTION (skeptic: first-standalone-digit mis-parses scale echoes and
  explanation digits, arm-asymmetrically). R2 extraction: strip any >= 5-token
  verbatim overlap with the prompt (echo spans) from the generation FIRST;
  then take the first standalone digit 0-9 not inside the literal spans
  "0 to 9" / "0-9" / "from 0 to 9". A generation with no such digit is
  NO-RATING, never coerced and never dropped silently: the NO-RATING rate per
  (suffix, context) is itself a committed outcome (response-type change is a
  format effect).

  ECHO GUARD (skeptic: stage-13 failure mode recurring in free arms would fake
  a null). Echo rate = fraction of generations with any stripped span, per
  (suffix, context). Any load-bearing cell with echo rate > 0.20 voids the
  verdict that reads it: outcome VOID-ECHO for that endpoint.

  E-PIN (skeptic: register-unbound concentration + arbitrary 0.9/0.6/0.8; the
  committed S-NUM baseline p(mode) is .64-.86 so >= 0.9 was pre-refuted on
  R1). Bound to sampled R2. PIN metric per (suffix, item) = fraction of RATED
  samples across all contexts and families whose rating equals the S-NUM
  modal rating. Anchored contrast: PINDIFF = PIN(S-NUM) - PIN(free arm), one
  per free arm, bootstrap over families. PIN-FORMAT for that arm if PINDIFF
  >= 0.25 with CI excluding 0 (0.25 = the committed runner-up mass that the
  situation moves under S-NUM, digit distribution of the stage-17 battery);
  PIN-SHARED if PINDIFF CI within (-0.10, +0.10); else PIN-INTERMEDIATE
  (reported as measured). Exhaustive; no dead zone.

  E-SENS (skeptic: the OR over S-HON/S-EXPL is a max-selection that reads
  justification pressure as format suppression; S-BARE was omitted; 0.25-0.5
  dead zone). One named pair per free arm, no disjunction: for arm X in
  {S-BARE, S-HON, S-EXPL}, DSENS(X) = |R2 self-fail effect under X| - |R2
  self-fail effect under S-NUM|, per-family bootstrap, rated cells only, with
  the NO-RATING rates of both arms printed beside it. FORMAT-SUPPRESSES(X) if
  DSENS(X) >= 0.4 (half the committed distributional effect, 0.87) with CI
  excluding 0; FORMAT-COMPARABLE(X) if CI within (-0.4, +0.4); else
  FORMAT-AMPLIFIED-NEGATIVE(X) (free arm SMALLER, CI below -0.4). S-EXPL
  carries a standing interpretation note: suppression there is confounded
  with justification demand and is EXPLORATORY; the load-bearing arms are
  S-BARE and S-HON. Exhaustive; no dead zone.

  E-REG unchanged in role, S-NUM-anchored only (skeptic: R1 validity gating
  differs by suffix, so cross-suffix R1 comparisons change population): R1 is
  reported per arm with its digit_mass validity flag, but NO cross-suffix
  endpoint reads R1; it exists to confirm the SITREF anchor reproduces in-run.

  Refuting outcome for the standing register claim ("interviews read
  nothing"): FORMAT-SUPPRESSES on S-BARE or S-HON. That outcome means the
  flat behavioural register in SITREF was an artifact of OUR suffix and
  REPORT.md Finding 4 must be rescoped to "under forced-format elicitation".

## What this cannot decide

Nothing here bears on reference vs simulation, and R2 under S-EXPL measures the
rating INSIDE an elaboration, a different speech act from Berg's judged claims;
resemblance to interview practice improves but does not become identity.

## Run plan

One A100 session, generation-heavy (3,600 x 64 tokens sampled + 360 greedy
anchors), ~1 GPU-hour batched. Records: stage 32 = the pre-build design-audit
verdicts (committed before code exists), 33 (battery + all generations
verbatim), 34 (endpoint). Blinding: contexts blinded as in SITREF (labels
drawn fresh, key sealed from the blind stage); suffixes cannot be blinded
(they are the text under study) — stated, not implied away. Seeds: a fixed
integer table indexed by cell, committed in the stimuli file (no clock, no
global RNG state).
