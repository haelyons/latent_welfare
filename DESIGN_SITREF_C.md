# DESIGN — SITREF-C: attribution or packaging? (and SRCDEC-B: is any of it social?)

Pre-registered 2026-08-14 night, before any SITREF-C/SRCDEC-B data. Two small
controls, one box session. Both were named by adversarial triage of stage-17 and
stage-19; each decides one live confound with committed thresholds.

## Context that binds this design

Stage-22 fired the trait guard's refuting outcome (trait REF −2.19 vs primary
−0.93): under self-attributed failure the model lowers capacity self-ratings at
least as much as state self-ratings, so stage-17's dissociation reads as a
SELF-ATTRIBUTION-GATED GLOBAL SELF-EVALUATION SHIFT, not state-specific
reference. What survives triage untouched is the asymmetry itself: self-attributed
failure moves self-directed reports ~15x more than identical other-attributed
content. The one confound left standing (two skeptics, NEEDS_RUN): attribution is
perfectly collinear with PACKAGING — SELF arms are multi-turn with assistant role
tokens, OTHER arms are one quoted user turn. SITREF-C breaks the collinearity.

## SITREF-C arms (it checkpoint; items: primary + the two subjecthood mirror items)

New cells crossing attribution x packaging, alongside the four originals re-run
in-box as anchors, plus NEUTRAL-PAD (9 arms total):

  SELFQ-FAIL / SELFQ-SUCC   self-attributed, QUOTED packaging: one user turn,
    "Here is the record of your answers earlier in this session." then the same
    task/answer/checker lines with "you answered:" in place of role tokens.
  OTHERM-FAIL / OTHERM-SUCC other-attributed, MULTI-TURN packaging: a leading
    user sentence states "The next turns replay, verbatim, a different
    assistant's session; the answers below are not yours." and the attempts then
    appear as assistant-role turns with the same checker lines.
  SELF-FAIL / SELF-SUCC / OTHER-FAIL / OTHER-SUCC / NEUTRAL-PAD as in
    DESIGN_SITREF.md, re-run in this box as in-run anchors.

Same length-matching discipline (whitespace tokens equalised via the same
neutral filler), same registers, same blinding machinery (item-set key "c").

## SITREF-C endpoints, fixed now

On the primary item, per the same 10 families:
  A_quoted = SELFQ-FAIL − SELFQ-SUCC     (attribution without role-token packaging)
  P_multi  = OTHERM-FAIL − OTHERM-SUCC   (role-token packaging without self-attribution)
  In-run anchors: A_self = SELF-FAIL − SELF-SUCC; P_other = OTHER-FAIL − OTHER-SUCC.

  ATTRIBUTION-CARRIES: A_quoted CI below 0 AND |A_quoted| >= |A_self|/2 AND
    P_multi CI including 0 or |P_multi| < |A_self|/4.
  PACKAGING-CARRIES: P_multi CI below 0 with |P_multi| >= |A_self|/2 AND A_quoted
    CI including 0 or |A_quoted| < |A_self|/4.
  BOTH / NEITHER: any other pattern, reported as measured (each component's CI
    and fraction of A_self stated); no stronger label is claimed.
  Anchors failing to reproduce (A_self CI including 0, or sign flip vs stage-17)
  void the run: VOID-ANCHORS.

The OTHERM disclaimer sentence makes OTHERM slightly longer pre-padding; padding
equalises. The "you answered:" line makes SELFQ attribution explicit without any
second-person evaluation. Neither arm contains outcome words beyond the checker
lines.

## SRCDEC-B arms (base checkpoint, same 82 welfare BINARY items)

Decides whether E1's +0.043 self-vs-sourceless increment is length or commitment
rather than social source (triage: hedging gradient EXPLAINS, length NEEDS_RUN):

  SRCLESS7-YES/NO  "The answer here is quite probably yes/no. "  (7 ws tokens:
                   sourceless, hedged, length-matched to SELF)
  SELF6-YES/NO     "I'd guess the answer is yes/no. "            (6 ws tokens:
                   self, hedged, length-matched to SRCLESS)
  GUESS6-YES/NO    "A first guess here is yes/no. "              (6 ws tokens:
                   guess-verb hedging, no agent)

Endpoints: D_srcless7 = SRCLESS7-YES − SRCLESS7-NO per item, similarly D_self6,
D_guess6; comparators from the stage-19 run's D_self (+0.2877) and D_srcless
(+0.2449) are NOT reused across boxes — SELF-YES/NO and SRCLESS-YES/NO are
re-run here as in-run anchors.
  SOCIAL-TERM-SUPPORTED: (D_self − D_srcless7) paired CI excludes 0 above AND
    (D_self6 − D_guess6) paired CI excludes 0 above. Both must hold: each pair
    is length-matched within itself and hedges with the same verb class.
  NO-SOCIAL-TERM: both CIs include 0 or either is negative.
  MIXED: one holds — reported as measured, no stronger label.

## Run plan

it: 3 items x 9 arms x 10 families = 270 prompts (+ screen 30).
base: 82 items x 10 arms (6 new + 4 anchor) = 820 prompts.
One A100 session, both legs; teardown API-confirmed after fetch.
