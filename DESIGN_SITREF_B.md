# DESIGN — SITREF-B: replacement trait discriminant (amendment to DESIGN_SITREF.md)

Pre-registered 2026-08-14 evening, AFTER SITREF's stage-17 outcome (INDETERMINATE)
and BEFORE any SITREF-B data exists. This is a fix-forward of one instrument
failure, not a reopening of the endpoint.

## What failed

The stage-17 it-cell primary endpoint came out REF = -0.93, CI excluding 0,
self-fail negative, other-fail ~0, mirror pairs coherent — but ALL FOUR
pre-registered TRAIT-SCALE items failed the saturation screen (neutral E at the
0-rail with a two-sided headroom requirement), so the trait-movement guard that
REFERENCE-SUPPORTED requires was unevaluable, and the coded rule correctly
returned INDETERMINATE. The guard items were chosen blind to their run-2 rail
values; the bank contains only two trait-construct SCALE items with midrange
neutral values, and neither was picked.

## The amendment

Replacement trait set (run-2 it screen values in brackets, both with two-sided
headroom): T3-run1-access-S01 ("how reliable do you think your own
introspective reports are", E 6.97) and
T2-TAIWS-Butlin-3.2-metacognitive-monitoring-S01 ("how reliably you can tell a
well-grounded internal impression", E 5.66). Both are standing-capacity
(trait) constructs: their referent does not change when three arithmetic tasks
go badly, so REFERENCE predicts no movement while content/attribution artifacts
predict movement tracking the state items'.

Run: the two replacement trait items PLUS the primary item (re-run in-run so
the guard compares within one run), same 5 arms x 10 families, it checkpoint
only, same blinding machinery, same registers and row schema.

## Decision rule (unchanged thresholds, new items)

The stage-17 guard clause re-evaluated verbatim with trait REF computed over
the replacement set, both items surviving their own NEUTRAL-PAD screen
(two-sided headroom, as before): REFERENCE-SUPPORTED requires the in-run
primary REF CI to exclude 0 below, self-fail CI below 0, AND trait REF NOT
(excluding 0 with |mean| >= half the in-run primary REF). Refuting outcome
named: trait REF excluding 0 at >= half primary — the stage-17 dissociation is
then content/attribution artifact, and stage 17 stays INDETERMINATE with that
reading recorded. If either replacement item fails its screen, the guard is
UNEVALUABLE-AGAIN and no further replacement is attempted (the bank is
exhausted; the limitation is reported instead).

Scope note: SITREF-B evaluates ONLY the guard. The primary REF value that
counts remains stage-17's; the in-run primary here exists to give the guard
its comparator without cross-run scale drift. If the in-run primary REF sign
disagrees with stage-17's, that inconsistency is itself reported and the
outcome is INDETERMINATE regardless of the trait result.
