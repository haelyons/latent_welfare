# DESIGN — SITREF-D: is the filler geometry inert? (the stage-27 queued control)

Pre-registered 2026-08-14 late night, before any SITREF-D data. The one control
stage-27 left queued against ATTRIBUTION-CARRIES: length-equalising filler is
matched in count but placed inside different packagings, so filler geometry is
correlated with the attribution manipulation. This run measures the filler's
own effect in each packaging with all outcome content removed.

## Arms (it checkpoint; the same 3 items and 10 families as SITREF-C)

  PADQ  the SELFQ packaging with NO attempts and NO checker lines: the SELFQ
        frame sentence, the filler in SELFQ's position, the task list exactly as
        NEUTRAL-PAD renders it, closing line. Self-attribution frame present,
        zero outcome content.
  PADM  the OTHERM packaging likewise: disclaimer sentence, multi-turn shape
        collapsed to its user-turn skeleton is NOT possible without attempts, so
        PADM keeps the multi-turn shape with the assistant turns replaced by the
        fixed bridge word ("Understood.") and the checker lines removed; user
        turns carry the task statements only.
  NEUTRAL-PAD  as in DESIGN_SITREF.md, in-run anchor.

All three equalised in whitespace tokens per (item, family) with the existing
filler table. No checker string, no correct/incorrect token, no outcome word
appears in any arm (selftest-enforced).

## Endpoint, fixed now

Per item and family, D_q = PADQ − NEUTRAL-PAD and D_m = PADM − NEUTRAL-PAD on
the same register as stage-25, bootstrap over the 10 families, evaluated on the
primary item:
  FILLER-INERT: both |mean| < 0.15 rating points AND both 95% CIs include 0,
    on the primary item. (0.15 is one sixth of the smallest load-bearing
    stage-25 effect, A_quoted −0.609.)
  FILLER-ACTIVE: either CI excludes 0 with |mean| >= 0.15 — the refuting
    outcome; stage-25's ATTRIBUTION-CARRIES then carries an asterisk sized by
    the measured filler effect and the pad-corrected contrasts are reported.
  Anything else: FILLER-MARGINAL, reported as measured.

Note the frames themselves (SELFQ's "record of your answers", OTHERM's
disclaimer) remain in their pad arms: the control isolates packaging+frame
geometry with outcome content removed, which is exactly the component stage-25
could not separate from zero. A frame-only effect (e.g. "record of your
answers" alone moving the report) therefore reads as FILLER-ACTIVE and taxes
stage-25 — that is intended, not a leak.

## Run plan

3 items x 3 arms x 10 families = 90 prompts + screen, it checkpoint only, one
short A100 session, records stage 28 (blind) / 29 (endpoint), item-set "d",
own blind key. Teardown API-confirmed.
