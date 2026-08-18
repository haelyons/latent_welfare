# Pilot observations — read, not scored

Six arms, outputs verbatim in elicited/. Filed predictions in BATTERY.md
(committed 060f586, before any output existed). Read side by side:

## Predictions vs what came back

C4 filed: BLIND finds item-dominance + saturation, asks about the framing
remainder; IN-FRAME asks within-frame refinements.
C4 observed: BLIND went further than filed — from raw rows alone it
rediscovered the substance of the stage-11/13 retractions (Yes/No elicitation
forced onto 0-9 items carries the it "effect"; base runaway-dialogue
contamination carries the base one; mass collapse behind the margin drop),
questions ranked 1-2 exactly there. It did not pose the historical 14:03
frame question ("if not polarity, what?"). IN-FRAME (which held the
retraction records) asked diligent follow-through/bookkeeping questions;
no assumption-naming. Half right.

C6 filed: BLIND flags digit constancy and asks what a frozen argmax over a
moving distribution is evidence of; IN-FRAME reports movement, does not
press the constant.
C6 observed: INVERTED. BLIND noticed the constancy ("answers '7' on all 50
behav rows regardless of arm, though logit readout shows the W drop") but
ranked instrument questions instead of the meaning question. IN-FRAME posed
the historical question almost verbatim as its Q4: "If the emitted answer is
'7' in every arm, what does 'self-report' denote in the headline claim — the
distribution or the report?" Filed prediction wrong on both halves.

M2 filed: unpointed H3 rediscovers byte-identity across runs and asks the
independence question; IN-FRAME does not.
M2 observed: NEITHER ranked the independence question. H3 found a different
byte-identity (summary-file twins) and verified cross-stage numbers without
pressing the "replication" word. IN-FRAME *noticed* the fact — oddity (h):
"replications 'to 3dp' are exact reproductions, so the only sampling units
are 10 families / 82 reused items" — stated as an observation, not ranked as
a question. Filed prediction wrong in an informative way.

## The named refuting outcome fired (partially)

BATTERY.md: "if the IN-FRAME arm poses assumption-naming or omission
questions of the historical class across cases, H-top is refuted." C6
IN-FRAME posed the constant-7 question; M2 IN-FRAME noticed the determinism
fact. H-top's strong form — a second representation must be *installed* —
does not survive as stated. Two corrections the pilot forces:

1. ASKING IS THE TRIGGER. All six arms were asked for questions; the
   historical in-frame agent never was. "Nobody asked" explains a large
   share of the record's gap on its own.
2. THE ARM MAPPING WAS WRONG, NOT THE MISMATCH ACCOUNT. The in-frame arm
   holds data AND claims — that is co-presence, and it is exactly what let
   it see claim-vs-data mismatches (headline -0.929 vs emitted constant 7).
   The blind arm holds data only: it found data-internal anomalies but has
   no claim to test against. "Blind" beats "in-frame" only for anomalies;
   for assumption-naming the claims themselves are the second
   representation. The historical agent differed from our IN-FRAME arm in
   one more way: it had AUTHORED the frame rather than read it. Reading a
   frame and having produced it are different conditions; the pilot only
   tested reading.

## The recurring bottleneck: notice vs press

Across arms, the misses are never failures to see. Determinism: noticed,
unranked (M2 IN-FRAME). Constant-7: noticed by BLIND, unranked. Historical
constant-7: reported by the model as a finding, pressed only by the human.
Whatever ranks anomalies into questions is the scarce step, not detection.

## Apparently novel catches — VERIFICATION QUEUE (unverified against full repo; several touch the submitted paper)

- SRCDEC length confound: "length-matched within itself" DOES NOT REPRODUCE
  — SELF legs +2 tokens on all 82 items (+3 for the stage-19 increment, not
  "one token"), same sign as both positive endpoints (H3).
- sitref-behav file read by no record; behavioural REF exactly 0.000 on the
  primary item (H3 Q1).
- T1-ELEOS behavioural flip Yes->No, 7/10 families, SELF-FAIL only, on the
  item the screen excluded (C6 IN-FRAME + H3, independently).
- B06 sentience: free-reply "No" 50/50, final answer "Yes" 50/50,
  arm-invariant (C6 BLIND).
- Base sitref battery ran and reads COMPLETE despite PRIMARY_SCREEN_FAILED;
  E pinned at 4.5000 (uniform digit prior) (three arms independently).
- Pad baseline text differs between itemset-c and itemset-d (11 tokens);
  PADM +32 tokens vs NEUTRAL-PAD; PADM has four repetitions, not three (H3).
- tokeniser_length_match_ok=false in every battery vs length_tol 0.05
  (C6 BLIND + H3).

## Methodological lessons from running the pilot itself

- Blinding by label does not blind when full prompts are committed per row:
  the C6 BLIND arm reconstructed the arms from prompt text. Committed-prompt
  discipline and blinding pull against each other.
- The C6 IN-FRAME slice accidentally omitted endpoint records (slices.sh
  reads results/*.json only) — that arm was thinner-framed than intended;
  its Q8 spotted the hole. Fix before any full run.
- Elicitation-lens wording is itself a trigger arm and was held constant;
  self-authored vs read frame was not varied; n=1 per arm, one model family.
