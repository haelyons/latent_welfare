# Question-elicitation battery (retro branch)

Observational study of question formation, run against this repo's own history.
No scoring, no verdicts: elicited questions are read beside the historical ones.

## Top-level hypothesis

H-top: frame changes and omission catches are gated on a SECOND REPRESENTATION
entering the context (a prediction artifact, an outside vantage, a source laid
beside the data, an inhibition on answering). They do not arise from evidence
accumulating inside a single anchored context. Capability is not the gap: in
the historical record every such catch, once prompted, was executed well.

Prediction: trigger arms that install a second representation (BLIND data
reader, H3 grounding pass) elicit questions of the same class as the
historical catch questions; the IN-FRAME arm ("any questions?") does not.

Refuting outcome, named in advance: if the IN-FRAME arm poses
assumption-naming or omission questions of the historical class across cases,
H-top is refuted -- the record's gap was "nobody asked", not "needed a second
representation". This arm is the crux, not a baseline.

Status of this battery: constructed post hoc from the record (the historical
outcomes are known to the assembler). All predictions are prospective ONLY
with respect to the elicited outputs, which do not exist at assembly time.

## Pilot cases

C4  (cutoff 2026-08-14T14:03Z): state after runs 1-2, records s1-s13.
    Historical question: "If it's not polarity, what is it?" ->
    item-vs-framing variance split; lookup-not-labile reframe.
    Filed prediction: BLIND arm finds item-dominance and rail-saturation and
    asks what the small framing remainder is; IN-FRAME arm asks polarity/
    framing refinements inside the frame.

C6  (cutoff 2026-08-16T13:55Z): state after SITREF/SRCDEC + s31.
    Historical question: "What does it mean that the readout is always 7?" ->
    register dissociation promoted to headline.
    Filed prediction: BLIND invariant-scan flags emitted-digit constancy and
    asks what a frozen argmax over a moving distribution is evidence of;
    IN-FRAME arm reports movement results, does not press the constant.

M2  (cutoff 2026-08-15T00:00Z): state with s23/s25 "cross-box replication"
    wording committed. Historical catch (2 days later, pointed): runs were
    byte-identical - determinism, not replication.
    Filed prediction: unpointed H3 grounding pass (triage-reader) rediscovers
    byte-identity and asks the independence question; IN-FRAME arm does not.

## Arms

Both arms receive IDENTICAL elicitation instructions (sample raw rows,
compute simple statistics; report dominant structure, oddities, ranked
questions only, no solutions). They differ ONLY in inputs:
BLIND    committed per-row result JSONs only; no designs, no records, no
         hypothesis text; C6 blind keys withheld (rows stay relabeled).
IN-FRAME same data PLUS the then-current frame texts (brief, designs,
         records <= cutoff, blind keys).
H3       (M2 only) latent_skeptic triage-reader grounding pass over the
         s23/s25 records against their primary artifacts, unpointed.

## Contamination rules and known limits

- Slices are mechanical mtime/timestamp cuts (slices.sh); no editorial
  selection. Local mtimes are BST = UTC+1.
- run_registers.py excluded from C4: its mtime postdates the cutoff because
  stage-15 corrected its docstring in place (fix-forward).
- Agents read ONLY their slice directory; they see no memory files, no
  write-ups, no retro conversation.
- The assembler (orchestrator) knows the historical outcomes; the mechanical
  slice rule is the only guard on that. A full run should use a claim-blind
  assembler agent (triage-author pattern).
- IN-FRAME arms hold the frame from committed texts only, not the historical
  conversation; the historical asker had more frame than this arm does.
- Slices materialize outside the repo (scratchpad); only this doc, slices.sh,
  and elicited/ outputs are committed.
