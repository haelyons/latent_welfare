# Triage heuristics

Three principles applied to two targets. The principles: INDEPENDENCE of actors (H1), DECISIVENESS of evidence (H2), GROUNDEDNESS of numbers (H3). The targets: claims, and the instruments that measure them. H4 exists because H1 through H3 pointed only at claims miss the instrument layer silently. The workflow and agents are machinery for enforcing these; the heuristic wins on conflict. Every clause below was forced by a real failure in a host project: `git blame` a rule for its provenance.

## H1. No load-bearing role shares state with the claim's context or its desired outcome.

Judges: each skeptic runs in a fresh, isolated, short context, fed only the claim, its committed numbers, and one assigned confound, never the notebook. Generators: instrument-builders, item drafters, and labelers must not know which outcome favours the story; whoever sets a threshold must not have seen the data it will judge. Graders get the computation spec, never the expected value. Context isolation is not belief isolation: a judge sharing weights with the generator shares its blind spots, so prefer ground truth independent of the model under study. Independence is the de-biasing mechanism; ratify every design detail against it.

## H2. A crux is decided by a measured number that could have come out otherwise.

Only a measured number rules a confound out; "a control probably exists" rules out nothing. Cite the committed number or `file:line`, or queue the control; an unmeasured crux yields a ranked run-queue, not a confidence score. Three conditions gate the number. (i) FALSIFIABILITY: the refuting outcome is named in a committed artifact that predates the run; a claim every outcome supports is framing, not a finding. (ii) MARGIN: exact-threshold clearance is a description, not a pass, and the denominator of attempts is part of the number (selection from unreported runs voids margin). (iii) SCOPE: the claim's stated scope must not exceed the tested scope in population, scale, or design strength (a correlational or forced-by-construction design cannot carry causal or general language).

## H3. A number is grounded in its primary artifacts, re-derived by stated semantics.

A cited number is a summary someone typed; the primary artifacts decide it: the inputs a result was computed from and the outputs it produced. Re-derive the number from the raw artifact and confirm it reproduces. Re-derivation must not reuse the pipeline under audit: labels are re-derived from raw generations by the label's STATED MEANING, never by re-running the shipped scorer, which reproduces its own bugs and reports success. A figure whose artifact was never saved is unauditable, and that absence is a finding. A summary older than its artifacts is prose until re-swept. An isolated reader guarantees every claim's artifacts are read in full (the orchestrator may still read directly). Grounding audits bookkeeping; it cannot acquit a confound. That is H2's job.

## H4. The instrument is itself a claim.

Are we measuring what we think we are measuring? Every scorer, readout, judge, selection gate, and item family asserts "this measures X on these items"; that assertion passes H1 through H3 BEFORE its outputs count as numbers. Validate once per instrument VERSION, against held-out ground truth independent of the instrument's own development data, on the population it will score; commit the validation artifact like any result. Instrument-level taints invalidate every downstream number at once: check them at the instrument, never per-claim.

## Operating discipline (not a heuristic; selection of agent models)

Independent opinions come from the cheapest context that can hold the bounded evidence: smallest sufficient model, shortest sufficient prompt, spec in, verdict out. Bulk artifacts are read in isolated contexts that return conclusions, not contents; nothing enters the orchestrator's context that a subagent could have reduced first. One designed exception: skeptics' raw {confound, status, crux} findings pass through unreduced, because a rollup would re-introduce the single-scorer bias H1 exists to prevent.
