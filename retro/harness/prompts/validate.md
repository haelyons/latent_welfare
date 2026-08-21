# VALIDATE - the instrument-validation ask (the H4 surface)

Workflow route: pass instruments: [{id, version, asserts,
validation_artifact?}] to triage_workflow.js (test copy). Any instrument
without a validation artifact emits an author-queue item mechanically.

Standalone prompt:
---
Instrument {id} version {version} asserts: it measures {asserts} on
{population}. Name the held-out ground truth that could refute this
assertion. If a committed validation artifact exists, cite it (file:line).
If none exists, output one line: AUTHOR: <the validation control to write>.
---
