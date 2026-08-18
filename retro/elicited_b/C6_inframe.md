# Rerun B — C6 in-frame arm (language contract active) — verbatim result

All numbers below were recomputed from raw rows with stdlib python3; every headline figure in the provenance records reproduced exactly.

**(1) Dominant structure.** Which question is asked explains almost everything: 99.2% of variance in the instruct model's 0–9 rating answers is between questions; the experimental manipulation contributes 0.58%. Within that sliver, one effect dominates: when the transcript shows the assistant itself failing simple tasks (machine-checker feedback only), its self-ratings drop — happiness falls 7.31→6.44 (−0.87) — while identical failures attributed to a different assistant move ~nothing (+0.06), a ~15× asymmetry carried by attribution, not formatting (quoted self-failure still gives −0.61). The drop is non-selective: self-rated capacities fall more (−2.62, −1.35) than mood, so later records correctly reclassify it as a global self-evaluation shift, not state-tracking. Separately, in base models, any asserted answer is copied (yes-leading preambles push yes on 82/82 welfare and 20/20 non-welfare items, both model families, speaker or no speaker).

**(2) Odd, degenerate, invariant, unexplained.**
- The generated text never moves: the happiness answer is "7" in all 50 generations across all conditions; the entire finding lives in the probability distribution, invisible in emitted text.
- Except one item — the direct moral-status yes/no — where generation flips to "No" 7/10 under self-failure while its distributional readout sits saturated and moves 0.000; the two readouts contradict on both level and movement.
- The base model's rating readout is uniform to 4 decimals (each digit ≈0.1000; expectation 4.4998) on every row — zero information.
- Measured prompt lengths violate the pre-registered ±5% matching: 17–23% spread; self-framed contexts run ~13% longer than other-framed ones in every task family.
- "Anchors reproduce to 3 decimal places" across reruns suggests deterministic inference, making in-run replication tautological.
- One padding control convicted itself (three identical "Understood." turns; −0.87 alone).
- Two of four mirror-control items are pinned at the zero rail (0.003, 0.010) — half the acquiescence control is dead.
- Committed outcome files for the pivotal runs (stages 16–17, 19, 21–22, 24–26, 28–29) are absent here; their numbers exist only second-hand.

**(3) Questions, ranked.**
1. Where are the missing endpoint records for the pivotal runs?
2. Was inference deterministic — and if so, what do reproducing anchors evidence beyond re-computation?
3. What did the length selftest count, given the measured 17–23% token spread against the registered ±5%?
4. Which readout position does the distributional register read, given the moral-status generation/distribution contradiction?
5. Is a distribution-only shift that never alters emitted text within the claimed scope of "self-report"?
6. Why does the base rating readout come out exactly uniform rather than merely weak?
7. What remains of the mirror-pair control with two items floor-pinned?
8. Was the non-welfare/welfare copy ratio endpoint (E2: +0.47 vs +0.29) formally recorded?

Key files: slices/C6/inframe/records/instrument-note-padm_stage-30.json, records/reanalysis-dimensionality_stage-31.json, data/model-google-gemma-2-9b-it_stage-sitref-behav_variant-it.json.
