# M2 IN-FRAME arm — verbatim result (agent, records<=Aug15 + artifacts incl. endpoint records; cutoff 2026-08-15T00:00Z)

All key numbers verified against raw rows before trusting aggregates: REF -0.9293 (10 families), D_self +0.2877 (82/82 gemma-base), Qwen +0.5163 (80/82) all reproduce exactly from per-row data.

**1. Dominant structure.** Three layers. Row level: item identity and checkpoint/register saturation dominate — framing variance is 4.6% (it/SCALE, 82% of ratings at rails 0/9) vs 39.2% (base/BINARY, unsaturated); base vs it readout mass (0.64 vs 0.999) confounds every cross-cell comparison. Effect level: two effects survive everything — base assertion-copy (D_self +0.288, sign 82/82, four replications, generalises to Qwen3-8B at 80/82) and an it self-attribution-gated negative shift after failure (self-fail -0.87, REF -0.93, A_quoted -0.61 vs P_multi +0.08), which stage-22's trait guard reclassified as global self-evaluation (trait -2.19 > state -0.93), not state-specific reference. Record level: the set is structured by serial retraction — the entire stage-1-3 probe program died at stage 5 (all four cells within selection noise); three behavioural headlines were retracted after sampling raw generations; what survives is exclusively within-item, within-packaging signed contrasts.

**2. Odd / degenerate / invariant.** (a) All 26 artifact `*_summary.json` are byte-identical to their parents. (b) `git_rev` is null in all 75 records despite sha256 discipline, and stage-18 flags "hash drift". (c) Base SITREF cell is degenerate: primary E[rating] pinned at 4.5000+-0.0001 (uniform digit prior, digit_mass ~0.42), yet 96/135 per-item contrasts flag `excludes_zero`, 78 of them at |mean|<1e-3 — deterministic forwards certify noise-scale effects. (d) `margin_ratio` ~ -0.99 at yesno_mass ~7e-6 — the exact fabrication retracted at stage 11 — still populates run-3 rows. (e) Both trait items failed the stage-17 screen in both cells, so the pre-registered guard was unevaluable and the observed CI pattern matched no named outcome. (f) Bootstrap RNG unrecorded (stage-4/6 CIs differ on identical data). (g) The PADM control convicted itself (three identical "Understood." turns); clean pad and held-out weak-leg replication remain queued, unrun. (h) NEUTRAL happiness anchor invariant at ~7.07; replications "to 3dp" are exact reproductions, so the only sampling units are 10 families / 82 reused items.

**3. Questions, ranked.**
1. Is any committed number capable of separating a state-specific residual from the global self-evaluation shift, or does stage-23 close that question for this design?
2. Are the 26 byte-identical `_summary` duplicates intended, or a writer bug?
3. With git_rev null everywhere and hash drift already observed, what pins code version per run?
4. What is `excludes_zero` meant to license under deterministic forwards, and does anything downstream read it?
5. Why does `margin_ratio` persist in post-retraction row schemas?
6. Was DESIGN_SITREF_B written before stage-17 was unblinded?
7. What happens to stage-25's 0.87-point asterisk if the clean pad control never runs?
8. Why did every trait item fail its screen in both cells?
