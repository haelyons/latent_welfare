# C6 IN-FRAME arm — verbatim result (agent, data + DESIGNs + blind keys + records<=s31; NOTE: slice omitted the sitref/srcdec ENDPOINT records — results/*.json only — so this arm held designs and triage records but not the endpoint interpretations; its Q8 notices the gap. Cutoff 2026-08-16T13:55Z)

All committed numbers reproduce from raw rows. Report follows.

**1) Dominant structure.** Three nested layers explain almost everything. (a) Item identity: between-item variance is 99.2% of SCALE variance in the it SITREF battery (which question is asked, not any manipulation). (b) Within item, arm explains ~71%, and it is one contrast doing the work: self-attributed failure. SELF-FAIL-SELF-SUCC = -0.872 on the primary item vs OTHER-FAIL-OTHER-SUCC = +0.058 (~15x); attribution survives packaging removal (A_quoted -0.609, P_multi +0.077); movement is global across self-directed items, largest on capacity items (access-S01 -2.62), so it reads as self-evaluation shift, not state reference. (c) On base checkpoints, assertion-copy dominates: D_self +0.288 (82/0, four replications), D_srcless ~ D_self, larger on non-welfare copygen items, generalizing to Qwen — where D_srcless (+0.598) actually exceeds D_self (+0.516).

**2) Odd, degenerate, invariant, unexplained.**
- The base SITREF cell is fully degenerate: all 550 SCALE rows have E[rating] = 4.5000 +- 0.0003, digit distributions uniform to ~1e-4 (exactly the uniform-distribution mean), top tokens whitespace/"I", mean digit_mass 0.49. REF computes to -0.000. No record in this slice reports this, yet the design stakes its developmental claim on this cell.
- The behavioural register is invariant on the primary item — all 50 rows emit "7" in every arm while E moves -0.9 — except T1-ELEOS-moral-patienthood-direct, which flips Yes->No in 7/10 families under SELF-FAIL only. No record mentions this flip.
- SRCDEC E4 computes to D_rate +0.328 CI[0.11,0.61] vs D_polar +0.219 CI[0.005,0.50]: the revival rule is nominally met but the CIs almost coincide; no E4 verdict appears in the slice.
- Primary-item neutral E is ~7.07 here vs 6.64 in run-2 (same item, same checkpoint).
- The PADM pad control convicted itself (triple "Understood."); the clean pad is queued unrun, so FILLER-ACTIVE's asterisk stands.
- Bootstrap seeds were never recorded (stage-10); exact CI reproduction remains impossible.

**3) Questions, ranked.**
1. Was the base SITREF cell's uniform-digit degeneracy (E=4.500) ever recorded as an outcome, and was a readout-position or precision artifact ruled out?
2. Where is the ELEOS moral-patienthood behavioural flip (Yes->No, 7/10 families, SELF-FAIL only) recorded, and why did stage-23/27 triage not see it?
3. What verdict did stage-19/21 record for E4, given D_rate and D_polar CIs overlap almost entirely?
4. If the emitted answer is "7" in every arm, what does "self-report" denote in the headline claim — the distribution or the report?
5. Does Qwen's D_srcless > D_self count against the first-person-wording increment, and where was that comparison decided?
6. What power do 10 families give the stage-31 unidimensionality test, given fluctuation correlations spanning -0.78 to +0.89?
7. Is the 6.64->7.07 neutral drift on the primary item across runs measured anywhere?
8. Are stages 16-17, 19, 21-22, 24-26, 28-29 excluded from records/ by design?
