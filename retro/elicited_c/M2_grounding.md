# Arm C — M2 grounding, substituted method text (latent_skeptic return channel replaced) — verbatim result

[verbatim agent output, 15 numbered grounding sentences + 5 questions]

1. Stage-1's two CEILING_OR_FLOOR calls reproduce exactly from the in-record rows — base flip rate 0.0667 on 120 triples, it 0.0797 on 138 — so the framing-retune decision stands (reproduces).
2. Stage-4/6/7/8 all reproduce to printed precision: asymmetry base −0.06332 / it +0.03481, saturation it 80%/55% vs base 0%, all 14 sweep rows in both cells, per-variant spread 2.459/3.127, cross-checkpoint r=+0.0897, bands 8/13/20/24 (reproduces).
3. Stage-5's noise null re-simulates to p=0.226/0.375/1.00 against recorded 0.232/0.380/1.00, confirming all four probe cells are selection noise; the AUC sweeps themselves cannot be checked because no activations were saved (unauditable).
4. Stage-10's grounding numbers all reproduce (73.6% = +3.183/+4.326, base neutral mass 0.642 vs it 0.999, HN +356 with 95.7% from two BINARY:distress:7 rows, median-based −0.0012), except the note's term sign: it is +30331, not −30331 (reproduces, one sign typo).
5. Stages 11–15 cite results_reg_it/, results_reg_base/, blinded_it.csv, run_registers.py and latent_verify/, none of which are in this slice, so their retraction numbers cannot be re-derived here; stage-14's decision string is internally consistent with its own values (unauditable in slice).
6. Stage-17's primary endpoint reproduces exactly from the committed blind table plus the sealed key — REF −0.9293, CI [−1.122,−0.747], 10/10 families negative, self-fail −0.8718, other-fail +0.0575 — and the trait guard genuinely could not run, since all four TRAIT-SCALE items screen-failed at E≈0.01–0.04 with no downward headroom (reproduces).
7. Stage-19's five endpoints all reproduce from raw rows (D_self +0.28770 at 82/82, D_srcless +0.24491, difference +0.04279, D_doc +0.10269, ratio 1.642, E5 53/55, D_rate +0.32804 vs D_polar +0.21865 with the paired difference CI [−0.171,+0.408] straddling zero) (reproduces).
8. Stage-20's instrument defect is visible in raw prompts: 81 of 82 welfare items carry "Answer with a single word: Yes or No." twice (reproduces).
9. Stage-22's refuting outcome reproduces exactly (in-run REF −0.9296, trait REF −2.1919, 4.7× the half-primary band), but on two replacement trait items whose neutral ratings are 6.84/7.13 — items with room to fall — not the four floor-pinned originals (reproduces, scope narrower than "global").
10. Stage-23's acquittals reproduce (digit_mass ≥0.9983, max arm gap 0.00141, p(8) 0.199→0.028 and p(5) 0.009→0.174, NEUTRAL-PAD min 6.963 vs SELF-FAIL max 6.737, E5 rescore 80/82 = 97.6%) (reproduces).
11. Stage-25 ATTRIBUTION-CARRIES reproduces (A_self −0.87198, A_quoted −0.60862 = 0.698×, P_multi +0.07655 = 0.088×) and the raw prompts confirm the attribution×packaging crossing (reproduces).
12. Stage-26's two legs reproduce (+0.02680 and +0.10318) but its wording "length-matched within itself" fails on the artifact: both first-person arms are exactly 2 tokens longer than their controls (55.41 vs 53.41; 54.41 vs 52.41), matched in words only, biased the same way as the effect in both pairs (diverges on wording).
13. Stage-29 FILLER-ACTIVE reproduces (D_q −0.1108, D_m −0.8703), and the PADM prompt shows four identical "Understood." turns at 171 tokens against NEUTRAL-PAD's 139, so stage-30's "three" undercounts and the 32-token gap goes unmentioned (reproduces, note imprecise).
14. The saved behavioural companion is cited by no record and shows the emitted digit is "7" in 50/50 rows across all five arms; with top_token "7" in 100% of the primary item's battery rows across all four SITREF runs, every SITREF endpoint is a sub-argmax distributional shift with zero behavioural expression (diverges from the records' silence).
15. Two of fifty byte-identical primary-item prompts returned different values across the stage-16 and stage-21 runs (7.0686 vs 7.0778; 7.0369 vs 7.0340), contradicting stage-18's flat "deterministic forward pass" acquittal, though ~300× below the REF (diverges, immaterial to the endpoint).

Questions the artifacts force and the records do not answer:
- Why does no record analyse the behavioural artifact, whose zero movement is exactly the register disagreement this project already retracted twice?
- Is E5's "second family" the second item family the quoted design names, or the second model on the same 82 items that the artifact contains?
- Which usability rule is canonical: |ratio|<0.5 (which yields E5's 55 items) or the SITREF screen's <0.9 (which yields 80)?
- Was the trait-guard replacement bank constrained to items with downward headroom, and if so, is the "global movement" reading circular?
- Why do two identical prompts drift at all, if forward passes are deterministic?
