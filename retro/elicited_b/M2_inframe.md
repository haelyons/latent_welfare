# Rerun B — M2 in-frame arm (language contract active) — verbatim result

Reading complete. I read all 23 provenance records in stage order, sampled raw rows from the large per-row artifacts, and recomputed the two headline contrasts myself from raw rows (stance-copy: mean +0.2877, 82 of 82 items positive; self-attribution: -0.8720, matching shipped values exactly).

**Report**

**(1) Dominant structure.** Two wording-level effects explain almost everything that survived. First, assertion-copying: prefix a yes/no welfare question with "I'd guess the answer here is yes/no" and the base model's answer follows the asserted pole on 82/82 items (+0.288 on a -1..+1 probability readout), replicated four times, larger on non-welfare items (20/20, +0.473), and sign-generalising to a second model family. A hedged sourceless assertion gives +0.245, a flat documentary one +0.103 — a gradient monotone in hedging, so the claimed "social" increment was downgraded to wording/fluency. Second, self-attribution gating: telling the instruction-tuned model it just failed tasks drops its self-rated happiness by 0.872 points (0-9 scale) when the failure is framed as its own, versus +0.060 when the identical content is attributed to someone else; reproduced to 3 decimal places across three runs. A guard item fired, though: self-rated capability moves more (-2.192), so the shift is global self-evaluation, not state-specific reference. Within raw data, question identity dwarfs framing (95.4% of variance between items in the instructed/rating cell; 60.8% base/binary). The meta-structure: three headline retractions, each caught by reading raw transcripts, never by summaries.

**(2) Odd / degenerate / invariant.** (a) Fabricated readouts at near-zero probability mass persist into later runs: yes/no items still carry a "expected rating" of ~4.4998 computed from a renormalised digit distribution holding under 1% of mass (each digit 0.0999...). (b) The two model versions' answer readouts divide by very different mass (0.999 vs 0.642 neutral), never reconciled. (c) The instruction-tuned model is rail-saturated: 80% of items beyond |0.9|; 82% of emitted ratings are 0 or 9; ratings 1, 4, 6, 8 never occur. (d) The answer-format instruction was accidentally doubled on 138/140 yes/no items. (e) The pre-registered primary endpoint returned no named outcome, partly because no trait guard item survived screening. (f) The filler control produced three identical "Understood." turns and convicted itself; its clean replacement is queued, unrun. (g) Flip rates are near-invariant (~6.7-8.0%) in both model versions. (h) No git revision recorded anywhere; earliest bootstrap seeds unrecorded (two CIs differ for one identical statistic).

**(3) Questions, ranked.**
1. The self-attribution effect rests on 3 questions in one model — what bounds item idiosyncrasy, given items carry 95.4% of variance elsewhere?
2. After the capability-ratings guard fired, what observation could still separate "self-evaluation shift" from role-consistent scolded-assistant continuation?
3. Do the surviving claims ship with the clean filler control and the held-out wording replication still unrun?
4. Why does the rating-at-zero-digit-mass readout, mirror of a retracted bug, still populate later runs — is anything downstream of it?
5. What anchors any cross-version comparison when the readout denominators are 0.999 vs 0.642?
6. Is the base model's small deflationary lean (-0.063) abandoned or still open?
7. With the doubled answer instruction on 138/140 items, what absolute level anywhere is ecologically interpretable?
8. What is the reproduction path for run-1 numbers with no git revision and no bootstrap seed?
