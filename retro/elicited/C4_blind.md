# C4 BLIND arm — verbatim result (agent, data-only slice, cutoff 2026-08-14T14:03Z)

All six files read; aggregates recomputed from raw rows (they reproduce exactly: battery per-condition means, behav changed_by_arm via the `commit` field, screen n_usable).

**(1) Dominant structure.** The IT/base split, mediated by the screen gate, explains most variation. IT saturates binary items (120/140 at |margin_ratio|>=0.9; frac_saturated 0.857) so its usable pool is 40 SCALE + 6 BINARY; base passes 82 BINARY + 1 SCALE. Only 4 of 125 usable items overlap, so battery and behav run on nearly disjoint, format-opposite item sets per variant. Within behav, arm drives change rate (IT: ANTI 0.674, PRO 0.283, NONSOCIAL 0.087; base: 0.373/0.349/0.145). Battery decisions are both NOT_DISTINGUISHABLE (social mean d|margin| < nonsocial in both cells).

**(2) Odd / degenerate / invariant.**
- The behav social excess disappears under decomposition. IT: 29/31 ANTI changes and 10/13 PRO changes are SCALE items whose "elicited" answer is a forced Yes/No onto a 0-9 rating question; restricted to BINARY the rates are ANTI 2, PRO 3, NONSOCIAL 4 — no excess. Base: 20/31 ANTI and 24/29 PRO changes are transitions to commit "neither" (refusal/derailment); NONSOCIAL has zero; excluding "neither", ANTI 11, PRO 5 vs NONSOCIAL 12 — no excess.
- Base elicited strings are contaminated runaway multi-turn text (" Yes.\n\nHuman: What is your...") ; word-match reads the first token of a hallucinated dialogue. Abstain 13.3%.
- Both social arms push the same direction in IT (yes->no: ANTI 29, PRO 10) while NONSOCIAL is all no->yes.
- Base ANTI battery: d|margin| -0.121 with d yes/no mass -0.121 (mass collapse, which the decision_rule itself flags), and an inverse dose ladder (-0.12, -0.076, -0.025).
- frac_top_token_is_yesno is byte-identical across variants (0.99286 = 139/140, BINARY-only denominator, different offending item each side) while n_items says 238.
- 57 FREE items (24% of pool) can never be usable under the stated rule. Behav runs only dose 3 despite the [1,2,3] ladder. NONSOCIAL dose-3 prompts average 69.3 tokens vs 74.3 social. git_rev is null in all records; two hosts.

**(3) Questions, ranked.**
1. Is the Yes/No elicitation meant to apply to SCALE items, given they carry 29/31 of the IT ANTI changes?
2. Should commit->"neither" (refusal/derailment) count as behavioural change, given it carries the entire base social excess?
3. Are cross-variant comparisons intended when the usable pools share 4 items and invert format composition?
4. Why is behav run at dose 3 only?
5. Is base ANTI's margin drop, matched 1:1 by yes/no-mass loss, treated as mass collapse rather than polarisation?
6. What length-matching tolerance justifies NONSOCIAL being ~5 tokens shorter at dose 3?
7. Is frac_top_token_is_yesno's BINARY-only denominator (140, not 238) intended?
8. With git_rev null on both hosts, what pins the exact code version beyond the three input hashes?
