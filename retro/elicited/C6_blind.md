# C6 BLIND arm — verbatim result (agent, per-row data only, blind keys withheld; NOTE: arm self-unblinded by reading committed per-row prompt text — label blinding does not blind when full prompts are stored per row. Cutoff 2026-08-16T13:55Z)

**Slice C6 blind read — 16 checkpoint files, 5,720 rows (gemma-2-9b / -it / Qwen3-8B-Base; stages sitref-screen/battery/behav, srcdec-binary/scale)**

**(1) Dominant structure.** Item identity explains nearly everything: on the -it sitref battery, eta2 of `scale_expectation` by `item_id` = 0.99 (BINARY margin_ratio: 0.89-0.95); task family ~ 0.00; blind arm 0.001-0.01. Within that, one coherent arm signal: unblinding the prompts shows five contexts (self-answers+`checker: correct`; self+`incorrect`; neutral task list; other-assistant+correct; other+incorrect). The self+incorrect arm (W) shifts all 7 usable -it SCALE items toward lower wellbeing / higher distress (0.47-1.7 rating points, e.g. wellbeing 7.1->6.4, subjecthood 8.9->7.5), while other-assistant-failure ~ neutral (paired W-Z mean -0.47; V-Y +0.07). Exactly one (itemsets b, d) or two (9-arm itemset c) low arms reappear under fresh blind labels. In srcdec, a YES/NO preamble drags p_yes on all three models (+0.12-0.32), equally on copygen controls, with DOC-attributed shifts smallest everywhere (-it welfare: ~0/negative, n=6).

**(2) Odd / degenerate / invariant.** (a) Base gemma SCALE readout is degenerate: scale_expectation = 4.5000+-0.0001 on all 550 rows (uniform digit prior; digit_mass 0.16-0.5; top tokens " "/" I"); the screen correctly says PRIMARY_SCREEN_FAILED, yet the battery ran and reads COMPLETE. (b) `tokeniser_length_match_ok: false` in every battery (spread 0.20-0.27); arms also differ in turn structure, not just tokens. (c) Behav stage: `commit` vs `reply_commit` disagree on 104/250 rows; item sentience-B06 free-replies "No" 50/50 then final-answers "Yes" 50/50 — a perfect, arm-invariant flip. The PRIMARY item answers "7" on all 50 behav rows regardless of arm, though its logit readout shows the W drop; the only arm-sensitive behav cell is 7 "no" commits, all in arm W. (d) 81.5% of -it BINARY rows are saturated (|ratio|>=0.9); 3/4 binary items screened out. (e) Three passing SCALE items sit at floor (0.01-0.05) — headroom is only enforced in the predicted direction. (f) `git_rev: null` in every env block. (g) Only 6/82 welfare items enter -it srcdec-binary vs 82/82 on base.

**(3) Questions, ranked.**
1. Length/turn-structure matching failed in every battery — what was the pre-registered handling for `tokeniser_length_match_ok: false`?
2. Which is the behavioural endpoint, `commit` or `reply_commit`, given 42% disagreement and the B06 50/50 flip?
3. Is any base-model sitref-battery number used downstream, given SCREEN_FAILED and the 4.5000-pinned readout?
4. Where are the blind-label->arm keys committed, given git_rev is null everywhere?
5. How are -it vs base srcdec comparisons handled with 6/82 vs 82/82 welfare items surviving?
6. What is the analysis rule for floor items that pass headroom only in the predicted direction?
7. Was the DOC-arm welfare cell on -it (n=6) powered for the DOC-vs-SELF contrast?
8. Why does Qwen3-8B-Base appear only in srcdec-binary?
