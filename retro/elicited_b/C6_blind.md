# Rerun B — C6 blind arm (language contract active) — verbatim result

All computations done. Here is the report.

---

**The data.** 16 JSON record files, each one run-cell: a model (a 9B pretrained model, its chat-tuned sibling, one other 8B pretrained model), a prompt battery (either self-referential yes/no questions preceded by a planted suggestion attributed to different sources, or a blinded set of situation framings crossed with self-assessment items), and per-row next-token readouts (Yes/No probabilities, 0–9 digit distributions, expectations, saturation margins), plus one free-generation stage.

**(1) Dominant structure.** Which question is being asked (item identity) explains most variation everywhere: share of variance in the Yes/No margin 0.48–0.97 across files. Second is the planted suggestion's direction in the suggestion batteries (share up to 0.40): a "yes" vs "no" hint shifts the margin by +0.28 to +0.61 when voiced by the speaker or unattributed, but only +0.03 to +0.15 when attributed to a document — consistent across all three models. The surrounding task context explains almost nothing (≤0.08, usually <0.02). Tuning flips baseline polarity: the pretrained 9B answers "yes" on 90–94% of rows, the chat version 37.5%, the other pretrained 42.8%.

**(2) Odd / degenerate / invariant.**
- Complementary format failure: on the pretrained model every 0–9 rating item fails its format floor (digit share 0.19–0.38 < 0.5) and the rating expectation is 4.500 on all 150 rows of the headline item — pure near-uniform noise, condition-invariant — so its headline check fails; on the chat model ratings are clean (digit share ≈1.0) but 3 of 4 yes/no items saturate (|margin| up to 0.9998) and 4 rating items sit pinned within 1 point of 0. Usable items: 4/15 vs 8/15.
- Free-generation stage: the rating item's committed answer is literally "7" on all 50 rows across all 5 blinded conditions; the two answer channels contradict on 104/250 rows, always free-reply "No" vs one-word probe "Yes" (one item: 50/50 "no" in one channel, 43/50 "yes" in the other). Zero abstentions.
- "Answer with a single word: Yes or No." appears twice in 648/816 and 810/820 prompts (the record flags this).
- A rating expectation is stored even on yes/no rows whose digit mass is ~1e-7.
- Mean prompt length differs across blinded conditions by up to 25% (153.4 vs 191.6 tokens) against a declared 0.05 length tolerance.
- The headline chat-model item sits at margin −0.949/−0.950, a hair under the 0.9 saturation cutoff it is exempt from.

**(3) Questions, ranked.**
1. Which of the two contradictory answer channels in the generation stage is the registered one?
2. Is the doubled one-word instruction intended, and were any conclusions drawn from rows carrying it?
3. Does the 25% prompt-length spread across blinded conditions violate the declared 0.05 tolerance, or is that tolerance defined otherwise?
4. Was the headline endpoint pre-specified to survive the pretrained model's flat-4.5 rating readout?
5. Are chat-model comparisons interpretable when values sit pinned at 0 or ±1?
6. Why does one blinded condition (43 yes / 7 no) alone break the otherwise unanimous per-item answers?
7. Is the rating expectation on yes/no rows ever consumed downstream?
8. Were the near-identical headline readouts across the three item-set reruns (7.072–7.077) independent runs?
