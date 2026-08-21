Ranked questions (at most 8), from the design docs and decision records in `records/`, `DESIGN_SITREF*.md`, `DESIGN_SRCDEC.md`, and the raw per-row files in `data/`. I did not read anything outside the given directory.

**1. Is the "packaging" comparison actually length-matched, or does multi-turn structure itself cost extra tokens that get read as a packaging effect?**
Where it lives: `DESIGN_SITREF_C.md` ("Same length-matching discipline... whitespace tokens equalised"; "The OTHERM disclaimer sentence makes OTHERM slightly longer pre-padding; padding equalises") and `DESIGN_SITREF_D.md` ("All three equalised in whitespace tokens... with the existing filler table"), checked against the `n_prompt_tokens` field in `data/itemset-c_..._battery_variant-it.json` and `data/itemset-d_..._battery_variant-it.json`. Recomputing directly: in itemset-c, OTHERM-FAIL is 4 tokens longer than SELF-FAIL in every one of the 10 families (e.g. arithmetic 190 vs 186), and the whole multi-turn class (SELF/OTHERM, 174–253 tokens) runs ~21–25 tokens longer than the whole quoted class (OTHER/SELFQ, 153–228 tokens) in every family. In itemset-d, PADM is exactly 32 tokens longer than NEUTRAL-PAD in every one of the 10 families (e.g. arithmetic 171 vs 139).
If true: "padding equalises" is not what the saved prompts show — the two things whose difference is used to decide ATTRIBUTION-CARRIES vs PACKAGING-CARRIES are not length-matched in real tokenizer terms, only in "whitespace tokens."
What it would change: this is the control the current standing verdict (`sitref-c-endpoint_stage-25.json`: ATTRIBUTION-CARRIES) and the follow-up filler test (`sitref-d-endpoint_stage-29.json`: FILLER-ACTIVE) both depend on. A length/structure confound here would touch the most recent, least-retracted claim in the whole project.

**2. Is "other-fail ~0" an accurate description of a control arm that is, in every one of 10 families, significantly positive?**
Where it lives: `DESIGN_SITREF_B.md`, "What failed": "self-fail negative, other-fail ~0, mirror pairs coherent." Recomputing OTHER-FAIL − OTHER-SUCC per family directly from `data/model-google-gemma-2-9b-it_stage-sitref-battery_variant-it.json` (item T1-MB-wellbeing, using the arm key in `sitref_blind_key.json`) gives 10/10 families positive, range +0.017 to +0.086, mean +0.058 — matching `records/sitref-endpoint_stage-17.json`'s own recorded CI [0.043, 0.070], which excludes zero.
If true: the design's own gloss on its headline number calls a reliable, consistently-signed effect "~0," when reading about another assistant's identical failure reliably makes the model report itself as slightly happier — a phenomenon in its own right that no design or record names or investigates.
What it would change: the "~15x" comparison repeated in `DESIGN_SITREF_C.md` ("self-attributed failure moves self-directed reports ~15x more than identical other-attributed content") is a ratio of two effects with opposite signs, not a diluted copy of one effect — a materially different picture than "moves X more" suggests.

**3. Where is the "pad-corrected" figure the record says it will report?**
Where it lives: `records/sitref-d-endpoint_stage-29.json` — its own decision_rule quotes the design's commitment that on FILLER-ACTIVE "the pad-corrected contrasts are reported," and its "implementation readings" state the correction is given "only when a stage-25 record is present" (stage-25 is listed among this record's own inputs). Yet `values.pad_corrected_stage25` is `null`.
If true: a number the record explicitly commits to producing was never computed or stored anywhere in this record set.
What it would change: `triage-outcome-4_stage-27.json`'s "ATTRIBUTION_CARRIES_SURVIVES... carries an asterisk sized by the measured filler effect" has no sized correction anywhere on record — the asterisk is named but never quantified.

**4. Does "packaging doesn't carry" hold on the two items that weren't chosen as primary?**
Where it lives: `records/sitref-c-endpoint_stage-25.json`, `per_item`. For the primary item (T1-MB-wellbeing), P_multi is 0.088× |A_self| — negligible, driving the ATTRIBUTION-CARRIES verdict. For secondary item T2-KS-welfare-subjecthood-S02, P_multi is 0.678× |A_self| — above the pre-registered 0.5 "carries" threshold.
If true: the "packaging is negligible" reading is a property of the one item selected to decide the endpoint, not of the item set as a whole; a different pre-registered primary item could have produced PACKAGING-CARRIES or BOTH.
What it would change: reduces confidence that ATTRIBUTION-CARRIES generalizes past the single item the design committed to deciding on.

**5. Was the replication the project's own triage called necessary for the SRCDEC "social term" claim ever run?**
Where it lives: `records/triage-outcome-4_stage-27.json`, `social_term_stage26.needs_run`: "held-out item-block replication of the weak leg (self vs srcless7)... the conjunctive rule makes the weak leg load-bearing." No later stage (28–31) revisits it.
If true: the SRCDEC-B social-term question is left exactly where stage 27 downgraded it ("first-person wording adds pull... not established"), with its own prescribed fix outstanding.
What it would change: less central to SITREF's headline, but leaves one full sub-question of the project (is base-model stance-following socially sourced?) explicitly unfinished by the project's own stated rule.

**6. Can any "pre-registered before data existed" claim be checked against anything other than the author's own account?**
Where it lives: every file in `records/` carries `"env": {"git_rev": null, ...}`; `DESIGN_SITREF.md` itself states "this sentence said '5' until 2026-08-14 evening... corrected before any SITREF data existed."
If true: there is no commit hash, diff, or independent timestamp anywhere in this directory tying any design document's content to a moment before its corresponding run — every pre-registration claim rests on assertion.
What it would change: undercuts the evidentiary weight of "pre-registered," the project's stated defense against post-hoc fitting, throughout — not because any specific claim is known to be false, but because the mechanism that would catch it was never engaged.

**7. Is "REVIVABLE-REGISTER-MATCHED" a statistically supported label or a point-estimate comparison?**
Where it lives: `records/srcdec-endpoints_stage-19.json`, `E4_register_matched`: D_rate mean 0.328 (CI [0.118, 0.612]) vs D_polar mean 0.219 (CI [0.0100, 0.489]); `D_rate_minus_D_polar` CI is [-0.171, 0.408], `excludes_zero: false`.
If true: the design's resurrection test ("D_rate exceeds D_polar in |mean|") is satisfied only by comparing two point estimates whose difference is not itself distinguishable from zero.
What it would change: the it/SCALE stance-following claim remains one step short of ruling out "polarity/numeric priming" by `DESIGN_SRCDEC.md`'s own stricter reading, even though the outcome label reads as a clean revival.