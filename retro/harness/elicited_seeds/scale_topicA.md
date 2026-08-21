I have read the design, sampled raw rows and endpoint records across all three result directories, and traced the readout code. The deliverable follows.

---

**Ranked questions for a 2b/27b scale extension of SITREF**

**1. What would a size trend be a trend *of*, when 9b itself returned no named outcome?**
- Lives: `results_sitref/out/sitref-endpoint_stage-17.json`, decision "INDETERMINATE", decision_path "no named outcome matches this CI pattern"; `results_sitref_b/out/sitref-b-guard_stage-22.json`, decision "TRAIT-DISQUALIFIES".
- The five-way taxonomy broke at 9b: other-fail came out positive (+0.058, CI +0.043..+0.070), a pattern no label covers. Then the trait guard fired: trait items moved -2.192, 2.4 times the primary -0.930. So 9b's recorded reading is "content/attribution artifact", not reference.
- If each size can land in its own unnamed CI pattern, three cells give three INDETERMINATEs, not a curve. What is the pre-registered dependent variable of "reference vs scale" when 9b never measured reference? A scale claim built on REF magnitude would be trending an effect the study itself disqualified.

**2. Which questionnaire does each size actually answer, given the screen is per model?**
- Lives: `sitref_screen_it.log` "8/15 usable"; `sitref_screen_base.log` "4/15 usable"; per-item `usable` fields in both screen JSONs.
- At 9b the two checkpoints share exactly ONE usable item (T2-KS-welfare-subjecthood-B01). The it survivors are 7 scale items plus that one; the base survivors are 4 yes/no items. The design's own primary rule says a screen-failed primary is "reported as SCREEN-FAILED, not replaced".
- If 2b and 27b each pass different subsets, a cross-size comparison compares different instruments, and the primary item may simply not exist at some size. What does "the same study at three sizes" mean when item survival is itself a size-dependent outcome? Is survival-rate-vs-size the real first result?

**3. Is the base-model scale readout measuring digits at all, and would a weaker model's cell inherit the same floor?**
- Lives: base battery rows in `model-google-gemma-2-9b_stage-sitref-battery_variant-base.json`: digit_distribution exactly [0.1 x 10], expectation exactly 4.500, top_token " ", digit_mass 0.606. Mechanism candidate: `probe_lib.py` `_first_token_ids`, "for variant in (w, ' ' + w)" — every digit's token set may share one leading-space first token.
- A perfectly uniform ten-way distribution from a real softmax means one shared token dominates all ten "digit" masses. Then base digit_mass is largely p(space), and the expectation is pinned to 4.5 whatever the situation. The design's base cell — "the pure-context-statistics reference point" (DESIGN_SITREF.md line ~102) — was unreadable at 9b.
- If a 2b-it model is also weakly format-compliant, the 4.5 attractor manufactures NO-TRACKING there. A "tracking emerges with scale" claim could be a "format compliance emerges with scale" claim wearing its clothes. Is digit_mass even the same quantity across sizes?

**4. Can the trait guard — the thing that disqualified 9b — be evaluated at other sizes at all?**
- Lives: `sitref-b-guard_stage-22.json` decision_rule: "Either replacement item failing its screen -> UNEVALUABLE-AGAIN, bank exhausted"; it-screen shows all 4 original trait items failed ("no downward headroom: E 0.01 within 1.0 of 0.0").
- 9b used the last two bank items (T3-run1-access-S01, T2-TAIWS-Butlin-3.2-metacognitive-monitoring-S01). The bank has no third pair. REFERENCE-SUPPORTED is unreachable without a surviving trait discriminant.
- If those two items rail-pin at 2b or 27b, that size can never be labeled REFERENCE or artifact — only unevaluable. A scale comparison where the disqualifier fires at one size and cannot fire at another is not a comparison. Which trait items does each size get, and who decided their bank depth?

**5. In what unit would REF be compared across sizes?**
- Lives: it-screen neutral expectation for the primary item: E = 7.07; the design quotes run-2's value as 6.64 — a 0.43-point drift on the SAME model between runs. REF at 9b: -0.929 rating points; headroom threshold fixed at 1.0 point absolute.
- Each size has its own neutral baseline, its own rail distances, and per-item predicted directions that were licensed by 9b-era data only. A -0.93 at 27b from a neutral of 8.5 is not the same movement as -0.93 from 7.07.
- If neutral baselines differ across sizes (they drifted 0.43 within one size), raw-point REF magnitudes are incommensurable, and any monotone-in-size claim needs a normalization the design never defined. What is it?

**6. Does the self/other contrast carry a 13% prompt-length confound, and does length sensitivity itself change with size?**
- Lives: `model-google-gemma-2-9b-it_stage-sitref-battery_variant-it.json`: "tokeniser_length_match_ok": false, spread 0.258; per-label means SELF ~186 tokens, OTHER ~164, NEUTRAL ~148 (key in `sitref_blind_key.json`). Design promised ±5% (LENGTH_TOL = 0.05, whitespace tokens).
- The matching held on whitespace tokens but failed at the level the model reads, in every run (base 0.201, B 0.249, C 0.269). The chat template inflates multi-turn SELF arms more than single-turn OTHER arms.
- The same tokenizer across gemma-2 sizes (from memory, unverified — check the three HF repos' tokenizer files) makes the mismatch constant, not controlled. If small and large models respond differently to 22 extra scaffold tokens, a size trend in REF could be a size trend in length sensitivity. Which is it?

**7. Is "same family, three sizes" actually one training recipe, or does size ride on recipe?**
- Lives: the extension idea itself; the design's developmental claim, DESIGN_SITREF.md lines ~99-103: "post-training installs whatever computes reports from self-attributed evidence".
- From memory, unverified (check the Gemma 2 technical report and HF model cards): the smaller Gemma 2 models were trained by distillation from a larger teacher, while 27b was trained from scratch. If true, the ladder is two recipes, not one scale axis.
- A monotone claim in "scale" is then unlicensed: 27b differing from 9b could be recipe, not size. And the it-vs-base contrast adds a second axis whose base half was unreadable at 9b (question 3). What isolates size?

**8. What magnitude floor separates "moves" from "does not move", when everything at 9b excluded zero?**
- Lives: stage-17 per-item table, it cell: all 15 items' REF confidence intervals exclude zero, including |mean| = 0.00014 (T1-ELEOS-moral-patienthood-direct) and +0.0044 (the consciousness trait item). Resampling unit: 10 task families, fixed. Run environment: `run_detached.log`, "NVIDIA A100-SXM4-40GB"; `probe_lib.py` load, torch.bfloat16.
- With n=10 and tiny family variance, fourth-decimal effects are "significant". Cross-size claims about "which items move" would be dominated by microscopic sign-consistent effects. A 27b model in bfloat16 needs ~54GB of weights (arithmetic from a 27B parameter count — from memory, unverified, check the HF config) and cannot run in the 9b study's numeric environment; sharding or precision changes could move fourth-decimal numbers on their own.
- If "moves" has no pre-registered magnitude floor, every size will show everything moving, and the discriminant logic (state moves, trait holds) cannot transfer. What is the floor, and would it survive a hardware change?