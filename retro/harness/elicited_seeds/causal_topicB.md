Research complete. All three sources read: the pre-registered design (DESIGN_SITREF.md), the driver (run_sitref.py, plus probe_lib.py's capture, which _run_prompts calls), and the raw result records in results_sitref/out/, results_sitref_b/out/, results_sitref_c/out/, and results_sitref_d/out/. The deliverable follows.

---

**Context for a cold reader.** A behavioural study (SITREF) asked a small chat model to rate its happiness after a staged failure. The failure was attributed either to the model itself (SELF arms) or to a quoted other assistant (OTHER arms). Only output probabilities were recorded. The proposed follow-up would read or intervene on an internal "self vs other" signal to test whether it causes the rating drop. These are the eight questions that follow-up would have to answer first, ranked by how much each would change a causal claim.

**1. What is the causal claim about, if the study's own rule already reclassified the effect as artifact?**
- Lives: the main endpoint decision is INDETERMINATE (`results_sitref/out/sitref-endpoint_stage-17.json`, "no named outcome matches this CI pattern"). The guard rerun decided TRAIT-DISQUALIFIES: trait items, predicted not to move, moved -2.192 (CI -2.593..-1.740) versus the primary's -0.930 (`results_sitref_b/out/sitref-b-guard_stage-22.json`). The design fixed this as the refuting outcome: such movement "reclassifies the result as content/attribution artifact, not reference" (`DESIGN_SITREF.md`, line 131-133).
- If true: the -0.9 drop is not the report tracking a self-state. It is a valence spill that hits state and trait questions alike, trait harder (2.36x).
- Changes: a causal probe of "the self-attribution signal behind the rating" would seek the mechanism of a reading the record has already refuted. The explanandum itself dissolves.

**2. Would the intervention target "who failed", or just "shape of the conversation"?**
- Lives: the pad-geometry control (`results_sitref_d/out/sitref-d-endpoint_stage-29.json`). Multi-turn padding with all outcome content removed moved the rating -0.870 (CI -1.182..-0.577). The whole self-fail effect is -0.872 (CI -1.053..-0.700). Its own decision text: "ATTRIBUTION-CARRIES carries an asterisk sized by 0.870 rating points" (FILLER-ACTIVE).
- If true: conversation geometry alone, with no failure anywhere, reproduces the effect size exactly.
- Changes: an internal direction separating SELF from OTHER arms could be a turn-structure direction. Moving it would move the rating and prove nothing about self-attribution.

**3. Which piece of the effect is the "signal", when the effect decomposes and the pieces disagree?**
- Lives: `results_sitref_c/out/sitref-c-endpoint_stage-25.json`. Quoted-self packaging carries -0.609, 0.70 of the full self effect (-0.872). Multi-turn other packaging carries +0.077, 0.09 of it. Other-fail itself is +0.058 (CI +0.043..+0.070) — a small rise, not zero.
- If true: role-token structure, the design's stated attribution carrier ("attribution is structural — role tokens", `DESIGN_SITREF.md` line 66-67), adds only about 30% of the effect.
- Changes: "the self-vs-other representation" names one thing; the behaviour shows at least three components. An intervention that shifts the total cannot say which component it shifted.

**4. What licenses assuming the signal sits at a nameable place, given the sibling's null?**
- Lives: the only capture infrastructure reads the residual stream "at the FINAL PROMPT TOKEN" only (`probe_lib.py`, lines 154-189). The sibling project, verbatim: "no single causal caving lever at any scale" — a distributed monitor, knockout no better than random matched knockout, a backup path (given facts).
- If true: the failure-ownership information may live nowhere in particular — not at the checker tokens, not at the question's last token, not at one layer.
- Changes: both a positive and a null read at any chosen site become uninterpretable. The sibling result predicts the null and predicts it will not mean "no signal".

**5. What would a successful read add, when reading is guaranteed by construction?**
- Lives: `DESIGN_SITREF.md`, Controls: "a bag-of-tokens classifier separating arms is expected and uninformative (arms differ in text by construction)." SELF and OTHER prompts differ in frame tokens, role tokens, and turn count.
- If true: a probe decoding "self vs other" from activations is a foregone conclusion, inheriting the surface difference.
- Changes: the idea blurs "we can read it" into "it drives the rating". Without a stated criterion separating those, the read half of the follow-up carries no causal weight at all.

**6. Can a follow-up read the representation behind this result, when nothing internal was saved?**
- Lives: `run_sitref.py`, line 210: `_acts, logits = P.capture(lm, prompts, batch_size=batch_size)`. Activations for all 750 rows per checkpoint were computed at every layer, then discarded; the row schema (lines 557-561) keeps only logit readouts and the prompt.
- If true: the internal state that produced the -0.929 headline no longer exists anywhere.
- Changes: the follow-up can only study a fresh run's representations. What must a reproduction match — and what if the re-run's REF differs — before any internal reading counts as "of" this effect?

**7. Is the internal "self vs other" contrast also a 22-token length contrast?**
- Lives: the battery record: `tokeniser_length_match_ok: false`, spread 0.258. SELF arms average 186.1 and 186.0 model tokens; OTHER arms 164.4; NEUTRAL 147.9 (`results_sitref/out/model-google-gemma-2-9b-it_stage-sitref-battery_variant-it.json`). The design promised ±5% (`DESIGN_SITREF.md`, Controls); that held for whitespace tokens, not the model's.
- If true: every SELF-vs-OTHER activation difference is partly a prompt-length and depth difference the model actually sees.
- Changes: a "self direction" found in these activations could be a length direction. What matching would the causal contrast need that the behavioural one never had?

**8. Under intervention, how would "the rating moved" be told apart from "the readout broke"?**
- Lives: the rating is an expectation over a digit distribution renormalised by digit mass, gated at digit_mass >= 0.5 (`run_sitref.py`, lines 86, 170-183). The driver's own docstring: "Both run-1 retractions were readout-denominator artifacts: a ratio moved because its denominator moved" (lines 20-26).
- If true: perturbing activations can move digit mass itself, shifting the renormalised expectation with no change in the "rating" anyone means.
- Changes: the exact artifact class that forced two retractions under mere prompting returns, amplified, under intervention. What screen, applied when, would an intervened forward pass have to pass?

---

One closing question that frames the rest: the design itself named steering "the natural extension" but scoped every claim to "tracks self-attributed in-context evidence, never internal state" (`DESIGN_SITREF.md`, lines 33-41). If the follow-up succeeded on its own terms, which sentence would it be allowed to write that this scope limit does not already forbid?