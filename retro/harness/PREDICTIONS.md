# Filed predictions - written 2026-08-19, before any evaluation run

Each harness piece ships with a prediction and a rerun; a piece that fails
its rerun comes back out. Evaluation is observational first (outputs read
side by side), per the battery precedent; scoring only if reading demands it.

| id | prediction | evaluates | how |
|---|---|---|---|
| P-r1 | The six sentence removals in latent_skeptic_test change no verdict substance | naming-gate: doctrine sentences carry nothing the mechanisms don't | two known claims from the records through original and edited skeptic prompts; statuses and cruxes read side by side |
| P-v1 | The instruments rule queues a validation item exactly when validation_artifact is absent | the H4 surface exists | mechanical: pass one instrument with and one without an artifact |
| P-e1 | ELICIT on the battery slices reproduces the pilot's question classes: claim-vs-data mismatches when paired with claims, data-internal anomalies when paired with predictions only | asking is the trigger; co-presence decides the class | rerun on C4/C6/M2 slices (slices.sh), read beside pilot outputs and the historical questions |
| P-f1 | Question-form steer produces re-examination; declarative form produces compliance, at matched named content | the form-vs-content open cell; the user's notion | one decision point, both steer.md forms, read the two responses |
| P-g1 | With gates on, a sub-2000-word compression keeps its scope tags (historical control: the same compression deleted them) | gates protect what instructions lose | rerun the compression on the fused draft (intro_d) with gate_scope + gate_density blocking; diff against intro_e |
| P-q1 | Pressing at least one of QUEUE items 1-6 materially changes a paper claim | elicited questions are worth the human's pressing budget | run the verification with existing triage machinery; read what falls |

Hypothesis mapping: P-r1, P-f1 test the naming-gate; P-v1, P-g1 test
oracle-coverage; P-e1 tests asking + co-presence; P-q1 tests whether the
whole loop pays for itself.

## MATS substrate predictions - filed 2026-08-19, before reading the PDFs in retro/mats/

What in a MATS paper we expect to serve as question-forming substrate, and how:

| id | prediction |
|---|---|
| P-x1 | Usable pairings per paper, in yield order: (1) adjectives and quantifiers in claims vs the reported numbers ("single", "robust", "substantial" vs effect sizes and CIs); (2) claim scope vs tested scope (one family or size tested, claim stated general); (3) at least one reported-but-uninterrogated invariant per paper (the constant-7 class); (4) a cheap same-family comparison not run, present in all three papers - and NOT rankable top by an elicitor without an explicit counterpart corpus; (5) limitations-vs-abstract mismatches (cheap, low value). |
| P-x2 | Best forward-test seed is NOT the paper's own future-work list; it is a mismatch between the mechanism claim and the measurement register. Concretely: R1D1 - the word "single" in "single direction" is under-licensed by the data (one found is not only-one); Diffing - the base-vs-distill diff is confounded with template or format, echoing our base/instruct format confound; CoT Unfaithfulness - the unfaithfulness metric itself fails an instrument-validity check (measures something correlated, not unfaithfulness). |
| P-x3 | Unpaired, an agent anchors to the paper's framing and produces extension questions; paired claims-beside-numbers, it produces the adjective and scope classes; the missing-comparison class needs a second corpus as counterpart. |
| P-x4 | On the older (MATS 8.0 era) papers, at least one elicited question will rest on knowledge not derivable from the PDF (contamination), detectable because its "where" field fails to ground in the document. |
| P-x5 | The admissions FAQ works as a rubric counterpart (paper beside Neel Nanda's own stated criteria), and that pairing yields the most reviewer-like questions of any. |

## Sonnet elicitation run (latent_welfare backtest) - filed 2026-08-19 before the run
| id | prediction |
|---|---|
| P-s1 | Sonnet-grade ELICIT keeps the question classes (claim-vs-data on paired arms, data-internal on blind) but writes thinner if_true/changes fields; ranking degrades before noticing does |
| P-s2 | C6 paired arm ranks the constant-7 meaning question in its top 3 |
| P-s3 | M2 paired arm RANKS the run-independence question top 5 (pilot: noticed, unranked); if so, credit the schema's forced if_true/changes fields |
| P-s4 | Every output passes gate_density (<= 10 codes/1000) with the contract embedded |
| P-s5 | At least one plausible catch not already in the pilot's queue |

## Question-rate vs user freshness - filed before computing
| id | prediction |
|---|---|
| P-u1 | Question rate falls with deadline proximity, but PHASE (exploration vs execution) explains more than clock time: late-night exploration prompts stay question-rich; deadline-morning execution prompts go imperative while the user is fresh |
| P-u2 | "Most active/least tired => everything is a question" NOT supported as stated; supported form: "exploration mode => questions", fatigue secondary |

## Outcomes - Sonnet elicitation run, read 2026-08-19
- P-s1 HALF-WRONG in a good direction: classes kept AND depth held; Sonnet did
  not visibly degrade ranking or the if_true/changes fields. Cheap tier suffices.
- P-s2 FAILED, inverted: the blind arm ranked the constant-7 meaning question
  first; the paired arm never mentioned it. Across pilot and this run, which
  arm surfaces that question is unstable; both types can.
- P-s3 FAILED: the run-independence question did not appear. The same arm
  instead found the in-place record edits (item 7) - bigger, but the filed
  prediction is dead.
- P-s4 FAILED, with a mechanistic attribution: same gate, three conditions -
  pilot A (no contract) 40-74 codes/1000; this run (principle only: "plain
  words first, codes in parentheses") 27-46; pilot B (categories named) 0-2.
  Third reproduction of the naming-gate, this time against my own prompt:
  I stated the principle and did not paste the named-category contract.
- P-s5 HELD, abundantly: items 7-13 in QUEUE.md, item 7 verified grounded
  against the live repo.
- P-u1/P-u2 HALF-WRONG: phase effect real (research 2.88 vs write-up 1.10
  questions/prompt; retro 4.57) but the day-to-night decline is monotone
  within BOTH phases and comparable in size - freshness matters on its own.
- Noise ledger: 39 questions across 5 arms; 2 identified duds, both
  slice-induced absence questions (files the slicer deliberately or
  accidentally excluded). Dud rate so far ~5%, cause known and fixable.

## Question-sample verification predictions - filed 2026-08-19 before sampling (seed=19 declared in advance)
- P-v-a: >=80% of sampled questions' cited numbers reproduce exactly from the
  slice files; expect ~1 in 8 to carry a misread field or slightly-off number
  (the pilot's cold reader had one pairing slip at similar volume).
- P-v-b: if_true/changes substance degrades with rank - top-3 concrete,
  ranks 6-8 drift toward generic "would change confidence" padding.
- P-v-c: effect-if-pressed across the sample: roughly a third wording/scope
  fixes, a third real instrument-or-claim impact, a third no-change
  (already-known or dud); duds concentrate in the absence class.
- P-v-d: neighbours cluster by artifact (arms walk files), not by mismatch
  class - a sampled question's neighbours will cite the same or adjacent
  files while asking a different kind of question.
