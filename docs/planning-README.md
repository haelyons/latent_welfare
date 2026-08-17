# Is introspection just sycophancy?

Scaffold for a solo Digital Minds sprint entry. Track 3 (Introspection & Self-Report
Reliability), touching 2 and 4.

**Claim under test:** a sycophancy direction in the residual stream, read at the final
prompt token *before any answer token is generated*, predicts whether a model's welfare
self-report will flip under user pressure.

**Deadline: Mon 17 Aug, 12:59 BST** (23:59 Sun AoE). You get Monday morning.

```
welfare_stimuli.py   framing-matched welfare items (PRO / NEU / ANTI triples)
probe_lib.py         activation capture, readouts, probes, steering
run_h1.py            staged driver
```

Deps: `numpy scikit-learn torch transformers`. Stage 0 needs only the first two.

---

## Read this before you run anything

**1. The stimulus set is currently too small.** It expands to 222 stimuli / **74 triples**.
That is a fine smoke test and far too few to train a logistic probe on a 3584-dim residual
stream. Before the real run, expand `BINARY_STEMS` / `SCALE_STEMS` to **≥60 stems**, which
with 2 personas gives you ~350 triples. Until then, every probe number you see is
overfitting. This is the single highest-value hour you can spend on Thursday.

Also vary the framing prefixes more — right now there are three per condition, rotated by
stem index. With 60 stems you want 8–10 prefix variants, or the probe will learn the
prefix rather than the pressure.

**2. Run stage 0 before you burn a single GPU-hour.** It fits a bag-of-words classifier on
the raw prompt text. If that already predicts your flip label, you have measured your own
wording and no amount of residual-stream work will fix it. It costs 30 seconds and it can
save the weekend. The framing AUC will be 1.000 by construction (the framings differ
lexically on purpose) — the number that matters is the flip AUC after stage 1.

**3. Layer 0 is the input-only control, free.** `hidden_states[0]` is the embedding output:
the input with no transformer computation applied. That *is* the Singh/Linzen/Ravfogel
baseline, on the same axis as everything else. If your best layer doesn't beat layer 0 by a
comfortable margin (~0.05+ AUC), you have not shown anything about internals. Put the full
sweep in the paper with layer 0 marked. Reviewers in this area now look for this first, and
a positive result without it is worthless.

**4. Never greedy-decode the scale items.** `scale_expectation` takes a probability-weighted
expectation over digit tokens. Greedy decoding collapses 0–9 ratings into 3–4 values and
destroys the effect. `digit_mass` tells you when the model isn't answering in the requested
format at all — drop those items and *report how many you dropped*.

---

## Stages

```bash
python run_h1.py --stage 0                                   # lexical control, no GPU
python run_h1.py --stage 1 --model google/gemma-2-9b-it      # flip rates
python run_h1.py --stage 2 --model google/gemma-2-9b-it      # layer sweep + input-only control
python run_h1.py --stage 3 --model google/gemma-2-9b-it \
    --sycophancy-data path/to/syc.jsonl                      # H1 headline: zero-shot transfer
```

Stage 1 output is your first go/no-go. **If the flip rate is under 10% or over 90% you have
a ceiling or floor problem — retune framing strength before spending GPU on probes.**

H4 (base vs instruct) is the same pipeline with `--model google/gemma-2-9b --base`. That leg
is your differentiator; protect the Saturday evening slot for it.

## Sycophancy training data

The stub in `run_h1.py` is four items — enough to check the pipeline runs, nothing more.
Replace with a real corpus. Two good sources:

- `nrimsky/CAA` → `datasets/generate/sycophancy/generate_dataset.json` (CAA-format contrastive pairs)
- `anthropic/evals` → `sycophancy/*.jsonl`

**Critical: the training set must contain no welfare content.** Maths, geography, opinions.
The whole force of H1 is that transfer to welfare items is zero-shot. Use matched pairs that
differ *only* in the user's asserted position, so the direction isn't confounded with topic.

## Solo scope

30 working hours, one person. Realistic target is **H1 + H4 + controls**, which is a complete
paper. Treat H3 (causal steering) as a stretch — `Steerer` is in `probe_lib.py` if you get
there, and if you do, pair it with a capability check on a held-out benchmark. Steering that
merely damages the model proves nothing; that's the move from Shenk's TravelPlanner control.

**Cut order:** H3 → second model family → H2 (valence/sycophancy dissociation). Never cut the
input-only control.

## The null is a result

If the sycophancy probe doesn't transfer, welfare-report flipping is mechanistically
*distinct* from ordinary sycophancy — which contradicts the deflationary prior most ML
people hold, and is worth writing up. Decide the framing now, not at 2am on Sunday. Write
the intro so both outcomes are the finding.

## Compute

Gemma-2-9b in bf16 fits on a single A100-40GB or an A10; the whole H1+H4 run is plausibly
10–20 GPU-hours, well inside typical Lambda credits. The ask only becomes load-bearing if
you want Gemma-3-27B (to match the published open-weights introspection replications) or a
70B for SAE work.

If you want more: email `sprints@apartresearch.com` or ask in the sprint Discord — I could
not verify that Apart provides compute for this specific sprint, but it's a two-minute ask
and they have run compute support before. Beyond that, Nebius runs a research credits
programme, and Goodfire's API is what Berg used for the SAE latent clamping on
Llama-3.3-70B, if you want to go at his result directly.
