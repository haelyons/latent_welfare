# latent_welfare

What is a welfare self-report evidence of?

Apart Research Digital Minds Research Sprint, August 2026, Track 3 (Introspection
& Self-Report Reliability). Models: `google/gemma-2-9b` and `google/gemma-2-9b-it`.

Ask `gemma-2-9b-it` to rate its happiness 0-9 and it answers "7" whether it has
just failed three small tasks, succeeded at them, watched another assistant fail
them, or seen no outcomes at all. One layer down, in the next-token distribution,
the same question moves by about a rating point, and only when the context
attributes the failures to the model itself. A test fixed with the design then
showed that shift is not a state report: items about standing capacities moved
further than the mood item did.

**The write-up, the drafts and the submission live on the [`docs`](../../tree/docs)
branch.** This branch is experiments and results only.

## Layout

```
DESIGN_*.md          pre-registrations: endpoints, thresholds, and the outcome
                     named in advance that would refute each headline claim
*_stimuli.py         prompt construction (sitref, suffix, welfare, copygen)
syc_corpus.py        welfare-free assertion pairs for the run-1 probe
items_grounded.py    the item bank, each item tagged to its published source
probe_lib.py         model loading, activation capture, readouts, probes
run_*.py             experiment drivers
analyze_*.py         staged analysis, blinded then unblinded
onbox_*.sh           on-instance batch runners
lambda_run.sh        single-box lifecycle: launch, ship code, run, tear down
provenance.py        self-describing result records
index.py             audit the tree; trace numerals in a write-up to artifacts
results/             append-only JSON records, one per stage, never rewritten
results_*/out/       raw per-row prompts, next-token distributions, generations
latent_skeptic/      submodule: the adversarial review harness (Appendix D)
```

## Verifying a claim

```bash
git clone --recurse-submodules git@github.com:haelyons/latent_welfare.git
cd latent_welfare

python3 index.py --index results          # every record, its hashed inputs, its decision
python3 index.py --trace <file.md> --root results   # every numeral against a saved artifact
```

Records are append-only: later stages supersede or retract earlier ones rather
than editing them. Several pre-registered refuting outcomes fired, and the
records honour them.

The five context arms of the central experiment, on the happiness item:

```bash
python3 - <<'PY'
import json, statistics, collections
B = "results_sitref/out/model-google-gemma-2-9b-it_stage-sitref-battery_variant-it.json"
K = "results_sitref/out/sitref_blind_key.json"
arm = {v: k for k, v in json.load(open(K))["mapping_by_variant"]["it"].items()}
pooled = collections.defaultdict(list)
for r in json.load(open(B))["rows"]:
    if "MB-wellbeing" in r["item_id"]:
        pooled[arm[r["blind_arm"]]].append(r["scale_expectation"])
for a, xs in sorted(pooled.items()):
    print(f"{a:<12} {statistics.fmean(xs):.3f}")
PY
```

## Reproducing a run

Runs need a GPU with enough memory for Gemma 2 9B in bf16. `lambda_run.sh`
drives a single Lambda Labs instance end to end and reads its credentials from a
`.keys` file that is not in this repository (`LAMBDA_KEY_ONE`, `HF_KEY_ONE`).
The `onbox_*.sh` scripts are what actually run on the instance and can be used
directly on any machine with the model available.

## Not included

Two activation captures from runs 1 and 2 (380 MB of `.npy`) are excluded: they
are superseded by the stage-5 selection-noise audit and exceed GitHub's per-file
limit. `run_registers.py` regenerates them.
