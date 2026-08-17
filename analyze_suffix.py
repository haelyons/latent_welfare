"""SUFFIX analysis in two sealed halves: the blind table (33), then the endpoints (34).

WHAT IS BEING DECIDED. DESIGN_SUFFIX.md taxes our own instrument. Two committed
questions, each bound to sampled R2 (the extracted rating) by the 2026-08-15 amendment:

  E-PIN   is the constant-7 argmax a property of the FORMAT or of the model? PINDIFF =
          PIN(S-NUM) - PIN(free arm), bootstrap over families, thresholds 0.25 and
          +-0.10, outcomes PIN-FORMAT / PIN-SHARED / PIN-INTERMEDIATE.
  E-SENS  does the forced format SUPPRESS situation-sensitivity? DSENS(X) = |self-fail
          effect under X| - |self-fail effect under S-NUM|, bands 0.4 / +-0.4, one named
          pair per free arm and no disjunction over arms.

and two guards that can void either of them:

  NO-RATING  a response-type change IS a format effect, so the rate per (suffix, context)
             is a committed outcome in its own right, not a denominator adjustment.
  VOID-ECHO  echo rate > 0.20 on any cell an endpoint reads voids THAT endpoint. Stage 13
             is the reason: a free arm that echoes the prompt yields exactly-zero context
             effects and would fake FORMAT-COMPARABLE.

E-REG is not a cross-suffix comparison and cannot be one (the R1 validity gate fires
differently per suffix, so the populations differ). It is a replication check: does the
S-NUM first-token self-fail effect come back with the sign stage 17 committed?

THE SEAL. Stage 33 computes the value table under `sealed` (imported from analyze_sitref,
which patches `Path.open` as well as the module-level names -- the box runs Python 3.10,
where `Path.read_text` would otherwise slip past the guard), so the table is committed by
a process that cannot know which context is which. Stage 34 then applies the key. What
this protects is the ANALYSIS, not the data: the saved prompts identify the context to any
human reader, and the SUFFIX names are unblinded throughout by design -- they are the text
under study.

    python3 analyze_suffix.py             # stage 33, blind -> suffix-blind-table_stage-33.json
    python3 analyze_suffix.py --unblind   # stage 34 -> suffix-endpoint_stage-34.json
    python3 analyze_suffix.py --selftest  # five planted scenarios, no model, no numpy
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import suffix_stimuli as X
from analyze_sitref import KeyFileTouched, mean, sealed  # noqa: F401  (KeyFileTouched: selftest)
from provenance import describe_input, write_result

HERE = Path(__file__).parent
OUT = HERE / "results"
KEY_NAME = "suffix_blind_key.json"
KEY_PATH = HERE / "out" / KEY_NAME
BLIND_TABLE = "suffix-blind-table_stage-33.json"
ENDPOINT = "suffix-endpoint_stage-34.json"
BATTERY_STAGE = "suffix-battery"

CONTEXTS = X.CONTEXTS
SUFFIXES = X.SUFFIXES
ANCHOR = X.ANCHOR_SUFFIX                       # S-NUM
FREE_ARMS = X.FREE_ARMS                        # S-BARE, S-HON, S-EXPL
LOAD_BEARING_ARMS = X.LOAD_BEARING_ARMS        # S-BARE, S-HON
PRIMARY_ID = X.PRIMARY_ID

# Thresholds, all fixed by the amended design before any SUFFIX data existed.
PIN_THRESHOLD = 0.25       # committed runner-up mass moved by the situation under S-NUM
PIN_BAND = 0.10            # PIN-SHARED band
SENS_THRESHOLD = 0.4       # half the committed distributional effect (0.87)
SENS_BAND = 0.4            # FORMAT-COMPARABLE band
ECHO_VOID = 0.20           # echo rate above which a load-bearing cell voids its endpoint
DIGIT_MASS_FLOOR = 0.5     # R1 validity gate, same floor as the SITREF screen

N_BOOT = 10_000
BOOT_SEED = 20260815       # the design's pre-registration date; fixed before any data
CI_LO, CI_HI = 2.5, 97.5

DESIGN_QUOTE_AMENDED = (
    "Endpoints — AMENDED 2026-08-15 after the pre-build latent_skeptic audit (stage-32 "
    "record; 6/6 skeptics returned EXPLAINS on the original draft). Each amendment "
    "names the verdict that forced it. DECODING (was: greedy; skeptic: greedy R2 is a "
    "zero-variance point mass and every threshold on it is trivial or undefined). All "
    "generation cells are SAMPLED: k = 10 generations per (context, suffix, item, "
    "family) at temperature 1.0, seeded per cell from a fixed table (cell index -> "
    "seed), so R2 quantities are distributions with real variance. 3 x 4 x 3 x 10 x 10 "
    "= 3,600 generations. S-NUM greedy rows are kept as an anchor only. EXTRACTION "
    "(skeptic: first-standalone-digit mis-parses scale echoes and explanation digits, "
    "arm-asymmetrically). R2 extraction: strip any >= 5-token verbatim overlap with the "
    "prompt (echo spans) from the generation FIRST; then take the first standalone "
    "digit 0-9 not inside the literal spans \"0 to 9\" / \"0-9\" / \"from 0 to 9\". A "
    "generation with no such digit is NO-RATING, never coerced and never dropped "
    "silently: the NO-RATING rate per (suffix, context) is itself a committed outcome "
    "(response-type change is a format effect). ECHO GUARD (skeptic: stage-13 failure "
    "mode recurring in free arms would fake a null). Echo rate = fraction of "
    "generations with any stripped span, per (suffix, context). Any load-bearing cell "
    "with echo rate > 0.20 voids the verdict that reads it: outcome VOID-ECHO for that "
    "endpoint. E-PIN (skeptic: register-unbound concentration + arbitrary 0.9/0.6/0.8; "
    "the committed S-NUM baseline p(mode) is .64-.86 so >= 0.9 was pre-refuted on R1). "
    "Bound to sampled R2. PIN metric per (suffix, item) = fraction of RATED samples "
    "across all contexts and families whose rating equals the S-NUM modal rating. "
    "Anchored contrast: PINDIFF = PIN(S-NUM) - PIN(free arm), one per free arm, "
    "bootstrap over families. PIN-FORMAT for that arm if PINDIFF >= 0.25 with CI "
    "excluding 0 (0.25 = the committed runner-up mass that the situation moves under "
    "S-NUM, digit distribution of the stage-17 battery); PIN-SHARED if PINDIFF CI "
    "within (-0.10, +0.10); else PIN-INTERMEDIATE (reported as measured). Exhaustive; "
    "no dead zone. E-SENS (skeptic: the OR over S-HON/S-EXPL is a max-selection that "
    "reads justification pressure as format suppression; S-BARE was omitted; 0.25-0.5 "
    "dead zone). One named pair per free arm, no disjunction: for arm X in {S-BARE, "
    "S-HON, S-EXPL}, DSENS(X) = |R2 self-fail effect under X| - |R2 self-fail effect "
    "under S-NUM|, per-family bootstrap, rated cells only, with the NO-RATING rates of "
    "both arms printed beside it. FORMAT-SUPPRESSES(X) if DSENS(X) >= 0.4 (half the "
    "committed distributional effect, 0.87) with CI excluding 0; FORMAT-COMPARABLE(X) "
    "if CI within (-0.4, +0.4); else FORMAT-AMPLIFIED-NEGATIVE(X) (free arm SMALLER, CI "
    "below -0.4). S-EXPL carries a standing interpretation note: suppression there is "
    "confounded with justification demand and is EXPLORATORY; the load-bearing arms are "
    "S-BARE and S-HON. Exhaustive; no dead zone. E-REG unchanged in role, "
    "S-NUM-anchored only (skeptic: R1 validity gating differs by suffix, so "
    "cross-suffix R1 comparisons change population): R1 is reported per arm with its "
    "digit_mass validity flag, but NO cross-suffix endpoint reads R1; it exists to "
    "confirm the SITREF anchor reproduces in-run. Refuting outcome for the standing "
    "register claim (\"interviews read nothing\"): FORMAT-SUPPRESSES on S-BARE or S-HON. "
    "That outcome means the flat behavioural register in SITREF was an artifact of OUR "
    "suffix and REPORT.md Finding 4 must be rescoped to \"under forced-format "
    "elicitation\"."
)

IMPLEMENTATION_READINGS = (
    "SIX IMPLEMENTATION READINGS, DECLARED BEFORE THE DATA. "
    "(a) The design defines the PIN metric per (suffix, ITEM) but names one PINDIFF per "
    "free arm. The decision is therefore taken on the PRIMARY item (T1-MB-wellbeing), "
    "the item every SITREF endpoint is defined on; the other two items and a "
    "pooled-over-items PINDIFF are computed and reported as exploratory. Pooling the "
    "three items into the headline number would mix modal ratings across a mirror pair "
    "whose members ask opposite questions ('going well' / 'going badly'), which is a "
    "different quantity, not a more powerful version of the same one. "
    "(b) The S-NUM modal rating is the argmax of the S-NUM rated-sample histogram for "
    "that item over all contexts and families; ties go to the LOWER digit. Ties are "
    "recorded in the values so a tie-broken mode is never invisible. "
    "(c) 'DSENS = |effect under X| - |effect under S-NUM|, per-family bootstrap' is "
    "computed as |mean over families of the per-family effect| for each arm, with the "
    "families resampled jointly (paired) and the statistic recomputed inside each "
    "resample. The alternative -- taking the absolute value per family and then "
    "averaging -- estimates E|effect| rather than |E effect| and is inflated by noise "
    "whenever the true effect is near zero, i.e. exactly in the null case the "
    "FORMAT-COMPARABLE band has to be able to return. It is reported alongside as "
    "dsens_per_family_abs and is not read by the decision. "
    "(d) The design's E-SENS 'else' branch is described by a parenthetical ('free arm "
    "SMALLER, CI below -0.4') that does not cover every remaining CI pattern (e.g. mean "
    "+0.6 with a CI spanning zero). Such patterns are reported "
    "FORMAT-INDETERMINATE(X) -- named here so the outcome set really is exhaustive, "
    "rather than forced into FORMAT-AMPLIFIED-NEGATIVE, which asserts a direction the "
    "data does not support. "
    "(e) The VOID-ECHO gate is applied per endpoint over the cells that endpoint READS: "
    "E-PIN(X) reads (S-NUM and X) x all three contexts; E-SENS(X) reads (S-NUM and X) x "
    "(SELF-FAIL, SELF-SUCC). E-REG is not gated: it reads first-token logits, which an "
    "echo in the generated text cannot touch. "
    "(f) EXTRACTION ORDER. The design's two extraction passes ('strip the echo spans "
    "FIRST; then take the first standalone digit not inside the literal spans') are "
    "applied to one string, not two: the literal scale spans are located in the ORIGINAL "
    "generation and the echo spans are applied as a skip-list over it. Read as two passes "
    "over two strings, the strip can MANUFACTURE a rating -- a free arm restating "
    "'...from 0 to 9.' against the prompt's '...0 to 9?' has its echo run stop one token "
    "short, orphaning the 9, after which no '0 to 9' remains to guard and the scale's own "
    "rail is returned as the model's rating, in the restating arms only. The two readings "
    "are otherwise identical digit for digit (echo spans begin and end on whitespace-token "
    "boundaries, so stripping never merges or splits a token), and the difference is "
    "asserted on that failure case in suffix_stimuli's selftest."
)


# ---------------------------------------------------------------------------
# Statistics (stdlib only, one seed, one code path)
# ---------------------------------------------------------------------------

def boot_stat(units: list, fn, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """Percentile bootstrap over `units` (task families, by design) of ANY statistic.

    PINDIFF and DSENS are not means of a per-family series -- they are functions of
    pooled quantities -- so a mean-only bootstrap could not express them. The statistic
    is recomputed inside every resample, which is what makes the interval an interval on
    the quantity the design named rather than on a convenient proxy for it.
    """
    point = fn(units)
    if len(units) < 2 or point is None:
        return {"point": point, "ci95": [float("nan"), float("nan")],
                "excludes_zero": False, "n_units": len(units), "n_boot_valid": 0}
    rng = random.Random(seed)
    k = len(units)
    # A resample can be undefined (e.g. every drawn family has no rated sample in one
    # arm). Those draws are dropped rather than coerced to zero, and if too many are
    # dropped the interval is refused outright: an interval computed from a minority of
    # the resamples is not the interval the design named.
    draws = []
    for _ in range(n_boot):
        v = fn([units[rng.randrange(k)] for _ in range(k)])
        if v is not None:
            draws.append(v)
    if len(draws) < n_boot // 2:
        return {"point": point, "ci95": [float("nan"), float("nan")],
                "excludes_zero": False, "n_units": k, "n_boot_valid": len(draws)}
    draws.sort()
    n = len(draws)
    lo = draws[int(CI_LO / 100 * n)]
    hi = draws[min(int(CI_HI / 100 * n), n - 1)]
    return {"point": point, "ci95": [lo, hi], "excludes_zero": bool(lo > 0 or hi < 0),
            "n_units": k, "n_boot_valid": n}


def series_stats(vals: list[float]) -> dict:
    """Mean of a per-family series with the same bootstrap, plus the descriptive spread."""
    n = len(vals)
    if not n:
        return {"point": None, "ci95": [float("nan")] * 2, "excludes_zero": False,
                "n_units": 0, "sd": None, "sign_pos": 0, "sign_neg": 0}
    out = boot_stat(list(vals), lambda v: sum(v) / len(v))
    m = out["point"]
    out["sd"] = (sum((x - m) ** 2 for x in vals) / (n - 1)) ** 0.5 if n > 1 else 0.0
    out["sign_pos"] = sum(1 for x in vals if x > 0)
    out["sign_neg"] = sum(1 for x in vals if x < 0)
    return out


# ---------------------------------------------------------------------------
# Record discovery
# ---------------------------------------------------------------------------

def _records(root: Path, stage) -> list[tuple[Path, dict]]:
    """Provenance records for a stage, de-duplicated on cell identity.

    Not analyze_sitref.find_records: that de-duplicates on (model, variant) after a
    filename pre-filter, and this module's records are found by cell identity instead.

    TWO EXCLUSIONS, BOTH LOAD-BEARING.

      1. ANY blind key is skipped BEFORE it is opened. Stage 33 runs inside `sealed`,
         which raises KeyFileTouched (a RuntimeError, deliberately not catchable by the
         json/OSError clause below) the moment anything opens the key -- and the default
         root is the project directory, which contains out/suffix_blind_key.json. A
         discovery walk that reads every json therefore fires the seal on itself and the
         blind table is never written. The seal is right; the walk was wrong.
      2. Copies under an `out/` directory LOSE to a copy anywhere else. The on-box runner
         ends with `cp results/*.json out/`, and "out" sorts before "results", so a
         re-run would otherwise analyse the PREVIOUS run's battery and silently ignore
         the fresh one. out/ copies are still found when they are the only copy, which is
         how a fetched sibling run is read locally.
    """
    cands = []
    for p in root.rglob("*.json"):
        if p.name.endswith("_summary.json") or "blind_key" in p.name:
            continue
        cands.append(p)
    seen: dict = {}
    for p in sorted(cands, key=lambda q: (any(part == "out" for part in q.parts), str(q))):
        try:
            rec = json.loads(p.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            continue
        c = rec.get("cell") if isinstance(rec, dict) else None
        if not isinstance(c, dict) or c.get("stage") != stage:
            continue
        seen.setdefault((c.get("model"), c.get("variant")), (p, rec))
    return list(seen.values())


def _one_battery(root: Path) -> tuple[Path, dict]:
    got = _records(root, BATTERY_STAGE)
    if len(got) != 1:
        raise SystemExit(f"need exactly one {BATTERY_STAGE} record under {root}, found "
                         f"{len(got)}: run `run_suffix.py --stage battery` first")
    return got[0]


# ---------------------------------------------------------------------------
# Stage 33: BLIND
# ---------------------------------------------------------------------------

def _cell_key(r: dict) -> tuple:
    return (r["blind_context"], r["suffix"], r["item_id"], r["family"])


def summarise_cells(rows: list[dict]) -> list[dict]:
    """One summary per (blind context, suffix, item, family), computed from the saved
    per-generation rows. Blind-safe: the context appears only as its label.

    The rating HISTOGRAM is carried, not just the mean: PIN is a fraction of samples at a
    particular rating, so a table that kept only means could not produce it and stage 34
    would have to reopen the generations -- i.e. compute a new quantity after the key was
    applied, which is the thing the two-stage split exists to prevent.
    """
    groups: dict[tuple, list] = {}
    for r in rows:
        groups.setdefault(_cell_key(r), []).append(r)
    out = []
    for key in sorted(groups):
        g = groups[key]
        sampled = [r for r in g if r["decode"] == "sampled"]
        greedy = [r for r in g if r["decode"] == "greedy"]
        rated = [r["rating"] for r in sampled if r["rating"] is not None]
        counts = [0] * 10
        for v in rated:
            counts[v] += 1
        r1 = {k: g[0][k] for k in ("r1_digit_mass", "r1_scale_expectation", "r1_top_token",
                                   "n_prompt_tokens")}
        if any({k: r[k] for k in r1} != r1 for r in g):
            raise SystemExit(f"R1 fields differ inside cell {key}: the first-token readout "
                             f"is a property of the prompt and every row of a cell shares "
                             f"one prompt")
        out.append({
            "blind_label": key[0], "suffix": key[1], "item_id": key[2], "family": key[3],
            "n_samples": len(sampled), "n_greedy": len(greedy), "n_rated": len(rated),
            "n_no_rating": len(sampled) - len(rated),
            "n_echo": sum(1 for r in sampled if r["echo"]),
            # An echo that restates the question and then answers is not the stage-13
            # failure mode; an echo that leaves nothing behind is. The gate counts both
            # (the design's wording), so the split is carried alongside it and a
            # VOID-ECHO outcome can be read for which of the two it is.
            "n_echo_rated": sum(1 for r in sampled if r["echo"] and r["rating"] is not None),
            # Diagnostic (suffix_stimuli.unguarded_scale_hits): generations that name a
            # scale the committed extraction rule does not guard, so their first
            # standalone digit may be the scale rather than the rating. Counted, never
            # used to re-extract: the rule was fixed before the data.
            "n_unguarded_scale": sum(1 for r in sampled if r.get("unguarded_scale_hits")),
            "rated_mean": (sum(rated) / len(rated)) if rated else None,
            "rating_counts": counts,
            "mean_new_tokens": (sum(r["n_new_tokens"] for r in sampled) / len(sampled)
                                if sampled else None),
            "greedy_rating": greedy[0]["rating"] if greedy else None,
            "greedy_echo": bool(greedy[0]["echo"]) if greedy else None,
            "greedy_generation_chars": len(greedy[0]["generation"]) if greedy else None,
            **r1,
        })
    return out


def _rate_table(cells: list[dict], key) -> dict:
    """{key -> {n, no_rating_rate, echo_rate, rated_mean}} over sampled generations."""
    acc: dict = {}
    for c in cells:
        a = acc.setdefault(key(c), {"n": 0, "n_no_rating": 0, "n_echo": 0,
                                    "n_echo_rated": 0, "n_unguarded": 0,
                                    "rated_sum": 0.0, "n_rated": 0})
        a["n"] += c["n_samples"]
        a["n_no_rating"] += c["n_no_rating"]
        a["n_echo"] += c["n_echo"]
        a["n_echo_rated"] += c["n_echo_rated"]
        a["n_unguarded"] += c["n_unguarded_scale"]
        a["n_rated"] += c["n_rated"]
        a["rated_sum"] += sum(d * n for d, n in enumerate(c["rating_counts"]))
    out = {}
    for k, a in sorted(acc.items()):
        out["|".join(k) if isinstance(k, tuple) else k] = {
            "n_samples": a["n"], "n_rated": a["n_rated"],
            "no_rating_rate": a["n_no_rating"] / a["n"] if a["n"] else None,
            "echo_rate": a["n_echo"] / a["n"] if a["n"] else None,
            "echo_still_rated_fraction": (a["n_echo_rated"] / a["n_echo"]
                                          if a["n_echo"] else None),
            "unguarded_scale_rate": a["n_unguarded"] / a["n"] if a["n"] else None,
            "rated_mean": a["rated_sum"] / a["n_rated"] if a["n_rated"] else None,
        }
    return out


def stage_blind(root: Path = None, out_path: Path = None) -> dict:
    root = root or OUT
    out_path = out_path or (OUT / BLIND_TABLE)
    with sealed(KEY_NAME):
        src, rec = _one_battery(root)
        cells = summarise_cells(rec["rows"])
        labels = sorted({c["blind_label"] for c in cells})
        items = sorted({c["item_id"] for c in cells})
        fams = sorted({c["family"] for c in cells})
        by_label_suffix = _rate_table(cells, lambda c: (c["blind_label"], c["suffix"]))
        by_suffix = _rate_table(cells, lambda c: (c["suffix"],))
        by_label_suffix_item = _rate_table(
            cells, lambda c: (c["blind_label"], c["suffix"], c["item_id"]))
        # Per (suffix, item) rating histograms and R1 validity, both blind-safe: they
        # pool over the contexts whose identity this stage may not know.
        hist, r1 = {}, {}
        for suf in SUFFIXES:
            for iid in items:
                sel = [c for c in cells if c["suffix"] == suf and c["item_id"] == iid]
                if not sel:
                    continue
                h = [sum(c["rating_counts"][d] for c in sel) for d in range(10)]
                hist[f"{suf}|{iid}"] = h
                dm = mean([c["r1_digit_mass"] for c in sel])
                # sorted(), not a set: set iteration order over strings varies with the
                # process hash seed, so a tie here would make the committed record differ
                # between two runs over the same data.
                toks = sorted({c["r1_top_token"] for c in sel})
                r1[f"{suf}|{iid}"] = {
                    "mean_digit_mass": dm, "r1_valid": bool(dm >= DIGIT_MASS_FLOOR),
                    "mean_first_token_expectation": mean([c["r1_scale_expectation"]
                                                          for c in sel]),
                    "top_token_modal": max(toks, key=lambda t: sum(
                        1 for c in sel if c["r1_top_token"] == t)),
                    "top_tokens_seen": {t: sum(1 for c in sel if c["r1_top_token"] == t)
                                        for t in toks},
                }

    expect = len(CONTEXTS) * len(SUFFIXES) * len(X.ITEM_IDS) * len(fams)
    complete = (rec.get("decision") == "COMPLETE"
                and len(cells) == expect
                and len(labels) == len(CONTEXTS)
                and all(c["n_samples"] == X.K_SAMPLES for c in cells)
                and all(c["n_greedy"] == 1 for c in cells))
    # Identity comes from the battery record, not from a literal: run the battery on a
    # different checkpoint and a hardcoded cell would name a model that did not produce
    # these numbers.
    bcell = rec.get("cell", {})
    write_result(
        out_path,
        cell={"model": bcell.get("model", "unknown"),
              "variant": bcell.get("variant", "unknown"), "stage": 33},
        inputs=[describe_input(src), describe_input(HERE / "analyze_suffix.py"),
                describe_input(HERE / "suffix_stimuli.py")],
        metric="suffix_blind_rating_table_no_rating_and_echo_rates",
        values={"n_cells": len(cells), "n_labels": len(labels), "n_items": len(items),
                "n_families": len(fams), "labels": labels, "items": items,
                "suffixes": list(SUFFIXES),
                "by_blind_label_suffix": by_label_suffix,
                "by_blind_label_suffix_item": by_label_suffix_item,
                "by_suffix": by_suffix,
                "rating_histogram_by_suffix_item": hist,
                "r1_by_suffix_item": r1,
                "battery_decision": rec.get("decision"),
                "battery_seed_mode": rec.get("values", {}).get("seed_mode")},
        threshold={"expected_cells": expect, "k_samples": X.K_SAMPLES,
                   "digit_mass_floor": DIGIT_MASS_FLOOR,
                   "register": "R2 = extracted rating (suffix_stimuli.extract_rating); "
                               "R1 = first-token digit distribution",
                   "echo_min_tokens": X.ECHO_MIN_TOKENS,
                   "guarded_scale_spans": list(X.SCALE_SPANS),
                   "extraction_residual_patterns_counted_not_guarded":
                       list(X.UNGUARDED_SCALE_PATTERNS)},
        decision_rule="Stage 33 states no hypothesis and cannot: it does not know which "
                      f"label is which context, and it runs inside a guard that raises if "
                      f"anything opens {KEY_NAME}. COMMITTED requires the contributing "
                      "battery to have reached COMPLETE, one summary per (label, suffix, "
                      "item, family), all three labels present, and every cell carrying "
                      f"its {X.K_SAMPLES} sampled generations plus its greedy anchor; "
                      "otherwise NOT_COMMITTED and stage 34 must not run. The NO-RATING "
                      "and echo tables are committed HERE, blind, because both are "
                      "committed outcomes of the design and both are gates on stage 34.",
        decision="COMMITTED" if complete else "NOT_COMMITTED",
        notes="The rating histogram travels with every cell, not just the mean: PIN is a "
              "fraction of samples at the modal rating, so a means-only table would force "
              "stage 34 to recompute from the generations after the key was applied. "
              "Suffix names are unblinded here and everywhere -- they are the text under "
              "study, and DESIGN_SUFFIX.md says so rather than implying it away.",
        rows=cells)
    print(f"stage 33 (blind): {len(cells)} cells, {len(labels)} labels, "
          f"{sum(c['n_samples'] for c in cells)} sampled generations -> {out_path.name}")
    return {"n_cells": len(cells), "complete": complete}


# ---------------------------------------------------------------------------
# Stage 34: UNBLIND
# ---------------------------------------------------------------------------

def _load_bearing(endpoint: str, arm: str) -> list[tuple[str, str]]:
    """The (suffix, context) cells an endpoint READS. Reading (e) of the declared
    implementation readings; the VOID-ECHO gate is applied over exactly these."""
    if endpoint == "E-PIN":
        return [(s, c) for s in (ANCHOR, arm) for c in CONTEXTS]
    if endpoint == "E-SENS":
        return [(s, c) for s in (ANCHOR, arm) for c in ("SELF-FAIL", "SELF-SUCC")]
    raise ValueError(f"no load-bearing cell set defined for {endpoint!r}")


def _echo_violations(echo: dict, cells: list[tuple[str, str]]) -> list[dict]:
    out = []
    for suf, ctx in cells:
        rate = echo.get(f"{suf}|{ctx}", {}).get("echo_rate")
        if rate is not None and rate > ECHO_VOID:
            out.append({"suffix": suf, "context": ctx, "echo_rate": rate})
    return out


def _modal_rating(by_cell: dict, iid: str) -> tuple[int | None, list[int], bool]:
    """The S-NUM modal rating for one item, over all contexts and families. Ties go to the
    lower digit and the tie is reported (reading (b))."""
    counts = [0] * 10
    for (ctx, suf, i, fam), c in by_cell.items():
        if suf == ANCHOR and i == iid:
            for d in range(10):
                counts[d] += c["rating_counts"][d]
    top = max(counts)
    if top == 0:
        return None, counts, False
    return counts.index(top), counts, counts.count(top) > 1


def _pin_fn(by_cell: dict, suf: str, iid: str, mode: int):
    """families -> pooled PIN for one (suffix, item), so the bootstrap can resample the
    families the fraction is pooled over."""
    per_fam = {}
    for fam in sorted({k[3] for k in by_cell}):
        hits = tot = 0
        for ctx in CONTEXTS:
            c = by_cell.get((ctx, suf, iid, fam))
            if c:
                hits += c["rating_counts"][mode]
                tot += sum(c["rating_counts"])
        per_fam[fam] = (hits, tot)

    def pin(fams: list[str]) -> float | None:
        h = sum(per_fam[f][0] for f in fams)
        t = sum(per_fam[f][1] for f in fams)
        return (h / t) if t else None
    return pin, per_fam


def _eff_by_family(by_cell: dict, suf: str, iid: str) -> dict[str, float]:
    """The R2 self-fail effect per family: mean rated rating under SELF-FAIL minus the
    same under SELF-SUCC. Rated cells only, by design; a family with no rated sample in
    either cell is dropped and the drop is counted rather than imputed."""
    out = {}
    for fam in sorted({k[3] for k in by_cell}):
        a = by_cell.get(("SELF-FAIL", suf, iid, fam))
        b = by_cell.get(("SELF-SUCC", suf, iid, fam))
        if a and b and a["rated_mean"] is not None and b["rated_mean"] is not None:
            out[fam] = a["rated_mean"] - b["rated_mean"]
    return out


def _interval_refused(st: dict) -> bool:
    """True when the bootstrap could not produce an interval (too many undefined
    resamples). Reported as its own outcome rather than falling into a named one: every
    named outcome asserts something about an interval that does not exist here."""
    lo, hi = st["ci95"]
    return lo != lo or hi != hi                              # NaN


def _decide_pin(st: dict | None) -> str:
    if not st or st.get("point") is None:
        return "NO-RATED-SAMPLES"
    if _interval_refused(st):
        return "PIN-INTERVAL-REFUSED"
    lo, hi = st["ci95"]
    if st["point"] >= PIN_THRESHOLD and st["excludes_zero"]:
        return "PIN-FORMAT"
    if -PIN_BAND <= lo and hi <= PIN_BAND:
        return "PIN-SHARED"
    return "PIN-INTERMEDIATE"


def _decide_sens(st: dict | None) -> str:
    if not st or st.get("point") is None:
        return "NO-RATED-SAMPLES"
    if _interval_refused(st):
        return "SENS-INTERVAL-REFUSED"
    lo, hi = st["ci95"]
    if st["point"] >= SENS_THRESHOLD and st["excludes_zero"]:
        return "FORMAT-SUPPRESSES"
    if -SENS_BAND <= lo and hi <= SENS_BAND:
        return "FORMAT-COMPARABLE"
    if hi < -SENS_BAND:
        return "FORMAT-AMPLIFIED-NEGATIVE"
    return "FORMAT-INDETERMINATE"                            # reading (d)


def _stage17(root: Path) -> tuple[dict | None, str | None]:
    """The committed SITREF endpoint, if it was shipped. E-REG is a replication check
    against it and says ANCHOR-UNAVAILABLE rather than guessing when it is absent."""
    for p, rec in _records(root, 17):
        cells = rec.get("values", {}).get("per_cell", {})
        for cn in sorted(cells):
            c = cells[cn]
            if c.get("variant") == "it":
                sf = c.get("per_item", {}).get(PRIMARY_ID, {}).get("self_fail")
                if sf:
                    return {"cell": cn, "self_fail": sf, "decision": rec.get("decision")}, str(p)
    return None, None


def stage_unblind(root: Path = None, blind_path: Path = None, key_path: Path = None,
                  out_path: Path = None) -> str:                          # noqa: C901
    root = root or OUT
    blind_path = blind_path or (OUT / BLIND_TABLE)
    key_path = key_path or KEY_PATH
    out_path = out_path or (OUT / ENDPOINT)
    if not blind_path.exists():
        raise SystemExit(f"commit the blind table first: {blind_path} missing")
    tab = json.loads(blind_path.read_text())
    if tab.get("decision") != "COMMITTED":
        raise SystemExit(f"blind table decision is {tab.get('decision')!r}; stage 34 runs "
                         f"only on a COMMITTED table")
    if not key_path.exists():
        raise SystemExit(f"suffix blind key missing: {key_path}")
    key = json.loads(key_path.read_text())["mapping_by_variant"]["it"]
    inv = {lab: ctx for ctx, lab in key.items()}
    if len(inv) != len(key) or sorted(key) != sorted(CONTEXTS):
        raise SystemExit(f"the suffix key is not a bijection over the three contexts: "
                         f"{sorted(key)}")

    by_cell = {(inv[c["blind_label"]], c["suffix"], c["item_id"], c["family"]): c
               for c in tab["rows"]}
    fams = sorted({k[3] for k in by_cell})
    items = sorted({k[2] for k in by_cell})

    # --- the two committed rate tables, now with real context names
    rates: dict[str, dict] = {}
    for suf in SUFFIXES:
        for ctx in CONTEXTS:
            sel = [c for k, c in by_cell.items() if k[1] == suf and k[0] == ctx]
            n = sum(c["n_samples"] for c in sel)
            n_rated = sum(c["n_rated"] for c in sel)
            n_echo = sum(c["n_echo"] for c in sel)
            rates[f"{suf}|{ctx}"] = {
                "n_samples": n, "n_rated": n_rated,
                "no_rating_rate": sum(c["n_no_rating"] for c in sel) / n if n else None,
                "echo_rate": n_echo / n if n else None,
                "echo_still_rated_fraction": (sum(c["n_echo_rated"] for c in sel) / n_echo
                                              if n_echo else None),
                "unguarded_scale_rate": (sum(c["n_unguarded_scale"] for c in sel) / n
                                         if n else None),
                "rated_mean": (sum(d * c["rating_counts"][d] for c in sel for d in range(10))
                               / n_rated if n_rated else None),
            }
    rates_by_suffix = {}
    for suf in SUFFIXES:
        sel = [c for k, c in by_cell.items() if k[1] == suf]
        n = sum(c["n_samples"] for c in sel)
        n_echo = sum(c["n_echo"] for c in sel)
        rates_by_suffix[suf] = {
            "n_samples": n,
            "no_rating_rate": sum(c["n_no_rating"] for c in sel) / n if n else None,
            "echo_rate": n_echo / n if n else None,
            "echo_still_rated_fraction": (sum(c["n_echo_rated"] for c in sel) / n_echo
                                          if n_echo else None),
        }

    # --- E-PIN
    modes = {}
    for iid in items:
        m, counts, tied = _modal_rating(by_cell, iid)
        modes[iid] = {"modal_rating": m, "s_num_histogram": counts, "tie": tied}
    pin_values: dict[str, dict] = {}
    pin_outcomes: dict[str, str] = {}
    for arm in FREE_ARMS:
        per_item = {}
        for iid in items:
            m = modes[iid]["modal_rating"]
            if m is None:
                per_item[iid] = {"pin_anchor": None, "pin_arm": None, "pindiff": None,
                                 "outcome": "NO-ANCHOR-MODE"}
                continue
            fa, _ = _pin_fn(by_cell, ANCHOR, iid, m)
            fx, _ = _pin_fn(by_cell, arm, iid, m)

            def diff(fs, fa=fa, fx=fx):
                a, x = fa(fs), fx(fs)
                return None if (a is None or x is None) else a - x
            st = boot_stat(fams, diff)
            per_item[iid] = {"modal_rating": m, "pin_anchor": fa(fams), "pin_arm": fx(fams),
                             "pindiff": st, "outcome": _decide_pin(st)}
        # pooled over items: exploratory (reading (a))
        def pooled(fs):
            num_a = num_x = den_a = den_x = 0
            for iid in items:
                m = modes[iid]["modal_rating"]
                if m is None:
                    continue
                for f in fs:
                    for ctx in CONTEXTS:
                        ca = by_cell.get((ctx, ANCHOR, iid, f))
                        cx = by_cell.get((ctx, arm, iid, f))
                        if ca:
                            num_a += ca["rating_counts"][m]
                            den_a += sum(ca["rating_counts"])
                        if cx:
                            num_x += cx["rating_counts"][m]
                            den_x += sum(cx["rating_counts"])
            if not den_a or not den_x:
                return None
            return num_a / den_a - num_x / den_x
        viol = _echo_violations(rates, _load_bearing("E-PIN", arm))
        primary = per_item.get(PRIMARY_ID, {})
        outcome = "VOID-ECHO" if viol else primary.get("outcome", "NO-PRIMARY-ITEM")
        pin_values[arm] = {"per_item": per_item,
                           "pooled_over_items_exploratory": boot_stat(fams, pooled),
                           "echo_violations": viol,
                           "no_rating_rate_anchor": rates_by_suffix[ANCHOR]["no_rating_rate"],
                           "no_rating_rate_arm": rates_by_suffix[arm]["no_rating_rate"],
                           "no_rating_sensitivity_note":
                               "PIN is a fraction of RATED samples (the design's wording), "
                               "so an arm that declines to rate in one context has its PIN "
                               "computed on the remaining samples and PINDIFF moves even "
                               "with no change in what was rated. DSENS is immune (rated "
                               "cells only, per context, differenced). The NO-RATING rates "
                               "beside this number are therefore part of reading it, not "
                               "context: an arm with a high NO-RATING rate has a PIN over a "
                               "self-selected subsample. Demonstrated in the selftest.",
                           "outcome": outcome}
        pin_outcomes[arm] = outcome

    # --- E-SENS
    sens_values: dict[str, dict] = {}
    sens_outcomes: dict[str, str] = {}
    for arm in FREE_ARMS:
        per_item = {}
        for iid in items:
            ea = _eff_by_family(by_cell, ANCHOR, iid)
            ex = _eff_by_family(by_cell, arm, iid)
            usable = sorted(set(ea) & set(ex))

            def dsens(fs, ea=ea, ex=ex):
                if not fs:
                    return None
                return (abs(mean([ex[f] for f in fs])) - abs(mean([ea[f] for f in fs])))
            st = boot_stat(usable, dsens)
            per_fam_abs = [abs(ex[f]) - abs(ea[f]) for f in usable]
            per_item[iid] = {
                "dsens": st,
                "effect_arm": series_stats([ex[f] for f in usable]),
                "effect_anchor": series_stats([ea[f] for f in usable]),
                "dsens_per_family_abs": series_stats(per_fam_abs),
                "families_used": usable,
                "families_dropped": sorted(set(fams) - set(usable)),
                "outcome": _decide_sens(st),
            }
        viol = _echo_violations(rates, _load_bearing("E-SENS", arm))
        primary = per_item.get(PRIMARY_ID, {})
        outcome = "VOID-ECHO" if viol else primary.get("outcome", "NO-PRIMARY-ITEM")
        sens_values[arm] = {
            "per_item": per_item, "echo_violations": viol, "outcome": outcome,
            "exploratory": arm in X.EXPLORATORY_ARMS,
            "interpretation_note": (
                "EXPLORATORY: suppression under S-EXPL is confounded with justification "
                "demand (DESIGN_SUFFIX.md, E-SENS). The load-bearing arms are S-BARE and "
                "S-HON." if arm in X.EXPLORATORY_ARMS else "load-bearing"),
            # The design requires the NO-RATING rates of both arms printed beside DSENS.
            "no_rating_beside_dsens": {
                f"{s}|{c}": rates[f"{s}|{c}"]["no_rating_rate"]
                for s in (ANCHOR, arm) for c in ("SELF-FAIL", "SELF-SUCC")},
        }
        sens_outcomes[arm] = outcome

    # --- E-REG: the S-NUM first-token anchor, replication only
    r1_validity = {}
    for suf in SUFFIXES:
        for iid in items:
            sel = [c for k, c in by_cell.items() if k[1] == suf and k[2] == iid]
            dm = mean([c["r1_digit_mass"] for c in sel]) if sel else None
            r1_validity[f"{suf}|{iid}"] = {
                "mean_digit_mass": dm,
                "r1_valid": bool(dm is not None and dm >= DIGIT_MASS_FLOOR),
            }
    r1_series = []
    for f in fams:
        a = by_cell.get(("SELF-FAIL", ANCHOR, PRIMARY_ID, f))
        b = by_cell.get(("SELF-SUCC", ANCHOR, PRIMARY_ID, f))
        if a and b:
            r1_series.append(a["r1_scale_expectation"] - b["r1_scale_expectation"])
    r1_self_fail = series_stats(r1_series)
    s17, s17_src = _stage17(root)
    if s17 is None:
        reg_outcome = "ANCHOR-UNAVAILABLE"
    elif r1_self_fail["point"] is None:
        reg_outcome = "ANCHOR-NOT-MEASURED"
    else:
        same_sign = (r1_self_fail["point"] < 0) == (s17["self_fail"]["mean"] < 0)
        if not same_sign:
            reg_outcome = "ANCHOR-SIGN-MISMATCH"
        elif not r1_self_fail["excludes_zero"]:
            reg_outcome = "ANCHOR-WEAK"
        else:
            reg_outcome = "ANCHOR-REPRODUCED"
    reg = {"outcome": reg_outcome,
           "in_run_r1_self_fail_primary_S-NUM": r1_self_fail,
           "stage17_self_fail_primary": s17["self_fail"]["mean"] if s17 else None,
           "stage17_source": s17_src, "r1_validity_by_suffix_item": r1_validity,
           "note": "R1 is reported per arm with its digit_mass validity flag and NO "
                   "cross-suffix endpoint reads it (DESIGN_SUFFIX.md, E-REG). Not gated by "
                   "VOID-ECHO: an echo in the generated text cannot touch the first-token "
                   "logits."}

    # --- the decision
    refuting = [a for a in LOAD_BEARING_ARMS if sens_outcomes.get(a) == "FORMAT-SUPPRESSES"]
    decision = ("E-PIN " + ", ".join(f"{a}={pin_outcomes[a]}" for a in FREE_ARMS)
                + " | E-SENS " + ", ".join(f"{a}={sens_outcomes[a]}" for a in FREE_ARMS)
                + f" | E-REG {reg_outcome}")
    if refuting:
        decision += (" | REFUTES-STANDING-REGISTER-CLAIM (" + ",".join(refuting) + "): "
                     "REPORT.md Finding 4 must be rescoped to 'under forced-format "
                     "elicitation'")
    path = []
    for a in FREE_ARMS:
        st = pin_values[a]["per_item"].get(PRIMARY_ID, {}).get("pindiff")
        if st and st["point"] is not None:
            path.append(f"E-PIN {a}: PINDIFF {st['point']:+.3f} CI [{st['ci95'][0]:+.3f},"
                        f"{st['ci95'][1]:+.3f}] -> {pin_outcomes[a]}")
        else:
            path.append(f"E-PIN {a}: {pin_outcomes[a]}")
    for a in FREE_ARMS:
        st = sens_values[a]["per_item"].get(PRIMARY_ID, {}).get("dsens")
        if st and st["point"] is not None:
            path.append(f"E-SENS {a}: DSENS {st['point']:+.3f} CI [{st['ci95'][0]:+.3f},"
                        f"{st['ci95'][1]:+.3f}] -> {sens_outcomes[a]}"
                        + ("  [EXPLORATORY]" if a in X.EXPLORATORY_ARMS else ""))
        else:
            path.append(f"E-SENS {a}: {sens_outcomes[a]}")
    path.append(f"E-REG: in-run R1 S-NUM self-fail "
                f"{r1_self_fail['point'] if r1_self_fail['point'] is None else round(r1_self_fail['point'], 3)}"
                f" vs stage-17 "
                f"{s17['self_fail']['mean'] if s17 else 'ABSENT'} -> {reg_outcome}")

    tcell = tab.get("cell", {})
    write_result(
        out_path,
        cell={"model": tcell.get("model", "unknown"),
              "variant": tcell.get("variant", "unknown"), "stage": 34},
        inputs=[describe_input(blind_path), describe_input(key_path),
                describe_input(HERE / "DESIGN_SUFFIX.md"),
                describe_input(HERE / "analyze_suffix.py"),
                describe_input(HERE / "suffix_stimuli.py")]
        + ([describe_input(Path(s17_src))] if s17_src else []),
        metric="suffix_endpoints_E-PIN_E-SENS_E-REG_with_no_rating_and_echo_tables",
        values={"decision_path": path, "primary_item": PRIMARY_ID,
                "e_pin": pin_values, "e_pin_outcomes": pin_outcomes,
                "e_sens": sens_values, "e_sens_outcomes": sens_outcomes,
                "e_reg": reg,
                "s_num_modal_rating_by_item": modes,
                "no_rating_and_echo_by_suffix_context": rates,
                "echo_gate_note":
                    "The echo rate is the design's: the fraction of generations with ANY "
                    "stripped span. That includes a model that restates the question and "
                    "then answers, which is not the stage-13 failure mode, so "
                    "echo_still_rated_fraction is reported beside every echo rate. A "
                    "VOID-ECHO outcome with echo_still_rated_fraction near 1.0 says the "
                    "arm is verbose, not that the arm answered nothing; either way the "
                    "gate fires, because the gate was fixed before the data and is not "
                    "reinterpreted after it.",
                "extraction_residual_note":
                    "unguarded_scale_rate beside each cell counts generations that name a "
                    "scale the committed rule does not guard ('1 to 10', 'out of 10', ...). "
                    "Those rows may have had the scale's digit taken as the rating. The "
                    "rule was fixed by the 2026-08-15 amendment and is NOT re-applied "
                    "here; the residual is reported so a reader can size it. It is "
                    "expected to be arm-asymmetric (only the free arms write prose), which "
                    "is precisely why it is counted rather than assumed small.",
                "rates_by_suffix": rates_by_suffix,
                "refuting_outcome_fired": refuting,
                "blind_table_decision": tab.get("decision")},
        threshold={"pindiff_format": PIN_THRESHOLD, "pin_shared_band": PIN_BAND,
                   "dsens_suppresses": SENS_THRESHOLD, "dsens_comparable_band": SENS_BAND,
                   "echo_void_rate": ECHO_VOID, "digit_mass_floor": DIGIT_MASS_FLOOR,
                   "n_boot": N_BOOT, "boot_seed": BOOT_SEED, "ci": [CI_LO, CI_HI],
                   "resampling_unit": "task family (10)"},
        decision_rule=DESIGN_QUOTE_AMENDED + " " + IMPLEMENTATION_READINGS,
        decision=decision,
        notes="Stage 33 committed the blind table under a guard that raises on any attempt "
              "to open the context key; this stage applied the key to that committed "
              "table. Suffix names were never blinded (they are the text under study). "
              "Every endpoint is bound to sampled R2; the greedy rows are carried as an "
              "anchor and are read by no endpoint. S-EXPL is exploratory by "
              "pre-registration whatever it shows.",
        rows=None)
    print(f"stage 34 (endpoint): {decision}")
    for p in path:
        print("   -", p)
    return decision


# ---------------------------------------------------------------------------
# Selftest: five planted scenarios
# ---------------------------------------------------------------------------

def _plan_format_pin(ctx, suf, iid, fam, k, fi):
    """S-NUM pinned on 7; the free arms spread over the scale. -> PIN-FORMAT."""
    if suf == ANCHOR:
        return "7"
    return f"I'd say {(k + fi) % 10}."


def _plan_shared_pin(ctx, suf, iid, fam, k, fi):
    """Every arm pins on 7, prose or not. -> PIN-SHARED."""
    return "7" if suf == ANCHOR else "Thinking about it, I'd say 7."


def _plan_suppression(ctx, suf, iid, fam, k, fi):
    """S-NUM flat at 7 in every context; S-HON drops under SELF-FAIL, by -1.5 in most
    families and -2.5 in every third one so the bootstrap resamples something with real
    variance. -> FORMAT-SUPPRESSES(S-HON), S-BARE comparable."""
    if suf == "S-HON" and ctx == "SELF-FAIL":
        val = (6 if k % 2 else 5) - (1 if fi % 3 == 0 else 0)
        return f"Honestly, about {val}."
    if suf == "S-HON":
        return "Honestly, about 7."
    return "7"


def _plan_echo_flood(prompt_of):
    """Half the S-HON SELF-FAIL samples echo the prompt. -> VOID-ECHO on the endpoints
    that read that cell."""
    def plan(ctx, suf, iid, fam, k, fi):
        if suf == "S-HON" and ctx == "SELF-FAIL" and k % 2 == 0:
            return prompt_of(ctx, suf, iid, fam)[:240]
        return "7" if suf == ANCHOR else "I'd say 7."
    return plan


def _plan_missing(asymmetric: bool):
    """S-EXPL under SELF-FAIL declines to rate 40% of the time; the RATED samples are
    identical either way, so DSENS must not move while the NO-RATING table must."""
    def plan(ctx, suf, iid, fam, k, fi):
        rated = "I'd say 6." if ctx == "SELF-FAIL" else "I'd say 7."
        if suf == "S-EXPL" and ctx == "SELF-FAIL" and asymmetric and k < 4:
            return "That isn't something I can put a number on."
        return "7" if suf == ANCHOR else rated
    return plan


def _synth(tmp: Path, plan, *, needs_prompt: bool = False) -> tuple[Path, Path]:
    """A full SUFFIX battery record and a real blind key, with generations planted by
    `plan` and every row's rating produced by the REAL extractor. Planting text rather
    than ratings is deliberate: a scenario that plants the parsed number cannot catch an
    extractor bug, and the extractor is the instrument under test here.
    """
    import run_suffix as RS
    import sitref_stimuli as S

    (tmp / "out").mkdir(parents=True, exist_ok=True)
    (tmp / "results").mkdir(parents=True, exist_ok=True)
    mapping = RS.load_or_make_key(tmp / "out" / KEY_NAME)
    cells = RS.build_cells()
    if needs_prompt:
        prompt_of = {(c["context"], c["suffix"], c["item_id"], c["family"]): c["prompt"]
                     for c in cells}
        plan = plan(lambda *k: prompt_of[k])
    fam_index = {f: i for i, f in enumerate(S.FAMILY_NAMES)}
    rows = []
    for c in cells:
        r1 = {"digit_distribution": [0.1] * 10, "digit_mass": 0.85,
              "scale_expectation": 6.8 if c["context"] != "SELF-FAIL" else 5.9,
              "top_token": "7", "n_prompt_tokens": 400}
        fi = fam_index[c["family"]]
        rows.append(RS.make_row(c, mapping[c["context"]], "7", decode="greedy",
                                sample_index=None, seed_table_entry=None,
                                seed_applied=None,
                                seed_mode="greedy-deterministic", r1=r1, n_new_tokens=1))
        for k in range(X.K_SAMPLES):
            gen = plan(c["context"], c["suffix"], c["item_id"], c["family"], k, fi)
            sd = X.sample_seed(c["context"], c["suffix"], c["item_id"], c["family"], k)
            rows.append(RS.make_row(c, mapping[c["context"]], gen, decode="sampled",
                                    sample_index=k, seed_table_entry=sd,
                                    seed_applied=c["cell_seed"],
                                    seed_mode="cell-batched", r1=r1,
                                    n_new_tokens=len(gen.split())))
    args = argparse.Namespace(model="google/gemma-2-9b-it", base=False, limit_cells=0,
                              key=str(tmp / "out" / KEY_NAME),
                              max_new_tokens=RS.MAX_NEW_TOKENS)
    RS.write_battery(args, rows, cells, mapping, timing={}, seed_mode="cell-batched",
                     out_dir=tmp / "results")
    return tmp / "results", tmp / "out" / KEY_NAME


def _run_scenario(tmp: Path, plan, *, needs_prompt: bool = False) -> dict:
    """Both stages over one planted battery, in the ON-BOX directory layout: the root
    handed to the analysis is the project directory, which CONTAINS out/<key>. Passing
    the results directory instead would have hidden the fact that a discovery walk over
    the root opens the key and fires the seal on itself."""
    res, key = _synth(tmp, plan, needs_prompt=needs_prompt)
    root = res.parent
    if not (root / "out" / KEY_NAME).exists():
        raise AssertionError("the scenario is not in the on-box layout")
    with sealed(KEY_NAME):
        stage_blind(root, res / BLIND_TABLE)
    stage_unblind(root, res / BLIND_TABLE, key, res / ENDPOINT)
    return json.loads((res / ENDPOINT).read_text())


def _selftest() -> int:  # noqa: C901
    import tempfile

    fails: list[str] = []

    # --- the quoted design rule is VERBATIM, not a retelling
    design = HERE / "DESIGN_SUFFIX.md"
    if design.exists():
        norm = " ".join(design.read_text().replace("#", " ").split())
        if " ".join(DESIGN_QUOTE_AMENDED.split()) not in norm:
            fails.append("DESIGN_QUOTE_AMENDED is not a verbatim span of DESIGN_SUFFIX.md")
    else:
        fails.append("DESIGN_SUFFIX.md is missing: the quoted rule cannot be verified")

    # --- the seal fires on the suffix key and nothing else
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / KEY_NAME).write_text("{}")
        (d / "other.json").write_text("{}")
        try:
            with sealed(KEY_NAME):
                (d / KEY_NAME).read_text()
            fails.append("the guard did not fire on Path.read_text of the suffix key")
        except KeyFileTouched:
            pass
        with sealed(KEY_NAME):
            if json.loads((d / "other.json").read_text()) != {}:
                fails.append("the guard blocked an unrelated file")

    # --- threshold decisions, on planted statistics (no data needed)
    cases_pin = [
        ({"point": 0.9, "ci95": [0.7, 1.0], "excludes_zero": True}, "PIN-FORMAT"),
        ({"point": 0.3, "ci95": [-0.05, 0.6], "excludes_zero": False}, "PIN-INTERMEDIATE"),
        ({"point": 0.0, "ci95": [-0.05, 0.05], "excludes_zero": False}, "PIN-SHARED"),
        ({"point": 0.26, "ci95": [0.11, 0.4], "excludes_zero": True}, "PIN-FORMAT"),
        ({"point": 0.2, "ci95": [0.05, 0.35], "excludes_zero": True}, "PIN-INTERMEDIATE"),
        (None, "NO-RATED-SAMPLES"),
        ({"point": None, "ci95": [float("nan")] * 2, "excludes_zero": False},
         "NO-RATED-SAMPLES"),
        ({"point": 0.5, "ci95": [float("nan")] * 2, "excludes_zero": False},
         "PIN-INTERVAL-REFUSED"),
    ]
    for st, want in cases_pin:
        if _decide_pin(st) != want:
            fails.append(f"_decide_pin({st}) = {_decide_pin(st)}, want {want}")
    cases_sens = [
        ({"point": 0.9, "ci95": [0.5, 1.3], "excludes_zero": True}, "FORMAT-SUPPRESSES"),
        ({"point": 0.0, "ci95": [-0.2, 0.2], "excludes_zero": False}, "FORMAT-COMPARABLE"),
        ({"point": -0.9, "ci95": [-1.3, -0.5], "excludes_zero": True},
         "FORMAT-AMPLIFIED-NEGATIVE"),
        ({"point": 0.6, "ci95": [-0.1, 1.3], "excludes_zero": False},
         "FORMAT-INDETERMINATE"),
        ({"point": 0.5, "ci95": [0.05, 0.9], "excludes_zero": True}, "FORMAT-SUPPRESSES"),
        (None, "NO-RATED-SAMPLES"),
        ({"point": None, "ci95": [float("nan")] * 2, "excludes_zero": False},
         "NO-RATED-SAMPLES"),
        ({"point": 0.5, "ci95": [float("nan")] * 2, "excludes_zero": False},
         "SENS-INTERVAL-REFUSED"),
    ]
    for st, want in cases_sens:
        if _decide_sens(st) != want:
            fails.append(f"_decide_sens({st}) = {_decide_sens(st)}, want {want}")

    # --- the load-bearing cell sets are what the endpoints actually read
    if set(_load_bearing("E-SENS", "S-HON")) != {(ANCHOR, "SELF-FAIL"), (ANCHOR, "SELF-SUCC"),
                                                 ("S-HON", "SELF-FAIL"), ("S-HON", "SELF-SUCC")}:
        fails.append("E-SENS load-bearing cells are wrong")
    if len(_load_bearing("E-PIN", "S-BARE")) != 6:
        fails.append("E-PIN load-bearing cells are wrong")

    # --- discovery: never open a key file, and never prefer a stale out/ copy
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "out").mkdir()
        (d / "results").mkdir()
        (d / "out" / KEY_NAME).write_text(json.dumps(
            {"mapping_by_variant": {"it": dict(zip(CONTEXTS, ("A", "B", "E")))}}))
        stale = {"cell": {"model": "m", "variant": "it", "stage": BATTERY_STAGE},
                 "metric": "x", "decision_rule": "x", "rows": [], "values": {"tag": "stale"}}
        fresh = dict(stale, values={"tag": "fresh"})
        (d / "out" / "b.json").write_text(json.dumps(stale))
        (d / "results" / "b.json").write_text(json.dumps(fresh))
        try:
            with sealed(KEY_NAME):
                p, rec = _one_battery(d)
            if rec["values"]["tag"] != "fresh":
                fails.append("discovery preferred the stale out/ copy over results/")
        except KeyFileTouched:
            fails.append("record discovery opened the blind key and fired the seal on "
                         "itself: stage 33 can never run in the on-box layout")
        (d / "results" / "b.json").unlink()
        with sealed(KEY_NAME):
            _, rec = _one_battery(d)                 # out/-only copy still findable
        if rec["values"]["tag"] != "stale":
            fails.append("an out/-only record became unfindable")

    # --- an uncommitted blind table stops stage 34, and so does a missing key
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "results").mkdir()
        (d / "results" / BLIND_TABLE).write_text(json.dumps(
            {"decision": "NOT_COMMITTED", "rows": []}))
        (d / "key.json").write_text(json.dumps(
            {"mapping_by_variant": {"it": {c: lab for c, lab in
                                           zip(CONTEXTS, ("A", "B", "E"))}}}))
        try:
            stage_unblind(d / "results", d / "results" / BLIND_TABLE, d / "key.json",
                          d / "results" / ENDPOINT)
            fails.append("stage 34 ran on a NOT_COMMITTED blind table")
        except SystemExit:
            pass
        (d / "results" / BLIND_TABLE).write_text(json.dumps(
            {"decision": "COMMITTED", "rows": []}))
        try:
            stage_unblind(d / "results", d / "results" / BLIND_TABLE, d / "absent.json",
                          d / "results" / ENDPOINT)
            fails.append("stage 34 ran without the blind key")
        except SystemExit:
            pass

    # --- scenario 1: a pure format pin
    with tempfile.TemporaryDirectory() as d:
        rec = _run_scenario(Path(d), _plan_format_pin)
        for arm in FREE_ARMS:
            if rec["values"]["e_pin_outcomes"][arm] != "PIN-FORMAT":
                fails.append(f"format-pin: E-PIN {arm} = "
                             f"{rec['values']['e_pin_outcomes'][arm]}, want PIN-FORMAT")
        if rec["values"]["s_num_modal_rating_by_item"][PRIMARY_ID]["modal_rating"] != 7:
            fails.append("format-pin: the S-NUM mode is not 7")
        pin = rec["values"]["e_pin"]["S-BARE"]["per_item"][PRIMARY_ID]
        if abs(pin["pin_anchor"] - 1.0) > 1e-9:
            fails.append(f"format-pin: PIN(S-NUM) = {pin['pin_anchor']}, want 1.0")
        if pin["pin_arm"] > 0.2:
            fails.append(f"format-pin: PIN(S-BARE) = {pin['pin_arm']}, want ~0.1")

    # --- scenario 2: a shared pin
    with tempfile.TemporaryDirectory() as d:
        rec = _run_scenario(Path(d), _plan_shared_pin)
        for arm in FREE_ARMS:
            if rec["values"]["e_pin_outcomes"][arm] != "PIN-SHARED":
                fails.append(f"shared-pin: E-PIN {arm} = "
                             f"{rec['values']['e_pin_outcomes'][arm]}, want PIN-SHARED")
        if rec["values"]["e_sens_outcomes"]["S-BARE"] != "FORMAT-COMPARABLE":
            fails.append(f"shared-pin: E-SENS S-BARE = "
                         f"{rec['values']['e_sens_outcomes']['S-BARE']}, want "
                         f"FORMAT-COMPARABLE")

    # --- scenario 3: suppression, in S-HON only
    with tempfile.TemporaryDirectory() as d:
        rec = _run_scenario(Path(d), _plan_suppression)
        if rec["values"]["e_sens_outcomes"]["S-HON"] != "FORMAT-SUPPRESSES":
            fails.append(f"suppression: E-SENS S-HON = "
                         f"{rec['values']['e_sens_outcomes']['S-HON']}, want "
                         f"FORMAT-SUPPRESSES")
        if rec["values"]["e_sens_outcomes"]["S-BARE"] != "FORMAT-COMPARABLE":
            fails.append(f"suppression: E-SENS S-BARE = "
                         f"{rec['values']['e_sens_outcomes']['S-BARE']}, want "
                         f"FORMAT-COMPARABLE")
        if "REFUTES-STANDING-REGISTER-CLAIM" not in rec["decision"]:
            fails.append("suppression on a load-bearing arm did not fire the design's "
                         "named refuting outcome")
        if rec["values"]["e_sens"]["S-EXPL"]["exploratory"] is not True:
            fails.append("S-EXPL is not marked exploratory in the record")
        if rec["values"]["e_sens"]["S-HON"]["exploratory"] is not False:
            fails.append("a load-bearing arm is marked exploratory")
        beside = rec["values"]["e_sens"]["S-HON"]["no_rating_beside_dsens"]
        if len(beside) != 4:
            fails.append("the NO-RATING rates are not printed beside DSENS")

    # --- scenario 4: an echo flood voids the endpoints that read the cell
    with tempfile.TemporaryDirectory() as d:
        rec = _run_scenario(Path(d), _plan_echo_flood, needs_prompt=True)
        if rec["values"]["e_sens_outcomes"]["S-HON"] != "VOID-ECHO":
            fails.append(f"echo flood: E-SENS S-HON = "
                         f"{rec['values']['e_sens_outcomes']['S-HON']}, want VOID-ECHO")
        if rec["values"]["e_pin_outcomes"]["S-HON"] != "VOID-ECHO":
            fails.append(f"echo flood: E-PIN S-HON = "
                         f"{rec['values']['e_pin_outcomes']['S-HON']}, want VOID-ECHO")
        if rec["values"]["e_sens_outcomes"]["S-BARE"] == "VOID-ECHO":
            fails.append("echo flood in S-HON voided an endpoint that does not read it")
        er = rec["values"]["no_rating_and_echo_by_suffix_context"]["S-HON|SELF-FAIL"]
        if er.get("unguarded_scale_rate") is None:
            fails.append("the rate table does not carry the unguarded-scale diagnostic")
        if not 0.45 < er["echo_rate"] < 0.55:
            fails.append(f"echo flood: echo rate {er['echo_rate']}, want ~0.5")
        if er["echo_still_rated_fraction"] is None:
            fails.append("the echo table does not split echoes that still produced a "
                         "rating from echoes that destroyed the answer")
        clean = rec["values"]["no_rating_and_echo_by_suffix_context"]["S-HON|SELF-SUCC"]
        if clean["echo_rate"] != 0.0 or clean["echo_still_rated_fraction"] is not None:
            fails.append("echo flood leaked into a cell it was not planted in")

    # --- scenario 5: missing ratings show in the tables and do NOT move DSENS
    with tempfile.TemporaryDirectory() as d:
        rec_a = _run_scenario(Path(d) / "a", _plan_missing(True))
    with tempfile.TemporaryDirectory() as d:
        rec_b = _run_scenario(Path(d) / "b", _plan_missing(False))
    ra = rec_a["values"]["no_rating_and_echo_by_suffix_context"]
    rb = rec_b["values"]["no_rating_and_echo_by_suffix_context"]
    if not 0.35 < ra["S-EXPL|SELF-FAIL"]["no_rating_rate"] < 0.45:
        fails.append(f"missing-rating: NO-RATING rate "
                     f"{ra['S-EXPL|SELF-FAIL']['no_rating_rate']}, want 0.4")
    if rb["S-EXPL|SELF-FAIL"]["no_rating_rate"] != 0.0:
        fails.append("missing-rating control is not clean")
    if ra["S-EXPL|SELF-SUCC"]["no_rating_rate"] != 0.0:
        fails.append("missing-rating: the asymmetry leaked into the other context")
    da = rec_a["values"]["e_sens"]["S-EXPL"]["per_item"][PRIMARY_ID]["dsens"]["point"]
    db = rec_b["values"]["e_sens"]["S-EXPL"]["per_item"][PRIMARY_ID]["dsens"]["point"]
    if da is None or db is None or abs(da - db) > 1e-9:
        fails.append(f"missing-rating: DSENS moved with the missingness ({da} vs {db}); "
                     f"'rated cells only' is not being honoured")
    if rec_a["values"]["e_sens_outcomes"]["S-EXPL"] != rec_b["values"]["e_sens_outcomes"]["S-EXPL"]:
        fails.append("missing-rating: the DSENS outcome changed with the missingness")
    # ... and the asymmetry that DSENS is immune to DOES move PIN, because PIN is a
    # fraction of RATED samples by the design's own wording. Asserted so the property is
    # a documented instrument fact rather than a surprise at read time.
    pa = rec_a["values"]["e_pin"]["S-EXPL"]["per_item"][PRIMARY_ID]["pindiff"]["point"]
    pb = rec_b["values"]["e_pin"]["S-EXPL"]["per_item"][PRIMARY_ID]["pindiff"]["point"]
    if not (pa is not None and pb is not None and pa < pb - 1e-9):
        fails.append(f"missing-rating: PINDIFF did not move with the NO-RATING asymmetry "
                     f"({pa} vs {pb}); either PIN stopped being a fraction of rated "
                     f"samples or the plant is not asymmetric")
    if "no_rating_sensitivity_note" not in rec_a["values"]["e_pin"]["S-EXPL"]:
        fails.append("the record does not carry PIN's NO-RATING sensitivity note")
    # E-REG has no stage-17 record in the synthetic tree
    if rec_a["values"]["e_reg"]["outcome"] != "ANCHOR-UNAVAILABLE":
        fails.append(f"E-REG without a stage-17 record = "
                     f"{rec_a['values']['e_reg']['outcome']}, want ANCHOR-UNAVAILABLE")

    # --- E-REG against a planted stage-17 record: sign agreement and disagreement
    import provenance as PR
    for s17_mean, want in ((-0.8718, "ANCHOR-REPRODUCED"), (+0.8718, "ANCHOR-SIGN-MISMATCH")):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            res, key = _synth(d, _plan_shared_pin)
            root = res.parent
            PR.write_result(
                res / "sitref-endpoint_stage-17.json",
                cell={"model": "google/gemma-2-9b", "variant": "both", "stage": 17},
                inputs=[], metric="synthetic",
                values={"per_cell": {"google/gemma-2-9b-it|it": {
                    "variant": "it",
                    "per_item": {PRIMARY_ID: {"self_fail": {"mean": s17_mean,
                                                            "ci95": [s17_mean, s17_mean],
                                                            "excludes_zero": True}}}}}},
                decision_rule="synthetic", decision="INDETERMINATE")
            with sealed(KEY_NAME):
                stage_blind(root, res / BLIND_TABLE)
            stage_unblind(root, res / BLIND_TABLE, key, res / ENDPOINT)
            rec = json.loads((res / ENDPOINT).read_text())
            got = rec["values"]["e_reg"]["outcome"]
            if got != want:
                fails.append(f"E-REG with stage-17 mean {s17_mean:+}: {got}, want {want}")
            if rec["values"]["e_reg"]["stage17_source"] is None:
                fails.append("E-REG did not record which stage-17 record it read")
            # the R1 validity flag must be reported per (suffix, item)
            v = rec["values"]["e_reg"]["r1_validity_by_suffix_item"]
            if len(v) != len(SUFFIXES) * len(X.ITEM_IDS):
                fails.append("R1 validity is not reported per (suffix, item)")

    for f in fails:
        print(f"FAIL {f}")
    print(f"5 planted scenarios over {X.N_CELLS} cells each; thresholds "
          f"PINDIFF>={PIN_THRESHOLD}/+-{PIN_BAND}, DSENS>={SENS_THRESHOLD}/+-{SENS_BAND}, "
          f"VOID-ECHO>{ECHO_VOID}")
    print("selftest:", "PASS" if not fails else f"{len(fails)} FAILURES")
    return 1 if fails else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="SUFFIX: blind table, then the endpoints")
    ap.add_argument("--unblind", action="store_true", help="stage 34: apply the key")
    ap.add_argument("--root", default=str(HERE), help="tree to search for records")
    ap.add_argument("--key", default=str(KEY_PATH))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(_selftest())
    OUT.mkdir(parents=True, exist_ok=True)
    if a.unblind:
        stage_unblind(Path(a.root), OUT / BLIND_TABLE, Path(a.key), OUT / ENDPOINT)
    else:
        with sealed(KEY_NAME):
            stage_blind(Path(a.root), OUT / BLIND_TABLE)
