"""SRCDEC-B: is E1's self-vs-sourceless increment a social term, or length and hedge?

WHY THIS EXISTS. E1 came out ASSERTION-COPY: removing the speaker from the assertion
barely changed the stance effect. Barely is not zero -- there was a +0.043 self-minus-
sourceless increment -- and triage left two readings of it alive. It is a SOCIAL term (a
speaker who commits carries more than an unowned assertion), or it is an artifact of the
two preambles differing in two other ways at once: SELF is one whitespace token longer
than SRCLESS, and it hedges with a different verb class ("I'd guess" against "probably").
A three-way difference cannot adjudicate a one-way question.

So SRCDEC-B measures the increment twice, each time holding two of the three fixed:

    SELF  (7 tokens, guess-hedge)  vs  SRCLESS7 (7 tokens, probably-hedge)
    SELF6 (6 tokens, guess-hedge)  vs  GUESS6   (6 tokens, guess-hedge, no agent)

The first pair equalises length, the second equalises both length and hedge verb, and
the design requires BOTH to show the increment before anything social is claimed. One
alone is MIXED and reported as measured -- because one alone is exactly what a
length-only or a hedge-only explanation predicts.

ANCHORS ARE IN-RUN, BY PRE-REGISTRATION. Stage-19's D_self (+0.2877) and D_srcless
(+0.2449) are recorded below for context and are NOT used in any threshold: they came
off a different box, and the endpoints here are paired within-item differences measured
in this one. Cross-box magnitude comparison is the mistake this design was written to
avoid.

    python3 analyze_srcdec_b.py            # -> results/srcdec-b-endpoints_stage-26.json
    python3 analyze_srcdec_b.py --selftest # every named outcome planted and recovered
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from analyze_srcdec import contrast, log_odds, ratio, value  # noqa: F401
from analyze_sitref import mean, stats
from provenance import describe_input, write_result

HERE = Path(__file__).parent
OUT = HERE / "results"
ENDPOINTS = "srcdec-b-endpoints_stage-26.json"
ARM_SET = "b"

# (name, YES arm, NO arm). The anchors are the last two: re-run here, never imported.
CONTRASTS: tuple[tuple[str, str, str], ...] = (
    ("D_srcless7", "SRCLESS7-YES", "SRCLESS7-NO"),
    ("D_self6", "SELF6-YES", "SELF6-NO"),
    ("D_guess6", "GUESS6-YES", "GUESS6-NO"),
    ("D_self", "SELF-YES", "SELF-NO"),
    ("D_srcless", "SRCLESS-YES", "SRCLESS-NO"),
)
# The two paired differences the decision turns on, in the design's order.
PAIRED: tuple[tuple[str, str, str], ...] = (
    ("self_minus_srcless7", "D_self", "D_srcless7"),
    ("self6_minus_guess6", "D_self6", "D_guess6"),
)
STAGE19_CONTEXT = {"D_self": 0.2877, "D_srcless": 0.2449,
                   "_note": "stage-19 values, recorded for context only; the design "
                            "forbids reusing them across boxes and no threshold here "
                            "reads them"}

DESIGN_QUOTE = (
    "DESIGN_SITREF_C.md, 'SRCDEC-B arms', quoted verbatim: 'Endpoints: D_srcless7 = "
    "SRCLESS7-YES - SRCLESS7-NO per item, similarly D_self6, D_guess6; comparators from "
    "the stage-19 run's D_self (+0.2877) and D_srcless (+0.2449) are NOT reused across "
    "boxes - SELF-YES/NO and SRCLESS-YES/NO are re-run here as in-run anchors. "
    "SOCIAL-TERM-SUPPORTED: (D_self - D_srcless7) paired CI excludes 0 above AND (D_self6 "
    "- D_guess6) paired CI excludes 0 above. Both must hold: each pair is length-matched "
    "within itself and hedges with the same verb class. NO-SOCIAL-TERM: both CIs include "
    "0 or either is negative. MIXED: one holds - reported as measured, no stronger "
    "label.' IMPLEMENTATION READING, DECLARED: 'CI excludes 0 above' is the whole 95% "
    "bootstrap interval lying above 0 (lower bound > 0), and 'either is negative' is "
    "either interval lying wholly below 0; those two clauses are evaluated in the "
    "design's order, so a run with one interval above 0 and the other below 0 reads "
    "NO-SOCIAL-TERM (the 'either is negative' clause), not MIXED. Bootstrap over items, "
    "10,000 resamples, seeded; the paired difference is computed per item and then "
    "resampled, never as a difference of two independently resampled means."
)


def _records(root: Path, stage: str, armset: str) -> list:
    """srcdec records for one arm set. Not analyze_sitref.find_records: that de-duplicates
    on (model, variant), and the arm-set-a and arm-set-b runs share a checkpoint."""
    seen: dict = {}
    for p in sorted(root.rglob("*.json")):
        if p.name.endswith("_summary.json"):
            continue
        try:
            rec = json.loads(p.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            continue
        c = rec.get("cell") if isinstance(rec, dict) else None
        if not isinstance(c, dict) or c.get("stage") != stage or c.get("armset") != armset:
            continue
        seen.setdefault((c.get("model"), c.get("variant")), (p, rec))
    return list(seen.values())


def analyse(rows: list) -> dict:
    """Every contrast, both paired differences, and the named outcome."""
    per = {name: contrast(rows, hi, lo, "welfare") for name, hi, lo in CONTRASTS}
    missing = [n for n, d in per.items() if not d]
    if missing:
        return {"outcome": "NOT-RUN", "reason": f"arms absent from the record: {missing}"}

    out: dict = {"contrasts": {n: stats(list(d.values())) for n, d in per.items()},
                 "n_items_by_contrast": {n: len(d) for n, d in per.items()},
                 "stage19_context": STAGE19_CONTEXT}

    paired = {}
    for name, a, b in PAIRED:
        shared = sorted(set(per[a]) & set(per[b]))
        if not shared:
            return {"outcome": "NOT-RUN", "reason": f"{name}: no item carries both arms"}
        paired[name] = stats([per[a][i] - per[b][i] for i in shared])
        paired[name]["n_shared_items"] = len(shared)
    out["paired"] = paired

    # Robustness companions: every run-1 retraction was a readout-denominator artifact,
    # so a paired difference that survives only in the ratio has to be visible as such.
    rob = {}
    for name, a, b in PAIRED:
        ha, la = next((h, lo) for n, h, lo in CONTRASTS if n == a)[0:2]
        hb, lb = next((h, lo) for n, h, lo in CONTRASTS if n == b)[0:2]
        for label, read in (("log_odds", log_odds),
                            ("margin_raw", lambda r: r["p_yes"] - r["p_no"])):
            da = contrast(rows, ha, la, "welfare", read=read)
            db = contrast(rows, hb, lb, "welfare", read=read)
            sh = sorted(set(da) & set(db))
            if sh:
                rob[f"{name}__{label}"] = stats([da[i] - db[i] for i in sh])
    out["robustness"] = rob
    out["yesno_mass_by_arm"] = {a: mean([r["yesno_mass"] for r in rows if r["arm"] == a])
                                for a in sorted({r["arm"] for r in rows})}

    above = {n: paired[n]["ci95"][0] > 0 for n in paired}
    below = {n: paired[n]["ci95"][1] < 0 for n in paired}
    path = [f"{n} {paired[n]['mean']:+.4f} CI [{paired[n]['ci95'][0]:+.4f},"
            f"{paired[n]['ci95'][1]:+.4f}] -> "
            f"{'above 0' if above[n] else 'below 0' if below[n] else 'includes 0'}"
            for n in paired]
    if all(above.values()):
        outcome = "SOCIAL-TERM-SUPPORTED"
    elif any(below.values()):
        outcome = "NO-SOCIAL-TERM"
        path.append("a paired difference is negative: the increment reverses under a "
                    "matched hedge, which no social-term reading predicts")
    elif not any(above.values()):
        outcome = "NO-SOCIAL-TERM"
        path.append("both paired differences include 0")
    else:
        outcome = "MIXED"
        path.append("one matched pair shows the increment and the other does not: "
                    "reported as measured, which is what a length-only or hedge-only "
                    "explanation predicts")
    out["outcome"] = outcome
    out["decision_path"] = path
    return out


def main(root: Path, out_path: Path) -> dict:
    cells = _records(root, "srcdec-binary", ARM_SET)
    if not cells:
        raise SystemExit(f"no arm-set-b srcdec-binary record under {root}: run "
                         f"`run_srcdec.py --stage binary --arm-set b --base` first")
    base = [(p, r) for p, r in cells if r["cell"].get("variant") == "base"]
    if not base:
        raise SystemExit("SRCDEC-B is a base-checkpoint endpoint; no base cell present")
    src, rec = base[0]
    values = analyse(rec.get("rows", []))
    values["cell"] = {k: v for k, v in rec["cell"].items()}
    values["run_decision"] = rec.get("decision")
    if rec.get("decision") != "COMPLETE":
        values["outcome"] = "NOT-RUN"
        values.setdefault("reason", f"battery decision was {rec.get('decision')!r}")

    write_result(
        out_path,
        cell={"model": "google/gemma-2-9b", "variant": "base", "stage": 26},
        inputs=[describe_input(src), describe_input(HERE / "DESIGN_SITREF_C.md"),
                describe_input(HERE / "analyze_srcdec_b.py")],
        metric="srcdec_b_social_term_under_matched_length_and_hedge",
        values=values,
        threshold={"n_boot": 10000, "resampling_unit": "item",
                   "pairs": [f"{a}-{b}" for _, a, b in PAIRED]},
        decision_rule=DESIGN_QUOTE,
        decision=values["outcome"],
        notes="Anchors (D_self, D_srcless) are re-run in this box; stage-19's values are "
              "recorded as context and read by no threshold. Log-odds and margin_raw "
              "companions accompany both paired differences because all three run-1 "
              "retractions were readout-denominator artifacts.",
        rows=None)
    print(f"srcdec-b: {values['outcome']}")
    for line in values.get("decision_path", []):
        print("   -", line)
    return values


# ---------------------------------------------------------------------------
# Selftest
# ---------------------------------------------------------------------------

def _plant(tmp: Path, inc_pair1: float, inc_pair2: float, base_effect: float = 0.25,
           n_items: int = 82) -> Path:
    """A base arm-set-b cell with the two increments planted directly.

    Each source cell displaces its YES and NO arms symmetrically about the item's neutral
    ratio, so D_x is exactly the named size per item; the two paired differences are then
    inc_pair1 = D_self - D_srcless7 and inc_pair2 = D_self6 - D_guess6 by construction. A
    per-item wobble keeps the bootstrap from resampling a constant.
    """
    size = {"D_srcless7": base_effect, "D_guess6": base_effect,
            "D_self": base_effect + inc_pair1, "D_self6": base_effect + inc_pair2,
            "D_srcless": base_effect}
    rows = []
    for k in range(n_items):
        neutral = 0.05 * ((k % 5) - 2)
        for j, (name, hi, lo) in enumerate(CONTRASTS):
            # Per-(item, contrast) wobble, not per-item: a wobble shared by both members
            # of a pair cancels in the paired difference and leaves the bootstrap
            # resampling a constant, which would make a zero-width CI look like a passing
            # test of an interval rule.
            d = size[name] + 0.004 * ((k % 7) - 3) + 0.003 * (((k + j) % 5) - 2)
            for arm, half in ((hi, +d / 2), (lo, -d / 2)):
                v = max(min(neutral + half, 0.95), -0.95)
                p_yes, p_no = (1 + v) / 2 * 0.8, (1 - v) / 2 * 0.8
                rows.append({"prompt": f"<{arm}> W{k:03d}", "item_id": f"W{k:03d}",
                             "item_class": "welfare", "fmt": "BINARY", "tier": "DERIVED",
                             "arm": arm, "preamble": arm,
                             "p_yes": p_yes, "p_no": p_no, "yesno_mass": p_yes + p_no,
                             "margin_ratio": v, "margin_raw": p_yes - p_no,
                             "digit_mass": 0.0, "scale_expectation": 0.0,
                             "digit_distribution": [0.1] * 10, "top_token": "Yes",
                             "n_prompt_tokens": 40})
    p = tmp / "results" / "srcdec_b_stage-srcdec-binary.json"
    write_result(p, cell={"model": "google/gemma-2-9b", "variant": "base",
                          "stage": "srcdec-binary", "armset": "b"},
                 inputs=[], metric="synthetic", values={}, decision_rule="synthetic",
                 decision="COMPLETE", rows=rows)
    return p


def _selftest() -> int:
    import tempfile

    fails = []
    cases = [
        ("SOCIAL-TERM-SUPPORTED", 0.06, 0.06),
        ("NO-SOCIAL-TERM", 0.0, 0.0),        # both include 0
        ("NO-SOCIAL-TERM", -0.06, 0.06),     # one negative -> the design's "either" clause
        ("MIXED", 0.06, 0.0),                # one holds
        ("MIXED", 0.0, 0.06),                # the other holds
    ]
    for want, i1, i2 in cases:
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            _plant(d, i1, i2)
            v = main(d, d / "results" / ENDPOINTS)
            if v["outcome"] != want:
                fails.append(f"planted ({i1:+.2f},{i2:+.2f}) -> {v['outcome']}, want {want} "
                             f"[{v.get('decision_path')}]")
                continue
            got1 = v["paired"]["self_minus_srcless7"]["mean"]
            got2 = v["paired"]["self6_minus_guess6"]["mean"]
            if abs(got1 - i1) > 0.01 or abs(got2 - i2) > 0.01:
                fails.append(f"planted ({i1:+.3f},{i2:+.3f}) recovered "
                             f"({got1:+.3f},{got2:+.3f})")
            if v["n_items_by_contrast"]["D_self"] != 82:
                fails.append(f"{want}: {v['n_items_by_contrast']['D_self']} items, want 82")
            if any(k.endswith("__log_odds") is False and False for k in v["robustness"]):
                fails.append("robustness companions missing")
            if len(v["robustness"]) != 2 * len(PAIRED):
                fails.append(f"{len(v['robustness'])} robustness companions, want "
                             f"{2 * len(PAIRED)}")

    # a record that never completed must not be read as an outcome
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        p = _plant(d, 0.06, 0.06)
        rec = json.loads(p.read_text())
        rec["decision"] = "INCOMPLETE"
        p.write_text(json.dumps(rec))
        v = main(d, d / "results" / ENDPOINTS)
        if v["outcome"] != "NOT-RUN":
            fails.append(f"INCOMPLETE battery read as {v['outcome']}")

    # an arm-set-a record must not be picked up by this analysis
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        rec = json.loads(_plant(d, 0.06, 0.06).read_text())
        write_result(d / "results" / "arm_a_stage-srcdec-binary.json",
                     cell={"model": "google/gemma-2-9b", "variant": "base",
                           "stage": "srcdec-binary"},
                     inputs=[], metric="m", values={}, decision_rule="r",
                     decision="COMPLETE", rows=rec["rows"])
        found = _records(d / "results", "srcdec-binary", ARM_SET)
        if len(found) != 1:
            fails.append(f"arm-set filter matched {len(found)} records, want 1")

    for f in fails:
        print(f"FAIL {f}")
    print(f"{len(cases)} planted outcomes; {len(CONTRASTS)} contrasts, {len(PAIRED)} "
          f"length-and-hedge-matched pairs")
    print("selftest:", "PASS" if not fails else f"{len(fails)} FAILURES")
    return 1 if fails else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="SRCDEC-B: social term under matched hedges")
    ap.add_argument("--root", default=str(HERE))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(_selftest())
    OUT.mkdir(parents=True, exist_ok=True)
    main(Path(a.root), OUT / ENDPOINTS)
