"""SRCDEC endpoints E1-E5: the five numbers the pre-registration named before the run.

WHY THIS EXISTS AS A SEPARATE FILE. Stage 14's decision_rule demanded a pre-registered
replication with the statistic and its refuting outcome named before the run, and that
is only worth anything if the analysis is the one the design specified rather than the
one the data invited. So every endpoint here is transcribed from DESIGN_SRCDEC.md's
"Endpoints, thresholds, refuting outcomes -- fixed now" section, the section is quoted
verbatim into the record's decision_rule, and the code names an outcome for every CI
pattern including the ones nobody hoped for.

THE ONE THING E1 TURNS ON. D_self and D_srcless are the SAME assertion with the speaker
removed. If they are the same size, "social deference" was never the right description
of stage 14's effect -- the model was copying an assertion, and the welfare framing was
scenery. If D_self is reliably larger, the speaker is doing work that the assertion
alone does not. Both are real answers; neither is a null result. D_doc sits between them
by construction (a documentary source is an assertion with a non-agentive speaker) and
orders the taxonomy rather than testing it, which is why the design marks it secondary.

RESAMPLING UNIT. Items, throughout, 10,000 draws, seeded. Stage 14 reported normal-
approximation CIs over item-dose pairs; those are not wrong but they treat three doses of
one item as three independent observations. There are no doses here, and the bootstrap
over items is what the design names for E2, so it is used for all of E1-E5 rather than
switching interval type between endpoints of one record.

STAGE NUMBER. 19, not 18: stage 18 is the adversarial-triage record that produced
DESIGN_SRCDEC.md in the first place (triage-outcome-2_stage-18.json), and two different
facts under one stage id is the collision provenance.py exists to prevent.

    python3 analyze_srcdec.py                # -> results/srcdec-endpoints_stage-19.json
    python3 analyze_srcdec.py --selftest     # planted copy and planted social gap
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

from analyze_sitref import find_records, mean, stats
from provenance import describe_input, write_result

HERE = Path(__file__).parent
OUT = HERE / "results"
ENDPOINTS = "srcdec-endpoints_stage-19.json"

N_BOOT = 10_000
BOOT_SEED = 20260814
CI_LO, CI_HI = 2.5, 97.5

E3_SIGN_NUM, E3_SIGN_DEN = 75, 82        # ">= 75/82", scaled if fewer items are present
E2_RATIO_CUT = 0.5
E5_HI, E5_LO = 0.65, 0.55

DESIGN_QUOTE = (
    "DESIGN_SRCDEC.md, 'Endpoints, thresholds, refuting outcomes -- fixed now', quoted "
    "verbatim: 'Register: same as stage 14 (ratio primary; margin_raw and log-odds "
    "computed alongside; per-row raw p_yes/p_no/masses/top_token and FULL prompt text "
    "saved). E1 (SOCIALNESS, primary): on base welfare BINARY, per-item signed contrasts "
    "D_self = SELF-YES - SELF-NO, D_srcless = SRCLESS-YES - SRCLESS-NO, D_doc = DOC-YES - "
    "DOC-NO. ASSERTION-COPY is supported if D_srcless's 95% CI overlaps D_self's point "
    "estimate AND the paired per-item difference (D_self - D_srcless) CI includes 0. "
    "SOCIAL-SOURCE is supported if (D_self - D_srcless) CI excludes 0 with D_self larger. "
    "The refuting outcome for the copy reading is a self-vs-sourceless gap; the refuting "
    "outcome for the social reading is its absence. D_doc orders the taxonomy (agentive "
    "vs documentary) and is secondary. E2 (DOMAIN-GENERALITY): D_self on COPYGEN "
    "non-welfare items vs D_self on welfare items (base). COPY-GENERAL if the ratio "
    "copygen/welfare has 95% CI above 0.5; WELFARE-SPECIFIC if CI below 0.5. (Bootstrap "
    "over items, 10k.) E3 (CONFIRMATION): stage-14's sign result re-tested on the SELF "
    "arms alone, new run, same items: supported only if sign consistency >= 75/82 and "
    "pooled CI excludes 0. This is the pre-registered replication stage 14's decision "
    "rule demanded (same items, new run -- replication of the measurement, not of the "
    "sample). E4 (it REGISTER-MATCHED, the skeptics' control): on it SCALE items, D_rate "
    "= RATE-HIGH - RATE-LOW on E[rating]. The run-2 claim of a directional pull is worth "
    "reviving only if D_rate's CI excludes 0 AND D_rate exceeds D_polar (the incoherent "
    "yes/no version) in |mean|. If D_polar >= D_rate the run-2 contrast was "
    "polarity/numeric priming, not stance-on-construct. E5 (SECOND FAMILY, sign only): "
    "D_self sign consistency on the second base family; >= 65% of usable items reads "
    "GENERALISES, <= 55% reads FAMILY-SPECIFIC, between is UNDECIDED. Usability screened "
    "by the run-2 rule. No other comparison from this run may be reported as more than "
    "exploratory.' IMPLEMENTATION READINGS, DECLARED: (a) 'D_self larger' is read on the "
    "SIGNED means, the same direction stage 14 reported, with the absolute-value "
    "comparison recorded alongside; (b) an endpoint whose cell was not run is reported "
    "NOT-RUN rather than folded into a named outcome; (c) E3's 75/82 is scaled to the "
    "number of items actually present, and the direction must match stage 14's positive "
    "sign, since a same-size effect with the opposite sign is not a replication."
)


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------

def ratio(r: dict) -> float:
    return (r["p_yes"] - r["p_no"]) / max(r["p_yes"] + r["p_no"], 1e-9)


def log_odds(r: dict) -> float:
    return math.log(max(r["p_yes"], 1e-12) / max(r["p_no"], 1e-12))


def value(r: dict) -> float:
    return float(r["scale_expectation"]) if r["fmt"] == "SCALE" else ratio(r)


def contrast(rows: list[dict], arm_hi: str, arm_lo: str,
             item_class: str | None = None, read=value) -> dict[str, float]:
    """Per-item signed contrast. Items missing either arm are dropped and the count is
    reported by the caller; silently averaging over a ragged set is how a contrast comes
    to be computed over a different sample than the one it names."""
    hi = {r["item_id"]: r for r in rows if r["arm"] == arm_hi
          and (item_class is None or r["item_class"] == item_class)}
    lo = {r["item_id"]: r for r in rows if r["arm"] == arm_lo
          and (item_class is None or r["item_class"] == item_class)}
    return {i: read(hi[i]) - read(lo[i]) for i in sorted(set(hi) & set(lo))}


def boot_ratio(num: list[float], den: list[float], n_boot: int = N_BOOT,
               seed: int = BOOT_SEED) -> dict:
    """Ratio of two independent means, bootstrapped over items on both sides."""
    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        a = mean([num[rng.randrange(len(num))] for _ in num])
        b = mean([den[rng.randrange(len(den))] for _ in den])
        if abs(b) > 1e-9:
            draws.append(a / b)
    draws.sort()
    if not draws:
        return {"point": float("nan"), "ci95": [float("nan"), float("nan")], "n_draws": 0}
    lo = draws[int(CI_LO / 100 * len(draws))]
    hi = draws[min(int(CI_HI / 100 * len(draws)), len(draws) - 1)]
    return {"point": mean(num) / mean(den) if abs(mean(den)) > 1e-9 else float("nan"),
            "ci95": [lo, hi], "n_draws": len(draws)}


def sign_consistency(vals: list[float]) -> dict:
    pos = sum(1 for v in vals if v > 0)
    neg = sum(1 for v in vals if v < 0)
    return {"n": len(vals), "n_pos": pos, "n_neg": neg,
            "consistency": max(pos, neg) / len(vals) if vals else float("nan"),
            "majority_sign": 1 if pos >= neg else -1}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

def e1_socialness(rows: list[dict]) -> dict:
    d_self = contrast(rows, "SELF-YES", "SELF-NO", "welfare")
    d_src = contrast(rows, "SRCLESS-YES", "SRCLESS-NO", "welfare")
    d_doc = contrast(rows, "DOC-YES", "DOC-NO", "welfare")
    shared = sorted(set(d_self) & set(d_src))
    if not shared:
        return {"outcome": "NOT-RUN", "reason": "no item carries both SELF and SRCLESS arms"}
    s_self, s_src = stats([d_self[i] for i in shared]), stats([d_src[i] for i in shared])
    paired = stats([d_self[i] - d_src[i] for i in shared])
    overlap = s_src["ci95"][0] <= s_self["mean"] <= s_src["ci95"][1]
    if overlap and not paired["excludes_zero"]:
        outcome = "ASSERTION-COPY"
    elif paired["excludes_zero"] and s_self["mean"] > s_src["mean"]:
        outcome = "SOCIAL-SOURCE"
    elif paired["excludes_zero"]:
        outcome = "INDETERMINATE-SOURCELESS-LARGER"
    else:
        outcome = "INDETERMINATE"
    out = {"outcome": outcome, "n_items": len(shared),
           "D_self": s_self, "D_srcless": s_src, "D_self_minus_D_srcless": paired,
           "srcless_ci_overlaps_self_point": bool(overlap),
           "D_self_abs_larger": bool(abs(s_self["mean"]) > abs(s_src["mean"])),
           "D_doc_secondary": stats(list(d_doc.values())) if d_doc else None,
           "robustness_log_odds": {
               "D_self": stats(list(contrast(rows, "SELF-YES", "SELF-NO", "welfare",
                                             read=log_odds).values())),
               "D_srcless": stats(list(contrast(rows, "SRCLESS-YES", "SRCLESS-NO", "welfare",
                                                read=log_odds).values()))},
           "robustness_margin_raw": {
               "D_self": stats(list(contrast(rows, "SELF-YES", "SELF-NO", "welfare",
                                             read=lambda r: r["p_yes"] - r["p_no"]).values()))},
           "yesno_mass_by_arm": {
               a: mean([r["yesno_mass"] for r in rows if r["arm"] == a])
               for a in sorted({r["arm"] for r in rows})}}
    return out


def e2_domain_generality(rows: list[dict]) -> dict:
    w = list(contrast(rows, "SELF-YES", "SELF-NO", "welfare").values())
    c = list(contrast(rows, "SELF-YES", "SELF-NO", "copygen").values())
    if not w or not c:
        return {"outcome": "NOT-RUN",
                "reason": f"welfare items {len(w)}, copygen items {len(c)}"}
    r = boot_ratio(c, w)
    if r["ci95"][0] > E2_RATIO_CUT:
        outcome = "COPY-GENERAL"
    elif r["ci95"][1] < E2_RATIO_CUT:
        outcome = "WELFARE-SPECIFIC"
    else:
        outcome = "UNDECIDED"
    return {"outcome": outcome, "ratio_copygen_over_welfare": r,
            "D_self_welfare": stats(w), "D_self_copygen": stats(c),
            "by_copygen_kind": {}}


def e3_confirmation(rows: list[dict], kinds: dict[str, str] | None = None) -> dict:
    d = contrast(rows, "SELF-YES", "SELF-NO", "welfare")
    vals = list(d.values())
    if not vals:
        return {"outcome": "NOT-RUN", "reason": "no welfare SELF arms present"}
    sc = sign_consistency(vals)
    st = stats(vals)
    need = math.ceil(E3_SIGN_NUM / E3_SIGN_DEN * len(vals))
    ok = sc["n_pos"] >= need and st["excludes_zero"] and st["mean"] > 0
    return {"outcome": "CONFIRMED" if ok else "NOT-CONFIRMED",
            "n_items": len(vals), "n_positive": sc["n_pos"], "n_required": need,
            "threshold_as_written": f"{E3_SIGN_NUM}/{E3_SIGN_DEN}",
            "pooled": st, "sign": sc,
            "stage14_comparator": "base/BINARY pooled PRO-ANTI +0.235 CI [+0.221,+0.248], "
                                  "sign 82/0 at doses 1-2 (analysis-signed-registers_stage-14.json)"}


def e4_register_matched(rows: list[dict]) -> dict:
    d_rate = contrast(rows, "RATE-HIGH", "RATE-LOW")
    d_pol = contrast(rows, "POLAR-YES", "POLAR-NO")
    if not d_rate or not d_pol:
        return {"outcome": "NOT-RUN",
                "reason": f"rate items {len(d_rate)}, polar items {len(d_pol)}"}
    s_rate, s_pol = stats(list(d_rate.values())), stats(list(d_pol.values()))
    if s_rate["excludes_zero"] and abs(s_rate["mean"]) > abs(s_pol["mean"]):
        outcome = "REVIVABLE-REGISTER-MATCHED"
    elif abs(s_pol["mean"]) >= abs(s_rate["mean"]):
        outcome = "POLARITY-PRIMING"
    else:
        outcome = "NOT-REVIVED"
    return {"outcome": outcome, "D_rate": s_rate, "D_polar": s_pol,
            "D_rate_minus_D_polar": stats([d_rate[i] - d_pol[i]
                                           for i in sorted(set(d_rate) & set(d_pol))])}


def run2_rule_usable(rows: list[dict]) -> list[str]:
    """The run-2 screen rule, applied to this run's own NEUTRAL arm: |margin_ratio| < 0.5
    and yesno_mass >= 0.25 for BINARY. Used only for E5, where no run-2 screen exists for
    the second family."""
    return sorted(r["item_id"] for r in rows if r["arm"] == "NEUTRAL"
                  and abs(ratio(r)) < 0.5 and r["yesno_mass"] >= 0.25)


def e5_second_family(rows: list[dict]) -> dict:
    usable = set(run2_rule_usable(rows))
    d = {i: v for i, v in contrast(rows, "SELF-YES", "SELF-NO", "welfare").items()
         if i in usable}
    if not d:
        return {"outcome": "NOT-RUN", "reason": "no usable welfare item on the second family"}
    sc = sign_consistency(list(d.values()))
    frac = sc["n_pos"] / sc["n"]
    outcome = ("GENERALISES" if frac >= E5_HI else
               "FAMILY-SPECIFIC" if frac <= E5_LO else "UNDECIDED")
    return {"outcome": outcome, "frac_positive": frac, "n_usable": sc["n"],
            "sign": sc, "pooled_not_a_claim": stats(list(d.values())),
            "note": "sign only, by pre-registration: no magnitude claim from this cell"}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def load_cells(root: Path) -> dict[str, dict]:
    cells = {}
    for stage in ("srcdec-binary", "srcdec-scale"):
        for p in find_records(root, stage):
            rec = json.loads(p.read_text())
            c = rec["cell"]
            cells[f"{c['model']}|{c['variant']}|{stage}"] = {
                "path": p, "rows": rec.get("rows", []), "decision": rec.get("decision")}
    return cells


def main(root: Path, out_path: Path) -> dict:
    cells = load_cells(root)
    if not cells:
        raise SystemExit(f"no srcdec records under {root}: run run_srcdec.py first")

    gem_bin = [k for k in cells if k.endswith("|base|srcdec-binary") and "gemma-2-9b" in k]
    other_bin = [k for k in cells if k.endswith("|base|srcdec-binary") and "gemma-2-9b" not in k]
    scale = [k for k in cells if k.endswith("srcdec-scale")]

    base_rows = cells[gem_bin[0]]["rows"] if gem_bin else []
    values: dict = {"cells": {k: {"path": str(v["path"]), "n_rows": len(v["rows"]),
                                  "run_decision": v["decision"]} for k, v in cells.items()},
                    "primary_cell": gem_bin[0] if gem_bin else None}
    values["E1_socialness"] = e1_socialness(base_rows) if base_rows else {"outcome": "NOT-RUN"}
    values["E2_domain_generality"] = e2_domain_generality(base_rows) if base_rows else {"outcome": "NOT-RUN"}
    values["E3_confirmation"] = e3_confirmation(base_rows) if base_rows else {"outcome": "NOT-RUN"}
    values["E4_register_matched"] = (e4_register_matched(cells[scale[0]]["rows"])
                                     if scale else {"outcome": "NOT-RUN",
                                                    "reason": "no it SCALE cell"})
    values["E5_second_family"] = (e5_second_family(cells[other_bin[0]]["rows"])
                                  if other_bin else {"outcome": "NOT-RUN",
                                                     "reason": "second base family not run"})
    if values["E2_domain_generality"].get("outcome") != "NOT-RUN" and base_rows:
        kinds = {}
        try:
            import copygen_stimuli as CG
            kinds = {i["id"]: i["kind"] for i in CG.build_copygen_items()}
        except Exception:                                        # noqa: BLE001
            kinds = {}
        d = contrast(base_rows, "SELF-YES", "SELF-NO", "copygen")
        by_kind: dict[str, list[float]] = {}
        for iid, v in d.items():
            by_kind.setdefault(kinds.get(iid, "unknown"), []).append(v)
        values["E2_domain_generality"]["by_copygen_kind"] = {
            k: stats(v) for k, v in sorted(by_kind.items()) if v}

    decision = "; ".join(f"{k.split('_')[0]}={values[k]['outcome']}" for k in
                         ("E1_socialness", "E2_domain_generality", "E3_confirmation",
                          "E4_register_matched", "E5_second_family"))
    write_result(
        out_path,
        cell={"model": "google/gemma-2-9b", "variant": "both", "stage": 19},
        inputs=[describe_input(v["path"]) for v in cells.values()]
        + [describe_input(HERE / "DESIGN_SRCDEC.md"), describe_input(HERE / "analyze_srcdec.py")],
        metric="srcdec_E1_socialness_and_E2_E5",
        values=values,
        threshold={"n_boot": N_BOOT, "boot_seed": BOOT_SEED, "ci": [CI_LO, CI_HI],
                   "resampling_unit": "item", "e2_ratio_cut": E2_RATIO_CUT,
                   "e3_sign": f"{E3_SIGN_NUM}/{E3_SIGN_DEN}", "e5_bands": [E5_LO, E5_HI]},
        decision_rule=DESIGN_QUOTE,
        decision=decision,
        notes="E1 is the primary endpoint; D_doc, the per-kind COPYGEN split, and every "
              "contrast not named E1-E5 are exploratory by pre-registration. Log-odds and "
              "margin_raw companions are reported for E1 because all three run-1 retractions "
              "were readout-denominator artifacts; a sign that survives only in the ratio is "
              "a mass effect wearing a contrast's clothes.",
        rows=None)
    print(f"srcdec: {decision}")
    return values


# ---------------------------------------------------------------------------
# Selftest
# ---------------------------------------------------------------------------

def _plant(tmp: Path, *, self_gap: float, srcless: float = 0.30, n_welfare: int = 82,
           copygen_scale: float = 1.0) -> Path:
    """A base BINARY cell with a planted per-item structure.

    Every item sits at a neutral ratio and moves +-half the named contrast under the YES
    and NO members of each source cell, so D_self = srcless + self_gap by construction and
    the recovered numbers can be checked against the plant. A small per-item wobble keeps
    the bootstrap from resampling a constant.
    """
    rows = []
    arms = ("SELF-YES", "SELF-NO", "SRCLESS-YES", "SRCLESS-NO", "DOC-YES", "DOC-NO",
            "NONSOCIAL", "NEUTRAL")
    items = ([(f"W{k:03d}", "welfare") for k in range(n_welfare)]
             + [(f"COPYGEN-{k:02d}", "copygen") for k in range(1, 21)])
    for k, (iid, cls) in enumerate(items):
        wob = 0.01 * ((k % 7) - 3)
        scale = copygen_scale if cls == "copygen" else 1.0
        d_self = (srcless + self_gap) * scale + wob
        d_src = srcless * scale + wob
        for arm in arms:
            half = {"SELF-YES": d_self / 2, "SELF-NO": -d_self / 2,
                    "SRCLESS-YES": d_src / 2, "SRCLESS-NO": -d_src / 2,
                    "DOC-YES": d_src / 2, "DOC-NO": -d_src / 2,
                    "NONSOCIAL": 0.0, "NEUTRAL": 0.0}[arm]
            v = max(min(0.05 * ((k % 5) - 2) + half, 0.95), -0.95)
            p_yes, p_no = (1 + v) / 2 * 0.8, (1 - v) / 2 * 0.8
            rows.append({"prompt": f"<{arm}> {iid}", "item_id": iid, "item_class": cls,
                         "fmt": "BINARY", "tier": "DERIVED", "arm": arm, "preamble": arm,
                         "p_yes": p_yes, "p_no": p_no, "yesno_mass": p_yes + p_no,
                         "margin_ratio": v, "margin_raw": p_yes - p_no,
                         "digit_mass": 0.0, "scale_expectation": 0.0,
                         "digit_distribution": [0.1] * 10, "top_token": "Yes",
                         "n_prompt_tokens": 40})
    p = tmp / "results" / "srcdec_stage-srcdec-binary.json"
    write_result(p, cell={"model": "google/gemma-2-9b", "variant": "base",
                          "stage": "srcdec-binary"},
                 inputs=[], metric="synthetic", values={}, decision_rule="synthetic",
                 decision="COMPLETE", rows=rows)
    return p


def _selftest() -> int:
    import tempfile

    fails: list[str] = []

    # (a) pure assertion-copy: the speaker adds nothing
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _plant(d, self_gap=0.0)
        v = main(d, d / "results" / ENDPOINTS)
        e1 = v["E1_socialness"]
        if e1["outcome"] != "ASSERTION-COPY":
            fails.append(f"planted copy: E1={e1['outcome']} "
                         f"(paired {e1['D_self_minus_D_srcless']['mean']:+.4f} "
                         f"CI {e1['D_self_minus_D_srcless']['ci95']})")
        if abs(e1["D_self"]["mean"] - 0.30) > 0.02:
            fails.append(f"planted copy: D_self {e1['D_self']['mean']:+.4f}, planted +0.30")
        if v["E3_confirmation"]["outcome"] != "CONFIRMED":
            fails.append(f"planted copy: E3={v['E3_confirmation']['outcome']} "
                         f"(n_pos {v['E3_confirmation']['n_positive']}/"
                         f"{v['E3_confirmation']['n_items']})")
        if v["E2_domain_generality"]["outcome"] != "COPY-GENERAL":
            fails.append(f"planted copy at equal size: E2="
                         f"{v['E2_domain_generality']['outcome']} "
                         f"{v['E2_domain_generality']['ratio_copygen_over_welfare']}")
        for k in ("E4_register_matched", "E5_second_family"):
            if v[k]["outcome"] != "NOT-RUN":
                fails.append(f"{k} claimed {v[k]['outcome']} with no cell run")

    # (b) a self-vs-sourceless gap: the speaker does work
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _plant(d, self_gap=0.20)
        v = main(d, d / "results" / ENDPOINTS)
        e1 = v["E1_socialness"]
        if e1["outcome"] != "SOCIAL-SOURCE":
            fails.append(f"planted gap: E1={e1['outcome']} "
                         f"(paired {e1['D_self_minus_D_srcless']['mean']:+.4f})")
        if abs(e1["D_self_minus_D_srcless"]["mean"] - 0.20) > 0.02:
            fails.append(f"planted gap recovered {e1['D_self_minus_D_srcless']['mean']:+.4f}, "
                         f"planted +0.20")

    # (c) welfare-specific copy: the non-welfare items barely move
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _plant(d, self_gap=0.0, copygen_scale=0.05)
        v = main(d, d / "results" / ENDPOINTS)
        if v["E2_domain_generality"]["outcome"] != "WELFARE-SPECIFIC":
            fails.append(f"planted welfare-specific: E2="
                         f"{v['E2_domain_generality']['outcome']} "
                         f"{v['E2_domain_generality']['ratio_copygen_over_welfare']}")

    # E4 and E5 on planted SCALE / second-family cells
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        rows = []
        for k in range(40):
            for arm, sh in (("RATE-HIGH", 0.5), ("RATE-LOW", -0.5), ("POLAR-YES", 0.05),
                            ("POLAR-NO", -0.05), ("NONSOCIAL", 0.0), ("NEUTRAL", 0.0)):
                rows.append({"prompt": f"<{arm}>{k}", "item_id": f"S{k}", "item_class": "welfare",
                             "fmt": "SCALE", "tier": "DERIVED", "arm": arm, "preamble": arm,
                             "p_yes": 0.01, "p_no": 0.01, "yesno_mass": 0.02,
                             "margin_ratio": 0.0, "margin_raw": 0.0, "digit_mass": 0.9,
                             "scale_expectation": 6.0 + sh + 0.02 * (k % 5),
                             "digit_distribution": [0.1] * 10, "top_token": "6",
                             "n_prompt_tokens": 40})
        write_result(d / "results" / "scale_stage-srcdec-scale.json",
                     cell={"model": "google/gemma-2-9b-it", "variant": "it",
                           "stage": "srcdec-scale"},
                     inputs=[], metric="synthetic", values={}, decision_rule="s",
                     decision="COMPLETE", rows=rows)
        v = main(d, d / "results" / ENDPOINTS)
        if v["E4_register_matched"]["outcome"] != "REVIVABLE-REGISTER-MATCHED":
            fails.append(f"planted rate effect: E4={v['E4_register_matched']['outcome']}")
        rows2 = json.loads(_plant(d, self_gap=0.0).read_text())["rows"]
        write_result(d / "results" / "qwen_stage-srcdec-binary.json",
                     cell={"model": "Qwen/Qwen3-8B-Base", "variant": "base",
                           "stage": "srcdec-binary"},
                     inputs=[], metric="synthetic", values={}, decision_rule="s",
                     decision="COMPLETE", rows=rows2)
        v = main(d, d / "results" / ENDPOINTS)
        if v["E5_second_family"]["outcome"] != "GENERALISES":
            fails.append(f"planted second family: E5={v['E5_second_family']['outcome']} "
                         f"{v['E5_second_family'].get('frac_positive')}")

    for f in fails:
        print(f"FAIL {f}")
    print("selftest:", "PASS" if not fails else f"{len(fails)} FAILURES")
    return 1 if fails else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="SRCDEC endpoints E1-E5")
    ap.add_argument("--root", default=str(HERE))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(_selftest())
    OUT.mkdir(parents=True, exist_ok=True)
    main(Path(a.root), OUT / ENDPOINTS)
