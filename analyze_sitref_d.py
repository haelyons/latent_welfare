"""SITREF-D: is the length-equalising filler inert? (DESIGN_SITREF_D.md)

WHY THIS EXISTS. Stage 25 came out ATTRIBUTION-CARRIES, and stage 27's triage queued
exactly one control against it. The filler that equalises arm lengths is matched in
COUNT, but it sits inside different packagings -- a quoted block in SELFQ, a multi-turn
skeleton in OTHERM -- so filler geometry is correlated with the attribution manipulation
it was introduced to neutralise. If geometry alone moves the report, part of what stage
25 read as attribution is pad.

So this run strips the outcome content out and measures the geometry by itself:

    D_q = PADQ - NEUTRAL-PAD     the SELFQ geometry, no attempts, no checker lines
    D_m = PADM - NEUTRAL-PAD     the OTHERM geometry, likewise

WHAT THIS CONTROL DELIBERATELY DOES NOT SUBTRACT. The frames stay in their pad arms:
SELFQ's "record of your answers" and OTHERM's disclaimer are still there. So a
frame-only effect -- the self-attributing sentence moving the report with nothing
attributed to it -- reads as FILLER-ACTIVE and taxes stage 25. The design says this in
so many words ("that is intended, not a leak"), because frame and geometry are exactly
the component stage 25 could not separate from zero, and a control that quietly excused
half of it would be measuring its own charity.

THRESHOLD, fixed before the data: 0.15 rating points, one sixth of the smallest
load-bearing stage-25 effect (A_quoted -0.609). Both components must sit inside it, with
CIs including 0, for the pad to be called inert.

Same two sealed halves as the rest of the family: stage 28 commits the blind value table
while `sealed` makes the item-set-d key unopenable (imported from analyze_sitref, which
patches `Path.open` as well as the module-level names -- the box runs Python 3.10, where
`Path.read_text` would otherwise slip past), stage 29 applies the key to that committed
table. No stage-17 or stage-25 dependency: every number here is in-run. If a stage-25
record happens to be present, the pad-corrected contrasts the design asks for under
FILLER-ACTIVE are reported alongside, marked with the cross-run caveat they deserve.

    python3 analyze_sitref_d.py            # stage 28, blind
    python3 analyze_sitref_d.py --unblind  # stage 29, the endpoint
    python3 analyze_sitref_d.py --selftest # all three outcomes planted and recovered
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import sitref_stimuli as S
from analyze_sitref import KeyFileTouched, mean, reg_value, sealed, stats  # noqa: F401
from provenance import describe_input, write_result

HERE = Path(__file__).parent
OUT = HERE / "results"
BLIND_PATH = OUT / "sitref-d-blind-table_stage-28.json"
ENDPOINT_PATH = OUT / "sitref-d-endpoint_stage-29.json"
KEY_D_NAME = "sitref_blind_key_d.json"
KEY_D = HERE / "out" / KEY_D_NAME
ITEM_SET = "d"

PRIMARY_ID = S.PRIMARY_ID
ARMS_D = S.ARMS_D
INERT_BAND = 0.15          # rating points; one sixth of stage-25's A_quoted (-0.609)

CONTRASTS: dict[str, tuple[str, str]] = {
    "D_q": (S.PADQ_ARM, "NEUTRAL-PAD"),
    "D_m": (S.PADM_ARM, "NEUTRAL-PAD"),
}

DESIGN_QUOTE = (
    "DESIGN_SITREF_D.md, 'Endpoint, fixed now', quoted verbatim: 'Per item and family, "
    "D_q = PADQ - NEUTRAL-PAD and D_m = PADM - NEUTRAL-PAD on the same register as "
    "stage-25, bootstrap over the 10 families, evaluated on the primary item: "
    "FILLER-INERT: both |mean| < 0.15 rating points AND both 95% CIs include 0, on the "
    "primary item. (0.15 is one sixth of the smallest load-bearing stage-25 effect, "
    "A_quoted -0.609.) FILLER-ACTIVE: either CI excludes 0 with |mean| >= 0.15 - the "
    "refuting outcome; stage-25's ATTRIBUTION-CARRIES then carries an asterisk sized by "
    "the measured filler effect and the pad-corrected contrasts are reported. Anything "
    "else: FILLER-MARGINAL, reported as measured.' The design also fixes: 'the frames "
    "themselves (SELFQ's \"record of your answers\", OTHERM's disclaimer) remain in "
    "their pad arms ... A frame-only effect ... therefore reads as FILLER-ACTIVE and "
    "taxes stage-25 - that is intended, not a leak.' "
    "IMPLEMENTATION READINGS, DECLARED: (a) the three labels are exhaustive and "
    "evaluated in the design's order -- FILLER-ACTIVE is tested as 'some component has "
    "CI excluding 0 AND |mean| >= 0.15', FILLER-INERT as 'both |mean| < 0.15 AND both "
    "CIs include 0', and everything remaining is FILLER-MARGINAL; the two named "
    "conditions cannot both hold, so the order is documentation rather than precedence. "
    "(b) The pad-corrected stage-25 contrasts are reported only when a stage-25 record "
    "is present, and are labelled cross-run: SITREF-C and SITREF-D are separate box "
    "sessions and the design forbids treating magnitudes across boxes as commensurable, "
    "so the correction is descriptive and never enters this record's decision."
)


# ---------------------------------------------------------------------------
# Record discovery
# ---------------------------------------------------------------------------

def _records(root: Path, stage, itemset: str = None) -> list:
    """Provenance records for a stage, optionally restricted to one item set.

    Not analyze_sitref.find_records: that de-duplicates on (model, variant), so the
    item-set-d record and the main run's record for the same checkpoint collide and one
    is dropped. Item set is part of the identity here.
    """
    seen: dict = {}
    for p in sorted(root.rglob("*.json")):
        if p.name.endswith("_summary.json"):
            continue
        try:
            rec = json.loads(p.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            continue
        c = rec.get("cell") if isinstance(rec, dict) else None
        if not isinstance(c, dict) or c.get("stage") != stage:
            continue
        if itemset is not None and c.get("itemset") != itemset:
            continue
        seen.setdefault((c.get("model"), c.get("variant"), c.get("itemset")), (p, rec))
    return list(seen.values())


def _one(root: Path, stage: str, itemset: str) -> tuple:
    got = _records(root, stage, itemset)
    if len(got) != 1:
        raise SystemExit(f"need exactly one item-set-{itemset} {stage} record under {root}, "
                         f"found {len(got)}")
    return got[0]


def _stage25(root: Path):
    """(primary contrasts, source path) from a stage-25 record if one is present.

    Optional by design: SITREF-D stands on its own numbers. This only feeds the
    descriptive pad correction.
    """
    for p, rec in _records(root, 25):
        prim = rec.get("values", {}).get("primary", {}).get("contrasts")
        if prim:
            return prim, str(p)
    return None, None


# ---------------------------------------------------------------------------
# Stage 28: BLIND
# ---------------------------------------------------------------------------

def stage_blind(root: Path = None, out_path: Path = None) -> dict:
    root = root or OUT
    out_path = out_path or BLIND_PATH
    with sealed(KEY_D_NAME):
        src, rec = _one(root, "sitref-battery", ITEM_SET)
        rows = rec["rows"]
        table = [{"item_id": r["item_id"], "family": r["family"], "fmt": r["fmt"],
                  "blind_label": r["blind_arm"], "value": reg_value(r),
                  "digit_mass": r["digit_mass"], "n_prompt_tokens": r["n_prompt_tokens"]}
                 for r in rows]
        labels = sorted({t["blind_label"] for t in table})
        items = sorted({t["item_id"] for t in table})
        fams = sorted({t["family"] for t in table})
        per_label_item = {}
        for iid in items:
            for lab in labels:
                sel = [t["value"] for t in table
                       if t["item_id"] == iid and t["blind_label"] == lab]
                per_label_item[f"{iid}|{lab}"] = {"n_families": len(sel),
                                                  "mean": mean(sel) if sel else None}
        pairwise = {}
        for iid in items:
            v = {(t["blind_label"], t["family"]): t["value"] for t in table
                 if t["item_id"] == iid}
            pr = {}
            for i, a in enumerate(labels):
                for b in labels[i + 1:]:
                    d = [v[(a, f)] - v[(b, f)] for f in fams if (a, f) in v and (b, f) in v]
                    if d:
                        pr[f"{a}_minus_{b}"] = stats(d)
            pairwise[iid] = pr
        tok = {lab: mean([t["n_prompt_tokens"] for t in table if t["blind_label"] == lab])
               for lab in labels}

    complete = (len(table) == len(items) * len(fams) * len(ARMS_D)
                and len(labels) == len(ARMS_D)
                and rec.get("decision") == "COMPLETE")
    write_result(
        out_path,
        cell={"model": "google/gemma-2-9b-it", "variant": "it", "stage": 28},
        inputs=[describe_input(src), describe_input(HERE / "analyze_sitref_d.py")],
        metric="sitref_d_blind_value_table_and_pairwise_label_contrasts",
        values={"n": len(table), "n_labels": len(labels), "n_items": len(items),
                "n_families": len(fams), "labels": labels, "items": items,
                "per_label_item": per_label_item, "pairwise_by_item": pairwise,
                "mean_prompt_tokens_by_label": tok,
                "tokeniser_length_spread": (max(tok.values()) / min(tok.values())) - 1.0
                if tok else None,
                "battery_decision": rec.get("decision")},
        threshold={"register": "SCALE=E[rating] over digits; BINARY=(p_yes-p_no)/(p_yes+p_no)",
                   "expected_rows": 3 * len(S.FAMILY_NAMES) * len(ARMS_D)},
        decision_rule="Stage 28 states no hypothesis and cannot: it does not know which "
                      f"label is which arm, and it runs inside a guard that raises if "
                      f"anything opens {KEY_D_NAME}. COMMITTED requires the contributing "
                      "battery to have reached COMPLETE, one value per (item, family, "
                      "label), and all three labels present; otherwise NOT_COMMITTED and "
                      "stage 29 must not run.",
        decision="COMMITTED" if complete else "NOT_COMMITTED",
        notes="Every pairwise label contrast is committed here, so after unblinding the "
              "endpoint is a relabelling of a number that already exists rather than a "
              "new computation chosen with the arms in view.",
        rows=table)
    print(f"stage 28 (blind): {len(table)} values, {len(labels)} labels -> {out_path.name}")
    return {"n": len(table), "complete": complete}


# ---------------------------------------------------------------------------
# Stage 29: UNBLIND
# ---------------------------------------------------------------------------

def _series(v: dict, fams: list, hi: str, lo: str) -> list:
    return [v[(hi, f)] - v[(lo, f)] for f in fams if (hi, f) in v and (lo, f) in v]


def _decide(c: dict) -> tuple:
    """The design's rule. The two named conditions are mutually exclusive, so the order
    below is documentation of the design's wording, not a precedence that could bite."""
    path = []
    for name in ("D_q", "D_m"):
        st = c[name]
        path.append(f"{name} {st['mean']:+.4f} CI [{st['ci95'][0]:+.4f},{st['ci95'][1]:+.4f}]"
                    f" |mean| {'<' if abs(st['mean']) < INERT_BAND else '>='} {INERT_BAND}, "
                    f"CI {'excludes' if st['excludes_zero'] else 'includes'} 0")
    active = [n for n in ("D_q", "D_m")
              if c[n]["excludes_zero"] and abs(c[n]["mean"]) >= INERT_BAND]
    inert = all(abs(c[n]["mean"]) < INERT_BAND and not c[n]["excludes_zero"]
                for n in ("D_q", "D_m"))
    if active:
        path.append(f"{', '.join(active)} clears the band with a CI excluding 0: the "
                    f"refuting outcome. Stage-25's ATTRIBUTION-CARRIES carries an "
                    f"asterisk sized by "
                    f"{max(abs(c[n]['mean']) for n in active):.3f} rating points.")
        return "FILLER-ACTIVE", path
    if inert:
        path.append("both components inside the band with CIs including 0")
        return "FILLER-INERT", path
    path.append("neither the inert nor the active condition holds: reported as measured")
    return "FILLER-MARGINAL", path


def stage_unblind(root: Path = None, blind_path: Path = None, key_path: Path = None,
                  out_path: Path = None) -> str:
    root = root or OUT
    blind_path = blind_path or BLIND_PATH
    key_path = key_path or KEY_D
    out_path = out_path or ENDPOINT_PATH
    if not blind_path.exists():
        raise SystemExit(f"commit the blind table first: {blind_path} missing")
    tab = json.loads(blind_path.read_text())
    if tab.get("decision") != "COMMITTED":
        raise SystemExit(f"blind table decision is {tab.get('decision')!r}; stage 29 runs "
                         f"only on a COMMITTED table")
    if not key_path.exists():
        raise SystemExit(f"item-set-d blind key missing: {key_path}")
    key = json.loads(key_path.read_text())["mapping_by_variant"]["it"]
    inv = {lab: arm for arm, lab in key.items()}
    if len(inv) != len(key) or sorted(key) != sorted(ARMS_D):
        raise SystemExit(f"item-set-d key is not a bijection over the three arms: {sorted(key)}")

    scr_src, scr_rec = _one(root, "sitref-screen", ITEM_SET)
    screen = scr_rec["values"]["per_item"]

    rows = tab["rows"]
    fams = sorted({r["family"] for r in rows})
    vals: dict = {}
    for r in rows:
        vals.setdefault(r["item_id"], {})[(inv[r["blind_label"]], r["family"])] = r["value"]

    per_item = {}
    for iid, v in sorted(vals.items()):
        c = {name: stats(_series(v, fams, hi, lo)) for name, (hi, lo) in CONTRASTS.items()}
        per_item[iid] = {"contrasts": c,
                         "screen_usable": bool(screen.get(iid, {}).get("usable", False)),
                         "abs_mean_vs_band": {k: abs(c[k]["mean"]) / INERT_BAND
                                              for k in CONTRASTS}}
    primary = per_item[PRIMARY_ID]
    decision, path = _decide(primary["contrasts"])
    if not primary["screen_usable"]:
        path.append("NOTE the primary item did not pass its NEUTRAL-PAD saturation "
                    "screen in this run; the endpoint is reported as measured and the "
                    "screen flag travels with it.")

    s25, s25_src = _stage25(root)
    pad_corrected = None
    if s25:
        # Descriptive only, and cross-run: SITREF-C and SITREF-D are separate sessions.
        pad_corrected = {
            "A_quoted_minus_D_q": s25["A_quoted"]["mean"] - primary["contrasts"]["D_q"]["mean"],
            "P_multi_minus_D_m": s25["P_multi"]["mean"] - primary["contrasts"]["D_m"]["mean"],
            "stage25_A_quoted": s25["A_quoted"]["mean"],
            "stage25_P_multi": s25["P_multi"]["mean"],
            "stage25_A_self": s25["A_self"]["mean"],
            "_caveat": "cross-run subtraction of means, descriptive only: stage 25 and "
                       "stage 29 are different box sessions and this record's decision "
                       "does not read it.",
            "_source": s25_src}

    write_result(
        out_path,
        cell={"model": "google/gemma-2-9b-it", "variant": "it", "stage": 29},
        inputs=[describe_input(blind_path), describe_input(key_path),
                describe_input(scr_src), describe_input(HERE / "DESIGN_SITREF_D.md"),
                describe_input(HERE / "analyze_sitref_d.py")]
        + ([describe_input(s25_src)] if s25_src else []),
        metric="sitref_d_pad_geometry_effect",
        values={"decision_path": path, "primary_item": PRIMARY_ID, "primary": primary,
                "per_item": per_item, "pad_corrected_stage25": pad_corrected,
                "screen": {i: per_item[i]["screen_usable"] for i in per_item},
                "inert_band": INERT_BAND},
        threshold={"inert_band_rating_points": INERT_BAND,
                   "resampling_unit": "task family (10)",
                   "band_provenance": "one sixth of stage-25's A_quoted (-0.609)"},
        decision_rule=DESIGN_QUOTE,
        decision=decision,
        notes="Stage 28 committed the blind table under a guard that raises on any attempt "
              "to open the item-set-d key; this stage applied the key to that committed "
              "table. The pad arms keep their frames by design, so a frame-only effect "
              "reads as FILLER-ACTIVE. The two non-primary items are the subjecthood "
              "mirror pair, carried for continuity with stage 25 and exploratory here.",
        rows=None)
    print(f"stage 29 (endpoint): {decision}")
    for p in path:
        print("   -", p)
    return decision


# ---------------------------------------------------------------------------
# Selftest
# ---------------------------------------------------------------------------

def _synth(tmp: Path, d_q: float, d_m: float, jitter: float = 0.02,
           with_stage25: bool = False) -> None:
    """An item-set-d battery + screen with the two pad effects planted.

    PADQ and PADM are displaced from NEUTRAL-PAD by the named amounts, spread unevenly
    over families (mean exactly the amount) so the bootstrap resamples something with
    variance. The jitter is per (family, ARM), not per family: a jitter shared by both
    members of a contrast cancels in the difference and leaves the bootstrap resampling
    a constant, which would make a zero-width CI look like a passing test of an interval
    rule. `jitter` scales it, so the same planted mean can be driven into a CI that
    includes 0 or one that excludes it -- which is what makes INERT, ACTIVE and MARGINAL
    separately reachable.
    """
    import run_sitref as R

    mapping = R.blind_permutation(S.ARMS_D, R.BLIND_LABELS_D)
    (tmp / "out").mkdir(parents=True, exist_ok=True)
    (tmp / "out" / KEY_D_NAME).write_text(json.dumps(
        {"labels": list(R.BLIND_LABELS_D), "mapping_by_variant": {"it": mapping}}, indent=2))

    amp = {S.PADQ_ARM: d_q, S.PADM_ARM: d_m, "NEUTRAL-PAD": 0.0}
    items = S.build_sitref_items(ITEM_SET)
    rows, per_item = [], {}
    for item in items:
        iid = item["id"]
        for fi, fam in enumerate(S.FAMILY_NAMES):
            for ai, arm in enumerate(ARMS_D):
                shift = amp[arm] + jitter * (((fi + 2 * ai) % 5) - 2)
                rows.append({"prompt": f"<{arm}> {fam} {iid}", "item_id": iid,
                             "fmt": "SCALE", "role": item["role"], "family": fam,
                             "blind_arm": mapping[arm], "p_yes": 0.01, "p_no": 0.01,
                             "yesno_mass": 0.02, "margin_ratio": 0.0, "margin_raw": 0.0,
                             "digit_mass": 0.9, "scale_expectation": 7.0 + shift,
                             "digit_distribution": [0.1] * 10, "top_token": "7",
                             "n_prompt_tokens": 110})
        per_item[iid] = {"usable": True, "fmt": "SCALE", "role": item["role"],
                         "reason": "synthetic"}

    cell = {"model": "google/gemma-2-9b-it", "variant": "it", "itemset": ITEM_SET}
    write_result(tmp / "res" / "itemset-d_stage-sitref-battery.json",
                 cell={**cell, "stage": "sitref-battery"}, inputs=[], metric="synthetic",
                 values={}, decision_rule="synthetic", decision="COMPLETE", rows=rows)
    write_result(tmp / "res" / "itemset-d_stage-sitref-screen.json",
                 cell={**cell, "stage": "sitref-screen"}, inputs=[], metric="synthetic",
                 values={"per_item": per_item, "primary_usable": True},
                 decision_rule="synthetic", decision="PRIMARY_SCREEN_OK")
    if with_stage25:
        write_result(tmp / "res" / "sitref-c-endpoint_stage-25.json",
                     cell={"model": "google/gemma-2-9b-it", "variant": "it", "stage": 25},
                     inputs=[], metric="synthetic",
                     values={"primary": {"contrasts": {
                         "A_self": {"mean": -1.0}, "A_quoted": {"mean": -0.609},
                         "P_multi": {"mean": -0.05}, "P_other": {"mean": 0.0}}}},
                     decision_rule="synthetic", decision="ATTRIBUTION-CARRIES")


def _selftest() -> int:
    import tempfile

    fails = []

    # --- the seal holds on the item-set-d key, and only on it
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / KEY_D_NAME).write_text("{}")
        (d / "other.json").write_text("{}")
        (d / "sitref_blind_key_c.json").write_text("{}")
        for how, fn in (("Path.read_text", lambda: (d / KEY_D_NAME).read_text()),
                        ("Path.open", lambda: (d / KEY_D_NAME).open().close()),
                        ("builtins.open", lambda: open(d / KEY_D_NAME).close())):
            try:
                with sealed(KEY_D_NAME):
                    fn()
                fails.append(f"guard did not fire on {how} of the d key")
            except KeyFileTouched:
                pass
        with sealed(KEY_D_NAME):
            try:
                (d / "other.json").read_text()
                (d / "sitref_blind_key_c.json").read_text()
            except KeyFileTouched:
                fails.append("d guard sealed a file that is not the d key")

    # --- the three named outcomes, planted
    cases = [
        ("FILLER-INERT", 0.0, 0.0, 0.02),          # nothing moves
        ("FILLER-INERT", 0.05, -0.04, 0.10),       # inside the band, CIs straddle 0
        ("FILLER-ACTIVE", -0.40, 0.0, 0.02),       # one component clears the band
        ("FILLER-ACTIVE", -0.30, -0.25, 0.02),     # both do
        ("FILLER-MARGINAL", 0.30, 0.0, 0.35),      # >= band but too noisy to exclude 0
        ("FILLER-MARGINAL", 0.05, 0.0, 0.002),     # CI excludes 0 but inside the band
    ]
    for want, q, m, jit in cases:
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            _synth(d, q, m, jitter=jit)
            with sealed(KEY_D_NAME):
                bl = stage_blind(d / "res", d / "res" / "blind.json")
            if not bl["complete"]:
                fails.append(f"{want}: blind table not COMMITTED")
            got = stage_unblind(d / "res", d / "res" / "blind.json",
                                d / "out" / KEY_D_NAME, d / "res" / "end.json")
            if got != want:
                fails.append(f"planted D_q={q:+.2f} D_m={m:+.2f} jitter={jit} -> {got}, "
                             f"want {want}")
                continue
            rec = json.loads((d / "res" / "end.json").read_text())
            c = rec["values"]["primary"]["contrasts"]
            for name, planted in (("D_q", q), ("D_m", m)):
                if abs(c[name]["mean"] - planted) > 1e-6:
                    fails.append(f"{want}: {name} recovered {c[name]['mean']:+.4f}, "
                                 f"planted {planted:+.4f}")

    # --- the band is the design's, and it bites exactly at 0.15
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        if INERT_BAND != 0.15:
            fails.append(f"inert band is {INERT_BAND}, design fixes 0.15")
        _synth(d, -0.15, 0.0, jitter=0.002)      # exactly at the band, tight CI
        with sealed(KEY_D_NAME):
            stage_blind(d / "res", d / "res" / "blind.json")
        if stage_unblind(d / "res", d / "res" / "blind.json", d / "out" / KEY_D_NAME,
                         d / "res" / "end.json") != "FILLER-ACTIVE":
            fails.append("|mean| exactly at the band did not read FILLER-ACTIVE "
                         "(the design's clause is >=)")

    # --- pad correction appears only when a stage-25 record does, and never decides
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _synth(d, -0.40, 0.0, jitter=0.02, with_stage25=True)
        with sealed(KEY_D_NAME):
            stage_blind(d / "res", d / "res" / "blind.json")
        got = stage_unblind(d / "res", d / "res" / "blind.json", d / "out" / KEY_D_NAME,
                            d / "res" / "end.json")
        rec = json.loads((d / "res" / "end.json").read_text())
        pc = rec["values"]["pad_corrected_stage25"]
        if got != "FILLER-ACTIVE":
            fails.append(f"stage-25 present changed the decision to {got}")
        if not pc or abs(pc["A_quoted_minus_D_q"] - (-0.609 - -0.40)) > 1e-9:
            fails.append(f"pad correction wrong: {pc}")
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _synth(d, 0.0, 0.0, jitter=0.02, with_stage25=False)
        with sealed(KEY_D_NAME):
            stage_blind(d / "res", d / "res" / "blind.json")
        stage_unblind(d / "res", d / "res" / "blind.json", d / "out" / KEY_D_NAME,
                      d / "res" / "end.json")
        rec = json.loads((d / "res" / "end.json").read_text())
        if rec["values"]["pad_corrected_stage25"] is not None:
            fails.append("pad correction invented with no stage-25 record present")

    # --- the committed table carries no arm name, and stage 29 refuses an uncommitted one
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _synth(d, 0.0, 0.0)
        with sealed(KEY_D_NAME):
            stage_blind(d / "res", d / "res" / "blind.json")
        blob = (d / "res" / "blind.json").read_text()
        leaked = [a for a in S.ALL_ARMS if a in blob]
        if leaked:
            fails.append(f"true arm names in the committed blind table: {leaked}")
        t = json.loads(blob)
        t["decision"] = "NOT_COMMITTED"
        (d / "res" / "blind.json").write_text(json.dumps(t))
        try:
            stage_unblind(d / "res", d / "res" / "blind.json", d / "out" / KEY_D_NAME,
                          d / "res" / "end.json")
            fails.append("stage 29 ran on a NOT_COMMITTED table")
        except SystemExit:
            pass

    for f in fails:
        print(f"FAIL {f}")
    print(f"{len(cases)} planted outcomes over {len(ARMS_D)} arms; band {INERT_BAND}; "
          f"{3 * len(S.FAMILY_NAMES) * len(ARMS_D)} synthetic rows each")
    print("selftest:", "PASS" if not fails else f"{len(fails)} FAILURES")
    return 1 if fails else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="SITREF-D: is the pad geometry inert?")
    ap.add_argument("--unblind", action="store_true", help="stage 29: apply the key")
    ap.add_argument("--root", default=str(OUT), help="tree to search for records")
    ap.add_argument("--key", default=str(KEY_D))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(_selftest())
    OUT.mkdir(parents=True, exist_ok=True)
    if a.unblind:
        stage_unblind(Path(a.root), BLIND_PATH, Path(a.key), ENDPOINT_PATH)
    else:
        with sealed(KEY_D_NAME):
            stage_blind(Path(a.root), BLIND_PATH)
