"""SITREF-C: is the asymmetry about attribution, or about packaging? (DESIGN_SITREF_C.md)

WHY THIS EXISTS. Stage 17 measured a ~15x asymmetry -- self-attributed failure moves
self-directed reports far more than identical other-attributed content -- and stage 22
then fired the trait guard's refuting outcome, so the surviving reading is a
self-attribution-gated GLOBAL self-evaluation shift rather than state-specific
reference. The asymmetry itself survived every triage pass. But in those five arms
attribution was perfectly collinear with PACKAGING: SELF arms were multi-turn with
assistant role tokens, OTHER arms were one quoted user turn. No statistic computed on
that design can say which of the two the asymmetry rides on, because no cell varied one
while holding the other. SITREF-C adds the two missing cells and this file reads them.

  A_quoted = SELFQ-FAIL - SELFQ-SUCC    self-attribution WITHOUT role-token packaging
  P_multi  = OTHERM-FAIL - OTHERM-SUCC  role-token packaging WITHOUT self-attribution

THE ANCHORS ARE PART OF THE MEASUREMENT, not a courtesy. Every threshold below is a
fraction of A_self measured IN THIS RUN, never of stage-17's number: the two runs are
different boxes, and a magnitude compared across boxes is a magnitude nobody can defend.
If the in-run A_self does not reproduce -- CI including 0, or a sign flip against stage
17 -- the run is VOID-ANCHORS and no component is interpreted, because the yardstick
every threshold is stated in has failed.

SAME TWO SEALED HALVES as analyze_sitref.py: stage 24 commits the blind
per-(label, item, family) value table while `sealed` makes the item-set-c key file
unopenable, and stage 25 applies the key to that committed table. The guard is imported
rather than reimplemented -- it patches `Path.open` as well as the module-level `open`
names, which is what makes it hold on the box's Python 3.10, where `Path.read_text`
routes through an accessor that captured `io.open` at import time.

    python3 analyze_sitref_c.py            # stage 24, blind
    python3 analyze_sitref_c.py --unblind  # stage 25, the endpoint
    python3 analyze_sitref_c.py --selftest # every named outcome planted and recovered
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
BLIND_PATH = OUT / "sitref-c-blind-table_stage-24.json"
ENDPOINT_PATH = OUT / "sitref-c-endpoint_stage-25.json"
KEY_C_NAME = "sitref_blind_key_c.json"
KEY_C = HERE / "out" / KEY_C_NAME
ITEM_SET = "c"

PRIMARY_ID = S.PRIMARY_ID
ARMS_C = S.ARMS_C

# The four per-family contrasts. Names are the design's.
CONTRASTS: dict[str, tuple[str, str]] = {
    "A_self": ("SELF-FAIL", "SELF-SUCC"),
    "A_quoted": ("SELFQ-FAIL", "SELFQ-SUCC"),
    "P_other": ("OTHER-FAIL", "OTHER-SUCC"),
    "P_multi": ("OTHERM-FAIL", "OTHERM-SUCC"),
}
HALF, QUARTER = 0.5, 0.25

DESIGN_QUOTE = (
    "DESIGN_SITREF_C.md, 'SITREF-C endpoints, fixed now', quoted verbatim: 'On the "
    "primary item, per the same 10 families: A_quoted = SELFQ-FAIL - SELFQ-SUCC "
    "(attribution without role-token packaging); P_multi = OTHERM-FAIL - OTHERM-SUCC "
    "(role-token packaging without self-attribution). In-run anchors: A_self = SELF-FAIL "
    "- SELF-SUCC; P_other = OTHER-FAIL - OTHER-SUCC. ATTRIBUTION-CARRIES: A_quoted CI "
    "below 0 AND |A_quoted| >= |A_self|/2 AND P_multi CI including 0 or |P_multi| < "
    "|A_self|/4. PACKAGING-CARRIES: P_multi CI below 0 with |P_multi| >= |A_self|/2 AND "
    "A_quoted CI including 0 or |A_quoted| < |A_self|/4. BOTH / NEITHER: any other "
    "pattern, reported as measured (each component's CI and fraction of A_self stated); "
    "no stronger label is claimed. Anchors failing to reproduce (A_self CI including 0, "
    "or sign flip vs stage-17) void the run: VOID-ANCHORS.' "
    "IMPLEMENTATION READING, DECLARED, for the residual patterns the design assigns to "
    "BOTH/NEITHER without splitting them: define carries(X) = X's CI below 0 AND |X| >= "
    "|A_self|/2, and negligible(X) = X's CI includes 0 OR |X| < |A_self|/4. Then BOTH is "
    "reported when both components carry, or when one carries and the other is neither "
    "carrying nor negligible (it sits between the quarter and the half, so it is "
    "contributing and the design's stronger single-factor label is unavailable); NEITHER "
    "is reported when no component carries. Both labels are reported as measured, with "
    "every component's CI, sign and fraction of |A_self| in values. A screen-failed "
    "primary item is reported VOID-ANCHORS with that reason named, since the anchors are "
    "defined on the primary item and cannot be established on a saturated readout."
)


# ---------------------------------------------------------------------------
# Record discovery
# ---------------------------------------------------------------------------

def _records(root: Path, stage, itemset: str = None) -> list:
    """Provenance records for a stage, optionally restricted to one item set.

    Deliberately NOT analyze_sitref.find_records: that one keys its de-duplication on
    (model, variant), so an item-set-c record and the main run's record for the same
    checkpoint collide and one is dropped. Item set is part of the identity here.
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


def _stage17_self_fail(root: Path) -> tuple:
    """(mean, source path) of stage-17's in-run A_self, for the sign-flip anchor check."""
    for p, rec in _records(root, 17):
        cells = rec.get("values", {}).get("per_cell", {})
        for name, c in cells.items():
            if name.endswith("|it") and "primary" in c and "self_fail" in c["primary"]:
                return float(c["primary"]["self_fail"]["mean"]), str(p)
    raise SystemExit(f"no stage-17 endpoint record with an it-cell primary under {root}: "
                     f"the sign-flip anchor check cannot be evaluated")


# ---------------------------------------------------------------------------
# Stage 24: BLIND
# ---------------------------------------------------------------------------

def stage_blind(root: Path = None, out_path: Path = None) -> dict:
    root = root or OUT
    out_path = out_path or BLIND_PATH
    with sealed(KEY_C_NAME):
        src, rec = _one(root, "sitref-battery", ITEM_SET)
        rows = rec["rows"]
        table = [{"item_id": r["item_id"], "family": r["family"], "fmt": r["fmt"],
                  "blind_label": r["blind_arm"], "value": reg_value(r),
                  "digit_mass": r["digit_mass"], "n_prompt_tokens": r["n_prompt_tokens"]}
                 for r in rows]
        labels = sorted({t["blind_label"] for t in table})
        items = sorted({t["item_id"] for t in table})
        fams = sorted({t["family"] for t in table})
        # Blind summaries: per (label, item) means and every pairwise label contrast.
        # Computable without knowing which label is which arm, which is the point.
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

    complete = (len(table) == len(items) * len(fams) * len(ARMS_C)
                and len(labels) == len(ARMS_C)
                and rec.get("decision") == "COMPLETE")
    write_result(
        out_path,
        cell={"model": "google/gemma-2-9b-it", "variant": "it", "stage": 24},
        inputs=[describe_input(src), describe_input(HERE / "analyze_sitref_c.py")],
        metric="sitref_c_blind_value_table_and_pairwise_label_contrasts",
        values={"n": len(table), "n_labels": len(labels), "n_items": len(items),
                "n_families": len(fams), "labels": labels, "items": items,
                "per_label_item": per_label_item, "pairwise_by_item": pairwise,
                "mean_prompt_tokens_by_label": tok,
                "tokeniser_length_spread": (max(tok.values()) / min(tok.values())) - 1.0
                if tok else None,
                "battery_decision": rec.get("decision")},
        threshold={"register": "SCALE=E[rating] over digits; BINARY=(p_yes-p_no)/(p_yes+p_no)",
                   "expected_rows": 3 * len(S.FAMILY_NAMES) * len(ARMS_C)},
        decision_rule="Stage 24 states no hypothesis and cannot: it does not know which "
                      f"label is which arm, and it runs inside a guard that raises if "
                      f"anything opens {KEY_C_NAME}. COMMITTED requires the contributing "
                      "battery to have reached COMPLETE, one value per (item, family, "
                      "label), and all nine labels present; otherwise NOT_COMMITTED and "
                      "stage 25 must not run.",
        decision="COMMITTED" if complete else "NOT_COMMITTED",
        notes="Rows are the value table the endpoint is computed from. Pairwise label "
              "contrasts are committed here so the endpoint's arithmetic is fixed before "
              "the key is applied: after unblinding, every number below is a relabelling "
              "of one of these.",
        rows=table)
    print(f"stage 24 (blind): {len(table)} values, {len(labels)} labels -> {out_path.name}")
    return {"n": len(table), "complete": complete}


# ---------------------------------------------------------------------------
# Stage 25: UNBLIND
# ---------------------------------------------------------------------------

def _series(v: dict, fams: list, hi: str, lo: str) -> list:
    return [v[(hi, f)] - v[(lo, f)] for f in fams if (hi, f) in v and (lo, f) in v]


def _below_zero(st: dict) -> bool:
    """'CI below 0' in the design's sense: the whole interval is negative."""
    return st["ci95"][1] < 0


def _decide(c: dict, s17_self: float, primary_usable: bool) -> tuple:
    path = []
    a_self, a_q, p_m = c["A_self"], c["A_quoted"], c["P_multi"]
    path.append(f"anchor A_self {a_self['mean']:+.3f} CI [{a_self['ci95'][0]:+.3f},"
                f"{a_self['ci95'][1]:+.3f}]; stage-17 A_self {s17_self:+.3f}")
    path.append(f"anchor P_other {c['P_other']['mean']:+.3f} CI "
                f"[{c['P_other']['ci95'][0]:+.3f},{c['P_other']['ci95'][1]:+.3f}]")
    if not primary_usable:
        path.append("primary item failed its NEUTRAL-PAD saturation screen; the anchors "
                    "are defined on it and cannot be established")
        return "VOID-ANCHORS", path
    if not a_self["excludes_zero"]:
        path.append("A_self CI includes 0: the yardstick every threshold is stated in "
                    "did not reproduce")
        return "VOID-ANCHORS", path
    if (a_self["mean"] < 0) != (s17_self < 0):
        path.append("A_self sign flipped against stage-17")
        return "VOID-ANCHORS", path

    ref = abs(a_self["mean"])
    for name, st in (("A_quoted", a_q), ("P_multi", p_m)):
        path.append(f"{name} {st['mean']:+.3f} CI [{st['ci95'][0]:+.3f},{st['ci95'][1]:+.3f}]"
                    f" = {abs(st['mean']) / ref:.2f} x |A_self|")

    def carries(st):
        return _below_zero(st) and abs(st["mean"]) >= HALF * ref

    def negligible(st):
        return (not st["excludes_zero"]) or abs(st["mean"]) < QUARTER * ref

    ca, cp = carries(a_q), carries(p_m)
    if ca and negligible(p_m):
        return "ATTRIBUTION-CARRIES", path
    if cp and negligible(a_q):
        return "PACKAGING-CARRIES", path
    if ca or cp:
        path.append("one component carries and the other is neither carrying nor "
                    "negligible, or both carry: reported as measured")
        return "BOTH", path
    path.append("no component reaches half of |A_self| with a CI below 0")
    return "NEITHER", path


def stage_unblind(root: Path = None, blind_path: Path = None, key_path: Path = None,
                  out_path: Path = None) -> str:
    root = root or OUT
    blind_path = blind_path or BLIND_PATH
    key_path = key_path or KEY_C
    out_path = out_path or ENDPOINT_PATH
    if not blind_path.exists():
        raise SystemExit(f"commit the blind table first: {blind_path} missing")
    tab = json.loads(blind_path.read_text())
    if tab.get("decision") != "COMMITTED":
        raise SystemExit(f"blind table decision is {tab.get('decision')!r}; stage 25 runs "
                         f"only on a COMMITTED table")
    if not key_path.exists():
        raise SystemExit(f"item-set-c blind key missing: {key_path}")
    key = json.loads(key_path.read_text())["mapping_by_variant"]["it"]
    inv = {lab: arm for arm, lab in key.items()}
    if len(inv) != len(key) or sorted(key) != sorted(ARMS_C):
        raise SystemExit(f"item-set-c key is not a bijection over the nine arms: {sorted(key)}")

    scr_src, scr_rec = _one(root, "sitref-screen", ITEM_SET)
    screen = scr_rec["values"]["per_item"]
    s17_self, s17_src = _stage17_self_fail(root)

    rows = tab["rows"]
    fams = sorted({r["family"] for r in rows})
    vals: dict = {}
    for r in rows:
        vals.setdefault(r["item_id"], {})[(inv[r["blind_label"]], r["family"])] = r["value"]

    per_item = {}
    for iid, v in sorted(vals.items()):
        c = {name: stats(_series(v, fams, hi, lo)) for name, (hi, lo) in CONTRASTS.items()}
        for arm in ARMS_C:
            if arm != "NEUTRAL-PAD":
                c[f"{arm}_vs_NEUTRAL"] = stats(_series(v, fams, arm, "NEUTRAL-PAD"))
        ref = abs(c["A_self"]["mean"]) or float("nan")
        per_item[iid] = {
            "contrasts": c,
            "fraction_of_A_self": {k: abs(c[k]["mean"]) / ref for k in CONTRASTS},
            "sign": {k: (1 if c[k]["mean"] > 0 else -1 if c[k]["mean"] < 0 else 0)
                     for k in CONTRASTS},
            "screen_usable": bool(screen.get(iid, {}).get("usable", False)),
        }

    primary = per_item[PRIMARY_ID]
    decision, path = _decide(primary["contrasts"], s17_self,
                             primary_usable=primary["screen_usable"])

    # Secondary, exploratory by pre-registration: the mirror pair rides along as the
    # acquiescence control. A genuine state effect moves the two members in OPPOSITE
    # directions; a response bias moves them the same way, in every packaging.
    mirror = {}
    pair = [i for i in S.ITEM_IDS_C if i != PRIMARY_ID]
    if len(pair) == 2 and all(i in per_item for i in pair):
        a, b = pair
        mirror = {"pair": pair,
                  "opposite_signs": {k: bool(per_item[a]["contrasts"][k]["mean"]
                                             * per_item[b]["contrasts"][k]["mean"] < 0)
                                     for k in CONTRASTS},
                  "means": {k: [per_item[a]["contrasts"][k]["mean"],
                                per_item[b]["contrasts"][k]["mean"]] for k in CONTRASTS}}

    write_result(
        out_path,
        cell={"model": "google/gemma-2-9b-it", "variant": "it", "stage": 25},
        inputs=[describe_input(blind_path), describe_input(key_path),
                describe_input(scr_src), describe_input(s17_src),
                describe_input(HERE / "DESIGN_SITREF_C.md"),
                describe_input(HERE / "analyze_sitref_c.py")],
        metric="sitref_c_attribution_vs_packaging",
        values={"decision_path": path, "primary_item": PRIMARY_ID,
                "primary": primary, "per_item": per_item, "mirror_pair": mirror,
                "stage17_A_self_mean": s17_self,
                "thresholds_in_units_of_A_self": {
                    "carries_at": HALF, "negligible_below": QUARTER,
                    "A_self_abs": abs(primary["contrasts"]["A_self"]["mean"])},
                "screen": {i: per_item[i]["screen_usable"] for i in per_item}},
        threshold={"resampling_unit": "task family (10)", "carries": HALF,
                   "negligible": QUARTER},
        decision_rule=DESIGN_QUOTE,
        decision=decision,
        notes="Stage 24 committed the blind table under a guard that raises on any attempt "
              "to open the item-set-c key; this stage applied the key to that committed "
              "table. Every threshold is a fraction of the IN-RUN A_self, never of stage "
              "17's, because the two are different boxes. The mirror pair and the "
              "arm-vs-NEUTRAL contrasts are exploratory by pre-registration.",
        rows=None)
    print(f"stage 25 (endpoint): {decision}")
    for p in path:
        print("   -", p)
    return decision


# ---------------------------------------------------------------------------
# Selftest
# ---------------------------------------------------------------------------

def _synth(tmp: Path, a_self: float, a_quoted: float, p_multi: float,
           p_other: float = 0.0, primary_usable: bool = True) -> None:
    """A complete item-set-c battery + screen + stage-17 record with planted contrasts.

    Each FAIL arm is displaced from its SUCC partner by the named amplitude, spread
    unevenly over families (mean exactly the amplitude, +-10%) so the bootstrap has
    variance to resample instead of a constant.
    """
    import run_sitref as R

    mapping = R.blind_permutation(S.ARMS_C, R.BLIND_LABELS_C)
    (tmp / "out").mkdir(parents=True, exist_ok=True)
    (tmp / "out" / KEY_C_NAME).write_text(json.dumps(
        {"labels": list(R.BLIND_LABELS_C), "mapping_by_variant": {"it": mapping}}, indent=2))

    amp = {"SELF-FAIL": a_self, "SELFQ-FAIL": a_quoted,
           "OTHERM-FAIL": p_multi, "OTHER-FAIL": p_other}
    items = S.build_sitref_items(ITEM_SET)
    rows, per_item = [], {}
    for item in items:
        iid = item["id"]
        for fi, fam in enumerate(S.FAMILY_NAMES):
            for arm in ARMS_C:
                shift = amp.get(arm, 0.0) * (1 + 0.1 * ((fi % 5) - 2))
                v = 7.0 + shift + 0.01 * ((fi % 3) - 1)
                rows.append({"prompt": f"<{arm}> {fam} {iid}", "item_id": iid,
                             "fmt": "SCALE", "role": item["role"], "family": fam,
                             "blind_arm": mapping[arm], "p_yes": 0.01, "p_no": 0.01,
                             "yesno_mass": 0.02, "margin_ratio": 0.0, "margin_raw": 0.0,
                             "digit_mass": 0.9, "scale_expectation": v,
                             "digit_distribution": [0.1] * 10, "top_token": "7",
                             "n_prompt_tokens": 120})
        per_item[iid] = {"usable": bool(primary_usable or iid != PRIMARY_ID),
                         "fmt": "SCALE", "role": item["role"], "reason": "synthetic"}

    cell = {"model": "google/gemma-2-9b-it", "variant": "it", "itemset": ITEM_SET}
    write_result(tmp / "res" / "itemset-c_stage-sitref-battery.json",
                 cell={**cell, "stage": "sitref-battery"}, inputs=[], metric="synthetic",
                 values={}, decision_rule="synthetic", decision="COMPLETE", rows=rows)
    write_result(tmp / "res" / "itemset-c_stage-sitref-screen.json",
                 cell={**cell, "stage": "sitref-screen"}, inputs=[], metric="synthetic",
                 values={"per_item": per_item,
                         "primary_usable": per_item[PRIMARY_ID]["usable"]},
                 decision_rule="synthetic", decision="PRIMARY_SCREEN_OK")
    write_result(tmp / "res" / "sitref-endpoint_stage-17.json",
                 cell={"model": "google/gemma-2-9b", "variant": "both", "stage": 17},
                 inputs=[], metric="synthetic",
                 values={"per_cell": {"google/gemma-2-9b-it|it":
                                      {"primary": {"self_fail": {"mean": -0.8718}}}}},
                 decision_rule="synthetic", decision="INDETERMINATE")


def _selftest() -> int:
    import tempfile

    fails = []

    # --- the seal holds on the item-set-c key, including through Path.read_text
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / KEY_C_NAME).write_text("{}")
        (d / "other.json").write_text("{}")
        for how, fn in (("Path.read_text", lambda: (d / KEY_C_NAME).read_text()),
                        ("Path.open", lambda: (d / KEY_C_NAME).open().close()),
                        ("builtins.open", lambda: open(d / KEY_C_NAME).close())):
            try:
                with sealed(KEY_C_NAME):
                    fn()
                fails.append(f"guard did not fire on {how} of the c key")
            except KeyFileTouched:
                pass
        with sealed(KEY_C_NAME):
            if json.loads((d / "other.json").read_text()) != {}:
                fails.append("guard blocked an unrelated file")
        # the main run's key must NOT be sealed by the c guard, and vice versa: the two
        # blinds are separate files and each stage seals its own
        (d / "sitref_blind_key.json").write_text("{}")
        with sealed(KEY_C_NAME):
            try:
                (d / "sitref_blind_key.json").read_text()
            except KeyFileTouched:
                fails.append("c guard sealed the main run's key file")

    cases = [
        ("ATTRIBUTION-CARRIES", dict(a_self=-1.0, a_quoted=-0.9, p_multi=-0.05)),
        ("PACKAGING-CARRIES", dict(a_self=-1.0, a_quoted=-0.05, p_multi=-0.9)),
        ("BOTH", dict(a_self=-1.0, a_quoted=-0.7, p_multi=-0.7)),
        ("NEITHER", dict(a_self=-1.0, a_quoted=0.0, p_multi=0.0)),
        ("VOID-ANCHORS", dict(a_self=0.0, a_quoted=-0.9, p_multi=-0.05)),
        ("VOID-ANCHORS", dict(a_self=+1.0, a_quoted=-0.9, p_multi=-0.05)),   # sign flip
    ]
    for want, kw in cases:
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            _synth(d, **kw)
            with sealed(KEY_C_NAME):
                bl = stage_blind(d / "res", d / "res" / "blind.json")
            if not bl["complete"]:
                fails.append(f"{want}: blind table not COMMITTED")
            got = stage_unblind(d / "res", d / "res" / "blind.json",
                                d / "out" / KEY_C_NAME, d / "res" / "end.json")
            if got != want:
                fails.append(f"planted {kw} -> {got}, want {want}")
            rec = json.loads((d / "res" / "end.json").read_text())
            if want != "VOID-ANCHORS":
                p = rec["values"]["primary"]["contrasts"]
                for name, planted in (("A_self", kw["a_self"]), ("A_quoted", kw["a_quoted"]),
                                      ("P_multi", kw["p_multi"])):
                    if abs(p[name]["mean"] - planted) > 1e-6:
                        fails.append(f"{want}: {name} recovered {p[name]['mean']:+.4f}, "
                                     f"planted {planted:+.4f}")
                if abs(p["P_other"]["mean"]) > 1e-6:
                    fails.append(f"{want}: P_other moved with nothing planted")

    # --- a screen-failed primary voids the anchors
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _synth(d, a_self=-1.0, a_quoted=-0.9, p_multi=-0.05, primary_usable=False)
        with sealed(KEY_C_NAME):
            stage_blind(d / "res", d / "res" / "blind.json")
        got = stage_unblind(d / "res", d / "res" / "blind.json", d / "out" / KEY_C_NAME,
                            d / "res" / "end.json")
        if got != "VOID-ANCHORS":
            fails.append(f"screen-failed primary -> {got}, want VOID-ANCHORS")

    # --- the blind table must not contain a true arm name, and stage 25 must refuse an
    #     uncommitted table
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _synth(d, a_self=-1.0, a_quoted=-0.9, p_multi=-0.05)
        with sealed(KEY_C_NAME):
            stage_blind(d / "res", d / "res" / "blind.json")
        blob = (d / "res" / "blind.json").read_text()
        leaked = [a for a in ARMS_C if a in blob]
        if leaked:
            fails.append(f"true arm names in the committed blind table: {leaked}")
        t = json.loads(blob)
        t["decision"] = "NOT_COMMITTED"
        (d / "res" / "blind.json").write_text(json.dumps(t))
        try:
            stage_unblind(d / "res", d / "res" / "blind.json", d / "out" / KEY_C_NAME,
                          d / "res" / "end.json")
            fails.append("stage 25 ran on a NOT_COMMITTED table")
        except SystemExit:
            pass

    for f in fails:
        print(f"FAIL {f}")
    print(f"{len(cases)} planted outcomes over {len(ARMS_C)} arms; "
          f"{3 * len(S.FAMILY_NAMES) * len(ARMS_C)} synthetic rows each")
    print("selftest:", "PASS" if not fails else f"{len(fails)} FAILURES")
    return 1 if fails else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="SITREF-C: attribution vs packaging")
    ap.add_argument("--unblind", action="store_true", help="stage 25: apply the key")
    ap.add_argument("--root", default=str(OUT), help="tree to search for records")
    ap.add_argument("--key", default=str(KEY_C))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(_selftest())
    OUT.mkdir(parents=True, exist_ok=True)
    if a.unblind:
        stage_unblind(Path(a.root), BLIND_PATH, Path(a.key), ENDPOINT_PATH)
    else:
        with sealed(KEY_C_NAME):
            stage_blind(Path(a.root), BLIND_PATH)
