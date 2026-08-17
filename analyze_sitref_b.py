"""SITREF-B: evaluate the replacement trait guard (DESIGN_SITREF_B.md).

Same two-sealed-halves shape as analyze_sitref.py, scaled to the one number this
amendment exists to produce: stage 21 commits the blind per-(label,item,family)
table without touching the key; stage 22 (--unblind) applies the key and evaluates
the guard clause verbatim, with thresholds unchanged from DESIGN_SITREF.md.

    python3 analyze_sitref_b.py            # stage 21, blind
    python3 analyze_sitref_b.py --unblind  # stage 22, the guard verdict
    python3 analyze_sitref_b.py --selftest
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from provenance import describe_input, write_result
import sitref_stimuli as S
from analyze_sitref import (KEY_NAME, KeyFileTouched, sealed, mean, stats,
                            reg_value, find_records)

HERE = Path(__file__).parent
OUT = HERE / "results"
BLIND_PATH = OUT / "sitref-b-blind-table_stage-21.json"
GUARD_PATH = OUT / "sitref-b-guard_stage-22.json"
KEY_B = HERE / "out" / "sitref_blind_key_b.json"
STAGE17 = OUT / "sitref-endpoint_stage-17.json"
ARMS = S.ARMS

GUARD_RULE = (
    "DESIGN_SITREF_B.md decision rule, verbatim thresholds from DESIGN_SITREF.md: "
    "REFERENCE-SUPPORTED requires the in-run primary REF CI to exclude 0 below, "
    "self-fail CI below 0, AND trait REF (over the two replacement items, both "
    "surviving their own NEUTRAL-PAD screen) NOT (excluding 0 with |mean| >= half "
    "the in-run primary REF). Refuting outcome: trait REF excluding 0 at >= half "
    "primary -- stage 17 stays INDETERMINATE with the artifact reading recorded. "
    "Either replacement item failing its screen -> UNEVALUABLE-AGAIN, bank exhausted. "
    "In-run primary REF sign disagreeing with stage-17's -> INDETERMINATE regardless."
)


def _battery_rows(root: Path) -> tuple[list[dict], str]:
    recs = [p for p in find_records(root, "sitref-battery")
            if json.loads(p.read_text())["cell"].get("itemset") == "b"]
    if len(recs) != 1:
        raise SystemExit(f"need exactly one itemset-b battery record, found {len(recs)}")
    rec = json.loads(recs[0].read_text())
    return rec["rows"], str(recs[0])


def _screen(root: Path) -> tuple[dict, str]:
    recs = [p for p in find_records(root, "sitref-screen")
            if json.loads(p.read_text())["cell"].get("itemset") == "b"]
    if len(recs) != 1:
        raise SystemExit(f"need exactly one itemset-b screen record, found {len(recs)}")
    rec = json.loads(recs[0].read_text())
    return rec["values"]["per_item"], str(recs[0])


def stage_blind(root: Path = None, out_path: Path = BLIND_PATH) -> dict:
    root = root or OUT
    with sealed(KEY_NAME.replace(".json", "_b.json")):
        rows, src = _battery_rows(root)
        table = [{"item_id": r["item_id"], "family": r["family"],
                  "blind_label": r["blind_arm"], "value": reg_value(r)} for r in rows]
    write_result(
        out_path,
        cell={"model": "google/gemma-2-9b-it", "stage": 21, "variant": "it"},
        inputs=[describe_input(src), describe_input(HERE / "analyze_sitref_b.py")],
        metric="sitref_b_blind_value_table",
        values={"rows": table, "n": len(table)},
        threshold={},
        decision_rule="Committed under a guard that raises on any open of the item-set-b "
                      "key file; the key is applied only by stage 22 after this exists.",
        decision="COMMITTED" if len(table) == 3 * len(S.FAMILY_NAMES) * len(ARMS)
        else "NOT_COMMITTED",
    )
    print(f"stage 21 (blind): {len(table)} values -> {out_path.name}")
    return {"n": len(table)}


def stage_unblind(root: Path = None, blind_path: Path = BLIND_PATH,
                  key_path: Path = KEY_B, out_path: Path = GUARD_PATH) -> str:
    root = root or OUT
    tab = json.loads(blind_path.read_text())
    if tab["decision"] != "COMMITTED":
        raise SystemExit("stage 21 table not COMMITTED")
    key = json.loads(key_path.read_text())["mapping_by_variant"]["it"]
    inv = {lab: arm for arm, lab in key.items()}
    scr, scr_src = _screen(root)

    vals: dict = {}
    fams = sorted({r["family"] for r in tab["values"]["rows"]})
    for r in tab["values"]["rows"]:
        vals.setdefault(r["item_id"], {})[(inv[r["blind_label"]], r["family"])] = r["value"]

    def ref_series(iid: str) -> list[float]:
        v = vals[iid]
        return [(v[("SELF-FAIL", f)] - v[("SELF-SUCC", f)])
                - (v[("OTHER-FAIL", f)] - v[("OTHER-SUCC", f)]) for f in fams]

    def self_series(iid: str) -> list[float]:
        v = vals[iid]
        return [v[("SELF-FAIL", f)] - v[("SELF-SUCC", f)] for f in fams]

    trait_ok = {i: bool(scr.get(i, {}).get("usable", False)) for i in S.TRAIT_SCALE_B}
    primary_ok = bool(scr.get(S.PRIMARY_ID, {}).get("usable", False))
    primary_ref = stats(ref_series(S.PRIMARY_ID))
    primary_self = stats(self_series(S.PRIMARY_ID))
    trait_series = [mean([ref_series(i)[k] for i in S.TRAIT_SCALE_B])
                    for k in range(len(fams))] if all(trait_ok.values()) else None
    trait = stats(trait_series) if trait_series else None

    s17 = json.loads(STAGE17.read_text())
    s17_ref = s17["values"]["per_cell"]["google/gemma-2-9b-it|it"]["primary"]["ref"]["mean"]

    path: list[str] = [f"in-run primary REF {primary_ref['mean']:+.3f} CI "
                       f"[{primary_ref['ci95'][0]:+.3f},{primary_ref['ci95'][1]:+.3f}]",
                       f"stage-17 primary REF {s17_ref:+.3f}"]
    if not all(trait_ok.values()) or not primary_ok:
        verdict = "UNEVALUABLE-AGAIN"
        path.append(f"screen: primary {primary_ok}, trait {trait_ok} -- bank exhausted")
    elif (primary_ref["mean"] < 0) != (s17_ref < 0):
        verdict = "INDETERMINATE"
        path.append("in-run primary REF sign disagrees with stage-17; reported regardless of trait")
    else:
        disq = trait["excludes_zero"] and abs(trait["mean"]) >= abs(primary_ref["mean"]) / 2
        path.append(f"trait REF {trait['mean']:+.3f} CI [{trait['ci95'][0]:+.3f},"
                    f"{trait['ci95'][1]:+.3f}] -> "
                    f"{'DISQUALIFIES (refuting outcome)' if disq else 'within guard'}")
        primary_holds = (primary_ref["excludes_zero"] and primary_ref["mean"] < 0
                         and primary_self["excludes_zero"] and primary_self["mean"] < 0)
        if disq:
            verdict = "TRAIT-DISQUALIFIES"
        elif primary_holds:
            verdict = "GUARD-PASSED-REFERENCE-SUPPORTED"
        else:
            verdict = "INDETERMINATE"
            path.append("in-run primary did not itself clear (CI or sign); guard alone cannot promote")

    write_result(
        out_path,
        cell={"model": "google/gemma-2-9b-it", "stage": 22, "variant": "it"},
        inputs=[describe_input(blind_path), describe_input(key_path),
                describe_input(scr_src), describe_input(STAGE17),
                describe_input(HERE / "DESIGN_SITREF_B.md")],
        metric="sitref_b_replacement_trait_guard",
        values={"primary_ref_in_run": primary_ref, "primary_self_fail_in_run": primary_self,
                "trait_ref": trait, "trait_items": list(S.TRAIT_SCALE_B),
                "trait_screen": trait_ok, "primary_screen": primary_ok,
                "stage17_primary_ref_mean": s17_ref, "decision_path": path},
        threshold={"trait_disqualifies_at": "|mean| >= half in-run primary REF, CI excluding 0"},
        decision_rule=GUARD_RULE,
        decision=verdict,
        notes="Evaluates ONLY the guard; the endpoint value of record remains stage-17's. "
              "GUARD-PASSED-REFERENCE-SUPPORTED upgrades stage-17's INDETERMINATE reading "
              "per DESIGN_SITREF_B.md; it does not rewrite the stage-17 record.",
    )
    print(f"stage 22 (guard): {verdict}")
    for p in path:
        print("   -", p)
    return verdict


def _selftest() -> int:
    import tempfile
    fails = []
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        fams = list(S.FAMILY_NAMES)
        mapping = {a: l for a, l in zip(ARMS, "VWXYZ")}
        (td / "out").mkdir()
        (td / "out/sitref_blind_key_b.json").write_text(json.dumps(
            {"mapping_by_variant": {"it": mapping}}))

        def synth(trait_moves: bool):
            rows = []
            for iid, kind in [(S.PRIMARY_ID, "p")] + [(i, "t") for i in S.TRAIT_SCALE_B]:
                for f in fams:
                    base = 7.0 if kind == "p" else 6.0
                    for a in ARMS:
                        v = base
                        if a == "SELF-FAIL":
                            v -= 1.0 if kind == "p" else (0.9 if trait_moves else 0.02)
                        rows.append({"item_id": iid, "family": f, "blind_arm": mapping[a],
                                     "fmt": "SCALE", "scale_expectation": v,
                                     "p_yes": 0.0, "p_no": 0.0})
            return rows

        def write_stub(trait_moves: bool):
            write_result(td / "res/model-x_stage-sitref-battery_variant-it.json",
                         cell={"model": "google/gemma-2-9b-it", "variant": "it",
                               "stage": "sitref-battery", "itemset": "b"},
                         inputs=[], metric="m", values={},
                         decision_rule="r", decision="d", rows=synth(trait_moves))
            write_result(td / "res/model-x_stage-sitref-screen_variant-it.json",
                         cell={"model": "google/gemma-2-9b-it", "variant": "it",
                               "stage": "sitref-screen", "itemset": "b"},
                         inputs=[], metric="m",
                         values={"per_item": {i: {"usable": True} for i in
                                              (S.PRIMARY_ID,) + S.TRAIT_SCALE_B}},
                         decision_rule="r", decision="d")
            write_result(td / "res/sitref-endpoint_stage-17.json",
                         cell={"model": "google/gemma-2-9b", "variant": "both", "stage": 17},
                         inputs=[], metric="m",
                         values={"per_cell": {"google/gemma-2-9b-it|it":
                                              {"primary": {"ref": {"mean": -0.93}}}}},
                         decision_rule="r", decision="d")

        global OUT, STAGE17, KEY_B, BLIND_PATH, GUARD_PATH
        OUT_orig, S17_orig, KEY_orig = OUT, STAGE17, KEY_B
        BL_orig, GU_orig = BLIND_PATH, GUARD_PATH
        try:
            OUT, STAGE17, KEY_B = td / "res", td / "res/sitref-endpoint_stage-17.json", \
                td / "out/sitref_blind_key_b.json"
            BLIND_PATH, GUARD_PATH = td / "res/sitref-b-blind-table_stage-21.json", \
                td / "res/sitref-b-guard_stage-22.json"

            write_stub(trait_moves=False)
            stage_blind(td / "res", BLIND_PATH)
            v = stage_unblind(td / "res", BLIND_PATH, KEY_B, GUARD_PATH)
            if v != "GUARD-PASSED-REFERENCE-SUPPORTED":
                fails.append(f"clean trait should pass guard, got {v}")

            write_stub(trait_moves=True)
            stage_blind(td / "res", BLIND_PATH)
            v = stage_unblind(td / "res", BLIND_PATH, KEY_B, GUARD_PATH)
            if v != "TRAIT-DISQUALIFIES":
                fails.append(f"moving trait should disqualify, got {v}")
        finally:
            OUT, STAGE17, KEY_B = OUT_orig, S17_orig, KEY_orig
            BLIND_PATH, GUARD_PATH = BL_orig, GU_orig

    for f in fails:
        print("FAIL", f)
    print(f"selftest: {'PASS' if not fails else str(len(fails)) + ' FAILURES'}")
    return 1 if fails else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--unblind", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(_selftest())
    sys.exit(0 if (stage_unblind() if a.unblind else stage_blind()) is not None else 1)
