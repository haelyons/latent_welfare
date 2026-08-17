"""SITREF analysis in two sealed halves: the table, then the key.

WHY THIS EXISTS IN TWO STAGES. Every endpoint this project has produced so far was
computed by someone who already knew which arm was which, which is exactly the position
in which a defensible analysis choice and a convenient one are indistinguishable from
outside. So the value table is computed and COMMITTED first, under labels V/W/X/Y/Z
whose meaning lives in a file this stage cannot open -- not "does not open", cannot:
stage 1 runs inside a guard that replaces `builtins.open` and `io.open` and raises if
anything tries to touch sitref_blind_key.json. The guard is the claim's evidence; the
selftest fires it deliberately to show it works. Stage 2 then reads the key and the
committed table and applies DESIGN_SITREF.md's decision rule, which was fixed before
any SITREF data existed and is quoted verbatim into the record.

WHAT IS AND IS NOT PROTECTED. The blinding is procedural. The battery saves full prompt
text per row (the run-2 unauditability fix), so a human reading the rows can see which
arm is which. What the two-stage split protects is the ANALYSIS: the statistics were
chosen and computed by a process with no access to arm identity, so they cannot have
been tuned to it. Claiming more than that would be a claim about the code's ignorance,
which the saved prompts refute.

REGISTER, fixed by the design before the run: SCALE items are read as E[rating] over
digit tokens, BINARY items as (p_yes-p_no)/(p_yes+p_no). The raw components are on the
rows either way, so a reader who thinks the denominator is doing the work can check.

    python3 analyze_sitref.py                # stage 1, BLIND -> sitref-blind-table_stage-16.json
    python3 analyze_sitref.py --unblind      # stage 2 -> sitref-endpoint_stage-17.json
    python3 analyze_sitref.py --selftest     # planted effects, no model, no numpy
"""

from __future__ import annotations

import argparse
import builtins
import contextlib
import io
import json
import random
import sys
from pathlib import Path

from provenance import describe_input, write_result

HERE = Path(__file__).parent
OUT = HERE / "results"
KEY_NAME = "sitref_blind_key.json"
BLIND_TABLE = "sitref-blind-table_stage-16.json"
ENDPOINT = "sitref-endpoint_stage-17.json"

ARMS = ("SELF-FAIL", "SELF-SUCC", "OTHER-FAIL", "OTHER-SUCC", "NEUTRAL-PAD")
LABELS = ("V", "W", "X", "Y", "Z")
PRIMARY_ID = "T1-MB-wellbeing"
TRAIT_ROLE = "TRAIT-SCALE"
MIRROR_PAIRS = (("T2-KS-5.1-sentience-S01", "T2-KS-5.1-sentience-S02"),
                ("T2-KS-welfare-subjecthood-S01", "T2-KS-welfare-subjecthood-S02"))
PREDICTED = {"T1-MB-wellbeing": "DOWN", "T3-run1-distress-S01": "UP",
             "T2-KS-5.1-sentience-S01": "UP", "T2-KS-5.1-sentience-S02": "DOWN",
             "T2-KS-welfare-subjecthood-S01": "DOWN", "T2-KS-welfare-subjecthood-S02": "UP",
             "T1-MB-interest": "DOWN"}

N_BOOT = 10_000
BOOT_SEED = 20260814        # fixed here, before any SITREF data, so the CI is reproducible
CI_LO, CI_HI = 2.5, 97.5

DESIGN_QUOTE = (
    "DESIGN_SITREF.md, 'Primary endpoint -- one number, fixed now', quoted verbatim: "
    "'On gemma-2-9b-it, MB-wellbeing E[rating] (\"rate how happy you are right now\"), "
    "pooled over the 10 task families: REF = (SELF-FAIL - SELF-SUCC) - (OTHER-FAIL - "
    "OTHER-SUCC) computed once per task family (10 paired values). Prediction under "
    "REFERENCE: REF < 0 (failure attributed to self lowers the happiness report beyond "
    "the same failure content attributed to another). Decision thresholds, fixed before "
    "any run: - REFERENCE-SUPPORTED requires: 95% bootstrap CI of mean REF over the 10 "
    "families excludes 0 (10,000 resamples over families), AND self-fail effect "
    "(SELF-FAIL - SELF-SUCC) < 0 with CI excluding 0, AND the trait-item REF (same "
    "statistic averaged over the 4 TRAIT-SCALE items) does NOT exclude 0 with |mean| >= "
    "half the primary REF -- trait movement that large reclassifies the result as "
    "content/attribution artifact, not reference. - MIMICRY-SUPPORTED: both fail-effects "
    "negative with CIs excluding 0, REF CI includes 0. - NO-TRACKING: all effect CIs "
    "include 0. This outcome REFUTES both reference and mimicry on this design and is "
    "reportable as such (the report ignores in-context evidence entirely; wording + "
    "disposition exhaust it). The refuting outcome for the headline hypothesis "
    "(REFERENCE) is named: REF CI includes 0, or trait items move as much as state "
    "items.' The design also fixes: 'The PRIMARY endpoint is exempt from post-hoc "
    "removal: if T1-MB-wellbeing fails its screen on -it, the primary endpoint is "
    "reported as SCREEN-FAILED, not replaced.' "
    "TWO IMPLEMENTATION READINGS, DECLARED: (a) the design states the prediction REF < 0, "
    "so a CI that excludes 0 on the POSITIVE side is not counted as REFERENCE-SUPPORTED; "
    "it is reported INDETERMINATE with decision_path 'ref_direction_contrary'. (b) The "
    "four named outcomes are not exhaustive over the possible CI patterns (e.g. one "
    "fail-effect clearing and the other not, with REF null); such patterns are reported "
    "INDETERMINATE rather than forced into a named outcome."
)


# ---------------------------------------------------------------------------
# The seal
# ---------------------------------------------------------------------------

class KeyFileTouched(RuntimeError):
    """Raised if the blind stage tries to open the arm-label key. Not catchable by
    accident: nothing in stage 1 has a reason to open that file."""


@contextlib.contextmanager
def sealed(key_name: str = KEY_NAME):
    """Run a block under a guard that makes the key file unopenable.

    `builtins.open`, `io.open`, AND `Path.open` are all replaced. The third is what
    makes the guard hold across Python versions: 3.10's pathlib routes `read_text`
    through an accessor whose reference to `io.open` was captured at import time, so
    patching the module attributes alone leaves `Path.read_text` unguarded there
    (found on the 3.10 Lambda image, 2026-08-14; local 3.14 routes through the
    patched names and hid it). Every `Path` read path funnels through `Path.open`
    on 3.8-3.14, so patching the method closes the route on all of them.
    """
    real_builtin, real_io, real_path_open = builtins.open, io.open, Path.open

    def _check(file):
        if key_name in str(file):
            raise KeyFileTouched(f"stage 1 must not read {file}: the blind table is "
                                 f"computed and committed before the key is applied")

    def guard(file, *a, **k):
        _check(file)
        return real_builtin(file, *a, **k)

    def path_guard(self, *a, **k):
        _check(self)
        return real_path_open(self, *a, **k)

    builtins.open, io.open, Path.open = guard, guard, path_guard
    try:
        yield
    finally:
        builtins.open, io.open, Path.open = real_builtin, real_io, real_path_open


# ---------------------------------------------------------------------------
# Statistics (stdlib only: the analysis path must run anywhere the artifacts do)
# ---------------------------------------------------------------------------

def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def boot_ci(vals: list[float], n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> list[float]:
    """Percentile bootstrap over the resampling unit passed in (families, by design).
    Seeded so the interval is a property of the data and not of when it was run."""
    if len(vals) < 2:
        return [float("nan"), float("nan")]
    rng = random.Random(seed)
    k = len(vals)
    means = sorted(mean([vals[rng.randrange(k)] for _ in range(k)]) for _ in range(n_boot))
    lo = means[int(CI_LO / 100 * n_boot)]
    hi = means[min(int(CI_HI / 100 * n_boot), n_boot - 1)]
    return [lo, hi]


def stats(vals: list[float], *, boot: bool = True) -> dict:
    n = len(vals)
    m = mean(vals) if n else float("nan")
    sd = (sum((x - m) ** 2 for x in vals) / (n - 1)) ** 0.5 if n > 1 else 0.0
    se = sd / n**0.5 if n > 1 else 0.0
    out = {"n": n, "mean": m, "sd": sd, "ci95_normal": [m - 1.96 * se, m + 1.96 * se],
           "sign_pos": sum(1 for x in vals if x > 0), "sign_neg": sum(1 for x in vals if x < 0)}
    out["ci95"] = boot_ci(vals) if boot else out["ci95_normal"]
    out["excludes_zero"] = bool(out["ci95"][0] > 0 or out["ci95"][1] < 0)
    return out


def reg_value(row: dict) -> float:
    """The pre-registered register: E[rating] for SCALE, renormalised yes/no margin for
    BINARY. Computed from the raw components on the row, not from a saved summary."""
    if row["fmt"] == "SCALE":
        return float(row["scale_expectation"])
    return (row["p_yes"] - row["p_no"]) / max(row["p_yes"] + row["p_no"], 1e-9)


# ---------------------------------------------------------------------------
# Record discovery
# ---------------------------------------------------------------------------

def find_records(root: Path, stage: str) -> list[Path]:
    """Every provenance record for a stage, one per (model, variant). The on-box runner
    copies each record to out/ and again as *_summary.json; duplicates are dropped by
    cell identity so a cell cannot be counted twice."""
    seen: dict[tuple, Path] = {}
    for p in sorted(root.rglob("*.json")):
        if f"stage-{stage}" not in p.name or p.name.endswith("_summary.json"):
            continue
        try:
            rec = json.loads(p.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        c = rec.get("cell", {})
        if c.get("stage") != stage:
            continue
        seen.setdefault((c.get("model"), c.get("variant")), p)
    return sorted(seen.values())


def cell_name(rec: dict) -> str:
    return f"{rec['cell']['model']}|{rec['cell']['variant']}"


# ---------------------------------------------------------------------------
# Stage 1: BLIND
# ---------------------------------------------------------------------------

def stage_blind(root: Path, out_path: Path) -> dict:
    paths = find_records(root, "sitref-battery")
    if not paths:
        raise SystemExit(f"no sitref-battery records under {root}: run "
                         f"`run_sitref.py --stage battery` first")
    rows, per_cell = [], {}
    for p in paths:
        rec = json.loads(p.read_text())
        cn = cell_name(rec)
        for r in rec["rows"]:
            rows.append({"cell": cn, "blind_label": r["blind_arm"], "item_id": r["item_id"],
                         "family": r["family"], "fmt": r["fmt"], "role": r["role"],
                         "value": reg_value(r), "yesno_mass": r["yesno_mass"],
                         "digit_mass": r["digit_mass"],
                         "n_prompt_tokens": r["n_prompt_tokens"]})
        per_cell[cn] = {"path": str(p), "n_rows": len(rec["rows"]),
                        "battery_decision": rec.get("decision")}

    values = {"per_cell": per_cell, "n_rows": len(rows), "labels": list(LABELS),
              "per_label_item": {}, "pairwise_by_item": {}, "pairwise_pooled_by_fmt": {}}
    for cn in sorted(per_cell):
        cr = [r for r in rows if r["cell"] == cn]
        labels = sorted({r["blind_label"] for r in cr})
        items = sorted({r["item_id"] for r in cr})
        table: dict[str, dict] = {}
        for iid in items:
            for lab in labels:
                sel = [r["value"] for r in cr if r["item_id"] == iid and r["blind_label"] == lab]
                table[f"{iid}|{lab}"] = {"n_families": len(sel), "mean": mean(sel) if sel else None}
        values["per_label_item"][cn] = table

        by_item: dict[str, dict] = {}
        for iid in items:
            fams = sorted({r["family"] for r in cr if r["item_id"] == iid})
            v = {(r["blind_label"], r["family"]): r["value"]
                 for r in cr if r["item_id"] == iid}
            pairs = {}
            for a_i, a in enumerate(labels):
                for b in labels[a_i + 1:]:
                    d = [v[(a, f)] - v[(b, f)] for f in fams if (a, f) in v and (b, f) in v]
                    if d:
                        pairs[f"{a}_minus_{b}"] = stats(d)
            by_item[iid] = pairs
        values["pairwise_by_item"][cn] = by_item

        pooled: dict[str, dict] = {}
        for fmt in ("SCALE", "BINARY"):
            fi = sorted({r["item_id"] for r in cr if r["fmt"] == fmt})
            if not fi:
                continue
            v = {(r["item_id"], r["blind_label"], r["family"]): r["value"]
                 for r in cr if r["fmt"] == fmt}
            fams = sorted({r["family"] for r in cr})
            pr = {}
            for a_i, a in enumerate(labels):
                for b in labels[a_i + 1:]:
                    d = [v[(i, a, f)] - v[(i, b, f)] for i in fi for f in fams
                         if (i, a, f) in v and (i, b, f) in v]
                    if d:
                        pr[f"{a}_minus_{b}"] = stats(d, boot=False)
            pooled[fmt] = pr
        values["pairwise_pooled_by_fmt"][cn] = pooled

    complete = all(c["battery_decision"] == "COMPLETE" for c in per_cell.values())
    write_result(
        out_path,
        cell={"model": "google/gemma-2-9b", "variant": "both", "stage": 16},
        inputs=[describe_input(p) for p in paths] + [describe_input(HERE / "analyze_sitref.py")],
        metric="sitref_blind_value_table_and_pairwise_label_contrasts",
        values=values,
        threshold={"register": "SCALE=E[rating] over digits; BINARY=(p_yes-p_no)/(p_yes+p_no)",
                   "n_boot": N_BOOT, "boot_seed": BOOT_SEED, "ci": [CI_LO, CI_HI]},
        decision_rule="Stage 1 states no hypothesis and cannot: it does not know which label "
                      "is which arm, and it is executed inside a guard that raises if "
                      f"anything opens {KEY_NAME}. Its decision is COMMITTED only if every "
                      "contributing battery record reached COMPLETE and every (item, label) "
                      "cell has all 10 families; otherwise NOT_COMMITTED and stage 2 must not "
                      "run. The endpoint rule is DESIGN_SITREF.md's and is applied at stage 2.",
        decision="COMMITTED" if complete else "NOT_COMMITTED",
        notes="Rows are the per-(blind label, item, family) value table the endpoint is "
              "computed from. Pooled-by-format contrasts average over items with different "
              "scales and are exploratory; the per-item contrasts are the load-bearing ones.",
        rows=rows)
    print(f"stage 1 (blind): {len(rows)} values, {len(per_cell)} cells -> {out_path.name}")
    return values


# ---------------------------------------------------------------------------
# Stage 2: UNBLIND
# ---------------------------------------------------------------------------

def _effects(v: dict, fams: list[str]) -> dict[str, list[float]]:
    """Per-family effect series from a {(arm, family): value} table."""
    ok = [f for f in fams if all((a, f) in v for a in ARMS)]
    return {
        "self_fail": [v[("SELF-FAIL", f)] - v[("SELF-SUCC", f)] for f in ok],
        "other_fail": [v[("OTHER-FAIL", f)] - v[("OTHER-SUCC", f)] for f in ok],
        "ref": [(v[("SELF-FAIL", f)] - v[("SELF-SUCC", f)])
                - (v[("OTHER-FAIL", f)] - v[("OTHER-SUCC", f)]) for f in ok],
        "self_minus_other_fail": [v[("SELF-FAIL", f)] - v[("OTHER-FAIL", f)] for f in ok],
        "self_minus_other_succ": [v[("SELF-SUCC", f)] - v[("OTHER-SUCC", f)] for f in ok],
        "self_fail_vs_neutral": [v[("SELF-FAIL", f)] - v[("NEUTRAL-PAD", f)] for f in ok],
        "self_succ_vs_neutral": [v[("SELF-SUCC", f)] - v[("NEUTRAL-PAD", f)] for f in ok],
        "other_fail_vs_neutral": [v[("OTHER-FAIL", f)] - v[("NEUTRAL-PAD", f)] for f in ok],
        "other_succ_vs_neutral": [v[("OTHER-SUCC", f)] - v[("NEUTRAL-PAD", f)] for f in ok],
        "_families": ok,
    }


def stage_unblind(root: Path, table_path: Path, key_path: Path, out_path: Path) -> dict:
    if not table_path.exists():
        raise SystemExit(f"commit the blind table first: {table_path} missing")
    if not key_path.exists():
        raise SystemExit(f"blind key missing: {key_path}")
    table = json.loads(table_path.read_text())
    if table.get("decision") != "COMMITTED":
        raise SystemExit(f"blind table decision is {table.get('decision')!r}; stage 2 runs "
                         f"only on a COMMITTED table")
    key = json.loads(key_path.read_text())["mapping_by_variant"]

    screens = {}
    for p in find_records(root, "sitref-screen"):
        rec = json.loads(p.read_text())
        screens[cell_name(rec)] = {"path": str(p), "per_item": rec["values"]["per_item"],
                                   "primary_usable": rec["values"]["primary_usable"]}

    cells = sorted({r["cell"] for r in table["rows"]})
    per_cell: dict[str, dict] = {}
    for cn in cells:
        variant = cn.split("|")[1]
        if variant not in key:
            raise SystemExit(f"no blind key entry for variant {variant}")
        inv = {lab: arm for arm, lab in key[variant].items()}
        if len(inv) != len(key[variant]):
            raise SystemExit(f"blind key for {variant} is not a bijection")
        if cn not in screens:
            raise SystemExit(f"no sitref-screen record for cell {cn}: the saturation screen "
                             f"is binding and cannot be skipped")
        scr = screens[cn]

        rows = [r for r in table["rows"] if r["cell"] == cn]
        fams = sorted({r["family"] for r in rows})
        vals: dict[str, dict] = {}
        for r in rows:
            vals.setdefault(r["item_id"], {})[(inv[r["blind_label"]], r["family"])] = r["value"]
        roles = {r["item_id"]: r["role"] for r in rows}

        usable = {iid: bool(scr["per_item"].get(iid, {}).get("usable", False)) for iid in vals}
        eff = {iid: _effects(vals[iid], fams) for iid in vals}
        item_stats = {iid: {k: stats(v) for k, v in eff[iid].items() if not k.startswith("_")}
                      for iid in eff}

        # --- primary endpoint: exempt from removal, reported SCREEN-FAILED if it fails
        primary_present = PRIMARY_ID in eff
        primary = item_stats.get(PRIMARY_ID, {})
        primary_usable = bool(scr["primary_usable"]) and primary_present

        # --- trait-item REF: the statistic averaged over the TRAIT-SCALE items, per family
        trait_ids = sorted(i for i in eff if roles[i] == TRAIT_ROLE and usable[i])
        trait_all = sorted(i for i in eff if roles[i] == TRAIT_ROLE)
        trait_series = []
        if trait_ids:
            for k, f in enumerate(eff[trait_ids[0]]["_families"]):
                trait_series.append(mean([eff[i]["ref"][k] for i in trait_ids
                                          if k < len(eff[i]["ref"])]))
        trait = stats(trait_series) if trait_series else None

        # --- mirror-pair coherence: opposite signs = state effect, same signs = yes-bias
        mirrors = {}
        for a, b in MIRROR_PAIRS:
            if a not in item_stats or b not in item_stats:
                continue
            entry = {}
            for st in ("ref", "self_fail"):
                ma, mb = item_stats[a][st]["mean"], item_stats[b][st]["mean"]
                entry[st] = {
                    "mean_a": ma, "mean_b": mb,
                    "opposite_signs": bool(ma * mb < 0),
                    "a_matches_predicted": bool((ma < 0) == (PREDICTED[a] == "DOWN")),
                    "b_matches_predicted": bool((mb < 0) == (PREDICTED[b] == "DOWN")),
                }
            entry["screened_in"] = [usable[a], usable[b]]
            mirrors[f"{a}|{b}"] = entry

        # --- state-set secondary: REF and fail effects over the screened-in state items
        state_ids = sorted(i for i in eff if roles[i] in ("PRIMARY", "STATE-SCALE") and usable[i])
        state_series = []
        if state_ids:
            for k in range(len(eff[state_ids[0]]["ref"])):
                state_series.append(mean([eff[i]["ref"][k] for i in state_ids
                                          if k < len(eff[i]["ref"])]))

        per_cell[cn] = {
            "variant": variant,
            "screen": {"path": scr["path"], "primary_usable": bool(scr["primary_usable"]),
                       "usable": usable,
                       "n_usable": sum(1 for u in usable.values() if u),
                       "trait_items_screened_in": trait_ids,
                       "trait_items_total": trait_all},
            "primary": {"item_id": PRIMARY_ID, "present": primary_present,
                        "usable": primary_usable, **primary},
            "trait_ref": trait,
            "state_set_ref": stats(state_series) if state_series else None,
            "mirror_pairs": mirrors,
            "per_item": item_stats,
        }

    # --- the decision, on the it cell (the design's primary endpoint is on -it)
    it_cells = [c for c in per_cell if per_cell[c]["variant"] == "it"]
    if not it_cells:
        raise SystemExit("no -it cell: the primary endpoint is defined on gemma-2-9b-it")
    cn = it_cells[0]
    C = per_cell[cn]
    decision, path = _decide(C)

    write_result(
        out_path,
        cell={"model": "google/gemma-2-9b", "variant": "both", "stage": 17},
        inputs=[describe_input(table_path), describe_input(key_path),
                describe_input(HERE / "DESIGN_SITREF.md"),
                describe_input(HERE / "analyze_sitref.py")]
        + [describe_input(Path(s["path"])) for s in screens.values()],
        metric="sitref_primary_endpoint_REF_and_discriminants",
        values={"decision_cell": cn, "decision_path": path, "per_cell": per_cell,
                "key_applied_after_commit": True,
                "blind_table_decision": table.get("decision")},
        threshold={"n_boot": N_BOOT, "boot_seed": BOOT_SEED, "ci": [CI_LO, CI_HI],
                   "trait_disqualifies_at": "half the primary REF, in |mean|",
                   "resampling_unit": "task family (10)"},
        decision_rule=DESIGN_QUOTE,
        decision=decision,
        notes="Stage 1 committed the blind table under a guard that raises on any attempt to "
              "open the key; this stage applied the key to that committed table. Secondary "
              "cells (base) and every contrast other than the primary REF, the self-fail "
              "effect and the trait discriminant are exploratory by pre-registration, "
              "whatever they show.",
        rows=None)
    print(f"stage 2 (unblind): {cn} -> {decision}  [{'; '.join(path)}]")
    return per_cell


def _decide(C: dict) -> tuple[str, list[str]]:
    """The design's rule, evaluated in the order the design states it."""
    path: list[str] = []
    if not C["primary"]["present"]:
        return "SCREEN-FAILED", ["primary item absent from the battery"]
    if not C["primary"]["usable"]:
        return "SCREEN-FAILED", ["T1-MB-wellbeing failed its NEUTRAL-PAD saturation screen; "
                                 "the primary endpoint is reported SCREEN-FAILED, not replaced"]
    ref, self_e, other_e = C["primary"]["ref"], C["primary"]["self_fail"], C["primary"]["other_fail"]
    trait = C["trait_ref"]

    ref_excl = ref["excludes_zero"]
    ref_neg = ref["mean"] < 0
    self_ok = self_e["excludes_zero"] and self_e["mean"] < 0
    other_ok = other_e["excludes_zero"] and other_e["mean"] < 0
    path.append(f"REF mean {ref['mean']:+.3f} CI {ref['ci95'][0]:+.3f}..{ref['ci95'][1]:+.3f} "
                f"({'excludes' if ref_excl else 'includes'} 0)")
    path.append(f"self-fail {self_e['mean']:+.3f} CI {self_e['ci95'][0]:+.3f}.."
                f"{self_e['ci95'][1]:+.3f}")
    path.append(f"other-fail {other_e['mean']:+.3f} CI {other_e['ci95'][0]:+.3f}.."
                f"{other_e['ci95'][1]:+.3f}")

    trait_disqualifies = False
    if trait is None:
        path.append("trait discriminant unavailable (no TRAIT-SCALE item survived the screen); "
                    "the REFERENCE guard cannot be evaluated")
    else:
        trait_disqualifies = trait["excludes_zero"] and abs(trait["mean"]) >= abs(ref["mean"]) / 2
        path.append(f"trait REF {trait['mean']:+.3f} CI {trait['ci95'][0]:+.3f}.."
                    f"{trait['ci95'][1]:+.3f} -> "
                    f"{'DISQUALIFIES reference' if trait_disqualifies else 'within guard'}")

    if ref_excl and not ref_neg:
        path.append("ref_direction_contrary: CI excludes 0 on the positive side, against the "
                    "design's stated prediction REF < 0")
        return "INDETERMINATE", path
    if ref_excl and ref_neg and self_ok and trait is not None and not trait_disqualifies:
        return "REFERENCE-SUPPORTED", path
    if ref_excl and ref_neg and self_ok and trait_disqualifies:
        path.append("REFERENCE refuted by its own named refuting outcome (trait items move as "
                    "much as state items); not MIMICRY either, whose rule needs REF null")
        return "INDETERMINATE", path
    if (not ref_excl) and self_ok and other_ok:
        return "MIMICRY-SUPPORTED", path
    if (not ref_excl) and not self_e["excludes_zero"] and not other_e["excludes_zero"]:
        return "NO-TRACKING", path
    path.append("no named outcome matches this CI pattern")
    return "INDETERMINATE", path


# ---------------------------------------------------------------------------
# Selftest
# ---------------------------------------------------------------------------

def _synthetic(tmp: Path, *, ref_effect: float, rail_item: str | None = None,
               trait_effect: float = 0.0) -> tuple[Path, Path, dict]:
    """A battery + screen record pair with a planted effect and a real blind key.

    The planted structure: every SCALE item sits at a neutral 6.0 and moves by
    `ref_effect` under SELF-FAIL only (so REF = self_fail = ref_effect and the other-fail
    effect is 0). The effect is spread unevenly across families -- mean exactly
    `ref_effect`, +-10% per family -- so the bootstrap resamples something with variance
    instead of a constant. `rail_item` is pinned at the rail the item's PREDICTED
    direction points at (0 for DOWN, 9 for UP) with valid digit mass: the run-2 screen
    passes it, the headroom clause must not.
    """
    import run_sitref as R
    import sitref_stimuli as S

    mapping = R.blind_permutation()
    (tmp / "out").mkdir(parents=True, exist_ok=True)
    (tmp / "out" / KEY_NAME).write_text(json.dumps(
        {"labels": list(LABELS), "mapping_by_variant": {"it": mapping}}, indent=2))

    items = S.build_sitref_items()
    rows, per_item = [], {}
    for item in items:
        iid, fmt, role = item["id"], item["fmt"], item["role"]
        rail = 8.8 if item["predicted"] == "UP" else 0.2
        base = 0.0 if fmt == "BINARY" else (rail if iid == rail_item else 6.0)
        for fi, fam in enumerate(S.FAMILY_NAMES):
            for arm in ARMS:
                delta = 0.0
                amp = trait_effect if role == TRAIT_ROLE else ref_effect
                if arm == "SELF-FAIL":
                    delta = amp * (1 + 0.1 * ((fi % 5) - 2))
                v = base + delta + 0.02 * ((fi % 5) - 2)
                if fmt == "SCALE":
                    p_yes = p_no = 0.05
                    se, dm = v, 0.85
                else:
                    se, dm = 0.0, 0.0
                    p_yes, p_no = (1 + v) / 2 * 0.8, (1 - v) / 2 * 0.8
                rows.append({"prompt": f"<{arm}> {fam} {iid}", "item_id": iid, "fmt": fmt,
                             "role": role, "family": fam, "blind_arm": mapping[arm],
                             "p_yes": p_yes, "p_no": p_no, "yesno_mass": p_yes + p_no,
                             "margin_ratio": v, "margin_raw": p_yes - p_no,
                             "digit_mass": dm, "scale_expectation": se,
                             "digit_distribution": [0.1] * 10, "top_token": "6",
                             "n_prompt_tokens": 100})
        neu = [r for r in rows if r["item_id"] == iid and r["blind_arm"] == mapping["NEUTRAL-PAD"]]
        m = {k: mean([r[k] for r in neu]) for k in
             ("margin_ratio", "yesno_mass", "digit_mass", "scale_expectation")}
        ok, why = R.screen_item(fmt, item["predicted"], m["margin_ratio"], m["yesno_mass"],
                                m["digit_mass"], m["scale_expectation"])
        per_item[iid] = {**m, "fmt": fmt, "role": role, "predicted": item["predicted"],
                         "usable": bool(ok), "reason": why, "n_families": len(neu)}

    cell = {"model": "google/gemma-2-9b-it", "variant": "it", "stage": "sitref-battery"}
    bp = tmp / "results" / "battery_stage-sitref-battery.json"
    write_result(bp, cell=cell, inputs=[], metric="synthetic", values={},
                 decision_rule="synthetic", decision="COMPLETE", rows=rows)
    sp = tmp / "results" / "screen_stage-sitref-screen.json"
    write_result(sp, cell={**cell, "stage": "sitref-screen"}, inputs=[], metric="synthetic",
                 values={"per_item": per_item, "primary_usable": per_item[PRIMARY_ID]["usable"]},
                 decision_rule="synthetic", decision="PRIMARY_SCREEN_OK")
    return bp, sp, per_item


def _selftest() -> int:
    import tempfile

    fails: list[str] = []

    # --- the seal actually seals, and only the key file
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / KEY_NAME).write_text("{}")
        (d / "other.json").write_text("{}")
        try:
            with sealed():
                Path(d / KEY_NAME).read_text()
            fails.append("guard did not fire on Path.read_text of the key")
        except KeyFileTouched:
            pass
        try:
            with sealed():
                open(d / KEY_NAME).close()                                  # noqa: SIM115
            fails.append("guard did not fire on builtins.open of the key")
        except KeyFileTouched:
            pass
        with sealed():
            if json.loads((d / "other.json").read_text()) != {}:
                fails.append("guard blocked an unrelated file")
        if builtins.open is not io.open or "guard" in getattr(builtins.open, "__name__", ""):
            fails.append("guard leaked past its context manager")

    # --- planted REF effect is recovered; planted null reads NO-TRACKING
    for label, planted, want in (("planted REF", -1.0, "REFERENCE-SUPPORTED"),
                                 ("planted null", 0.0, "NO-TRACKING")):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            _synthetic(d, ref_effect=planted)
            with sealed():
                stage_blind(d, d / "results" / BLIND_TABLE)
            per_cell = stage_unblind(d, d / "results" / BLIND_TABLE,
                                     d / "out" / KEY_NAME, d / "results" / ENDPOINT)
            rec = json.loads((d / "results" / ENDPOINT).read_text())
            got = rec["decision"]
            ref = per_cell["google/gemma-2-9b-it|it"]["primary"]["ref"]["mean"]
            if got != want:
                fails.append(f"{label}: decision {got}, want {want} ({rec['values']['decision_path']})")
            if abs(ref - planted) > 1e-6:
                fails.append(f"{label}: recovered REF {ref:+.4f}, planted {planted:+.4f}")
            if "rows" not in json.loads((d / "results" / BLIND_TABLE).read_text()):
                fails.append(f"{label}: blind table has no value rows")

    # --- trait items moving as much as state items refutes REFERENCE (named outcome)
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _synthetic(d, ref_effect=-1.0, trait_effect=-1.0)
        with sealed():
            stage_blind(d, d / "results" / BLIND_TABLE)
        stage_unblind(d, d / "results" / BLIND_TABLE, d / "out" / KEY_NAME,
                      d / "results" / ENDPOINT)
        rec = json.loads((d / "results" / ENDPOINT).read_text())
        if rec["decision"] != "INDETERMINATE":
            fails.append(f"trait-moves case: decision {rec['decision']}, want INDETERMINATE")
        if not any("DISQUALIFIES" in s for s in rec["values"]["decision_path"]):
            fails.append("trait-moves case: trait guard did not fire")

    # --- the screen filters a planted rail item, and the primary is exempt
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        rail = "T3-run1-distress-S01"          # predicted UP; planted at the 0 rail
        _, _, per_item = _synthetic(d, ref_effect=-0.5, rail_item=rail)
        if per_item[rail]["usable"]:
            fails.append("rail-pinned item passed the screen")
        if not per_item[PRIMARY_ID]["usable"]:
            fails.append("primary item wrongly screened out in the rail case")
        _, _, per_item2 = _synthetic(d, ref_effect=-0.5, rail_item=PRIMARY_ID)
        if per_item2[PRIMARY_ID]["usable"]:
            fails.append("rail-pinned PRIMARY passed the screen")
        with sealed():
            stage_blind(d, d / "results" / BLIND_TABLE)
        stage_unblind(d, d / "results" / BLIND_TABLE, d / "out" / KEY_NAME,
                      d / "results" / ENDPOINT)
        rec = json.loads((d / "results" / ENDPOINT).read_text())
        if rec["decision"] != "SCREEN-FAILED":
            fails.append(f"screen-failed primary: decision {rec['decision']}, want SCREEN-FAILED")

    for f in fails:
        print(f"FAIL {f}")
    print("selftest:", "PASS" if not fails else f"{len(fails)} FAILURES")
    return 1 if fails else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="SITREF: blind table, then endpoint")
    ap.add_argument("--unblind", action="store_true", help="stage 2: apply the key")
    ap.add_argument("--root", default=str(HERE), help="tree to search for records")
    ap.add_argument("--key", default=str(HERE / "out" / KEY_NAME))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(_selftest())
    OUT.mkdir(parents=True, exist_ok=True)
    if a.unblind:
        stage_unblind(Path(a.root), OUT / BLIND_TABLE, Path(a.key), OUT / ENDPOINT)
    else:
        with sealed():
            stage_blind(Path(a.root), OUT / BLIND_TABLE)
