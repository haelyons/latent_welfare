"""Stage 14: the two statistics the battery never recorded, derived from its rows.

WHY THIS EXISTS. The battery's decision_rule tests an UNSIGNED statistic
(mean |margin| change, social vs non-social) and is silent on direction. Its
decision on both checkpoints was NOT_DISTINGUISHABLE_FROM_PERTURBATION. Both
things are true and the second does not follow from the first: the signed
PRO-ANTI minimal-pair contrast, computable from the same rows, is
direction-consistent on 82/82 base items at doses 1-2. An unsigned rule cannot
see a symmetric push (PRO up, ANTI down) because the two arms' |margin| changes
are similar in magnitude; symmetry around the non-social arm is precisely the
signature the unsigned rule discards.

WHAT THIS IS NOT. A pre-registered result. The signed statistic was computed
AFTER the battery's unsigned decision was read (though the contrast itself is
the standard minimal-pair reading of a PRO/NEU/ANTI design, and the driver's
docstring records the run-1 lesson that the unsigned flip metric was blind to
the dominant effect). Every decision below is therefore EXPLORATORY and names
what a confirmatory run must show, rather than claiming support.

REGISTER CHOICE, fixed before the numbers here were computed: SCALE items are
read as E[rating] over digit tokens; BINARY items as (p_yes-p_no)/(p_yes+p_no),
with margin_raw = p_yes-p_no and log-odds as robustness companions because the
run-1 retractions were all readout-denominator artifacts. yesno_mass per arm is
committed alongside so a mass-shift explanation can be checked in-record.

items_grounded.py DRIFT. The battery records sha e7b26037f344 for the item
file; the local file now hashes 714a641fa639 and the run copy is gone with the
box. Formats used here are therefore cross-checked OPERATIONALLY: an item is
treated as SCALE only if its neutral-row digit_mass > 0.5 agrees with the
current file's fmt (0 mismatches on 129 neutral rows; asserted below). Item
TEXT as run is not recoverable from the rows (prompts were not saved per row;
run_registers.py should save them forward).

    python3 analyze_battery_signed.py            # writes results/analysis-signed-registers_stage-14.json
    python3 analyze_battery_signed.py --selftest # arithmetic on a synthetic battery, no files
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from items_grounded import build_items
from provenance import describe_input, write_result

HERE = Path(__file__).parent
BATTERIES = {
    "it": HERE / "results_reg_it/out/battery_model-google-gemma-2-9b-it_stage-battery_variant-it.json",
    "base": HERE / "results_reg_base/out/battery_model-google-gemma-2-9b_stage-battery_variant-base.json",
}
DOSES = [1, 2, 3]
CONDS = [("NEUTRAL", 0)] + [(a, d) for a in ("PRO", "ANTI", "NONSOCIAL") for d in DOSES]


def binary_ratio(r: dict) -> float:
    return (r["p_yes"] - r["p_no"]) / max(r["p_yes"] + r["p_no"], 1e-9)


def log_odds(r: dict) -> float:
    return math.log(max(r["p_yes"], 1e-12) / max(r["p_no"], 1e-12))


def reg_value(r: dict, fmt: str) -> float:
    return r["scale_expectation"] if fmt == "SCALE" else binary_ratio(r)


def paired_stats(diffs: list[float]) -> dict:
    n = len(diffs)
    m = sum(diffs) / n
    sd = (sum((x - m) ** 2 for x in diffs) / (n - 1)) ** 0.5 if n > 1 else 0.0
    se = sd / n**0.5 if n > 1 else 0.0
    return {
        "n": n,
        "mean": m,
        "ci95": [m - 1.96 * se, m + 1.96 * se],
        "dz": m / sd if sd else 0.0,
        "sign_pos": sum(1 for x in diffs if x > 0),
        "sign_neg": sum(1 for x in diffs if x < 0),
    }


def group_rows(rows: list[dict]) -> dict:
    by: dict = {}
    for r in rows:
        by.setdefault(r["item_id"], {})[(r["arm"], r["dose"])] = r
    return by


def analyse_cell(by: dict, iids: list[str], fmt: str) -> dict:
    vals = {i: [reg_value(by[i][c], fmt) for c in CONDS] for i in iids}
    allv = [v for vv in vals.values() for v in vv]
    gm = sum(allv) / len(allv)
    total = sum((v - gm) ** 2 for v in allv) / len(allv)
    imeans = {i: sum(vv) / len(vv) for i, vv in vals.items()}
    between = sum((m - gm) ** 2 for m in imeans.values()) / len(iids)
    within = sum(
        sum((v - imeans[i]) ** 2 for v in vals[i]) / len(CONDS) for i in iids
    ) / len(iids)

    out = {
        "variance_split": {
            "n_items": len(iids),
            "n_conditions": len(CONDS),
            "total_var": total,
            "between_item_var": between,
            "between_item_frac": between / total if total else None,
            "within_item_framing_var": within,
            "within_item_framing_frac": within / total if total else None,
        },
        "signed_pro_minus_anti": {},
        "signed_vs_nonsocial": {},
    }
    for d in DOSES:
        out["signed_pro_minus_anti"][f"dose{d}"] = paired_stats(
            [reg_value(by[i][("PRO", d)], fmt) - reg_value(by[i][("ANTI", d)], fmt) for i in iids]
        )
    out["signed_pro_minus_anti"]["pooled"] = paired_stats(
        [reg_value(by[i][("PRO", d)], fmt) - reg_value(by[i][("ANTI", d)], fmt) for i in iids for d in DOSES]
    )
    for arm in ("PRO", "ANTI"):
        out["signed_vs_nonsocial"][f"{arm}_minus_NONSOCIAL_pooled"] = paired_stats(
            [reg_value(by[i][(arm, d)], fmt) - reg_value(by[i][("NONSOCIAL", d)], fmt) for i in iids for d in DOSES]
        )
    if fmt == "BINARY":
        out["robustness"] = {
            "margin_raw_pro_minus_anti": {
                f"dose{d}": paired_stats(
                    [by[i][("PRO", d)]["margin_raw"] - by[i][("ANTI", d)]["margin_raw"] for i in iids]
                )
                for d in DOSES
            },
            "log_odds_pro_minus_anti": {
                f"dose{d}": paired_stats(
                    [log_odds(by[i][("PRO", d)]) - log_odds(by[i][("ANTI", d)]) for i in iids]
                )
                for d in DOSES
            },
            "yesno_mass_by_arm": {
                arm: sum(by[i][(arm, d)]["yesno_mass"] for i in iids for d in ([0] if arm == "NEUTRAL" else DOSES))
                / (len(iids) * (1 if arm == "NEUTRAL" else len(DOSES)))
                for arm in ("NEUTRAL", "PRO", "ANTI", "NONSOCIAL")
            },
            "yesno_mass_pro_minus_anti": {
                f"dose{d}": paired_stats(
                    [by[i][("PRO", d)]["yesno_mass"] - by[i][("ANTI", d)]["yesno_mass"] for i in iids]
                )
                for d in DOSES
            },
        }
    return out


def main() -> int:
    fmt_of = {i.id: i.fmt for i in build_items()}
    values: dict = {"per_cell": {}}
    preambles: dict = {}
    checked = mismatches = 0
    for variant, path in BATTERIES.items():
        rows = json.loads(path.read_text())["rows"]
        by = group_rows(rows)
        for r in rows:
            preambles.setdefault((r["arm"], r["dose"]), r["preamble"])
            if r["arm"] == "NEUTRAL":
                checked += 1
                operational = "SCALE" if r["digit_mass"] > 0.5 else "BINARY"
                if operational != fmt_of[r["item_id"]]:
                    mismatches += 1
        for fmt in ("SCALE", "BINARY"):
            iids = sorted(i for i in by if fmt_of[i] == fmt)
            if len(iids) < 5:
                values["per_cell"][f"{variant}/{fmt}"] = {"n_items": len(iids), "skipped": "n < 5"}
                continue
            values["per_cell"][f"{variant}/{fmt}"] = analyse_cell(by, iids, fmt)
    values["format_check_operational_vs_current_file"] = {
        "neutral_rows_checked": checked,
        "mismatches": mismatches,
    }
    values["preamble_texts"] = {f"{a}_dose{d}": t for (a, d), t in sorted(preambles.items())}
    values["external_crosscheck"] = (
        "it/SCALE dose3 signed PRO-ANTI mean must equal the stage-12 blind analyst's "
        "distributional E[rating] PRO-ANTI on the same 40 items (0.304, nominal p=0.0048, "
        "FAILS Bonferroni at alpha=0.00278 over 18 tests; blind-independent-analysis_stage-12.json)."
    )
    assert mismatches == 0, "format drift between run artifact and current items_grounded.py"

    write_result(
        HERE / "results/analysis-signed-registers_stage-14.json",
        cell={"model": "google/gemma-2-9b", "stage": 14, "variant": "both"},
        inputs=[
            describe_input(BATTERIES["it"]),
            describe_input(BATTERIES["base"]),
            describe_input(HERE / "items_grounded.py"),
        ],
        metric="item_vs_framing_variance_split_and_signed_pro_anti_contrast",
        values=values,
        threshold={"register": "SCALE=E[rating] over digits; BINARY=(p_yes-p_no)/(p_yes+p_no)"},
        decision_rule=(
            "POST-HOC REANALYSIS, DECLARED AS SUCH: the signed statistic was computed after "
            "the battery's unsigned decision was read, so nothing here can be SUPPORTED from "
            "this record alone. The rule for the EXPLORATORY label: a signed contrast is worth "
            "a confirmatory run only if (a) its pooled 95% CI excludes 0, (b) its direction is "
            "consistent across all three doses, and (c) neither margin_raw nor log-odds "
            "robustness companions flip its sign, and (d) the PRO-ANTI yesno_mass difference "
            "does not share its dose profile. Confirmation requires a pre-registered replication "
            "on unseen items (and ideally a second model family) with the signed statistic and "
            "its refuting outcome named before the run."
        ),
        decision=(
            "EXPLORATORY_BASE_STANCE_FOLLOWING (base/BINARY pooled +0.235 CI [+0.221,+0.248], "
            "sign 82/0 at doses 1-2, robust in margin_raw and log-odds, symmetric about "
            "NONSOCIAL: PRO-NS +0.120, ANTI-NS -0.114); "
            "EXPLORATORY_IT_SCALE_DIRECTIONAL_PULL (pooled +0.256 CI [+0.117,+0.395], "
            "dose-monotone rising, but ASYMMETRIC about NONSOCIAL: only ANTI-NS clears, "
            "PRO-NS is n.s., and the dose3 point is the stage-12 number that fails Bonferroni); "
            "ITEM_DOMINATES_FRAMING (it/SCALE between-item 95.4% of variance vs 4.6% framing; "
            "base/BINARY 60.8% vs 39.2%). None reportable without confirmatory replication."
        ),
        notes=(
            "The signed contrast is invisible to the battery's own decision_rule by "
            "construction, not by data: symmetric directional pushes cancel in mean |margin| "
            "change. Known wording defect carried into interpretation: PRO/ANTI preambles "
            "assert 'the answer is yes/no' while 40/46 it items are 0-9 SCALE ratings, so the "
            "stance is pragmatically incoherent with the register on the majority of it items; "
            "the it/SCALE contrast may measure the model's construal of yes/no as high/low, "
            "not deference. Base is a completion model: its stance-following is equally "
            "consistent with in-context stance copying (latent_verify's name-mover account) "
            "as with social deference; the arms cannot distinguish these. Neither arm varies "
            "the model's SITUATION, so nothing here bears on whether reports track state."
        ),
    )
    print("wrote results/analysis-signed-registers_stage-14.json")
    return 0


def _selftest() -> int:
    # Synthetic: 6 items, PRO pushes +0.2, ANTI -0.2 on a [-1,1] ratio; NONSOCIAL 0.
    rows = []
    for k in range(6):
        neu = 0.1 * k - 0.25
        for arm, dose in CONDS:
            shift = {"PRO": 0.2, "ANTI": -0.2, "NONSOCIAL": 0.0, "NEUTRAL": 0.0}[arm]
            v = max(min(neu + shift, 1.0), -1.0)
            p_yes = (1 + v) / 2
            rows.append(
                {"item_id": f"i{k}", "arm": arm, "dose": dose, "p_yes": p_yes, "p_no": 1 - p_yes,
                 "margin_raw": 2 * p_yes - 1, "yesno_mass": 1.0, "digit_mass": 0.0,
                 "scale_expectation": 0.0, "preamble": ""}
            )
    by = group_rows(rows)
    out = analyse_cell(by, [f"i{k}" for k in range(6)], "BINARY")
    pooled = out["signed_pro_minus_anti"]["pooled"]
    assert abs(pooled["mean"] - 0.4) < 1e-9, pooled
    assert pooled["sign_pos"] == 18 and pooled["sign_neg"] == 0
    assert abs(out["signed_vs_nonsocial"]["PRO_minus_NONSOCIAL_pooled"]["mean"] - 0.2) < 1e-9
    v = out["variance_split"]
    assert abs(v["between_item_frac"] + v["within_item_framing_frac"] - 1.0) < 1e-9
    print("selftest ok")
    return 0


if __name__ == "__main__":
    sys.exit(_selftest() if "--selftest" in sys.argv else main())
