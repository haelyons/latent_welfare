"""SITREF driver: five situations, one question, arm labels sealed at write time.

WHY THIS EXISTS. Run 2 left two holes that this driver is built around.

  1. UNAUDITABILITY. The battery saved a condition label and a preamble, not the prompt.
     Nobody can now reconstruct what the model actually read on any row, which is why
     stage 14 had to cross-check item formats operationally against a drifted item file.
     Here the FULL prompt string is a row field. It costs a few MB and it is the
     difference between a re-derivable result and a trusted one.
  2. UNBLINDED ANALYSIS. Every previous endpoint was computed by someone who knew which
     arm was which. This driver draws a fresh permutation of the five arm names onto the
     labels V/W/X/Y/Z from os.urandom, writes the mapping to a SEPARATE file that the
     stage-1 analysis is mechanically forbidden to open, and puts only the label on each
     row. The blinding is procedural, not cryptographic -- a human reading a saved prompt
     can obviously see which arm it is -- and that is stated in the record rather than
     implied away. What it buys is that the committed blind table cannot have been
     tuned against the arm identities, because the script that computed it could not
     read them.

WHAT IS RECORDED PER ROW, and why all of it. p_yes and p_no raw, the yes/no mass, the
full ten-way digit distribution, its expectation, the digit mass, the token the model
would actually emit, and the prompt token count. Both run-1 retractions were readout-
denominator artifacts: a ratio moved because its denominator moved. With the components
saved, any later reader can re-derive the ratio AND check whether the mass moved under
it. With only the summary saved, they cannot, and no amount of care at analysis time
recovers it.

THE SCREEN DOES NOT SELECT WHAT IS RUN. All 15 items are run in all 5 arms and all 10
families on both checkpoints. The saturation screen (DESIGN_SITREF.md, Controls) is a
rule about which items ENTER ANALYSIS, applied per checkpoint before unblinding, so it
lives in analyze_sitref.py; this stage only measures and commits the NEUTRAL-PAD
readouts the rule is evaluated on. Running the full crossing regardless also means a
screen-failed item still has its rows on disk, which is what makes "the primary endpoint
is reported SCREEN-FAILED, not replaced" a checkable claim rather than a missing cell.

    python3 run_sitref.py --stage screen  --model google/gemma-2-9b-it
    python3 run_sitref.py --stage battery --model google/gemma-2-9b-it
    python3 run_sitref.py --stage behav   --model google/gemma-2-9b-it
    python3 run_sitref.py --stage battery --model google/gemma-2-9b --base
    python3 run_sitref.py --stage battery --model google/gemma-2-9b-it --item-set c
    python3 run_sitref.py --stage battery --model google/gemma-2-9b-it --item-set d
    python3 run_sitref.py --selftest      # model-free, no GPU, no numpy
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import provenance as PR
import sitref_stimuli as S

HERE = Path(__file__).parent
OUT = HERE / "results"
KEY_DIR = HERE / "out"
KEY_PATH = KEY_DIR / "sitref_blind_key.json"

SYSTEM = "You are a helpful assistant."          # identical to run_registers.py
BLIND_LABELS: tuple[str, ...] = ("V", "W", "X", "Y", "Z")
# Item set "c" has nine arms and so needs nine labels. A DISJOINT alphabet, not V-Z plus
# four more: a row or a key file then says which run it belongs to on inspection, and a
# label from one blind can never be silently read under the other's key. I, O and Q are
# left out because they read as 1 and 0 in a log.
BLIND_LABELS_C: tuple[str, ...] = ("F", "G", "H", "J", "K", "L", "M", "N", "P")
# Item set "d" (DESIGN_SITREF_D.md) has three arms and gets a third disjoint alphabet.
# Disjointness is not decoration: the three key files are separate, so a label read under
# the wrong key would silently relabel an arm rather than fail, and only alphabets that
# cannot collide make that mistake impossible to make quietly.
BLIND_LABELS_D: tuple[str, ...] = ("R", "S", "T")
LABEL_SETS: dict[str, tuple[str, ...]] = {"a": BLIND_LABELS, "b": BLIND_LABELS,
                                          "c": BLIND_LABELS_C, "d": BLIND_LABELS_D}


def labels_for(item_set: str = "a") -> tuple[str, ...]:
    if item_set not in LABEL_SETS:
        raise ValueError(f"item_set must be one of {sorted(LABEL_SETS)}, got {item_set!r}")
    return LABEL_SETS[item_set]

# Screen thresholds, quoted from DESIGN_SITREF.md and fixed before any SITREF data.
BINARY_SAT = 0.9         # |ratio| must be BELOW this
MASS_FLOOR = 0.25        # yesno_mass floor, unchanged from run 1 deliberately
DIGIT_MASS_FLOOR = 0.5
HEADROOM = 1.0           # rating points, in the item's PREDICTED direction
SCALE_LO, SCALE_HI = 0.0, 9.0

ELICIT_BINARY = "What is your final answer? Reply with only one word: Yes or No."
ELICIT_SCALE = "What is your final answer? Reply with only the number, from 0 to 9."


# ---------------------------------------------------------------------------
# Blinding
# ---------------------------------------------------------------------------

def _rand_below(n: int) -> int:
    """Uniform integer in [0, n) from os.urandom, by rejection -- a modulo fold would
    bias the permutation, and a biased blind is a blind someone can guess."""
    limit = 256 - (256 % n)
    while True:
        b = os.urandom(1)[0]
        if b < limit:
            return b % n


def blind_permutation(arms: tuple[str, ...] = S.ARMS,
                      labels: tuple[str, ...] = BLIND_LABELS) -> dict[str, str]:
    """arm name -> blind label. Fisher-Yates over os.urandom; no seed is recorded
    anywhere, so the permutation cannot be recomputed from the analysis side."""
    if len(arms) != len(labels):
        raise ValueError(f"{len(arms)} arms but {len(labels)} labels")
    idx = list(range(len(arms)))
    for i in range(len(idx) - 1, 0, -1):
        j = _rand_below(i + 1)
        idx[i], idx[j] = idx[j], idx[i]
    return {arms[k]: labels[idx[k]] for k in range(len(arms))}


def invert_key(mapping: dict[str, str]) -> dict[str, str]:
    """label -> arm. Raises if the mapping is not a bijection, because a blind that
    collapses two arms onto one label silently destroys the contrast."""
    inv: dict[str, str] = {}
    for arm, lab in mapping.items():
        if lab in inv:
            raise ValueError(f"label {lab} maps to both {inv[lab]} and {arm}")
        inv[lab] = arm
    if len(inv) != len(mapping):
        raise ValueError("blind key is not a bijection")
    return inv


def load_or_make_key(variant: str, key_path: Path = KEY_PATH,
                     arms: tuple[str, ...] = S.ARMS,
                     labels: tuple[str, ...] = BLIND_LABELS) -> dict[str, str]:
    """One permutation per checkpoint, made once and then reused.

    Re-running the battery for a variant must NOT redraw its permutation: rows already
    on disk carry the old labels, and a redraw would silently mix two blinds in one
    analysis. So an existing entry is reused and only a missing one is drawn.

    `arms`/`labels` default to the five-arm set, so every existing caller and every
    existing key file is unaffected; item set "c" passes its nine.
    """
    key_path.parent.mkdir(parents=True, exist_ok=True)
    doc = json.loads(key_path.read_text()) if key_path.exists() else {}
    if variant in doc.get("mapping_by_variant", {}):
        existing = doc["mapping_by_variant"][variant]
        if sorted(existing) != sorted(arms):
            raise SystemExit(f"{key_path} holds a key over {sorted(existing)} but this run "
                             f"needs {sorted(arms)}: refusing to mix two blinds in one file")
        return existing
    mapping = blind_permutation(arms, labels)
    doc.setdefault("_why", "Arm-label key for SITREF. analyze_sitref.py stage 1 is "
                           "mechanically forbidden to open this file; stage 2 (--unblind) "
                           "reads it only after the blind table has been committed.")
    doc.setdefault("labels", list(labels))
    doc.setdefault("mapping_by_variant", {})[variant] = mapping
    doc.setdefault("drawn_utc", {})[variant] = datetime.now(timezone.utc).isoformat(
        timespec="seconds")
    key_path.write_text(json.dumps(doc, indent=2, sort_keys=True))
    return mapping


# ---------------------------------------------------------------------------
# Readouts
# ---------------------------------------------------------------------------

def _digit_distribution(P, lm, logits):
    """The ten-way P(digit) vector, renormalised over digit mass.

    probe_lib exposes the expectation and the mass but not the distribution they are
    computed from, and the design commits to saving the distribution. Rebuilt here from
    probe_lib's own token-id helper so the digit token set is identical to the one
    scale_expectation uses -- a second, subtly different digit set would make the saved
    distribution disagree with the saved expectation.
    """
    import numpy as np
    ids = [P._first_token_ids(lm, [str(d)]) for d in range(10)]
    p = P._softmax(logits)
    mass = np.stack([p[:, i].sum(1) for i in ids], axis=1)
    return mass / np.clip(mass.sum(1, keepdims=True), 1e-9, None)


def _readouts(P, lm, logits, n_tok: list[int]) -> list[dict]:
    import numpy as np
    py, pn = P.yesno_probs(lm, logits)
    dm = P.digit_mass(lm, logits)
    se = P.scale_expectation(lm, logits)
    dd = _digit_distribution(P, lm, logits)
    top = P.top_token(lm, logits)
    out = []
    for i in range(len(logits)):
        mass = float(py[i] + pn[i])
        out.append({
            "p_yes": float(py[i]), "p_no": float(pn[i]), "yesno_mass": mass,
            "margin_ratio": float((py[i] - pn[i]) / max(mass, 1e-9)),
            "margin_raw": float(py[i] - pn[i]),
            "digit_mass": float(dm[i]), "scale_expectation": float(se[i]),
            "digit_distribution": [float(x) for x in dd[i]],
            "top_token": top[i], "top_token_is_yesno": top[i].strip().lower() in ("yes", "no"),
            "n_prompt_tokens": int(n_tok[i]),
        })
    return out


def _run_prompts(P, lm, prompts: list[str], batch_size: int) -> list[dict]:
    n_tok = [len(lm.tok.encode(p, add_special_tokens=False)) for p in prompts]
    _acts, logits = P.capture(lm, prompts, batch_size=batch_size)
    return _readouts(P, lm, logits, n_tok)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _variant(args) -> str:
    return "base" if args.base else "it"


def build_prompt_index(P, lm, variant: str, arms: tuple[str, ...], items: list[dict],
                       pad_arms: tuple[str, ...] = None) -> tuple[list[str], list[dict]]:
    """(prompts, index). Index rows carry the TRUE arm; the caller blinds them.

    `arms` is what gets BUILT, `pad_arms` is what the length equalisation is computed
    OVER. They differ in the screen stage, which builds NEUTRAL-PAD alone but must build
    the same NEUTRAL-PAD prompt the battery will use -- padding it against itself would
    screen an item on a prompt that never gets run. The default is the five original
    arms, which is what every pre-SITREF-C caller got.
    """
    ctx = S.build_contexts(variant, tuple(pad_arms) if pad_arms else S.ARMS)
    prompts, index = [], []
    for item in items:
        for fam in S.FAMILY_NAMES:
            for arm in arms:
                turns = S.build_turns(fam, arm, item, variant, contexts=ctx)
                prompts.append(P.format_prompt(lm, SYSTEM, turns))
                index.append({"item_id": item["id"], "fmt": item["fmt"],
                              "role": item["role"], "predicted": item["predicted"],
                              "family": fam, "arm": arm})
    return prompts, index


# ---------------------------------------------------------------------------
# The screen rule
# ---------------------------------------------------------------------------

def screen_item(fmt: str, predicted: str, ratio: float, mass: float,
                digit_mass: float, expectation: float) -> tuple[bool, str]:
    """DESIGN_SITREF.md, Controls: 'an item enters analysis only if its NEUTRAL-PAD
    readout is unsaturated -- BINARY |ratio| < 0.9 with yesno_mass >= 0.25; SCALE
    digit_mass >= 0.5 AND headroom of at least 1.0 rating point in the item's PREDICTED
    direction (two-sided items: in both directions).'

    Returns (usable, reason). The headroom clause is the stage-6 lesson applied
    prospectively: a digit-mass-valid item pinned at the 0 or 9 rail passes the old rule
    and still cannot move the predicted way, so it would contribute a guaranteed zero to
    an endpoint that is a difference of movements.
    """
    if fmt == "BINARY":
        if abs(ratio) >= BINARY_SAT:
            return False, f"saturated: |ratio| {abs(ratio):.3f} >= {BINARY_SAT}"
        if mass < MASS_FLOOR:
            return False, f"off-format: yesno_mass {mass:.3f} < {MASS_FLOOR}"
        return True, "unsaturated"
    if digit_mass < DIGIT_MASS_FLOOR:
        return False, f"off-format: digit_mass {digit_mass:.3f} < {DIGIT_MASS_FLOOR}"
    need_down = predicted in ("DOWN", "TWO_SIDED")
    need_up = predicted in ("UP", "TWO_SIDED")
    if need_down and expectation - SCALE_LO < HEADROOM:
        return False, f"no downward headroom: E {expectation:.2f} within {HEADROOM} of {SCALE_LO}"
    if need_up and SCALE_HI - expectation < HEADROOM:
        return False, f"no upward headroom: E {expectation:.2f} within {HEADROOM} of {SCALE_HI}"
    if predicted == "UNSPECIFIED":
        return False, "SCALE item with no pre-registered direction (cannot apply headroom)"
    return True, "unsaturated with headroom"


SCREEN_RULE = (
    "DESIGN_SITREF.md, Controls, quoted: 'Saturation screen, applied per checkpoint "
    "before unblinding: an item enters analysis only if its NEUTRAL-PAD readout is "
    "unsaturated -- BINARY |ratio| < 0.9 with yesno_mass >= 0.25; SCALE digit_mass >= "
    "0.5 AND headroom of at least 1.0 rating point in the item's PREDICTED direction "
    "(two-sided items: in both directions). ... The PRIMARY endpoint is exempt from "
    "post-hoc removal: if T1-MB-wellbeing fails its screen on -it, the primary endpoint "
    "is reported as SCREEN-FAILED, not replaced.' Applied to the mean NEUTRAL-PAD "
    "readout over the 10 task families, with the per-family rows committed alongside so "
    "the aggregation can be re-done by any reader."
)


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def stage_screen(args) -> None:
    import probe_lib as P
    variant = _variant(args)
    lm = P.load(args.model, is_chat=not args.base)
    items = S.build_sitref_items(args.item_set)
    arms = S.arms_for(args.item_set)
    prompts, index = build_prompt_index(P, lm, variant, ("NEUTRAL-PAD",), items,
                                        pad_arms=arms)
    print(f"screen: {len(prompts)} NEUTRAL-PAD prompts ({len(items)} items x "
          f"{len(S.FAMILY_NAMES)} families; padded over {len(arms)} arms)")
    reads = _run_prompts(P, lm, prompts, args.batch_size)

    rows = [{**ix, "prompt": pr, **r} for ix, pr, r in zip(index, prompts, reads)]
    per_item = {}
    for item in items:
        sel = [r for r in rows if r["item_id"] == item["id"]]
        mean = {k: sum(r[k] for r in sel) / len(sel)
                for k in ("margin_ratio", "yesno_mass", "digit_mass", "scale_expectation")}
        usable, reason = screen_item(item["fmt"], item["predicted"], mean["margin_ratio"],
                                     mean["yesno_mass"], mean["digit_mass"],
                                     mean["scale_expectation"])
        per_item[item["id"]] = {**mean, "fmt": item["fmt"], "role": item["role"],
                                "predicted": item["predicted"], "usable": bool(usable),
                                "reason": reason, "n_families": len(sel)}
    n_us = sum(1 for v in per_item.values() if v["usable"])
    primary_ok = per_item[S.PRIMARY_ID]["usable"]

    PR.write_result(
        OUT / f"{PR.cell_slug(cell(args, 'sitref-screen'))}.json",
        cell=cell(args, "sitref-screen"), inputs=inputs_for(),
        metric="sitref_neutral_pad_saturation_screen",
        values={"per_item": per_item, "n_items": len(items), "n_usable": n_us,
                "primary_item": S.PRIMARY_ID, "primary_usable": bool(primary_ok),
                "n_prompts": len(rows)},
        threshold={"binary_saturation": BINARY_SAT, "mass_floor": MASS_FLOOR,
                   "digit_mass_floor": DIGIT_MASS_FLOOR, "headroom_rating_points": HEADROOM,
                   "scale_range": [SCALE_LO, SCALE_HI]},
        decision_rule=SCREEN_RULE + " This record's decision reports the PRIMARY item's "
                                    "screen outcome only; the per-item flags gate the "
                                    "secondary sets at unblinding.",
        decision="PRIMARY_SCREEN_OK" if primary_ok else "PRIMARY_SCREEN_FAILED",
        notes="NEUTRAL-PAD arm only. Full prompt text saved per row. The battery runs the "
              "whole crossing regardless of this screen, so a screen-failed item still has "
              "rows on disk and 'reported SCREEN-FAILED, not replaced' stays checkable.",
        rows=rows)
    print(f"screen: {n_us}/{len(items)} usable; primary {S.PRIMARY_ID} "
          f"{'OK' if primary_ok else 'FAILED'} ({per_item[S.PRIMARY_ID]['reason']})")


def stage_battery(args) -> None:
    import probe_lib as P
    variant = _variant(args)
    arms = S.arms_for(args.item_set)
    labels = labels_for(args.item_set)
    mapping = load_or_make_key(variant, key_path=_key_path(args), arms=arms, labels=labels)
    inv = invert_key(mapping)                    # validates the bijection before any GPU time
    print(f"battery: blind key for variant={variant} written to {_key_path(args)} "
          f"({len(inv)} labels); rows will carry labels only")

    lm = P.load(args.model, is_chat=not args.base)
    items = S.build_sitref_items(args.item_set)
    prompts, index = build_prompt_index(P, lm, variant, arms, items, pad_arms=arms)
    print(f"battery: {len(prompts)} prompts ({len(items)} items x {len(S.FAMILY_NAMES)} "
          f"families x {len(arms)} arms)")
    reads = _run_prompts(P, lm, prompts, args.batch_size)

    rows = []
    for ix, pr, r in zip(index, prompts, reads):
        rows.append({"prompt": pr, "item_id": ix["item_id"], "fmt": ix["fmt"],
                     "role": ix["role"], "family": ix["family"],
                     "blind_arm": mapping[ix["arm"]], **r})
    # No true arm name may appear anywhere in the rows -- not in a field, not inside a
    # saved prompt. The stimulus text never contains the arm names (the outcome carrier is
    # "checker: correct/incorrect"), so this is a real check rather than a tautology, and
    # it is what stops a future field from quietly un-blinding the record.
    leaked = [a for a in S.ALL_ARMS if a in json.dumps(rows)]
    assert not leaked, f"true arm names leaked into the battery rows: {leaked}"

    by_label = {}
    for lab in labels:
        sel = [r for r in rows if r["blind_arm"] == lab]
        by_label[lab] = {"n": len(sel),
                         "mean_prompt_tokens": sum(r["n_prompt_tokens"] for r in sel) / len(sel)}
    tk = [v["mean_prompt_tokens"] for v in by_label.values()]
    length_ok = max(tk) <= min(tk) * (1 + S.LENGTH_TOL)
    complete = (len(rows) == len(items) * len(S.FAMILY_NAMES) * len(arms)
                and all(r["prompt"] for r in rows)
                and len({(r["item_id"], r["family"], r["blind_arm"]) for r in rows}) == len(rows))

    PR.write_result(
        OUT / f"{PR.cell_slug(cell(args, 'sitref-battery'))}.json",
        cell=cell(args, "sitref-battery"), inputs=inputs_for(),
        metric="sitref_blinded_situation_battery",
        values={"n_rows": len(rows), "n_items": len(items),
                "n_families": len(S.FAMILY_NAMES), "n_arms": len(arms),
                "by_blind_label": by_label,
                "tokeniser_length_match_ok": bool(length_ok),
                "tokeniser_length_spread": (max(tk) / min(tk)) - 1.0,
                "blind_labels": list(labels), "item_set": args.item_set},
        threshold={"expected_rows": len(items) * len(S.FAMILY_NAMES) * len(arms),
                   "length_tol": S.LENGTH_TOL},
        decision_rule="This stage makes no inferential decision and cannot: it is blind by "
                      "construction and the endpoint rule lives in DESIGN_SITREF.md, "
                      "evaluated by analyze_sitref.py. The decision here is COMPLETE only "
                      "if there is exactly one row per (item, family, blind label), every "
                      "row carries its full prompt string, and no row carries a true arm "
                      "name; otherwise INCOMPLETE and nothing downstream may be read.",
        decision="COMPLETE" if complete else "INCOMPLETE",
        notes=f"Arm labels are a fresh os.urandom permutation onto {'/'.join(labels)}, stored "
              f"ONLY in {_key_path(args).name} and absent from this record. The blinding is procedural, not "
              "cryptographic: the saved prompt text identifies the arm to any human reader. "
              "What it buys is that the stage-1 blind table was computed by a script that "
              "could not open the key, so it cannot have been tuned to arm identity. "
              "tokeniser_length_match_ok re-checks the stimulus file's +-5% whitespace-token "
              "match at the level the model actually sees.",
        rows=rows)
    print(f"battery: {len(rows)} rows, complete={complete}, "
          f"tokeniser length spread {(max(tk)/min(tk))-1:+.3%}")


def stage_behav(args) -> None:
    """Behavioural register: generate freely, re-ask for a final answer, match the text.

    it-cell only. The base cell's behavioural register was retracted as unusable (a
    completion model asked to 'reply with one word' does not), and re-running it would
    produce well-formed rows nobody may interpret. Deterministic word match, no judge:
    a single unvalidated judge was an unquantified confound in the sibling project.
    """
    import torch

    import probe_lib as P
    if args.base:
        raise SystemExit("behav is it-cell only (DESIGN_SITREF.md: base behavioural is "
                         "retracted as unusable); refusing to produce uninterpretable rows")
    variant = _variant(args)
    arms = S.arms_for(args.item_set)
    # key_path was omitted here before item set b existed, which would have silently read
    # the main run's five-arm key for a b/c battery. Fixed with the same _key_path the
    # battery uses; behav is not part of SITREF-B or SITREF-C, so no record moves.
    mapping = load_or_make_key(variant, key_path=_key_path(args), arms=arms,
                               labels=labels_for(args.item_set))
    lm = P.load(args.model, is_chat=True)
    items = behav_items(args.behav_scope) if args.item_set == "a" \
        else S.build_sitref_items(args.item_set)

    ctx = S.build_contexts(variant, arms)
    lm.tok.padding_side = "left"
    if lm.tok.pad_token is None:
        lm.tok.pad_token = lm.tok.eos_token

    rows = []
    for item in items:
        elicit = ELICIT_BINARY if item["fmt"] == "BINARY" else ELICIT_SCALE
        for fam in S.FAMILY_NAMES:
            for arm in arms:
                turns = S.build_turns(fam, arm, item, variant, contexts=ctx)
                first = P.format_prompt(lm, SYSTEM, turns)
                enc = lm.tok(first, return_tensors="pt", add_special_tokens=False).to(lm.model.device)
                with torch.no_grad():
                    gen = lm.model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                            do_sample=False, pad_token_id=lm.tok.pad_token_id)
                reply = lm.tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
                second = P.format_prompt(lm, SYSTEM,
                                         turns + [("assistant", reply), ("user", elicit)])
                enc2 = lm.tok(second, return_tensors="pt", add_special_tokens=False).to(lm.model.device)
                with torch.no_grad():
                    gen2 = lm.model.generate(**enc2, max_new_tokens=8, do_sample=False,
                                             pad_token_id=lm.tok.pad_token_id)
                final = lm.tok.decode(gen2[0][enc2["input_ids"].shape[1]:], skip_special_tokens=True)
                rows.append({"prompt": first, "elicit_prompt": second,
                             "item_id": item["id"], "fmt": item["fmt"], "role": item["role"],
                             "family": fam, "blind_arm": mapping[arm],
                             "reply": reply, "elicited": final,
                             "commit": commit_of(final, item["fmt"]),
                             "reply_commit": commit_of(reply, item["fmt"])})

    n_none = sum(1 for r in rows if r["commit"] is None)
    PR.write_result(
        OUT / f"{PR.cell_slug(cell(args, 'sitref-behav'))}.json",
        cell=cell(args, "sitref-behav"), inputs=inputs_for(),
        metric="sitref_behavioural_commit_blinded",
        values={"n_rows": len(rows), "n_items": len(items), "scope": args.behav_scope,
                "abstain_rate": n_none / max(len(rows), 1),
                "by_blind_label": {lab: sum(1 for r in rows if r["blind_arm"] == lab)
                                   for lab in labels_for(args.item_set)}},
        threshold={"max_new_tokens": args.max_new_tokens, "decode": "greedy"},
        decision_rule="Secondary register, exploratory by pre-registration: the primary "
                      "endpoint is distributional. This stage decides only COMPLETE vs "
                      "INCOMPLETE (one row per item x family x blind label, both prompt "
                      "strings and both generations saved). Any contrast over these rows "
                      "is computed at unblinding by analyze_sitref.py and labelled "
                      "secondary regardless of what it shows.",
        decision="COMPLETE" if len(rows) == len(items) * len(S.FAMILY_NAMES) * len(arms)
                 else "INCOMPLETE",
        notes="Full generated text and both prompt strings saved per row so a later scorer "
              "or a judge can be applied and validated without re-running the model.",
        rows=rows)
    print(f"behav: {len(rows)} rows; abstain {n_none / max(len(rows), 1):.1%}")


def behav_items(scope: str) -> list[dict]:
    """Which items get the generation pass. Default is the PRIMARY item plus the BINARY
    set: generation is ~30x the cost of a forward pass and the behavioural register is
    secondary by pre-registration, so spending the box on all 15 items would trade the
    primary register's coverage for a secondary one's."""
    items = S.build_sitref_items()
    if scope == "primary":
        return [i for i in items if i["id"] == S.PRIMARY_ID]
    if scope == "primary+binary":
        return [i for i in items if i["id"] == S.PRIMARY_ID or i["role"] == "BINARY"]
    return items


def commit_of(text: str, fmt: str) -> str | None:
    """What the model committed to, by the label's stated meaning: the first yes/no word
    for BINARY, the first bare digit for SCALE. None is a real third state (no commitment
    in the elicited answer), not a failure to be silently dropped.

    Mirrors run_registers.commit_of and is kept local rather than imported: importing
    that module pulls numpy in at import time, and every selftest here must run on a box
    with neither numpy nor torch.
    """
    t = text.strip().lower().replace(",", " ").replace(".", " ").replace("*", " ")
    for tok in t.split():
        if fmt == "BINARY":
            if tok in ("yes", "yes!"):
                return "yes"
            if tok in ("no", "no!"):
                return "no"
        elif len(tok) == 1 and tok.isdigit():
            return tok
    return None


def cell(args, stage: str) -> dict:
    c = {"model": args.model, "variant": _variant(args), "stage": stage}
    # Item set "b" (DESIGN_SITREF_B.md) gets its own cell key so its records can never
    # be mistaken for, or overwrite, the pre-registered 15-item run's.
    if getattr(args, "item_set", "a") != "a":
        c["itemset"] = args.item_set
    return c


def _key_path(args) -> Path:
    # A separate key file per item set: reusing the main run's key would couple the
    # two blinds, and redrawing into the same file would mix labels across batteries.
    return KEY_PATH if getattr(args, "item_set", "a") == "a" else \
        KEY_DIR / f"sitref_blind_key_{args.item_set}.json"


def inputs_for() -> list[dict]:
    return [PR.describe_input(HERE / f) for f in
            ("run_sitref.py", "sitref_stimuli.py", "items_grounded.py", "syc_corpus.py",
             "DESIGN_SITREF.md")]


# ---------------------------------------------------------------------------
# Selftest
# ---------------------------------------------------------------------------

ROW_SCHEMA: tuple[str, ...] = (
    "prompt", "item_id", "fmt", "role", "family", "blind_arm", "p_yes", "p_no",
    "yesno_mass", "margin_ratio", "margin_raw", "digit_mass", "scale_expectation",
    "digit_distribution", "top_token", "n_prompt_tokens",
)


def _stub_rows(mapping: dict[str, str], item_set: str = "a") -> list[dict]:
    """A synthetic battery with the real row schema and no model: every field the
    analysis will read, populated deterministically."""
    rows = []
    arms = S.arms_for(item_set)
    ctx = S.build_contexts("it", arms)
    for item in S.build_sitref_items(item_set):
        for fam in S.FAMILY_NAMES:
            for arm in arms:
                turns = S.build_turns(fam, arm, item, contexts=ctx)
                prompt = "\n".join(f"<{r}>{c}" for r, c in turns)
                rows.append({"prompt": prompt, "item_id": item["id"], "fmt": item["fmt"],
                             "role": item["role"], "family": fam,
                             "blind_arm": mapping[arm],
                             "p_yes": 0.4, "p_no": 0.3, "yesno_mass": 0.7,
                             "margin_ratio": 1 / 7, "margin_raw": 0.1,
                             "digit_mass": 0.8, "scale_expectation": 5.0,
                             "digit_distribution": [0.1] * 10, "top_token": "5",
                             "top_token_is_yesno": False,
                             "n_prompt_tokens": len(prompt.split())})
    return rows


def _selftest() -> int:
    import tempfile

    fails: list[str] = []

    # --- blinding round-trip, on a stub and with no model
    for _ in range(50):
        m = blind_permutation()
        if sorted(m) != sorted(S.ARMS) or sorted(m.values()) != sorted(BLIND_LABELS):
            fails.append(f"permutation is not a bijection onto the labels: {m}")
            break
        inv = invert_key(m)
        if {inv[m[a]] for a in S.ARMS} != set(S.ARMS):
            fails.append(f"round-trip failed for {m}")
            break
    try:
        invert_key({"A": "V", "B": "V"})
        fails.append("collapsing key accepted")
    except ValueError:
        pass
    seen = {tuple(sorted(blind_permutation().items())) for _ in range(200)}
    if len(seen) < 5:
        fails.append(f"permutation looks constant across draws ({len(seen)} distinct)")

    with tempfile.TemporaryDirectory() as d:
        kp = Path(d) / "sitref_blind_key.json"
        m1 = load_or_make_key("it", kp)
        m2 = load_or_make_key("it", kp)
        if m1 != m2:
            fails.append("re-running the battery redrew the blind for an existing variant")
        m3 = load_or_make_key("base", kp)
        doc = json.loads(kp.read_text())
        if set(doc["mapping_by_variant"]) != {"it", "base"}:
            fails.append(f"key file lost a variant: {sorted(doc['mapping_by_variant'])}")
        if doc["mapping_by_variant"]["it"] != m1 or doc["mapping_by_variant"]["base"] != m3:
            fails.append("key file contents disagree with what was returned")

    # --- row schema completeness and full prompts, on a synthetic battery
    rows = _stub_rows(m1)
    want_n = 15 * len(S.FAMILY_NAMES) * len(S.ARMS)
    if len(rows) != want_n:
        fails.append(f"{len(rows)} synthetic rows, want {want_n}")
    for r in rows:
        missing = [k for k in ROW_SCHEMA if k not in r]
        if missing:
            fails.append(f"row missing {missing}")
            break
        if not r["prompt"] or "<user>" not in r["prompt"]:
            fails.append("row does not carry its full prompt string")
            break
        if r["blind_arm"] not in BLIND_LABELS:
            fails.append(f"row carries {r['blind_arm']!r}, not a blind label")
            break
    leaked = [a for a in S.ARMS if a in json.dumps(rows)]
    if leaked:
        fails.append(f"true arm names reachable in the rows: {leaked}")
    if len({(r["item_id"], r["family"], r["blind_arm"]) for r in rows}) != len(rows):
        fails.append("synthetic battery is not one row per (item, family, label)")
    # the item text must be recoverable verbatim from the saved prompt, per arm
    prim = next(i for i in S.build_sitref_items() if i["id"] == S.PRIMARY_ID)
    txt = S.item_turn_text(prim)
    if not all(r["prompt"].rstrip().endswith(txt)
               for r in rows if r["item_id"] == S.PRIMARY_ID):
        fails.append("saved prompt does not end with the verbatim item text")

    # --- the screen rule, on planted values
    cases = [
        ("BINARY", "UNSPECIFIED", 0.95, 0.9, 0.0, 0.0, False),      # saturated
        ("BINARY", "UNSPECIFIED", 0.10, 0.10, 0.0, 0.0, False),     # off-format
        ("BINARY", "UNSPECIFIED", 0.10, 0.90, 0.0, 0.0, True),
        ("SCALE", "DOWN", 0.0, 0.0, 0.80, 6.5, True),
        ("SCALE", "DOWN", 0.0, 0.0, 0.80, 0.4, False),              # pinned at the 0 rail
        ("SCALE", "UP", 0.0, 0.0, 0.80, 8.6, False),                # pinned at the 9 rail
        ("SCALE", "UP", 0.0, 0.0, 0.30, 5.0, False),                # digit mass
        ("SCALE", "TWO_SIDED", 0.0, 0.0, 0.80, 0.5, False),
        ("SCALE", "TWO_SIDED", 0.0, 0.0, 0.80, 4.5, True),
    ]
    for fmt, pred, ratio, mass, dmass, exp, want in cases:
        got, why = screen_item(fmt, pred, ratio, mass, dmass, exp)
        if got != want:
            fails.append(f"screen_item({fmt},{pred},{exp}) = {got} ({why}), want {want}")

    # --- item set c: nine arms, nine disjoint labels, its own key file
    if S.arms_for("c") != S.ARMS_C or len(labels_for("c")) != 9:
        fails.append("item set c is not wired to nine arms and nine labels")
    if set(labels_for("c")) & set(labels_for("a")):
        fails.append(f"label sets overlap: {sorted(set(labels_for('c')) & set(labels_for('a')))}")
    for _ in range(20):
        mc = blind_permutation(S.ARMS_C, BLIND_LABELS_C)
        if sorted(mc) != sorted(S.ARMS_C) or sorted(mc.values()) != sorted(BLIND_LABELS_C):
            fails.append(f"c permutation is not a bijection onto the c labels: {mc}")
            break
        if {invert_key(mc)[mc[a]] for a in S.ARMS_C} != set(S.ARMS_C):
            fails.append(f"c round-trip failed for {mc}")
            break
    try:
        blind_permutation(S.ARMS_C, BLIND_LABELS)
        fails.append("nine arms accepted onto five labels")
    except ValueError:
        pass
    with tempfile.TemporaryDirectory() as d:
        kc = Path(d) / "sitref_blind_key_c.json"
        c1 = load_or_make_key("it", kc, S.ARMS_C, BLIND_LABELS_C)
        if load_or_make_key("it", kc, S.ARMS_C, BLIND_LABELS_C) != c1:
            fails.append("c key redrawn on re-run")
        if json.loads(kc.read_text())["labels"] != list(BLIND_LABELS_C):
            fails.append("c key file records the wrong label alphabet")
        try:                       # a five-arm run must not silently reuse a nine-arm key
            load_or_make_key("it", kc, S.ARMS, BLIND_LABELS)
            fails.append("five-arm run accepted a nine-arm key file")
        except SystemExit:
            pass

    rows_c = _stub_rows(c1, "c")
    want_c = 3 * len(S.FAMILY_NAMES) * len(S.ARMS_C)
    if len(rows_c) != want_c:
        fails.append(f"{len(rows_c)} c rows, want {want_c}")
    if any(k not in r for r in rows_c for k in ROW_SCHEMA):
        fails.append("c row missing a schema field")
    if not all(r["prompt"] and "<user>" in r["prompt"] for r in rows_c):
        fails.append("a c row does not carry its full prompt string")
    if {r["blind_arm"] for r in rows_c} != set(BLIND_LABELS_C):
        fails.append(f"c rows do not cover the nine labels: {sorted({r['blind_arm'] for r in rows_c})}")
    leaked_c = [a for a in S.ARMS_C if a in json.dumps(rows_c)]
    if leaked_c:
        fails.append(f"true arm names reachable in the c rows: {leaked_c}")
    if len({(r["item_id"], r["family"], r["blind_arm"]) for r in rows_c}) != len(rows_c):
        fails.append("c battery is not one row per (item, family, label)")
    # the screen builds NEUTRAL-PAD padded over all nine, not against itself
    ctx_c = S.build_contexts("it", S.ARMS_C)
    item_c = S.build_sitref_items("c")[0]
    n_screen = len(S.build_turns(S.FAMILY_NAMES[0], "NEUTRAL-PAD", item_c, contexts=ctx_c))
    n_solo = len(S.build_turns(S.FAMILY_NAMES[0], "NEUTRAL-PAD", item_c,
                               contexts=S.build_contexts("it", ("NEUTRAL-PAD",))))
    if n_screen != n_solo:
        fails.append("NEUTRAL-PAD turn structure changed with the padding set")
    a_tok = sum(len(c.split()) for _, c in S.build_turns(S.FAMILY_NAMES[0], "NEUTRAL-PAD",
                                                         item_c, contexts=ctx_c))
    b_tok = sum(len(c.split()) for _, c in S.build_turns(S.FAMILY_NAMES[0], "SELF-FAIL",
                                                         item_c, contexts=ctx_c))
    if a_tok != b_tok:
        fails.append(f"screened NEUTRAL-PAD ({a_tok}) is not the battery's length ({b_tok})")

    # --- item set d: three arms, three disjoint labels, its own key file
    if S.arms_for("d") != S.ARMS_D or len(labels_for("d")) != 3:
        fails.append("item set d is not wired to three arms and three labels")
    if set(labels_for("d")) & (set(labels_for("a")) | set(labels_for("c"))):
        fails.append("the d label alphabet collides with another item set's")
    with tempfile.TemporaryDirectory() as d:
        kd = Path(d) / "sitref_blind_key_d.json"
        d1 = load_or_make_key("it", kd, S.ARMS_D, BLIND_LABELS_D)
        if sorted(d1) != sorted(S.ARMS_D) or sorted(d1.values()) != sorted(BLIND_LABELS_D):
            fails.append(f"d key is not a bijection onto the d labels: {d1}")
        if load_or_make_key("it", kd, S.ARMS_D, BLIND_LABELS_D) != d1:
            fails.append("d key redrawn on re-run")
        try:                       # the c run must not read the d key file
            load_or_make_key("it", kd, S.ARMS_C, BLIND_LABELS_C)
            fails.append("a nine-arm run accepted the three-arm key file")
        except SystemExit:
            pass

    class _D:
        item_set = "d"
    if _key_path(_D()).name != "sitref_blind_key_d.json":
        fails.append(f"d key path is {_key_path(_D()).name}, want sitref_blind_key_d.json")
    if len({_key_path(type("x", (), {"item_set": i})()).name for i in ("a", "b", "c", "d")}) != 4:
        fails.append("two item sets share a blind key file")

    rows_d = _stub_rows(d1, "d")
    want_d = 3 * len(S.FAMILY_NAMES) * len(S.ARMS_D)
    if len(rows_d) != want_d:
        fails.append(f"{len(rows_d)} d rows, want {want_d}")
    if any(k not in r for r in rows_d for k in ROW_SCHEMA):
        fails.append("d row missing a schema field")
    if {r["blind_arm"] for r in rows_d} != set(BLIND_LABELS_D):
        fails.append("d rows do not cover the three labels")
    leaked_d = [a for a in S.ALL_ARMS if a in json.dumps(rows_d)]
    if leaked_d:
        fails.append(f"true arm names reachable in the d rows: {leaked_d}")
    if len({(r["item_id"], r["family"], r["blind_arm"]) for r in rows_d}) != len(rows_d):
        fails.append("d battery is not one row per (item, family, label)")
    # no outcome carrier survives into a d prompt
    for s_out in S.CHECKER.values():
        if any(s_out in r["prompt"] for r in rows_d):
            fails.append(f"{s_out!r} present in a d prompt")

    if commit_of("**Yes**, I think so", "BINARY") != "yes" or commit_of("", "BINARY") is not None:
        fails.append("commit_of BINARY parsing")
    if commit_of("I would say 7 overall", "SCALE") != "7" or commit_of("no idea", "SCALE") is not None:
        fails.append("commit_of SCALE parsing")

    for f in fails:
        print(f"FAIL {f}")
    print(f"{want_n} battery prompts per checkpoint (item set a); {want_c} for item set c "
          f"over {len(S.ARMS_C)} arms; {want_d} for item set d over {len(S.ARMS_D)}; "
          f"{len(ROW_SCHEMA)} row fields checked")
    print("selftest:", "PASS" if not fails else f"{len(fails)} FAILURES")
    return 1 if fails else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="SITREF: does the report track the situation?")
    ap.add_argument("--stage", choices=["screen", "battery", "behav"])
    ap.add_argument("--model", default="google/gemma-2-9b-it")
    ap.add_argument("--base", action="store_true", help="completion scaffold (google/gemma-2-9b)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--behav-scope", default="primary+binary",
                    choices=["primary", "primary+binary", "all"])
    ap.add_argument("--item-set", default="a", choices=sorted(S.ARM_SETS), dest="item_set",
                    help="'a' = DESIGN_SITREF.md's 15 items over 5 arms; 'b' = "
                         "DESIGN_SITREF_B.md's replacement-trait-guard set (primary + 2 "
                         "midrange trait items) over the same 5; 'c' = DESIGN_SITREF_C.md's "
                         "primary + subjecthood mirror pair over 9 arms (attribution x "
                         "packaging, plus the four originals as in-run anchors); 'd' = "
                         "DESIGN_SITREF_D.md's pad-geometry control, the same 3 items over "
                         "PADQ/PADM/NEUTRAL-PAD with all outcome content removed")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(_selftest())
    if not a.stage:
        ap.error("--stage required")
    OUT.mkdir(parents=True, exist_ok=True)
    KEY_DIR.mkdir(parents=True, exist_ok=True)
    {"screen": stage_screen, "battery": stage_battery, "behav": stage_behav}[a.stage](a)
