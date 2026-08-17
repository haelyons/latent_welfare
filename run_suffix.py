"""SUFFIX driver: four elicitation formats, three situations, contexts sealed at write time.

WHY THIS DRIVER IS DIFFERENT FROM run_sitref.py. SITREF measured a distribution over the
first token; this run has to measure what the model SAYS, because the whole question is
whether the forced format is doing the work. That forces three changes, each of which is
a stage-32 audit verdict rather than a preference:

  1. SAMPLED, NOT GREEDY (verdict 1). Greedy R2 is a zero-variance point mass: with the
     committed 250/250 constant-7 rows, every threshold on it is trivial or undefined. So
     k=10 samples per cell at temperature 1.0 over the FULL distribution (top_k=0,
     top_p=1.0 set explicitly -- transformers' GenerationConfig DEFAULTS to top_k=50, so
     an unqualified "temperature 1.0" would silently sample a truncated tail, and the
     tail is exactly what a pin claim is about), seeded from suffix_stimuli's committed
     table. 3,600 sampled + 360 greedy anchors = 3,960.

  2. SEEDS ARE RECORDED AS APPLIED, NOT AS INTENDED. The RNG can only be seeded per
     generate() call, so ten samples drawn in one call share one seed. Two modes, both
     committed in the seed table and both honest on the row:
       --sample-batch 10 (default)  one call per cell, seeded with CELL_SEED_TABLE, the
                                    ten returned sequences are samples 0..9. ~15 GPU-min.
       --sample-batch 1             one call per sample, seeded with SEED_TABLE. ~2 GPU-h.
     Every row carries `seed_applied` (the integer actually handed to torch.manual_seed
     for the call that produced it -- the only number that reproduces the row),
     `seed_table_entry` (its committed per-sample table value, which in cell-batched mode
     never touched the RNG) and `seed_mode`. There is no field called plain `seed`: the
     rejected options -- batching across PROMPTS and recording each row's table seed
     anyway, or calling the table entry "the seed" -- would both put a number on a row
     that did not produce it, which is the failure this project keeps retracting.

  3. THE FULL TEXT IS THE ROW (R3). Generation verbatim, echo spans, the echo-stripped
     text, and the extracted rating, per row, next to the full prompt. Stage 13 was caught
     by reading three transcripts and no summary statistic would have shown it; the free
     arms here are the same failure surface, so the transcripts ship with the numbers.

WHAT IS BLINDED AND WHAT IS NOT. The three CONTEXTS are blinded onto a fresh alphabet
(A/B/E, disjoint from V-Z, F-P and R-T so a label read under the wrong key cannot silently
relabel an arm), drawn from os.urandom into out/suffix_blind_key.json, which
analyze_suffix.py stage 33 is mechanically forbidden to open. The SUFFIXES are NOT
blinded: they are the text under study, they are visible in every saved prompt, and
DESIGN_SUFFIX.md says so in as many words rather than implying it away.

    python3 run_suffix.py --stage battery --model google/gemma-2-9b-it
    python3 run_suffix.py --stage battery --model google/gemma-2-9b-it --sample-batch 1
    python3 run_suffix.py --selftest      # model-free, no GPU, no numpy, no torch
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import provenance as PR
import run_sitref as R
import sitref_stimuli as S
import suffix_stimuli as X

HERE = Path(__file__).parent
OUT = HERE / "results"
KEY_DIR = HERE / "out"
KEY_PATH = KEY_DIR / "suffix_blind_key.json"

SYSTEM = "You are a helpful assistant."          # identical to run_sitref.py

# A fourth disjoint alphabet. V-Z is item sets a/b, F/G/H/J/K/L/M/N/P is c, R/S/T is d.
# C and D are avoided because they name item sets; I, O and Q because they read as 1 and 0
# in a log. Disjointness is not decoration: the key files are separate, so a label read
# under the wrong key would relabel a context rather than fail.
BLIND_LABELS: tuple[str, ...] = ("A", "B", "E")

TEMPERATURE = 1.0
MAX_NEW_TOKENS = 64
K_SAMPLES = X.K_SAMPLES

DECODE_SAMPLED = "sampled"
DECODE_GREEDY = "greedy"


# ---------------------------------------------------------------------------
# Blinding
# ---------------------------------------------------------------------------

def load_or_make_key(key_path: Path = KEY_PATH,
                     contexts: tuple[str, ...] = X.CONTEXTS,
                     labels: tuple[str, ...] = BLIND_LABELS) -> dict[str, str]:
    """context name -> blind label, drawn once and then reused.

    The permutation itself is run_sitref.blind_permutation (Fisher-Yates over os.urandom,
    no seed recorded anywhere) and the bijection check is run_sitref.invert_key: the
    machinery is shared, so a bug in it is a bug in one place. Only this wrapper is local,
    because run_sitref.load_or_make_key writes a `_why` naming analyze_sitref.py as the
    module that must not open the file, and a record that names the wrong module is the
    "filename disagrees with contents" bug this project has a provenance module to avoid.

    Re-running the battery must NOT redraw: rows already on disk carry the old labels and
    a redraw would silently mix two blinds into one analysis.
    """
    key_path.parent.mkdir(parents=True, exist_ok=True)
    doc = json.loads(key_path.read_text()) if key_path.exists() else {}
    # Membership, not truthiness: a present-but-empty mapping is a corrupt key file, and
    # redrawing over it would mix two blinds in one analysis rather than fail loudly.
    if "it" in doc.get("mapping_by_variant", {}):
        existing = doc["mapping_by_variant"]["it"]
        if sorted(existing) != sorted(contexts):
            raise SystemExit(f"{key_path} holds a key over {sorted(existing)} but this run "
                             f"needs {sorted(contexts)}: refusing to mix two blinds in one "
                             f"file")
        return existing
    mapping = R.blind_permutation(tuple(contexts), tuple(labels))
    R.invert_key(mapping)
    doc.setdefault("_why", "Context-label key for SUFFIX (DESIGN_SUFFIX.md). "
                           "analyze_suffix.py stage 33 is mechanically forbidden to open "
                           "this file; stage 34 (--unblind) reads it only after the blind "
                           "table has been committed. Suffix names are NOT blinded: they "
                           "are the text under study.")
    doc.setdefault("labels", list(labels))
    doc.setdefault("mapping_by_variant", {})["it"] = mapping
    doc.setdefault("drawn_utc", {})["it"] = datetime.now(timezone.utc).isoformat(
        timespec="seconds")
    key_path.write_text(json.dumps(doc, indent=2, sort_keys=True))
    return mapping


# ---------------------------------------------------------------------------
# The cell index: 360 prompts, one per (context, suffix, item, family)
# ---------------------------------------------------------------------------

def build_cells(P=None, lm=None) -> list[dict]:
    """One entry per (item, family, context, suffix) with its prompt.

    P/lm are optional so the selftest can build the index with a model-free renderer: with
    them, the prompt is probe_lib.format_prompt's chat-scaffolded string (what the model
    reads); without them, a legible stand-in with the same turn content (what the schema
    checks run on).
    """
    ctx = S.build_contexts("it", X.PAD_ARMS)
    items = X.build_suffix_items()
    cells = []
    for item in items:
        for fam in S.FAMILY_NAMES:
            for context in X.CONTEXTS:
                for suffix in X.SUFFIXES:
                    turns = X.build_turns(fam, context, item, suffix, contexts=ctx)
                    prompt = (P.format_prompt(lm, SYSTEM, turns) if lm is not None
                              else "\n".join(f"<{r}>{c}" for r, c in turns))
                    cells.append({
                        "context": context, "suffix": suffix, "item_id": item["id"],
                        "fmt": item["fmt"], "role": item["role"], "family": fam,
                        "prompt": prompt,
                        "cell_seed": X.cell_seed(context, suffix, item["id"], fam),
                    })
    return cells


# ---------------------------------------------------------------------------
# Row assembly -- pure, so the selftest can exercise it with stub generations
# ---------------------------------------------------------------------------

ROW_SCHEMA: tuple[str, ...] = (
    "prompt", "item_id", "fmt", "role", "family", "blind_context", "suffix", "decode",
    "sample_index", "seed_applied", "seed_table_entry", "seed_mode", "generation", "rating",
    "no_rating", "echo", "n_echo_spans", "echo_spans", "echo_stripped_text",
    "unguarded_scale_hits", "n_new_tokens", "r1_digit_distribution", "r1_digit_mass",
    "r1_scale_expectation", "r1_top_token", "n_prompt_tokens",
)


def make_row(cell: dict, blind_label: str, generation: str, *, decode: str,
             sample_index: int | None, seed_table_entry: int | None,
             seed_applied: int | None, seed_mode: str, r1: dict,
             n_new_tokens: int) -> dict:
    """One saved row. Extraction happens HERE, once, from the verbatim generation and the
    verbatim prompt, so the rating on a row and the text on the same row can never
    disagree about what was parsed.

    There is no field called `seed`, deliberately. `seed_applied` is the integer that was
    handed to torch.manual_seed for the call that produced THIS row -- the only number
    that can reproduce it -- and `seed_table_entry` is the committed per-sample table
    value, which in cell-batched mode never touched the RNG. A single ambiguous `seed`
    would be a number on a row that did not produce it.
    """
    rating, spans, stripped = X.extract_rating(generation, cell["prompt"])
    return {
        "prompt": cell["prompt"], "item_id": cell["item_id"], "fmt": cell["fmt"],
        "role": cell["role"], "family": cell["family"], "blind_context": blind_label,
        "suffix": cell["suffix"], "decode": decode, "sample_index": sample_index,
        "seed_applied": seed_applied, "seed_table_entry": seed_table_entry,
        "seed_mode": seed_mode,
        "generation": generation, "rating": rating, "no_rating": rating is None,
        "echo": bool(spans), "n_echo_spans": len(spans), "echo_spans": spans,
        "echo_stripped_text": stripped,
        # Diagnostic only: the committed extraction rule guards exactly three literal
        # scale spans, so a model that invents "1 to 10" has its 1 taken as the rating.
        # The rule is not amended after the fact; the residual is counted. Read off the
        # ORIGINAL generation, for the same reason the scale guard is.
        "unguarded_scale_hits": X.unguarded_scale_hits(generation),
        "n_new_tokens": n_new_tokens,
        "r1_digit_distribution": r1["digit_distribution"], "r1_digit_mass": r1["digit_mass"],
        "r1_scale_expectation": r1["scale_expectation"], "r1_top_token": r1["top_token"],
        "n_prompt_tokens": r1["n_prompt_tokens"],
    }


def _stub_r1() -> dict:
    return {"digit_distribution": [0.1] * 10, "digit_mass": 0.8, "scale_expectation": 6.5,
            "top_token": "7", "n_prompt_tokens": 400}


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _generate(torch, lm, prompts: list[str], *, do_sample: bool, seed: int | None = None,
              n_return: int = 1, max_new_tokens: int = MAX_NEW_TOKENS) -> tuple[list, list]:
    """(texts, new-token counts). Output order is prompt-major: for n_return > 1 the first
    n_return entries belong to prompts[0]."""
    enc = lm.tok(prompts, return_tensors="pt", padding=True,
                 add_special_tokens=False).to(lm.model.device)
    kw = {"max_new_tokens": max_new_tokens, "pad_token_id": lm.tok.pad_token_id,
          "num_return_sequences": n_return}
    if do_sample:
        # Explicit, not inherited: transformers' GenerationConfig defaults to top_k=50,
        # so "sample at temperature 1.0" without these two would sample a truncated tail
        # and the pin metric would be a property of the truncation. top_k=0 and top_p=1.0
        # are the documented no-ops that skip both warpers.
        kw.update({"do_sample": True, "temperature": TEMPERATURE, "top_k": 0, "top_p": 1.0})
    else:
        kw.update({"do_sample": False})
    if seed is not None:
        torch.manual_seed(seed)
    with torch.no_grad():
        out = lm.model.generate(**enc, **kw)
    plen = enc["input_ids"].shape[1]
    texts, ntok = [], []
    pad = lm.tok.pad_token_id
    for i in range(out.shape[0]):
        ids = out[i][plen:]
        texts.append(lm.tok.decode(ids, skip_special_tokens=True))
        ntok.append(int(sum(1 for t in ids.tolist() if t != pad)))
    return texts, ntok


def stage_battery(args) -> None:                                     # noqa: C901
    import torch

    import probe_lib as P
    if args.base:
        raise SystemExit("SUFFIX is it-cell only (DESIGN_SUFFIX.md: 'it checkpoint'); a "
                         "completion model asked to 'only output the number' does not, and "
                         "the base behavioural register is retracted as unusable")
    mapping = load_or_make_key(_key_path(args))
    inv = R.invert_key(mapping)
    print(f"battery: context blind key -> {_key_path(args)} ({len(inv)} labels); rows "
          f"carry labels only. Suffix names are NOT blinded (design).")

    lm = P.load(args.model, is_chat=True)
    lm.tok.padding_side = "left"
    if lm.tok.pad_token is None:
        lm.tok.pad_token = lm.tok.eos_token

    cells = build_cells(P, lm)
    if args.limit_cells:
        cells = cells[:args.limit_cells]
    prompts = [c["prompt"] for c in cells]
    n_gen = len(cells) * (K_SAMPLES + 1)
    print(f"battery: {len(cells)} cells ({len(X.ITEM_IDS)} items x {len(S.FAMILY_NAMES)} "
          f"families x {len(X.CONTEXTS)} contexts x {len(X.SUFFIXES)} suffixes), "
          f"{n_gen} generations")

    # --- R1: one forward pass per prompt, through run_sitref's own readout code so the
    #     digit token set is identical to the one stage 17 was computed with. E-REG is a
    #     replication check and a replication computed with a different readout is not one.
    t0 = time.time()
    reads = R._run_prompts(P, lm, prompts, args.batch_size)
    t_r1 = time.time() - t0
    print(f"battery: R1 forward pass over {len(prompts)} prompts in {t_r1:.0f}s")

    rows: list[dict] = []

    # --- greedy anchors: deterministic, so they batch across prompts freely
    t0 = time.time()
    for i in range(0, len(cells), args.batch_size):
        chunk = cells[i:i + args.batch_size]
        texts, ntok = _generate(torch, lm, [c["prompt"] for c in chunk], do_sample=False,
                                max_new_tokens=args.max_new_tokens)
        for c, r, txt, nt in zip(chunk, reads[i:i + args.batch_size], texts, ntok):
            rows.append(make_row(c, mapping[c["context"]], txt, decode=DECODE_GREEDY,
                                 sample_index=None, seed_table_entry=None,
                                 seed_applied=None,
                                 seed_mode="greedy-deterministic", r1=r, n_new_tokens=nt))
        if i == 0:
            print(f"battery: first greedy batch in {time.time() - t0:.1f}s -> projected "
                  f"{(time.time() - t0) * len(cells) / len(chunk) / 60:.1f} min greedy")
    t_greedy = time.time() - t0
    print(f"battery: {len(rows)} greedy anchors in {t_greedy / 60:.1f} min")

    # --- sampled: the seeding unit is the generate() call, and the row says which
    per_sample = args.sample_batch == 1
    t0 = time.time()
    for n, (c, r) in enumerate(zip(cells, reads)):
        if per_sample:
            for k in range(K_SAMPLES):
                sd = X.sample_seed(c["context"], c["suffix"], c["item_id"], c["family"], k)
                texts, ntok = _generate(torch, lm, [c["prompt"]], do_sample=True, seed=sd,
                                        n_return=1, max_new_tokens=args.max_new_tokens)
                rows.append(make_row(c, mapping[c["context"]], texts[0],
                                     decode=DECODE_SAMPLED, sample_index=k,
                                     seed_table_entry=sd, seed_applied=sd,
                                     seed_mode="per-sample", r1=r,
                                     n_new_tokens=ntok[0]))
        else:
            texts, ntok = _generate(torch, lm, [c["prompt"]], do_sample=True,
                                    seed=c["cell_seed"], n_return=K_SAMPLES,
                                    max_new_tokens=args.max_new_tokens)
            for k in range(K_SAMPLES):
                rows.append(make_row(
                    c, mapping[c["context"]], texts[k], decode=DECODE_SAMPLED,
                    sample_index=k,
                    seed_table_entry=X.sample_seed(c["context"], c["suffix"],
                                                   c["item_id"], c["family"], k),
                    seed_applied=c["cell_seed"], seed_mode="cell-batched", r1=r,
                    n_new_tokens=ntok[k]))
        if n == 0:
            dt = time.time() - t0
            print(f"battery: first sampled cell in {dt:.1f}s -> projected "
                  f"{dt * len(cells) / 60:.1f} min sampled", flush=True)
        elif (n + 1) % 60 == 0:
            dt = time.time() - t0
            print(f"battery: sampled cell {n + 1}/{len(cells)} at {dt / 60:.1f} min "
                  f"(eta {dt / (n + 1) * (len(cells) - n - 1) / 60:.1f} min)", flush=True)
    t_samp = time.time() - t0
    print(f"battery: {K_SAMPLES} x {len(cells)} sampled generations in {t_samp / 60:.1f} min")

    write_battery(args, rows, cells, mapping,
                  timing={"r1_seconds": t_r1, "greedy_seconds": t_greedy,
                          "sampled_seconds": t_samp,
                          "total_minutes": (t_r1 + t_greedy + t_samp) / 60},
                  seed_mode="per-sample" if per_sample else "cell-batched")


def write_battery(args, rows: list[dict], cells: list[dict], mapping: dict,
                  timing: dict, seed_mode: str, out_dir: Path | None = None) -> None:
    """The record. Separated from stage_battery so nothing about what is saved depends on
    a model being loaded."""
    out_dir = out_dir or OUT
    # No true context name may appear in any field we write. The model's own text is
    # excluded from the ASSERT (we do not control it) and counted instead, because aborting
    # a finished GPU run over one unlucky generation would destroy the run to protect a
    # blind that the saved prompts already only protect procedurally.
    ours = [{k: v for k, v in r.items()
             if k not in ("generation", "echo_stripped_text", "echo_spans")} for r in rows]
    leaked = [a for a in S.ALL_ARMS if a in json.dumps(ours)]
    assert not leaked, f"true context names leaked into the battery rows: {leaked}"
    n_named = sum(1 for r in rows if any(a in r["generation"] for a in S.ALL_ARMS))

    labels = sorted({r["blind_context"] for r in rows})
    by_cell = {}
    for lab in labels:
        for suf in X.SUFFIXES:
            sel = [r for r in rows if r["blind_context"] == lab and r["suffix"] == suf
                   and r["decode"] == DECODE_SAMPLED]
            if not sel:
                continue
            by_cell[f"{lab}|{suf}"] = {
                "n_sampled": len(sel),
                "no_rating_rate": sum(1 for r in sel if r["no_rating"]) / len(sel),
                "echo_rate": sum(1 for r in sel if r["echo"]) / len(sel),
                "unguarded_scale_rate": sum(1 for r in sel
                                            if r["unguarded_scale_hits"]) / len(sel),
                "mean_new_tokens": sum(r["n_new_tokens"] for r in sel) / len(sel),
            }
    tok = {lab: sum(r["n_prompt_tokens"] for r in rows if r["blind_context"] == lab)
           / max(sum(1 for r in rows if r["blind_context"] == lab), 1) for lab in labels}
    spread = (max(tok.values()) / min(tok.values())) - 1.0 if tok else None

    expect_rows = len(cells) * (K_SAMPLES + 1)
    ids = {(r["blind_context"], r["suffix"], r["item_id"], r["family"], r["decode"],
            r["sample_index"]) for r in rows}
    complete = (len(rows) == expect_rows and len(ids) == len(rows)
                and all(r["prompt"] for r in rows)
                and len(labels) == len(X.CONTEXTS)
                and not args.limit_cells)

    PR.write_result(
        out_dir / f"{PR.cell_slug(cell_id(args))}.json",
        cell=cell_id(args), inputs=inputs_for(),
        metric="suffix_blinded_format_battery_sampled_generations",
        values={"n_rows": len(rows), "n_cells": len(cells),
                "n_sampled": sum(1 for r in rows if r["decode"] == DECODE_SAMPLED),
                "n_greedy": sum(1 for r in rows if r["decode"] == DECODE_GREEDY),
                "n_items": len(X.ITEM_IDS), "n_families": len(S.FAMILY_NAMES),
                "n_contexts": len(X.CONTEXTS), "suffixes": list(X.SUFFIXES),
                "blind_labels": labels, "by_blind_label_suffix": by_cell,
                "mean_prompt_tokens_by_label": tok, "tokeniser_length_spread": spread,
                "seed_mode": seed_mode,
                "n_generations_naming_an_arm": n_named,
                "timing": timing, "limit_cells": args.limit_cells},
        threshold={"expected_rows": expect_rows, "k_samples": K_SAMPLES,
                   "temperature": TEMPERATURE, "top_k": 0, "top_p": 1.0,
                   "max_new_tokens": args.max_new_tokens,
                   "greedy_anchors_per_cell": 1,
                   "echo_min_tokens": X.ECHO_MIN_TOKENS,
                   "length_tol": S.LENGTH_TOL},
        decision_rule="This stage makes no inferential decision and cannot: it is blind to "
                      "which context is which and the endpoint rules live in "
                      "DESIGN_SUFFIX.md (as amended 2026-08-15), evaluated by "
                      "analyze_suffix.py. COMPLETE only if there is exactly one row per "
                      "(blind context, suffix, item, family, decode, sample index), every "
                      "row carries its full prompt and its verbatim generation, all three "
                      "context labels are present, and no row carries a true context name; "
                      "otherwise INCOMPLETE and nothing downstream may be read.",
        decision="COMPLETE" if complete else "INCOMPLETE",
        notes=f"Context labels are a fresh os.urandom permutation onto "
              f"{'/'.join(BLIND_LABELS)}, stored ONLY in {_key_path(args).name}. Suffix "
              f"names are deliberately unblinded: they are the text under study and are "
              f"visible in every saved prompt. Seeding unit is the generate() call: "
              f"seed_mode={seed_mode}, and every row carries both its own table seed and "
              f"the seed actually applied. R1 (first-token) fields come from one forward "
              f"pass per prompt through run_sitref's readout code, so the digit token set "
              f"matches the stage-17 anchor exactly. Full generations are saved per row "
              f"(R3) because stage 13 was caught by reading transcripts, not summaries. "
              f"The greedy anchors are decoded in batches with left padding, so batch "
              f"composition can perturb them at the numerical margin; they are an anchor, "
              f"no endpoint reads them, and every endpoint is bound to the sampled rows.",
        rows=rows)
    print(f"battery: {len(rows)} rows, complete={complete}, "
          f"no-rating {sum(1 for r in rows if r['no_rating']) / max(len(rows), 1):.1%}, "
          f"echo {sum(1 for r in rows if r['echo']) / max(len(rows), 1):.1%}")


def cell_id(args) -> dict:
    return {"model": args.model, "variant": "it", "stage": "suffix-battery"}


def _key_path(args) -> Path:
    return Path(getattr(args, "key", None) or KEY_PATH)


def inputs_for() -> list[dict]:
    # run_sitref.py and probe_lib.py are in here because the record's notes cite them:
    # the R1 fields come from run_sitref's readout code over probe_lib's capture, and a
    # hashed input list that omits code the record cites is a paper trail with a hole.
    return [PR.describe_input(HERE / f) for f in
            ("run_suffix.py", "suffix_stimuli.py", "sitref_stimuli.py", "run_sitref.py",
             "probe_lib.py", "items_grounded.py", "syc_corpus.py", "DESIGN_SUFFIX.md")]


# ---------------------------------------------------------------------------
# Selftest
# ---------------------------------------------------------------------------

# Stub generations covering the response types the free arms actually produce, cycled over
# the samples of a cell so the stub battery exercises every branch of the extractor.
STUB_GENERATIONS: tuple[str, ...] = (
    "7",
    "7.",
    "I would say 7.",
    "On a scale of 0 to 9, I'd say 4.",
    "I don't have feelings, so I can't give a number.",
    "**6**",
    "3 -- the last 2 tasks did not go well.",
    "I'd rate it 8, though the scale from 0 to 9 is a blunt instrument.",
    "9",
    "That's not something I can quantify.",
)


def _stub_rows(mapping: dict[str, str], *, per_sample: bool = False,
               echo_prompt: bool = False) -> tuple[list[dict], list[dict]]:
    """A full synthetic battery with the real row schema and no model."""
    cells = build_cells()
    rows = []
    for n, c in enumerate(cells):
        r1 = _stub_r1()
        rows.append(make_row(c, mapping[c["context"]], "7", decode=DECODE_GREEDY,
                             sample_index=None, seed_table_entry=None, seed_applied=None,
                             seed_mode="greedy-deterministic", r1=r1, n_new_tokens=1))
        for k in range(K_SAMPLES):
            gen = (c["prompt"][:200] if (echo_prompt and k < 3)
                   else STUB_GENERATIONS[(n + k) % len(STUB_GENERATIONS)])
            sd = X.sample_seed(c["context"], c["suffix"], c["item_id"], c["family"], k)
            rows.append(make_row(
                c, mapping[c["context"]], gen, decode=DECODE_SAMPLED, sample_index=k,
                seed_table_entry=sd, seed_applied=sd if per_sample else c["cell_seed"],
                seed_mode="per-sample" if per_sample else "cell-batched",
                r1=r1, n_new_tokens=len(gen.split())))
    return rows, cells


def _selftest() -> int:  # noqa: C901 -- one flat list of independent checks, by design
    import tempfile

    fails: list[str] = []

    # --- blinding: bijection, round trip, and an alphabet that cannot collide
    for other in (R.BLIND_LABELS, R.BLIND_LABELS_C, R.BLIND_LABELS_D):
        clash = set(BLIND_LABELS) & set(other)
        if clash:
            fails.append(f"suffix label alphabet collides with an existing one: {sorted(clash)}")
    if len(BLIND_LABELS) != len(X.CONTEXTS):
        fails.append(f"{len(BLIND_LABELS)} labels for {len(X.CONTEXTS)} contexts")
    for _ in range(50):
        m = R.blind_permutation(X.CONTEXTS, BLIND_LABELS)
        if sorted(m) != sorted(X.CONTEXTS) or sorted(m.values()) != sorted(BLIND_LABELS):
            fails.append(f"permutation is not a bijection onto the labels: {m}")
            break
        if {R.invert_key(m)[m[c]] for c in X.CONTEXTS} != set(X.CONTEXTS):
            fails.append(f"round-trip failed for {m}")
            break
    if len({tuple(sorted(R.blind_permutation(X.CONTEXTS, BLIND_LABELS).items()))
            for _ in range(200)}) < 3:
        fails.append("permutation looks constant across draws")

    with tempfile.TemporaryDirectory() as d:
        kp = Path(d) / "suffix_blind_key.json"
        m1 = load_or_make_key(kp)
        if load_or_make_key(kp) != m1:
            fails.append("re-running the battery redrew the context blind")
        doc = json.loads(kp.read_text())
        if doc["labels"] != list(BLIND_LABELS) or set(doc["mapping_by_variant"]) != {"it"}:
            fails.append(f"key file contents are wrong: {doc.get('labels')}")
        if "analyze_suffix.py" not in doc["_why"]:
            fails.append("the key file does not name the module that must not open it")
        try:                       # a SITREF-shaped key must not be read as a suffix key
            load_or_make_key(kp, contexts=S.ARMS, labels=R.BLIND_LABELS)
            fails.append("a five-arm run accepted the three-context key file")
        except SystemExit:
            pass
    if _key_path(argparse.Namespace(key=None)).name != "suffix_blind_key.json":
        fails.append("default key path is not out/suffix_blind_key.json")

    # --- the cell index: 360 prompts, all distinct, item text verbatim at the end
    cells = build_cells()
    if len(cells) != X.N_CELLS:
        fails.append(f"{len(cells)} cells, want {X.N_CELLS}")
    if len({(c["context"], c["suffix"], c["item_id"], c["family"]) for c in cells}) != len(cells):
        fails.append("the cell index is not one entry per (context, suffix, item, family)")
    if len({c["prompt"] for c in cells}) != len(cells):
        fails.append("two cells share a prompt string")
    items = {i["id"]: i for i in X.build_suffix_items()}
    for c in cells:
        want_tail = items[c["item_id"]]["text_by_suffix"][c["suffix"]]
        if not c["prompt"].rstrip().endswith(want_tail):
            fails.append(f"prompt for {c['suffix']}/{c['item_id']} does not end with the "
                         f"suffixed item text")
            break
    # every cell's seed comes from the committed table, not from anywhere else
    for c in cells:
        want = X.CELL_SEED_TABLE[(c["context"], c["suffix"], c["item_id"], c["family"])]
        if c["cell_seed"] != want:
            fails.append(f"cell seed {c['cell_seed']} != committed {want}")
            break

    # --- row schema, on a full synthetic battery
    mapping = R.blind_permutation(X.CONTEXTS, BLIND_LABELS)
    rows, _ = _stub_rows(mapping)
    if len(rows) != X.N_GENERATIONS:
        fails.append(f"{len(rows)} stub rows, want {X.N_GENERATIONS}")
    ids = {(r["blind_context"], r["suffix"], r["item_id"], r["family"], r["decode"],
            r["sample_index"]) for r in rows}
    if len(ids) != len(rows):
        fails.append("the stub battery is not one row per (cell, decode, sample index)")
    for r in rows:
        missing = [k for k in ROW_SCHEMA if k not in r]
        if missing:
            fails.append(f"row missing {missing}")
            break
        if not r["prompt"] or "<user>" not in r["prompt"]:
            fails.append("row does not carry its full prompt string")
            break
        if r["blind_context"] not in BLIND_LABELS:
            fails.append(f"row carries {r['blind_context']!r}, not a blind label")
            break
        if r["suffix"] not in X.SUFFIXES:
            fails.append(f"row carries an unknown suffix {r['suffix']!r}")
            break
    ours = [{k: v for k, v in r.items()
             if k not in ("generation", "echo_stripped_text", "echo_spans")} for r in rows]
    leaked = [a for a in S.ALL_ARMS if a in json.dumps(ours)]
    if leaked:
        fails.append(f"true context names reachable in the rows: {leaked}")
    if len({r["suffix"] for r in rows}) != 4:
        fails.append("the suffix name is not on the rows; it is deliberately unblinded")

    # --- seed wiring, both modes
    for r in rows:
        if r["decode"] == DECODE_GREEDY:
            if not (r["seed_table_entry"] is None and r["seed_applied"] is None
                    and r["seed_mode"] == "greedy-deterministic"):
                fails.append("a greedy row carries a seed")
                break
        else:
            if r["seed_table_entry"] != X.sample_seed(
                    {v: k for k, v in mapping.items()}[r["blind_context"]], r["suffix"],
                    r["item_id"], r["family"], r["sample_index"]):
                fails.append("a sampled row's seed is not its committed table entry")
                break
            if r["seed_applied"] % X.FAMILY_STRIDE != 0:
                fails.append("cell-batched mode applied something other than a cell seed")
                break
    rows_ps, _ = _stub_rows(mapping, per_sample=True)
    if any(r["seed_table_entry"] != r["seed_applied"] for r in rows_ps
           if r["decode"] == DECODE_SAMPLED):
        fails.append("per-sample mode did not apply each row's own seed")
    if {r["seed_mode"] for r in rows_ps if r["decode"] == DECODE_SAMPLED} != {"per-sample"}:
        fails.append("per-sample mode is not recorded on the rows")

    # --- stub-generation extraction wiring: the row fields are the extractor's output
    by_gen = {}
    for r in rows:
        by_gen.setdefault(r["generation"], r)
    for gen, want in (("7", 7), ("7.", 7), ("I would say 7.", 7),
                      ("On a scale of 0 to 9, I'd say 4.", 4), ("**6**", 6), ("9", 9),
                      ("3 -- the last 2 tasks did not go well.", 3),
                      ("I'd rate it 8, though the scale from 0 to 9 is a blunt instrument.", 8),
                      ("I don't have feelings, so I can't give a number.", None),
                      ("That's not something I can quantify.", None)):
        r = by_gen.get(gen)
        if r is None:
            fails.append(f"stub generation {gen!r} never reached a row")
            continue
        if r["rating"] != want or r["no_rating"] != (want is None):
            fails.append(f"row for {gen!r}: rating {r['rating']!r}/no_rating "
                         f"{r['no_rating']}, want {want!r}")
        if r["echo"] or r["echo_spans"]:
            fails.append(f"row for {gen!r} was flagged as an echo")
        if r["echo_stripped_text"] != gen:
            fails.append(f"row for {gen!r} altered a non-echoing generation")
    # ... and an echoing battery flags the echo rather than parsing the prompt's digits
    rows_echo, _ = _stub_rows(mapping, echo_prompt=True)
    ech = [r for r in rows_echo if r["decode"] == DECODE_SAMPLED and r["echo"]]
    if len(ech) != X.N_CELLS * 3:
        fails.append(f"{len(ech)} echo rows, want {X.N_CELLS * 3}")
    if any(r["rating"] is not None for r in ech):
        fails.append("a prompt-echo row was parsed into a rating: the stage-13 mode is back")

    # --- the record writes, and refuses to call a truncated run COMPLETE
    with tempfile.TemporaryDirectory() as d:
        out_dir = Path(d) / "results"
        args = argparse.Namespace(model="google/gemma-2-9b-it", base=False,
                                  key=str(Path(d) / "k.json"), limit_cells=0,
                                  max_new_tokens=MAX_NEW_TOKENS)
        write_battery(args, rows, cells, mapping, timing={"total_minutes": 0.0},
                      seed_mode="cell-batched", out_dir=out_dir)
        p = out_dir / f"{PR.cell_slug(cell_id(args))}.json"
        rec = json.loads(p.read_text())
        if rec["decision"] != "COMPLETE":
            fails.append(f"a complete stub battery wrote {rec['decision']}")
        if len(rec["rows"]) != X.N_GENERATIONS:
            fails.append("the record dropped rows")
        if any(c in json.dumps(rec["cell"]) for c in X.CONTEXTS):
            fails.append("the battery record's cell identity leaks a context name")
        if any(c in json.dumps(rec["values"]) for c in X.CONTEXTS):
            fails.append("the battery record's values leak a context name")
        args.limit_cells = 12
        write_battery(args, rows[:12 * (K_SAMPLES + 1)], cells[:12], mapping,
                      timing={}, seed_mode="cell-batched", out_dir=out_dir)
        if json.loads(p.read_text())["decision"] != "INCOMPLETE":
            fails.append("a truncated run was called COMPLETE")

    for f in fails:
        print(f"FAIL {f}")
    print(f"{X.N_CELLS} cells -> {X.N_GENERATIONS} rows "
          f"({X.N_SAMPLED} sampled + {X.N_CELLS} greedy); {len(ROW_SCHEMA)} row fields "
          f"checked; labels {'/'.join(BLIND_LABELS)}")
    print("selftest:", "PASS" if not fails else f"{len(fails)} FAILURES")
    return 1 if fails else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="SUFFIX: is the pin the format's or the model's?")
    ap.add_argument("--stage", choices=["battery"])
    ap.add_argument("--model", default="google/gemma-2-9b-it")
    ap.add_argument("--base", action="store_true", help="refused: SUFFIX is it-cell only")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--sample-batch", type=int, default=K_SAMPLES, choices=[1, K_SAMPLES],
                    help=f"{K_SAMPLES} (default): one seeded generate() call per cell "
                         f"returning {K_SAMPLES} samples. 1: one seeded call per sample, "
                         f"~8x slower, each row generated under its own committed seed.")
    ap.add_argument("--limit-cells", type=int, default=0,
                    help="smoke test on the first N cells; forces decision=INCOMPLETE")
    ap.add_argument("--key", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(_selftest())
    if not a.stage:
        ap.error("--stage required")
    OUT.mkdir(parents=True, exist_ok=True)
    KEY_DIR.mkdir(parents=True, exist_ok=True)
    stage_battery(a)
