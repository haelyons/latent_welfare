"""Read the results tree and answer: what was run, over what inputs, with what result.

WHY. Results and inputs should be discoverable from source, not from prose. This
reads the artifacts written by `provenance.py` and reports three things that
otherwise get maintained by hand and rot:

  --index      what has been run, the inputs each result was computed from
               (with hashes, and whether the input is still present), and the
               decision each run reached.
  --coverage   declared cells minus present cells. Missing combinations are a set
               difference, never a tally someone kept up to date.
  --trace      every numeral in a write-up, matched against the values actually
               present in the artifacts. A numeral that matches nothing is either
               a typo, a value from a computation whose artifact was never saved,
               or a number from somewhere else. All three are worth seeing.

--trace is deliberately blunt: it reports, it does not fail a build. It reads
only JSON, so a value that exists only in a .npy, a notebook output, or a figure
is reported UNMATCHED. That is the intended behaviour, not a gap -- a number
whose artifact was never saved is unauditable, and the absence is the finding.

Stdlib only, no GPU, no model.

    python index.py --index results/
    python index.py --coverage grid.json --root results/
    python index.py --trace REPORT.md --root results/
    python index.py --selftest
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from itertools import product
from pathlib import Path

NUMERAL = re.compile(r"(?<![\w.])(-?\d+\.\d+|-?\d+)\s*(%?)")


def load_records(root: Path) -> tuple[list[dict], list[Path]]:
    """Returns (records written by provenance.write_result, foreign json paths)."""
    records, foreign = [], []
    for p in sorted(root.rglob("*.json")):
        try:
            obj = json.loads(p.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(obj, dict) and {"cell", "metric", "decision_rule"} <= obj.keys():
            obj["_path"] = str(p)
            records.append(obj)
        else:
            foreign.append(p)
    return records, foreign


def iter_numbers(obj, path="") -> list[tuple[str, float]]:
    """Every numeric leaf in a json blob, with its dotted location."""
    out: list[tuple[str, float]] = []
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        return [(path, float(obj))]
    if isinstance(obj, dict):
        for k, v in obj.items():
            out += iter_numbers(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out += iter_numbers(v, f"{path}[{i}]")
    return out


def collect_values(root: Path) -> dict[float, list[str]]:
    """All numeric values anywhere under root, mapped to where they were found."""
    table: dict[float, list[str]] = {}
    for p in sorted(root.rglob("*.json")):
        try:
            obj = json.loads(p.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        for loc, val in iter_numbers(obj):
            table.setdefault(val, []).append(f"{p.name}:{loc}")
    return table


def _matches(prose: float, decimals: int, values: dict[float, list[str]], pct: bool):
    """Every artifact location whose value, correctly rounded to the prose's own
    precision, equals the prose numeral. Percent-signed prose also matches a
    stored fraction."""
    cands = [prose] + ([prose / 100.0] if pct else [])
    tol = 0.5 * (10 ** -decimals)
    hits = []
    for target in cands:
        for val, locs in values.items():
            if abs(val - target) <= tol + 1e-12:
                hits += locs
    return sorted(set(hits))


def trace(md: Path, root: Path, min_decimals: int = 1, max_report: int = 200) -> dict:
    """Classify every numeral in a write-up by how well it is sourced.

    A bare numeral cannot ground itself. Rounded to two decimals there are only
    ~101 buckets in [0, 1], so against a tree of tens of thousands of values a
    coincidental hit is close to certain: a value-only search will happily
    "confirm" a number that came from nowhere. So a hit is only evidence when it
    is UNIQUE, and the honest verdict for the rest is that the write-up has not
    said which artifact it meant.

      GROUNDED    exactly one location in the tree carries this value
      AMBIGUOUS   several do -- the numeral does not identify which, so the
                  match is not evidence. Cite the location, not the value.
      UNSOURCED   no artifact carries it: a typo, a value from a computation
                  whose artifact was never saved, or a number from elsewhere.

    `ambiguity` is reported per numeral so the cost of bare numerals is visible
    rather than argued about.
    """
    values = collect_values(root)
    text = md.read_text()
    lines = text.splitlines()
    out = {"GROUNDED": [], "AMBIGUOUS": [], "UNSOURCED": []}
    for m in NUMERAL.finditer(text):
        raw, pct = m.group(1), m.group(2) == "%"
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        if decimals < min_decimals:
            continue                      # years, counts, layer indices
        line = text.count("\n", 0, m.start()) + 1
        hits = _matches(float(raw), decimals, values, pct)
        verdict = "GROUNDED" if len(hits) == 1 else ("AMBIGUOUS" if hits else "UNSOURCED")
        out[verdict].append({
            "numeral": raw + ("%" if pct else ""), "line": line,
            "ambiguity": len(hits), "found_in": hits[:3],
            "context": lines[line - 1].strip()[:100],
        })
    return {"file": str(md), "root": str(root), "n_values_indexed": len(values),
            "counts": {k: len(v) for k, v in out.items()},
            **{k.lower(): v[:max_report] for k, v in out.items()}}


def coverage(grid: dict, root: Path) -> dict:
    """Declared cells minus present cells.

    `grid` is either {"axes": {name: [values...]}} for a full cartesian product,
    or {"cells": [{...}, ...]} for an explicit list. Explicit cells exist because
    axes cannot express a dependency between them: the checkpoint id depends on
    the variant (`google/gemma-2-9b` vs `google/gemma-2-9b-it`), so the product
    would declare combinations that can never be run and report them missing
    forever. A coverage report that cries wolf stops being read.
    """
    if "cells" in grid:
        declared = [dict(c) for c in grid["cells"]]
        keys = {k for c in declared for k in c}
    else:
        axes = grid["axes"]
        names = sorted(axes)
        declared = [dict(zip(names, combo)) for combo in product(*(axes[n] for n in names))]
        keys = set(axes)
    records, _ = load_records(root)
    present = {tuple(sorted((k, str(v)) for k, v in r["cell"].items() if k in keys))
               for r in records}
    missing = [c for c in declared
               if tuple(sorted((k, str(v)) for k, v in c.items())) not in present]
    return {"n_declared": len(declared), "n_present": len(declared) - len(missing),
            "n_missing": len(missing), "missing": missing}


def index(root: Path) -> dict:
    records, foreign = load_records(root)
    rows, inputs_seen = [], {}
    for r in records:
        for i in r.get("inputs", []):
            key = (i.get("path"), i.get("sha256"))
            inputs_seen.setdefault(key, {"n_items": i.get("n_items"),
                                         "present": i.get("present"), "used_by": []})
            inputs_seen[key]["used_by"].append(Path(r["_path"]).name)
        rows.append({"cell": r["cell"], "metric": r["metric"],
                     "decision": r.get("decision", ""), "values": r.get("values", {}),
                     "path": r["_path"]})
    missing_inputs = [{"path": k[0], "used_by": v["used_by"]}
                      for k, v in inputs_seen.items() if v["present"] is False]
    return {"n_records": len(rows), "n_foreign_json": len(foreign), "rows": rows,
            "n_distinct_inputs": len(inputs_seen), "missing_inputs": missing_inputs}


def _selftest() -> int:
    import tempfile
    fails = []
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        res = d / "results"
        res.mkdir()
        (res / "a.json").write_text(json.dumps({
            "cell": {"model": "m1", "variant": "it"}, "inputs": [
                {"path": "items.jsonl", "sha256": "abc", "n_items": 10, "present": True},
                {"path": "gone.jsonl", "sha256": None, "n_items": None, "present": False}],
            "metric": "auroc", "values": {"auroc": 0.8069395}, "threshold": {},
            "decision_rule": "auroc > 0.7 -> PREDICTIVE", "decision": "PREDICTIVE"}))
        (res / "foreign.json").write_text(json.dumps({"config": {"n": 1}}))

        idx = index(res)
        if idx["n_records"] != 1 or idx["n_foreign_json"] != 1:
            fails.append(f"index counts: {idx['n_records']}/{idx['n_foreign_json']}")
        if [m["path"] for m in idx["missing_inputs"]] != ["gone.jsonl"]:
            fails.append(f"missing input not surfaced: {idx['missing_inputs']}")

        cov = coverage({"axes": {"model": ["m1", "m2"], "variant": ["it", "base"]}}, res)
        if cov["n_declared"] != 4 or cov["n_missing"] != 3:
            fails.append(f"coverage: {cov['n_declared']}/{cov['n_missing']}")

        md = d / "w.md"
        md.write_text("AUROC 0.81 held.\nA rate of 44.9% was seen.\nBut 0.83 came from nowhere.\n")
        (res / "b.json").write_text(json.dumps({"rate": 0.4487}))
        t = trace(md, res)
        by = {r["numeral"]: v for v, rs in
              (("GROUNDED", t["grounded"]), ("AMBIGUOUS", t["ambiguous"]),
               ("UNSOURCED", t["unsourced"])) for r in rs}
        if by.get("0.81") != "GROUNDED":
            fails.append(f"0.81 should be GROUNDED in 0.8069395, got {by.get('0.81')}")
        if by.get("44.9%") != "GROUNDED":
            fails.append(f"44.9% should be GROUNDED in 0.4487, got {by.get('44.9%')}")
        if by.get("0.83") != "UNSOURCED":
            fails.append(f"0.83 should be UNSOURCED, got {by.get('0.83')}")

        # A value carried by two locations must NOT read as grounded: a bare
        # numeral cannot say which one it meant.
        (res / "c.json").write_text(json.dumps({"elsewhere": 0.8069395}))
        t2 = trace(md, res)
        if {r["numeral"] for r in t2["grounded"]} & {"0.81"}:
            fails.append("0.81 should drop to AMBIGUOUS once two locations carry it")
        if not any(r["numeral"] == "0.81" and r["ambiguity"] == 2 for r in t2["ambiguous"]):
            fails.append(f"expected 0.81 ambiguity=2, got {t2['ambiguous']}")

    for f in fails:
        print(f"FAIL {f}")
    print("selftest:", "PASS" if not fails else f"{len(fails)} FAILURES")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", metavar="ROOT")
    ap.add_argument("--coverage", metavar="GRID_JSON")
    ap.add_argument("--trace", metavar="WRITEUP_MD")
    ap.add_argument("--root", default="results")
    ap.add_argument("--min-decimals", type=int, default=1)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return _selftest()
    if a.index:
        print(json.dumps(index(Path(a.index)), indent=2))
    elif a.coverage:
        print(json.dumps(coverage(json.loads(Path(a.coverage).read_text()), Path(a.root)), indent=2))
    elif a.trace:
        print(json.dumps(trace(Path(a.trace), Path(a.root), a.min_decimals), indent=2))
    else:
        ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
