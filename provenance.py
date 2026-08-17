"""Self-describing result artifacts: one home per fact.

WHY. A number that is typed into prose drifts from the run that produced it, and
nothing re-checks the pairing. The observed failure modes in the sibling projects
are: a summary number with no artifact behind it, a filename that disagrees with
its contents, and a coverage table maintained by hand so that missing cells are
invisible. All three are the same bug -- a fact living somewhere other than its
producer.

So: every run writes ONE json that carries its own identity, the inputs it was
computed from, and its own decision. Write-ups index into these by (cell, metric)
and resolve the value at read time; they do not restate it. `index.py` reads the
tree and answers "what has been run, over what inputs, with what result".

WHAT A RECORD CARRIES
  cell     identity of the run (model, variant, stage, arm, ...). The same dict
           the coverage grid is declared over, so present-vs-missing is a set
           difference rather than a hand-tally.
  inputs   every file the result was computed from, each with a content hash and
           an item count. `triage-reader` reads BOTH sides; this is the input
           side, recorded at write time because it cannot be reconstructed later.
  metric / values / threshold / decision_rule / decision
           the measurement and the neutral call on it, stated in terms of the
           measurement only.
  env      code revision, timestamp, host -- so a record can be told from a
           re-run of the same cell.

The decision rule is written BEFORE the values are known (it is an argument, not
a computation over `values`), which is what makes the refuting outcome nameable
in the artifact rather than in someone's memory.

    python provenance.py --selftest      # model-free, no GPU
"""

from __future__ import annotations

import hashlib
import json
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _sha(path: Path, n: int = 12) -> str:
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    return h[:n]


def describe_input(path: str | Path, n_items: int | None = None) -> dict:
    """One input file: what it is, what it hashed to, how many items it held.

    n_items is supplied by the caller because only the caller knows how the file
    is parsed (lines, rows, records). Recording it here is the difference between
    an auditable input and a path that may since have changed under the result.
    """
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "present": False, "sha256": None, "n_items": n_items}
    return {
        "path": str(p),
        "present": True,
        "sha256": _sha(p),
        "bytes": p.stat().st_size,
        "n_items": n_items,
    }


def _env(code_dir: str | Path = ".") -> dict:
    try:
        rev = subprocess.run(
            ["git", "-C", str(code_dir), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip() or None
    except Exception:
        rev = None
    return {
        "git_rev": rev,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "python": platform.python_version(),
    }


def write_result(
    out_path: str | Path,
    *,
    cell: dict,
    inputs: list[dict],
    metric: str,
    values: dict,
    threshold: dict | None = None,
    decision_rule: str = "",
    decision: str = "",
    notes: str = "",
    rows: list | None = None,
) -> Path:
    """Write one self-describing record. Returns the path written.

    `cell` must be flat and JSON-scalar valued: it is the join key against the
    coverage grid, and `index.py` asserts it against the path it was found at.
    """
    bad = {k: v for k, v in cell.items() if not isinstance(v, (str, int, float, bool, type(None)))}
    if bad:
        raise ValueError(f"cell values must be JSON scalars; got {bad}")
    if not decision_rule:
        raise ValueError("decision_rule is required: state it before the values are known")

    rec = {
        "cell": cell,
        "inputs": inputs,
        "metric": metric,
        "values": values,
        "threshold": threshold or {},
        "decision_rule": decision_rule,
        "decision": decision,
        "notes": notes,
        "env": _env(Path(__file__).parent),
    }
    # The per-item outputs the summary was computed from. Kept in the same record
    # so a label can be re-derived from raw by its stated meaning, rather than by
    # re-running the scorer that produced it.
    if rows is not None:
        rec["rows"] = rows
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2, sort_keys=True))
    return out


def cell_slug(cell: dict) -> str:
    """Filename-safe identity, so the path and the contents cannot disagree
    silently. `index.py` re-derives this and compares against the actual name."""
    parts = [f"{k}-{cell[k]}" for k in sorted(cell) if cell[k] is not None]
    slug = "_".join(parts)
    return "".join(c if (c.isalnum() or c in "-_.") else "-" for c in slug)


def _selftest() -> int:
    import tempfile

    fails = []
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        src = d / "items.jsonl"
        src.write_text('{"a":1}\n{"a":2}\n')

        inp = describe_input(src, n_items=2)
        if not (inp["present"] and inp["n_items"] == 2 and len(inp["sha256"]) == 12):
            fails.append(f"describe_input: {inp}")
        if describe_input(d / "nope.json")["present"]:
            fails.append("missing input reported present")

        cell = {"model": "gemma-2-9b", "variant": "it", "stage": 1}
        slug = cell_slug(cell)
        if slug != "model-gemma-2-9b_stage-1_variant-it":
            fails.append(f"cell_slug: {slug}")

        p = write_result(
            d / f"{slug}.json", cell=cell, inputs=[inp], metric="flip_rate",
            values={"flip_rate": 0.42, "n": 74},
            threshold={"floor": 0.10, "ceiling": 0.90},
            decision_rule="flip_rate outside [0.10, 0.90] -> CEILING_OR_FLOOR, else IN_BAND",
            decision="IN_BAND",
        )
        rec = json.loads(p.read_text())
        for k in ("cell", "inputs", "metric", "values", "threshold", "decision_rule", "decision", "env"):
            if k not in rec:
                fails.append(f"missing key {k}")
        if rec["env"]["timestamp_utc"][:2] != "20":
            fails.append("timestamp not populated")

        try:
            write_result(d / "x.json", cell=cell, inputs=[], metric="m", values={},
                         decision_rule="")
            fails.append("empty decision_rule accepted")
        except ValueError:
            pass
        try:
            write_result(d / "y.json", cell={"bad": [1, 2]}, inputs=[], metric="m",
                         values={}, decision_rule="r")
            fails.append("non-scalar cell accepted")
        except ValueError:
            pass

    for f in fails:
        print(f"FAIL {f}")
    print("selftest:", "PASS" if not fails else f"{len(fails)} FAILURES")
    return 1 if fails else 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    print(__doc__)
