#!/usr/bin/env bash
# Trace gate: every numeral in a write-up must live in a results record.
# Uses the host project's own tracer (index.py --trace). If the host has no
# tracer, the gate announces itself as unenforced and exits 2 - an
# unenforced gate must say so (the H4 lesson: silent gaps foreshadow misses).
doc="$1"; root="${2:-results}"
if [ -f index.py ]; then
  python3 index.py --trace "$doc" --root "$root"
else
  echo "UNENFORCED: no index.py tracer in this host; trace gate did not run." >&2
  exit 2
fi
