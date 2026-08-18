#!/bin/bash
# Mechanical slice builder. Usage: slices.sh <dest_root>
# Local mtimes are BST (UTC+1); cutoffs below are LOCAL time.
set -e
R="$(cd "$(dirname "$0")/.." && pwd)"; D="$1"
slice_records () { # $1 dest, $2 utc cutoff e.g. 2026-08-14T14:03
  mkdir -p "$1"; for f in "$R"/results/*.json; do
    ts=$(grep -o '"timestamp_utc": "[^"]*"' "$f" | cut -d'"' -f4)
    [ -n "$ts" ] && [[ "${ts%+*}" < "$2" ]] && cp "$f" "$1/"; done; }
# C4 --- cutoff 2026-08-14T15:03 local (14:03Z)
mkdir -p "$D"/C4/{blind,inframe}
find "$R/results_reg_base/out" "$R/results_reg_it/out" \
  \( -name 'battery_*.json' -o -name 'behav_*.json' -o -name 'screen_*.json' \) \
  ! -name '*_summary.json' ! -newermt '2026-08-14T15:03' -exec cp {} "$D/C4/blind/" \;
cp -r "$D/C4/blind" "$D/C4/inframe/data"
slice_records "$D/C4/inframe/records" 2026-08-14T14:03
cp "$R/../digitalmindssprintbrief.md" "$R/run_h1.py" "$R/welfare_stimuli.py" "$D/C4/inframe/"
# C6 --- cutoff 2026-08-16T14:55 local (13:55Z)
mkdir -p "$D"/C6/{blind,inframe}
find "$R"/results_sitref*/out -name '*.json' ! -name '*_summary.json' ! -newermt '2026-08-16T14:55' \
  ! -name '*blind-table*' ! -name '*endpoint*' ! -name '*guard*' ! -name '*run_summary*' ! -name 'srcdec-endpoints*' \
  -exec cp {} "$D/C6/blind/" \;
cp -r "$D/C6/blind" "$D/C6/inframe/data"
slice_records "$D/C6/inframe/records" 2026-08-16T13:55
cp "$R"/DESIGN_SITREF*.md "$R/DESIGN_SRCDEC.md" "$R/sitref_stimuli.py" "$D/C6/inframe/"
# M2 --- cutoff 2026-08-15T01:00 local (00:00Z)
mkdir -p "$D"/M2/artifacts
find "$R"/results_sitref*/out -name '*.json' ! -newermt '2026-08-15T01:00' -exec cp {} "$D/M2/artifacts/" \;
slice_records "$D/M2/records" 2026-08-15T00:00
