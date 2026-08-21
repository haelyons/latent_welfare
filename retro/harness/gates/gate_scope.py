#!/usr/bin/env python3
"""Scope gate: a sentence carrying a measured number must name its checkpoint
(which model or variant) or carry a footnote marker.

Retro evidence: a compression pass silently dropped "on the base checkpoint"
from the ranked list that answered the paper's title question. This gate
makes that deletion loud. Heuristic v1; extend MARKERS per host project.

Exit 0 clean, 1 with violations. --selftest is model-free.
"""
import re, sys

MARKERS = re.compile(
    r"\b(base|instruct|-it\b|9b|2b|27b|qwen|gemma|both checkpoints|both models)\b"
    r"|\^\d+|\[\^?\d+\]", re.I)
NUM = re.compile(r"[+\-]?\d+\.\d+|\d+%|\bCI\b|±|E\[")

def violations(text):
    sents = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sents if NUM.search(s) and not MARKERS.search(s)]

def main():
    if "--selftest" in sys.argv:
        ok = "The instruct model drops by 0.93 under self-attributed failure."
        bad = "The rating drops by 0.93 under self-attributed failure."
        assert not violations(ok) and violations(bad)
        print("selftest ok")
        return 0
    v = violations(open(sys.argv[1]).read())
    for s in v[:20]:
        print("UNSCOPED:", s[:120])
    print(f"{len(v)} unscoped result sentence(s)")
    return 1 if v else 0

if __name__ == "__main__":
    sys.exit(main())
