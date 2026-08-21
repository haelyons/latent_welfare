#!/usr/bin/env python3
"""Density gate: coined project codes per 1000 words of prose.

A code is a token a first-time reader cannot parse: project-minted all-caps
labels and stage references. The whitelist holds ordinary all-caps English
and field-standard acronyms. Threshold from the retro's rerun-B result:
readable text sat under 10 codes per 1000 words; anchored text sat at 40+.

Exit 0 under threshold, 1 over. --selftest is model-free.
"""
import re, sys

WHITELIST = {"A","I","OK","NO","YES","AND","OR","NOT","THE","TO","IN","ON",
             "GPU","API","JSON","CI","PDF","URL","CPU","RAM","II","III","IV",
             "TODO","NB","PS","QA","AUROC","LLM","MATS","FAQ","BST","UTC"}
CODE = re.compile(r"\b[A-Z][A-Z0-9_\-]{2,}\b")
STAGE = re.compile(r"\bstage[- ]?\d+\b", re.I)
THRESHOLD = 10.0

def density(text):
    words = len(text.split())
    if not words:
        return 0.0, 0, 0
    codes = [t for t in CODE.findall(text) if t not in WHITELIST]
    n = len(codes) + len(STAGE.findall(text))
    return 1000.0 * n / words, n, words

def main():
    if "--selftest" in sys.argv:
        good = ("The model answers seven on every row while the underlying "
                "distribution moves under self-attributed failure. ") * 30
        bad = ("The SITREF REF drops under SELFQ but not OTHERM; PADM "
               "convicts the pad; stage-25 stands as committed. ") * 30
        dg, _, _ = density(good)
        db, _, _ = density(bad)
        assert dg <= THRESHOLD < db, (dg, db)
        print(f"selftest ok: {dg:.1f} (pass) vs {db:.1f} (fail)")
        return 0
    d, n, w = density(open(sys.argv[1]).read())
    print(f"{sys.argv[1]}: {d:.1f} codes/1000 words ({n} codes, {w} words); threshold {THRESHOLD}")
    return 0 if d <= THRESHOLD else 1

if __name__ == "__main__":
    sys.exit(main())
