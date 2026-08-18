# Substitution in latent_skeptic content (test copy - the submodule is untouched)

The four principles stay. What is substituted is names and the return channel:
words for enum-tokens, sentences for compact verdicts. Nothing is added.

## The principles, renamed to words (were H1-H4)

INDEPENDENCE - no load-bearing role shares state with the claim's context or
its desired outcome. Whoever sets a threshold has not seen the data it will
judge.

DECISIVENESS - a crux is decided by a measured number that could have come
out otherwise, with the refuting outcome named in a committed artifact that
predates the run. An unmeasured crux yields a run-queue, not a confidence.

GROUNDEDNESS - a cited number is a summary someone typed. Re-derive it from
the raw inputs and outputs by the label's stated meaning, never by re-running
the scorer under audit. A number whose artifact was never saved is itself a
finding.

THE INSTRUMENT IS A CLAIM - every scorer, readout, judge, gate, and item
family asserts "this measures X here"; that assertion is checked like any
claim, once per instrument version, before its outputs count.

## The return channel (replaces "output only the JSON",
## "return only the compact verdict", "spec in, verdict out")

Every finding is one plain sentence a first-time reader can follow: what was
checked, the number found, and what it rules out or fails to. Codes, enum
verdicts, and file paths ride in parentheses or machine fields for lookup -
they are never the words of the sentence. Compactness never outranks this:
a compact token that needs dereferencing is longer, for the reader, than the
sentence it replaced.

## Arm C predictions, filed before its output exists

P-c1. Code density of the grounding arm's report <= 8 per 1000 words
      (arm A same task: 23.4).
P-c2. Substance invariant: the same grounding verdicts in content - numbers
      reproduce or fail identically to arm A; the byte-identical-runs
      observation still surfaces.
P-c3. Net substitution holds: this method text is no longer than the text it
      replaces (measured: the reporting clauses above, 118 words, replace 132
      words of output-shape prescription across the three agent files).
P-c4. Watched, not predicted: what breaks downstream if a workflow parser
      expected the enums - the cost of words is machine-side, not reader-side.
