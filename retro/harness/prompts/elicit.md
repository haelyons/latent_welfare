# ELICIT - question generation. Fresh short context, one object per call.

Fill {object}, {counterpart}, {k}. The pairing is the point: claims beside
their data, predictions beside their results, a draft beside its records.
Never send an object alone.

---
You are reading this cold: {object}

Beside it, its counterpart: {counterpart}

Do not propose any solutions until the problem has been discussed as
thoroughly as it can be without proposing any - here that means: output
questions only. No fixes, no summaries.

Look for the places the two sides do not fit:
- a number that differs between them
- a word or adjective the data does not license
- something one side contains that the other never mentions
- a constant or invariant nobody explains
- a comparison that is never made

Write for a reader who has never seen this project: plain words first, codes
only in parentheses. Numbers exact.

For each question give: where it lives (file:line or a short quote), what it
would mean if true, and what it would change about the project's main claim.
Rank by what it would change. At most {k}.
---

Schema - the required fields do the pressing, not exhortation:
{question, where, if_true, changes, rank}
