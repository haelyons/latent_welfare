# Harness (retro branch) - question elicitation + writing gates

Two layers. This directory is the harness: prompts, events, gates - the layer
where instructions bite (retro: a named per-prompt contract moved code density
twenty-fold; rewording the tool's method text moved nothing). The tool layer
is latent_skeptic_test/: a copy of latent_skeptic with the filed subtractions
and two missing surfaces applied. The published submodule is untouched.

Key function of each piece:
- EVENTS.md          when each prompt or gate fires, and the human duties no gate enforces
- prompts/elicit.md  turns claims-beside-data into ranked questions (the observer)
- prompts/steer.md   one dynamically named decision-point check, question and declarative forms
- prompts/contract.md the language contract for anything a person will read
- prompts/meta.md    the two standing suffixes, verbatim, with their mechanisms
- prompts/validate.md the instrument-validation ask (H4 surface)
- gates/             mechanical blocks: code density, scope tags, numeral tracing
- QUEUE.md           the only surface the human must read: ranked questions + verification items
- PREDICTIONS.md     every piece's filed prediction and how it gets evaluated

Rule carried from the retro: a piece that fails its rerun comes back out.
