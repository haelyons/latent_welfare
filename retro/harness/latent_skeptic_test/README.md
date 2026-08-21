# latent_skeptic

Adversarial verification for empirical research claims, built from stateless parts. `HEURISTICS.md` is the binding contract; everything else is a way of applying it.

The parts, each independently usable and self-describing:

- `HEURISTICS.md`: the rules. Read first; nothing normative lives anywhere else.
- `agents/*.md`: claim-blind roles (reader grounds numbers in artifacts, runner executes one control verbatim, author writes one instrument from a spec). Each file is its own documentation. Invoke any of them directly with the Agent tool, alone or in a composition of your own, provided the composition honours the heuristics: fresh contexts, bounded evidence, one lens per agent.
- `triage_workflow.js`: an EXAMPLE composition, full per-claim confound fan-out with run/author queues. Self-documented in its header; other compositions can be produced the same way.

Install: from the host project root, `cp <clone>/agents/*.md .claude/agents/` (the harness does not discover agents inside the clone). Keep `settings.local.json` scoped to the instance ssh/scp/python; never `Bash(*)`.
