// EXAMPLE COMPOSITION of the latent_skeptic parts (rules: HEURISTICS.md; roles: agents/*.md).
// Invoke: Workflow({ scriptPath: "<clone>/triage_workflow.js", args: { claims: [
//   { id: "<slug>", text: "<one-sentence claim>", evidence: "<committed numbers/file:line only>", confounds: [] }
// ], execute_runs: false } })
//   text: the claim, not the story. evidence: the bounded slice only (H1 sets why). confounds []: general
//   defaults below; override only to add a project-specific one. Optional: ground:false skips the reader
//   pass; author:true fans claim-blind authors over the author_queue; execute_runs:true only after reading
//   the queues (pass 1 is free of GPU: skeptics acquit what committed numbers kill, the rest queue).
// Between passes, make each run_queue item a full command (e.g. prefix the GPU launcher); author_queue
//   items are specs, written to controls/ for review before running.
// Output per claim: grounding (reader verdict), findings = raw {confound, status, crux} (no rollup by
//   design; for a NULL claim an EXPLAINS can be the claim holding, the crux says which), run_queue,
//   author_queue, runner_results. Top-level shared_queue = controls needed by >= 2 claims; run once.
// Config: model inherits the session; per-role efforts set below; agent model: fields are defaults,
//   downsizable per the Operating discipline. Keep rounds:1 (breadth is one-confound-per-skeptic, not
//   re-sampling). Keep prompt strings free of em-dashes.
export const meta = {
  name: 'claim-triage',
  description: 'Adversarial triage of measured research claims: an isolated reader grounds each claim in its primary artifacts (H3); fresh independent skeptics (one confound each) reason over the committed numbers and emit a raw status + crux per confound (H1; H2 number-gates: falsifiability, margin, scope); unresolved cruxes become a run-queue (existing controls) or author-queue (controls to write); an optional read-only runner executes the run-queue. Queue items shared across claims are hoisted to a shared_queue so instrument-level checks run once, not per-claim (H4). No rolled-up score or verdict (the human reads the cruxes).',
  phases: [
    { title: 'Ground' },
    { title: 'Skeptics' },
    { title: 'Adjudicate' },
    { title: 'Verify' },
    { title: 'Author' },
  ],
}

// Prompts are short and reused; schemas are kept to a small flat tail. Routing
// rules live in the prompt, not in nested schema. The runner's discipline lives
// once in agents/triage-runner.md, so the per-call message stays tiny.

// General confound menu (not project-specific). A claim may override via its
// own confounds: field. Tier A (first two) are intervention-validity confounds
// general to any ablation; the rest are general measurement confounds.
const DEFAULT_CONFOUNDS = [
  'off-distribution ablation',
  'downstream compensation (self-repair)',
  'selection bias',
  'readout artifact',
  'regime specificity',
  'construct validity',
  'scale incomparability',
  'noise floor',
  'ceiling or floor effect',
  'single-case overfit',
]

// ---- SKEPTIC (the analytical agent): one fresh reasoner per (claim x confound).
// H1. Tool-less by instruction: it decides only from the evidence in the prompt.
// Reason in prose, then emit the 4-field tail. No nested schema.
const SKEPTIC = (claim, confound) => `You audit ONE measured claim through ONE assigned confound. Findings here are numbers measured from a model, not text a model wrote. Default stance: assume the claim is an ARTIFACT of your confound until a committed number rules it out.

Decide only from the evidence below; do not call tools or read files. If you cannot name the deciding number, it is NEEDS_RUN, not RULED_OUT. Do not invent a control.

CONFOUND: ${confound}
CLAIM: ${claim.text}
EVIDENCE (the only data you may treat as established): ${claim.evidence}

Reason in prose, then output the JSON tail:
- status RULED_OUT only if you can name the committed number / file:line that makes your confound unable to explain the claim; put that number in next.
- status EXPLAINS if the evidence is consistent with your confound producing the claim.
- status NEEDS_RUN if the committed data cannot decide it; then next is ONE line, tagged "RUN: <existing script + flags>" if an existing control settles it, else "AUTHOR: <the control to write>".
- crux: the one fact your verdict hinges on. If the claim is itself a null ("X is uninformative / diffuse / at the noise floor"), a confound that EXPLAINS the statistic may BE the claim holding, not refuting it -- say which in the crux.
Number-gates (H2) that apply whatever your confound: a cited number decides only if a refuting outcome was possible and named before the run (a claim every outcome supports is framing -- treat as EXPLAINS); only with margin (exact-threshold clearance, a threshold authored after seeing the data, or selection from unreported attempts does not decide); and only within tested scope (stated scope exceeding the tested population, scale, or design strength is EXPLAINS on that ground).
Examples of next -- RULED_OUT: "same effect on held-out set, foo.json:42, n=30". NEEDS_RUN: "AUTHOR: mean/resample-ablation variant; only zero-ablation on record".`

const SKEPTIC_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['confound', 'status', 'crux', 'next'],
  properties: {
    confound: { type: 'string' },
    status: { type: 'string', enum: ['RULED_OUT', 'EXPLAINS', 'NEEDS_RUN'] },
    crux: { type: 'string' },
    next: { type: 'string', description: 'RULED_OUT: the committed number/file:line. NEEDS_RUN: one line tagged RUN: or AUTHOR:. EXPLAINS: brief why.' },
  },
}

// ---- ADJUDICATOR: dedup only. No score, no verdict label. The raw status + crux
// of every skeptic passes through to the human untouched (see the return); a
// rolled-up score masks the behaviour and mislabels null claims.
const ADJUDICATOR = (claim, verdicts, runnerResults) => `Dedup the controls the skeptics proposed for ONE claim. Do NOT score, do NOT emit a verdict label, do NOT reword cruxes (those pass through raw to the human).
${runnerResults ? `A runner produced new numbers: ${JSON.stringify(runnerResults)} -- drop any queued control it already settled.\n` : ''}VERDICTS: ${JSON.stringify(verdicts)}
From the NEEDS_RUN entries take the next-line: tagged "RUN:" -> run_queue, tagged "AUTHOR:" -> author_queue. Dedup near-identical controls: if several confounds need the same control (e.g. one matched-resample-ablation answers off-distribution AND self-repair AND noise-floor), emit it ONCE. Output ONLY the two distinct lists.`

const ADJ_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['run_queue', 'author_queue'],
  properties: {
    run_queue: { type: 'array', items: { type: 'string' }, description: 'distinct existing-control commands the runner can execute' },
    author_queue: { type: 'array', items: { type: 'string' }, description: 'distinct controls that do not exist yet; to author' },
  },
}

// ---- RUNNER: thin and reused. All discipline lives in agents/triage-runner.md;
// the per-call message is just the item. The control decides its own verdict;
// the runner returns it verbatim.
const RUNNER = (item) => `Run this control and report its result verbatim:\n${item}`

const RUNNER_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['matches', 'teardown_ok'],
  properties: {
    result: { type: 'string', description: "the control's own decision / numbers, verbatim" },
    matches: { type: 'string', enum: ['claim', 'confound', 'ambiguous', 'not_runnable'] },
    teardown_ok: { type: 'boolean' },
  },
}

// ---- AUTHOR: fresh, claim-blind agent that writes ONE new control from a spec.
// Discipline lives in agents/triage-author.md. It writes, it does not run.
const AUTHOR = (spec, path) => `Write a new measurement control from this spec, to ${path}.\nSPEC: ${spec}\nRead the existing scripts in this repo to match conventions and get the measurement correct.`

const AUTHOR_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['control_path', 'run_command', 'measures', 'decision_rule'],
  properties: {
    control_path: { type: 'string' },
    run_command: { type: 'string', description: 'existing launcher + this script, ready for the runner' },
    measures: { type: 'string' },
    decision_rule: { type: 'string', description: 'neutral threshold on the measured number; no claim/confound reference' },
  },
}

// ---- READER: fresh agent that GROUNDS a claim by reading its primary artifacts (H3). It reads the INPUT
// (items / labels / directional-training data a result was computed from) AND the OUTPUT (generations,
// per-token values, result JSONs), re-derives the claim's headline number from the raw artifact, and reports
// whether it reproduces. It reads in its own context (preserving the orchestrator's) and never runs or edits.
// Discipline lives in agents/triage-reader.md. This does not replace the orchestrator's own reading; it
// guarantees the primary data is read in full, for every claim.
const READER = (c) => `Ground this claim by reading the PRIMARY artifacts behind it and checking its numbers reproduce from the raw data. You read; you do not run or edit.\n\nCLAIM: ${c.text}\nEVIDENCE (cites the input data/labels and the output/result files): ${c.evidence}\n\nOpen the INPUTS (the items, labels, or directional-training data a result was computed from) AND the OUTPUTS (the generations, per-token values, result JSONs). Re-derive the claim's headline number(s) from the raw artifact. grounding="reproduces" if it matches; "diverges" if the raw data gives a different number; "unauditable" if the artifact needed to check it was never saved (that absence IS the finding); "absent" if the cited file is missing. Put both numbers in observed; one line in note.`

const READER_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['grounding', 'note'],
  properties: {
    artifacts_read: { type: 'array', items: { type: 'string' }, description: 'paths actually opened' },
    observed: { type: 'string', description: "the number(s) re-derived from the raw artifact vs the claim's" },
    grounding: { type: 'string', enum: ['reproduces', 'diverges', 'unauditable', 'absent'] },
    note: { type: 'string', description: 'one line: the mismatch, or which artifact was never saved' },
  },
}

// Orchestration. All agents inherit the session model; effort per role.

const A = (typeof args === 'string' ? JSON.parse(args) : args) || {}   // runtime may deliver args as a JSON string
const claims = A.claims || []
if (!claims.length) {
  log('Pass { claims: [{id, text, evidence, confounds?}], execute_runs?, rounds? }.')
  return { error: 'no claims provided' }
}
const execute = !!A.execute_runs
const rounds = A.rounds || 1
const ground = A.ground !== false              // H3: read the primary artifacts; ON by default (pass ground:false to skip)
// H4 surface: instruments are claims. Any instrument version without a committed validation
// artifact mechanically queues a validation control - the rule the doctrine had and the machinery lacked.
const instruments = A.instruments || []        // [{id, version, asserts, validation_artifact?}]
const instrument_queue = instruments
  .filter((i) => !i.validation_artifact)
  .map((i) => `AUTHOR: validation control for instrument ${i.id} v${i.version}: held-out ground truth for "${i.asserts}"`)
if (instrument_queue.length) log(`H4: ${instrument_queue.length} instrument(s) lack a validation artifact; queued.`)
log(`Triaging ${claims.length} claim(s); confounds=${DEFAULT_CONFOUNDS.length}; execute_runs=${execute}; ground=${ground}.`)

// H3 fan-out: one isolated reader per claim reads its cited inputs + outputs and re-derives the number.
// Returned as a separate leg (NOT fed to the tool-less skeptics, preserving H1).
const groundings = {}
if (ground) {
  const gs = await parallel(claims.map((c) => () =>
    agent(READER(c), { label: `read:${c.id}`, phase: 'Ground', agentType: 'triage-reader', model: 'opus', effort: 'low', schema: READER_SCHEMA })))
  claims.forEach((c, i) => { groundings[c.id] = gs[i] || null })
}

const triaged = await pipeline(
  claims,
  (c) => {                                       // skeptic fan-out (H1: fresh, tool-less, bounded slice)
    const confounds = (c.confounds && c.confounds.length) ? c.confounds : DEFAULT_CONFOUNDS
    const jobs = []
    for (const cf of confounds) for (let r = 0; r < rounds; r++)
      jobs.push(() => agent(SKEPTIC(c, cf), { label: `skeptic:${c.id}:${cf.slice(0, 20)}`, phase: 'Skeptics', model: 'sonnet', effort: 'medium', schema: SKEPTIC_SCHEMA }))
    return parallel(jobs).then((v) => ({ claim: c, verdicts: v.filter(Boolean) }))
  },
  (r) => agent(ADJUDICATOR(r.claim, r.verdicts, null), { label: `adjudicate:${r.claim.id}`, phase: 'Adjudicate', model: 'sonnet', effort: 'low', schema: ADJ_SCHEMA }).then((adj) => ({ ...r, adjudication: adj })),
)

let final = triaged
if (execute) {                                   // H2: only this step makes new evidence; gated + non-mutating
  final = await parallel(triaged.map((r) => () => {
    const queue = (r.adjudication && r.adjudication.run_queue) || []
    if (!queue.length) return Promise.resolve(r)
    return parallel(queue.map((item) => () => agent(RUNNER(item), { label: `run:${r.claim.id}`, phase: 'Verify', agentType: 'triage-runner', model: 'sonnet', effort: 'low', schema: RUNNER_SCHEMA })))
      .then((runs) => {
        const ran = runs.filter(Boolean)
        return agent(ADJUDICATOR(r.claim, r.verdicts, ran), { label: `re-adjudicate:${r.claim.id}`, phase: 'Adjudicate', model: 'sonnet', effort: 'low', schema: ADJ_SCHEMA }).then((adj) => ({ ...r, runner_results: ran, adjudication: adj }))
      })
  }))
}

let authored = null
if (A.author) {                                  // claim-blind author fan-out over the (deduped) author_queue
  authored = await parallel(final.map((r) => () => {
    const aq = (r.adjudication && r.adjudication.author_queue) || []
    if (!aq.length) return Promise.resolve({ id: r.claim.id, controls: [] })
    return parallel(aq.map((spec, idx) => () => agent(AUTHOR(spec, `controls/${r.claim.id}_${idx + 1}.py`), {
      label: `author:${r.claim.id}:${idx + 1}`, phase: 'Author', agentType: 'triage-author', model: 'opus', effort: 'medium', schema: AUTHOR_SCHEMA
    }))).then((cs) => ({ id: r.claim.id, controls: cs.filter(Boolean) }))
  }))
}

// H4 hoist: a control that several CLAIMS need is instrument-level (taints them all at once); surface it
// ONCE in shared_queue so it is checked at the instrument, not re-run per claim. Pure string dedup;
// per-claim queues are left intact for provenance.
const seenBy = {}
for (const r of final) {
  const adj = r.adjudication || {}
  for (const item of [...(adj.run_queue || []), ...(adj.author_queue || [])]) {
    const k = item.trim().toLowerCase()
    ;(seenBy[k] = seenBy[k] || { item, claims: [] }).claims.push(r.claim.id)
  }
}
const shared_queue = Object.values(seenBy).filter((e) => e.claims.length >= 2)

return {
  claims: final.map((r) => ({
    id: r.claim.id,
    grounding: groundings[r.claim.id] || null,  // H3: did the claim's numbers reproduce from the raw input+output artifacts
    findings: (r.verdicts || []).map((v) => ({ confound: v.confound, status: v.status, crux: v.crux })),  // raw ground truth; read these
    run_queue: (r.adjudication && r.adjudication.run_queue) || [],
    author_queue: (r.adjudication && r.adjudication.author_queue) || [],
    runner_results: r.runner_results || null,
  })),
  shared_queue,   // H4: controls needed by >= 2 claims; run these once, at the instrument
  instrument_queue,   // H4: instrument versions with no validation artifact; author these first
  authored,
}
