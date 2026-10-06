# Post-M18.1 — Typed Turn Admission Simplification

Baseline: `c01a076430e896cbc3a95389c2fe932c0c9f6d2d`
(`Complete M18 interactive application`). M18 remains closed and accepted within
its recorded scope. This follow-up does not revise its historical acceptance.

## Problem and resulting behavior

The interpreter already returns a typed Execute, scientific Answer or guidance
Answer. Three generic admission checks then independently classified the
utterance with the same bounded command-verb predicate. An Execute for a request
beginning “Plan and execute” could be rejected because `plan` was absent from
that predicate; a typed scientific or guidance Answer could be rejected merely
because the utterance began with a recognized command verb.

This change removes only those three generic linguistic vetoes:

- `dialogue_execution.admit` no longer requires its typed Execute utterance to
  match `is_execution_command`.
- `scientific_dialogue.admit` no longer rejects a typed scientific Answer when
  its utterance matches that predicate.
- `scientific_guidance.admit` no longer rejects a typed guidance Answer when its
  utterance matches that predicate.

The interpreter's existing typed decision owns generic language intent. Its
prompts, wire tags, decision schemas and normalization remain unchanged. No verb
list expansion, replacement classifier, workflow heuristic or automatic planning
fallback is introduced.

This deliberately changes the guarantee: a wrongly supplied typed Execute for a
question can proceed to the existing Planner. Code no longer vetoes that generic
language misclassification. Likewise, a wrongly supplied typed Answer is not
reclassified as execution by this predicate. Passing mechanical and scientific
contracts does not prove that the provider correctly understood user intent.

## Retained boundaries

Generic Execute admission still checks the current base, absent parameter delta
and registered tool target. The existing Planner, semantic compiler, preflight,
actual-plan output selection, sequential runtime, scientific owners, authority
and evidence validation remain responsible for executable behavior. Answer
admission retains exact revision/output/subject references and accepted-evidence
grounding. Structured inputs retain their existing normalization, immutable
submission identity and binding path.

`is_execution_command` remains available for M17.2's narrower explicit
guidance-candidate selection evidence. Exact complete current-turn command
clauses, conditional/negated-command rejection, candidate identity,
applicability and generation checks remain unchanged. The existing generic
candidate-bypass guard still rejects attempts to route a referenced guidance
candidate through generic Execute instead of the candidate-selection branch.
The separate guidance candidate-reference checks remain intact.

No scientific algorithm, tool contract, Registry entry, EpiZoo code, Session
persistence schema, HTTP endpoint, browser flow or recovery behavior changes.
Safe current-turn structured-input semantic descriptors and browser uploads
remain deferred; this change does not add either capability.

## Validation and acceptance

Focused checks pass, using scripted providers and temporary fixtures:

- New typed-admission tests: **37 passed in 1.86 seconds**. Eight utterances,
  including the complete qualified R2 and a deliberately misclassified result
  question, pass both existing Execute wire forms up to a stop before planning.
  The tests retain invalid-contract/input and stale-generation rejection, exact
  bindings, generic candidate-bypass rejection and M17.2 selection admission.
- Scientific dialogue, guidance and provider-contract tests: **93 passed in
  12.51 seconds**. Command-like answer/guidance wording retains exact evidence
  and subject admission; unavailable evidence and invalid references still fail.
- Turn decisions, interactive Session core/boundary and candidate selection:
  **89 passed in 17.85 seconds**.
- Existing missing-request-input and unavailable-v4-selector checks: **2 passed
  in 1.52 seconds**. Required input binding remains owned by the existing
  planning contracts, not inferred during generic turn admission.

The focused checks total **221 passed**. After source and tests stabilized, one
relevant regression across `tests/application` and the six existing dialogue,
intent and guidance authority suites passes **492 tests in 584.66 seconds**,
exit 0, with no failures, skips or warnings. The full repository regression is
not run for this bounded change.

A separate temporary-Session reproduction supplies the complete qualified R2
with a scripted `execute_plan / inspect_scATAC` decision. Normal interpretation
and admission preserve the exact operator input binding and reach the execution
stop. It records one scripted interpreter call, zero live provider calls, zero
forbidden planning/scientific attempts, generation zero and no Runs/Revisions.
This qualifies the deterministic correction, not live-provider consistency or
scientific correctness on new data. The earlier three-call live qualification
demonstrated the old command-gate rejection; no live call is repeated here.

`git diff --check` passes. The M17.2 predicate body and candidate-selection,
turn-decision prompt/schema, orchestration, provider, Session-store, interactive
and EpiZoo source remain unchanged. Content manifests for the entire existing
`evals/` directory, real web workspace and operator configuration match their
pre-change values; all 125 pre-existing nonignored untracked eval files remain
untouched. Real user Sessions, the running service and preserved acceptance
sources are not changed. No safe input descriptor or upload is implemented.

Post-M18.1 software acceptance passes and implementation is ready for review.
Changes remain unstaged. No commit or push is performed or authorized by this
record. The running server is not restarted; live browser use of the changed
source is outside this qualification.
