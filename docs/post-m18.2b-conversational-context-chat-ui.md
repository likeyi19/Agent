# Post-M18.2b — Conversational Context Projection & Chat-First UI

Baseline: `6a8a477774c1e33b9c97669a6e59b490e97f1ea0`
(`Fix scientific literal validation`). Local `main`, local `origin/main` and
remote `main` matched before editing. The tracked tree was clean; the 125
pre-existing nonignored untracked eval files were preserved separately from
the two ignored eval files.

## Gap and ownership

The accepted Post-M18.2a correction remains unchanged. Its demonstrated failure
was the literal validator treating `zero` inside `non‑zero` as an unbound number.
Claim/support selection stays with the LLM; accepted values, Revision/output
identity, provenance and concrete evidence bindings stay with Agent. This task
does not repeat the historical diagnostic or loosen evidence validation.

The interpreter previously received individual retained output handles without
their shared accepted step identity or current-turn supplied-input availability.
Several handles can represent one inspection result; their count alone cannot
establish several independent results. The prior-discussion focus also cannot
stand for a newly created result before any scientific question has been answered.
The separate live operational-route failure establishes provider variability,
not proof that either projection gap caused every historical failure.

## One derived interpreter projection

The existing `dialogue` projection now contains:

- Registry-admitted supplied field names, shallow JSON types and registered
  consumer expectations, plus presence/omission counts. No supplied value,
  nested key, path, resource qualification or readiness conclusion is exposed.
  `source_complete` means required request-source keys are present; tools still
  validate their contents. Unknown field names are omitted.
- Opaque `gN` groups for cooutputs with the exact same Revision, run, step and
  accepted-step digest. Evidence must agree completely after removing only
  locator name/output-key attribution. The lexicographically smallest
  `(name, output_key)` supplies canonical attribution within an equivalent
  group; this never selects among distinct scientific steps.
- `@current_result`, `@most_recently_created` and `@previous_turn_result` scopes.
  Current uses the captured active Revision. Creation scopes exclude retained
  old-run outputs. Most-recent creation stays distinct from navigation. Previous
  turn means the immediate preceding interaction's accepted created result or
  captured completed scientific target/comparison, including its subject; it
  never scans backward for a convenient result.

The LLM interprets natural language and selects an offered handle using the
existing scientific-answer target shape. Agent derives exact persisted
Revision/output/digest bindings and checks accepted evidence. Multiple groups
or subjects remain ambiguous. Unknown, unavailable or stale references fail
closed. Legacy output/focus targets retain their existing contracts.

Only bounded capture metadata is added to the existing interaction snapshot;
there is no new Session/Revision scientific state or authority. The capture pins
the revision prefix and previous-turn availability so later completion cannot
promote a pending result or discussion. Legacy snapshots remain readable.
Read-only context, answers and retries do not invoke scientific production or
owner reconstruction. Existing evidence readers and answer generation are reused.

## Chat-first static browser

One centered column contains chronological user/Agent bubbles, the multiline
composer, then scientific details. Result actions open/focus the existing view.
Compact result labels and output summaries remain visible; accepted results,
evidence, reports/figures, supported detail and revision history use progressive
disclosure. Full identities remain in expandable metadata and tooltips.
Guidance cards remain attached to their persisted Agent response.

Session creation/reopening, per-turn model and operator-input choices, polling,
cancellation, retries, evidence/artifacts, M17.2 guidance selection, navigation,
continuation, child branching and refresh/restart use their existing handlers
and APIs. Browser convenience storage remains the same four non-scientific
fields. No frontend framework or build tool is introduced.

## Deferred behavior and limits

Execute-and-explain remains deferred: the typed execution decision selects a
registered execution request, and its completion returns an execution receipt.
Output retention does not encode a second scientific-answer intent. A future
explicit completion contract can invoke the existing evidence-bound answer path
after acceptance; this task adds no automatic second answer call.

Uploads remain deferred. Future bounded server staging should produce a
server-owned admitted resource, then use the same structured input/projection
and existing `AgentRequest.inputs` boundary. Browsers must not provide arbitrary
server paths. Safe descriptors establish availability, not scientific format.
Existing context bounds still apply; broad biology and universal live-provider
intent accuracy are not qualified by this work.

Planner, semantic compiler, preflight/runtime, Registry, scientific tools,
authority/evidence owners, EpiZoo and M17.2 are unchanged. This record authorizes
no commit or push.

## Focused validation and provider limitation

Development checks pass: **182 dialogue/evidence/provider/guidance/decision
tests**, **93 web tests**, and **30 context/static tests** after the final capture
checks. The latter includes 19 new application cases and three new static cases.
They cover safe/partial input descriptions, canonical cooutput equivalence and
permutation, distinct-step ambiguity, navigation versus creation, retained
outputs, subject/comparison handling, pending completion and failure,
unchanged-generation capture races, legacy four-key persisted snapshots, stale
references, stored-response retries and progressive disclosure. Provider and
scientific work are prohibited in read-only fixture paths. The retained-output
metadata fixture isolates its missing executable-authority publication gate;
normal evidence/admission and the separate authority owner suites remain intact.

The single stable relevant regression across `tests/application`, `tests/web`
and the six existing dialogue/intent/guidance authority suites passes **636
tests in 717.33 seconds**, exit 0. All `RUN_*` gates are disabled, with no
exclusions. All **404 frozen source/test/static/resource files** remain unchanged
during the run. The full repository suite is not run.

A read-only projection of the preserved real inspection at Revision
`7d439aee77b5748a9964afe19b3f0050ec1941569b8f2bd26e6530912e997d7d`
contains 12 cooutputs in one `g0` group, canonically attributed to `density`.
Current and most-recently-created scopes are available; previous-turn is
unavailable because the immediate prior interaction produced no result. The
7,885-byte projection exposes neither source paths nor scientific values.
No provider or science runs for this check.

The native Firefox fresh-Session flow selects PBMC and submits exactly
`Inspect the selected scATAC-seq dataset.` through ordinary form/HTTP/application
admission with `groq-gpt-oss-120b`. Its sole live interpreter/SDK call returns
`clarify / ambiguous_subject`. The actual prompt contains `input_path` presence,
string type, the registered inspection description and complete request-source
keys, with no supplied value/path. All result scopes are correctly unavailable
in the empty Session. The interpreter schema offers registered execution.
No Planner, compiler, preflight, scientific tool or verification is invoked.

This is a separate provider intent result despite available safe input context.
The requested successful inspection and both follow-ups are **not qualified**.
`What did you find?` and `Summarize the result you just created.` are not submitted
after the first turn creates no accepted result. No semantic retry, phrase
matching, scripted replacement in the live path or deterministic route repair
follows. The fresh browser's compact controls, single column at 1600/1024 widths,
conversation/composer order, refresh and fresh-server restoration pass with no
additional work or Session mutation.

Two earlier diagnostic setup attempts stopped before any provider/science call:
an optional empty `interactions` field assumption and a progress-helper argument
collision. Both external checks are corrected and independently exercised before
the actual call. They leave two empty Sessions; the actual flow appends one
clarification to a separate new Session. No pre-existing Session is modified.

The original 125 nonignored untracked eval files and two ignored eval files
remain unchanged: 127 files, 499,991,240 bytes, aggregate manifest SHA-256
`2f79c4ee68a75a2a5519ea8c4c7634c8a75cc09ff88fabaddd23b6d0277d357a`.
All 31 pre-existing web-workspace files and the operator configuration retain
their original content manifests. The PBMC H5AD's bytes and mode remain unchanged.
Credentials come from the existing Conda `agent` environment, with only presence
checked; no environment configuration or plaintext credential file is created.
Credentials are excluded from the browser environment and diagnostic records.

## Native rich-UI acceptance and review status

A separate strictly read-only native Firefox check reopens preserved Session
`528a944d-2b4c-48f6-8dd1-2efbe81e2f36` and its accepted inspection. All **12
checks pass**: compact/closed default details, one column at 1600/1024 widths,
result-button focus/open, accepted evidence, output selection, accepted report
delivery/digest, report privacy, revision controls/browsing, refresh and a fresh
server. Chat and details screenshots confirm the continuous conversation,
attached composer and scientific details below it. Historical responses remain
exact, including their previously recorded failures; reads do not regenerate them.

This check adds zero provider constructions, logical/SDK calls, inspections,
scientific execution or submissions. Every workspace file and Session JSON is
unchanged. All owned servers/browser processes stop, ports close and temporary
Firefox profiles are removed. Cancellation, guidance execution and branching
remain covered by the relevant software suites and prior M18 acceptance; this
read-only browser check does not execute those actions.

Review scope is four application files (`scientific_dialogue.py`,
`turn_context.py`, `turn_decisions.py`, `turns.py`), the three existing static
frontend files, `tests/application/test_interaction_context.py`,
`tests/web/test_static.py`, README, AGENTS and this record. Diagnostic scripts,
logs, projections and screenshots stay outside Git under
`/tmp/agent-post-m18.2b.x46kavlm/`; the live record is in `browser-final/` and
the successful rich-UI record is in `browser-read-only-smoke-corrected/`.

Post-M18.2b is **ready for software/UI review**, with the live-provider limitation
above. The requested live scientific-answer chain has not passed. No additional
scientific capability, authority, language guessing, execute-and-explain, upload,
framework, full repository regression, commit or push is introduced.

Final `git diff --check` passes. All changes remain unstaged. Local `main`, local
`origin/main` and remote `main` still match the baseline SHA. A final scan confirms
the configured credential is absent from all owned diagnostic artifacts.
