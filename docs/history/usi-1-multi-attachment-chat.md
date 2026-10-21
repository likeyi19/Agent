# USI-1 — Multi-attachment chat and turn binding

2026-10-15 (Asia/Shanghai). The composer now uploads real files and binds their
server references to the submitted turn. Bounded native-browser acceptance uses
scripted providers and genuine H5AD inspection. It does not qualify live-provider
language quality, biological validity, or the deferred scientific handoffs.
UA3.6.4a/b/c/d remain **PARTIAL**. The initial live-provider blocker is
resolved by the explicit floor-routing correction and bounded acceptance below.
The failed attempt remains recorded separately.

## Baseline and preservation

Entry: branch `main`, HEAD `4a9d8d11402350c5875abc0e187e4750c1bf3447`
(`Complete UA3.6.4b-d scientific integration simplification`), following
`bc9963840394a5df802857d7ce09794d3e990379`. The tracked working tree and index were
clean; `evals/` was already untracked. `git fetch origin` completed without merge
or reset; HEAD and `origin/main` agree, with zero ahead/behind commits. The remote
is `git@github.com:likeyi19/Agent.git`. The initial implementation phase performed no stage, commit, push, deployment,
resource download, or production configuration update. The separately authorized
operator configuration change is recorded in the final closeout attempt below.

Tests use isolated temporary workspaces. Browser evidence is outside Git at
`/home/likeyi/agent-usi1-9jq50uzy/` (**R**). No accepted scientific source,
checkpoint, result, authority, EpiZoo implementation, Registry, Planner, compiler,
or scientific owner changes. The existing 1,274 eval files are preserved; the
verification inventory records size and nanosecond mtime, not full dataset hashes.

## Existing boundaries and implementation

`H5ADUploadAdmission` already streams bounded chunks, hashes bytes, validates
paths and lengths, leases bounded capacity, fsyncs, publishes without overwrite,
and registers completed sources through `LocalResourceAdmission`. Existing
H5AD composition, typed scientific role selection, owner validation, and normal
interactive submission are reused. No second file store, catalog, role registry,
scientific dispatcher, provider prompt, or workflow engine is introduced.

The existing catalog record gains an optional `session_id`. Absent fields remain
omitted, preserving old record bytes and digests. A neutral `attachment` input
representation has no scientific operation. Session-scoped transfer identities
include the session, so equal client transfer tokens in different sessions cannot
collide. The completed `local-…` reference is issued by the existing server
registration owner; client filenames never become storage paths or trusted
references.

The same streaming handler now also serves
`PUT /api/v1/sessions/{session_id}/uploads/{upload_id}`. The session must exist.
`GET /api/v1/sessions/{session_id}/attachments` exposes only that session's safe
completed metadata. Scoped files are omitted from the legacy global picker.
Existing advanced upload and registered-input routes remain available.

`SubmitTurn.attachment_ids` is an ordered, unique, bounded list of up to 16 server
references. The interactive application resolves the records, checks session
scope, and captures immutable identity/digest/label snapshots in the existing
interaction submission. Both validation and worker submission carry the same
references; they participate in exact-retry identity. Scientific selection also
checks scope, preventing bypass through `resource_id` or `resource_ids`.
No-attachment submissions retain their historical submission shape. Scientific
Revisions and authorities gain no attachment state.

`TurnView.attachments` projects safe captured labels and references without paths
or hashes. `staged` and `offered_for_analysis` describe this particular submission,
not permanent file roles or scientific acceptance. For a successfully accepted
execution, the display derives `consumed` from terminal acceptance, which already
passed the resource owner’s `validate_result` guard. It does not revalidate science
or infer consumption from upload success. Actual eligibility and lineage remain
in existing accepted execution/evidence. Reads
and exact completed retries do not rescan historical uploaded bytes. Neutral
binding checks catalog identity; scientific consumers retain their existing
source-integrity and eligibility checks.

## User-visible behavior and format limits

The multiline composer keeps its layout and model selector. **Attach files** opens
a multiple-file picker. Repeated selections append; each file shows transfer
progress, registration outcome, retry and removal. Files stream directly from the
browser without whole-file JavaScript buffering. A failed or incomplete transfer
blocks Send, including programmatic form submission. Removing a file only removes
it from the draft; it does not delete shared or captured server bytes. Existing
advanced uploads remain under **Advanced inputs**.

Completed draft references are restored against the session catalog on refresh.
Interrupted transfers whose responses were not captured require explicit removal
and reselection; this is not resumable chunk upload. Captured attachments remain
in conversation history after refresh and server restart. Only references echoed by a captured turn clear from the draft; an HTTP 202
receipt alone does not clear them. History explicitly labels unresolved staged files,
even when a supported analysis succeeds.

| Representation through the new composer | USI-1 behavior |
| --- | --- |
| `.h5ad` candidate | Existing registered H5AD composition and scientific validation; genuine `inspect_scATAC` verified. Filename supplies only a candidate hint, never format validity, species, feature compatibility or model eligibility. |
| Multiple H5AD candidates | Existing ordered group/role mechanism; no first-match source choice. Existing scientific limits and ambiguity handling remain. |
| FASTQ, BAM, external fragments, FASTA, FAI, BED, TSV, TXT, CSV and compressed files | Bytes are uploaded, hashed, retained and associated with the turn as neutral attachments. No decompression, format claim, scientific role, or automatic execution follows. |
| Existing advanced registered inputs | Existing qualified workflows and explicit declarations remain available; no automatic conversion of neutral attachments into those roles. |

A new H5AD and an earlier selected scientific source are not silently substituted
for one another: the composer requests removal of the conflicting selection.
There is no extension-based scientific handler for TSV grouping or raw libraries.
H5AD candidates remain subject to existing clarification/failure mechanisms.
Inspection needs no EpiZoo checkpoint or reference resource; this unit does not
claim new EpiZoo inference or biological acceptance.

## Verification

The new API tests use the existing small sparse H5AD fixture and actual production
inspection with scripted interpretation/planning. They verify two persisted
references, genuine H5AD execution, staged/unresolved second-file state, scope
bypass rejection, byte/metadata conflicts, size/path protection, catalog tampering,
invalid references, corrupt H5AD rejection by its owner, restart, exact retries,
and stable source identity across staged and later scientific use. Legacy Session
and resource tests cover records without attachment metadata.

Final focused attachment/upload/static run: **148 passed in 3.60 seconds**.
Command: agent-environment Python, `PYTHONPATH=src`, `pytest -q` over
`tests/web/test_chat_attachments.py`, `test_static.py`, `test_upload_static.py`,
`test_multi_source_selection_static.py`, `test_upload_api.py`, and
`tests/application/test_uploads.py` and `test_interactive_schemas.py`. Nineteen tests directly cover the new backend
binding. An existing stale static expectation was corrected to include the
already-supported FASTA paired-index control.

FASTQ compatibility follow-up: **77 passed, 6 skipped in 9.09 seconds** over
`tests/web/test_fastq_execution.py`, `test_fastq_selection.py`, and
`tests/application/test_fastq_uploads.py`. The broader run exposed a collection
compatibility mistake in the new scope check; it was corrected to traverse
`RegisteredInputCollection.members` as well as independent input groups.

The single broader affected run (all `tests/web`, upload application tests, and
interactive schema tests) finished **691 passed, 19 failed, 6 skipped, 3 warnings
in 755.55 seconds**, before the collection correction. Fifteen failures were that
corrected collection regression, covered by the passing FASTQ follow-up above.
Two assertion failures reproduce against an isolated archive of committed HEAD:
`test_invalid_explicit_declaration_never_reaches_scientific_production[0.7-changes8]`
and `test_selection_transport_rejects_unbounded_ambiguous_or_path_inputs[changes1]`.
Two rich-UI scientific-turn checks exceeded their existing 20-second presentation
wait while science had succeeded; their isolated final-code recheck passes
**2 tests in 93.13 seconds**, with the same timeouts. No scientific production behavior or timeout policy was changed to make
these tests pass.

The directly affected application regression (`test_local_resources`,
`test_interactive_boundary`, `test_interactive_session_core`,
`test_interactive_rich`, `test_sessions`) finished **291 passed, 1 failed in 23.25
seconds**. Its annotation-rationale detail failure reproduces at committed HEAD
(**1 failed, 1 passed, 20 deselected in 1.86 seconds**). The baseline check of the
two other assertion failures plus one FASTQ case gives **2 failed, 1 passed,
3 warnings in 12.05 seconds**. These pre-existing failures remain outside this
bounded implementation; no green full regression or broad product PASS is claimed.
Logs are preserved under R with `usi1-` filenames. The isolated rich-UI rerun
is `R/usi1-rich-rerun.log`. The bounded USI-1 checks pass; the wider regression
remains non-green because of the three reproduced baseline assertions.

Native Firefox, through real DOM controls and normal HTTP/application paths,
passes 11 recorded checks: pending-upload progress/submission blocking; one-file
upload; incremental multi-selection; independent removal; draft refresh; visible
size failure; retry/removal recovery; successful transient-error retry; multiline
H5AD execution with both references and explicit staged status; evidence/report
and report download; and fresh-server history restoration. A test-only middleware
adds transfer delay and one transient capacity error. Scientific execution is
real `inspect_scATAC`, once per acceptance session, on the existing 2 × 3 sparse
fixture. Selected profile `beta` constructs `model/beta` through the normal path.
Restart uses a provider factory that raises if constructed, and executes zero
science. `R/browser.log`, `restart-browser.log`, `native-final.log`, `browser.png`,
`session.json`, `server.py`, `browser.py`, and `run.py` record/reproduce this scope.
The initial `/tmp` Firefox launch failed because the installed browser could not
access its profile there; the successful fixture uses the home-directory location.

## Initial model configuration inspection

The actual external operator file is
`/home/likeyi/program/agent-web-config/operator-web.json`, SHA-256
`50ddd7846dd276f11465f98040b285218fb459acbe480bb8aa90350f2283e880`.
It contains only enabled/default `groq-gpt-oss-120b`, provider `groq`, model
`openai/gpt-oss-120b`, label **Groq GPT-OSS 120B**. Loading its actual profile
configuration into an isolated facade returns that exact `/api/v1/models` choice
without constructing a provider. No production server was running at inspection.
`R/configured-model-catalog.json` records the result. Uploads are already configured with a 268,435,456-byte per-file limit and two
concurrent transfers; those settings are unchanged. The browser builds its picker
from this endpoint and submits the selected profile normally. OpenRouter
`openai/gpt-5.6-sol` is **not configured in this operator deployment**. No profile
or credential was invented or changed during that initial phase. The later
authorized profile addition and its live-provider limitation are recorded below.

## Deferred work

USI-2 should resolve task-specific roles for these existing identities, preserve
explicit clarification continuation, and bind source-qualified FASTQ/BAM library
context. It should not re-upload bytes or turn file types into permanent roles.
USI-3–6 retain qualified resource resolution, raw preprocessing handoff, BED and
cell-selection requirements, MAESTRO grouping/resources, cross-species EpiZoo,
post-training and ordinary analysis integration. USI-7 retains further deployment and UX work; the bounded operator-profile
addition was explicitly brought into this USI-1 closeout task. Authentication remains the existing single-lab
boundary; session scope does not introduce a new account authentication system.

The new composer does not yet offer a general historical-attachment reuse picker,
resumable transfers, or neutral-file scientific adoption. File removal is draft
removal, and existing storage capacity/retention limits remain unchanged.

## Authorized change inventory

- `src/agent/application/local_resources.py`: additive neutral representation and optional session scope, preserving legacy serialization.
- `src/agent/application/uploads.py`: shared scoped streaming/publication and neutral-file storage.
- `src/agent/application/interactive.py`: exact attachment binding, scope enforcement, retry identity and accepted-use projection.
- `src/agent/application/interactive_schemas.py`: bounded immutable attachment display contract.
- `src/agent/web/app.py`: scoped upload/list routes and turn reference transport.
- `src/agent/web/static/app.js`, `index.html`, `styles.css`: composer interaction, progress/errors, durable-capture-aware draft clearing, restoration and history.
- `tests/web/test_chat_attachments.py`: nineteen backend integration cases.
- `tests/application/test_interactive_schemas.py`: additive view-field expectation.
- `tests/web/test_upload_static.py`: paired FASTA/fragments index expectation.
- `docs/usi-1-multi-attachment-chat.md`: this record.

The original attachment scope is these twelve files. The final routing correction
also includes `src/agent/providers/openrouter_planning.py` and
`tests/unit/providers/test_openrouter_planning.py`: fourteen files total. The external operator
configuration and all acceptance fixtures, browser profiles and observations stay
outside Git. No UA3.6.4 status or deferred scientific integration is promoted.

## Initial closeout attempt — blocked (2026-10-15)

Historical failed attempt; superseded by the routing resolution below.

The actual entry baseline was rechecked: `main` at
`4a9d8d11402350c5875abc0e187e4750c1bf3447`, matching `origin/main` after a
successful fetch, with an empty index. All twelve prior USI-1 file hashes matched
`R/final-files.json`; the only subsequent repository edit in this closeout is
this record. The production implementation needed no provider or routing changes.

**Configuration and default policy.** The existing strict Web configuration and
built-in `openrouter` factory support the requested model. The authorized external
`operator-web.json` now also contains enabled, structured-output profile
`openrouter-gpt-5-6-sol`, provider `openrouter`, exact model
`openai/gpt-5.6-sol`, label **OpenRouter GPT-5.6 Sol**, timeout 60 seconds.
All original Groq and non-model settings are preserved. Groq remains the default
and selectable: the accepted model-layer closeout explicitly distinguishes
current development-primary qualification from operator-owned Web defaults.
Both credential variables were present after activating the existing `agent`
environment; no secret value was printed, copied into fixtures, or changed.

Before replacement, the complete original configuration was backed up to
`R/closeout/operator-web.before.json`, mode **0600**, inside a mode **0700**
directory outside Git. The original digest is recorded above. The new SHA-256 is
`e626de8d88ed3c223d076c62d17b67f45861c81509b2cbdddc55b947c523279f`.
`R/closeout/configuration-change.json` records only safe configuration metadata.

**Actual API and native browser.** An isolated Web server loaded that exact
external configuration, changing only its workspace/upload roots to an external
acceptance directory. It used the existing default provider factory and SDK;
scientific execution was guarded against for this availability smoke. The actual
`/api/v1/models` returned both exact profiles and labels, with Groq marked default.
Native Firefox rendered both server-provided options, selected GPT-5.6 Sol,
submitted “Please clarify the intended analysis.” through the normal composer,
and persisted `profile_id=openrouter-gpt-5-6-sol` in the resulting turn. Refresh
preserved the profile and turn. No Run or Revision was created. No production
server was restarted or production Session mutated.

**Actual invocation failed.** There was exactly **one completion request**, to
`https://openrouter.ai/api/v1/`, with model `openai/gpt-5.6-sol`, 19,751 prompt
bytes, 4,096 maximum completion tokens, 60-second timeout, zero SDK retries,
strict structured output, `require_parameters=true`, `allow_fallbacks=false`,
and the existing $2/$10 per-million prompt/completion price caps and zero request
fee cap. Response healing, compression and web plugins remained disabled. The
SDK returned **NotFoundError / HTTP 404**, without a model completion. There was
no retry, Groq substitution, semantic repair or scientific execution.

The application rendered its existing `clarify / invalid_decision` response.
This is **not** a provider-authored clarification or a successful invocation.
`browser.log` records the passing UI/transport assertions; its PASS line does
not establish provider success. The enclosing `run_live.py` check exited **1**
because no `returned_model` existed, preserving the actual failed outcome.
`R/closeout/provider-observation.json`, `live-result.log`, `models.json`,
`session.json`, `browser-timeline.jsonl` and `model-selection.png` provide the
separate provider and browser observations.

Read-only discovery then succeeded: the catalog contained the exact model among
462 entries, with advertised base pricing $4/$20 per million tokens. The endpoint
listing contained seven entries. Six reported status `0` and prices at least
$4/$20; the sole entry priced within $2/$10 reported status `-2`. These observations
point to upstream routing availability under the preserved cost policy, rather
than a missing profile, absent credential variable, local factory failure or
model-output error. The original 404 response body was not retained, so the exact
server rejection reason is not asserted. Discovery made no additional completion
requests. Evidence: `model-discovery.json` and `endpoint-discovery-http.json`.
An initial SDK endpoint-discovery attempt failed locally with `ValueError`,
recorded separately in `endpoint-discovery.json`; ordinary bounded HTTP GET
provided the endpoint evidence. Existing price safeguards were not relaxed.

**Final bounded regression.** **151 passed, 6 skipped, 1 deselected in 11.70
seconds**, exit 0. Command: agent-environment Python, `PYTHONPATH=src`,
`python -B -m pytest -q` over `tests/web/test_chat_attachments.py`,
`test_fastq_execution.py`, `test_fastq_selection.py`, `test_model_profiles.py`,
`test_multi_source_selection_transport.py`, `test_static.py`,
`test_upload_static.py`, and `tests/application/test_fastq_uploads.py`, with
`--deselect='tests/web/test_multi_source_selection_transport.py::test_selection_transport_rejects_unbounded_ambiguous_or_path_inputs[changes1]'`.
That one exclusion is the already reproduced baseline assertion documented above;
it is not classified as a pass. The other two known baseline assertions remain
unchanged and were not rerun. Collection `.members`, ordered groups and singles,
session scope, existing FASTQ selection, corrupt H5AD rejection, real H5AD
inspection, and normal provider selection/failure routing are covered. Six skips
remain existing gated FASTQ cases. No broad scientific suite or F3 qualification
rerun was performed. Log: `R/closeout/final-focused.log`.

**Preservation and Git outcome.** All 1,274 pre-existing eval files and all 46
files in the production Web workspace retain their recorded size and nanosecond
mtime. The eleven implementation/test files retain their prior accepted hashes;
only this documentation changes during final closeout. These are bounded metadata
and source-hash checks, not full scientific dataset rehashing. Accepted scientific
sources, results, authorities and checkpoints were not used or changed by the
isolated smoke. Credentials and the external configuration backup remain outside
Git. `git diff --check` passes. The index remains empty; ten tracked implementation
files are modified, this record and the attachment test remain new/untracked,
and `evals/` remains pre-existing/untracked.

**Outcome of that attempt: USI-1 was not formally closed.** The user's commit/push authorization is
conditional on genuine GPT-5.6 Sol availability. The live 404 leaves that
prerequisite unsatisfied, so no files were staged, no completion commit was
created, and no push was attempted. HEAD, local main, local origin/main and a
fresh remote-main read still agree on the baseline above. Completion requires a
successful bounded invocation under an available approved routing/cost policy;
a model-list/UI pass alone is insufficient. UA3.6.4a/b/c/d remain **PARTIAL**.
USI-2 was not started.

## Final routing resolution and acceptance (2026-10-15)

**Baseline.** The follow-up started on `main` at the same `4a9d8d1` baseline,
with an empty index and the twelve USI-1 files pending. Fetch completed and
`HEAD...origin/main` remained 0/0. Evidence for this follow-up is
`R/routing/` (**Q**). No historical acceptance file was overwritten.

**Confirmed rejection and current eligibility.** One authorized diagnostic
completion repeated the ordinary Web interpreter request with the original exact
model, strict schema, 4,096-token cap, 60-second timeout, zero SDK retries,
`require_parameters=true`, `allow_fallbacks=false`, and $2/$10/$0 price filters.
Its sanitized HTTP 404 message was: “No endpoints found that satisfy the max
price for this request.” This confirms price filtering for the reproduced
failure; the earlier unretained response body remains unknown.

Fresh read-only discovery showed seven endpoints. The $2/$10 entry is
`openai/flex`, now status 0, with `response_format` and `structured_outputs`.
Standard OpenAI/Azure entries start at $4/$20; regional Azure/Bedrock entries
are $4.40/$22 and OpenAI fast is $8/$40 per million tokens. All entries advertised
`discount=0`. The account-visible model list includes the exact model, and the
key endpoint succeeded with non-free, non-expired flags. Full private account
policy settings were neither exposed nor changed. The successful request below
establishes eligibility for this account and execution location at that time.

The [service-tier documentation](https://openrouter.ai/docs/guides/features/service-tiers)
explains that flex endpoints require explicit opt-in. Ordinary price sorting
alone does not admit them. The [floor variant](https://openrouter.ai/docs/guides/routing/model-variants/floor)
admits flex capacity for the same model and sorts by price, with greater latency
and availability variability. This explains why the standard request had no route
within its caps even after flex became available. Current
[model-page pricing](https://openrouter.ai/openai/gpt-5.6-sol) is $4/$20.
Cached search results advertised an earlier 50% promotion; the freshly opened
[discount collection](https://openrouter.ai/collections/discounted-models/)
no longer listed GPT-5.6 Sol and says promotions may end without notice.
No promotional discount is assumed. Published `max_price` documentation does
not specify how temporary promotional discounts interact with filtering; this
resolution uses explicit advertised flex pricing and observed billing instead.

**Minimal correction.** The external profile now explicitly requests
`openai/gpt-5.6-sol:floor` and displays **OpenRouter GPT-5.6 Sol (floor / flex
eligible)**. Profile ID stays `openrouter-gpt-5-6-sol`; Groq settings and default
are unchanged. A second mode-0600 backup is
`Q/operator-web.before-floor.json`. New operator SHA-256:
`f59312491c2719ae93bc3366724335550a713e5c76cf6bd84ddbc01a0558477e`.
No configuration schema or provider abstraction changes. The adapter now accepts
the exact base model for an explicitly requested, unstacked `:floor` suffix
**only when routing metadata echoes the full requested slug**. Existing
pipeline/fallback checks still apply. Unknown substitutions, missing identity
proof, unrelated variants and additional attempts fail closed. The real response
below confirms that this narrowly scoped identity handling is necessary.
Scientific code, prompts, typed decision parsing, authority and lineage are unchanged.

**Real invocation and browser acceptance.** Exactly one corrected completion
was made through the real default factory and native Firefox composer on an
isolated server loading the updated operator file. It returned
`model=openai/gpt-5.6-sol`, `service_tier=flex`, provider OpenAI,
`requested=openai/gpt-5.6-sol:floor`, strategy `direct`, attempt 1 and
`finish_reason=stop`, with one available endpoint. The genuine strict JSON was:

```json
{"turn_schema_version":1,"decision":{"kind":"answer_guidance","targets":[],"candidate":null}}
```

The existing parser accepts this as `Answer`; with no evidence
in the empty Session, normal admission returns `clarify / unavailable_context`.
This is distinct from the earlier provider failure's `invalid_decision`. It
qualifies transport, identity, strict JSON and typed-contract usability, not the
model's intent quality or a scientific answer. Both dropdown choices, explicit
floor label, selected-profile submission, persisted profile and refresh pass.
No science, Runs, Revisions, alternative-model fallback or retry occurred.
`Q/acceptance/live-result.log` exits 0. Raw completion, safe usage/routing
observations, Session view, timeline and screenshot are beside it. The diagnostic
failure is separately preserved under `Q/diagnostic/`.

**Cost accounting.** The successful request used 5,706 prompt and 169 completion
tokens (137 reasoning tokens), costing **$0.0159535** as reported by OpenRouter.
Its 5,703 cache-write tokens are billed at the separately advertised $2.50/M;
three uncached tokens use $2/M, and completion uses $10/M. These reconstruct the
reported charge exactly. The unchanged `max_price` prompt/completion/request
fields are not an all-inclusive cache-write-price ceiling. The initial
prompt-price-only reservation of $0.114806 omitted that premium; the corrected
cache-aware upper estimate for the same 36,923-token conservative input bound
and 4,096 output tokens is $0.1332675, still below the recorded $0.25 per-request
budget. There were two requests total this follow-up (one diagnostic 404, one
successful completion), no repeated exploratory calls. No cost limit was raised
or removed. No output or cost usage was returned for the failed request; zero
billing for that attempt is not inferred. `Q/acceptance-assessment.json` preserves
these distinctions. Higher-context tier prices can exceed the caps; long-context
availability and continuous flex availability are not qualified by this smoke.

**Verification and delivery scope.** Focused provider/factory/Web tests pass
**129 tests in 1.98 seconds**, including eleven new floor identity cases. Command:
`PYTHONPATH=src python -B -m pytest -q tests/unit/providers/test_openrouter_planning.py tests/unit/providers/test_model_registry.py tests/web/test_model_profiles.py`.
The earlier **151 passed, 6 skipped, 1 known baseline assertion deselected**
attachment regression remains applicable; unchanged attachment code was not
retested broadly. The three historical baseline assertions remain unresolved.
The two routing files above join the original twelve-file commit scope; external
configuration, credentials, browser workspaces, scientific artifacts and `evals/`
are excluded. The scoped source diff and documentation were reviewed and
`git diff --check` passes. Preservation and final Git-delivery receipts are stored
under Q and reported with the completion commit hash in the final response.

The routing blocker is resolved within the existing price-filter policy. Bounded
USI-1 implementation and acceptance are complete; formal delivery requires the
authorized normal commit/push and matching local/remote refs. No production Web
process was restarted; a deployment must load the updated external configuration
and corrected adapter. Flex is an explicitly selected capacity tier of the
qualified base model, not a new F3 stability qualification. UA3.6.4a/b/c/d remain
**PARTIAL**; USI-2–7 and UA3.6.5 scientific handoffs remain deferred. USI-2 was not started.
