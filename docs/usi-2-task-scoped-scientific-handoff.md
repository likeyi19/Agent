# USI-2 — Task-scoped attachment roles and scientific handoff

Current status: **USI-2 is COMPLETE within its defined implementation and bounded
synthetic scientific/software acceptance scope**; final review is complete. See the
[USI-2C closeout](#usi-2c--bam-conversational-handoff-and-usi-2-integration-closeout)
for BAM handoff, shared FASTQ/BAM coordination, validation and limitations. One
previously reproduced baseline transport assertion remains failing; no green
full-repository claim is made. UA3.6.4a/b/c/d remain PARTIAL. The following
checkpoints are retained as historical evidence.

## USI-2A checkpoint (historical)

2026-10-15 (Asia/Shanghai). **PARTIAL; not ready for USI-2 closeout.**
The shared attachment-to-role bridge and a genuine inspection continuation work.
The requested FASTQ/BAM source-bound context generation and incomplete FASTQ
library continuation are **not implemented**. Passing inspection, existing raw
owner tests, or live Interpreter observations does not satisfy B.1. The remaining
B.1 implementation belongs to USI-2, not a reclassification as USI-3 work.
UA3.6.4a/b/c/d remain PARTIAL.

## Baseline and preservation

Starting branch: `main`, HEAD and local `origin/main`:
`36a1b4921a7db935027cf53b90cd478a90ba224e`
(`Complete USI-1 multi-attachment chat and model integration`). The tracked
working tree and index were clean; `evals/` was already untracked.
`git ls-remote origin refs/heads/main` independently returned the same commit.
No reset, checkout, staging, commit, push, or production configuration update.

Evidence root **R** is `/home/likeyi/agent-usi2-zyuhds5p/`, outside Git and
historical acceptance directories. `R/baseline.json` records tracked-file
SHA-256 hashes, the original index digest, and size/mtime_ns for all **1,274**
pre-existing eval files. Eval preservation is a metadata inventory, not a claim
of freshly hashing all biological datasets. New test workspaces, provider
observations, browser profile, fixture files and results are under R.

The operator configuration remains at SHA-256
`f59312491c2719ae93bc3366724335550a713e5c76cf6bd84ddbc01a0558477e`.
No historical source, checkpoint, Session, Run, Revision or authority was
intentionally opened for mutation. No scientific download, GPU work, model
checkpoint loading, full matrix densification or production upload-limit change.

## Existing machinery and confirmed gaps

Reused unchanged: streaming scoped upload and server-issued references;
`SubmitTurn.attachment_ids`; immutable local registration; Registry consumer
contracts and request aliases; `request_roles` / `answer_roles`; ordinary
`Interaction.prerequisite`; generation, predecessor and exact-retry admission;
Planner/compiler/preflight/executor; accepted evidence, reports and delivery.
Existing resource and parameter continuation remain their respective owners.

The USI-1 facade captured neutral attachments for presentation, but did not offer
them through `RegisteredInputGroup`. That group required every selected file to
receive a distinct scientific role. This is correct for an explicit scientific
selection but inappropriate for a chat attachment list containing unrelated files.
Also, role questions required at least two consumers, preventing a question about
which of several attachments supplies a single dataset.

The raw gaps are distinct and remain open:

1. Neutral uploads have `.attachment` storage and no read membership declaration.
   Existing FASTQ collections require exact typed library/layout/read/lane/chunk/
   compression attribution in registered member records and owner-compatible
   source locators. Generic role associations currently map a single source to a
   single reviewed consumer, not a partially declared raw library.
2. Scientific parameter continuation exposes reviewed Registry scalar parameters.
   Library namespace, shared-library membership, barcode interpretation/correction
   and whitelist identity are owner declarations, not interchangeable tool scalar
   arguments. There is no current typed user-language handoff for that complete
   declaration set. A BAM CB observation cannot fill it.
3. `scatac_library_context.build_scatac_library_processing_context` requires an
   actual exact intake manifest and qualified declaration/resource bindings.
   Both producer `library_context` ports remain request-only. The existing
   application resource continuation chooses pinned InputSets; it does not
   construct and durably bind this newly generated source-specific resource.

Consequently, candidate role binding alone cannot safely solve B.1. It still
requires a coordinated extension of task-scoped raw membership/declarations,
verified intake preparation, context publication and persisted continuation
identity. No provisional authority, invented intake, historical PBMC context
substitution, or hidden producer workflow was introduced to bridge these gaps.
This is outstanding implementation work, not a claim that the scientific owners
cannot support the intended workflow or that the researcher must author a context.

## Implemented application behavior

`RegisteredInputGroup` gains optional `attachment_candidates=True`, omitted from
legacy serialization. The facade sets it for attachment-only submissions with no
explicit scientific selection or nonempty structured inputs. It captures up to
16 ordered candidates, matching the existing attachment bound. Role associations
remain bounded to eight; no tool or permanent role registry is added.

The existing reviewed source-port set now includes `inspect_scATAC.dataset`.
Its exact `input_path` request alias is resolved explicitly, not by first match.
For candidate groups, typed decisions may bind only a subset. Unassigned files
stay staged, cannot enter another compiled consumer implicitly, and are not
hashed as scientific dependencies. Their registration metadata still must match.
Actual selected-source consumption checks exact registered bytes before science.
Neutral representation can be proposed for a reviewed role, but the unchanged
scientific owner determines whether it can read and qualify the source. In
particular, the current H5AD owner still requires its supported representation;
a neutral `.attachment` is not silently renamed or qualified.

The Interpreter receives safe labels, representation hints, positions and opaque
`source.N` handles. It receives no resource IDs, paths, file hashes or file
contents. The prompt distinguishes candidate attachments from an explicit
all-required selection. No deterministic language parser or semantic retry is
added. A single missing role is offered through the same `request_roles` schema
in candidate context, followed by the existing `answer_roles` decision.

A pending candidate-role answer may explicitly attach additional files. Existing
members retain their exact identities and order; new identities append. Repeated
members must agree exactly and retain their relative order. No unrelated Session
attachment is discovered or inherited. Existing scientific selections, resource
choices and original declarations cannot be replaced through this path.
`_execution_objective` continues to supply the original utterance to the Planner.
The original question does not create a Run, Revision or scientific authority.

Presentation derives the current turn's offered/consumed subset from its admitted
roles and the existing accepted completion guard. A continuation displays both
captured and added candidates. Earlier turns retain their historical staged
status; no history is rewritten. A failed owner check never displays consumption.
The single-file H5AD composer path remains available. Mixed attachments and new
files answering a pending role question use candidate capture. The chat layout
and advanced explicit scientific-input controls are unchanged.

## Implementation inventory

| File | Change |
| --- | --- |
| `src/agent/application/local_resources.py` | Candidate designation, bounded membership, subset role replay, inspection consumer, exact consumption guards; legacy all-required groups remain strict. |
| `src/agent/application/interactive.py` | Scoped attachment-only candidate capture and accepted role-specific attachment display. |
| `src/agent/application/dialogue_execution.py` | Preserve designation during replay and merge explicitly appended candidates into the same pending objective. |
| `src/agent/application/turn_decisions.py` | Single-role question support and conditional candidate instructions/schema. |
| `src/agent/application/turns.py` | Safe candidate-context projection through the existing role offer. |
| `src/agent/application/responses.py` | Plain-language inspection-role clarification. |
| `src/agent/web/static/app.js` | Candidate submission for mixed attachments and pending role replies; preserve existing single H5AD and explicit selections. |
| `tests/application/test_attachment_candidate_contract.py` | Candidate/legacy distinction, consumer substitution rejection, bounds, exact role replay and immutable registration. |
| `tests/web/test_attachment_role_handoff.py` | HTTP uploads, real inspection, original-objective continuation, append/restart/retry, staged status, privacy and fail-closed cases. |
| `tests/application/test_pending_role_contract.py` | Remove the obsolete assertion that a single missing role is always invalid. |
| `tests/web/test_multi_source_selection_static.py` | Permit reading the existing role-clarification response while still prohibiting browser-authored role assignments. |
| `docs/usi-2-task-scoped-scientific-handoff.md` | This record. |

## Scripted-provider scientific evidence

New HTTP cases use the ordinary scoped upload, Web worker and interactive facade.
Only provider decisions are scripted. Production `inspect_scATAC`, compilation,
preflight, independent verification and accepted reporting execute normally on a
2 × 3 sparse simulated fixture. A role question executes zero science; its answer
executes inspection once, with the original objective in the persisted request.
The selected source is consumed and the unrelated attachment remains staged.
The same immutable source records can receive different task-local associations.

An additional-file case starts with a neutral attachment, captures the inspection
objective, then uploads the H5AD on the reply. The original neutral identity stays
first and staged; the new H5AD supplies `source.1`. Another case recreates the
application while the question is pending and answers without reupload. Completed
history/retries work after source removal with a provider factory that raises if
constructed.

Negative coverage includes foreign Session references (including the direct
candidate-group boundary), unknown references, stale generations, reordered
captured sources, changed selected bytes, exact-retry conflicts, invalid neutral
matrix bytes, conflicting/unassigned consumers and candidate bounds. Unused staged
bytes do not gate an independent accepted scientific inspection. Candidate role
assignment is not scientific resource qualification.

The existing library-context and FASTQ/BAM application tests are regression
witnesses only. Their operator/test helper still constructs per-source context.
Neither a new chat FASTQ context nor a new chat BAM context was generated by this
implementation. There is no new raw fragments/QC/selection/matrix result or H5AD
download acceptance attributable to USI-2. No biological provenance, cell purity,
barcode correction history, default thresholds or real-data validity is qualified.

## Native Web acceptance

`R/run-browser.py`, `server.py`, `browser.py` and copied read-only browser helpers
run actual headless Firefox through its native file input and rendered controls.
The isolated server uses the normal Web/application path and scripted providers.
`R/native.log` reports success; `browser-timeline.jsonl`, browser/server logs,
`browser-session.json` and the native screenshot `browser.png` retain evidence.

Verified: neutral upload and submitted objective; visible task-role clarification
with zero science; refresh while pending; H5AD upload in the reply; original-task
continuation; one actual inspection; distinct staged/consumed labels; accepted
evidence and report download; fresh server and browser history restoration with
zero providers and science. The accepted fixture Session is
`f0feb027-96be-44e9-b4f3-2d7a035e01c8`, generation 1, Revision
`08258e2d1baa6fb88b0bd1aedbd9efe972b20d54222f918a5cc8c25f44f2e231`.
This is a real inspection product witness, not a FASTQ/BAM preprocessing witness.

## Bounded live-provider observation

Existing credentials were made available by activating the installed `agent`
environment; secret values were not printed or persisted. The existing operator
profile `openrouter-gpt-5-6-sol` and qualified floor routing were loaded unchanged
into an isolated application. `R/live/observe.py` limits network requests to two
Interpreter calls, no SDK retries, a conservative total reservation below $0.50,
and no Planner or scientific execution. No operator configuration was edited.

The genuine initial utterance asks to inspect one of two attached datasets without
specifying which. GPT-5.6 Sol returned `request_roles`, target `inspect_scATAC`,
required role `dataset`, with no assignment. The reply, “Use the first attached
dataset for the inspection,” returned `answer_roles`, pending `@roles`, assigning
`inspect_scATAC.dataset` to `source.0`. Both original structured decisions were
accepted without repair. Both responses identify `openai/gpt-5.6-sol`; routing
metadata preserves the requested `openai/gpt-5.6-sol:floor` identity.

The two requests reserved **$0.345168** in total; provider-reported usage costs
were **$0.028331** and **$0.0237285**, totaling **$0.0520595**. The subsequent Planner
request was intentionally blocked by the acceptance harness before network I/O,
so the reply ends with `PLANNING_PROVIDER_ERROR` and no Revision. This is bounded
live role interpretation/continuation evidence, **not** a complete live execution
or a production provider failure. Exact observations and terminal views are in
`R/live/{observations,first,answer}.json`; no extra exploratory completions occurred.

## Regression and preservation closeout

The single bounded affected-domain run completed **888 passed, 2 failed,
6 skipped, 3 warnings in 561.75 seconds**, exit 1. All 18 new candidate-contract
and HTTP handoff cases passed. The 35 selected modules cover attachment/role,
resource/parameter continuation, H5AD and fragments composition, FASTQ/BAM
upload/composition/execution, source-bound library context, target-model resource
continuation, existing scientific role execution and browser/static contracts.
All `RUN_*` variables were unset; there were no case exclusions inside those
modules. The six qualified Chromap production cases remained explicitly gated.
This was not a full repository regression.

The exact executable command and complete module list are in
`R/regression-command.txt`; `R/regression.py`, `regression.log` and
`regression-result.json` retain the runner, pytest results and wall time. It uses
`/home/likeyi/anaconda3/envs/agent/bin/python -B -m pytest -q`, `PYTHONPATH=src`,
`PYTHONDONTWRITEBYTECODE=1`, and `--basetemp=R/regression-temp`.

Failure classification:

- `test_selection_transport_rejects_unbounded_ambiguous_or_path_inputs[changes1]`
  expects 422 for a single unknown resource, while the existing supported
  singleton selection returns 404. An isolated `git archive HEAD` of the exact
  starting commit independently reproduces **1 failed in 1.86 seconds**.
  `R/baseline-command.txt` and `baseline-test.log` record it. No unrelated
  production change was made.
- The static `test_multi_source_retry_freezes_captured_selection_and_singles_keep_the_old_wire`
  rejected any substring `roles`, including the new read of the existing
  `missing_input_roles` response. The assertion now prohibits `body.roles` and
  explicitly checks the read-only clarification condition. This test-only
  correction does not change the production bytes exercised by the bounded run
  or native browser. The final static rerun is recorded below.

Final static rerun: **40 passed in 0.11 seconds**, exit 0, over
`tests/web/test_multi_source_selection_static.py`, `test_upload_static.py` and
`test_static.py`, using the same Python and `PYTHONPATH=src`, with
`--basetemp=R/static-final-temp`. Output is `R/static-final.log`. No broad suite
was repeated after this test-only correction. `git diff --check` passes.

The two other previously reproduced USI-1 baseline assertions—invalid explicit
Leiden declaration and application annotation rationale—are outside this selected
module set and remain unchanged/unrepaired. No green full-suite claim is made.

Final preservation: all 1,274 existing eval records retain their original size
and mtime_ns; the original index bytes are unchanged and the staged diff is empty.
Of 681 tracked baseline files, 672 are byte-identical; the nine intended modified
tracked files are seven production files and two test files. Three new files
(two tests and this record) remain untracked, alongside pre-existing `evals/`.
HEAD remains the accepted USI-1 commit. Operator configuration retains the hash
above. `R/preservation.json` and `final-status.txt` record the final check. No
source, test or static bytes changed during the bounded run; the sole subsequent
code-tree change is the documented static assertion correction.

## Remaining scope and architecture review

B.1 and incomplete raw-library objective continuation remain USI-2 prerequisites
for closeout. Required next work is the typed raw declaration/membership extension
and exact owner-built context handoff described above, followed by both FASTQ and
BAM positive/negative chat and browser acceptance. The current upload limit and
resource-selection policy remain unchanged.

USI-3 retains general qualified resource resolution, production upload-size
qualification and appropriately qualified Cell Selection policy. USI-4–USI-6
retain later raw/feature, MAESTRO grouping/resources and cross-species EpiZoo /
post-training application handoffs as scoped by the existing stage records.
USI-7 retains broader deployment and UX qualification. This change does not
silently move the missing B.1 work into those later units.

Registry count/contracts, Planner/compiler/preflight, executor, Session/Revision
storage, authority owners, raw scientific algorithms, EpiZoo and MAESTRO science
are unchanged. The implementation extends the existing binding and pending-turn
mechanisms; it introduces no new engine, parser, catalog or authority. It is a
reviewable partial shared handoff, not fulfillment of the full requested vertical
integration. **Recommendation: keep USI-2 open.**


## USI-2B — FASTQ conversational intake and source-bound context

2026-10-15 (Asia/Shanghai). FASTQ vertical implementation is complete within the
bounded single-library, supported-layout, canonical human/mouse contract below.
Final regression/preservation results are recorded at the end of this section.
**USI-2 and UA3.6.4a/b/c/d remain PARTIAL.** No BAM source-context handoff was added.

### Starting state and preservation boundary

HEAD, local `origin/main`, and freshly queried remote `refs/heads/main` all remain
`36a1b4921a7db935027cf53b90cd478a90ba224e`; branch is `main`. The index was empty
of staged changes. The nine tracked USI-2A modifications, its three new files,
and pre-existing `evals/` were present and were continued in place. No reset,
checkout, replacement of USI-2A, staging, commit, or push was performed.

New evidence root **B** is `/home/likeyi/agent-usi2b-2_kndari/`. `B/baseline.json`
captures the starting tracked and three USI-2A file hashes, index bytes, operator
configuration hash, and size/mtime_ns inventory of the existing eval files.
All new scientific fixtures, source views, contexts, Sessions, Runs, browser
profiles, and provider observations are isolated under B or pytest temporary
workspaces. No reference download, GPU execution, checkpoint load, whole-matrix
densification, historical dataset mutation, production configuration change, or
upload-limit change was required.

### Smallest owner handoffs

The three gaps identified at the historical checkpoint were confirmed in the
existing implementation and linked UA3.6.4b/UA3.4.3/UA3.6.3a/c records:
neutral attachment records lacked read declarations; source/library declarations
were outside Registry scalar parameters; and source-bound context creation was
confined to operator-assisted helpers such as `qualified_companion`.

The implementation connects those boundaries as follows:

| Responsibility | Implementation and invariant |
| --- | --- |
| Language interpretation | `turn_decisions.py` extends existing `request_roles` / `answer_roles` with an optional closed `FastqDeclaration`, separate from generic consumer roles. It offers only opaque captured source/resource handles. |
| Task declaration and continuation | New `fastq_handoff.py` coordinates declaration merge, missing requirements, captured catalog choice, owner preparation, and pure durable replay. It reuses the existing role source-append rule, argument binder, `Interaction.prerequisite`, generation checks, and original-objective execution. |
| Byte admission | `local_resources.py::derive_fastq_member` copies exact neutral-upload bytes into an owner-compatible typed representation through atomic publication and ordinary pinned registration. Original upload records remain immutable. |
| Reusable resources | `ScientificInputSet.fastq_whitelist` stores the existing owner's closed whitelist identity alongside a FASTQ companion. An InputSet with this field cannot also contain a source-specific library context. Legacy definition/catalog digests omit the absent extension and remain unchanged. |
| Scientific preparation | Actual `raw_scatac.inspect_raw_scATAC` produces the source intake; `verify_raw_scatac_intake` independently reconstructs/checks it. The existing library-context builder and publisher produce the exact request resource. |
| Scientific execution | The normal Planner, compiler, preflight, Registry, executor, independent verifiers, RunStore, Session/Revision, report and delivery paths consume the resulting ordinary FASTQ collection and request inputs. |
| Presentation | `interactive.py` projects the original captured attachments with correct offered/consumed/staged use. `responses.py` renders concrete missing FASTQ requirements. The composer recognizes that clarification when adding a file. |

`session_state.py` admits the narrow `field=fastq` prerequisite shape using the
same origin/binding/generation machinery. It adds no Session store or Revision
state. `dialogue_execution.py` and `turns.py` dispatch that variant and preserve
existing parameter continuation. `resource_continuation.py` and `web/config.py`
serialize/read the optional reusable whitelist identity. Registry contracts,
request-only `library_context`, scientific owner algorithms, EpiZoo and MAESTRO
are unchanged. There is no public context-generation tool or second workflow
planner.

### Declaration, missing-member and source identity contracts

The typed variant declares one library: supported 10x ATAC assay; layout
`tenx-atac-r1-r2-r3.v1` or `tenx-atac-r1-i2-r2.v1`; explicit namespace/shared-library
membership; human/mouse species; raw barcode sequences requiring whitelist
correction; and member `(source handle, read role, lane, chunk)` records. Up to
16 captured attachments remain available; unrelated candidates are not bound.
This is the existing chat bound, not a new production upload limit. Scientific
consumer roles remain a different level and are empty in the raw variant.

Missing fields remain absent, never guessed. If layout is absent, Agent asks for
it before diagnosing the missing read. Once declared, `_fastq_member` and
`LocalResourceAdmission._fastq_groups` apply the existing owner's allowed roles,
lane/chunk membership and completeness rules. Optional I1 and both owner layouts
remain supported. Compression is detected from the uploaded bytes. No original
filename parser assigns scientific roles or reconstructs barcode history.

An accepted declaration cannot be reassigned in a reply. New source references
must be explicitly included in that reply; existing captured members keep their
identity and relative order. The complete collection uses the existing owner's
canonical resource-ID ordering; scientific locators use its sorted path contract.
The original user objective, captured source/record digests, admitted declarations,
resource catalog, origin/binding turn, generation and predecessor are durable.
An incomplete library creates no scientific intake, context, Run or Revision.
A missing reusable resource produces an actionable provisioning question while
preserving the task. Explicit resource handles and the existing unique declared
FASTQ-default policy are supported; this is not compatibility-based discovery.

Typed source views are under the managed workspace's `fastq_handoffs/<identity>`
directory. Each derived member has the original byte SHA-256/size, Session scope,
and attribution identifying the original registration and record digest.
Mechanical filenames encode only explicit member declarations required by the
existing intake owner. Registration verifies the copied bytes and checks the
originals again. Original immutable neutral registrations are not retyped.
Foreign-session sources, source mutation, duplicate slots, invalid references and
conflicting declarations fail closed. Unrelated staged files remain unused.

### Source-bound context and execution lifecycle

After complete declarations and reusable resources are available, preparation:

1. Resolves and verifies the exact captured uploads and registers their typed
   member views through existing local admission.
2. Registers the complete FASTQ collection using existing completeness checks.
3. Produces a real intake manifest and runs the existing independent intake
   verifier; no manifest or execution result is fabricated.
4. Calls `build_scatac_library_processing_context` for the exact intake groups,
   declared namespace/membership and raw-sequence/whitelist policy. The owner
   reinspects the pinned whitelist and validates barcode/intake semantics.
5. Publishes the owner-generated context and captures its path/digest, exact
   intake identity, deterministic handoff identity, source registrations and
   selected resource definition in the existing Interaction admission.
6. Supplies the generated context through the unchanged request-only port, then
   executes the user's original objective through the normal scientific plan.

Preparation is a request-resource admission operation, not accepted result
science or a hidden Run. It performs one genuine intake production and independent
verification. The planned registered intake then performs its normal production
and verification too. The test confirms those intake **content digests match**;
the source-specific context is consumed by the ordinary FASTQ producer. Full
read synchronization, barcode correction, reference/runtime eligibility,
alignment, packaging and producer verification remain with existing owners.

Source views/context publication are immutable. Repeated completed HTTP turns
reuse persisted outcomes and invoke no preparation or science. An existing
context publication is reusable only at its exact expected digest; it is never
overwritten. A failed or interrupted preparation can leave unaccepted managed
artifacts; it creates no provisional scientific authority. This does not claim
new automatic crash recovery for interrupted preparation. Existing explicit
interactive interruption/recovery behavior remains in force.

History validation replays captured declarations and resource choice without
opening source files or rebuilding science. It checks the deterministic derived
source IDs, allowed locators, context/intake receipt shape and source identity.
Normal consumption still revalidates registration bytes and scientific context
binding. The owner negative tests retain wrong-intake/context, whitelist,
barcode-policy, namespace and membership rejection.

Selection thresholds are unchanged: explicit scoped declarations or an existing
qualified policy are required. A focused test deliberately omits threshold
bindings and reaches the existing compiler-owned parameter clarification, with
zero registered production calls and no Revision. No permissive test threshold
became a general default.

### Real bounded scientific acceptance

`tests/web/test_fastq_chat_handoff.py` uses normal scoped HTTP uploads, interpreted
turn submission, persisted clarification, explicit missing-file upload and
original-task continuation. Its scripted provider chooses the ordinary five-step
semantic plan; no scientific owner is mocked into success. Reusable reference,
QC/TSS and whitelist fixtures are provisioned **without** a per-upload context.
The application's new handoff alone creates that context after member completion.

The fixture reuses the qualified tiny Chromap patterns and existing pinned
executable/build record `/tmp/m116c-test-build-identity.json`. Its synthetic
reference has one 12,000-base `chrTiny`, one cCRE, one barcode, and five read pairs
at two distinct loci. The resulting canonical matrix is sparse **1 × 1**, with
one nonzero and total count one. The `human`/`hg38` contract labels do not qualify
this synthetic reference as a biological human reference. Explicit zero depth
and TSS thresholds are software-test declarations only. Candidate selection is
not biological cell calling.

The accepted plan is exactly:

`inspect_raw_scATAC → prepare_scATAC_fragments → compute_scATAC_qc → select_scATAC_cells → build_scATAC_cell_by_ccre`.

Source-bound context/intake pins, untouched neutral source records, one accepted
Revision at generation 1, the original request in the Run, staged unrelated
notes, producer verification/alignment, exact retries and fresh application
reopening are asserted. All five eligible scientific downloads are fetched and
checked against pinned SHA-256: `matrix.h5ad`, `barcodes.tsv.gz`, `lengths.tsv.gz`,
`decisions.tsv.gz`, and `selected.tsv.gz`. Tampered context identity, intake digest
and context locator are rejected by durable-state validation.

Focused negatives cover missing layout before missing reads; partial declaration
and later upload; missing configured resources; wrong layout/read role; duplicate
or reassigned slots; forged handles; foreign Session; changed original/derived
bytes; stale generation; incompatible species/resource declarations; invalid raw
barcode policy; and unrelated staged attachments. Existing library-context owner
tests cover incompatible whitelist and wrong intake/source bindings rather than
duplicating their algorithms in application tests.

### Native Firefox and live Interpreter observations

Native Firefox evidence is under `B/browser-final/`. Actual composer controls
upload R1/R2 plus an unrelated text attachment, submit the natural-language task,
show missing genomic R3 with no science, refresh, upload R3 and continue the
original task. The normal application reaches generation 1 and the verified
five-step result. Attachment presentation keeps notes staged and marks only the
three read files consumed. `scientific-audit.json` records the exact accepted Run.

A fresh server then restores the four conversation messages and accepted Revision
with a provider factory that raises if called. Native report viewing and browser
fetches verify all five scientific downloads. Refresh preserves conversation,
Revision and generation. `finish-run.py` passes; `browser.png`,
`browser-session.json` and `browser-timeline.jsonl` retain the visual evidence.
All owned browser/server processes are stopped.

Two development checks are retained honestly. The first native attempt exposed
an overly strict new replay check comparing declaration order with the collection
owner's canonical resource ordering; it failed before planned execution and was
fixed. The next native run completed science, but its harness expected the direct
owner counter to include the Registry's already-captured intake callable. That
counter covers preparation only. The accepted Run supplies the registered intake
record; remaining browser delivery/restart checks completed against that same
accepted result without rerunning science. Neither harness issue is hidden by a
claim that every exploratory command passed.

Exactly two live GPT-5.6 Sol Interpreter requests used the configured OpenRouter
`openai/gpt-5.6-sol:floor` route, strict structured output, returned-model checks,
zero SDK retries, no fallback and no semantic repair. Conservative reservations
were **$0.210090 + $0.236400 = $0.446490**; reported usage costs were
**$0.035561 + $0.039366 = $0.074927**. The production provider configuration and
Groq default were unchanged. `B/live/observations.json` records responses and
routing/usage; `first.json` and `answer.json` retain application views.

Both calls used the new request/answer variant and exact R1/R2/R3 handles. The
initial task correctly clarified missing R3 and unavailable reusable resources;
the reply completed membership and retained the resource question. No preparation,
Planner or scientific execution occurred. However, the first response also emitted
an unrelated EpiZoo species scalar because the initial raw schema offered generic
scalar scopes. That observation led to a narrow schema/admission correction:
FASTQ arguments now include only its registered scientific consumers; species is
carried only by the typed library declaration. An explicit negative test rejects
that unrelated scope. The recorded provider output was not edited or retried;
it would be rejected by the tightened final admission. Thus live evidence supports
bounded member/declaration interpretation but **does not qualify final-schema live
end-to-end reasoning or scientific execution**. The two-call budget is exhausted;
no further live experiment was dispatched.

### Limits and next USI-2C step

This is one explicitly grouped library with at most 16 chat candidates, supported
10x ATAC layouts, existing human/mouse canonical resources and current runtime
qualification. It does not establish biological FASTQ acceptance, calibrated
thresholds, large-library upload usability, general resource compatibility
resolution, arbitrary species or live end-to-end product quality. Changing a
captured operator catalog is not an automatic repair of an existing pending task.
Reusable production reference/whitelist resources still require independent
qualification and explicit operator deployment; no production default was installed.

BAM remains the smallest next step: extend the same typed task declaration and
owner-coordination boundary for exact BAM source, corrected-CB and coordinate/
duplicate-pair provenance, then generate its source-bound context through existing
intake/library owners and qualify normal continuation/execution/delivery. Do not
reimplement the BAM scientific owner or change its request-only context port.
No USI-3 resource resolver, MAESTRO/EpiZoo integration or broader UI work began.


### Final validation and uncommitted file inventory

The final qualified tiny vertical test passes **1 test in 24.57 seconds**, exit 0,
with the explicitly enabled qualified Chromap fixture. It includes genuine
scientific production and verification, source/context coupling, five downloads,
retry/restart and durable-state tamper rejection. Command and output are
`B/qualified-command.json` and `B/qualified.log`.

The single bounded affected-domain run covers 19 directly relevant application,
web and library-context modules: **612 passed, 10 skipped, one failed in
345.69 seconds**, exit 1. All `RUN_*` gates were disabled; skips are explicit
Chromap qualification gates, including the separately executed new positive.
No test exclusions or repeated broad suite were used. The exact module list and
command are `B/bounded-command.json`; output is `B/bounded.log`.

The sole failure was the static assertion
`test_multi_source_retry_freezes_captured_selection_and_singles_keep_the_old_wire`,
which still expected the literal single-reason expression
`clarification.reason === "missing_input_roles"`. The composer now checks both
ordinary-role and FASTQ clarification reasons. The assertion was updated to
require reading the clarification reason and both supported constants, while
retaining the prohibition on browser-assigned roles and internal identities.
This is a test-only correction after the bounded run. The focused static rerun
passes **40 tests in 0.10 seconds**, exit 0, across
`test_multi_source_selection_static.py`, `test_upload_static.py` and `test_static.py`;
output is `B/static-final.log`. Production files were not changed after the run.
No green full-repository regression claim is made; the previously recorded
unrelated USI-1/USI-2A failures were neither fixed nor concealed.

The two new test modules contain **28 tests**. All are covered by the final
bounded run plus the separately enabled positive workflow. Earlier development
runs are exploratory evidence, not substituted for those final results. All
**522 frozen source/test/static files** retained their hashes throughout both
final executions (`B/frozen.json`, `B/verification.json`). The only subsequent
source/test-tree change is the static assertion correction above.

The complete working tree relative to accepted USI-1 contains 13 modified tracked
files and six new implementation/test/documentation files, plus unchanged
pre-existing `evals/`. The tracked files are:

- `src/agent/application/dialogue_execution.py`
- `src/agent/application/input_sets.py`
- `src/agent/application/interactive.py`
- `src/agent/application/local_resources.py`
- `src/agent/application/resource_continuation.py`
- `src/agent/application/responses.py`
- `src/agent/application/session_state.py`
- `src/agent/application/turn_decisions.py`
- `src/agent/application/turns.py`
- `src/agent/web/config.py`
- `src/agent/web/static/app.js`
- `tests/application/test_pending_role_contract.py` (retained USI-2A change)
- `tests/web/test_multi_source_selection_static.py` (USI-2A plus the correction above)

New files are this combined historical/current record,
`src/agent/application/fastq_handoff.py`,
`tests/application/test_fastq_declaration_contract.py`,
`tests/web/test_fastq_chat_handoff.py`, and the retained USI-2A tests
`tests/application/test_attachment_candidate_contract.py` and
`tests/web/test_attachment_role_handoff.py`.

The final preservation audit records unchanged index bytes, an empty staged diff,
all **1,274** pre-existing eval size/mtime_ns records unchanged, unchanged accepted
HEAD/remote tracking, and unchanged external operator configuration SHA-256
`f59312491c2719ae93bc3366724335550a713e5c76cf6bd84ddbc01a0558477e`.
`B/preservation.json` identifies every change since the uncommitted USI-2A starting
snapshot; `B/final-status.txt` records the final Git status. `git diff --check`
passes. No commit or push was performed.

**USI-2B is complete within this bounded scripted-provider scientific/software
and native-browser scope.** Final-schema live-provider quality and biological
FASTQ product qualification remain explicitly unqualified. USI-2 remains open
for the exact BAM handoff and acceptance in USI-2C.

## USI-2B.1 — FASTQ Byte-Efficient Source Handoff and Intake Reuse

**Partial: byte efficiency remains unresolved on this deployment.** This follow-up
adds a capacity-reserved, size-bounded copy fallback and measures the existing
intake lifecycle. It does not eliminate full-byte source duplication or claim a
storage/performance improvement. USI-2A/B evidence above is preserved. USI-2 remains
partial pending BAM handoff; no UA3.6.4 milestone is promoted.

Evidence root `B1` is `/home/likeyi/agent-usi2b1-kripbvpy`, outside Git. The starting
HEAD and local `origin/main` are
`36a1b4921a7db935027cf53b90cd478a90ba224e`. Actual starting status is 13 modified
tracked files, the six untracked USI-2A/B source/test/documentation files listed
above, pre-existing `evals/`, and an empty index. `B1/baseline.json` captures file
hashes, index/operator identity and the 1,274 pre-existing eval size/mtime records.
No reset, checkout, staging, commit or push is performed.

The original `derive_fastq_member` streams every source byte into a new temporary
regular file, verifies source and destination digests, and atomically publishes
that file under the scientific filename. A newly consumed three-file library has
three uploaded originals plus three full-size typed copies. The publication link
is from the independent temporary copy, never from the uploaded source; it is not
a hard-link source-sharing shortcut. Exact sequential retries reuse the existing
published inode after hashing. Incomplete declaration/member continuation creates
no typed files. A different handoff identity (captured candidates, declarations or
resource choice) can create further copies. Pre-publication abrupt process death
can leave an unaccepted temporary file; no new garbage collector or crash-recovery
system is added. Exceptions clean temporary files; an interruption after complete
file publication but before registration can retry without another copy.

Direct source reuse is incompatible with several current contracts, not just an
application filename check. `raw_scatac._kind` selects FASTQ by suffix;
`_raw_fastq._name` establishes syntactic membership from canonical names;
`inspect_fastq_inputs` includes those paths and grouping facts in the intake;
`reinspect_fastq_sources` independently enforces the FASTQ suffix and exact source
snapshot during downstream qualification. The Registry exposes paths and layout,
not an authenticated typed-member mapping. Supporting opaque `.attachment` paths
would need a coordinated owner input, manifest/reinspection and Registry contract
change with legacy identity preservation. Renaming the immutable upload, symlinks,
or sharing its inode would violate existing ownership/integrity assumptions.
That redesign is not introduced here.

The actual `/home` deployment is `/dev/sda`, ext4, with approximately 14.2 TiB
available at the probe. A real Linux `FICLONE` ioctl from an opened 1 MiB source to
a distinct regular file on that filesystem fails with `errno 95 / EOPNOTSUPP`.
`B1/reflink.json` records the probe and filesystem. No automatic-copy command or
silent reflink fallback is used. A reflink path for hypothetical deployments is
not added when it cannot improve this deployment.

The selected last-resort implementation remains in the existing local-resource
owner. Before copying it reserves the exact pinned source length using
`posix_fallocate`; insufficient space, quota exhaustion or unsupported reservation
fails with `FASTQ_SOURCE_CAPACITY_UNAVAILABLE`, before byte copying/publication.
This reservation is stronger than a racy free-space comparison and is per member,
not an entire-library transaction. The copy uses bounded 1 MiB reads, writes at
most the pinned length, rejects early EOF and checks one extra byte for growth.
It preserves all original digest verification, Session scope, neutral registration,
explicit member metadata, no-overwrite publication, fsync and retry behavior.
Existing upload limits remain unchanged (configured per-file cap, supported maximum
8 GiB); no new biological or size eligibility is inferred. Metadata or subsequent
members can still fail for lack of space; no partial scientific Run is accepted.
Empty inputs reserve no data blocks and still require scientific qualification.
Filesystems without allocation reservation now fail explicitly rather than silently
performing unreserved copying. This Linux fallback is not a portable CoW solution.

`B1/measure_storage.py` measures actual `derive_fastq_member` plus registration on
32 MiB of random storage-only bytes before and after the change, with exact retries.
The data are not FASTQ qualification. Both runs use fresh files and a warm page
cache; these single observations are not a statistical performance comparison.

| Metric | Before | After |
| --- | ---: | ---: |
| Input and typed logical size | 33,554,432 bytes each | 33,554,432 bytes each |
| New typed allocation (`st_blocks × 512`) | 33,554,432 bytes | 33,554,432 bytes |
| Initial handoff elapsed | 0.268865 s | 0.261693 s |
| Process logical reads (`rchar`) | 167,811,150 bytes | 167,811,150 bytes |
| Process storage writes (`write_bytes`) | 33,558,528 bytes | 33,558,528 bytes |
| Process storage reads (`read_bytes`) | 0, cached | 0, cached |
| Exact retry elapsed | 0.033720 s | 0.036268 s |
| Retry storage writes / new copies | 0 / 0 | 0 / 0 |

`B1/storage-before.json` and `storage-after.json` include distinct source/view inode
identities and all counters. The 4 KiB beyond data writes includes registration
work. Process counters do not measure physical device traffic including all
filesystem metadata; zero disk reads reflects caching, not zero data consumption.
On this non-CoW ext4 probe the independent inode and allocated blocks corroborate
full duplication. The initial handoff still performs roughly five source-lengths
of logical reads (copy plus required source/destination hashing); retries perform
roughly two. Other lifecycle integrity checks add full-file hashing. Neither this
cost nor large-library storage requirements have been optimized. No large PBMC
benchmark or extrapolated large-file timing is claimed.

The qualified lifecycle measures **two owner production calls, nine independent
intake verifier calls, one context construction, one fragments production/alignment
and one independent fragments verification** per successful first workflow.
The nine include preparation and step acceptance plus existing downstream
evidence/report source-step revalidation (`report.evidence._prepare_evidence`
invokes `verify_step`, which reopens raw intake sources). The initial two-verifier
assumption counted only the direct preparation/execution stages and was incorrect
for the complete presentation lifecycle. The instrumented test exposed this; no
production change removes or hides these existing calls. Preparation
produces and verifies the exact intake needed by the context owner. The ordinary
registered plan then produces and verifies its own accepted step. Both manifests
have the same content SHA within that workflow, but different publication paths.
The preparation manifest is not an accepted Run or prior-output authority.

`prior_outputs.bind_output` requires a terminal successful Run, an accepted step,
reusable schema-2 authority and an owner-qualified output port. Preparation has
none of those execution anchors; moreover `inspect_raw_scATAC` has no schema-2
scientific publication owner for historical prior-output reuse. Its ordinary step
has accepted verification, which is distinct from reusable artifact authority.
The fragments Registry port accepts a request-only library context; no registered
context producer supplies it within a plan. Moving context construction between
accepted steps would require a new execution seam or tool/authority contract, not
merely skipping a duplicate function. Equal manifest SHA cannot supply that
missing authority. The normal explicit `StepOutputRef` intake-to-fragments binding
and independent verification remain intact.

Actual intake work is bounded by 256 records, 2 MiB decoded bytes per source,
16 KiB per line, and a 4 MiB encoded budget for compressed reading (plus bounded
buffer/magic read overhead), with at most 128 candidate files. It does not perform
a full-library content hash or alignment. The context builder checks intake and
whitelist declarations/resources, not full FASTQ sequences. Full-data producer
work and application integrity hashes are separate costs.

`B1/measure_intake.py` runs two standalone production/verification pairs over three
plain synthetic files totaling **19,046,670 bytes**, 40,000 records each. Each
invocation observes only 256 records per file: 54,418 / 11,410 / 54,418 decoded
bytes, **120,246 total**, all stopping at `record-limit`. Production takes
0.119634 / 0.068813 seconds; verification takes 0.035811 / 0.037006 seconds.
The extra registered-equivalent pair costs 0.105819 seconds in this observation.
Logical reads are 180,517 bytes per production and 144,901 per verification,
including manifest IO; all physical read counters are zero from caching.
`B1/intake-measurement.json` records bounds, timings and identical manifest SHA.
The standalone second pair is a timing surrogate, explicitly not an accepted Run.
The qualified witness below counts both real lifecycle call sites, including
evidence revalidation. A separate nine-verification batch over the same moderate
fixture takes **0.324258 seconds total**, with 1,303,162 logical read bytes and zero
storage reads/writes in the warm-cache process. `B1/verification-cost.json` and
`measure_verifications.py` retain all nine timings. These are standalone bounded
cost observations, not reconstructed timings for the accepted workflow. Keeping
these checks avoids disproportionate authority/evidence changes; no duplicate
calls are hidden or relabelled as eliminated.

Validation uses the existing `agent` Python environment with `PYTHONPATH=src`.
The focused declaration/storage suite passes **30 tests in 3.64 seconds**
(`B1/focused-final.log`). Eleven cases were added: allocation ENOSPC/EDQUOT/
unsupported failures; source shrink/growth; pre-publication failure and
post-publication interruption/retry; conflicting declarations; and real intake
plus independent verification for both layouts in plain/gzip form. Existing
foreign-Session, source/destination tampering and immutable neutral-source cases
remain. The first development run exposed a test fixture without a Registry;
that fixture was corrected before the passing focused run.

The single affected-domain regression covers local resources, FASTQ declarations,
attachment candidates, registered FASTQ integration, chat/role handoff, raw FASTQ,
raw manifest, library context and public intake/verifier modules: **881 passed,
one skipped in 112.01 seconds**, exit 0. All `RUN_*` gates were disabled. Exact
argv/environment and the eleven-module list are `B1/domain-command.json`; output
is `B1/domain.log`. The skip is the separately enabled scientific witness. No
full-repository regression or unrelated historical fix is claimed.

The qualified attempt first stopped before workflow execution because the new
timing wrapper did not preserve the callable signature required by Registry
default validation. Adding `functools.wraps` fixed that test instrumentation.
The next attempt executes the real qualified Chromap workflow once, producing a
successful five-step Run and one Revision, then stops at the newly added assertion
expecting two intake verifications; the observed count is nine. That pytest
invocation is **one failed test in 22.07 seconds**, not a green test result.
`B1/qualified-final.log` preserves the failure and measured counts. The assertion
is corrected to nine; timing properties are now recorded before assertions.
The read-only follow-up also corrected the new context assertion to unpack the
context loader's `(path, context, digest)` return value. These are test-only
corrections. The full qualified test is not rerun to repeat alignment solely for
an assertion correction; per-call timings from that attempt were not persisted
before its counter assertion and are therefore not claimed.

`B1/finish_witness.py` completes the remaining checks in a fresh process against
that exact accepted Run, with provider construction, intake, context construction,
fragments production, alignment and independent scientific reconstruction patched
to fail if called. **The follow-up passes**, with zero forbidden calls and all
**64 workspace files byte-unchanged**. Evidence is `B1/witness-closeout.json` and
`witness-closeout.log`. It verifies report delivery, all five digest-pinned downloads,
Session/Revision restore, repeated exact submission retries, forged context/intake/
handoff rejection, and accepted schema-2 authority for fragments, QC, selection
and matrix. Existing owner-negative coverage remains in the domain run, including
wrong intake/context, whitelist and barcode contracts. Source views and original
neutral records retain exact Session attribution and separate inode identities.

The witness starts with ordinary R1/R2/notes attachment upload, captures the original
objective, clarifies missing R3 without production, uploads R3 and resumes the same
task. The notes remain staged and unconsumed. The existing five-step plan executes
intake → fragments → QC → explicit selection → matrix. Preparation and accepted
intake manifests are byte-identical within this run; the producer's explicit
intake arguments resolve to the accepted step, while its context binds the same
intake SHA. Every downstream manifest path/SHA edge matches the actual upstream
result. No preparation artifact is imported as accepted authority.

Comparison with the preserved USI-2B native witness checks actual original/typed
FASTQ source SHA/size, fragments historical-source authority, canonical fragment
stream digest, producer qualification, QC profile/counts, explicit selection and
ordered identities, plus matrix profile/content. Three consumed source sizes are
555 / 215 / 555 bytes, with unchanged source SHA values. Both runs produce two
fragment records with support five, one observed/selected barcode, and a sparse
1 × 1 matrix with one nonzero and total count one. The exact logical matrix SHA is
`309b70ad30bee2cc5f5dc439f5df0e70692dfce633399149f06269811c52c19c`;
ordered cell/feature digests and matrix file SHA also match. Paths, timestamps,
source-bound intake/context identities and execution authorities properly belong
to the new workspace/Run and are not asserted equal across runs. This is bounded
synthetic scientific/software acceptance, not biological qualification or a
large-library benchmark.

The exact qualified commands/environment are `B1/qualified-command.json` and
`qualified-final-command.json`; they use `RUN_CHROMAP_QUALIFICATION=1` and
`AGENT_CHROMAP_QUALIFICATION_RECORD=/tmp/m116c-test-build-identity.json`. The
follow-up command is:

```bash
PYTHONPATH=src /home/likeyi/anaconda3/envs/agent/bin/python /home/likeyi/agent-usi2b1-kripbvpy/finish_witness.py
```

All 533 frozen source/test/static/resource files remain unchanged during the
recorded domain/qualified attempts (`B1/frozen.json`, `verification.json`,
`frozen-qualified-final.json`, `qualified-final-verification.json`). Subsequent
test-only wrapper/count/tuple-return corrections are explicitly described above;
they do not alter production behavior. No second broad regression is run. No
native browser repetition is needed: successful transport, composer, payloads,
model routing and browser presentation are unchanged, and the new storage error
uses the existing error boundary. Prior USI-2B browser evidence remains scoped as
recorded. No paid LLM calls or operator profile changes occur.

The narrow architectural review retains uploaded byte identity in the existing
storage owner, task declarations in the application, FASTQ qualification and
context construction in their scientific owners, and accepted execution in the
normal Registry/RunStore path. No scientific algorithm, Registry, authority,
report owner, Planner, global cache, virtualization service or new storage layer
changes. The only B.1 production edit is the capacity-reserved bounded copy in
`src/agent/application/local_resources.py`; tests extend
`tests/application/test_fastq_declaration_contract.py` and
`tests/web/test_fastq_chat_handoff.py`; this documentation is the fourth changed
file relative to the B.1 starting snapshot.

**USI-2B.1 remains PARTIAL.** Full-byte duplication is neither eliminated nor
reduced on current ext4; duplicate bounded intake/evidence checks are retained
with measured justification. The concrete next decision is whether to provision
a reflink-capable managed workspace and qualify explicit CoW publication, or to
schedule a coordinated typed-source owner/manifest/Registry contract extension.
No filesystem migration or such extension is performed implicitly. Large inputs
still need another full source-size allocation per distinct handoff plus repeated
integrity hashing; interrupted attempts and distinct handoffs can retain extra
files. Capacity reservation prevents unreserved copy attempts, not all future
storage failures. USI-2C can proceed independently on the preserved scientific
boundaries, but this record does not close B.1 byte efficiency, USI-2, or biological
FASTQ acceptance.

Final preservation audit (`B1/preservation.json`, `final-status.txt`) confirms the
same 13 modified tracked files and six untracked USI-2A/B files, plus unchanged
pre-existing `evals/`. Only the four B.1 files named above differ from the starting
snapshot. The entire historical document prefix retains its starting SHA-256;
all 1,274 eval size/mtime records, index bytes, empty staged diff, HEAD/local remote
tracking and operator configuration remain unchanged. Operator SHA-256 remains
`f59312491c2719ae93bc3366724335550a713e5c76cf6bd84ddbc01a0558477e`.
`git diff --check` passes. No biological resource, model checkpoint, historical
accepted workspace, production operator configuration, commit or push is changed.

## USI-2B.2 — FASTQ Input Contract Simplification and Copy-Free Handoff

**Complete within bounded synthetic scientific/software scope.** The normal chat
FASTQ path now binds task roles to the original uploaded bytes and executes the
existing scientific owners without creating another FASTQ data file. This resolves
the byte-copy limitation measured in B.1 on the same ext4 deployment. Historical
A/B/B.1 records above remain unchanged. USI-2 remains partial pending USI-2C BAM;
no BAM handoff or unrelated scientific milestone is implemented or closed.

Evidence root `B2` is `/home/likeyi/agent-usi2b2-_g7332tf`, outside Git. The actual
starting state is main at `36a1b4921a7db935027cf53b90cd478a90ba224e`, 13 modified
tracked files, six untracked USI-2 implementation/test/documentation files,
pre-existing `evals/`, and an empty index. `B2/baseline.json` records hashes and
Git status; `B2/before/` preserves the relevant starting source/test/doc bytes.
The user's correctness-preserving simplification principle is also recorded in
`AGENTS.md`: preserve accepted scientific data, meaning and provenance, while
allowing justified coordinated interface changes and explicit retirement of
unnecessary implementation pathways.

The bounded dependency review distinguishes the following responsibilities:

| Dependency | Requirement and implementation decision |
| --- | --- |
| Read roles, required/optional members, lane/chunk groups, synchronization, barcode meaning | Essential science; retain the same FASTQ parser, layout tables, full-record scanner, context owner and producer/verifier. |
| Captured source IDs, Session ownership, digest pins, immutable role binding, accepted execution lineage | Essential integrity; retain existing local catalog, admission, generation checks, Registry and RunStore/authority. |
| FASTQ suffix dispatch and synthetic filename role extraction | Historical adapter; allow explicit member records for original canonical files. Keep filename interpretation for legacy path inputs only. |
| A separate regular file with a generated scientific name | Unnecessary representation; remove data copying, temporary source publication, capacity reservation and destination hashing. |
| Bounded preparation/step/evidence reinspection | Distinct existing verification boundaries; retain them, measure actual calls, and avoid a separate authority/cache project. |

The existing `raw_input_paths` input port now accepts either its legacy paths/
non-recursive directories or a closed list of explicit FASTQ members, each with
`path`, `library_id`, `role`, `lane` and `chunk`. No new Registry tool, second input
port, side-channel role map or workflow engine is introduced. The name of the
existing argument is retained; its Registry description and Python annotation
now document the typed member form. The application resolves captured handles to
real registered paths; clients/providers do not supply these physical locators.
Assay and layout remain their existing explicit scientific arguments.

`bind_fastq_member` replaces `derive_fastq_member`. It verifies the captured neutral
record, Session ownership and pinned bytes, then publishes only a task-scoped
metadata record in the existing local catalog, with the **same source path** and
source digest. It does not retype or modify the neutral upload. Attribution binds
its original registration ID and record digest; conflicting registration retries
fail closed. Existing collection metadata and fingerprints bind these role records.
Scientific input rows are derived from those records, rather than separately
invented by the Planner. Compression in the storage metadata comes from source
magic; scientific compression observations still come independently from reading
actual bytes. The storage role record is not scientific eligibility or authority.

The FASTQ owner normalizes the closed member shape, rejects mixed path/member
forms, duplicate paths/role slots, unknown roles, directory/traversal/symlink
substitutions and invalid declarations. Typed sources must already be canonical
regular files; inode aliases remain rejected by discovery. Grouping uses explicit
library/lane/chunk declarations. The same bounded parser checks actual bytes,
compression, read counts/names, supported layouts and barcode read meaning.
Invalid labels/content never become ready merely because the input is typed.
Read-role proposals are declarations: the owner checks their structural/content
consistency, not an unobservable biological truth inferred from filenames.

The existing raw-intake manifest already represents `GroupingBasis.EXPLICIT` and
`EvidenceSource.USER_DECLARATION`. Those fields now carry exact per-file
`member-declaration` facts under the existing extensible FASTQ evidence contract.
No fabricated filename evidence or synthetic sample token is emitted for typed
sources. Actual parse/compression/coverage and synchronization remain observations.
A suffix-match diagnostic is retained only for legacy filename inputs; it has no
scientific meaning for an explicitly typed opaque path. No manifest schema bump
is needed for these existing structural/provenance semantics.

`declared_sources` reconstructs the exact explicit member binding from manifest
facts, checks per-file/group evidence completeness, and supplies the same intake
engine for producer source revalidation. Independent intake verification still
reconstructs from the actual registered arguments and compares canonical evidence.
`_fragments_binding.fresh_intake` uses these declarations for new manifests and its
existing path adapter for legacy manifests. Source-only reinspection no longer
requires a filename suffix from a `FileRecord` already declared as FASTQ; content
validation and snapshot checks remain. The existing context builder/validator is
unchanged and binds the independently verified intake SHA.

Downstream production already gets roles from owner-generated `read-meaning`
evidence. Its full-record scanner detects gzip by magic. The pinned Chromap
`SequenceBatch::InitializeLoading` uses `gzopen` on each explicitly supplied role
path (`/tmp/agent-m112b-qualification/candidate/src/sequence_batch.cc`), without a
suffix requirement. Native invocation is captured in the qualified witness:
`-1`, `-2`, and `-b` point exactly to the original uploaded `.attachment` files,
including gzip R1/R3 and plain barcode R2. No scientific algorithms, alignment
flags, preprocessing parameters, reference, whitelist or selection defaults change.
There is no source symlink, source hard link, reflink, FUSE path or hidden
source-copy fallback.
Legitimate fragments, QC, selection, matrix and presentation artifacts are still
published normally.

The application receipt uses `source_mode: original`, included in its handoff
identity, so a new operation cannot collide with an older interrupted copy-based
preparation. Durable replay checks the exact original paths and declarations.
A small read-only branch interprets accepted older receipts without this field;
it retains their recorded materialized paths and identities. Legacy path-based
intake and its canonical manifest bytes remain supported through the same parser.
Historical files/evidence are not migrated or rewritten. Previously composed
`fastq-science.v1` path-only request snapshots are not silently converted into new
typed executions: new work should recompose from the catalog and generate context
against its exact new intake. Operator/test context preparation now uses that
same typed input representation. Old accepted results remain interpretable.

The final compatibility review added a narrow check when consuming a legacy
filename-based intake: its filename-derived role/lane/chunk must agree with any
explicit registered task metadata. Thus an old intake cannot replace a different
explicit task role assignment on the same files. This check is confined to legacy
manifests; new typed manifests compare their actual declarations and do not use
filenames. The original copy pipeline, copy-only capacity error/reservation and
copy-publication tests are removed. Their replacement tests cover immutable
metadata publication, source binding, no data allocation, source mutation,
foreign/forged records, retries and both plain/gzip layouts. Conventional naming
for the older typed-upload entry remains an inexpensive initial-upload/legacy
adapter, not another source materialization stage.

`B2/measure_storage.py` measures real binding on `/dev/sda`, ext4, using a 32 MiB
simulated original plus a subsequent bounded three-member intake probe. It checks
all newly created files, source inode, allocation and exact source SHA.
`B2/storage.json` records the following observations:

| Metric | B.1 measured full-copy fallback | B.2 original-source binding |
| --- | ---: | ---: |
| Main source logical/allocated size | 33,554,432 bytes | 33,554,432 bytes |
| Additional full-size FASTQ copies | 1 | **0** |
| Additional FASTQ source-data allocation | 33,554,432 bytes | **0 bytes** |
| New role metadata allocation | Included in prior measurement | 4,096 bytes (831-byte JSON plus empty lock) |
| Initial binding elapsed | 0.261693 s | 0.060480 s |
| Initial process logical reads | 167,811,150 bytes | 67,113,660 bytes |
| Initial process storage writes | 33,558,528 bytes | 4,096 bytes |
| Exact retry elapsed | 0.036268 s | 0.030991 s |
| Retry allocation / storage writes | 0 / 0 | 0 / 0 |

These are individual warm-cache observations, with different simulated payloads,
not a controlled throughput study or multi-gigabyte timing claim. Both source
files have the same physical allocation on ext4. Process `read_bytes` is zero
because data were cached; it does not imply zero reading. B.2 still hashes the
original for binding integrity (roughly two source lengths in this probe), while
copy reads and destination integrity hashes disappear. The original path/inode
and SHA remain unchanged. The subsequent original-path intake takes 0.101015 s
and independent verification 0.027293 s, together 120,240 logical read bytes and
20,480 storage-write bytes for the small intake artifact. That is legitimate
scientific metadata, not duplicated FASTQ data. No full PBMC benchmark is run.

The qualified end-to-end witness uses real attachment upload, R1/R2 plus notes,
missing-R3 clarification, additional-file continuation and the captured original
objective. It executes the ordinary five-step intake → fragments → QC → explicit
selection → matrix plan with qualified Chromap and independent verification. The
notes remain staged. Its handoff directory contains only intake/context JSON,
with no `sources` directory. Source attribution and accepted fragments authority
pin the original uploaded compressed/plain bytes. Preparation and registered
intakes share the same manifest content SHA within this new representation;
the producer explicitly consumes the accepted step, not invented preparation
Run authority. Four downstream owners retain accepted schema-2 authority.

The instrumented lifecycle records two intake productions (0.125243 / 0.079698 s),
nine intake verifications (0.287593 s total), one context construction (0.005538 s),
one alignment/fragment production and one independent fragments verification.
`B2/qualified.xml` and `owner-timings.json` preserve all timings. The nine checks
include existing evidence/report revalidation. No extra calls occur on exact retry
or restored presentation. Intake still has bounded record/byte budgets; full
source hashing and actual scientific processing remain separate responsibilities.
This task removes source copying, not the necessary accepted-evidence boundaries.

Comparison with the accepted USI-2B native witness checks the decoded bytes of each
intended read role, source pins, explicit intake-to-context-to-producer edges,
context namespace/membership/barcode policy/whitelist meaning, canonical fragment
stream, producer qualification, QC profile/counts, selection and ordered matrix
identities. The new encoded R1/R3 SHA values legitimately differ because they are
gzip and the source paths/evidence now represent original opaque uploads. Their
decoded role SHA values exactly match B; R2's original SHA also matches. Both runs
produce two fragment records with support five, one selected barcode and an
int64 sparse 1 × 1 matrix with one nonzero/total count one. Logical matrix SHA is
`309b70ad30bee2cc5f5dc439f5df0e70692dfce633399149f06269811c52c19c`;
ordered cell/feature digests and matrix file SHA also match. New intake/context
and execution identities correctly remain specific to their new provenance.

The qualified test passes **1 test in 22.36 seconds**, exit 0, on its first B.2
scientific execution. It includes all five pinned downloads, report delivery,
source/context tamper rejection, exact retries and application restart. The
additional fresh-process `B2/finish_witness.py` repeats read-only delivery,
lineage/baseline comparisons and retries with providers and scientific owner
calls forbidden: **passes, zero forbidden calls, all 61 workspace files unchanged**
(`witness-closeout.json`). Separately, `B2/audit_history.py` loads the preserved B.1
materialized-path Session, checks its four accepted authorities and reconstructs
its legacy intake exactly: **passes, all 64 historical workspace files unchanged**
(`historical-audit.json`). That historical audit performs bounded inspection only,
not production. No historical digest is updated to make compatibility pass.

Development validation passes 35 application checks (one gated skip), 58 owner/
integration checks, and 120 composition/upload/web checks (six gated skips).
The new owner module contains 29 positive/negative cases covering both layouts,
optional I1, mixed/plain/gzip compression, explicit provenance, bounded inspection,
independent full-record scanning, lane separation, malformed bytes/gzip, read-name
mismatch, missing/wrong layout, duplicate/forged members, mixed declarations,
unsafe paths, source mutation and changed evidence. Existing context/whitelist
negative tests remain unchanged.

The single final affected regression passes **1,271 tests, 7 skipped in 202.38
seconds**, exit 0, across 19 relevant source/admission/FASTQ/context/raw-intake/
producer-contract modules. It includes BAM cases at the shared raw dispatch
boundary; no BAM conversational change is made. All `RUN_*` gates are disabled;
the seven scientific qualification skips are explicit, with the separate new
qualified witness above enabled. All **534 frozen source/test/static/resource
files** retain their hashes during both final runs (`B2/frozen.json`,
`verification.json`). The subsequent legacy-role guard and its regression test
are the only later production/test edits; their targeted suite passes **165
checks in 48.79 seconds**, exit 0 (`legacy-guard.log`). The guard does not affect
the explicit-manifest scientific witness. No second alignment or broad regression
is performed for that legacy-only correction.

Exact final argv/environment/module lists are in `B2/qualified-command.json` and
`domain-command.json`, with output in the corresponding logs. The qualified run
uses `RUN_CHROMAP_QUALIFICATION=1` and the unchanged build record
`AGENT_CHROMAP_QUALIFICATION_RECORD=/tmp/m116c-test-build-identity.json`. The final
legacy guard check is:

```bash
PYTHONPATH=src /home/likeyi/anaconda3/envs/agent/bin/python -m pytest -q tests/application/test_fastq_declaration_contract.py tests/application/test_fastq_composition.py tests/application/test_fastq_resources.py --basetemp=/home/likeyi/agent-usi2b2-_g7332tf/legacy-guard-temp
```

No native browser repetition or paid LLM call is needed: composer/payload routing
and successful presentation contracts are unchanged, and the real HTTP workflow
is exercised. Providers are scripted for deterministic integration. Synthetic
resources labelled human/hg38 are not biological reference qualification, cell
purity evidence or large-library throughput acceptance.

Relative to the B.2 starting snapshot, changed files are:

- `AGENTS.md` and this appended acceptance record.
- `src/agent/application/local_resources.py` and `fastq_handoff.py`.
- `src/agent/tools/data/_raw_fastq.py`, `raw_scatac.py` and `_fragments_binding.py`.
- `src/agent/orchestration/registry.py` (input description).
- `tests/application/test_fastq_declaration_contract.py` and `test_fastq_uploads.py`.
- `tests/web/test_fastq_chat_handoff.py` and `test_fastq_execution.py`.
- New `tests/tools/data/test_explicit_fastq_sources.py`.

The six production modules have a **net increase of 81 lines** compared with B.1,
not a claimed line-count reduction (`B2/implementation-delta.json`). The application
loses the full-copy machinery; the owner gains closed declaration validation and
manifest reconstruction. There is one scientific parser/producer path, with a
small legacy filename adapter and receipt reader, not parallel scientific engines
or a permanent copy fallback. No new file/role registry, cache, authority schema,
Planner or scientific algorithm is introduced. Registry tool count remains 23;
EpiZoo, MAESTRO, BAM science and provider/operator settings are unchanged.

USI-2B.2 is ready for review and USI-2C can proceed on the preserved ownership
boundaries. No B.2-specific blocker remains for that next task. B.2 does not start
BAM work or close USI-2. Original-source lifetime and integrity remain necessary
for new FASTQ execution; no cross-user sharing, hidden resource substitution or
large-data biological acceptance is implied.

The final preservation audit (`B2/preservation.json`, `final-status.txt`) confirms
20 modified tracked files and seven new implementation/test/documentation files,
plus unchanged pre-existing `evals/`. Twelve existing files differ from the B.2
starting snapshot, with one new owner test module. All 1,274 pre-existing eval
size/mtime records, index bytes, empty staged diff, HEAD/local remote tracking,
and the entire historical documentation prefix are unchanged. Production operator
SHA-256 remains
`f59312491c2719ae93bc3366724335550a713e5c76cf6bd84ddbc01a0558477e`.
`git diff --check` passes. Nothing is staged, committed or pushed; protected
scientific resources and historical results are not modified.

## USI-2C — BAM Conversational Handoff and USI-2 Integration Closeout

2026-10-15 (Asia/Shanghai). BAM conversational handoff is implemented and its
bounded synthetic scientific, HTTP and native-browser witnesses pass. Final
integrated regression and preservation results are recorded below. This is not
real biological BAM qualification or full live-provider scientific acceptance.
UA3.6.4a/b/c/d remain PARTIAL; no USI-3 work is started.

### Baseline and preserved work

Evidence root **C** is `/home/likeyi/agent-usi2c-ve1brutj`, outside Git and prior
acceptance directories. The actual starting branch is `main`; HEAD and local
`origin/main` are `36a1b4921a7db935027cf53b90cd478a90ba224e`. The starting tree
contains 20 modified tracked files, seven untracked USI-2 implementation/test/doc
files, pre-existing `evals/`, and an empty index. `C/baseline.json` and `C/before/`
preserve exact starting source/document bytes, index/operator hashes, Git status,
and the 1,274 pre-existing eval size/mtime records. A later read-only remote query
also reports that same main commit (`remote-main.txt`). No reset, checkout, clean,
staging, commit, push, production configuration change or historical migration.

### Owner boundaries and input simplification

The public corrected-CB paired-ATAC producer already supports the requested route.
Its existing profile requires one BAM/group/library, human/hg38 or mouse/mm10,
exact reference names/lengths and assembly, corrected CB identifiers, unshifted
paired alignments and duplicate pairs retained. The existing decoder, pair
processing, MAPQ policy, Tn5 shifts, aggregation, independent reconstruction,
packaging and authority remain unchanged. Original FASTQs, coordinate sorting and
BAI/CSI are not prerequisites. The neutral arbitrary-species BAM route remains
separate and unchanged.

The missing boundaries were neutral attachment format dispatch, typed historical
source declarations, and application-owned context creation. `_raw_bam` previously
required a `.bam` suffix for direct discovery and recorded-source reinspection;
`raw_scatac` also dispatched solely by suffix. These are representations, not
scientific evidence. The existing `raw_input_paths` argument now accepts closed
BAM rows `{path, input_kind: "BAM"}` alongside its explicit FASTQ-member form and
legacy path/directory form. The Registry description documents both typed forms;
there is no new tool, argument port, side-channel map or authority schema.

BAM rows require canonical absolute regular files and reject mixed representations,
unknown/extra fields, duplicate paths, directories, traversal and symlink
substitution. The existing discovery owner still rejects inode aliases. Explicit
format declarations permit opaque filenames; actual BAM decoding, header/tag/read
observations and scientific eligibility remain owner checks. Arbitrary bytes do
not become valid BAM because of a label. Recorded BAM reinspection trusts its
already-declared `FileRecord` kind while rechecking the actual source and bytes.
The producer reconstructs explicit-file intake from those exact recorded BAM
sources; legacy directory/mixed-root reconstruction retains its existing roots.

No manifest version change is needed: existing `InputKind.BAM`, explicit selection,
one-group-per-BAM identity and user-declaration/observation distinctions already
represent this input. Typed and legacy representations use one scientific parser.
Legacy named BAM registration/composition remains supported. Existing intake,
context and accepted artifact bytes are neither migrated nor rewritten.

### Shared task and source binding

`application/fastq_handoff.py` is consolidated into
`application/raw_handoff.py`; its internal imports are updated, with no permanent
compatibility wrapper. Historical JSON records do not encode that module name.
The shared module owns capture, immutable declaration merge, exact source-handle
resolution, resource choice, prerequisite generation, original-objective
continuation, intake verification, context publication and pure durable replay.
There is no separate BAM pending-task engine or copied BAM workflow module.

`BamDeclaration` contains an optional captured source handle, assay, namespace,
species, source assembly, corrected-CB policy, supported source profile and offered
resource handle. The profile explicitly means unshifted paired ATAC coordinates
with retained duplicate pairs. The barcode policy explicitly means already
corrected CB identifiers. These are declared processing history, not facts proved
from observed CB tags, reference names or coordinates. Missing values remain
missing. FASTQ retains its own layout/read-role/lane/chunk and raw-barcode/whitelist
semantics; none of those member requirements is imposed on BAM.

The Interpreter receives only safe source/resource handles and labels. Its existing
`request_roles` / `answer_roles` schema gains the closed BAM variant. Raw scalar
arguments are restricted to the respective registered consumers. Scientific
thresholds still require explicit supported declarations; there are no new defaults.
The Agent rejects reassigned captured sources/declarations, forged resource handles,
changed catalogs, wrong pending fields, stale generations and foreign-session
references through the existing admission and prerequisite machinery.

The existing local catalog now has one `bind_raw_source` operation. FASTQ's former
format-specific binding wrapper is removed. It validates the immutable neutral
record, Session scope and exact pinned source bytes, then registers task-local
format metadata pointing to the **same original path** and SHA/size. Attribution
pins the neutral registration and record digest. It does not retype the upload,
rename it, create a data copy, symlink, hard link or CoW view. Exact registration
retry retains the same record. Unrelated candidate attachments stay staged.
H5AD continues to use the same upload/catalog identity mechanisms and its own
scientific reader; its validation is not forced into the raw-data contract.

### Necessary clarification and context lifecycle

The representative request asks to preprocess a paired scATAC BAM and produce a
matrix, but omits assembly and corrected-barcode/coordinate/duplicate history.
Agent asks for those declarations, preserves the source and original objective,
and performs zero intake/context/producer work while they are missing. A valid
answer continues that original request, including across a recreated application.
An invalid answer cannot produce an accepted Run or Revision.

Reusable BAM companions are offered through the existing InputSet catalog only
when they contain no source-specific library context. A choice must use an exact
offered handle or the existing unique explicitly configured default policy;
there is no new compatibility search or first-match choice. Missing resources
produce a provisioning question while retaining the task. The synthetic fixtures
provide reusable reference/QC resources, never a per-upload context.

Once the declarations and resources are complete, the shared handoff:

1. Binds the exact original upload through the existing local resource owner.
2. Runs real BAM intake on the original opaque path and independent intake
   verification with the actual registered arguments.
3. Passes that exact intake to `build_scatac_library_processing_context`, with
   one declared namespace/group, corrected identifiers and already-corrected policy.
4. Publishes the context without overwrite and records source/declaration/resource
   identity, intake SHA and context locator/SHA in the existing Interaction record.
5. Composes the existing `bam-science.v1` request with its unchanged request-only
   library-context port, then invokes normal planning, compilation and execution.

The handoff directory contains intake/context JSON only. It is not a Run or
provisional scientific authority. The planned intake executes and is independently
verified normally; its digest matches the preparation intake. The producer binds
the actual accepted upstream step and the source-specific context. Source/context
substitution fails closed. Interrupted preparation may leave unaccepted managed
metadata, but creates no authority; no new recovery engine or garbage collector
is claimed. Completed retries and history reads perform no preparation or science.

### Scientific and native product evidence

`tests/web/test_bam_chat_handoff.py` exercises scoped neutral uploads, an unrelated
note, missing-information clarification, application restart while pending, typed
answer, original-task execution, authority, delivery and completed retry/restart.
Only provider decisions are scripted. The actual owner route is:

`inspect_raw_scATAC → prepare_scATAC_bam_fragments → compute_scATAC_qc → select_scATAC_cells → build_scATAC_cell_by_ccre`.

This plan is provider-authored test data, not a hardcoded application template.
The fixture reuses the accepted tiny paired-ATAC design: unsorted BAM, two paired
primary templates, no index, synthetic chr2/chr1 reference, and explicit zero
selection thresholds. The newly uploaded 259-byte BAM remains neutral and byte-
identical at its original `.attachment` path. Scientific production yields two
fragment records with support two, one observed/selected barcode and an int64
sparse **1 × 2** matrix with **one nonzero and total count two**. Logical matrix SHA:
`5b621373bb2175896becdd9f0f8e8c90eab8755e978f071d460c779a0a16d41f`.
Matrix file SHA:
`42a7f0971d9754ab6370abea64a9db3869e7c84de6e7eb09b1a5e54f7cb38ac9`.
These agree between the HTTP and native-browser witnesses.

The first focused BAM scientific test passes **1 test in 17.30 seconds** with one
pytest XML-format warning (`C/qualified.log`, `qualified.xml`). Its counters show
one registered intake, one BAM production and one independent BAM reconstruction.
Preparation additionally runs one intake (0.087900 s) and one context construction.
Nine independent intake verifications cover preparation, acceptance and the
existing evidence/report source-step checks. Those nine take 0.204935 s in total.
Their repeated bounded work is retained for the same distinct verification
boundaries documented in B.1/B.2. Context/producer/verification do not repeat on
completed retries. No new scientific cache or authority shortcut is introduced.

`C/finish_witness.py` reopens that exact accepted result in a fresh process with
providers, intake, context construction, BAM production and independent scientific
reconstruction forbidden. It checks the original request, every upstream path/SHA
edge, four accepted schema-2 authorities, exact original-source attribution,
forged source/declaration/context rejection, report and all five pinned downloads,
and repeated completed submissions. It passes with **zero forbidden calls and all
46 workspace files unchanged** (`witness-closeout.json`). It also reads the accepted
B.1 materialized-source and B.2 original-source FASTQ Sessions and their four
accepted authorities: **64 and 61 historical files respectively remain unchanged**.
The small read-only legacy FASTQ receipt interpretation remains; no copy fallback
is restored.

Native Firefox uses the real composer and native file input (`C/browser/`). It
uploads BAM plus notes, submits the ordinary task, displays the scientific question
with zero science, refreshes while pending, answers in the same conversation,
executes the real five-step route and exposes the report/results. The BAM is shown
as consumed; notes remain staged. Browser fetches verify all five original bytes:
`matrix.h5ad`, `barcodes.tsv.gz`, `lengths.tsv.gz`, `decisions.tsv.gz`, `selected.tsv.gz`.
A fresh server restores four conversation messages and the accepted Revision with
providers and scientific owners forbidden. `native.log` reports success;
`browser.png`, timeline, logs and Session snapshot retain evidence. Session is
`1f14ea68-ea48-4dbf-b788-97bbb5c93b04`, generation 1, Revision
`48f921717d290035261a3fdd4c6c86224d659675e57ac483bf5b08152263e196`.
All owned browser/server processes are stopped. No UI redesign was needed; the
composer only gains the existing BAM-declaration clarification reason.

These are bounded synthetic scientific/software results. Human/hg38 fixture labels
are not biological reference qualification; explicit corrected-CB/source-history
declarations are not independently established biological provenance. Selection
remains candidate selection, not validated cell calling or calibrated thresholds.

### Bounded live-provider observations

After the typed contracts stabilized, exactly two Interpreter calls used the
unchanged configured OpenRouter `openai/gpt-5.6-sol:floor` route with strict output,
returned-model checks, cost filters, no fallback, zero SDK retries and no semantic
repair. Both responses identify `openai/gpt-5.6-sol`. The harness capped total
conservative reservation at $0.60; actual reservations were **$0.201116 + $0.184576
= $0.385692**. Reported costs were **$0.0340585 + $0.030441 = $0.0644995**.
Credentials were loaded through the installed environment and not printed.

The initial response selected `request_roles` for the exact BAM source, scATAC
assay, human species and the two explicit selection operands. Assembly and
processing-history declarations remained absent. The reply selected `answer_roles`
and correctly supplied hg38, corrected-CB/already-corrected policy and the
unshifted/retained-duplicate source profile. Both decisions passed typed admission.
It left namespace null even though the scripted witness uses “sample”; that
conservative language interpretation is preserved, not repaired. Missing reusable
resources were intentional in this isolated semantic observation. The original
task therefore remains pending library name and resource availability. There are
zero Planner or scientific calls and no handoff artifacts. Raw observations and
views are in `C/live/`. This supports bounded BAM declaration interpretation and
continuation, **not complete live-provider execution or universal intent quality**.
No further paid call was made to obtain a more favorable result.

### Storage and bounded architecture review

`C/measure_storage.py` uses 32 MiB of clearly simulated storage-only bytes on the
same `/dev/sda` ext4 filesystem. It inventories every new file and checks original
path, inode, allocation, size and SHA across initial binding and exact retry.
`storage.json` records **zero full-size BAM copies and zero additional BAM data
allocation**. The only new nonempty file is a 696-byte local catalog JSON, allocated
4,096 bytes; its lock is empty. Initial binding takes 0.059145 s with 67,109,565
logical read bytes and 4,096 process storage-write bytes. Retry takes 0.045570 s,
creates no files and writes zero storage bytes. Source integrity hashing remains
approximately two source lengths; cached physical reads are zero. These single
warm-cache observations are not a throughput benchmark or biological BAM test.

The bounded A/B/C review confirms one upload/catalog source identity, shared opaque
source handles and task-role admission, and the existing prerequisite/Session
lifecycle for roles, raw declarations, resources and parameters. FASTQ and BAM
share the coordination in `raw_handoff.py`, with small source-specific declaration,
resource and preparation/replay branches. Scientific algorithms remain with their
owners. No new registry, workflow engine, source store, authority mechanism or
provider-specific scientific rule is introduced. There are still 23 Registry tools.

Removed/replaced: format-only BAM suffix restrictions for explicitly typed sources;
FASTQ-only lifecycle/module naming; the redundant FASTQ binding wrapper. No BAM
copy adapter is added. Legacy path dispatch and the historical FASTQ receipt reader
remain only where actual historical interpretation depends on them. Source-bound
intake/context/execution identities and independent owner checks remain necessary.
Thirteen production modules, counting the module rename as one change, gain **175
net lines** over B.2 (`implementation-delta.json`); this is not a line-count reduction.
The reduction is in duplicated pathways: there is one raw coordination lifecycle,
not parallel FASTQ and BAM implementations. EpiZoo, MAESTRO and neutral BAM science,
provider routing, Groq default and production resource settings are unchanged.

### Validation, preservation and closeout

Focused development checks pass 65 tests (one gated skip), then 25 tests, then
60 tests covering the final declaration/binding and pending-answer additions.
The final focused 60-test run takes 9.83 seconds (`focused-final.log`). New negatives
cover unsupported declarations/profile, species/assembly conflicts, corrected-CB
policy, immutable merge, forged handles, mixed/unsafe source forms, invalid bytes,
source mutation, foreign Sessions and stale generation. Existing owner suites
retain full pair/tag/reference/context/coordinate-policy qualification; their
scientific algorithms are not reimplemented in application tests.

The refactored final shared path passes the qualified FASTQ chat workflow:
**1 test in 21.85 seconds**, exit 0 (`fastq-qualified.log`). This rechecks real
Chromap, both source/context edges, original-byte inputs, producer verification,
report/downloads and exact retry/restart after the shared lifecycle change.

The single final integrated regression covers **47 modules** and completes with
**2,045 passed, 7 skipped, 1 failed in 870.94 seconds**, exit 1. All `RUN_*` gates
are disabled, with no test exclusions inside those modules. The seven skips are
explicit FASTQ/Chromap qualification gates; the separately enabled FASTQ witness
above passed. The new BAM positive runs normally in this regression, including
its final original-objective and exact matrix assertions. Both new BAM test modules
contain **33 cases**, all passed. The exact argv, environment, module list, XML,
output and runner result are `C/integrated-command.json`, `integrated.xml`,
`integrated.log` and `integrated-result.json`.

The sole failure is
`test_selection_transport_rejects_unbounded_ambiguous_or_path_inputs[changes1]`:
an unknown singleton resource returns 404/`LOCAL_RESOURCE_UNAVAILABLE`; the test
expects 422. This is the unchanged failure independently reproduced at the exact
USI-1 starting commit and documented in USI-2A (`R/baseline-test.log`). It is not
introduced by C and was neither deselected nor repaired as unrelated work. No
new failing case, second broad regression, or green full-repository result is
claimed. The other previously recorded unrelated baseline assertions were outside
this affected module list.

All **536 frozen source/test/static/resource files** retain exact hashes throughout
both final executions (`C/frozen.json`, `fastq-qualified-result.json`,
`integrated-result.json`). No production/test edits follow these runs. The earlier
focused BAM/native witnesses preceded the final small binding-wrapper retirement,
import naming cleanup and stronger test assertions; the final regression and
qualified FASTQ run exercise that final shared implementation. These changes do
not alter the browser payload or successful presentation contract.

Relative to the B.2 snapshot, the implementation changes twelve existing production
files and replaces the untracked `application/fastq_handoff.py` with the shared
`application/raw_handoff.py`. The existing FASTQ declaration/chat tests update
internal imports/binding calls; new tests are
`tests/application/test_bam_declaration_contract.py` and
`tests/web/test_bam_chat_handoff.py`. This record is appended, with only its top
current-status pointer updated. All text from the historical USI-2A checkpoint
through the end of B.2 is byte-preserved.

The final tree relative to accepted USI-1 has **22 modified tracked files and nine
new untracked implementation/test/documentation files**, plus unchanged pre-existing
`evals/`. `C/final-status.txt` lists them all; `preservation.json` identifies changes
relative to B.2 and the deliberate internal module replacement. Index bytes remain
unchanged and the staged diff is empty. All **1,274** original eval size/mtime
records, HEAD/local remote tracking, and operator SHA-256
`f59312491c2719ae93bc3366724335550a713e5c76cf6bd84ddbc01a0558477e`
remain unchanged. `git diff --check` passes. No biological downloads, model loading,
GPU work, full-matrix densification, production upload-limit change or protected
scientific-resource mutation occurred.

**USI-2C is COMPLETE within the bounded scope above. Recommend formal closeout of
USI-2 as COMPLETE within its defined implementation and bounded scientific/software
acceptance scope**, with the known baseline test failure disclosed. There is no
remaining central BAM/shared-handoff implementation blocker. The next action is
review of this uncommitted diff and acceptance record before separately authorized
commit/push; no commit or push has been performed.

This does not qualify genuine biological source-processing history, calibrated
cell selection, production-scale upload usability, generic resource resolution,
arbitrary-species public BAM, or a complete live-provider scientific workflow.
Original sources remain necessary for new consumption. UA3.6.4a/b/c/d and the later
USI-3–USI-7 product work remain open within their previously declared scopes.

## Final review and authorized closeout

2026-10-15 (Asia/Shanghai). **USI-2 — COMPLETE within its defined implementation
and bounded synthetic scientific/software acceptance scope.** The bounded final
review accepts the combined A/B/B.1/B.2/C implementation for the user-authorized
single commit and normal push. Earlier uncommitted checkpoints above remain
historical. The final commit identity and remote synchronization are reported in
the delivery response, without a second documentation-only commit.

The review confirms shared `raw_handoff.py` coordination, original-byte FASTQ/BAM
binding, distinct scientific declarations, existing prerequisite continuation and
unchanged scientific qualification/authority owners. The retired FASTQ handoff
module and copy-only machinery are absent. No production or test correction was
needed. All 536 files frozen for the final C acceptance still match exactly, so
the qualified FASTQ/BAM, native Firefox, restoration/download, storage and bounded
live Interpreter evidence above is retained without repeating scientific execution.
The integrated result remains **2,045 passed, 7 skipped, 1 known baseline failure**:
unknown singleton resource HTTP 404 versus expected 422, independently reproduced
at USI-1. No newly introduced failure or entirely green regression is claimed.

The exact commit inventory contains 31 implementation, test and documentation
files. The final review preserves all 1,274 pre-existing eval size/mtime records
and the production operator configuration hash recorded above. External evidence,
scientific datasets, historical results, credentials and `evals/` are excluded
from staging. Review/preservation records are outside Git at
`/home/likeyi/agent-usi2-closeout-kc7xobho/`.

This closeout does not qualify genuine biological FASTQ/BAM workflows, complete
live-LLM scientific execution, generalized qualified-resource auto-selection,
production-scale uploads, scientifically qualified universal Cell Selection
defaults, or complete external-fragments/custom-BED handoff. UA3.6.4a/b/c/d remain
**PARTIAL**. USI-3 through USI-7 and UA3.6.5 remain future work; none begins here.
