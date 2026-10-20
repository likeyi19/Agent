# UA3.6.5-R7 — Internal state, lifecycle and resource reuse

2026-10-20. R7 simplifies the existing file-backed architecture. It does not add a
Task entity, database, event framework, recovery coordinator or second resource
catalog. R5 interpretation/planning/answers and R6 scientific owners remain the
execution and evidence boundaries. The final closeout instruction explicitly
authorizes one reviewed R7 commit and normal push to `origin/main`; earlier
no-commit statements below describe their historical acceptance stages.

Acceptance evidence is isolated at `/home/likeyi/agent-ua365-r7-o3pqsmyz`.
Final validation and disposition are recorded below; development checks are not
additional independent acceptance totals.

## A. Baseline and protected state

Before modification, HEAD, local main, origin/main and independently queried
remote main all resolved to `5c5c1bab74b4260dca968e5b3107932bf6a8ae3b`.
The index and tracked worktree were clean. Only existing `evals/` was untracked.
The audit read current README/AGENTS contracts, R4–R6 records and the application,
Session, interpreter, resource, Web, RunStore and runtime consumers below.

`protected-baseline.json` records 262,453 existing files, extending the preceding
R6 protected inventory with current normal configuration/workspace/library and
both R6 acceptance directories. It records size, nanosecond mtime and mode;
1,478 bounded files also have SHA-256, including operator configuration and small
normal-workspace, library and eval records. This is a bounded preservation check,
not a new full-content checksum of every checkpoint or biological dataset.

All development uses the `agent` Conda environment, private temporary workspaces
and disabled provider/scientific opt-in test gates. Live acceptance uses a copied
operator configuration, independent uploads/workspace, Firefox profile and local
port 8879. Credentials come from the existing environment and are neither printed
nor copied into Git. The configured scientific library remains read-only.

## B. Authoritative lifecycle and consumers

| Fact | Authoritative owner and current readers/writers |
| --- | --- |
| Conversation identity and ordered interaction history | `AnalysisSession` / `FileSessionStore`; `turns.respond` captures before interpretation; interactive/Web reads project persisted history. |
| Exact original utterance, selected model/config fingerprint, inputs and attachments | `Interaction.submission`; `interactive._checked_submission` compares exact retries; input/resource owners validate new consumption. A later upload cannot replace a captured binding. |
| Base Revision/generation, offered handles and prerequisite origin | Immutable `Interaction.snapshot`; `turn_context`, `scientific_dialogue`, `dialogue_execution`, `raw_handoff`, resource continuation and guidance selection consume the captured context. |
| Interpreted/admitted intent and clarification | `Interaction.admitted`, `status`, `prerequisite`; typed interpreter and admission owners write these. Clarification creates no execution checkpoint. |
| Reserved scientific request/run identity and selected/retained outputs | `SessionTurn`; `sessions` and the two admitted-plan callbacks reserve it before runtime execution. Its Run ID is request-derived. |
| Plan, tool attempts, cancellation, native completion and verified results | `FileRunStore` / `AgentRuntime`; execution leases, revision checks, cancellation sidecars and scientific publication/authority owners remain authoritative. |
| Application evidence/report completion | `SessionTurn.completion_files` and result digest; `_record_result` pins successful composition; `recover` checks those bytes before first activation. |
| Accepted result version, exact selected outputs and parent | Immutable `AnalysisRevision`; publication uses the captured base, accepted Run/step anchors and explicit retained outputs. No implicit inheritance. |
| User-selected result | Session active pointer and generation, changed atomically with `Navigation`; stale completion publishes a historical child without stealing the active selection. |
| Displayed response | First-write immutable `Interaction.presentation`; reads/retries never regenerate model wording. It is not scientific authority. |
| Current processing activity | Existing OS-held per-turn processing lease; it disappears on process exit. File existence alone is never an activity claim. |
| HTTP pre-capture acknowledgement | Bounded `_LocalWorkers` receipts/futures. They account for capacity and pre-capture errors, not durable scientific execution or restart replay. |
| Private uploaded source identity | Existing local resource registration, scoped to a Session; scientific owners establish measurement compatibility. |
| Shared references/models | One configured `ScientificResourceLibrary`; explicit installer/sharing action and immutable identities, not Session history or display labels. |

The actual path remains: create Session → validate submission → take per-turn
lease → capture interaction → interpret → clarify/answer/navigate or admit science
→ reserve SessionTurn and mark submission atomically → RunStore execution → pin
application completion → publish Revision with generation CAS → save display.
Subsequent questions and navigation read accepted identities. Recovery reconciles
existing records; scientific resume is a separate explicit runtime operation.

## C. Overlap audit and decisions

1. **Interaction status / SessionTurn status: distinct responsibilities with one
   confirmed duplicate.** Interpretation, clarification and answering are not Run
   phases. Scientific reservation/acceptance must survive missing display. Keep
   these phases separate; remove clarification-only SessionTurn writes and the
   unused production `record_no_run_turn` API. Old rows remain readable. Reserve
   execution and mark the interaction submitted in one metadata transaction.
2. **Session completion / RunStore lifecycle: necessary distinction.** A successful
   scientific Run can still lack verified reporting or displayed wording. Keep
   `run_succeeded` and `ready` as acceptance checkpoints; do not mirror every
   runtime phase into Session metadata. `_view` reads runtime progress directly.
3. **Navigation / active pointer / generation: justified materialized state.**
   Navigation preserves previous-active history, distinct from scientific parentage.
   The pointer is the current choice and generation protects concurrent selection.
   They are validated and published atomically, not independently synchronized.
4. **Captured snapshots / immutable history: justified capture, expensive repeated
   decoding.** Offered context, defaults/parameter focus, completion availability,
   predecessor ambiguity, selected registration and resource choices can change
   after capture. Reconstructing them from today's Registry or final history would
   silently change old meaning. Keep their bounded materialization. Reuse the
   fully validated immutable Session object only when the current file bytes are
   exactly identical; remove serialization followed by another historical decode
   during every write.
5. **Output locators in Run/Turn/Revision: necessary separate roles.** Run steps own
   genuine results; Turn selections describe requested acceptance; Revision locators
   pin accepted result members and retained ancestry. They are not three copies of
   scientific payloads. Existing agreement/lineage checks remain.
6. **Prerequisite origin / captured reference / current pending declaration:
   necessary temporal roles.** Origin captures the objective and initial inputs;
   follow-up capture pins the exact pending declaration and binding source. New
   registrations add explicit roles without replacing captured members. Keep these
   references and validators; a blanket deduplication would break continuation.
7. **Presentation / recomputable view: preserve history, derive status.** Exact
   displayed wording remains immutable. Steps, current active markers and status
   are projected. Do not regenerate historical explanations on reopen.
8. **Worker receipts / durable history: necessary short-lived boundary.** Receipts
   cover the interval before an interaction exists and bounded local scheduling.
   They never replay work after restart. The missing piece was truthful display
   of a captured turn whose processing lease is gone, not another job table.

## D. Implemented simplifications and migrated consumers

`ResearchAgentApplication.sessions` is now a lazily retained facade. A respond
call still receives its own facade for `_interaction_session_id`, sharing only
its FileSessionStore. Per-turn model applications share that store with their
interactive owner. Model profiles and language context are not cached globally.
One-shot use still creates no Session directory.

FileSessionStore keeps at most **one** immutable decoded snapshot. Every load
still acquires the metadata lock, checks root/path/file constraints and reads
bounded bytes. Exact equality to the previously validated bytes permits reuse.
Changes, even same-size edits with restored mtime, require normal checksum,
strict JSON, shape, identity and historical validation. A fresh process performs
full decoding. No mtime/generation-only trust or scientific-authority cache is
introduced. Concurrent writers remain serialized by the existing flock; writes
retain temporary-file fsync, atomic replace and directory fsync. Constructors
validate the new frozen state before serialization; the redundant decode-after-
serialize validation is removed.

New clarification writers store only Interaction. The legacy nonexecution
SessionTurn writer is removed. Historical clarification/failed/cancelled rows
remain supported by the existing reader. Tests now check that historical rows
remain readable and new clarification has no Run, Revision or SessionTurn.
Both admitted-plan consumers update execution reservation and submitted status
in the same atomic Session write, eliminating an intermediate inconsistent state.

Unused shared-model discovery is removed from `validate_submission`; completed
retries no longer discover models before their identity check. Discovery occurs
only for an actual new interpreted turn. It cannot make a completed retry depend
on unrelated library installation changes.

## E. Recovery, navigation and user continuity

Read-only status probes the **existing** processing lease without creating files.
If a captured interactive response is absent and no processor holds its lease,
the view reports `INTERACTIVE_PROCESSING_INTERRUPTED`. It re-reads Session state
after the probe to avoid misclassifying a just-published completion. Successful
accepted science remains successful; ongoing orphaned phases become visibly
interrupted. No Run state is rewritten and no provider/native tool is invoked.
The browser stops indefinite presentation polling and enables continued use.

The Web exposes the existing `recover_turn(..., complete_presentation=True)`
operation through **Recover saved progress**. Under the same lease, it can finish
application presentation for a successful Run or display deterministic completion
for an already accepted result. A pre-execution/answer interruption is explicitly
closed with an unavailable response, allowing a new message. It does not retry a
model answer or resume nonterminal native science. Existing cancellation arbitration
and explicit runtime resume remain unchanged. Missing Run reservations stay
truthful and are not silently resubmitted.

A browser-local recent-conversation selector retains up to 30 successfully opened
Session IDs and short labels. It offers no server-wide Session enumeration and
creates no identity system. Opening always reads authoritative server state.
The two-panel scientific results/conversation layout remains unchanged.

Conversation reads now accept an exact `before_turn_id` cursor, and the browser
can load earlier messages. The default 100-message page is no longer an access
barrier to older conversation. The selected active Revision remains in the bounded
result summary even when it falls outside the most recent revisions. Exact old
results also remain accessible through their historical message and revision API.

## F. Resource reuse and final library workflow

The R4 library lifecycle already has the required ownership boundary and is retained:

* Uploads are private registrations. Session source offers recover exact previous
  attachments; current-turn selected members retain their positions and identities.
  Accepted results use exact Revision/step references; reuse does not copy data to
  the library or require sharing permission.
* Discovery combines common installed references with only this Session's private
  references. Opaque handles map to labels in the UI. Selection ambiguity remains
  explicit. Consuming scientific owners check species/assembly/method/measurement
  compatibility; discovery does not globally qualify every resource for every task.
* Sharing requires an actual supported reference role and explicit `shared_use`.
  Existing `install_private` copies the required closure into library-owned storage.
  Stable operator installations may remain indexed in place. Minimal version,
  provenance and usage basis are recorded by the existing installer.
* Matrices, raw reads, task groups and ordinary experimental uploads cannot become
  common references through that action. Private-source loss after sharing does not
  remove the independently owned shared reference.

No sharing UX or library schema changes are needed. Selected-file integrity checks,
private isolation, ownership-copy tests and cross-Session reuse tests remain. The
only reuse-lifecycle change is removing discovery from validation/completed retries.

The broader consumer audit also found two historical-read defects, fixed without
changing scientific producers. Annotation rationale now passes the manifest's
actual schema version to the profile reader; accepted schema-1 replay profiles
are not compared against schema-2 completion profiles. Standalone/historical UMAP
results with no explicit clustering dependency no longer block unrelated questions
while annotation-group offers are built. They offer no grouping source. Explicitly
recorded but corrupt dependencies still fail closed. Tests retain tamper checks and
cover both annotation versions, descriptive UMAP comparison and exact R6 grouping.

## G. Retired machinery and retained readers

Removed: clarification-only SessionTurn production, `record_no_run_turn`, separate
post-reservation submitted-status writes, per-property Session facade/store creation,
write-time full historical decoding, repeated unchanged-file decoding, and unused
model discovery before completed retry/admission validation. Migrated consumers are
`sessions`, `turns`, `dialogue_execution`, `service`, `interactive`, Web API/static UI
and their owner tests. The two historical reader fixes additionally touch
`dialogue_evidence` and `annotation_groups`. Legacy fixture construction is explicit
test code.

Retained: current session envelope/schema, historical nonexecution rows, accepted
Run/Revision identities, R4 legacy resource readers, R6 versioned scientific readers,
atomic storage, leases, cancellation, exact evidence/download validation and the
R5 planner/answer interface. No old/new parallel production architecture is added.

## H. Verification and real product acceptance

Native Firefox read-only checks cover all four new accepted revisions, their reports
and three PCA scientific downloads. Initial reads preserve 68 workspace files;
a fresh independent Web process repeats them over 73 files after the explicitly
simulated fixtures below were added. Both phases make zero additional provider or
scientific production calls. Main Session generation is 7 with four revisions;
its chronological user/Agent text survives refresh and independent process restart.

Separate **simulated** fixtures seed 205 interactions for browser paging (410 user/
Agent messages) and an interrupted pre-execution interaction. Normal browser controls
load all pages, recover the interrupted response and survive refresh with zero
provider/science calls. These do not establish real scientific crash recovery.

The real journey makes **19 provider calls**: seven interpreter, six planner
(including the missing-file repair), four output-selection, one scientific answer
and one factual-review call. There are four accepted scientific executions: one
inspection and PCA at 8, 6 and 4 components, each with one persisted step attempt.
Entry instrumentation, enabled before the three PCA executions, records three
executor entries and three native Scanpy PCA calls. The earlier inspection is
accounted by its exact persisted successful Run, not retroactive entry tracing.
Questions, clarification, navigation, reads and restarts add no production.
AnnData parsing during input observation/download integrity is not counted as
native scientific production.

A separate fresh-process read of the original normal workspace preserves all 228
file hashes, historical reports and the original 79,569,234-byte annotated H5AD
(download SHA-256 `c94c784c53ed55d99015fdcb25fc82fbeb295526b926ba0b455acb1998ef763f`).
It also reads accepted legacy annotation rationale with providers, runtime and
scientific-owner verification forbidden. Pure Session decoding additionally checks
30 preserved normal/R6 records without changing them.

The broad development regression (`tests/application tests/web`) completed with
2,857 passed, 19 failed, 10 skipped and three warnings in 1,412.94 s. It was run
while the changes were still under review and is **not a clean final acceptance**.
The failures identified seven legacy annotation-detail reader cases, four standalone
UMAP dialogue cases, six obsolete clarification-checkpoint assertions (species and
raw input), and two instrumented Rich Web 20-second polling expirations after the
selection step had succeeded. All functional failures are addressed by the reader
fixes and consumer migration above. Polling deadlines are not changed; the Rich
cases are rerun serially below. The successful isolated guidance/branch check takes
47.10 s overall, including fixture setup, with its ordinary per-turn deadline.

Final focused runs (overlapping, not additive):

| Evidence | Scope | Result |
| --- | --- | --- |
| `final-owners.log` | Session/state/concurrency, interactive schemas/core, conversation context, scientific answers, versioned details, UMAP comparison, Web API/static and raw clarification | 265 passed, 42.34 s |
| `recovery-final.log` | Final recovery refinement including old clarification checkpoints, concurrent lease and exact retry tests | 73 passed, 22.61 s |
| `resources-lifecycle.log` | Shared ownership/cross-Session reuse/privacy, RunStore integrity and initial R7 cache/lifecycle tests | 37 passed, 28.19 s |
| `historical-readers-final.log` | Annotation detail, UMAP comparison and exact R6 grouping/projection | 39 passed, three warnings, 23.00 s |
| `final-rich-declarations.log` | Complete Rich Web, initial scientific declarations and interactive rich-result suites, including both previously expired polling cases with unchanged deadlines | 86 passed, 188.46 s |

The first development core run had seven obsolete expectations (six assertions
requiring clarification SessionTurns and one JavaScript fixture missing the new
recent-history rendering stub). They were migrated; the next core run passed
111 tests. The separate resource/lifecycle owner run passed 37 tests in 28.19 s.
These overlapping development counts are not added to the final regression total.

Native Firefox and the configured real OpenRouter `openai/gpt-5.6-sol:floor`
exercise Session `0a1fc693-547b-4605-8dda-f76a87b05c5e`. The preserved real PBMC
input is uploaded into the isolated workspace as `real-pbmc.data` (1,004 cells,
82,579 features). Inspection succeeds. A subsequent cited answer reports its
shape, CSR storage and nonzero count without scientific re-execution. The same
Session-private upload supplies PCA with eight components. Conflicting six/ten
component instructions produce a clarification without execution; the explicit
six-component follow-up succeeds. Switching to the original inspection and
executing four-component PCA creates a child of that exact historical Revision.
Navigation back and forward, creating another Session and returning through the
new browser-local selector preserve conversation and selected result.

There were two diagnostic setup/product outcomes before this successful path.
The first server launch omitted Conda activation, so provider construction failed
before capture; relaunch with the configured environment fixed the setup. A real
request to inspect without supplying any file was interpreted as execution; the
planner's missing-source repair ultimately returned unsupported. No tool executed.
This is preserved as an unsuccessful missing-file interaction, not counted as a
clarification pass or hidden through semantic repair by Agent. The subsequent
explicit upload/request succeeds. The live conflicting-parameter clarification
above is the actual clarification acceptance. This bounded test does not establish
general provider intent reliability or universal missing-file dialogue quality.

## I. Before/after architecture and limits

The functional path stays Planning → Orchestration → Scientific tools → verified
outputs, with Session-owned research continuity above it. The redesign removes
redundant records/transactions and repeats of pure metadata work; it does not
collapse scientific success into conversational success.

For an isolated 1,000-interaction, 213,044-byte synthetic Session, 20 repeated reads
had median 17.71 ms with the R6 store and 0.0836 ms with R7. Total repeated-read time
was 0.4002 s versus 0.00173 s. Cold loads still validate the entire document; those
single observations are not used as a performance comparison. This measures small
metadata history, not biological computation or all production Session shapes.

The 16 MiB Session file bound, 64 KiB submission bound and 256 KiB stored response
bound remain finite allocation controls. Writes still serialize the complete file
and validate new immutable history; this is not a claim of unlimited growth or
constant-time writes. The existing 16-source interpretation offer bound remains
explicit on overflow; users can choose exact attachments, with no silent recent
subset or first-match selection. Result context also remains bounded for providers.
No evidence from this bounded workload justifies replacing atomic files with a DB.

Recent conversations are local to this browser/origin, not an account-wide history.
A cleared browser can still reopen a known Session ID. Deployment authorization
remains the existing trusted laboratory/same-origin model; no multi-user discovery
or authorization mechanism is claimed. Interrupted native execution still needs
explicit supported runtime recovery; the UI recovery action never restarts it.
No new scientific method, annotation accuracy, EpiZoo capability or arbitrary
resource compatibility is qualified by this lifecycle acceptance.

## J. Final preservation, diff and disposition

**Initial Session-focused acceptance completed.** The Agent-wide follow-up below
supersedes this initial scope and owns the final R7 disposition. The
final focused suites pass, including every module affected by the broad development
failures. The broad development run is retained as a diagnostic, not relabeled as
a clean full regression. At this initial checkpoint, the live missing-file dialogue
failure was an integration defect requiring correction; the Agent-wide follow-up
below fixes it. The successful real clarification/continuation,
result reuse, branching and restart paths are independently recorded above.

`preservation-final.json` / `preservation-final.log` check all 262,453 protected
files: zero changed, zero missing, and zero added or removed historical evals.
The 1,478 baseline SHA-256 checks and 30 historical Session reads pass. This retains
the bounded metadata-versus-content distinction in section A. The independent
original-workspace read additionally preserves all 228 file hashes and the exact
historical report/download identities. Normal operator configuration, credentials,
common library and accepted historical data are unchanged.

`source-freeze-final.json` pins 563 source/test/static/resource and dependency
files; final comparison reports no changes after the final acceptance snapshot.
`git diff --check` passes and the index is empty. HEAD, local main, origin/main
and actual remote main remain the R6 baseline. The diff is limited to application
state/readers, Web consumers, affected tests, this record and README/AGENTS.
Scientific tools, orchestration/runtime, Registry and EpiZoo are unchanged.

The isolated acceptance server and Firefox have been stopped, with exact process
identity checked before shutdown (`stopped-processes.json`). Evidence and generated
acceptance inputs remain outside Git under the isolated directory above. No files
have been staged, committed or pushed. No subsequent milestone has begun.

## Agent-wide follow-up — expanded R7 scope

The expanded request reviews all Agent-owned production boundaries, retaining the
initial R7 changes and evidence. The earlier treatment of the missing-file failure
as merely model variability was insufficient: Agent had no initial empty-source
role handoff. That integration defect is corrected below. This follow-up was
completed within the same R7 milestone before Git closeout.

Evidence: `/home/likeyi/agent-ua365-r7-full-pogdirz3`. Initial HEAD, local main,
origin/main and independently queried remote main all remain
`5c5c1bab74b4260dca968e5b3107932bf6a8ae3b`. The initial R7 diff and newly created
R7 files were captured before this follow-up. Its protection baseline extends the
previous inventory to **262,703 files**, including the entire initial R7 evidence
directory. No protected historical record is migrated in place.

### A. Complete module and consumer inventory

`module-inventory.json` captures all **215 Python modules**: application 35,
orchestration 34, providers 6, reports 13, schemas 5, Web 4, root 2 and scientific
tools 116. It includes module purpose, definitions and imports.
`agent-owner-consumers.json` records the initial 99 Agent modules and import
consumers; `module-inventory-final.json` records final definitions and consumers,
including module imports inside functions. These are audit evidence, not runtime
catalogs. Static HTML/JavaScript/
CSS, pinned repository R scripts, CLI configuration, test discovery and benchmark
consumers were inspected alongside Python.

| Production modules / entry points | Actual responsibility and consumers |
| --- | --- |
| Root `__main__`, `application/cli`, `web/__main__`, package exports | One-shot CLI and explicit local Web launch; import exports are APIs, not another execution engine. |
| `application/service`, `workspace`, `schemas` | One-shot composition, output directories and compact application status; CLI and Session completion consume them. |
| `web/config`, `application/input_sets`, `providers/model_registry` | Operator-owned model/resource configuration, immutable factories, per-turn profiles; Web constructs the interactive facade. |
| `web/app`, static `app.js`, `index.html`, `styles.css` | Same-origin transport, bounded worker receipts, polling, upload/navigation/download controls; no semantic or scientific authority. |
| `application/interactive`, `interactive_schemas` | Exact submission/retry admission, model bundle selection, client-safe views, explicit recovery and pinned delivery. |
| `application/turns`, `turn_decisions`, `turn_context` | Captured user interaction, typed intent and exact declarations, context handles, dispatch to existing execution/navigation/answer owners. |
| `application/dialogue_execution`, `guidance_selection` | Scientific role/parameter prerequisites, scoped declarations, plan/output admission and explicit guidance-command evidence. |
| `application/raw_handoff`, `epizoo_handoff`, `resource_continuation`, `reference_supplementation` | Existing typed method declarations, exact file associations and resource acquisition continuations; native owners retain qualification. |
| `application/scientific_dialogue`, `scientific_guidance`, `dialogue_evidence`, `response_facts`, `responses` | Accepted evidence projection, cited prose and factual review, bounded advice and deterministic lifecycle text. |
| `application/session_state`, `session_store`, `sessions` | Conversation/history, exact immutable Revision locators, active selection and generation CAS; metadata never grants native authority. |
| `application/uploads`, `local_resources`, `shared_resources`, `shared_genome` | Bounded byte intake, private source identity, explicit common-library installation/discovery, private feature binding to a common genome. |
| `application/annotation_groups` | Exact accepted Leiden/source dependency projection into ordered TSV; never filename/count-based matching. |
| `application/matrix_delivery`, `table_delivery`, `analysis_delivery`, `annotation_delivery`, `model_delivery` | Pinned snapshots and format-specific privacy policies; interactive/Web downloads consume them. |
| `orchestration/planner`, `llm_planner`, `planning_model`, `planning_scope`, `semantic_prompt`, `semantic_wire_v4`, `_catalog_compaction` | Explicit deterministic CLI option or configured semantic Planner; Registry tool visibility, bounded provider wire, lossless prompt compaction. No model capability-selection phase in production. |
| `orchestration/planning_recovery`, `planning_diagnostics`, `error_policy` | Bounded plan acquisition/transport recovery, sanitized diagnostics, distinct tool retry policy; runtime consumes recorded policy identities. |
| `orchestration/semantic_compiler`, `active_context`, `prior_outputs` | Pure exact lowering of selected sources/dependencies, private historical identities, trusted accepted-output binding and integrity. |
| `orchestration/registry` and `annotation_registry`, `bam_fragments_registry`, `barcode_qc_registry`, `cell_selection_registry`, `external_fragments_registry`, `fragment_feature_registry`, `fragments_registry`, `matrix_adoption_registry`, `matrix_registry`, `species_adaptation_registry` | Immutable execution allowlist and owner-specific tool/result/port metadata; compiler, executor and evidence consume the same contracts. |
| `orchestration/runtime`, `executor`, `run_store`, `durable_tool_recovery` | Durable sequential execution, cancellation/lease arbitration, persisted checkpoints and exact published-step recovery. |
| `orchestration/verifier`, `verification_authority`, `target_model_binding` | Result checks, accepted-authority restoration and exact target-model handoff; no conversational interpretation. |
| `orchestration/raw_scatac_verifier`, `differential_accessibility_verifier` | Scientific verification implementations despite their directory: raw reconstruction and explicit/versioned independent DA verification. They are not generic admission stages. |
| `providers/openai_planning`, `groq_planning`, `gemini_planning`, `openrouter_planning` | Optional provider-specific transport/response semantics; no scientific routing or independent retry engine. |
| `report/evidence`, `visualization`, `analysis_report`; projections `annotation`, `bam_fragments`, `barcode_qc`, `cell_selection`, `external_fragments`, `fragment_features`, `fragments`, `matrix`, `matrix_adoption` | Reproducible evidence/figures/reports after accepted science; standalone verification APIs and bounded tool facts. |
| `schemas/orchestration`, `run_state`, `prior_output`, `verification_authority` and exports | Frozen domain/wire records, versioned readers, exact plan/source fingerprints; shared by all corresponding owners. |
| `tools/analysis`, `tools/data`, `tools/models`, `tools/scientific_authority`, cancellation helper and empty knowledge/package modules | Native scientific algorithms and publication contracts. Integration/call boundaries were reviewed; the 116-module inventory is not a claim to have requalified every algorithm. |
| `tests/application`, `tests/web`, unit orchestration/providers, integration/owner tests; `benchmarks/interactive`, `benchmarks/planner` | Active consumer contracts and isolated historical comparisons. Historical benchmark experiments must not remain live production planning APIs. Existing `evals/` is protected data. |

### B. Before/after responsibility map

```text
CLI / Web controls
  → Interactive admission + immutable Interaction capture
  → Interpreter: intent, offered source associations, explicit declarations
      missing files/roles → existing durable prerequisite → later upload/reply
      answer/navigation  → accepted evidence / Session selection (zero science)
      executable request → Planner: Registry tools and dependency selections
  → Compiler: exact arguments, dependencies and lineage
  → plan/output admission + atomic Session scientific reservation
  → Runtime / Executor / RunStore: preflight, sequential native calls, cancellation
  → scientific owner verification + accepted publication authority
  → Revision publication and active-generation CAS
  → evidence / figures / report / cited response / safe downloads
```

Before this follow-up, the role branch could represent incomplete associations only
after at least one candidate file existed. An empty initial request escaped into
planning, creating a failed planning Run and allowing a repair call to change its
classification to unsupported. Afterward, the same role branch represents zero,
partial or complete candidate membership. It requests inputs before any planning
Run. No new state type, workflow coordinator or storage system is introduced.

### C. Complexity decisions and guarantees

| Classification | Mechanism, concrete consumer and decision |
| --- | --- |
| SIMPLIFY / MERGE | Initial empty-source admission joins the existing `RequestRoles`/`RoleAnswer` lifecycle. `turns.respond`, durable replay and upload continuation share `_role_completion`; no second missing-upload state machine. |
| MOVE / REMOVE | `selection_request` and `parse_selection` have no production callers after R5. Move the historical experiment to `benchmarks/interactive/legacy_selection.py`; migrate its harness and remove obsolete production-test imports/assertions. |
| KEEP | Interpreter selects intent/targets; Planner selects actual tool dependencies. A target is not an executable workflow. Removing either responsibility would require language heuristics or give interpreter decisions direct execution authority. |
| KEEP | Output selection follows the actual plan. It determines user-visible retained results, distinct from workflow construction; existing 1–32 bound, exact keys and stale-descendant checks remain. No automatic output fanout. |
| KEEP | Compiler lowers semantic sources; preflight validates arbitrary `AgentPlan` callers and prior references; execution resolves newly produced values. The latter facts do not exist at compilation. |
| KEEP | Candidate preflight enables bounded planning repair. Execution preflight runs after application/provider callbacks and across public executor/resume entry points; it checks mutable historical integrity. A bare cached boolean would lose that guarantee. No new trust token/cache is introduced. |
| KEEP | Registered result validation, native publication verification and Run acceptance have different subjects. R6 accepted-authority reuse already prevents covered upstream reconstruction. No universal native replay is restored. |
| KEEP | Planning recovery remains at most one local repair/transport action plus explicit configured failover, with total provider-call bounds. A malformed plan is reparable; unknown capability, authentication failure and cancellation are different outcomes. Empty-file requests now avoid this loop entirely. |
| KEEP | `DeterministicPlanner` is still an explicit CLI/library capability and is used by offline native integration tests. Configured interactive applications never use it as an LLM fallback. Deleting it would remove a current public consumer, not just historical machinery. |
| KEEP | Explicit v3 wire callers, current v4 compilation and historical diagnostic/policy readers remain. There is no evidence that all v3 consumers have migrated; no silent wire reinterpretation. |
| KEEP | Typed raw/annotation/EpiZoo declarations encode method-dependent measurement, barcode, reference and grouping requirements. They share source/continuation mechanics but are not interchangeable scientific contracts. |
| KEEP | Numeric/exact-string declaration grounding preserves explicit values and scope; it does not repeat generic intent classification. M17.2 command evidence is limited to persisted guidance-candidate selection. |
| SIMPLIFY | Initial R7 exact-byte Session snapshot, atomic reservation and removal of clarification-only SessionTurns remain. This follow-up also covers zero-source clarification. |
| KEEP | Interaction, SessionTurn, Run lifecycle and presentation status remain separate temporal facts. A failed response cannot erase accepted science; worker receipts cover only pre-capture scheduling. |
| KEEP | Session snapshot versus active pointer/generation: immutable offered context is needed for retry/continuation; generation CAS prevents stale work from replacing a newer selection. |
| KEEP | Submission normalization at HTTP worker admission and again under the application lease protects different concurrency windows. Stored exact retries return before providers/library discovery. |
| KEEP | RunStore writes/checksums/leases and SessionStore locks/fsync are distinct authority boundaries. No shared mutable global cache or new queue is justified. |
| KEEP | Prior-output references pin run, step, port, authority and accepted-step digest. Revision selections are navigation records, not another copy of scientific payloads. |
| KEEP | Evidence, report and client-safe facts are projections with different intended consumers; report APIs can be called standalone. Exact displayed history is immutable, not regenerated evidence. |
| KEEP | Delivery adapters differ by legitimate privacy/format rules. They share the pinned snapshot owner; combining their scientific field policies into a generic export would expose or silently drop meaningful metadata. |
| KEEP | Upload byte identity and local-resource admission do not claim scientific compatibility. The selected native tool owns qualification. Current file aliases and filename suffixes do not establish eligibility. |
| KEEP | ScientificResourceLibrary installation, private registration and selected-file integrity remain separate. Explicit shared intent is necessary; repeated private data use needs no library copy. |
| KEEP | Provider adapters preserve distinct API/response/refusal/identity semantics. Common typed profiles and error classification already have shared owners; no new generic transport abstraction is justified. |
| KEEP | Closed schemas, allocation limits, safe error projections and bounded diagnostic traces protect persistence, privacy and client behavior. Internal exceptions are not returned as model-facing evidence. |
| HISTORICAL READ ONLY | Existing nonexecution SessionTurns, accepted annotation/feature/DA versions, old library deployment records and scoped recovery-policy records remain version-aware readers. No obsolete production writer is restored. |

### D–E. Implemented corrections and execution boundaries

`RegisteredInputGroup` now permits zero members only for an attachment-candidate
group. Ordinary selected groups still require members. An empty group is a pending
source set, never a scientific source or executable role association. Its path and
identity protections are unchanged. `available_input_view(..., allow_empty=True)`
is explicitly used at initial role capture; ordinary continuation with no newly
selected attachment keeps the captured group instead of replacing it with empty
membership.

The interpreter schema exposes existing role requests even when `selected_sources`
is empty. Raw/annotation/EpiZoo variants use their existing declaration contracts
with null source fields; empty source enumerations are removed from those schemas.
Prompt guidance distinguishes missing files from unsupported capability. It does
not invent a workflow, choose a file or infer scientific qualification.

For a typed `Execute` with an offered target but no inputs/selected result context,
`missing_file_request` reads that target's required offered Registry file roles.
It creates the same role prerequisite before Planner/application execution.
This deterministic check does not inspect utterance phrases, select upstream tools
or reclassify a genuinely unsupported decision. Supplied structured inputs,
nonempty candidate sets and selected result context retain their normal paths.
The direct Session API's separately supplied `execution_inputs` is also respected.
An overflowed Session input offer remains ambiguous; it is never treated as an
empty source set or silently narrowed to a first file.

Previously captured sources retain order and identity; later selected registrations
append through `_role_binding`. Ambiguous alternatives are interpreted explicitly.
Source type/byte integrity and native scientific compatibility remain unchanged.
Registry file roles cover the existing reusable upload consumers; special compound
raw/reference inputs continue through their typed handoffs. This is not a claim
that every arbitrary manifest can be uploaded as an interchangeable dataset.

The retired capability-selection prompt/parser now live only with their actual
historical benchmark consumer. Registry visibility/fingerprints and explicit
capability-scoped benchmark experiments remain; default production planning still
offers the Registry directly in one semantic call. Runtime, compiler, executor,
native algorithms and their scientific checks require no redesign for these fixes.

### F. Missing-file continuation

1. A supported inspection request arrives with no file.
2. Interpreter identifies the offered operation/role, or deterministic admission
   detects the required file role for its typed Execute.
3. The original utterance, declarations, model profile and pending role are stored
   in the Interaction. No SessionTurn, Run or Revision is created.
4. A later upload plus “Here is the file.” selects the offered exact source.
5. Existing `RoleAnswer` admission binds it and passes the original objective to
   Planner; it does not treat the short reply as a new analysis objective.
6. One accepted scientific execution publishes its result. Exact retries,
   historical reads and restart do not replay it.

Changing the objective starts a separate prerequisite rather than inheriting the
old task's sources. Missing parameters/resources, incompatible files, access errors
and unsupported capabilities retain distinct owners and outcomes.

### G–H. State/resource conclusions and consumer migration

Initial R7 Session caching, recovery, recent-session navigation, history paging,
versioned annotation rationale and historical UMAP reader fixes remain. No
additional cache or persistent state schema is needed. The Web continues through
the existing interactive facade and two-panel UI; there is no new technical form.

The unified library is unchanged. New private uploads remain private; exact
Session-local reuse uses captured registrations or accepted references. Shared
installation still requires explicit use intent and copies/indexes the existing
qualified closure. No new library catalog, promotion stage or universal resource
qualification has been added.

Migrated consumers are role projection, typed decision schema/prompt, initial
admission, captured role replay, raw schema projection and the benchmark harness.
Tests that previously asserted absence of any new prerequisite for a supported
unrelated empty-input task now assert its **new objective**, empty membership,
no inherited continuation and zero science. Historical records retain their
original bytes; only future missing-file turns use the expanded existing contract.

### I. Verification and real-product evidence

Final focused results overlap and must not be added into one unique-test count:

| Evidence | Coverage | Result |
| --- | --- | --- |
| `affected-boundaries-final.log` | Role continuation, Session lifecycle/core, initial declarations, Web API/upload reuse, compiler, bounded planning recovery, RunStore, cancellation and all benchmark tests | **715 passed, 170.69 s** |
| `resources-admission-final.log` | Empty-file and compound-role Web admission, initial declarations, local resource integrity, shared/private library and uploaded-reference owners | **303 passed, 74.36 s** |
| `admission-final.log` | Empty/source-role contracts, role continuation, immutable history, initial Session execution and recovery | **111 passed, 25.99 s** |
| `final-source-admission.log` | Final overflow distinction, Registry requirements across inspection/PCA/evaluation/transfer, typed raw/annotation/EpiZoo missing files, exact upload continuation and Session reuse | **69 passed, 18.65 s** |

The 715-test run preceded the final narrow overflow guard; the last row covers
that exact refinement. No opt-in `RUN_*` gate was enabled. These suites use scripted
semantic decisions and small scientific fixtures; they do not qualify provider
reliability or biological methods. The existing genuine small Leiden→evaluation
Web test also passed in the development role suite, preserving one native call
per step, exact source association, evidence, retries and fresh-process reads.

Development failures are retained, not waived: the first empty-view implementation
incorrectly replaced a continuation's unselected captured group, fixed by an
explicit initial-only empty view (50 subsequent tests passed). A proposed schema
test used an unavailable `jsonschema` dependency; it was replaced by a bounded
empty-enumeration check without adding a dependency. The extended role suite then
had 205 passes and one obsolete assertion requiring no new prerequisite for a new
inspection objective lacking an input; the updated assertion checks a new
objective and no inherited files/continuation. The first broader run had 711 passes
and four failures: three direct Session submission-shape regressions and one
literal prompt-text assertion. Actual input handling was fixed; the text assertion
now tests offered execution/omitted-parameter semantics. The clean final runs above
cover those failures. No timeout or scientific assertion was weakened.

The live test uses native Firefox and the unchanged configured OpenRouter
`openai/gpt-5.6-sol:floor`, through normal Web controls in isolated Session
`e0d177ad-cb67-4ba1-a416-201f5b02f7c2`:

* “Inspect my scATAC matrix and tell me its dimensions.” with no file produces a
  durable `request_roles` clarification. Generation stays zero; no SessionTurn,
  Run or Revision exists for it. There is one interpreter call and no Planner.
* Uploading the preserved real PBMC `real-pbmc.data` and replying only
  “Here is the file.” produces `answer_roles`, one normal planning/output-selection
  sequence and one accepted inspection. Its Run request contains the exact original
  objective, not the short upload reply. Accepted Revision:
  `dc69f99bf400d954497081f4eb7d999bf7953bd00d2ac14f0551aa8b1d23d560`.
* A subsequent result question answers **1,004 cells and 82,579 features** through
  accepted citations and factual review, with no scientific execution.
* Requesting four-component PCA followed by a five-neighbor graph reuses the same
  Session-private upload without another upload or library installation. One plan
  executes `pca_embed_cells` → `build_cell_neighbors`; accepted Revision:
  `096e28843fb41de976ed3c8ac39ec57176379a2c8e91878ce9f028da4f6ab3c8`.

`live-accounting.json` records **10 provider calls**: four interpreter, two Planner,
two output-selection, one scientific answer and one factual-review call. There
are exactly two successful Runs, with **one persisted attempt for each of the
three scientific steps**. Entry tracing records two executor entries, one native
PCA and one native neighbor call. Inspection is accounted by its accepted persisted
step; the imported Registry callable was not intercepted by the module-attribute
inspection wrapper. Seven AnnData reads include observation/delivery work and are
not seven scientific executions. No planning repair or failed planning Run occurs.

`browser-reads.json` and `browser-restart.json` cover both revisions' report access,
available scientific downloads, historical result inspection, refresh and a new
independent server process. Each preserves all **33 workspace file hashes**, adds
no files and invokes zero additional providers/native production. Eight conversation
texts, generation 2 and both exact revisions survive. The initial R7 live
branch/navigation evidence remains applicable and is not counted as repeated work.
The Web and Firefox processes used for this acceptance are now stopped.

A separate fresh-process read of the original normal workspace preserves all
**228 file hashes**, accepted legacy annotation rationale, original reports and
downloads with provider construction, runtime execution and scientific-owner
verification forbidden. `read-only-result.json` and `historical-read.log` retain
this independent historical check.

### J. Follow-up closeout

**COMPLETE within the expanded R7 engineering and bounded product scope.**
All Agent-owned module families and their consumer boundaries have been reviewed.
The initial state simplifications remain, the missing-file integration defect is
corrected, and retired semantic-selection APIs have left production. The review
does not require replacing correctly owned contracts merely to enlarge the diff.

`preservation-final.json` checks **262,703 protected files**, including the first
R7 acceptance directory: zero changed, zero missing, zero added/removed historical
evals. **1,718** bounded files have SHA-256 comparisons; the remainder use size,
nanosecond mtime and mode. Thirty original normal/R6 Session records decode
unchanged. This is not a claim to have rehashed every large dataset or checkpoint.
Normal operator configuration, credentials, installed models, shared library,
accepted Run/Session/Revision bytes and previous acceptance workspaces are preserved.

`final-source-freeze.json` pins **594** source, static, test, benchmark and dependency
files after the final overflow refinement. Its preceding snapshot is retained
separately; the final affected source check is explicit. Git diff review covers
all original and follow-up R7 changes. Scientific algorithms, Registry inventory,
EpiZoo, compiler, executor and RunStore are unchanged; the only orchestration module
change removes the retired selection prompt/parser. No production import depends
on its benchmark replacement.

Remaining bounds are explicit: provider decisions are not universally reliable;
only the described live missing-file/upload/answer/multistep path is qualified.
Compound raw/annotation/EpiZoo empty-input handling has scripted admission coverage,
not new live raw-science qualification. Selected sources still require native
compatibility; unrelated or ambiguous sources are never guessed. Existing Session
size, source-offer, history-page and local deployment limits remain as described
above. Native crash/recovery and expensive R5/R6 scientific methods were not
requalified by this work. No unresolved failure from the affected final suites
is hidden as a historical exception.

At follow-up acceptance, HEAD, local main, origin/main and actual remote main
remained the expected R6 baseline. `git diff --check` passed and the index was
empty. Nothing had been staged, committed or pushed at that checkpoint.

## Final engineering review and Git closeout

The final review covers all **36 R7 changed files**, including the original
Session/lifecycle implementation, Agent-wide follow-up, three new files,
benchmark migration, tests, documentation and Web assets. No substantive blocker
was found. Session conversation/history, Interaction intent/display, SessionTurn
scientific reservation/acceptance, RunStore execution, Revision outputs/parentage
and generation-checked Navigation retain their distinct authoritative roles.
Interpreter intent, Registry planning, compiler/preflight binding, native
scientific qualification and accepted-evidence presentation remain at their owners.
Production has no dependency on the retired semantic-selection APIs or their
benchmark replacement. Empty candidates, selected sources and source overflow
remain distinct; clarification preserves the objective without reserving science.

All **594** files in the final accepted source snapshot still match exactly.
The focused test logs and both live/browser acceptance records were checked,
including the final 69-test overflow coverage and the documented development
failures. No new code change or concrete test-coverage gap was found, so no
additional provider, native analysis or full regression was run. Final edits only
clarify documentation authorization and historical checkpoint wording.

Before staging, branch `main`, HEAD, local main, origin/main and independently
queried remote main matched the expected R6 baseline; the index was empty and the
working-tree inventory matched the accepted R7 record. Preservation was rechecked
against the existing **262,703-file** baseline, with **1,718 SHA-256** comparisons
and size/nanosecond-mtime/mode checks for the remainder. Historical eval membership
and all 30 historical Session decodes remain unchanged. Both acceptance directories
also have a separate final-review preservation snapshot. Evidence and Git
verification are retained outside the repository at
`/home/likeyi/agent-ua365-r7-closeout-657yiter`.

The explicitly authorized closeout is one commit named
`Complete UA3.6.5-R7 Agent internal architecture and lifecycle simplification`,
followed by normal push and independent remote-SHA, index, tracked-worktree and
protected-file verification. Only the reviewed 36 files belong in that commit;
historical `evals/`, credentials, data, models and acceptance artifacts are excluded.
The final task report records the actual commit and push outcome. No subsequent
milestone is included.
