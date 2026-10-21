# UA3.6.2b — Unified conversational state-to-planning integration

Implemented on 2026-10-12 against
`a856652aa1ef7fa2110ca27c542c66ddf62e991a` on `main`, the actual accepted
UA3.6.2a commit. This connects ordinary interpreted `execute_plan` turns to the
existing captured Session context, historical bindings and explicit retention.
Neighbors and raw QC/selection/matrix are representative witnesses of that shared
connection. Embedding, clustering and UMAP publications gain no new authority.

## A. Baseline and preservation

Admission verified HEAD, local `main`, existing `origin/main` and remote `main`
at the same commit above; remote verification used `git ls-remote`, without
fetching. The UA3.6.2a implementation and acceptance document were present. The
tracked worktree/index were clean and only `?? evals/` was untracked. No reset,
clean, fetch, pull, staging, commit or push is performed.

The preservation directory is `/tmp/agent-ua362b-preservation-9f_mb02k`.
`protected-before.json` inventories 2,882 protected files: 1,274 existing evals,
198 outputs, 208 preserved real-state files, one operator configuration,
23 UA3.5.5 scientific-state files, seven preparatory files, two human inputs,
four EpiZoo auxiliary resources, one checkpoint and 1,164 prior UA3.6.2a acceptance
files. Ordinary files are pinned by SHA-256, byte count, modification time and
mode. The 5,231,645,507-byte checkpoint is checked by metadata only; it is never
loaded or rehashed. `tracked-baseline.json` pins all 604 tracked baseline files
and `index-before` preserves exact index bytes. New scientific fixtures, Runs,
Sessions, evidence, reports, cache directories and live diagnostics use `/tmp`.

`preservation-final.json` reports **zero mismatches across all 2,882 protected
records** after live and domain acceptance. Evals and output inventories remain
exactly 1,274 and 198 files. All three local refs and index bytes are unchanged.
Of 604 baseline tracked files, 595 remain byte-identical; the nine intentional
tracked changes are seven shared production files and the two narrowly corrected
test files described below. The four new test modules and this document are new
files. `implementation-final-freeze.json` pins all 507 current source/test/benchmark
files; every byte remains unchanged across final acceptance. The earlier freeze
is retained, with its sole intentional test-deadline delta recorded separately.
No historical evaluation, dataset, publication or acceptance document is rewritten.

## B. Verified integration gap and reused architecture

At baseline, `dialogue_execution.execute` initialized its planning context to
`None`; selected M17.2 guidance candidates could explicitly supply one. The
ordinary `execute_plan` branch did not supply the captured current Revision's
qualified outputs. The existing specialized selection/matrix path and
`sessions.run(..., use_active_context=True)` already supported historical
bindings. Neither was replaced or copied.

The connected path remains:

```text
InteractiveAgentApplication.submit_turn
  → AnalysisSessions.respond → turns.respond
  → existing typed Interpreter decision and admission
  → dialogue_execution.execute + planning_context
  → existing scoped Planner → semantic compiler/preflight
  → PriorOutputRef / StepOutputRef → registered tools and verification
  → existing RunStore completion → SessionTurn → child AnalysisRevision
```

`AnalysisSession`, immutable `Interaction` capture, `SessionTurn`,
`AnalysisRevision` and `FileSessionStore` remain the sole conversation-state
records. No new state manager, memory store, workflow engine, result cache,
interpreter, compiler or scientific authority system is introduced.

The accepted boundary records remain [M15.3](m15.3-prior-output-binding.md),
[UA3.5 closeout](ua3.5-closeout.md),
[UA3.5.3 binding](ua3.5.3-unified-scientific-input-binding.md),
[UA3.6.1 declarations](ua3.6.1-shared-string-scientific-declarations.md) and
[UA3.6.2a processed reuse](ua3.6.2a-generalized-processed-output-reuse.md).

## C. Shared implementation seams

| File | Change |
| --- | --- |
| `application/sessions.py` | Build existing `ActivePlanningContext` from a persisted pre-execution Interaction, its exact base Revision, generation and captured locators. Return immutable handle-to-locator groups without creating a SessionTurn. |
| `application/turns.py` | Add the bounded recorded planning inventory to the existing Interpreter projection when qualified artifacts are available. |
| `application/turn_decisions.py` | Add generic `Execute.context`, with `none` and `current`; conditionally expose the choice and instructions. Existing decision wires default to `none`. |
| `application/dialogue_execution.py` | Admit and propagate that context through ordinary scoped planning; revalidate after output selection; atomically check the captured base/generation; select explicit retained locators through the existing SessionTurn field. Preserve context through exact pending declarations. |
| `orchestration/prior_outputs.py` | Classify valid accepted nonreusable outputs separately from corrupt recorded authority. Preserve the qualified owner DAG; follow exact argument references for selected reporting outputs lacking reusable authority. |
| `orchestration/verification_authority.py` | Share the accepted-step predicate without granting authority. Preserve the default Registry authority boundary. Handle the demonstrated owner-context recovery failure through the existing durable runtime boundary. |
| `orchestration/runtime.py` | Persist a failed owner-proof restoration with pending work skipped, without production, proof reconstruction or fallback. |

Four new test modules cover typed context decisions, captured context and retention,
ordinary processed continuation, and ordinary raw continuation. A pre-existing
neighbors cancellation fixture now invokes its actual executing registered
producer, rather than substituting an unrelated standalone publication; its
original cancellation assertions remain intact. The raw four-owner Web fixture
gets an explicit bounded 60-second test-only completion wait after its existing
20-second deadline also fails at the unchanged baseline; the generic helper
default remains 20 seconds. No Web production behavior or acceptance assertion
is weakened.

Registry remains 23 tools and 172 public arguments. Scientific owners, algorithms,
defaults, EpiZoo, source declarations, compiler, semantic ports, provider adapters,
Web source, upload and download contracts are unchanged.

## D. Ownership, capture and scope

The LLM interprets whether the utterance requests an independent task, computation
using the captured current analysis, an answer, navigation or an exact pending
continuation. `context=current` expresses that general execution distinction;
`context=none` preserves independent execution. No word such as resolution,
neighbors, QC or EpiZoo triggers a deterministic conversational recipe.

Agent resolves qualified outputs from the exact persisted Interaction's captured
base Revision and generation. It checks terminal successful source Runs, accepted
step anchors, recorded/explicitly qualified authority, unique registered producer
ports and complete semantic members through existing binding functions. Cooutput
aliases share one planning handle while every named locator remains explicit.
Empty Sessions and accepted unqualified reports contribute no executable handles.
A malformed stored authority or qualification never becomes ordinary absence.

Interpreter projection uses metadata checks only. The offered records describe
historical qualification, not current byte integrity or execution readiness.
Questions and navigation do not hash or reconstruct scientific artifacts merely
to show that inventory. Contextual execution performs the existing strict authority,
publication and dependency checks before planning and again after the last model
call. Corrupted eligible bytes remain an execution error.

Providers receive opaque `ctx.N` and `retain.N` handles with registered semantic
metadata, not paths, private Run IDs, authority digests, arbitrary Session objects
or unrestricted conversation history. The semantic Planner selects an exact
offered handle; the compiler validates and lowers it to existing `PriorOutputRef`.
Execution resolves the exact
accepted value. Same-plan inputs remain `StepOutputRef`. No path search,
first-match lookup, implicit alignment, step-order lineage or upstream recomputation
is added.

This stage's execution scope is the captured current Revision. A user can use an
older Revision through existing explicit navigation and then submit a new turn;
there is no new whole-history execution inventory. Existing scientific-answer
historical references remain separate. Current-turn inputs and declarations still
use their existing admission and canonical scalar/string binders. Historical
parameters and source-specific declarations are not copied into new inputs.

## E. Explicit output retention and consistency

Ordinary contextual output selection offers actual new plan outputs plus bounded
opaque handles for qualified captured upstream locators. Its conditional version-2
response contains `outputs` and `retain`; `retain=[]` is allowed. Version 1 stays
unchanged for independent execution and selected guidance. No outputs or steps are
automatically appended.

Agent rejects unknown/duplicate handles, duplicate or colliding names and more
than 32 combined outputs. Selected historical locators must be exact captured-base
members. Selection is recorded in existing `SessionTurn.retained_outputs`; normal
completion includes each exact locator in the child `AnalysisRevision.outputs`. Multiple
aliases are distinct named offers; Agent does not choose one by first match.
Historical names are not exposed as arbitrary prompt text, so a conflicting new
label is rejected rather than renamed silently.

Existing owner-qualified DAG consistency remains authoritative. For new accepted
reporting results without reusable owner authority, the supplemental consistency
check follows only actual persisted `PriorOutputRef` and `StepOutputRef` arguments,
validates exact resolved argument identities and checks their qualified upstream
closure against active publications. Control dependencies and literal paths do
not imply historical lineage. Qualified publications use their owner-described
DAG without duplicating this reporting-result traversal.

This permits an explicitly retained graph plus new verified Leiden/UMAP reporting
outputs while preserving their nonreusable status. Retaining graph N1 alongside a
new result whose explicit references consume incompatible N2 is rejected. It
never mints clustering/UMAP authority. Completion and activation continue through
existing application checks and generation protection.

## F. Genuine representative execution

Scenario A uses UA3.6.2a's real registered neighbors fixture: 32 explicitly ordered
Unicode cell IDs and a seed-17, 32 × 512 float32 embedding with two separated
groups. The source Run is accepted by normal `sessions.run`. A separate ordinary
interactive turn receives the utterance:

> Change the clustering resolution to 0.5 and update the UMAP.

Scripted semantic decisions exercise the actual Interpreter contract, scoped
Planner, compiler, binder, preflight, registered Leiden and UMAP bodies, verifiers,
RunStore and Session completion. The compiled cluster input is the exact accepted
neighbors `PriorOutputRef`; UMAP consumes the new cluster `StepOutputRef`.
Resolution is exactly 0.5 with its utterance literal/span. A new child Revision
activates; the source Revision and publication/input bytes remain unchanged.
Accepted evidence exposes 32 cells and two UMAP components. Both explicit graph
retention and `retain=[]` succeed. A later mixed Revision offers only its qualified
graph for another downstream modification, omitting its unqualified Leiden/UMAP.

| Actual observed body | Ordinary continuation |
| --- | ---: |
| Neighbors production / independent owner reconstruction | 0 / 0 |
| Source capture/check/embedding/cell-ID loads | 0 / 0 / 0 / 0 |
| Leiden / UMAP production | 1 / 1 |
| Semantic compilation / preflight | 1 / 2 |
| Existing clustering / UMAP verifier body calls | 15 / 15 |
| Graph structural validation / analysis H5AD reads | 66 / 64 |
| Shared file closure, without / with retained graph | 18 / 24 |

Repeated normal verifications, integrity checks and artifact reads are recorded as
such; they are not neighbors production or scientific reconstruction. Existing
UMAP's installed backend can prune tiny connectivity weights in its new output
copy. Leiden output preserves the source graph values; both outputs preserve
embedding values and cell order, and accepted source bytes remain unchanged.
No unchanged UMAP contract is strengthened to promise
identical connectivity weights in its new file.

The directly emitted witnesses and exact model captures are under
`/tmp/agent-ua362b-processed-final2/test_ordinary_resolution_chang{0,1}`.
`conversational-continuation-witness.json` records identities, exact resolution and
actual-body counters; `conversational-model-captures.json` stores prompt/schema
bytes and measurements. The retained witness's source authority is
`9fafcfba5c7229180135064f7f88f973fc73c3b61857a048977a5dfd0fa085c8`.

Scenario B uses existing qualified tiny BAM and external-fragment fixtures with
accepted fragments/QC/selection/matrix. Ordinary interactive `execute_plan`
explicitly uses accepted QC to select cells and optionally rebuild the canonical
matrix from accepted fragments plus its new same-plan selection. It uses the same
context/Planner/binding/retention path, not the older specialized selection follow-up.
The exact fixture-only declarations are depth ≥1 and TSS enrichment ≥0; these are
not empirical defaults. The existing binder yields canonical integers and retains
verbatim declaration spans. Only fragments and QC are explicitly retained; the
old selection/matrix are not retained as stale descendants. New selection and,
when requested, matrix production/independent verification run once each; all
covered upstream production and owner reconstruction are zero. Fresh-process
reopen/duplicate retries preserve state and source bytes without new providers or
science.

## G. Fail-closed and recovery behavior

Focused cases cover missing/corrupt authority or qualification, changed publication
bytes, exact accepted-record targeting, foreign/empty Sessions, unoffered context
handles, incompatible explicit sources, forged resolved-reference identities,
incompatible retained upstreams, ambiguous language clarification, invalid
retention selections and stale generation/base races after capture.

Read-only scientific questions remain answers. Independent new input inspection
has no active historical inputs or inherited declarations. Navigation and duplicate
submissions add no scientific work. A missing raw threshold remains an exact
pending original objective; its later explicit answer keeps the original captured
context without inheriting older thresholds. Changing the Revision/generation
cannot redirect that pending operation or activate an invalid child. Refresh and
fresh-process replay with archived original embedding inputs execute zero
production, owner reconstruction, source checks and model calls.

A broad affected check discovered two genuine baseline lifecycle incompatibilities,
reproduced at the committed UA3.6.2a SHA from an isolated `git archive`. Missing
neighbors publication bytes caused owner-context restoration to raise before the
runtime's ordinary completed-step error boundary. The narrowly repaired hook route
now records `PERSISTED_STEP_REVALIDATION_FAILED` and skips pending work; it does
not run without authority, retry science, reconstruct proofs or qualify legacy
outputs. Existing raw and prior-reference restoration paths retain their contracts.
The second cancellation test supplied a standalone publication from a different
output directory with no producing execution identity. Its fixture now calls the
genuine executing producer before requesting cancellation, preserving the original
successful-producer/skipped-consumers assertion. No owner contract is weakened.
Baseline reproduction records are under
`/tmp/agent-ua362b-baseline-milestone6-9y5s5bb1`.

## H. Validation, provider qualification and payload sizes

Domain runs use conda `agent`, `PYTHONPATH=src:.`, disabled `RUN_*` gates,
no pytest cache and isolated `/tmp` cache/output directories. Earlier direct
processed acceptance uses Python `-B`, `PYTHONDONTWRITEBYTECODE=1`,
`-p no:cacheprovider` and a `/tmp` basetemp; it does not explicitly override the
Numba/Matplotlib cache environment. No full 7,000-plus
suite, real PBMC preprocessing, foundation inference, adaptation training or
native browser redesign is run. Exact domain command records and stdout/stderr
are stored by `/tmp/agent-ua362b-run-selection.py` in each listed workspace.

The focused typed-decision suite passes 18 cases. Initial existing application,
binding, output-selection, pending continuation and prior-planning checks pass
370 cases in 87.35 seconds. Captured-context acceptance passes 19 cases in
36.67 seconds; processed acceptance passes 17 cases in 59.28 seconds. These
selections overlap later domains and must not be added as unique totals.

Completed domain selections (overlap with focused reruns is reported separately):

| Selection | Result | Complete command/log workspace |
| --- | --- | --- |
| Unit orchestration, after lifecycle repair | 938 passed; three existing warnings; 205.57 s | `/tmp/agent-ua362b-orchestration-final-5o6bcdvc` |
| Existing raw prior-output, qualification and context | 96 passed; 799.06 s | `/tmp/ua362b-existing-authority-context.log` |
| Remaining authority/application/dialogue/guidance, processed prior outputs and Chromap authority | 184 passed; three existing warnings; 1,524.10 s | `/tmp/agent-ua362b-authority-dialogue-jbzsdfgc` |
| Semantic-v4/scoped quality/planner benchmark replays and frozen full-size boundary | 116 passed; 89.52 s | `/tmp/agent-ua362b-semantic-planning-08uc_e07` |
| Focused owner recovery/cancellation boundary | Five passed; three existing warnings; 11.41 s | `/tmp/agent-ua362b-lifecycle-boundary-lx0e_cti` |

The final full Application/Web invocation records 2,281 passed, nine gated skips
and one raw multi-owner 20-second polling timeout in 713.01 seconds
(`/tmp/agent-ua362b-application-web-final-brsf8uor`). The rich threshold case passes
in that full run. With no other scientific pytest process running, the unchanged
UA3.6.2a baseline reproduces that same raw deadline failure and passes the rich
case in 59.85 seconds (`/tmp/agent-ua362b-baseline-web-quiescent-ongh63nf`).
Quiescent current reproduces the same raw timeout and passes the rich case in
60.45 seconds (`/tmp/agent-ua362b-current-web-quiescent-i5q_gxdr`). Bounded
post-deadline inspection confirms both comparison Runs succeeded, all four steps
succeeded, generation 1, one activated Revision and persisted presentation.
Final Session writes occur about 20.600/20.527 seconds after Run creation in
baseline/current; these are filesystem observations, not universal latency claims.
Evidence is `/tmp/ua362b-web-quiescent-postdeadline-state.json`.

Only the initial four-owner test's wait is raised to 60 seconds; generic default
and duplicate/replay waits remain 20 seconds. The entire affected external-fragments
Web module then passes **14 tests in 35.28 seconds**, with all original scientific,
provenance, immutable-history, download and replay assertions intact. The actual
four-owner case call takes 21.88 seconds. Complete log/command are under
`/tmp/agent-ua362b-external-fragments-web-final-ckwb0lr4`. No final full Application/Web
rerun is claimed after this test-only correction: the preceding full selection's
other 2,281 checks pass, and the changed module passes in its entirety.
The final all-new selection passes **66 cases, three existing dependency warnings,
in 380.80 seconds**, exit 0, against the final source freeze. It comprises 18 typed
context cases, 19 captured-context/consistency cases, 17 genuine processed
continuation cases and all 12 ordinary raw continuation cases. Complete command,
stdout/stderr and result are under
`/tmp/agent-ua362b-new-acceptance-final-2b_8jkjo/{command.txt,pytest.log,result.json}`.
The raw positives preserve exact bindings/retention and literal/span provenance;
all covered upstream bodies remain zero. Earlier raw test assertions mistakenly
expected string `"0"` instead of the existing binder's canonical integer `0`;
that fixture-only expectation was corrected, with no production binder change.
Final 12-case coverage supersedes the earlier split eight/four passing invocations.

The final command is:

```bash
PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 \
/home/likeyi/anaconda3/envs/agent/bin/python -B -m pytest \
tests/application/test_execution_context_decisions.py \
tests/authority_dag/test_captured_planning_context.py \
tests/authority_dag/test_conversational_processed_continuation.py \
tests/authority_dag/test_conversational_raw_context.py \
-q -p no:cacheprovider \
--basetemp=/tmp/agent-ua362b-new-acceptance-final-2b_8jkjo/pytest-temp
```

Every `RUN_*` key is removed and Numba, Matplotlib and XDG caches are
isolated by the recorded runner environment. The exact remaining authority
selection is:

```text
tests/authority_dag/test_application.py
tests/authority_dag/test_propagation.py
tests/authority_dag/test_intent_turns.py
tests/authority_dag/test_responses.py
tests/authority_dag/test_dialogue_evidence_authority.py
tests/authority_dag/test_scientific_dialogue_authority.py
tests/authority_dag/test_scientific_guidance_authority.py
tests/authority_dag/test_processed_prior_outputs.py
tests/chromap/test_authority_dag.py
tests/chromap/test_verification_authority.py
```

The 938-case command selects `tests/unit/orchestration`; the full Application/Web
command selects `tests/application tests/web`; the corrected Web command selects
`tests/web/test_external_fragments_execution.py --durations=3`. These use the same
runner flags and isolated environments, with exact argument lists in their
`command.txt` records. The 116-case selection is:

```text
tests/benchmarks/test_semantic_v4_acceptance.py
tests/benchmarks/test_scoped_quality.py
tests/benchmarks/test_planner_benchmark_harness.py
tests/benchmarks/test_planner_benchmark_replays.py
tests/external_fragments/test_integration.py::test_v4_complete_catalog_size_and_explicit_null_index
```

The 96-case command selects existing `test_prior_outputs.py`,
`test_qualification.py` and `test_context.py` under `tests/authority_dag` with
`-m pytest -q -p no:cacheprovider`; exact invocation/environment is
`/tmp/ua362b-existing-authority-context-command.txt`. The earlier 19-case command
is recorded in `/tmp/ua362b-captured-planning-context-final-command.txt`.
Initial 370-case selection is exactly existing `tests/application` modules
`test_turn_decisions.py`, `test_dialogue_provider_contract.py`,
`test_output_selection_contract.py`, `test_scientific_binding_contract.py`,
`test_scientific_binding_safety.py`, `test_scientific_parameter_continuation.py`
and `tests/unit/orchestration/test_prior_output_planning.py`; log
`/tmp/agent-ua362b-initial-contracts.log`.
Repeated warnings are existing `pkg_resources`, `louvain` and TBB/Numba notices.
The nine Application/Web skips are explicit pre-existing real/stress gates.
`git diff --check` passes. Overlapping runs are not summed as unique test totals. Initial broad unit orchestration had 935 passes and the
two reproduced baseline lifecycle failures; the final 938-case run passes after
the bounded repair and one added corruption case. Initial Application/Web had
2,280 passes, nine gated skips and two 20-second transport polling timeouts after
all scientific steps succeeded. The first post-change confirmation has all
19 context passes and those same two timeouts while another scientific suite was
running. Timing diagnosis/final results remain distinct from scientific failures.

The single actual live schedule uses the previously accepted development-primary
OpenRouter `openai/gpt-5.6-sol`, existing exact catalog discovery and compatible
strict-schema transport. It retains no SDK retries, no semantic retry, no model
fallback, provider `require_parameters=true`, `allow_fallbacks=false`, price caps
prompt 2/completion 10/request 0, and max completion tokens 4,096. A narrower
seven-completion/per-surface schedule and $1 cumulative worst-case reservation
wrap the accepted Runner without changing production provider or science logic.
The offline schedule accepts seven reservations and rejects an eighth before
contacting a provider. Script digest is
`1e81999f23ce73ef66593a6ecf671ed322efc4fedecfe9a936805a731c45e8dd`.

A direct-Python startup first lacked credentials supplied by Conda activation.
It stopped locally with zero catalog/completion calls, zero reservation/cost and
unchanged source state. Its record remains under
`/tmp/agent-ua362b-live-once-0v5gpq0_`. The configured activated `agent` environment
then performs the sole actual qualification schedule; activation changes no
credentials or operator configuration and is not a semantic retry.

| Live case | Actual decision and qualification surface | Outcome |
| --- | --- | --- |
| A1: resolution 0.5 and update UMAP | `execute_plan`, `context=current`, exact `argument` literal/span 36:39; full ordinary interactive scoped planning and registered execution | Pass; real Leiden/UMAP once each, exact Prior/Step references, neighbors/owner/source zero |
| A2: inspect separately supplied tiny H5AD independently | `execute_plan` / `inspect_scATAC`, `context=none`, no inherited declarations; Interpreter only | Pass; no execution handoff |
| A3: other earlier analysis | `clarify / ambiguous_subject`; Interpreter only | Pass; no execution handoff |
| A4: scientific graph question | scientific `Answer` targeting `g0`; Interpreter only | Pass; no execution handoff |

The live A1 output selector explicitly retains the original graph and selects
five new outputs: resolution, clustering key/count, UMAP key and UMAP H5AD. The
real intermediate clustering H5AD is produced and verified but not separately
selected as a Revision locator. Both new steps retain `artifact_authority=None`.
The child activates at generation 2; original Revision and all ten pinned
source/input files remain unchanged. The live actual-body counts match Scenario
A with retained graph: integrity 24, verifier bodies 15 each, graph checks 66 and
analysis reads 64. Provider-authored decisions are neither repaired nor scripted.

All four live cases pass in 55.831 seconds with one catalog request and exactly
seven completions. Worst-case reservation is **$0.588964**; all seven responses
report usage cost totalling **$0.0698180** (23,789 prompt and 1,084 completion
tokens). This is observed provider usage, not invoice reconciliation. Each
response model matches the requested model. No fallback, retry or hard stop
occurs. Three Interpreter-only probes use the genuine immutable captured source
projection, not a claimed full three-turn application/explanation run. No live
raw-science or answer-rendering qualification is inferred.

Exact command, sanitized stdout/stderr, script snapshot, `evidence/results.json`,
`sanitized-summary.json` and `scientific-lineage-witness.json` are under
`/tmp/agent-ua362b-live-activated-dg1rcxvj`. Preparation/zero-provider byte
measurements and guard validation are under
`/tmp/agent-ua362b-live-preparation-d22u5bu2/prepared`. All further provider calls
are held after this completed schedule.

All eight baseline/current semantic projection groups are byte-identical, including
v3/v4 catalogs/prompts/schemas, compiler contract, result fields and scope
fingerprint. The frozen full-v4 representative remains **33,998 bytes** under its
34,000-byte constraint. The existing complete embedding-analysis scoped fixture
remains 10,435 bytes. The unchanged unscoped complete-pipeline comparison remains
35,143 bytes; it is not the bounded frozen fixture or production scoped request.
No schema/prompt budget is increased.

Actual processed-continuation UTF-8 prompt plus canonical sorted compact schema:

| Surface | Prompt | Schema | Total bytes |
| --- | ---: | ---: | ---: |
| Interpreter without executable context, baseline and current | 16,831 | 11,198 | 28,029 |
| Interpreter with captured qualified context | 17,800 | 11,262 | 29,062 |
| Capability selector | 1,760 | 714 | 2,474 |
| Scoped semantic Planner | 6,048 | 4,019 | 10,067 |
| Retention-capable output selector | 2,083 | 430 | 2,513 |

The context-free Interpreter is byte-identical to baseline; qualified projection
adds 1,033 bytes in this actual fixture. Measurements are reproducible through
`/tmp/ua362b-provider-footprint.py`,
`/tmp/agent-ua362b-dialogue-footprint.py` and
`/tmp/ua362b-provider-footprint-comparison.json`; they are fixture measurements,
not universal size claims.

A broader offline benchmark attempt produced 586 passes and 49 setup errors in
155.56 seconds. All 49 errors share the pre-existing frozen frontier-stability
seed's paid reservation guard. The same setup error reproduces at unmodified
UA3.6.2a in 2.15 seconds, before any provider call, under
`/tmp/agent-ua362b-baseline-frontier-nji6o2w7`. That historical spending fixture
is not weakened, rewritten or claimed as passing. Current semantic planning and
frozen-size regression is selected separately and passes 116 cases in 89.52
seconds (`/tmp/agent-ua362b-semantic-planning-08uc_e07`).

## I. Remaining boundaries

This is one shared captured-current-context integration, not qualification of all
23 tools for conversational execution. Reports/evidence/viewable outputs remain
distinct from reusable scientific inputs. Historical UA3.5.5 artifacts are not
retroactively qualified. Embedding-plus-ordered-ID, clustered-analysis and UMAP
owner publication contracts remain future work where genuine continuation needs
them; extension must define exact upstream closure and scientific compatibility.
Arbitrary older execution targets require existing navigation before continuation.
Literal paths never supply inferred historical lineage, and source-specific
parameters/declarations are never automatically transferred.

Reviewed processed visualization and downloads remain UA3.6.2c work with explicit
payload, companion, privacy and transport contracts. Upload-only behavior, matrix
delivery, Web source and operator resources are preserved. Larger scientific
resource qualification, MAESTRO and adaptation interaction, universal input
composition and broad Planner redesign remain outside this stage.

UA3.6.2b implementation and bounded deterministic/live acceptance are complete.
The existing frontier-stability reservation-fixture errors are explicitly recorded
above and are not represented as a green historical benchmark. No full lightweight
suite or full application rerun after the test-only deadline correction is claimed.
Changes remain unstaged and uncommitted for review; no next-stage work, commit or
push is performed.
