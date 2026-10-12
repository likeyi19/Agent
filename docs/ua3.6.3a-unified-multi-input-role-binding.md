# UA3.6.3a — Unified multi-input scientific role binding

Implemented on 2026-10-12 against committed UA3.6.2c
`6d861d32331e7443e502f9c8c96a241d73295cf1`. One bounded ordered selection now
connects independently registered H5AD sources to reviewed Registry consumers
through the existing Interpreter, Planner, compiler and scientific owners.
This qualifies one genuine clustering/evaluation workflow and a second transfer
binding/compiler witness, not arbitrary multi-file execution across all tools.

## A. Baseline and preservation

Admission found HEAD, local `main`, local `origin/main` and independently read
remote `main` equal to the exact commit above. The tracked worktree and index
were clean, only existing `?? evals/` was untracked, and the UA3.6.2c document
was present. Remote inspection uses `git ls-remote origin refs/heads/main`.
No reset, clean, fetch, pull, staging, commit or push is performed.

`/tmp/agent-ua363a-preservation-a6o9hh9c/protected-before.json` inventories
**23,624 protected files**: 1,274 evaluations, 198 outputs, 208 real application
state files, one operator configuration, 23 UA3.5.5 scientific-state files,
seven preparatory files, two human inputs, four auxiliary resources, 1,164
UA3.6.2a acceptance files, 1,489 UA3.6.2b acceptance files, 19,253 UA3.6.2c
acceptance/browser files and one checkpoint. Ordinary records pin SHA-256,
size, mtime and mode. The 5,231,645,507-byte checkpoint is checked only by
metadata; it is never loaded or rehashed.

The record also pins index bytes, local refs and all 615 tracked baseline
contents. Tracked content is anchored to the admitted clean HEAD; the retained
`tracked-baseline-anchor.json` documents five concurrent editing/capture overlaps
and marks their later metadata rather than claiming initial mtimes for them.
Protected historical records are unaffected by those intentional source edits.
The external EpiZoo checkout remains clean at
`029cd631d0f9806a646c4a3b42ce10b958f2b67f`.

All new science, Sessions, Runs, reports, caches and acceptance logs use isolated
`/tmp` workspaces. Native Snap Firefox uses only a fresh home-directory profile;
old scientific state, profiles, configuration and evaluations are not used.
`implementation-final-freeze.json` pins all 518 current source/test/benchmark
files before the stable domain runs. All 518 remain byte-identical with unchanged
inventory after acceptance. The final comparison finds **zero mismatches across
all 23,624 protected records**; all 1,274 evaluation files and the complete
198-file output inventory remain unchanged. Index bytes and all three local refs
remain exact; the final remote `main` read also matches the admitted commit.
Of 615 tracked baseline files, 604 remain byte-identical; the 11 intentional
changes are the ten production files and existing executor test in section C.
Six new test modules and this document remain unstaged. The completed preservation,
freeze, test and browser records are summarized in
`/tmp/agent-ua363a-preservation-a6o9hh9c/completion-validation.json`.

## B. Confirmed gap and existing contracts

The ordinary Web `SubmitTurn` accepted one registered resource ID. Existing
`RegisteredInput` and H5AD companion composition bind a single `input_path`;
the explicitly attributed FASTQ collection describes one library. Neither
represents independent H5ADs assigned to different scientific roles.

The scientific contracts were already present: singleton semantic dataset and
analysis ports, typed request aliases, `PlanningProvenanceRole`,
`SemanticLineage`, same-plan channels and the shared exact-string binder.
No compiler or scientific contract repair is needed. The reused records are
[Web upload](ua3.1-web-upload.md),
[H5AD composition](ua3.2.2-h5ad-composition.md),
[FASTQ collection](ua3.4.3-fastq-web-upload.md),
[scientific declarations](ua3.5.3-unified-scientific-input-binding.md),
[capability readiness](ua3.5.4d-practical-capability-readiness.md),
[exact strings](ua3.6.1-shared-string-scientific-declarations.md),
[ordinary contextual planning](ua3.6.2b-unified-conversational-state-planning.md)
and [result delivery](ua3.6.2c-unified-scientific-result-delivery.md).

The shared path is:

```text
existing uploads/registered records → captured ordered selection
  → typed Interpreter roles → Registry request aliases
  → ordinary Planner/compiler/preflight → exact compiled consumer admission
  → existing executor/owners/verification → RunStore/Session Revision
  → existing accepted evidence/report/figures
```

Registry remains **23 tools and 172 public arguments**. The compiler, semantic
channels, provider transport/selection, paid-request guards, scientific authority,
numerical algorithms and accepted result-delivery contracts are unchanged.

## C. Minimal implementation

| File | Change |
| --- | --- |
| `src/agent/application/local_resources.py` | Immutable `RegisteredInputGroup`, ordered record capture, pure role replay/projection, Registry companion validation, exact compiled/resolved consumer checks and registered-byte admission. |
| `src/agent/application/interactive.py` | Admit the group through the existing submission/fingerprint/lease path. |
| `src/agent/application/turn_decisions.py` | Bounded typed `InputRole(tool, port, source)` and conditional offered role/schema instructions; lossless UTF-8 serialization for grouped Interpreter prompts only. |
| `src/agent/application/dialogue_execution.py` | Materialize proposed roles before scalar binding; replay exact attribution; preserve role-bearing label continuation; install operation-local registration checks. |
| `src/agent/application/turns.py` | Project safe captured source choices and preserve received inputs on duplicate role-bearing turns. |
| `src/agent/orchestration/executor.py` | Optional operation-local input-admission callback before and after each registered attempt; ordinary sequential execution, verification and retry policy remain the engine. |
| `src/agent/web/app.py` | Optional bounded ordered `resource_ids`, central group admission and the existing worker/facade handoff. |
| `src/agent/web/static/app.js`, `index.html`, `styles.css` | Small Add another input/numbered selection/Remove controls and exact frozen request selection. |

Six new test modules are:

```text
tests/application/test_multi_input_group_contract.py
tests/application/test_multi_input_role_decisions.py
tests/application/test_multi_input_role_admission.py
tests/web/test_multi_input_scientific_roles.py
tests/web/test_multi_source_selection_transport.py
tests/web/test_multi_source_selection_static.py
```

`tests/unit/orchestration/test_executor.py` also adds four meaningful callback
checks: rejection before production, rejection after the body, source admission
before a retry, and concurrent operation isolation/reset. No scientific-owner,
Registry, upload-service, operator-config or Session-schema file changes.

## D. Exact membership and scientific roles

`RegisteredInputGroup` is an inline typed turn binding, with composition
`scientific-role-selection.v1`, not a new registration record or collection store.
It captures two to eight ordered existing records. Each private member preserves
resource ID, immutable registration-record digest, original source locator,
declared input kind and safe display label. No input is copied to assign roles.
Repeated resource IDs or source locators fail closed.

Original received submission, generation, task text and context stay immutable.
Admitted role bindings and scientific inputs are separate projections. The
existing submission digest provides identity/idempotency; no new group identity
system or pending store is introduced.

The Interpreter receives only `source.0` through `source.7`, explicit selection
positions, safe labels, declared `h5ad` kind and offered tool/port/role metadata.
It receives no paths, resource IDs, digests, file contents or expanded history.
An ordinal identifies the exact captured order only when the user associates it
with a scientific role. Selection order never establishes biology.

Initial reviewed consumers are:

| Tool / semantic port | Registry-derived request input | Existing role |
| --- | --- | --- |
| `cluster_cells.analysis` | `analysis_path` | Branch-preserving neighbors input |
| `evaluate_cell_clustering.analysis` | `analysis_path` | Branch-preserving fixed analysis |
| `evaluate_cell_clustering.ground_truth_dataset` | `reference_h5ad_path` | Ground truth |
| `transfer_cell_labels.reference_dataset` | `reference_input_path` | Reference |
| `transfer_cell_labels.query_dataset` | `query_input_path` | Query |

This is an explicit qualification list of ports; aliases, artifact expectations,
lineage and argument domains are derived from their existing Registry metadata.
Canonical source selection requires one singleton matching the port field name,
or otherwise one unique singleton. Ambiguity rejects rather than selecting a
first match. Selected source paths cannot be supplied as arbitrary companions.
Explicit ordinary operator InputSets may provide supported Registry companions,
including existing reference/query embedding bundles; this does not qualify their
scientific content or invent model/species provenance.

Every selected member needs one distinct offered consumer association. Duplicate,
conflicting, absent or unsupported roles fail closed. Shared alias conflicts and
repeated actual consumers reject. The actual compiled plan must consume each
exact registered member at its authorized tool/port; another path, unauthorized
selected-source fanout or unregistered literal dataset input cannot substitute.
Same-plan outputs remain `StepOutputRef`; historical outputs retain their existing
authority requirements. Uploaded compact analysis is an explicit new scientific
input, not an upgrade of a historical clustering publication.

The LLM owns semantic role interpretation. Agent owns exact records, typed
associations, aliases, consumer checks and source integrity. Scientific owners
validate artifact content, ordered cells, labels, provenance and compatibility.
Structurally valid but semantically wrong Interpreter proposals remain a language
qualification boundary; compiled substitutions against captured roles are
deterministically rejected. No lexical role parser is introduced.

## E. User interaction, continuation and source lifetime

The existing chat composer gains a small Add another input action and numbered
safe selections with Remove controls. Click order is retained across refresh and
rechecked against the existing catalog. One source still sends `resource_id`;
two to eight send `resource_ids`. These fields are mutually exclusive. The same
32 KiB HTTP body bound, worker capacity and immutable retry mechanism remain.
Browsers transmit IDs, not paths, digests, roles or scientific plans.

Source-specific companion/resource controls remain separate. H5AD/BAM/fragments
single-source composition, completed FASTQ library selection and operator InputSet
submission keep their existing contracts. Independent groups admit only H5AD
records in this stage; raw members and completed FASTQ collection records reject
with bounded resource errors. Completing/uploads/selecting alone executes no
science, provider, Run or Revision.

Missing/ambiguous role association produces ordinary clarification. It is not
claimed to have automatic durable role-answer continuation. Once roles are
admitted, an omitted required `label_key` uses the existing compiler diagnostic
and `@parameters` continuation. Its answer retains the original objective,
selection, roles, generation/context and shared exact-string declaration.
Both omission of the selection and exact same-selection resubmission are tested.
A different or reordered selection cannot redirect the pending task. An unrelated
new task receives no old roles or parameter values.

New consumption verifies exact registration identities and current byte pins at
submission, planning admission, immediately before and after each affected
registered tool attempt, and first Session acceptance after verification/report
completion. The executor callback is repository-owned, ContextVar-scoped,
reset in `finally` and separate across concurrent operations. It is never a
provider input, persistent authority or another execution engine. Existing
scientific-owner stable source checks remain authoritative for scientific facts.

Accepted terminal reads, retries, evidence and fresh-instance restoration never
call first-acceptance checks or require current source files. Archived accepted
inputs are tested. Interactive recovery remains storage/presentation reconciliation
and never resumes unfinished science. Standalone `ResearchAgentApplication.resume`
is outside this Session role-admission qualification; changed sources cannot be
adopted later through first Session acceptance. No new automatic scientific-resume
workflow is claimed.

## F. Genuine scientific and secondary witnesses

The primary fixture uses explicit seed-17 32 × 512 float32 values, 32 ordered
Unicode cell IDs and a genuine existing neighbors construction with unchanged
defaults. Its graph H5AD and an independent sparse 32 × 4 annotated ground-truth
H5AD are separately uploaded and registered. No foundation inference/checkpoint
or biological data is required. Fixture graph construction occurs before the
measured conversational turn; registration alone grants no scientific authority.

The ordinary Web request is:

> Cluster the first selected dataset, then evaluate its fixed clustering against the second selected dataset as ground truth. Use ground-truth column "Cell 类型 Δ".

Typed roles associate the first source with `cluster_cells.analysis`, the second
with evaluation ground truth. The shared string binder preserves `Cell 类型 Δ`
exactly. The Planner supplies cluster → evaluation through the existing semantic
channel; ground truth never enters clustering arguments. Genuine Leiden and the
existing evaluator execute and pass normal scientific verification. Two clusters
produce NMI/ARI/AMI/Homogeneity 1.0 for this synthetic fixture. This is software
acceptance, not biological accuracy or label validation.

| Observed body | Ordinary successful turn | Upload / viewing / completed retry / fresh-instance reads |
| --- | ---: | ---: |
| Interpreter / Planner detailed body | 1 / 1 | 0 / 0 |
| Compiler / preflight | 1 / 2 | 0 / 0 |
| Clustering owner / Leiden body | 1 / 1 | 0 / 0 |
| Evaluation public owner body | 1 | 0 |
| Normal clustering / evaluation verifier | 15 / 15 | 0 / 0 |
| Evaluation source/metric reconstruction | 16 | 0 |
| Neighbors / EpiZoo inference during turn | 0 / 0 | 0 / 0 |
| Scripted model completions | 4 | 0 |

The evaluator's reconstruction count includes initial evaluation plus repeated
existing independent verification; it recomputes all four metrics, rather than
claiming metrics are computed only once. Existing accepted evidence records the
label key and exact user declaration; safe reports/figures and the accepted Run
and Revision persist. Refresh/reopen and a forbidden-provider fresh facade return
the same accepted result without science or a new Revision.

The secondary witness selects registered reference/query H5ADs and explicitly
supplied existing embedding/ordered-ID/species/checkpoint companions. The same
group/role path and `reference_label_key` binder reach normal compiler/preflight
and a recording executor that withholds science. Exact role lineage and reversed
selection with explicit corrected handles pass; swapped dataset/embedding
consumers reject. This qualifies reusable binding/compiler mechanics, not a second
executed scientific workflow or model/embedding provenance.

## G. Fail-closed findings and fixes

Focused negatives cover forged/missing/changed registrations, duplicate source
locators, unsupported raw/library kinds, arbitrary dataset-path companions,
unassigned/conflicting/absent roles, wrong ports, selected source substitution,
repeated consumers, swapped reference/query lineage, nonexistent columns, wrong
cell order, missing scientific provenance, source replacement, altered retries,
unrelated tasks and pending answers that attempt to switch inputs.

Association failures run no scientific owners and publish no successful Revision.
Content failures reach the existing correct owner: missing graph provenance stops
before Leiden; missing labels or changed ordered truth reach evaluation validation
and fail without a Revision. No Web content validator duplicates those checks.

Review demonstrated two byte-pin gaps and fixes:

- A scientifically valid replacement of ground truth during preceding Leiden
  could formerly pass whole-plan admission and produce evaluation on different
  bytes. Per-consumer registration checks now stop evaluation/source reconstruction
  at zero after the already-completed Leiden, with no successful Revision.
- A trailing HDF5 byte added when evaluation verification begins could leave
  scientific values valid while changing the registration SHA. First Session
  acceptance now rejects `LOCAL_RESOURCE_INTEGRITY_INVALID`. Its Run may already
  be scientifically successful; Session publication fails, and failed retries or
  recovery perform no additional work. Terminal accepted history remains readable.

A grouped label continuation also exposed tuple-versus-list equality of frozen
metadata; comparing both exact serialized structures repairs it without changing
membership or roles. Completed FASTQ collection admission now checks concrete
record kind before accessing H5AD fields, giving an ordinary resource error.

An exploratory five-neighbor fixture exposed a pre-existing owner/visualization
boundary: homogeneity `1.0000000000000002` passes owner tolerance but fails the
figure's strict range. No scientific/visualization algorithm is changed here;
the final fixture uses existing default neighbors and exact 1.0 scores. That
general visualization tolerance edge remains separate unqualified work.

## H. Validation and prompt measurements

All pytest commands use `/home/likeyi/anaconda3/envs/agent/bin/python -B`,
`PYTHONPATH=src:.`, `PYTHONDONTWRITEBYTECODE=1`, all `RUN_*` unset,
`-q -p no:cacheprovider`, isolated basetemp and `/tmp` Numba/Matplotlib/XDG caches.
`/tmp/agent-ua363a-run-selection.py` emits exact `command.txt`, complete
`pytest.log`, and `result.json` with exit/runner timing. No full 7,000-plus-test
suite, paid request, live semantic call, foundation inference, adaptation training
or real PBMC preprocessing is run.

| Stable final selection | Result |
| --- | --- |
| Six new modules plus existing executor, including four new guard checks | **125 passed**, three existing warnings, 33.97 s; exit 0. |
| Complete affected Application/Web domain | **2,416 passed**, nine existing gated skips, three existing warnings, 829.34 s; exit 0. |
| Compiler/Planner/Registry/executor/runtime, scientific evaluation/transfer and authority/context domains | **662 passed**, five warnings, 469.72 s; exit 0. |
| Final native Firefox | **19 checks passed**, 11 initial and eight fresh-server; exit 0. |

Selections overlap and are not unique aggregate counts. Earlier passing selections
include 259 local/H5AD/group checks, 315 model/scalar contracts, 90 pending/string
checks, 207 single-source Web/configuration checks, 550 input/model/pin checks and
88 final group/FASTQ/executor checks. They are supporting, overlapping evidence;
the stable final commands above define acceptance. The three repeated warnings
are existing `pkg_resources`, `louvain` and TBB/Numba notices.
The scientific/orchestration domain also emits two expected AnnData duplicate
observation-name warnings from the deliberate nonunique-cell-ID evaluation case.
All nine Application/Web skips are existing FASTQ qualification cases gated by
explicit `RUN_CHROMAP_QUALIFICATION=1` in `tests/chromap/conftest.py`: three
Application and six Web cases. No selected tests are excluded.

The exact focused command is:

```bash
PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 \
/home/likeyi/anaconda3/envs/agent/bin/python -B -m pytest \
  tests/application/test_multi_input_group_contract.py \
  tests/application/test_multi_input_role_decisions.py \
  tests/application/test_multi_input_role_admission.py \
  tests/web/test_multi_input_scientific_roles.py \
  tests/web/test_multi_source_selection_transport.py \
  tests/web/test_multi_source_selection_static.py \
  tests/unit/orchestration/test_executor.py \
  -q -p no:cacheprovider \
  --basetemp=/tmp/agent-ua363a-final-focused-f0pol5g1/pytest-temp
```

Its logs/command/exit record are in
`/tmp/agent-ua363a-final-focused-f0pol5g1`. Direct scientific and secondary witnesses
are respectively `test_multiple_uploads_roles_co0/multi-input-scientific-witness.json`
and `test_transfer_contract_shares_0/secondary-transfer-binding-witness.json`
under that `pytest-temp`. Actual counts are directly emitted test artifacts.

The exact complete Application/Web command is:

```bash
PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 \
/home/likeyi/anaconda3/envs/agent/bin/python -B -m pytest \
  tests/application tests/web -q -p no:cacheprovider \
  --basetemp=/tmp/agent-ua363a-app-web-final-v3631_eq/pytest-temp
```

Its complete records are
`/tmp/agent-ua363a-app-web-final-v3631_eq/{command.txt,pytest.log,result.json}`.
After it exits 0, the exact serial scientific/orchestration domain command is:

```bash
PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 \
/home/likeyi/anaconda3/envs/agent/bin/python -B -m pytest \
  tests/unit/orchestration/test_semantic_wire_v4.py \
  tests/unit/orchestration/test_semantic_prompt.py \
  tests/unit/orchestration/test_executor.py \
  tests/unit/orchestration/test_scientific_authority_dispatch.py \
  tests/unit/orchestration/test_semantic_registry_metadata.py \
  tests/unit/orchestration/test_llm_planner.py \
  tests/unit/orchestration/test_cancellation.py \
  tests/unit/orchestration/test_semantic_compiler.py \
  tests/unit/orchestration/test_registry.py \
  tests/unit/orchestration/test_llm_planner_v4_recovery.py \
  tests/unit/orchestration/test_prior_output_planning.py \
  tests/unit/orchestration/test_llm_planner_v4.py \
  tests/unit/orchestration/test_recovery_policy.py \
  tests/unit/orchestration/test_runtime.py \
  tests/unit/orchestration/test_milestone6_2_lifecycle.py \
  tests/unit/orchestration/test_milestone6_3_lifecycle.py \
  tests/tools/analysis/test_clustering_evaluation.py \
  tests/tools/analysis/test_label_transfer.py \
  tests/authority_dag/test_captured_planning_context.py \
  tests/authority_dag/test_conversational_processed_continuation.py \
  tests/authority_dag/test_processed_prior_outputs.py \
  tests/authority_dag/test_conversational_raw_context.py \
  -q -p no:cacheprovider \
  --basetemp=/tmp/agent-ua363a-domain-final-akuxgmj1/pytest-temp
```

Its corresponding command, full log and exit record are in
`/tmp/agent-ua363a-domain-final-akuxgmj1/{command.txt,pytest.log,result.json}`.
Both domain commands use the same isolated environment and disabled gates as the
focused run. Application/Web covers declaration binding, Session and upload
compatibility and existing UA3.6.2c delivery; the additional domain covers normal
scientific owners, compiler lineages, runtime/retries/cancellation and the
accepted UA3.6.2b raw/processed conversational continuations.

Prompt measurements and baseline replay are in
`/tmp/agent-ua363a-prompt-measurement-7u1qoa8u/{result.json,replay.py}`.
The frozen representative v4 prompt/schema remain byte-identical to admitted
HEAD: **21,013 + 12,985 = 33,998 UTF-8 bytes**, below the unchanged 34,000 limit.
Ordinary two-source Interpreter measurement is **30,059** bytes; eight ordinary
labels total **30,885**. The exact scientific witness differs only in ordinary
utterance/label text: its actual prompt/schema totals are **30,072**, **2,590**,
**9,777** and **2,015** bytes for Interpreter, scope selection, detailed planning
and output selection respectively.

Eight maximum 160-character Unicode labels produce **35,837** Interpreter bytes
after lossless grouped UTF-8 encoding, reduced by 10,176 bytes from redundant
ASCII escapes. This is a measured variable Interpreter-context stress case,
not a universal 34,000-byte claim. The frozen v4 budget is not increased; no
roles, labels or scientific requirements are truncated to force that stress
case into a different fixture's bound. Existing utterance/history context also
varies. Paths, resource IDs, hashes and file contents are absent from all measured
prompts/schemas. Model-facing semantic/provider quality remains unqualified by
these deterministic scripted tests.

Final native Firefox evidence is
`/tmp/agent-ua363a-native-firefox-guard-final-bltnhh8k/browser-summary.json`, with
exact commands in `commands.json`, retained server/browser logs and screenshots.
The new Snap-visible profile is `/home/likeyi/agent-ua363a-firefox-p0dtsxdx/profile`.
Firefox runs `/usr/bin/firefox --headless --no-remote --marionette
--remote-allow-system-access --profile` with that profile. Server/driver use conda
`agent -B` and the same private environment. Native interaction checks the ordered
picker, Add/Remove, pre-execution refresh, ordinary composer execution, accepted
evidence/report/result association, completed refresh and fresh server. Its
production/verification counts equal the successful-turn table above. Fresh server
uses a forbidden provider factory: zero clients/completions, science or new
Revisions; generation 1 and the one accepted Revision remain exact.
There are zero application JavaScript errors; internal Firefox notices remain
separately classified. Screenshots are visually reviewed. Servers and Firefox
processes are stopped after acceptance. This is native browser evidence, not an
API-only substitute or live-provider qualification.

## I. Remaining boundaries and next task

Only independently registered H5AD groups and the five reviewed ports are added.
No arbitrary raw-source grouping, experimental-design/list parser, assembly or
barcode inference, automatic resource provisioning, new species qualification,
MAESTRO resource/signature admission, adaptation strategy inference, processed
authority expansion or additional download role is implemented. Single-source
and explicit FASTQ library routes remain separate scientific contracts within
the same interaction infrastructure.

For the four core scenarios, raw preprocessing still needs exact source-specific
library/reference/QC contexts; MAESTRO needs qualified canonical inputs, groups and
explicit tissue/resources; EpiZoo needs existing qualified model/species/feature
resources; adaptation needs its owner-qualified target/source/SEAM declarations.
Reference transfer additionally needs scientifically compatible complete embedding
bundles. Selecting H5ADs alone establishes none of those facts. General role-answer
continuation, standalone unfinished group-run resume and live semantic accuracy
are not qualified here.

The recommended next task is **UA3.6.3b — Qualified Source-Specific Resource
Connection**, connecting one exact approved uploaded raw route through existing
operator and LocalResourceAdmission interfaces. UA3.6.3c should address only the
remaining reviewed declaration categories discovered there. No next-stage work,
staging, commit or push is performed.
