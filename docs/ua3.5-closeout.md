# UA3.5 — Conversational scientific workflow closeout

UA3.5 is complete within its accepted
conversational and scientific-workflow scope. It establishes a reusable natural-
language interaction mechanism and one genuine complete human scientific task.
This closeout introduces no capability expansion and does not start UA3.6.

## Baseline and publication scope

The reviewed baseline is `8bcad5b42cb812c89dfc232e3ceadea2fc5e44a7` on `main`.
Entry review finds 30 modified tracked files and 31 new non-evaluation files,
with a clean index and preserved untracked evaluations. Earlier stage records
describe their original checkpoints; this record supplies the milestone disposition.

The explicit allowlist contains **68 files: 19 production, 32 tests/fixtures, six reusable
benchmark sources and 11 documentation files**. Their exact paths are:

Production:

```text
src/agent/application/dialogue_evidence.py
src/agent/application/dialogue_execution.py
src/agent/application/interactive.py
src/agent/application/local_resources.py
src/agent/application/responses.py
src/agent/application/scientific_dialogue.py
src/agent/application/session_state.py
src/agent/application/session_store.py
src/agent/application/sessions.py
src/agent/application/turn_decisions.py
src/agent/application/turns.py
src/agent/orchestration/llm_planner.py
src/agent/orchestration/planning_diagnostics.py
src/agent/orchestration/registry.py
src/agent/orchestration/semantic_compiler.py
src/agent/orchestration/semantic_prompt.py
src/agent/tools/analysis/embedding_analysis.py
src/agent/web/app.py
src/agent/web/config.py
```

Tests:

```text
tests/application/test_clarification_wording.py
tests/application/test_clustering_parameter_evidence.py
tests/application/test_dialogue_provider_contract.py
tests/application/test_h5ad_composition.py
tests/application/test_human_epizoo_mapping.py
tests/application/test_initial_scientific_declarations.py
tests/application/test_output_selection_contract.py
tests/application/test_prerequisite_planner_handoff.py
tests/application/test_recovery_policy_submission.py
tests/application/test_scientific_binding_contract.py
tests/application/test_scientific_binding_safety.py
tests/application/test_scientific_parameter_continuation.py
tests/application/test_selection_parameter_evidence.py
tests/application/test_turn_decisions.py
tests/application/test_typed_turn_admission.py
tests/benchmarks/fixtures/offline_replays.json
tests/benchmarks/test_interactive_breadth.py
tests/benchmarks/test_interactive_frontier_transport.py
tests/benchmarks/test_interactive_stability.py
tests/integration/test_milestone4_llm_planner_runtime.py
tests/tools/analysis/test_embedding_analysis.py
tests/unit/orchestration/test_registry.py
tests/unit/orchestration/test_semantic_prompt.py
tests/web/test_bam_execution.py
tests/web/test_h5ad_analysis.py
tests/web/test_h5ad_composition_api.py
tests/web/test_leiden_resolution.py
tests/web/test_output_selection_api.py
tests/web/test_scientific_binding.py
tests/web/test_selection_clarification.py
tests/web/test_species_handoff.py
tests/web/test_uploaded_multiturn.py
```

Benchmarks:

```text
benchmarks/interactive/run_ua354a.py
benchmarks/interactive/run_ua354b.py
benchmarks/interactive/run_ua354c.py
benchmarks/interactive/run_ua355.py
benchmarks/interactive/run_ua35_closeout.py
benchmarks/interactive/ua354a_fixtures.py
```

Documentation:

```text
README.md
docs/ua3.5-closeout.md
docs/ua3.5-final-conversational-closeout-review.md
docs/ua3.5.1-conversational-prerequisite-handoff.md
docs/ua3.5.2-leiden-resolution-defaults.md
docs/ua3.5.3-unified-scientific-input-binding.md
docs/ua3.5.4a-live-semantic-qualification.md
docs/ua3.5.4b-prerequisite-handoff.md
docs/ua3.5.4c-output-selection-contract.md
docs/ua3.5.4d-practical-capability-readiness.md
docs/ua3.5.5-human-epizoo-user-workflow.md
```

The consolidated commit includes only reviewed UA3.5 production source, tests,
reusable benchmark source and documentation. Local evaluation records remain
untracked. Scientific datasets, checkpoints, generated scientific workspaces,
credentials and the external operator configuration are excluded. Publication is
one ordinary commit on `main` followed by a normal push; no history is rewritten.

## What users can do

Users can upload the supported H5AD, FASTQ, BAM and external-fragments forms,
select the applicable scientific input, and describe an eligible registered task
in chat. Explicit reviewed species and scalar declarations bind to that exact
input. Valid explicit parameters take precedence, conflicts fail closed, and
omitted optional parameters retain their scientific owner's defaults. Eligible missing scalar declarations produce compiler-owned prerequisite
questions; missing artifacts, resources and invalid authority remain failures. Ordinary-language
answers can complete or correct the captured request, which continues through
the usual Planner, compiler, execution and verification path. Results persist in
RunStore and Session/Revision state and can be inspected through the existing
browser evidence, safe report and image surfaces. Upload alone runs no science. Selected M17.2 guidance candidates retain their
existing missing-input failure contract; this milestone adds no candidate pending
workflow.

## Ownership and human deployment

One Interpreter and scoped Planner own language interpretation, operation
association, capability choice and semantic graph proposals. Registry/ToolSpec,
Agent's typed binder, source identity, qualified resources, compiler/preflight,
executor and existing Session records own deterministic admission, execution and
provenance. Scientific owners retain algorithms, numerical domains and defaults.
Web supplies inputs and renders interaction/results. No second parser/Planner,
workflow store, provider-specific science rule or application default engine is
introduced; authority and identity checks remain fail closed.

Agent invokes the original EpiZoo processing, datasets, model/configuration and
embedding extraction through `epizoo_embed_cells` → `analysis.epizoo_embedding`
→ `epizoo_cache`/`models.epizoo`. It does not duplicate transformer, tokenization,
normalization or forward-inference algorithms. Human and mouse share the original
joint checkpoint. The independently accepted Scanpy neighbors, fixed Leiden and
UMAP algorithms/defaults remain unchanged; UA3.5.2 only extracts the existing
Leiden scalar validator for shared admission. Existing convenience-wrapper overlap
is described in [UA3.5.5](ua3.5.5-human-epizoo-user-workflow.md), without a refactor.

The human default was admitted to this deployment's **external** operator
`epizoo_resources` catalog as `human-recorded-qualified`; the mouse entry and
Groq default/profile remain unchanged. Another deployment must explicitly admit
one reviewed human resource tuple through that existing catalog: `species=human`,
its local original-joint-checkpoint `checkpoint_path`, `resource_id`, `label`, `checkpoint_sha256`,
`frequencies_sha256`, `filter_indices_sha256`, accurate `qualification` and
`default=true`. Use the exact original human auxiliary files consumed by the
existing owner and the established pins in [UA3.5.5](ua3.5.5-human-epizoo-user-workflow.md#b-shared-model-human-mapping-and-invocation-ownership).
Preserve the mouse row and unrelated settings. This designation connects
qualified resources; it does not train or validate another human model or make an
arbitrary H5AD compatible. No private configuration, credentials or deployment
paths are embedded in this closeout record or committed as configuration.

## Acceptance evidence

| Bounded achievement | Accepted evidence and limits |
| --- | --- |
| Supported upload/registration | Preserved [UA3.1](ua3.1-web-upload.md) and [UA3.4 closeout](ua3.4.3-fastq-web-upload.md); exact input identity, declared FASTQ collections, qualified source-specific raw companions. Registration is not scientific authority. |
| Natural-language scientific planning | Existing Interpreter/scoped Planner; live four-step human request in UA3.5.5. No upload-specific recipe or additional Planner. |
| Registry-grounded typed parameters | [UA3.5.3](ua3.5.3-unified-scientific-input-binding.md) exact tool/argument/literal/span binding and actual-consumer checks; reviewed contracts only. |
| Explicit precedence and owner defaults | [UA3.5.2](ua3.5.2-leiden-resolution-defaults.md) and [UA3.5.4b](ua3.5.4b-prerequisite-handoff.md): live Leiden 0.7, omitted default 1.0 and paraphrase; no invalid-value fallback or biological default policy. |
| Compiler-owned prerequisite diagnosis | [UA3.5.4b](ua3.5.4b-prerequisite-handoff.md) repairs supported incomplete-candidate handoff; artifacts, resources and incompatible graphs remain authoritative failures. |
| Durable clarification and continuation | [UA3.5.1](ua3.5.1-conversational-prerequisite-handoff.md)/UA3.5.3 scripted software/browser coverage; UA3.5.4b live Selection partial answer, correction and verified completion. Synthetic threshold operands do not calibrate biology. |
| Initial dataset-species declaration | [Final conversational review](ua3.5-final-conversational-closeout-review.md): six genuine-LLM cases pass explicit human, omitted species, live answer, model-support-only mention, matching companion and conflict admission boundaries. Every scientific tool is withheld. |
| Qualified human/mouse resources | [UA3.5.4d](ua3.5.4d-practical-capability-readiness.md) original joint-resource evidence and UA3.5.5 actual human consumed pins; unique species defaults and explicit choice precedence. |
| Valid multi-step output selection | [UA3.5.4c](ua3.5.4c-output-selection-contract.md) exact offered pairs, unique nonblank Revision-wide names; UA3.5.5 persists 13 unrepaired selections. |
| Genuine human scientific execution | UA3.5.5 native Firefox + real provider + normal scientific executor: one foundation load/invocation, finite 32×512 embeddings, genuine neighbors, four Leiden clusters and finite 32×2 UMAP. |
| Verified RunStore/Session/Revision | Same genuine UA3.5.5 task: passed durable preflight, step/Run verification, succeeded Run and one activated Revision, generation 1. |
| Browser results/evidence/report/image | Same native journey opens all 13 selections, safe report and pinned PNG; refresh/new server restore without replay. Restoration is within one Python process. |

UA3.5.1–3 deterministic coverage uses scripted semantic decisions and tiny
fixtures; it does not establish live language accuracy or universal biological
quality. UA3.5.4a's failures and blocked cases remain preserved; UA3.5.4b and
UA3.5.4c repair and requalify their specific contracts. UA3.5.4d is a read-only
23-tool/172-parameter readiness assessment, not universal live qualification.

The earlier final initial-declaration stage passes **198 focused tests** and
**2,107 Application/Web tests, nine skipped**; its genuine-LLM matrix passes
**6/6 admission cases**, withholding every scientific tool. Those saved results
are reused, rather than repeated provider or foundation acceptance.

The new initial-declaration admission evidence and UA3.5.5's earlier combined
browser/provider/science task are **separate witnesses**. The genuine task used
a selected typed human companion; the final run supplied initial species through
chat and stopped before tools. No fictitious combined utterance-only scientific
journey is claimed, and no foundation inference is repeated for closeout.

## Remaining scope

All 23 tools remain registered, but they are not individually live-qualified.
The shared binder does not collect every required string/list/design declaration.
Some raw uploads need separately approved source-specific resource companions.
Generic processed-output cross-turn authority and raw embedding/compact-H5AD
downloads are not universally available. EpiZoo still requires exact compatible
full-axis sparse count-like data; peak projection or species guesses cannot
satisfy that contract. Cell Selection candidates 1000/4/20 are not approved
general defaults. Statistical cell calling, broad biological interpretation and
scientific generalization remain outside this milestone.

The uncaptured Selection partial-answer and unfinished-BAM fixture exceptions
remain unresolved diagnostic risks. Subsequent isolated/domain reruns pass;
these do not establish cause or justify labeling them flaky, pre-existing or
caused by UA3.5. Original failed records remain intact; no speculative hardening
is part of this closeout.

## Final validation and preservation

Closeout's first full invocation aborts with an instrumentation `INTERNALERROR`:
`test_symlinks_and_partial_write` deliberately replaces `os.replace`, which also
hits the temporary progress hook's `Path.replace`. The hook is removed; production
and tests remain unchanged. Its trace and failed command are retained.

The second invocation exposes two original M17.2 missing-input assertions and is
stopped after **2 failed, 1,784 passed, three skipped, two warnings in 1,476.02
seconds**, exit 2. New generic prerequisite synthesis incorrectly turns selected
guidance candidates into pending requests whose continuation lacks the candidate
context. The existing `dialogue_execution.execute` boundary now excludes selected
candidates from species/numeric pending creation, restoring their accepted failure
contract. No test expectation, candidate context/authority or ordinary pending
continuation is changed. This diagnosed repair is separate from the older
uncaptured Selection/BAM risks above.

The affected guidance and ordinary declaration/continuation suites first pass
**236 tests in 118.30 seconds**, exit 0. A subsequent complete regression records
**72 failed, 6,918 passed, 92 skipped, seven warnings and 56 fixture errors in
3,648.71 seconds**, exit 1. All original exceptions and failed states remain
intact. Its diagnosed repairs are limited to existing owners:

- Preserve the four-argument execution callback when the resource catalog is
  empty; forward any nonempty admitted catalog exactly. This restores the
  frozen zero-science harness interface without modifying historical benchmark
  implementations or suppressing their identity rejection.
- Represent an optional v3 argument with no possible input/ref as null-only;
  required unbindable arguments still fail. The parser and pin checks stay closed.
- Compact generic v4 instructions while retaining every handoff, scope, resource,
  authority and execution rule. The frozen 23-tool regression request's catalog/schema now total **33,998
  bytes**, within the unchanged **34,000-byte** assertion.
- Use existing frozen-Git simulation in offline breadth/stability fixtures and
  their module fixtures; their production baseline guards and negative tests
  remain strict. Refresh only current offline Interpreter fingerprints and add
  the registered nullable resource field to synthetic v3 replies. No historical
  response, manifest or evaluation artifact is repaired.
- Give only the five-step BAM presentation test a bounded 60-second wait. Its
  original Run/Revision succeeds after 21.329 seconds with no scientific error,
  while its old 20-second wait expires during finalization. All scientific and
  download assertions stay unchanged; this is distinct from the old uncaptured
  BAM admission exception.

The first integrated focused run preserves **2 failed, 871 passed in 230.36
seconds**, exit 1: a second stale schema literal and a missing instruction anchor.
Both exact traces are retained and corrected. The corrected whole focused selection passes **873 tests in 230.93 seconds**,
exit 0, with all 497 source/test/benchmark files unchanged and no observer
exceptions.
All attempts, exact commands, traces, states and independent reviews remain in
[closeout diagnostics](../evals/ua3.5_closeout_2026-10-10/diagnostics/). These are
CPU/offline contract tests; no paid-model, GPU/foundation, historical raw-science
or native-browser acceptance is repeated.

The final lightweight full repository regression disables every `RUN_*` gate and
uses no test exclusions:

```bash
PYTHONPATH=src:.:/tmp/agent-ua35-closeout-preservation-pl4ibn4_/observer \
PYTHONDONTWRITEBYTECODE=1 \
NUMBA_CACHE_DIR=/tmp/agent-ua35-closeout-preservation-pl4ibn4_/numba-cache \
MPLCONFIGDIR=/tmp/agent-ua35-closeout-preservation-pl4ibn4_/matplotlib-cache \
UA354C_ADMISSION_EXCEPTION_LOG=/tmp/agent-ua35-closeout-preservation-pl4ibn4_/final-recheck-admission-exceptions.jsonl \
/home/likeyi/anaconda3/envs/agent/bin/python -B -m pytest -q -p no:cacheprovider tests --basetemp \
/tmp/agent-ua35-closeout-preservation-pl4ibn4_/full-final-recheck-pytest-tmp --tb=long -p ua354c_admission_observer
```

**7,046 passed, 92 skipped, seven warnings in 3,743.65 seconds**, exit 0.
All **497 frozen source/test/benchmark files** retain exact SHA-256, size, mode
and mtime; the unchanged exception-only observer records zero exceptions.
The [command](../evals/ua3.5_closeout_2026-10-10/diagnostics/full-regression-final-recheck-command.json),
[complete log](../evals/ua3.5_closeout_2026-10-10/diagnostics/full-lightweight-regression-final-recheck.log)
and [validation](../evals/ua3.5_closeout_2026-10-10/diagnostics/full-regression-final-recheck-validation.json)
retain the exact run. The runner removes every `RUN_*` variable before launch;
reproduction needs fresh scratch paths and the archived exception observer.

The final staged-diff check also removes excess blank lines at EOF in two test
files. Their remaining bytes and ASTs, including source locations, match the
tested versions exactly; this post-regression formatting cleanup changes no
behavior. The [whitespace review](../evals/ua3.5_closeout_2026-10-10/diagnostics/post-regression-whitespace-review.json)
preserves both original tested files and their checksums.

Of **595 original repository files**, the 13 declared closeout source/test/README
changes, including those two formatting cleanups, are reviewed; the other **582**
retain SHA-256, size, mtime and mode.
All **1,161 original evaluation files**, **198 scientific outputs**, **208 real
state files**, retained UA3.5.5 science/preparation/input files (23/7/2), and
all **117 scientific-owner/schema files** remain exact. All 29 preexisting
benchmark files remain unchanged. The original EpiZoo repository stays clean at
`029cd631d0f9806a646c4a3b42ce10b958f2b67f`. The original checkpoint retains its
5,231,645,507-byte size, mode, mtime and established identity; it is neither
rehashed nor loaded. Four original small human/mouse auxiliaries match their
admitted pins. Credentials and external operator fields are untouched; operator
SHA-256 stays `50ddd7846dd276f11465f98040b285218fb459acbe480bb8aa90350f2283e880`,
mode `0664`. No private configuration, biological dataset, checkpoint or generated
workspace is staged. Local evaluations remain untracked.

[Preservation audit](../evals/ua3.5_closeout_2026-10-10/diagnostics/before-commit-preservation.json)
and [staged-content review](../evals/ua3.5_closeout_2026-10-10/diagnostics/staged-content-review.json)
record the exact allowlist, content/private-value screening, staged diff checks
and original-file invariance. The index remains byte-identical until deliberate
allowlist staging. Earlier uncommitted implementation is retained in the single
consolidated change; nothing is reset, cleaned, stashed, rebased or discarded.

## Disposition and next priority

**UA3.5 is complete within this accepted scope.** Review and required regression
pass. Publish the 68 reviewed files in exactly one commit on `main`,
`Complete UA3.5 conversational scientific workflow`, followed by normal push to
`origin/main`. The [publication receipt](../evals/ua3.5_closeout_2026-10-10/diagnostics/publication.json)
and final response record the resulting commit SHA, exact committed inventory
and local/remote alignment; those post-commit identities are outside this
self-contained commit. No force push or UA3.6 work is authorized by this record.

Scoped completion establishes the accepted interaction mechanism and qualified
user task; it does not imply universal 23-tool production readiness. The next
high-value bounded expansion is exact string declaration binding for `label_key`,
`reference_label_key` and `ground_truth_label_key`, through the existing reviewed
binder and scientific column-role checks. Evaluator/multi-H5AD upload composition
needs its own scope. This recommendation starts no new work or UA3.6.
