# UA3.5.4c — Generic output selection contract repair

UA3.5.4c is complete within its contract-repair and bounded admission scope.
The real A3→A4 mouse continuation passes: the model selects **12 uniquely named,
exact offered outputs across four planned steps**, and normal Application admission
reaches the existing evaluation executor. That executor deliberately withholds the
whole plan before tools. This is **2 PASS, 0 FAIL, 0 BLOCKED** for the stated
checks, not successful EpiZoo inference or a scientific result. UA3.5 remains open;
UA3.6 has not begun. No commit or push occurred.

## Baseline and preservation

Entry and final branch are `main`. HEAD, local `main` and local `origin/main`
remain `8bcad5b42cb812c89dfc232e3ceadea2fc5e44a7`; the clean index remains
byte-identical. Entry contains 30 modified tracked files, 20 new non-evaluation
implementation/test/report files and existing untracked evaluations. Final state
contains the same 30 modified tracked files and 24 new such files, plus evaluations.
All previous UA3.5.1–UA3.5.4b deltas are preserved. The entry snapshot is
`/tmp/agent-ua354c-preservation-mcdm_dgg`.

Of **584 original repository files**, only the already-modified
`src/agent/application/dialogue_execution.py` changes in this task; the other 583
remain exact in bytes, size, mtime and mode. All **941 original evaluation files**,
198 existing scientific outputs, 208 real-state files and the operator configuration
remain exact. All 117 initial scientific-owner/schema files are unchanged.
The checkpoint's size/mtime/mode and previously pinned digest remain unchanged;
the large binary hash is reused rather than recomputed. All 24 reused fixture
files and the registered H5AD's source digest remain exact. New workspaces are
isolated; no datasets or checkpoints are added to Git. No reset, clean, stage,
fetch, commit, push or EpiZoo repository modification occurs.

## Root cause and authoritative owners

The preserved [UA3.5.4b call 14](../evals/ua3.5.4b_live_handoff_2026-10-10/live/call-14.json)
is valid JSON but selects 32 entries, eight distinct repeated names with nine
extra duplicate entries, and `output_key="input_analysis_path},{"`, absent from
the offered `cluster_cells` step. It uses 622 completion tokens, below the limit.
Species resumption, compilation and Planner-internal preflight had already passed;
output validation correctly rejects the response before final Application admission.

`dialogue_execution.execute.accept` owns the actual-plan output context and
validation. It offers each exact step ID, tool and Registry-derived required
result keys. The existing closed schema permits 1–32 items containing `name`,
`step_id` and `output_key`. Every selected pair must belong to that offered step;
all three fields must be nonblank strings; names must be unique across the new
Revision. `SessionTurn` and `AnalysisRevision` preserve that uniqueness contract.
Names have no identifier grammar or length limit, and uniqueness compares exact
strings. Selecting the same pair under distinct names remains representable.

The previous instruction only said to select exact named outputs and avoid adding
steps. It omitted global name uniqueness and relevant-subset guidance. The raw
unoffered key is a model-authored semantic error; the missing generic instructions
are a contributing contract gap. Neither warrants approximate matching, key repair,
renaming, a smaller arbitrary output cap or automatic plan completion.

## Minimal implementation

Only one production file changes:

| File | Responsibility and change |
| --- | --- |
| `src/agent/application/dialogue_execution.py` | Explain the existing 1–32 bound, exact offered step/key pairs, nonblank Revision-wide unique names, a meaningful subset of requested results, and preservation of explicitly requested valid multiple outputs. |
| Same existing owner | After `model.complete()` returns, classify decoding/selection-validation `ValueError` through existing `PlannerError/PLANNER_OUTPUT_INVALID`, using `INTERNAL_AGENT_ERROR` consistently with the existing Planner. Provider dispatch, resource checks and Session linkage stay outside this catch. |

The schema, candidate expression and original validation conditions are unchanged,
as independently checked in [deterministic review](../evals/ua3.5.4c_output_selection_2026-10-10/diagnostics/deterministic-contract-review.json).
The existing `NO_AUTOMATIC_RECOVERY` policy already handles this code; no recovery,
retry or error-taxonomy addition is made. Previously rejected selections reported
`PLANNER_UNEXPECTED_ERROR` and “Planner raised an unexpected orchestration error.”
New rejected selections use `PLANNER_OUTPUT_INVALID` and existing safe text
“The operation failed validation or execution.” Historical records remain exact.

New files are [contract tests](../tests/application/test_output_selection_contract.py),
[HTTP tests](../tests/web/test_output_selection_api.py), the
[narrow evaluation entrypoint](../benchmarks/interactive/run_ua354c.py), and this report.
The entrypoint inherits frozen UA3.5.4a/4b discovery, transport, recording, guards,
explicit zero-recovery policy and bounded executor. Only its evaluation-local
schedule becomes A3/A4. It creates one ordinary Session with the retained registered
H5AD; no QC/neighbors fixture construction or production patch occurs. Prior fixture
metadata is recorded by digest only. All semantic responses in the live run are real.

## Deterministic validation

The new Application suite passes **20 tests in 2.32 seconds**, exit 0; the new HTTP
suite passes **2 in 1.72 seconds**, exit 0. They cover valid single/multi-step choices,
exact scoped pairs, unique names, rejected duplicate names/unoffered keys/fabricated
pairs, relevant subsets and explicitly requested multiple results. Compatibility
checks preserve exact-string names and duplicate pairs with distinct labels.
Malformed JSON, closed-object violations and blank names remain rejected. Invalid
selections run no executor, link no Session turn and create no Revision. A provider
`ValueError` remains outside the new classification catch.

The recorded four-step candidate and all 32 rejected selections are reconstructed
exactly in a structural replay through tiny-H5AD compilation/admission. This is not
reproduction of the original qualified resources, pending continuation or inference.
Prompt/schema assertions and deterministic response validation are tested separately;
scripted choices do not qualify language-model accuracy.

Phase 1 passes **651 tests, three existing warnings in 166.99 seconds**, exit 0:

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 /home/likeyi/anaconda3/envs/agent/bin/python -B -m pytest tests/application/test_prerequisite_planner_handoff.py tests/application/test_recovery_policy_submission.py tests/application/test_clarification_wording.py tests/application/test_turn_decisions.py tests/application/test_dialogue_provider_contract.py tests/application/test_clustering_parameter_evidence.py tests/application/test_h5ad_composition.py tests/application/test_scientific_binding_contract.py tests/application/test_scientific_binding_safety.py tests/application/test_scientific_parameter_continuation.py tests/application/test_selection_parameter_evidence.py tests/web/test_species_handoff.py tests/web/test_leiden_resolution.py tests/web/test_scientific_binding.py tests/web/test_selection_clarification.py tests/unit/orchestration/test_registry.py tests/unit/orchestration/test_semantic_prompt.py tests/unit/orchestration/test_semantic_compiler.py tests/unit/orchestration/test_planning_diagnostics.py tests/unit/orchestration/test_llm_planner_v4.py tests/unit/orchestration/test_llm_planner_v4_recovery.py tests/tools/analysis/test_embedding_analysis.py tests/application/test_output_selection_contract.py tests/web/test_output_selection_api.py tests/unit/orchestration/test_error_policy.py -q -p no:cacheprovider --basetemp /tmp/agent-ua354c-phase1-9sgl_7zb/pytest-tmp --tb=short
```

It includes the previous focused UA3.5.1–4b files plus new contract/HTTP cases and
existing error-policy coverage. Species, Leiden explicit/default values, selection
partial answers/corrections, invalid parameters, named-backend rejection, pending
identity and provenance regressions pass. Opt-in `RUN_*` gates are disabled and
live-provider dispatch is zero.

The first full Application/Web run records **1 failed, 2,045 passed, nine skipped,
three warnings in 557.72 seconds**, exit 1:

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 /home/likeyi/anaconda3/envs/agent/bin/python -B -m pytest tests/application tests/web -q -p no:cacheprovider --basetemp /tmp/agent-ua354c-phase2-wfte3sx5/pytest-tmp --tb=short
```

`test_registered_bam_unfinished_raw_recovery_checks_full_source[deleted]` fails
while creating its interrupted-intake fixture, before source deletion. Its Run
stops at `PLANNING_SCOPE_ACCEPTED`, with no detailed Planner dispatch or output
selection, no plan/steps/errors, and a generic facade `INTERACTIVE_APPLICATION_FAILED`.
The underlying exception was not captured. No causal attribution to the output
selection repair, or claim of a pre-existing/flaky defect, is made.

The unchanged whole BAM file then passes **33 tests in 34.69 seconds**. The complete
Application/Web recheck passes **2,046 tests, nine skipped, three existing warnings
in 551.19 seconds**, exit 0, without exclusions or source/test changes:

```bash
UA354C_ADMISSION_EXCEPTION_LOG=/tmp/agent-ua354c-phase2-diagnostic-vabc34uk/admission-exceptions.jsonl PYTHONPATH=src:/tmp/agent-ua354c-test-observer PYTHONDONTWRITEBYTECODE=1 /home/likeyi/anaconda3/envs/agent/bin/python -B -m pytest tests/application tests/web -q -p no:cacheprovider --basetemp /tmp/agent-ua354c-phase2-diagnostic-vabc34uk/pytest-tmp --tb=short -p ua354c_admission_observer
```

Its temporary pytest observer delegates `AnalysisSessions.respond` once, records
any targeted exception and rethrows it unchanged. It changes no decisions, results,
recovery or acceptance; it records zero exception rows. Both full-run results and
commands, the isolated BAM command, observer source/configuration and exact failed
records are preserved in [diagnostics](../evals/ua3.5.4c_output_selection_2026-10-10/diagnostics/).
The original exception remains unexplained; no production or test repair was made
for it. All **466 frozen source/test/static/resource files** remain unchanged during
each full/focused run. No full repository or heavyweight scientific regression is claimed.

Six narrow native Firefox checks pass, exit 0:

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 /home/likeyi/anaconda3/envs/agent/bin/python /tmp/agent-ua354c-browser/native_output_selection_check.py
```

They check the changed error bubble/code, refresh, exact retry/polling, reopen,
fresh-server restoration and JavaScript errors. There is one scripted model
construction, four semantic calls, zero live providers/science, one failed planning
attempt and no Session turn links/Revisions. Reads/retries construct no extra model
or scientific work. Scripts, logs and screenshots are in the
[browser archive](../evals/ua3.5.4c_output_selection_2026-10-10/diagnostics/native-browser/).
This is presentation acceptance, separate from the CLI live qualification.

## Bounded live mouse continuation

The exact accepted development-primary profile remains
`f1-openrouter-gpt-5-6-sol`, `openrouter / openai/gpt-5.6-sol`, strict Chat
Completions, 60-second timeout. Fresh discovery makes one catalog lookup and
confirms exact availability and existing public prices. All seven responses
identify the requested model, one upstream attempt and no modified pipeline.
Operator Groq defaults, credentials and resource configuration are unchanged.

There are **7 interface calls and 7 provider completions**: two Interpreter,
two capability selection, two detailed Planner and one output selection.
Provider failures, guard blocks, SDK retries, semantic repairs and failovers are
zero. Usage is **15,976 prompt tokens and 2,229 completion tokens**;
provider-reported cost is **$0.0615105** and conservative reservation **$0.489582**
under the unchanged $5 ceiling. Other inherited limits remain 48 completions,
six per turn, 4,096 output tokens, 150,000 conservative input byte/token units and
one dispatch per turn/schema. Exact routing, strict parameters and no fallback
remain enforced. No balance claim is made. New artifacts are screened against
configured secret values; SDK headers/client representations are not saved.

The exact command, dispatched once after the passing recheck, is:

```bash
PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 conda run --no-capture-output -n agent python -B -m benchmarks.interactive.run_ua354c --output evals/ua3.5.4c_output_selection_2026-10-10/live --workspace /tmp/agent-ua354c-live-20261010 --source-h5ad /tmp/agent-ua322-real-ynobks9w/Fang2021-first32-full-features.h5ad --operator-config /home/likeyi/program/agent-web-config/operator-web.json --fixtures-json /tmp/agent-ua354a-fixture-smoke-vn2q5o7o/smoke.json
```

[Raw calls and per-case records](../evals/ua3.5.4c_output_selection_2026-10-10/live/results.json)
retain decisions independently of their validation.

| Case / actual request | Live semantics | Agent-owned outcome | Disposition |
| --- | --- | --- | --- |
| A3: “Analyze this dataset with the available EpiZoo model.” | Execute EpiZoo; supported four-step candidate omits species | Compiler `MISSING_REQUIRED_SOURCE/species`; durable pending origin `a3`, exact source and `device=cuda`, generation 0; no execution | **PASS**, prerequisite handoff |
| A4: “The cells came from mice.” | `answer_prerequisite/@species/mouse`; resumes the original `a3` task; live four-step candidate and 12-output subset | Qualified mouse pins, compiler/Planner-internal preflight, exact selection validation and normal Application resource guard/linkage pass; executor preflight passes and whole plan is intentionally withheld | **PASS**, bounded admission only |

A4 preserves the registered sparse 32-cell, 1,341,077-feature source digest
`c41e93a8e9ca71025f13f3c3cea7fd27693b0b3d6f754814616ab53622b48c26`, original
task, `device=cuda` and exact `mouse-recorded-qualified` checkpoint/frequency/filter
pins. Its four steps are embedding → neighbors → Leiden → UMAP, authored by the
real Planner. No new numeric/default policy is supplied.

The selector chooses 12 of 56 offered fields: embedding/cell-ID products and
bounded metadata, neighbor analysis/count, clustering analysis/count/key and UMAP
analysis/key/dimensions. Names distinguish repeated `analysis_path` keys across
steps; every exact pair is offered. Accepted `SessionTurn.selections` equal this
response. The nonempty persisted plan and ordinary turn link establish that the
Application's resource guard and output validation completed.

Planner trace records `SEMANTIC_CANDIDATE_COMPILED`,
`CANDIDATE_PREFLIGHT_PASSED` and `FINAL_PLAN_ACCEPTED`. Separately,
[execution-boundary.json](../evals/ua3.5.4c_output_selection_2026-10-10/live/execution-boundary.json)
records actual executor preflight `passed=true`, `delegated=false` and withholding
of the whole plan before any tool. **Persisted `Run.preflight_verification` is null**:
the frozen evaluation executor returns without the normal execution checkpoint.
The executor observation and Planner trace prove their respective checks; no
passed durable-preflight field is invented.

The facade and linked turn remain `failed` with `EVALUATION_EXECUTION_WITHHELD`;
normal run verification also records `MISSING_STEP_RESULT` because no results were
produced. These expected consequences of the bounded executor are preserved,
not renamed into scientific success. There are zero scientific steps/Revisions
and generation remains 0. No EpiZoo inference, GPU workload, scientific-owner
production or independent scientific result verification runs in this live sequence. Actual foundation-resource
consumption and feature compatibility remain unqualified by this admission witness.

## Architecture and next work

No second Interpreter, Planner or output-selection system, per-tool language/output
parser, output-name heuristic, new default engine, fabricated authority, invalid-key
replacement or provider-specific rule is added. Scientific owners, Registry/ToolSpec,
compiler, binding, pending state, Session/Revision, resource qualification and
scientific provenance contracts remain unchanged. No 1000/4/20 policy is introduced.
The prior live matrix is preserved; Leiden, Cell Selection and negative behavior
are protected by deterministic regression rather than repeated score-seeking live calls.

The previous output-selection admission blocker is requalified within this one
real four-step continuation. Broader semantic reliability, owner-approved policies
and practical capability readiness remain UA3.5 work. The smallest next step is to
use the accepted tool/parameter inventory to select one bounded resource/default
readiness item with its existing scientific owner. Investigate the uncaptured BAM
fixture exception if it recurs; this task establishes no justified broader fix.
Full live-provider + real-science + browser acceptance and foundation consumption
remain UA3.6 work. No subsequent work is begun and nothing is committed or pushed.
