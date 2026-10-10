# UA3.5.4b — Prerequisite handoff and focused integration repair

UA3.5.4b is complete within this repair and bounded requalification scope. The
three previously rejected initial requests now reach existing compiler-owned
clarification. The complete live Cell Selection partial-answer/correction chain
executes and independently verifies. Leiden explicit/default behavior remains
intact. Mouse species binding and resumption pass, but its final admission is
still unqualified because the real output selector returns invalid selections.
The 14-case disposition is **13 PASS, 1 FAIL, 0 BLOCKED** with the limits below.
UA3.5 is not closed and UA3.6 has not begun. No commit or push occurred.

## Baseline and minimal changes

Entry and final branch are `main`; HEAD, local `main` and local `origin/main`
remain `8bcad5b42cb812c89dfc232e3ceadea2fc5e44a7`. The index is clean and
byte-identical. Entry contains 28 modified tracked files, 15 new implementation,
test/documentation/benchmark files and existing untracked evaluations. Final
state contains 30 modified tracked files and 20 new such files, plus evaluations.
Earlier changes are retained; no reset, clean, stage, fetch, commit or push occurs.
The entry snapshot is `/tmp/agent-ua354b-preservation-u8h05grg`.

The owning ambiguity was the scoped-v4 instruction in
`src/agent/orchestration/semantic_prompt.py`: “Unsupported means genuine
capability/input insufficiency only.” Recorded A1/A3/C1 detailed responses
identify the correct operation and missing species/thresholds, then choose that
offered branch. No candidate reaches compiler diagnosis or Session clarification.
The unchanged wire schema already represents the necessary absent sources.

Only four production files change in this task:

| File | Minimal change and owner |
| --- | --- |
| `src/agent/orchestration/semantic_prompt.py` | Distinguish an otherwise legitimate supported candidate awaiting explicit user declarations/scalar values from genuinely unsupported inputs/capabilities. Omit unavailable declaration sources, without inventing values; compiler and existing Application alone decide clarification eligibility. |
| `src/agent/application/interactive.py` | Use existing `PlanningRecoveryPolicy.to_dict()` for explicit-policy configuration hashing, replacing `asdict()` that retained a non-JSON `frozenset`. Default-`None` digest is unchanged. |
| `src/agent/application/turn_decisions.py` | One generic instruction preserves explicitly named backend/algorithm/implementation requirements, while permitting suitable registered choices for general operations. Unavailable named choices clarify instead of silently substituting. |
| `src/agent/application/responses.py` | Remove nonexistent retained/pending-state claims from invalid-value text; use generic unsupported-intent wording. Existing biological-answer wording and persisted history remain unchanged. |

The incomplete-candidate exception explicitly excludes required scientific
artifacts, reference/checkpoint resources, upstream authority, incompatible
lineage, ambiguous source identity and invalid DAGs. A candidate grants no
execution permission. No schema, parser, compiler, Registry, prerequisite binder,
Session state machine, scientific owner or algorithm changes are made. The
`scientific_parameter` flag is not treated as blanket clarification eligibility;
it also describes resources. Existing typed and owner checks remain authoritative.

New tests are `tests/application/test_prerequisite_planner_handoff.py`,
`test_recovery_policy_submission.py` and `test_clarification_wording.py`.
`tests/unit/orchestration/test_semantic_prompt.py` updates the changed instruction
contract. The handoff witnesses reproduce exact A1/A3/C1 utterances, input-name/
type shapes and missing-port boundaries, not a byte-identical replay of every
historical capability scope. Selection structural-context tests claim compiler
reachability only; existing genuine QC continuation suites qualify owner behavior.

The new [evaluation entrypoint](../benchmarks/interactive/run_ua354b.py) reuses the
unchanged UA3.5.4a runner, matrix, tiny owner fixtures and execution boundary. An
evaluation-local constructor factory supplies the ordinary application's existing
explicit scoped policy with zero retry/repair/failover actions and two planning
calls. No provider response, prompt, owner or production constructor is patched.
This avoids unnecessary repair requests; production recovery defaults are unchanged.

## Staged deterministic acceptance

Phase 1 passes **611 tests, three existing dependency warnings in 168.45 seconds**,
exit 0. It includes all prior UA3.5.3 focused files, semantic-prompt coverage and
the 29 new cases. The exact command is:

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 /home/likeyi/anaconda3/envs/agent/bin/python -B -m pytest tests/application/test_prerequisite_planner_handoff.py tests/application/test_recovery_policy_submission.py tests/application/test_clarification_wording.py tests/application/test_turn_decisions.py tests/application/test_dialogue_provider_contract.py tests/application/test_clustering_parameter_evidence.py tests/application/test_h5ad_composition.py tests/application/test_scientific_binding_contract.py tests/application/test_scientific_binding_safety.py tests/application/test_scientific_parameter_continuation.py tests/application/test_selection_parameter_evidence.py tests/web/test_species_handoff.py tests/web/test_leiden_resolution.py tests/web/test_scientific_binding.py tests/web/test_selection_clarification.py tests/unit/orchestration/test_registry.py tests/unit/orchestration/test_semantic_prompt.py tests/unit/orchestration/test_semantic_compiler.py tests/unit/orchestration/test_planning_diagnostics.py tests/unit/orchestration/test_llm_planner_v4.py tests/unit/orchestration/test_llm_planner_v4_recovery.py tests/tools/analysis/test_embedding_analysis.py -q -p no:cacheprovider --basetemp /tmp/agent-ua354b-phase1-mmlwu97v/pytest-tmp --tb=short
```

These tests distinguish recorded `unsupported` rejection from supported incomplete
candidate diagnosis, exact pending creation and fail-closed artifact/resource/source/
graph errors. Existing suites preserve species continuation, numeric partial answers,
atomic invalid/corrected thresholds, input identity, refresh/restart/idempotency,
Leiden defaults and evidence. Explicit-policy tests prove submission/validation,
deterministic serialization, the prior default-`None` digest and zero dispatch for
invalid submissions. Wording tests preserve immutable historical text. No existing
UA3.5.1–UA3.5.3 regression is observed.

Phase 2 passes **2,024 tests, nine skipped, three existing dependency warnings in
552.26 seconds**, exit 0, without exclusions:

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 /home/likeyi/anaconda3/envs/agent/bin/python -B -m pytest tests/application tests/web -q -p no:cacheprovider --basetemp /tmp/agent-ua354b-phase2-2erurgmg/pytest-tmp --tb=short
```

Both phases disable opt-in `RUN_*` gates, dispatch no live providers and preserve
all 464 frozen source/test/static/resource files during execution. No full
lightweight repository or heavyweight scientific acceptance is claimed.

Nine targeted native Firefox presentation checks pass, exit 0:

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 /home/likeyi/anaconda3/envs/agent/bin/python /tmp/agent-ua354b-browser/native_response_check.py
```

They check corrected N1/N2 bubbles, refresh, retry, reopen, fresh-server restoration
and JavaScript errors. Exactly two scripted Interpreter calls occur; Planner,
live-provider and science calls are zero, with no Runs/Revisions. Historical bad
wording remains exact in the deterministic reopen/retry fixture. No browser/science/
live-provider combined acceptance is implied. Native artifacts remain under
`/tmp/agent-ua354b-browser/run-yrm9dw6z`, with exact copies of the check script,
screenshots and logs in the [browser evidence archive](../evals/ua3.5.4b_live_handoff_2026-10-10/diagnostics/native-browser/).

## Bounded live acceptance

Fresh exact-model discovery confirms development-primary
`f1-openrouter-gpt-5-6-sol`, `openrouter / openai/gpt-5.6-sol`, strict Chat
Completions, 60-second timeout. All 38 responses identify that exact model and
report one upstream attempt, no modified pipeline and no fallback. The operator's
Groq configuration remains unchanged. Every semantic decision is live; no scripted
continuation, semantic repair, retry or failover supplies acceptance.

There are **38 interface calls and 38 provider completions**: 14 Interpreter,
9 capability selection, 9 detailed Planner and 6 output selection, plus one
catalog lookup. Provider failures and guard blocks are zero. Usage is 96,597
prompt tokens and 8,303 completion tokens; provider-reported cost is **$0.321828**.
The conservative reservation is **$2.770500** under the unchanged $5 ceiling.
Other unchanged bounds are 48 total completions, six per turn, 4,096 output
tokens, 150,000 conservative input byte/token units, SDK retries zero, exact
routing and one dispatch per turn/schema. Credentials and SDK headers are never
saved; artifact screening finds no configured secret values.

The [live record](../evals/ua3.5.4b_live_handoff_2026-10-10/live/results.json)
links raw prompts, schemas, responses and per-case Session/run/evidence records.
[Diagnostics](../evals/ua3.5.4b_live_handoff_2026-10-10/diagnostics/) retain exact
commands, logs, independent review and preservation. The runner uses the same
registered sparse 32-cell mouse H5AD, `device=cuda`, unchanged qualified mouse
catalog and genuine synthetic-only QC fixture as UA3.5.4a. No new biological
threshold policy or resource is admitted.

| Case / actual utterance | Live interpretation and candidate | Deterministic diagnosis, binding and resumption | Scientific/resource boundary | Status / owner |
| --- | --- | --- | --- | --- |
| A1: “Use this dataset to run the supported EpiZoo analysis.” | Execute EpiZoo; supported four-step candidate, species omitted | Compiler `MISSING_REQUIRED_SOURCE/species`; durable `missing_species` pending for `a1`; original source/device retained | Zero execution/revision | **PASS** |
| A2: “These cells are from human.” | `answer_prerequisite/@species/human`; supported embedding candidate | Exact `a1` task resumed with human; original source/device retained | `EPIZOO_RESOURCE_REQUIRED`; no compatible qualified human default; zero science | **PASS**, correct resource rejection (category 4) |
| A3: “Analyze this dataset with the available EpiZoo model.” | Execute EpiZoo; supported embedding candidate, species omitted | Same compiler diagnosis; exact `a3` pending/input/device | Zero execution/revision | **PASS** |
| A4: “The cells came from mice.” | `answer_prerequisite/@species/mouse`; supported four-step candidate | Exact `a3` task resumed; qualified mouse pins selected; compiler and Planner-internal candidate preflight pass | Output selector emits duplicate names and an unoffered key; later Application resource/durable execution admission never reached | **FAIL**, output-selection model/contract; species binding passes |
| B1: “Cluster these cells using Leiden with resolution 0.7.” | Scoped `cluster_cells.resolution=0.7`; one clustering step | Exact span `[49,52)`; validated actual consumer | Verified **0.7**, explicit-user provenance | **PASS** |
| B2: “Cluster these cells using Leiden.” | Execute; no resolution declaration/source | No unnecessary clarification; no resolution in admitted/compiled/resolved inputs | Verified owner default **1.0**, `existing_owner_default` | **PASS** |
| B3: “Group these cells with Leiden, setting its granularity to 0.7.” | Resolution interpreted from paraphrase; one step | Exact span `[58,61)`; correct scoped consumer | Verified **0.7**, explicit-user provenance | **PASS** |
| C1: “Select cells from these barcode QC results.” | Execute selection; supported candidate omits both thresholds | Compiler identifies exactly depth/TSS; pending origin/binding `c1`; exact QC pins retained | Zero selection before prerequisites complete | **PASS** |
| C2: “Keep barcodes with at least 1 QC fragment record.” | `answer_parameters`, depth literal `1`, span `[28,29)` | Owner-validated integer 1; pending binding `c2`, only TSS missing | Zero selection; no new Run/revision | **PASS** |
| C3: “Actually, require at least 2 QC fragment records instead.” | Scoped depth correction `2`, span `[27,28)` | Owner-validated replacement; binding `c3`; TSS still missing; QC identity unchanged | Zero selection; no new Run/revision | **PASS** |
| C4: “Use a minimum TSS enrichment of 0.5.” | `answer_parameters`, exact string `0.5`, span `[32,35)`; supported selection plan | Original `c1` task resumes with depth 2/TSS `0.5`; original QC pins | One real selection and independent verification; active Revision, generation 1 | **PASS** |
| N1: “Cluster these cells using Leiden with resolution -0.7.” | `clarify/invalid_parameter_value` | Corrected generic text; no substitution/pending fabrication | Zero Planner/science; owner invalid-value path unexercised live | **PASS**, bounded refusal only |
| N2: “Run two independent Leiden clusterings on these cells; use resolution 0.7.” | `clarify/unsupported_intent` | Corrected generic text; no guessed scope | Zero Planner/science; actual repeated-consumer binding unexercised live | **PASS**, bounded refusal only |
| N3: “Use EpiAgent to embed the cells in this dataset.” | `clarify/unsupported_intent`, without EpiZoo substitution | Registered capability boundary retained | Zero Planner/science; one focused backend refusal, not broad reliability | **PASS** |

C2/C3 preserve the original objective, QC manifest/digest, generation and resource
context. C4's accepted evidence records depth **2**, TSS **1/2**, both
`explicit_user_instruction` and `explicit_execution_argument`. Optional flank,
maximum-depth and nucleosome conditions remain `null` with owner-default origin.
No values 1000/4/20 are applied. Synthetic resource qualification and unassessed
statistical cell calling remain explicit. Four live plans reach execution: three
tiny clusterings and one selection; each passes normal verification. No foundation
inference, GPU work, upstream production rerun or scientific-policy calibration occurs.

## Residual failure and smallest next step

A4 proves live species interpretation, original-task resumption, exact input/device/
mouse resource selection, compilation and **Planner-internal preflight**. Its
output-selection call returns 32 entries including eight repeated names and
`output_key="input_analysis_path},{"`, which is absent from the offered step.
Existing exact-pair and unique-name validation rejects it. The facade reports
`PLANNER_UNEXPECTED_ERROR` with `IntentError`; the durable Run retains no plan,
steps or preflight result and the evaluation executor is never called. These
later absences do not erase the earlier Planner-preflight trace. No withheld
EpiZoo execution or successful final resource admission is claimed for A4.

The unoffered key is a **category 1** model-authored selection error. A contributing
**category 2** contract gap is that the existing selector prompt does not explain
unique revision-wide names or choosing a bounded relevant subset across multiple
steps. This response uses 622 completion tokens, below the limit, with no provider
failure: it is not truncation, infrastructure or a species-binding defect. No
per-prompt fix, additional attempt or relaxed validator is used to force a pass.

The smallest next UA3.5 task should clarify that existing generic output-selection
contract, preserve exact offered-pair/unique-name checks, and qualify one bounded
multi-step mouse continuation. More accurate typed reporting of rejected selections
can be reviewed at the same existing Application boundary, without a message taxonomy
or workflow redesign. This residual precedes broader qualified-resource/policy and
practical capability-readiness work. Full live-provider + real-science + browser
acceptance, foundation resource consumption/feature compatibility and broader
language/biological reliability remain UA3.6 work.

Historical rejected-plan messages remain immutable. N2 does not exercise the
accepted `AMBIGUOUS_PARAMETER_SCOPE` path; repeated-operation scope still fails
closed and requires a narrowed request. The targeted browser checks establish
correct current wording and preserved history, not a stale-state or workflow fix.

## Preservation

All **811 original evaluation files**, including every UA3.5.4a artifact, remain
exact in SHA-256, size, mtime and mode. The same holds for 198 existing scientific
outputs, 208 real-state files and operator configuration. All **117 initial
scientific-owner/schema files** are byte-identical. Of 579 original repository
files, only the four declared production files and one prompt-contract test change;
the other 574 remain exact. Earlier uncommitted deltas are preserved, with only
the documented additions layered over them. Reused fixture bytes and source
registration pins remain exact. Checkpoint size/mtime/mode and previously pinned
identity are unchanged; the large binary digest is reused, not recomputed.

No EpiZoo repository change, second Interpreter/Planner, per-tool language parser,
new default engine, parallel pending store, fabricated authority, automatic resource/
assembly inference, scientific algorithm change or 1000/4/20 policy is introduced.
New evidence/scientific workspaces are isolated; credentials and existing operator
state are untouched. The index and aligned refs remain exact; nothing is committed
or pushed.
