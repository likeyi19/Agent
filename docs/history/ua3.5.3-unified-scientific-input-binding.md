# UA3.5.3 — Unified Conversational Scientific Input Binding

UA3.5.3 extends the existing conversational input seam to canonical registered
scientific scalar parameters. It preserves the accepted species and Leiden
interactions and demonstrates first Cell Selection with two required thresholds,
partial answers, scoped corrections and durable continuation. Interpreter and
Planner decisions remain semantic proposals; Registry, compiler/preflight and
scientific owners retain execution and scientific authority.

## Repository baseline and preservation

Initial HEAD, `main` and local `origin/main` all equal
`8bcad5b42cb812c89dfc232e3ceadea2fc5e44a7` on `main`. The initial index is clean.
The worktree has 25 modified tracked files, five new implementation/documentation
files, and the existing untracked `evals/`. These are the actual inspected
UA3.5.1/UA3.5.2 changes, matching their records. Before edits, their bytes,
worktree diff and index are copied to
`/tmp/agent-ua353-preservation-n9prp96c/previous-work` and adjacent records.
No divergence, fetch, pull, reset, clean, stage, commit or push occurs.

The initial preservation inventory matches all 705 evaluation files, 198
scientific outputs, 208 existing application/Web state files and the operator
configuration, including SHA-256, size, mtime and mode. The original 5.23 GB
checkpoint retains its exact size/mtime/mode; its previously verified digest is
retained without rereading the large binary. All new execution and browser
fixtures live under isolated temporary workspaces. EpiZoo's repository and
historical outputs are not modified.

## Existing integration and the shared seam

UA3.5.1 and UA3.5.2 already use one Interpreter, `dialogue_execution.admit`,
`AgentRequest`, the admitted Planner wrapper, semantic compiler, preflight,
runtime and existing Session/Revision records. Their source/resource identities
and received submissions are already immutable. No working engine is replaced.

The missing seams were the declaration restricted to a single Leiden resolution,
its parameter resolver, species-only pending state, and the loss of the compiler's
complete missing-argument list in structured diagnostics. These are extended in
place:

- `ScopedArgument` generalizes the existing tool/argument/literal/character-span
  declaration. Historical `LeidenResolution`, singular `argument` wires and old
  species records remain valid. At most eight distinct declarations are admitted.
- The Interpreter offers exact Registry-derived tool/argument pairs, requiredness
  and registered descriptions, without scientific input values, paths or a new
  default table. It interprets associations and supplies canonical identities.
- Application resolves a scientific scalar's one registered request-source port.
  Integer conversion, finite float conversion and exact decimal/fraction strings
  follow the existing declared types. Registry validation delegates scientific
  domains to the existing owner. No Cell Selection grammar or threshold validator
  is added to conversation code.
- The compiler's existing complete missing-argument calculation is added as
  optional `argument_names` in its diagnostic. The existing sanitized diagnostic
  transport carries these canonical identifiers to Application; planning and
  recovery behavior are unchanged.
- Before execution, the actual compiled plan must contain exactly one consumer
  for each scoped declaration, with its exact accepted value. Another consumer
  sharing that request-input identity fails closed. The binder creates no steps,
  selects no arbitrary duplicate step and supplies no workflow completion.

Scientific input claims remain distinct. A human/mouse species answer uses the
existing species `ArgumentSpec`, exact registered H5AD and qualified resource
selection, rather than the numeric declaration contract. Species does not prove
feature compatibility or supply a missing qualified resource. The shared binder
also handles explicit scientific scalar values in a species answer, including a
validated correction to the same captured parameter.

## Clarification, continuation and defaults

The existing `Interaction.prerequisite` gains a bounded parameter-pending shape:
origin turn, exact latest accepted binding turn, tool, missing canonical names
and existing resource-context digest. Inputs stay in existing admitted records;
there is no second pending store or untyped execution dictionary.

A required-parameter question is grounded in the original compiler finding and
Registry scientific-parameter contract. If another missing prerequisite appears
in that finding, ordinary fail-closed failure remains visible rather than
inventing a policy. Omitted optional values never create a new question.

`answer_parameters` uses the same binder as initial `execute_plan`. An admitted
answer rechecks the originating failed RunStore attempt, original objective,
received inputs, exact pending references and captured Revision/generation.
Partial answers persist accepted inputs in the current immutable admitted record
and ask only for remaining required values. Explicit scoped corrections can
replace an earlier accepted parameter; a mixed invalid answer changes none of
the earlier accepted values. Invalid semantic decisions preserve the pending
binding. New ordinary tasks use their own received inputs and inherit no values
from an offered old clarification.

Once complete, the original objective and exact scientific inputs enter ordinary
planning, compilation, preflight, registered execution, independent verification
and publication. Immutable received submissions remain distinct from admitted
execution projections. Direct Session receipts also retain input fingerprints
for initial declarations and partial/final answers. Duplicate responses do not
replay models or science. A changed generation, dataset, source pin, resource
policy or prerequisite cannot redirect the pending task. Upload/registration
alone still performs no scientific execution.

The Cell Selection owner defines exactly two required thresholds:

| Parameter | Existing contract |
| --- | --- |
| `min_qc_fragment_records` | Required nonnegative integer |
| `min_tss_enrichment` | Required nonnegative integer or exact decimal/fraction string; floats rejected |
| `min_tss_flank_evidence` | Optional; omitted/null disables |
| `max_qc_fragment_records` | Optional; omitted/null disables |
| `max_nucleosome_signal` | Optional; omitted/null disables |

All domains and bounds remain the existing `scatac_selection_profile` owner's
rules through `ThresholdArgument`. No automatic 1000/4/20 policy exists.
Leiden omission retains the owner's effective `1.0`; neighbor/UMAP and other
omitted settings retain their own applicable defaults. Application neither
copies those defaults nor asks the LLM to invent them.

## Effective values and provenance

Effective values remain the accepted owner result/evidence, including exact
selection rational thresholds and disabled optional conditions. The additional
`selection_threshold_origins` projection checks exact accepted planned/resolved
arguments against the existing owner's normalization. It uses the existing
`explicit_execution_argument` and `existing_owner_default` vocabulary and pins
the accepted Run result. Historical incomplete summaries remain readable without
fabricating additional origins.

`explicit_user_parameters` proves user origin only through exact admitted Session
declarations, received utterance spans and recorded continuation links, with
matching final inputs and accepted planned/resolved arguments. Its payload names
the exact tool and step and each canonical parameter. A digest pins immutable
supporting interaction records, so later conversation/display completion changes
no attribution. Matching values in unrelated turns supply no provenance.

Existing reviewed evidence and scientific-answer claims expose these facts,
including nested rational numerator/denominator values. Parameter questions run
no scientific production or reconstruction. Web still only supplies user inputs
and renders conversation/evidence; no frontend parser, parameter dashboard or
scientific validation is introduced.

## Implementation files

UA3.5.3 changes `src/agent/application/turn_decisions.py`,
`dialogue_execution.py`, `turns.py`, `session_state.py`, `responses.py`,
`dialogue_evidence.py` and `scientific_dialogue.py`; and
`src/agent/orchestration/semantic_compiler.py`, `planning_diagnostics.py` and
`llm_planner.py`. README and this record document the stage. Earlier Web,
resource, Registry and scientific-owner UA3.5.1/UA3.5.2 changes remain together
in the unstaged worktree; UA3.5.3 adds no Web source or scientific-owner changes.

New tests are `tests/application/test_scientific_binding_contract.py`,
`test_scientific_binding_safety.py`, `test_scientific_parameter_continuation.py`,
`test_selection_parameter_evidence.py`, and
`tests/web/test_scientific_binding.py`, `test_selection_clarification.py`.

## Acceptance

Language and plan decisions use the existing scripted model-facing interfaces.
Tiny H5AD chains script EpiZoo inference and run real sparse downstream owners,
compiler/preflight, verification, Session and Web paths. Cell Selection fixtures
use existing real external-fragment/QC construction, pinned BEDTools and explicit
`synthetic_only` qualification; the normal selection owner and independent
verifier each execute once on a successful new task. No live-provider language
qualification or real biological threshold calibration is claimed.

The final focused command is:

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 /home/likeyi/anaconda3/envs/agent/bin/python -m pytest tests/application/test_turn_decisions.py tests/application/test_dialogue_provider_contract.py tests/application/test_clustering_parameter_evidence.py tests/application/test_h5ad_composition.py tests/application/test_scientific_binding_contract.py tests/application/test_scientific_binding_safety.py tests/application/test_scientific_parameter_continuation.py tests/application/test_selection_parameter_evidence.py tests/web/test_species_handoff.py tests/web/test_leiden_resolution.py tests/web/test_scientific_binding.py tests/web/test_selection_clarification.py tests/unit/orchestration/test_registry.py tests/unit/orchestration/test_semantic_compiler.py tests/unit/orchestration/test_planning_diagnostics.py tests/unit/orchestration/test_llm_planner_v4.py tests/unit/orchestration/test_llm_planner_v4_recovery.py tests/tools/analysis/test_embedding_analysis.py -q -p no:cacheprovider --basetemp=/tmp/agent-ua353-focused-final2 --tb=short
```

It passes **561 tests, three existing dependency warnings in 145.68 seconds**,
exit 0. Its log is `/tmp/agent-ua353-focused-final2.log`. The focused command
includes the following complete files:

| Acceptance | Test file | Passing cases |
| --- | --- | --- |
| H5AD species, resources, restart and retry | `tests/web/test_species_handoff.py` | 27 |
| Explicit/default Leiden and retained species-pending values | `tests/web/test_leiden_resolution.py` | 32 |
| Shared species-answer binding and scoped correction through downstream owners | `tests/web/test_scientific_binding.py` | 3 |
| First selection, partial/invalid/atomic corrections, subprocess restart, exact consumers, conflicts and unrelated tasks | `tests/application/test_scientific_parameter_continuation.py` | 19 |
| HTTP selection, refresh/fresh server and stale generation | `tests/web/test_selection_clarification.py` | 3 |
| Typed pairs, literal/type bounds and additional existing registered parameters | `tests/application/test_scientific_binding_contract.py` | 52 |
| Valid-checksum record corruption, changed source, direct receipts and conflicting current values | `tests/application/test_scientific_binding_safety.py` | 17 |
| Exact effective thresholds and immutable explicit-user attribution | `tests/application/test_selection_parameter_evidence.py` | 15 |

The selection execution fixture supplies genuine verified QC and omits exactly
the two required thresholds. Explicit acceptance values are depth `1` and TSS
`0.5` (or a scoped depth correction); these fixture values are not biological
policies. Actual owner normalization records TSS `1/2` and omitted flank `null`.
The selection production and independent verifier body execute once per accepted
new selection. Parameter answers, retries, refresh and restart reads run zero
instrumented selection science. Negative cases cannot execute arbitrary steps or
replace an invalid value with a default.

The full affected Application/Web command, without exclusions, is:

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 /home/likeyi/anaconda3/envs/agent/bin/python -m pytest tests/application tests/web -q -p no:cacheprovider --basetemp=/tmp/agent-ua353-regression-final2 --tb=short
```

It passes **1,995 tests, nine skipped, three existing dependency warnings in
523.79 seconds**, exit 0. Its log is `/tmp/agent-ua353-regression-final2.log`.
All `RUN_*` gates are disabled. No complete lightweight repository regression
is run. The final 460-file source/test/static/resource freeze remains exact
through both final commands. `git diff --check` passes.

Native Firefox command:

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 /home/likeyi/anaconda3/envs/agent/bin/python /tmp/agent-ua353-browser/native_check.py
```

All 12 checks pass, exit 0: input selection without execution; two-threshold
question; partial answer; invalid answer; refresh; fresh application/server;
ordinary continued execution; accepted parameter/provenance evidence; read-only
parameter answer; accepted-history refresh; and no application JavaScript
errors; and ambiguous repeated consumers with exact retry and no science.
Selection production and independent verification each execute once;
one QC deep verification occurs during execution. Read/refresh operations replay
no providers or science. Evidence, timeline, Firefox log and five screenshots are
under `/tmp/agent-ua353-browser/run-digi3ab3`, together with the additional
`ambiguous-scope.png` screenshot. Language calls remain scripted.

## Scope and next stage

The reusable boundary supports offered scientific scalar contracts with a unique
registered request-source identity and unambiguous actual consumer. It does not
make arbitrary species/resource/design declarations numeric parameters, resolve
duplicate same-tool step scopes, qualify every registered parameter, or establish
live-provider semantic reliability. Unsupported contracts fail closed. Scientific
algorithms, authority semantics, Registry inventory and EpiZoo are unchanged.
Accepted-value dialogue remains bounded to reviewed evidence projections; binding
a new scalar identity does not invent its scientific result projection. An
ambiguous compiled plan produces a typed scope clarification while preserving
the rejected RunStore attempt. The current browser also shows that attempt's
failure banner. A narrowed new request is needed for duplicate step scopes;
Agent performs no automatic scope selection.

## Completion and final preservation

UA3.5.3 is complete within this integration/software scope and ready for review.
Final HEAD, `main` and local `origin/main` remain
`8bcad5b42cb812c89dfc232e3ceadea2fc5e44a7`. The worktree has 28 tracked
modifications and 12 new implementation/documentation files, plus the preserved
untracked `evals/`. All changes remain unstaged; the index matches its initial
bytes. No commit or push occurs.

Final SHA-256/size/mtime/mode comparison matches all 705 evaluations, 198
scientific outputs, 208 existing state files and operator configuration. The
checkpoint metadata remains exact under the stated prior-digest policy. All
117 initial scientific-owner/schema files remain byte-identical; only the three
documented orchestration diagnostic files change in that layer. Earlier
UA3.5.1/UA3.5.2 documentation and new acceptance tests remain byte-identical;
their existing implementation behavior passes the final regressions. Existing
Web registration/upload, report/download, cancellation and navigation contracts
remain covered by the full affected domain suite. File registration alone
still executes no scientific task.

Preservation snapshots, final comparisons, source freeze, initial worktree/index
and implementation-change inventory are under
`/tmp/agent-ua353-preservation-n9prp96c`. Scientific authority and provenance
schemas, algorithms, numerical defaults and Registry inventory remain unchanged.
There is no second Interpreter/Planner, default engine, pending store, workflow
engine or per-tool natural-language parser. The only scoped UI limitation is the
ambiguity failure banner described above; live-provider language and biological
qualification remain separate work.

UA3.5.4 should qualify the real Interpreter/Planner's language behavior on these
three interactions, especially multi-value clarification, partial replies and
scoped corrections, while retaining compiler/owner enforcement and evaluating
provider errors independently. No new default threshold policy is justified by
this evidence. UA3.5.4 is not implemented; UA3.5 is not closed and UA3.6 is not
started. No commit or push is performed.
