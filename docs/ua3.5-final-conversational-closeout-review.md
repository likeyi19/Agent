# UA3.5 — Final conversational integration and closeout review

The initial explicit-species gap is closed through the existing Interpreter and
shared typed binder. The requested human utterance now resolves the existing
human resource tuple and reaches normal admission without a species companion
or redundant clarification. Final live qualification is **6 PASS, 0 FAIL,
0 BLOCKED**, with every scientific tool deliberately withheld. Final focused
checks pass **198 tests** and the complete Application/Web regression passes
**2,107 tests, nine skipped**.

The unchanged UA3.5.5 journey remains the actual combined browser/provider/science
completion witness. Together, these bounded records support **scoped UA3.5
closeout and one consolidated commit/push after review**. No commit, push,
foundation rerun or UA3.6 work occurs here. The uncaptured first Selection and
historical BAM exceptions remain disclosed diagnostic limitations.

## Baseline and preservation

On 2026-10-10, entry branch is `main`. HEAD, local `main` and local
`origin/main` equal `8bcad5b42cb812c89dfc232e3ceadea2fc5e44a7`; the index is
clean. Entry contains **30 modified tracked files, 28 new non-evaluation files**
and existing untracked evaluations. The entry snapshot is
`/tmp/agent-ua35-final-preservation-lmx5x7um`, with private copies of prior work,
the worktree patch, exact index and file inventories. Earlier UA3.5.1–UA3.5.5
changes are preserved rather than reapplied from HEAD.

This task modifies five already-modified production files and adds this report,
[one test file](../tests/application/test_initial_scientific_declarations.py),
[one benchmark](../benchmarks/interactive/run_ua35_closeout.py) and a new
[evaluation directory](../evals/ua3.5_final_conversational_closeout_2026-10-10/).
No previous report, test or evaluation runner changes.

Final state contains **30 modified tracked files and 31 new non-evaluation
files**, plus evaluations. Of **592 original repository files**, only the five
declared production files change; the other **587** remain exact in SHA-256,
size, mtime and mode. All **1,061 prior evaluation files**, 198 inventoried
scientific outputs, 208 real-state files and the operator configuration remain
exact. The retained UA3.5.5 scientific workspace's 23 files, preparatory state's
seven files and human input workspace's two files also remain exact.

All **117 initial scientific-owner/schema files** are unchanged. The index is
byte-identical; branch and all three refs retain the entry values. The original
EpiZoo repository is clean at `029cd631d0f9806a646c4a3b42ce10b958f2b67f`.
The [final preservation audit](../evals/ua3.5_final_conversational_closeout_2026-10-10/diagnostics/final-preservation-audit.json)
passes with no undeclared original-file changes. New scientific input/state
workspaces remain private and outside Git; no dataset or checkpoint is added.

The operator's activated `human-recorded-qualified` and original
`mouse-recorded-qualified` entries remain unchanged. Configuration SHA-256 is
`50ddd7846dd276f11465f98040b285218fb459acbe480bb8aa90350f2283e880`, mode
`0664`. Its Groq default, profiles, InputSets, credentials and upload settings
remain exact. No permanent F1 profile, companion or resource is added. The large
checkpoint is statted against the entry inventory; its established identity is
reused without rehashing or loading it. No reset, clean, stage, fetch, commit or
push occurs.

## Precise defect and minimal repair

The [UA3.5.5](ua3.5.5-human-epizoo-user-workflow.md) browser journey supplied
`species=human` through an explicitly selected companion. Its initial
`ExecutePlan` had no way to express that declaration solely from the utterance.
The shared `arguments` wire existed, but its Registry projection and admission
accepted numerical declarations only. EpiZoo's required `(str,)` species already
had the exact `human`/`mouse` choices and a singleton request-source port. Merely
opening that gate would still leave the initial upload without the resource
tuple: resource selection previously happened before interpretation or during
the existing pending-species answer.

The repair extends those existing seams:

| Production owner | Change relative to this task's entry |
| --- | --- |
| `orchestration/registry.py` | Add default-false `conversational_choice` planning metadata, enabled only for `epizoo_embed_cells.species`. Eligibility requires a reviewed direct scientific declaration; scientific arguments, callbacks, result contracts, defaults and ports stay exact. |
| `application/turn_decisions.py` | Add `ScalarChoiceArgument`, a subclass of the existing scoped declaration with one canonical `value`. Extend the same closed `arguments` schema/parser and prompt; project missing-species clarification only for a captured pending species request. Explain initial missing-declaration handoff to the existing compiler. Preserve historical numeric wires and the eight-identity limit. |
| `application/dialogue_execution.py` | Bind the canonical choice through the existing typed port and binder, select its resource through `select_epizoo_resource`, and recompose the original registered H5AD. Persist a bounded catalog receipt and replay exact binding/selection when validating stored records. |
| `application/turns.py` | Pass the already admitted resource catalog into ordinary initial execution admission. Preserve fresh ambiguous/unsupported/conflicting species refusals without inventing pending work; retain compiler-only missing-species creation. |
| `application/dialogue_evidence.py` | Reconstruct numeric/choice declarations with the same parser when checking accepted explicit-user provenance. |

For the exact requested sentence:

> Analyze these human scATAC-seq cells with EpiZoo, construct their neighbor graph,
> perform Leiden clustering, and compute UMAP.

the declaration is represented as:

```json
{"tool":"epizoo_embed_cells","argument":"species","literal":"human","start":14,"end":19,"value":"human"}
```

The LLM supplies the dataset association and canonical value. Exact text/span,
nonblank evidence, generic token boundaries, closed choices and Registry types
are checked deterministically. For example, scripted `literal="mice"` with
`value="mouse"` demonstrates typed normalization admission; Agent contains no
species alias extraction rule. A span proves textual grounding, not that a word
describes the dataset. The prompt assigns that distinction to the Interpreter:
model-support statements alone supply no dataset species.

The public projection now offers **14 existing numerical parameters plus one
reviewed species choice**. Other string, list, path, resource, source-history and
design contracts remain outside this extension. The `scientific_parameter`
flag alone does not grant conversational eligibility. Species answers still use
`answer_prerequisite.species`; their accompanying parameters and
`answer_parameters` retain the original numeric surface and scientific meaning.

An initial resource receipt captures the existing bounded operator catalog and
its digest. Revalidation reconstructs its typed rows, checks the digest, replays
the existing selector and compares exact admitted inputs and registered
attribution. It does not simply allow additional checkpoint/pin keys. Existing
explicit choices take priority, and selection may not replace any received
species, checkpoint, pins, device or unrelated inputs. The receipt is historical
application policy, not scientific authority or a new pending store.

## Authoritative responsibilities

The existing Interpreter interprets task intent, explicit dataset declarations,
normalization and association. The existing Planner proposes a supported graph
and chooses actual outputs. Agent resolves one registered request port, admits
exact declarations, preserves source identity, selects configured resources and
checks the actual plan's consumers. A declaration for EpiZoo may accompany a
downstream UMAP target, but it must have exactly one actual EpiZoo consumer;
duplicate consumers and guessed fanout fail closed.

The compiler, ordinary preflight, registered-input guard, Session/RunStore and
scientific owners retain their existing authority. A human declaration cannot
approve a mouse axis, arbitrary peaks, altered resources or incompatible
preprocessing. Required scientific input validation is never replaced by language
or resource availability. No schema-2 authority adapter, transport allowlist,
compiler, Session schema, scientific algorithm or recovery policy changes.

## Deterministic acceptance and regression

The initial new suite passes **44 tests in 3.97 seconds**, exit 0. Its
[command](../evals/ua3.5_final_conversational_closeout_2026-10-10/diagnostics/new-tests-command.json)
and [log](../evals/ua3.5_final_conversational_closeout_2026-10-10/diagnostics/new-tests.log)
cover exact human/mouse declarations, case/alias normalization supplied by the
model, Unicode spans, blank/unsupported/ungrounded values, closed schemas,
numeric-wire compatibility and the unchanged pending-answer surface. Ordinary
Application witnesses cover source-only registration, matching/conflicting
companions, explicit-resource priority and species conflicts, missing/ambiguous
defaults, source mutation, durable omission/resumption, mixed historical/plural
Leiden declarations, retry/reopen and valid-checksum record corruption. All
scientific tools are deliberately withheld; tiny structural H5ADs do not qualify
human inference compatibility.

Repeated EpiZoo consumers in the new witness are rejected by the existing
compiler's `AMBIGUOUS_OPTIONAL_INPUT_SCOPE` before the downstream shared scope
check. This establishes fail-closed behavior, not a newly reached
`AMBIGUOUS_PARAMETER_SCOPE` witness.

The complete focused recheck passes **740 tests, three existing warnings in
171.99 seconds**, exit 0. It includes prior species/Leiden/Selection binding,
continuation, safety, evidence, resource consumption/cache, prompt/compiler,
output-selection and HTTP error-policy coverage, with the new declaration suite.
Its [exact command](../evals/ua3.5_final_conversational_closeout_2026-10-10/diagnostics/focused-recheck-command.json)
and [log](../evals/ua3.5_final_conversational_closeout_2026-10-10/diagnostics/focused-recheck.log)
are preserved.

The complete Application/Web regression passes **2,095 tests, nine skipped,
three existing warnings in 558.27 seconds**, exit 0, without exclusions:

```bash
UA354C_ADMISSION_EXCEPTION_LOG=/tmp/agent-ua35-final-preservation-lmx5x7um/application-web-admission-exceptions.jsonl PYTHONPATH=src:.:/tmp/agent-ua35-final-preservation-lmx5x7um/observer PYTHONDONTWRITEBYTECODE=1 /home/likeyi/anaconda3/envs/agent/bin/python -B -m pytest tests/application tests/web -q -p no:cacheprovider --basetemp /tmp/agent-ua35-final-preservation-lmx5x7um/application-web-pytest-tmp --tb=short -p ua354c_admission_observer
```

[Command](../evals/ua3.5_final_conversational_closeout_2026-10-10/diagnostics/application-web-command.json),
[log](../evals/ua3.5_final_conversational_closeout_2026-10-10/diagnostics/application-web.log)
and [validation](../evals/ua3.5_final_conversational_closeout_2026-10-10/diagnostics/application-web-validation.json)
record all 468 frozen files unchanged and zero observer exception rows. The
Selection failure does not recur; neither does the older BAM failure. Their
original causes remain unresolved; no production/test fix or exclusions are
used to obtain these passes.

All opt-in `RUN_*` gates are disabled. These deterministic runs dispatch zero
live providers and no real foundation inference. All **468 frozen
source/test/static/resource files** retain bytes, size, mtime and mode during
both focused attempts and the isolated Selection check. The passing recheck
records zero observed exceptions; it establishes nonrecurrence, not the first
failure's cause.

The first focused run records **1 failed, 739 passed, three existing warnings in
172.08 seconds**, exit 1. The failure is
`test_partial_invalid_threshold_answer_refresh_and_fresh_server_resume_original_task`.
Its final answer has the correct original QC identity, depth 1 and TSS `0.5`,
accepted plan, linked turn and passed durable preflight. The selection attempt
starts and publication artifacts exist, but its Run and step remain `RUNNING`,
without a checkpointed result, verification or Revision; the facade stores
`INTERACTIVE_APPLICATION_FAILED`. The original exception was not observed.
This differs from UA4c's earlier pre-detailed-planning BAM failure. Neither
causal attribution to this declaration repair nor a pre-existing/flaky diagnosis
is justified.

The unchanged whole Selection Web file then passes **3 tests in 7.53 seconds**,
exit 0. The evaluation-local exception observer delegates `AnalysisSessions.respond`
once, captures exceptions for the Selection and previously implicated BAM
workspaces and rethrows unchanged. It alters no result, decision or recovery;
the isolated run records zero exception rows. Failed records, original and
subsequent commands/logs, observer source and validation script remain in
[diagnostics](../evals/ua3.5_final_conversational_closeout_2026-10-10/diagnostics/).
No source or test change is made to obtain a passing recheck of that failure.

The [independent static review](../evals/ua3.5_final_conversational_closeout_2026-10-10/diagnostics/initial-declaration-independent-review.json)
passes ten checks, including identical ToolSpec ASTs except the explicit
conversation opt-in and unchanged scientific owners. Development checks also
identified and corrected frozen-receipt/plain-registration replay and blank
evidence issues before the recorded acceptance. No full repository or heavy
scientific regression is claimed.

The [final independent code review](../evals/ua3.5_final_conversational_closeout_2026-10-10/diagnostics/final-code-review.json)
passes seven checks on the final contract. Registry executable contracts,
numerical admission, actual-consumer scope and all 117 scientific-owner/schema
files remain exact. The fresh semantic-refusal admission change is identified
explicitly; unchanged Session schemas do not imply every admission branch is
unchanged.

## Bounded genuine-LLM qualification

The first matrix is retained intact. Its raw runner disposition is **2 PASS,
3 FAIL, 1 BLOCKED**, with eleven actual completions. Initial human admission and
matching typed-human admission pass. The omitted-species and model-support-only
utterances return live `clarify/missing_species`; no pending request exists, so
ordinary admission reports `invalid_prerequisite` and the dependent answer is
blocked. The conflict is safely rejected, but the evaluator incorrectly requires
no Session turns at all: normal clarification owns a receipt with null request/Run
IDs and empty selections. The independent corrected evidence disposition is
**3 PASS, 2 FAIL, 1 BLOCKED**, without editing any raw record.

The [initial live review](../evals/ua3.5_final_conversational_closeout_2026-10-10/diagnostics/initial-live-review.json)
identifies a concrete Interpreter contract gap: the fresh schema offers a
pending-only missing-species branch and the instructions do not describe the
compiler-owned initial handoff. The final contract explicitly sends otherwise
supported selected-input commands with omitted/unresolved declarations through
`execute_plan` with those arguments absent. The schema offers `missing_species`
only for an actual captured species prerequisite, matching existing eligibility.
Parser compatibility and compiler diagnosis remain unchanged. Fresh semantic
ambiguity/unsupported/conflict refusals retain their actual reason without
creating pending state or science; pending answers retain their original meaning.

Final focused coverage is **56 new cases**, included with the two existing
decision/binding suites in **198 passing tests in 4.20 seconds**, exit 0.
[Command](../evals/ua3.5_final_conversational_closeout_2026-10-10/diagnostics/fresh-species-clarification-tests-command.json)
and [log](../evals/ua3.5_final_conversational_closeout_2026-10-10/diagnostics/fresh-species-clarification-tests.log)
verify the generic handoff instructions, context-dependent reason offering,
unchanged numeric pending schemas, fresh refusals without state fabrication and
rejection of unoffered prerequisite answers. No providers or scientific owners
run in these checks.

After that final production contract adjustment, the complete Application/Web
suite passes **2,107 tests, nine skipped, three existing warnings in 555.99
seconds**, exit 0, without exclusions:

```bash
UA354C_ADMISSION_EXCEPTION_LOG=/tmp/agent-ua35-final-preservation-lmx5x7um/application-web-final-admission-exceptions.jsonl PYTHONPATH=src:.:/tmp/agent-ua35-final-preservation-lmx5x7um/observer PYTHONDONTWRITEBYTECODE=1 /home/likeyi/anaconda3/envs/agent/bin/python -B -m pytest tests/application tests/web -q -p no:cacheprovider --basetemp /tmp/agent-ua35-final-preservation-lmx5x7um/application-web-final-pytest-tmp --tb=short -p ua354c_admission_observer
```

The [final command](../evals/ua3.5_final_conversational_closeout_2026-10-10/diagnostics/application-web-final-command.json),
[log](../evals/ua3.5_final_conversational_closeout_2026-10-10/diagnostics/application-web-final.log)
and [validation](../evals/ua3.5_final_conversational_closeout_2026-10-10/diagnostics/application-web-final-validation.json)
record all **468 frozen files unchanged**, disabled `RUN_*` gates, zero live
providers and zero observed exceptions. The first Selection and historical BAM
failures do not recur; their original exceptions remain unexplained.

The first executable runner is archived by its exact digest. The corrected
benchmark accepts ordinary nonexecution clarification receipts and counts the
initial eleven completions and **$0.828222** conservative reservation against the
same aggregate **48-completion/$5** limits. A post-contract qualification uses
fresh requests/workspace and the original single exact catalog evidence. It is
performed after a production contract repair; there is no automatic semantic
repair, provider retry, decision conversion or scripted continuation of the
failed requests. The original failures remain failures.

The final [live record](../evals/ua3.5_final_conversational_closeout_2026-10-10/live-recheck/results.json)
is **6 PASS, 0 FAIL, 0 BLOCKED**, with **19 interface calls and 19 actual provider
completions**: six Interpreter, five capability selection, five detailed Planner
and three output selection. Every semantic response is live. The accepted
development-primary profile remains `f1-openrouter-gpt-5-6-sol`, provider
`openrouter`, exact model `openai/gpt-5.6-sol`, strict Chat Completions adapter
and 60-second timeout. All final and initial responses identify that exact model,
one upstream attempt and an unmodified pipeline. No substitute profile is used.

Final usage is **49,087 prompt / 5,316 completion tokens**, with provider-reported
cost **$0.1569041**. Including the preserved initial eleven completions, this
task totals **30 interface calls and 30 provider completions**, **78,924 prompt /
8,238 completion tokens** and provider-reported cost **$0.2429516**. The aggregate
pre-dispatch reservation is **$2.209676**, already including the initial
**$0.828222**, under the unchanged **$5 ceiling**. One original exact-model catalog
lookup is reused; no second lookup occurs. No balance claim is made.

Bounds remain **48 aggregate completions, six per fresh turn, 4,096 output tokens,
150,000 conservative input byte/token units and one dispatch per turn/schema**.
SDK retries, provider failures, guard blocks, semantic repairs and profile
failovers are zero; routing fallback is disabled. Both paid matrices and their
commands, original runner bytes and discovery provenance remain intact. The
[independent final record review](../evals/ua3.5_final_conversational_closeout_2026-10-10/diagnostics/final-post-live-review.json)
checks these observations without replaying language or science.

The new [runner](../benchmarks/interactive/run_ua35_closeout.py) reuses frozen
UA4a transport/recording/spending guards and UA4b's explicit zero-recovery policy.
Its ordinary application receives the existing qualified human/mouse catalog and
an evaluation-local exact F1 profile. Every plan is withheld after actual ordinary
executor preflight, before **every registered scientific tool**, including any
unexpected model-selected operation. No scripted semantic response or repaired
output supplies acceptance.

Initial explicit-human, omitted species, the corresponding live pending answer,
model-support-only wording, matching human companion and conflicting mouse
companion are tested separately. The pending follow-up is dispatched only if the
preceding live turn creates the matching durable prerequisite. Raw prompts,
schemas, responses, Run/Session records and executor observations are retained
independently of the checks.

| Case / actual utterance and input | Live semantic decision | Ordinary Agent outcome | Result |
| --- | --- | --- | --- |
| A1: exact requested human sentence; selected H5AD only, no species companion | `execute_plan/compute_cell_umap`; scoped `species=human`, exact literal/span `[14,19)` | Original source retained; unique human default and exact pins selected; four-step plan, 18 exact uniquely named selections and ordinary turn linkage; executor preflight passes, all tools withheld | **PASS**, initial declaration/admission |
| A2: same four-operation request with species omitted | `execute_plan/compute_cell_umap`; no species argument | Compiler `MISSING_REQUIRED_SOURCE/species`; durable pending origin `a2`, original input/catalog and generation 0; executor not entered | **PASS**, necessary clarification |
| A3: “These are human cells.” in the live-created A2 pending task | `answer_prerequisite/@species/human` | Exact original `a2` objective/source resumed with human resources; four-step plan, 15 exact selections and normal linkage; executor preflight passes, all tools withheld | **PASS**, continuation/admission |
| A4: “EpiZoo supports human and mouse. Analyze these scATAC-seq cells with EpiZoo.” | `execute_plan/epizoo_embed_cells`; no dataset-species argument | Model-support mention supplies no dataset declaration; compiler creates durable missing-species pending origin `a4`; executor not entered | **PASS**, dataset association |
| B1: exact human sentence with matching typed human companion | Same explicit human declaration, accepted consistently | Typed declaration/source retained; four-step plan and nine exact selections; executor preflight passes, all tools withheld | **PASS**, companion compatibility |
| B2: exact human sentence with conflicting typed mouse companion | Same explicit human declaration, without overriding typed mouse | `conflicting_scientific_parameter`; only nonexecution clarification receipt with null request/Run IDs and empty selections; no Planner, executable turn, Run, Revision or executor | **PASS**, fail-closed conflict |

Admitted positives deliberately retain facade/turn failure
`EVALUATION_EXECUTION_WITHHELD`, zero scientific results/Revisions and generation
0. Normal verification also reports missing step results. Persisted durable
`Run.preflight_verification` remains null because the evaluation executor returns
without the normal execution checkpoint; its separate preflight observation and
Planner trace prove only their respective checks. No scientific success or input
feature compatibility is inferred from this run.

The final command, exit 0 in **143.55 seconds**, is:

```bash
PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 /home/likeyi/anaconda3/bin/conda run --no-capture-output -n agent python -B -m benchmarks.interactive.run_ua35_closeout --live --output evals/ua3.5_final_conversational_closeout_2026-10-10/live-recheck --workspace /tmp/agent-ua35-final-live-recheck-20261010 --source-h5ad /tmp/agent-ua355-human-input-kk2y0q98/human_pbmc_m12_canonical_32.h5ad --operator-config /home/likeyi/program/agent-web-config/operator-web.json --prior-budget-json evals/ua3.5_final_conversational_closeout_2026-10-10/live/results.json --discovery-json evals/ua3.5_final_conversational_closeout_2026-10-10/diagnostics/exact-discovery-reused.json
```

The [command record](../evals/ua3.5_final_conversational_closeout_2026-10-10/diagnostics/live-recheck-command.json)
and [log](../evals/ua3.5_final_conversational_closeout_2026-10-10/diagnostics/live-recheck.log)
retain the exact environment and frozen runner digest. Reproduction requires
fresh output/workspace paths. Discovery reuse and prior-budget input refer to
this evaluation's exact original evidence; a separate evaluation should perform
fresh discovery and set its own bounded budget.

Credentials are read from the existing Conda environment, never printed or copied.
SDK headers and client representations are not saved. The
[confidential-value screen](../evals/ua3.5_final_conversational_closeout_2026-10-10/diagnostics/confidential-value-screen.json)
checks new text artifacts and changed files against configured private values,
including encoded variants, and finds no matches. Private scientific datasets,
model bytes and operator configuration are not copied into the evaluation.

## Unchanged EpiZoo and accepted complete task

This work does not repeat the accepted foundation inference. The genuine
[UA3.5.5 journey](ua3.5.5-human-epizoo-user-workflow.md) remains the combined
native-browser, real-provider and real-science witness: original full-axis
32-cell human canonical counts, **32 × 512** finite embeddings, genuine neighbors,
four fixed-Leiden clusters, **32 × 2** finite UMAP, passed durable preflight and
step/Run verification, one active Revision and Firefox evidence/report/PNG access.
It had one foundation load and embedding invocation; those historical counts are
not new work performed here.

Human and mouse consume the same original jointly pretrained checkpoint
`6b2d13fdbd54a9b0d56efa5afa81bc4832b813f4eac8662e4d93cc08c9d9b39a`,
with their exact original species auxiliaries. No separate human model,
qualification rerun, adaptation or retraining is required. The invocation path
remains Registry `epizoo_embed_cells` → `analysis.epizoo_embedding` →
`epizoo_cache`/`models.epizoo` → original EpiZoo processing, datasets,
model/configuration and embedding extraction in `/home/likeyi/program/EpiZoo`.
It directly reuses the original implementation. Agent owns sparse/input/resource
contracts, faithful invocation, strict loading/cache lifecycle, artifacts and
provenance. Existing downstream Scanpy composition overlaps the original
convenience wrapper as documented in UA3.5.5; its accepted fixed-Leiden owners
remain unchanged. No model, tokenization, normalization or inference logic is
copied or redesigned.

The new live witness qualifies initial utterance binding/admission; the retained
UA3.5.5 witness qualifies actual science/browser completion with a selected typed
companion. They are distinguished explicitly rather than presented as a newly
observed combined utterance-only browser/science run.

## Accepted capabilities and remaining scope

The [saved-evidence review](../evals/ua3.5_final_conversational_closeout_2026-10-10/diagnostics/closeout-evidence-review.json)
passes fourteen checks and maps the requested user capabilities:

| User capability | Accepted bounded evidence |
| --- | --- |
| Upload and registration | UA3.5.5 actual Firefox upload; exact registered full source bytes; upload itself executes no science. |
| Natural-language planning | UA3.5.5 live four-step graph and exact output selection; this task separately checks initial species language/admission. |
| Explicit parameters | UA3.5.3 shared binding and safety; UA4b live Leiden explicit/paraphrase plus complete Selection partial-answer/correction chain. |
| Owner defaults | Live Leiden omission uses resolution 1.0; UA3.5.5 downstream optional values are omitted and original owners supply their settings. |
| Qualified resources | Existing human/mouse catalog resolution and exact original resources; UA3.5.5 proves actual human consumption. |
| Necessary clarification | Compiler-owned missing species and Selection thresholds, durable exact pending scope and no premature science. |
| Answer binding/continuation | Live species continuation/admission and UA4b's live Selection correction chain; exact original task/source retained. |
| Actual scientific execution | UA3.5.5 four ordinary owners, real unchanged EpiZoo inference and downstream science. |
| Verified Run/Revision | UA3.5.5 successful Run, all owner/Run verification, one activated Revision and generation 1. |
| Browser results/evidence | UA3.5.5 all 13 exact selections, safe report projection and PNG, refresh and fresh-server restoration in one Python process. |

These records do not qualify all 23 tools conversationally, arbitrary H5AD/peak
inference, broad language or biology. The 172 public parameter inventory and
requiredness remain unchanged. General required string/list/design declarations,
evaluator/multi-H5AD upload composition and source-specific raw companions remain
explicit development boundaries. Generic processed-output cross-turn authority
and raw embedding/compact-H5AD downloads remain unsupported; displayed summaries
are not scientific authority. Empirical Selection values 1000/4/20 remain
unqualified candidates, and statistical cell calling, purity and broad biological
interpretation are unassessed. No new default or resource provisioning occurs.

## Closeout recommendation and next expansion

**UA3.5 is ready for a single consolidated commit and push after scoped review.**
The shared conversational mechanisms, original human resource integration,
accepted complete user task and the final initial-declaration/admission repair
have the bounded evidence described above. All earlier unstaged work is retained;
this task neither stages nor commits any part of it and performs no push.

Closeout should retain the explicit capability, authority, resource, policy and
language/scientific scope limits. The original uncaptured Selection and BAM
exceptions remain diagnostic risks despite passing isolated and complete
rechecks; no unsupported causal explanation or speculative repair is included.
This recommendation is readiness for review, not a claim of universal reliability
or an automatic milestone closure. No further foundation inference is needed
merely to approve this unchanged owner's initial declaration integration.

The next highest-value bounded expansion is the shared exact-string declaration
contract for `label_key`, `reference_label_key` and `ground_truth_label_key`, as
ranked in [UA4d](ua3.5.4d-practical-capability-readiness.md). Reuse exact scoped
evidence, owner-validated column roles, atomic correction and compiler-owned
pending state; begin with complete trusted InputSets. Evaluator/multi-H5AD upload
composition must be scoped separately. This recommendation starts no new work,
authority expansion, raw-data provisioning, policy calibration or UA3.6.
