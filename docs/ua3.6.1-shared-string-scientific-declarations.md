# UA3.6.1 — Shared exact string scientific declarations

UA3.6.1 extends the existing conversational scalar binder to three reviewed label
columns. It adds no language parser, workflow engine, scientific algorithm,
resource admission, source registration or scientific authority contract.

## Baseline and preservation

The inspected baseline is `main` at
`63a74cfb2b5a3842c2e567e87798f21a26520340` (`Complete UA3.5 conversational
scientific workflow`). HEAD, local main, local origin/main and remote main agree.
The initial index/worktree is clean apart from the existing untracked `evals/`.
Remote inspection uses only `git ls-remote`; no fetch, pull, reset, clean,
stage, commit or push occurs.

The preservation inventory is
`/tmp/agent-ua361-preservation-yhka3b3g`: 1,249 original evaluation files,
198 outputs, 208 real application/state files, 23 UA3.5.5 scientific-state files,
seven preparatory-state files, two preserved human-input files and the external
operator configuration. The 5,231,645,507-byte checkpoint is recorded by
size/mtime/mode only; it is neither read, hashed nor loaded for this change.
All new scientific tests use isolated temporary workspaces. New live-language
records occupy their own directory and do not replace prior evaluations.
Final comparison and repository status are recorded below.

## Reused mechanism and boundaries

`ScopedArgument` remains the same canonical tool/argument, exact literal and
zero-based Unicode character start/end representation. The eight-declaration,
80-character literal, 4,096-character span and closed wire bounds remain intact.
Historical Leiden and separately normalized species-choice wires remain valid.

Only two production files change:

- `application/turn_decisions.py`: derive exact-string eligibility from existing
  Registry metadata; distinguish exact columns from numbers and normalized
  choices; validate literal/span and quotation/token boundaries without changing
  the column value; project the same existing decision schema and Interpreter
  instruction. Pending schema offers only the captured tool's declarations,
  including eligible corrections, instead of duplicating unrelated tools.
- `application/dialogue_execution.py`: resolve strings through exactly one
  singleton semantic consumer port and direct request source, with no upstream
  alternative; expose bounded type/provenance descriptions; validate pending
  requiredness through the same eligible-argument projection. This prevents
  transfer's already-supplied, nonconversational species/resource arguments from
  being mistaken for pending parameters.

No Registry metadata addition is needed. The existing conjunction identifies
exactly three registered arguments: exact `(str,)`, no choices, scientific
parameter, request-input-only, nonartifact and reference/ground-truth provenance,
plus the singleton port/source requirement. `scientific_parameter=True` alone is
insufficient. Argument identity remains canonical; the existing compiled-plan
check requires exactly one actual scientific consumer with the accepted value.

| Declaration | Exact request input | Reviewed role |
| --- | --- | --- |
| `evaluate_cell_clustering.label_key` | `label_key` | Evaluation-only ground truth |
| `transfer_cell_labels.reference_label_key` | `reference_label_key` | Biological labels in the reference dataset |
| `evaluate_cell_annotation.ground_truth_label_key` | `ground_truth_label_key` | Independent query ground truth |

The Interpreter owns language, operation association and explicit declarations.
Agent owns typed binding, captured identity, compiler/preflight, durable
continuation and exact consumer checks. The existing scientific owners validate
column existence, source roles, ordered cells, labels and compatibility. Web
still transports and presents ordinary application operations.

## Initial requests and pending answers

With complete trusted typed inputs, examples are:

- `Evaluate the clustering using the ground-truth column "cell_type".`
- `Transfer cell labels from the reference dataset using the column "major_type".`
- `Evaluate the transferred annotations using the ground-truth column "manual_annotation".`

The first example binds `literal="cell_type", start=55, end=64` only if those
positions match the actual utterance; quotation delimiters are excluded. Unicode,
case, embedded whitespace and punctuation remain exact. A numeric-looking column
such as `123` remains a string. Blank, fabricated, truncated, quote-inclusive,
normalized-choice and unsupported declarations fail closed. Scientific owners
retain their existing rejection of invalid keys, including surrounding whitespace.

An omitted required key enters the existing Planner/compiler. Only its exact
missing-argument diagnostic can create `@parameters` pending state when all
reported missing arguments are eligible. Missing artifacts/resources or mixed
noneligible requirements remain failures. A response such as
`Use the ground-truth column "Cell 类型 Δ".` binds that exact string and resumes
only the captured original objective and inputs.

Each of these tools has one required eligible string. Supplying it completes that
particular prerequisite immediately. Partial/correction tests therefore use
transfer's existing optional numerical declarations while its reference label
remains missing; they do not invent a second required string. Valid corrections
remain scoped to the pending tool. A mixed valid string/invalid numeric response
accepts neither change. A fresh unrelated task receives no old pending values.
Selected M17.2 guidance candidates retain their original missing-input failure
behavior and do not acquire a pending string workflow.

Received submissions stay immutable and separate from admitted execution inputs.
Session records retain exact literals, source utterance/spans, original/binding
turn links, Revision/generation and resource-context identity. Existing replay
validation checks these without providers or science. Accepted
`explicit_user_parameters` evidence proves origin through that exact linked
Session/Run/step; matching values in unrelated history provide no attribution.

## Deterministic and scientific acceptance

The focused tests extend existing binding/safety/turn/continuation and Web suites;
`test_string_scientific_execution.py` reuses existing compact clustering and
ordered ground-truth fixtures. These are synthetic software fixtures, not real
biological label validation or uploaded multi-H5AD composition.

Changed test files are `tests/application/test_scientific_binding_contract.py`,
`test_scientific_binding_safety.py`, `test_turn_decisions.py` (shared eligibility,
literal/source/schema and consumer checks); `test_initial_scientific_declarations.py`
(intentional 15→18/14→17 projection expansion, preserving the original 14 numerical
pairs and species choice); `test_scientific_parameter_continuation.py` and new
`test_string_scientific_execution.py` (durability, atomicity and real owner checks);
and `tests/web/test_scientific_binding.py`, `test_selection_clarification.py`
(ordinary HTTP interaction/recovery and upload-only invariance).

Semantic admission covers all three keys, excluded string categories, Unicode and
quote spans, exact inputs, compiler handoff, partial/atomic corrections, durable
recovery, stale source/Revision/generation/resource context, repeated consumers,
role-safe ports, unrelated tasks, selected guidance and upload-only behavior.

The normal Application and HTTP path executes the real registered clustering
evaluator, publishes its report and accepted Revision, and exposes exact user
provenance. The measured initial Application witness runs production once, with
15 verifier calls and 16 source reinspection calls during completion. Its existing
legacy revalidation can repeat; this observation is recorded rather than changed
or described as one verification. Accepted
retry and evidence reads perform zero production, verification, source
reinspection or provider calls. Missing columns, swapped sources and mismatched
ordered cells fail through existing owner validation without scientific Revision
publication. Transfer/annotation placeholder inputs qualify compiler/semantic
admission only; unqualified sources correctly fail owner execution.

Final focused acceptance passes **828 tests, seven existing dependency/fixture
warnings in 174.13s**, exit 0:

```bash
PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 /home/likeyi/anaconda3/envs/agent/bin/python -B -m pytest \
  tests/application/test_scientific_binding_contract.py \
  tests/application/test_scientific_binding_safety.py \
  tests/application/test_scientific_parameter_continuation.py \
  tests/application/test_string_scientific_execution.py \
  tests/application/test_turn_decisions.py \
  tests/application/test_dialogue_provider_contract.py \
  tests/application/test_initial_scientific_declarations.py \
  tests/application/test_clustering_parameter_evidence.py \
  tests/application/test_selection_parameter_evidence.py \
  tests/application/test_h5ad_composition.py \
  tests/web/test_scientific_binding.py \
  tests/web/test_selection_clarification.py \
  tests/web/test_species_handoff.py \
  tests/web/test_leiden_resolution.py \
  tests/unit/orchestration/test_registry.py \
  tests/unit/orchestration/test_semantic_compiler.py \
  tests/unit/orchestration/test_planning_diagnostics.py \
  tests/unit/orchestration/test_llm_planner_v4.py \
  tests/unit/orchestration/test_llm_planner_v4_recovery.py \
  tests/tools/analysis/test_clustering_evaluation.py \
  tests/tools/analysis/test_label_transfer.py \
  tests/tools/analysis/test_annotation_evaluation.py \
  tests/external_fragments/test_integration.py::test_v4_complete_catalog_size_and_explicit_null_index \
  -q -p no:cacheprovider --basetemp /tmp/agent-ua361-focused-recheck-final-pytest-tmp --tb=long
```

The final lexical/binding selection passes **291 tests in 21.50s**, exit 0:

```bash
PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 /home/likeyi/anaconda3/envs/agent/bin/python -B -m pytest tests/application/test_scientific_binding_contract.py tests/application/test_scientific_binding_safety.py tests/application/test_turn_decisions.py -q -p no:cacheprovider --basetemp=/tmp/agent-ua361-lexical-final2 --tb=long
```

Provider/benchmark compatibility passes **246 tests in 54.55s**, exit 0:

```bash
PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 /home/likeyi/anaconda3/envs/agent/bin/python -B -m pytest tests/benchmarks/test_interactive_frontier_transport.py tests/benchmarks/test_interactive_candidates.py tests/benchmarks/test_interactive_breadth.py tests/benchmarks/test_interactive_stability.py -q -p no:cacheprovider --basetemp=/tmp/agent-ua361-benchmark-contracts --tb=short
```

The affected Application/Web and orchestration regression passes **3,175 tests,
nine skipped, three existing dependency warnings in 733.53s**, exit 0:

```bash
PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 /home/likeyi/anaconda3/envs/agent/bin/python -B -m pytest tests/application tests/web tests/unit/orchestration -q -p no:cacheprovider --basetemp /tmp/agent-ua361-domain-final-pytest-tmp --tb=long
```

Both final focused/domain runs preserve all **498 frozen source/test/benchmark
files**. No exclusions or scientific-owner changes are used to obtain these
results.

All `RUN_*` gates are disabled. Final focused and domain invocations use isolated
Matplotlib/Numba caches and freeze source/test/benchmark bytes; results and
resolved commands are retained in
[development validation](../evals/ua3.6.1_string_semantic_2026-10-12/development-validation/).
The 7,000+ full lightweight suite is not run. `git diff --check` passes.

## Bounded live language and schema budget

The previously qualified benchmark candidate OpenRouter `openai/gpt-5.6-sol`,
with existing Conda credentials, is tested through the existing guarded
evaluation transport, discovery and spending guard. The Web operator's Groq
default and configured profiles remain unchanged.
The isolated runner further limits dispatch to seven Interpreter calls, one per
case and a $1 reservation. There are no retries, fallback, Planner calls, science,
checkpoint access or configuration/credential changes.

Exact command:

```bash
PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 /home/likeyi/anaconda3/bin/conda run --no-capture-output -n agent python -B /tmp/agent-ua361-semantic.py --live --output evals/ua3.6.1_string_semantic_2026-10-12
```

All **seven cases pass**: three initial declarations, one pending-answer fixture,
ambiguous alternatives, omitted/unknown ground-truth declaration and a clustering-output column
mentioned without independent ground truth. Positive outputs pass deterministic
exact binding; negative cases invent no label. The pending public context is an
explicit test fixture, not a live-created Session prerequisite. Language
qualification is separate from the scripted durable/scientific acceptance above.

Provider usage is 31,523 prompt plus 1,225 completion tokens, reported cost
**$0.0910470**, conservative reservation **$0.696550**. Raw prompts/schemas,
responses, usage, exact model discovery, independent replay and preservation
reviews are retained in
[the new evaluation directory](../evals/ua3.6.1_string_semantic_2026-10-12/).
All saved calls replay against final prompts/schemas and binding with no more
provider calls.

| Representative UTF-8 bytes | Prompt | Schema | Total |
| --- | ---: | ---: | ---: |
| Frozen full v4 Planner catalog, baseline and final | 21,013 | 12,985 | 33,998 |
| Initial Interpreter, baseline (15 declarations) | 14,375 | 9,735 | 24,110 |
| Initial Interpreter, final (18 declarations) | 15,439 | 10,927 | 26,366 |
| Matched pending fixture, baseline size comparison | 15,518 | 15,483 | 31,001 |
| Pending Interpreter, uncompact extension | 16,598 | 17,867 | 34,465 |
| Pending Interpreter, final | 16,598 | 11,615 | 28,213 |

The original **34,000-byte v4 full Prompt + Schema constraint is unchanged**.
The baseline pending row is a matched size-only fixture; the baseline could not
create or resume a required-string prerequisite.
Pending tool scoping removes 6,252 duplicate bytes without removing allowed
corrections or admission constraints. Exact-string instructions appear only when
such declarations are offered, preserving the original frozen inspection
prompt/schema fingerprints. No frozen expectation is changed to conceal drift.
The live model accepts the final schema. No policy budget increase is necessary.
Offline sizes reproduce with the same runner without `--live`; the exact
measurement witness is `schema-sizes.json` in the new evaluation directory.

## Findings, limits and next work

Development exposed stale scalar-count assertions after intentional expansion,
a same-tool schema-test expectation that omitted existing numerical corrections,
and lexical truncation of quoted/punctuated/combining-character names. Focused
regressions and shared lexical checks address these findings. An unconditional
new instruction initially changed a frozen inspection fingerprint; making the
instruction conditional on offered exact strings preserves the existing freeze.
No historical evaluation response, fixture replay or scientific artifact is
repaired. Historical uncaptured Selection/BAM exceptions remain diagnostic risks;
no speculative owner repair is made.

The initial development invocation records **277 passed, three failed in 22.19s**
in `/tmp/agent-ua361-initial-tests.log`: all three are the old projection-count
assertions, replaced with explicit checks for preserved numeric/choice contracts
plus the three new strings. Another development test expected only the missing
reference key and omitted two valid existing numerical correction fields; fixing
that new test expectation preserves the actual same-tool contract. Quotation
reproductions include interior `Type` from `"Cell Type α"` and malformed unmatched
delimiters; final focused regressions reject them. The normal owner negatives
retain their actual `INVALID_ARGUMENT` errors rather than replacing them with
new conversation-layer scientific validation.

Arbitrary paths, resource IDs, digests, devices, namespaces, barcode history,
assemblies, conditions, designs, lists/dictionaries and unreviewed free-form
operands remain excluded. General multi-H5AD upload composition, new
reference/query authority, processed-output cross-turn reuse, extra downloads,
raw-resource provisioning, calling policies and model adaptation remain outside
UA3.6.1. Labels are never inferred from AnnData metadata.

The next recommended task is **UA3.6.2**: trusted processed-output reuse followed
by verified artifact delivery, targeting “Change resolution to 0.5 and update
UMAP” while retaining the original Revision and reusing qualified
embeddings/neighbors. **UA3.6.3** covers role-safe multiple-H5AD composition;
**UA3.6.4** a qualified uploaded raw workflow; **UA3.6.5** combined live/native
Firefox/scientific/verified-result acceptance. None is implemented here.


## Completion and final repository state

UA3.6.1 implementation and bounded software/semantic acceptance are complete and
ready for review. HEAD, main and local origin/main remain
`63a74cfb2b5a3842c2e567e87798f21a26520340`; the index bytes remain exact and
nothing is staged, committed or pushed. The final worktree contains two modified
production files, seven modified existing test files, one new test file, this new
record and the preserved untracked evaluation directory with a separate new
qualification directory. Registry inventory remains **23 tools / 172 public
arguments**.

Final SHA-256/size/mtime/mode comparisons preserve every initially recorded
output, Session/RunStore/input/configuration/evaluation file. All **117 scientific
owner/schema files** remain exact; the four accepted EpiZoo auxiliary resource
pins also match their prior accepted record. Checkpoint metadata remains exact
under the no-read/no-rehash policy. The external EpiZoo checkout remains clean at
`029cd631d0f9806a646c4a3b42ce10b958f2b67f`.

The exact final preservation comparison is
`/tmp/agent-ua361-preservation-yhka3b3g/final-comparison.json`, copied alongside
stable test commands/results in development validation. All seven captured live
responses also replay against final code and identical prompt/schema bytes in
`final-binder-replay.json`, without another provider call. No demonstrated
implementation blocker remains within this scope. Groq string-language quality,
real biological labels, universal multi-H5AD composition and combined native
browser/live scientific acceptance are not qualified by this task.
