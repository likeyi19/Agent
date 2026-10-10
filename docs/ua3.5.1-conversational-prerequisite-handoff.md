# UA3.5.1 — Conversational prerequisite handoff and continuation

This focused stage lets a researcher answer a missing human/mouse species
declaration in chat and continue the original registered-H5AD EpiZoo request.
The ordinary Planner still chooses embedding and any requested downstream
neighbors, Leiden and UMAP operations. No workflow recipe or parameter parser
is added.

## Baseline and preservation

Implementation starts on `main` at
`8bcad5b42cb812c89dfc232e3ceadea2fc5e44a7`. HEAD, local `main` and local
`origin/main` match; the tracked worktree and index are initially clean, with
only the existing `evals/` untracked. No fetch, pull, reset, clean, commit or push
is performed.

Before editing, SHA-256 and metadata inventories match prior acceptance for all
705 evaluations, 198 scientific outputs, 208 application/Web state files and
operator configuration. The original 5.23 GB pretrained checkpoint matches its
previously verified size/mtime/mode; its recorded digest is retained without
rereading the large binary. Final preservation verification matches every
inventoried evaluation, output, state/configuration file and all 151 frozen
scientific tool/orchestration/schema files. The checkpoint metadata remains
exact. These checks write only temporary acceptance records outside the repo.

## Existing contracts and the small extension

- `Interaction.utterance`, `submission`, `base_revision_id` and `base_generation`
  already retain the original objective, exact registered source and validated
  companions. They remain immutable.
- An optional immutable `Interaction.prerequisite` stores only the originating
  turn, awaited `species` field and digest of the exact configured resource
  policy. Unset fields remain omitted so historical Session records retain their
  original encoding. There is no additional state store or queue.
- A follow-up captures `snapshot.prerequisite` with the exact originating and
  immediately preceding clarification turns. Session decoding checks these
  references and the captured Revision/generation.
- The additive version-1 Interpreter variant
  `answer_prerequisite(pending, species)` offers an opaque `@species` handle and
  only `human`/`mouse` choices when that exact clarification is available.
  The LLM interprets the user's language and normalizes its meaning. Agent
  checks the handle, awaited field, allowed value and existing Registry
  `ArgumentSpec`. No provider dictionary becomes execution inputs.
- An admitted continuation records its origin, species and completed registered
  binding in the existing admitted turn. Session decoding checks its original
  tool, source identity, retained companions and captured base. The received
  follow-up submission is preserved separately as the immutable received turn.

## Grounded question and ordinary continuation

The first question is created only when the ordinary semantic compiler reports
`MISSING_REQUIRED_SOURCE` or `MISSING_REQUIRED_BINDING` for
`epizoo_embed_cells/species`, the selected source uses `h5ad-science.v1`, and no
species declaration exists. The failed planning attempt remains in RunStore;
no scientific tool executes and no Revision is created. Presentation becomes a
normal chronological clarification rather than an execution-error bubble.
The existing Planner recovery policy is unchanged.

On an admitted answer, Agent rechecks that exact original compiler finding in
RunStore, objective, registered source bytes, generation and predecessor. It
validates the species using the existing Registry declaration, preserves every
previous companion, and calls the existing qualified resource-selection policy.
That policy is now owned by `application.local_resources` and reused by Web,
with the former Web imports/helper retained. A unique configured default must
apply to the declared species. Explicitly captured model pins remain exact;
compatible current typed selections can also be used. A conflict or changed
catalog fails closed. Choosing a known qualified model before declaring species
captures its exact pins while leaving species absent; the compiler still
establishes the missing declaration. Mouse qualification supplies no human default.

The original objective and completed typed inputs enter the normal Planner,
semantic compiler, preflight, registered owners, verification, evidence/report,
RunStore and Session/Revision admission. Plan and result consumption use the
completed exact registered binding, including explicit recovery. Scientific
compatibility remains with EpiZoo; species is a user assertion, not proof of
feature-vocabulary compatibility. Nothing infers species, projects peaks,
changes assembly, trains a model or repairs an incompatible dataset.

## User interaction, retry and recovery

```text
User:  Run EpiZoo, cluster the cells, and generate UMAP.
Agent: I need to know the species of the selected dataset. Is it human or mouse?
User:  Mouse.
Agent: The dataset is declared as mouse. [Actual accepted execution status.]
```

The original file need not be uploaded or selected again. The normal composer,
result details, evidence, report/figure delivery and existing eligible downloads
are reused without static-layout changes. A processed embedding/graph result
does not acquire a new scientific-file exporter in this stage.

Pending questions survive history retrieval, refresh and a fresh application or
server. Reads construct no provider and execute no science. Ambiguous species
answers retain the same origin. Unsupported species and conflicts execute no
science. A different selected source, new objective, cancellation, unrelated
turn, explicit scientific predecessor or Revision/generation change cannot
redirect the pending task. Only the immediately pending clarification chain is
offered; Agent never searches backward for an old task.

Existing turn IDs, duplicate checks and processing leases make identical
Interactive and direct Session retries return stored outcomes. Completed
clarifications are not offered again. Admission rechecks generation, active
Revision and latest interaction after the final provider call and before linking
the Run. Source/resource mutations fail through existing typed admission errors.
Missing qualified resources and scientific/runtime failures remain genuine
failures; they are not converted into generic conversational repair questions.

## Validation

Acceptance uses the existing `agent` Python environment, scripted structured
provider decisions and tiny registered H5AD fixtures. EpiZoo inference is
scripted through the established owner fixture; production scientific owners,
verification, reports, Session and Web boundaries run normally. No live provider
or expensive GPU inference is required. The earlier real UA3.2 mouse scientific
acceptance remains the biology qualification.

Focused command:

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 /home/likeyi/anaconda3/envs/agent/bin/python -m pytest tests/web/test_species_handoff.py tests/application/test_turn_decisions.py tests/application/test_h5ad_composition.py tests/web/test_h5ad_composition_api.py tests/web/test_config.py -q -p no:cacheprovider --basetemp=/tmp/agent-ua351-focused-final --tb=short
```

Affected-domain command (no full-repository regression):

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 /home/likeyi/anaconda3/envs/agent/bin/python -m pytest tests/application tests/web -q -p no:cacheprovider --basetemp=/tmp/agent-ua351-regression-final --tb=short
```

The final focused command passes **188 tests in 18.30 seconds**, with three
existing warnings, exit 0. The final affected-domain command passes **1,783
tests, nine skipped, three warnings in 439.23 seconds**, exit 0, with no
exclusions and all `RUN_*` gates disabled. All 453 frozen source/test/static and
resource files remain unchanged during the final validation. `git diff --check`
passes. No unresolved software blocker remains within this scope.

The new handoff file independently passes 27 cases in 16.79 seconds, with three
existing scientific-stack warnings. It covers embedding/full-chain admission,
companions and explicit pins, report download, complete-input/inspection behavior,
ambiguity, unsupported/conflicting species, changed task/source/catalog, source
mutation after provider calls, corrupt persisted continuation identities,
restart, direct and Web retries, completed replay rejection and stale navigation.

Native Firefox passes nine checks through the real file picker and ordinary
composer: prompt, pending refresh, fresh application/server, mouse continuation,
exact input/resources, accepted Revision/evidence/report, report attachment,
accepted refresh without replay, and no application JavaScript errors. Its
isolated records are under `/tmp/agent-ua351-browser/run-i5qyxdgm`; no real
Session, source, operator configuration or evaluation is used for browser writes.

## Changed files and review status

Production changes: `application/turn_decisions.py`, `turns.py`,
`dialogue_execution.py`, `responses.py`, `session_state.py`, `session_store.py`,
`sessions.py`, `interactive.py`, `local_resources.py`, and `web/app.py`,
`web/config.py` under `src/agent/`. The resource declaration/selector move retains
Web compatibility; HTTP and Web presentation do not interpret species text.

Tests: `tests/application/test_turn_decisions.py`, `test_h5ad_composition.py`,
`test_typed_turn_admission.py`; `tests/web/test_species_handoff.py`,
`test_h5ad_analysis.py`, `test_uploaded_multiturn.py`,
`test_h5ad_composition_api.py`. Documentation: this record
and the focused README interaction update.

Registry remains 23. Scientific tools/algorithms, EpiZoo, compiler/preflight,
RunStore and scientific-authority schemas are unchanged. Existing Revision
identity and admission contracts remain; Session adds only optional interaction
metadata. UA3.1–UA3.4 upload and result-delivery regressions are included above.
Changes remain unstaged and ready for review; no commit or push is performed.
Final status is 18 tracked modifications, two new files, the preserved untracked
`evals/`, and a clean unchanged index. Branch and local references remain at the
baseline commit.

Natural-language numeric values, QC thresholds, statistical designs, assemblies,
and other declarations still require their existing typed inputs. The smallest
next UA3.5 task is one explicit clustering-resolution assignment from the current
utterance, normalized by the Interpreter and bound to the existing Registry
type/range and command evidence. It is not implemented here.
