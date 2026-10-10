# UA3.5.2 — Explicit Leiden resolution and default reuse

This stage binds one natural-language scientific argument,
`cluster_cells.resolution`, through the existing Interpreter, Planner and typed
execution path. Omitted optional arguments retain their existing owner defaults.
The accepted clustering result supplies effective parameters; follow-up answers
read accepted evidence without running science.

## Baseline and implementation

Work continues on `main` at
`8bcad5b42cb812c89dfc232e3ceadea2fc5e44a7`, with HEAD, local `main` and local
`origin/main` equal. The existing UA3.5.1 18 tracked modifications and two new
files are captured before editing; its 453-file source/test/static/resource
freeze matches. The index is clean. Nothing is reset, cleaned, fetched, pulled,
staged, committed or pushed.

The version-1 Interpreter's `execute_plan` and `answer_prerequisite` variants
gain one optional typed declaration: exact tool `cluster_cells`, argument
`resolution`, numeric literal, and its current-utterance Unicode character span.
Historical decisions omitting the field remain accepted. The LLM identifies
intent and operation association. Agent verifies the exact literal/span,
numeric token boundaries and finite conversion, then delegates scalar validation
to the existing Registry argument contract. No language parser or arbitrary
argument dictionary is introduced.

The existing clustering owner's finite, non-boolean, strictly positive scalar
check is extracted without changing its algorithm or default. Registry admission
reuses that check. Application resolves the one request-input name from the
existing clustering semantic port; the normal compiler binds it. Before science,
the actual plan must contain exactly one clustering consumer with the admitted
resolution, and no other tool may consume that argument. A missing, repeated or
mismatched consumer fails closed; Agent adds no steps or workflow recipe.

Received submissions remain immutable. Their admitted projection adds only the
validated resolution and recomposes the existing registered H5AD binding while
preserving source identity, companions and resource policy. Direct Session
calls retain an original-input fingerprint for retries. Session decoding checks
the persisted declaration against its utterance and admitted inputs, including
boolean/value tampering and removed declaration metadata.

UA3.5.2 production changes are `application/turn_decisions.py`,
`dialogue_execution.py`, `turns.py`, `responses.py`, `dialogue_evidence.py`,
`scientific_dialogue.py`, `orchestration/registry.py`, and
`tools/analysis/embedding_analysis.py` under `src/agent/`. Tests change
`tests/application/test_turn_decisions.py` and `test_dialogue_provider_contract.py`,
`tests/unit/orchestration/test_registry.py`, and
`tests/tools/analysis/test_embedding_analysis.py`; new focused files are
`tests/application/test_clustering_parameter_evidence.py` and
`tests/web/test_leiden_resolution.py`. README and this record document the
behavior. Earlier UA3.5.1 changes remain together for review.

## Precedence and durable conversation

1. A valid explicit text value is used. An existing typed value must agree;
   conflicts clarify without choosing either source. Invalid declarations never
   fall back to a default.
2. An omitted resolution stays absent from execution inputs, preserving the
   clustering owner's `1.0` default. Neighbors retain `15`, Euclidean metric and
   seed `0`; UMAP retains `min_dist=0.5`, `spread=1.0` and seed `0`. Clustering
   retains seed `0`. These are existing execution defaults, not universally
   optimal biological settings. Application/Web contain no duplicate default
   table.
3. Required facts with no qualified applicable default still require declaration.
   Species is not inferred from the file or model. Operational/model-resource
   failures remain operational failures.

```text
User:  Run EpiZoo, perform Leiden clustering at resolution 0.7, and generate UMAP.
Agent: I need to know the species of the selected dataset. Is it human or mouse?
User:  Mouse.
Agent: [The ordinary accepted scientific result.]
User:  What resolution did you use for clustering?
Agent: [Accepted resolution 0.7 and explicit execution-argument origin.]
```

With species already supplied, the same request executes directly. Without a
resolution, ordinary execution uses the owner default without a resolution
question. A pending species reply can also declare a resolution, using the same
typed contract, for clustering already requested by the captured objective.
The Interpreter owns that association; actual-plan acceptance still requires
exactly one clustering consumer. Agent appends no steps, and an earlier
conflicting explicit value cannot be overridden.

The original objective, exact registered H5AD and effective inputs remain in the
existing UA3.5.1 turn/RunStore records. No incomplete plan is persisted as an
executable pending task. Refresh/restart retain the pending resolution; species
answers select the existing qualified mouse resource and resume ordinary
planning. Conflicting replies preserve the pending origin for a corrected answer.
Retries return accepted outcomes without replay. Stale generations, source
mutation, changed dataset/objective or revision context cannot redirect the old
parameter. A new omitted-value request does not inherit an earlier `0.7`.
Explanatory questions and unsupported/ambiguous decisions execute no science.

## Effective values and provenance

The existing owner result records effective `resolution` and `random_seed`,
including omitted defaults; the raw accepted evidence/report already records
these values. Those formats and historical bytes are unchanged.

A bounded Application evidence projection exposes `clustering_parameters` and
`clustering_resolution_origin`. Values come from the exact pinned accepted
clustering summary. Origin is `explicit_execution_argument` when the accepted
planned/resolved arguments include resolution, otherwise `existing_owner_default`.
The origin's source pointer and digest identify the accepted Run result, rather
than pretending that this derived fact was present in raw evidence.

For a final UMAP, this projection follows only its exact same-plan
`StepOutputRef` to a verified clustering step, with explicit dependency and
resolved-input checks. It never searches by path, tool order or an older run.
Missing attribution is unavailable. Historical effective values are read from
their accepted result, never reconstructed from today's `1.0` default; a test
preserves a recorded historical omitted value of `1.3`.

The new Session declaration additionally establishes the exact user text/span.
Generic evidence deliberately calls the source an explicit execution argument:
arbitrary historical typed arguments do not establish how a user supplied them.
Follow-up answers can state the accepted value and execution/default origin
through the existing reviewed claim mechanism. No generated explanation becomes
authority and no provider or science runs during historical reads.

Project policy for future work is valid explicit declarations first, established
scientifically applicable owner/configured defaults next, and clarification only
for genuine missing prerequisites. Empirical/resource defaults require existing
qualification and applicability. Record effective values and established origins;
never guess biological facts or recycle an example as a universal default. This
stage implements natural-language binding only for Leiden resolution, not all
23 tools or every provenance field.

## Acceptance and preservation

Validation uses the existing `agent` environment, scripted Interpreter/Planner
decisions and scripted EpiZoo inference. Tiny CPU scientific owners, compiler,
verification, Session, evidence/report and HTTP paths execute normally. No live
provider, expensive GPU inference, dependency installation or resource download
is used. Live-provider language quality remains separate qualification.

Final focused command:

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 /home/likeyi/anaconda3/envs/agent/bin/python -m pytest tests/application/test_turn_decisions.py tests/application/test_dialogue_provider_contract.py tests/application/test_clustering_parameter_evidence.py tests/web/test_leiden_resolution.py tests/unit/orchestration/test_registry.py tests/unit/orchestration/test_semantic_compiler.py tests/tools/analysis/test_embedding_analysis.py -q -p no:cacheprovider --basetemp=/tmp/agent-ua352-focused-final2 --tb=short
```

It passes **309 tests, three existing warnings in 34.83 seconds**, exit 0.

Affected-domain command, with no full-repository regression:

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 /home/likeyi/anaconda3/envs/agent/bin/python -m pytest tests/application tests/web -q -p no:cacheprovider --basetemp=/tmp/agent-ua352-regression-final2 --tb=short
```

It passes **1,886 tests, nine skipped, three existing warnings in 464.38
seconds**, exit 0, with no exclusions and all `RUN_*` gates disabled. The 455
frozen source/test/static/resource files remain unchanged through both final
runs, and `git diff --check` passes. No unresolved defect remains within this
software scope. The Web parameter file contributes 32 cases covering actual
consumption, omission, typed conflicts, invalid/ambiguous/unsupported intent,
pending restart/retry, stale/source/context changes, combined species replies,
accepted answers and persisted-record corruption. The evidence file contributes
17 cases, including exact upstream attribution and historical-value preservation.

Native Firefox passes **11 checks**, including normal file
picker/composer parameter capture, missing species, refresh/fresh server,
mouse continuation with actual `0.7`, unrelated defaults, accepted evidence and
report/download, zero-science follow-up, a new default `1.0` request, historical
bytes and no replay/JavaScript errors. Its isolated check record and screenshot
are under `/tmp/agent-ua352-browser/run-faoebz95`; two tiny CPU chains and two
scripted inference calls run. Browser/server and temporary home fixtures are
removed after acceptance.

Registry remains 23. Scientific algorithms/numerical defaults, EpiZoo,
Planner/compiler/preflight, RunStore and scientific-authority schemas are
unchanged. Only the shared scalar validator extraction and its Registry use
change scientific source files. Session/Revision compatibility and existing
upload/report/download behavior are covered by the affected regression. No new
store, default engine, Planner, UI editor or exporter is added.

Preservation checks match all **705 evaluations, 198 scientific outputs, 208
existing application/Web state files and operator configuration**, including
SHA-256 and metadata. The original 5.23 GB checkpoint retains exact
size/mtime/mode; its prior verified digest is reused without rereading it. Of
151 frozen scientific sources, 149 remain byte-exact; the two intentional
changes are the Registry validator and extracted owner helper. Expanding the
helper call restores an AST-identical clustering function. The original UA3.5.1
acceptance record and its species-handoff test remain byte-identical. Read-only
preservation records and the original worktree snapshot are under
`/tmp/agent-ua352-preservation-m0c_uuo6`; final test logs are
`/tmp/agent-ua352-focused-final2.log` and
`/tmp/agent-ua352-regression-final2.log`.

Final status is 25 tracked modifications and five new files, plus preserved
untracked `evals/`. The clean index and its original digest remain unchanged;
branch and all local refs remain at the baseline. UA3.5.1 and UA3.5.2 remain
unstaged together for review, with no commit or push.

The smallest next UA3.5.3 is one equally scoped explicit
`build_cell_neighbors.n_neighbors` declaration, with existing owner validation,
omitted-default behavior, pending-request retention and accepted-value answers.
It is not implemented here.
