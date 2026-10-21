# UA3.6.3c — Unified scientific declaration and clarification continuation

Implemented on 2026-10-12 against UA3.6.3b commit
`46316567a22baebe6932500fa18def773d4ae08c`. This extends the existing Interaction
prerequisite contract with durable scientific-role answers. It does not introduce
another conversation store, prerequisite solver, scientific workflow or language
parser. Final affected-domain acceptance is recorded below.

## A. Repository baseline

`HEAD`, local `main`, local `origin/main` and remote `refs/heads/main` all equal
the actual commit above. Remote alignment was checked with `git ls-remote`;
there was no fetch or pull. Tracked files and the index were clean at admission,
with only the existing `?? evals/`. The UA3.6.3b acceptance document was present.
Nothing is staged, committed, pushed, reset or cleaned.

Preservation root: `/tmp/agent-ua363c-preservation-h9ve1f54`.
Admission captures all 626 tracked files and 72,781 protected historical records:
1,274 evaluations, 198 outputs, 208 real-state files, operator configuration,
qualified production resources, EpiZoo auxiliary resources and previous scientific,
Run/Session, authority and browser acceptance artifacts. Ordinary files retain
SHA-256, size, modification time and mode comparisons. Previously metadata-only
large checkpoint/reference records retain that policy. The 5,231,645,507-byte
checkpoint is neither loaded nor rehashed. All new fixtures, Runs, Sessions,
reports, caches, browser profiles and provider capture live in isolated workspaces.
Final preservation and frozen implementation comparisons are recorded in L.

## B. Previously accepted machinery

The existing [prerequisite handoff](ua3.5.4b-prerequisite-handoff.md),
[scientific input binding](ua3.5.3-unified-scientific-input-binding.md),
[exact-string declarations](ua3.6.1-shared-string-scientific-declarations.md),
[state-to-planning integration](ua3.6.2b-unified-conversational-state-planning.md),
[ordered scientific roles](ua3.6.3a-unified-multi-input-role-binding.md) and
[qualified resource connection](ua3.6.3b-unified-qualified-resource-connection.md)
remain authoritative. Interaction snapshots capture the original request and
Session scope. The immutable Registry offers reviewed consumers and argument
contracts. LocalResourceAdmission owns exact registration and selected resource
identity. Planner, semantic compiler, preflight, scientific owners, RunStore and
generation-checked Revisions continue to perform execution and acceptance.

Existing species and scalar/string parameter prerequisites keep their closed
formats. A compatible explicit resource choice on a species-answer turn remains
permitted by that existing contract. Scientific authority and delivery formats
are unchanged.

## C. Actual confirmed gap

Previously `Clarify` retained a reason but no identified execution target or
missing-role requirements. An ordinary group execution required complete roles;
an ambiguous or partial association therefore clarified without a durable role
answer bridge. Parameter prerequisites already retained the exact task and could
accept partial grounded answers. The missing extension is a typed role request
and answer linked through that same pending Interaction machinery.

An operation that cannot be identified safely still uses ordinary clarification
without executable pending state. No operation is inferred from a reference/query
word, file name, selection position, step order or historical first match.

## D. Implementation

| File | Responsibility |
| --- | --- |
| `application/turn_decisions.py` | Immutable `RoleRequirement`, `RequestRoles`, `RoleAnswer`; closed `request_roles` / `answer_roles` wires and conditional provider-neutral schema/instructions. |
| `application/local_resources.py` | Pure partial-role validation using the existing reviewed Registry consumers, distinct source aliases and exact offered handles. Complete execution binding stays strict. |
| `application/session_state.py` | Add the closed pending `field=roles` distinction alongside unchanged legacy species/parameter records. |
| `application/dialogue_execution.py` | Capture, replay and continue exact pending bindings; preserve effective registered input through chained parameter answers and follow explicit continuation links to the original objective/declarations. |
| `application/turns.py` | Route typed role requests/answers through the existing Session lifecycle; offer original sources without reselection; apply the existing generic candidate veto to initial role requests. |
| `application/responses.py` | Present the remaining scientific roles in ordinary clarification text using generic readable formatting. Machine choices retain exact Registry identities. |

Paths above are relative to `src/agent/`. New tests are
`tests/application/test_pending_role_contract.py`,
`tests/application/test_role_declaration_continuation.py` and
`tests/web/test_role_clarification_scientific_continuation.py`; an existing
response-contract test checks the clarification text. Web transport and static
frontend are unchanged. Registry inventory remains 23 tools / 172 public arguments.
Scientific owners, EpiZoo, numerical methods, authority and download contracts
remain byte-identical to the baseline.

## E. Shared state transition

`request_roles` names an identified registered target and exact required consumers,
existing partial `InputRole` associations and original grounded declarations. It
captures a question; it creates no Plan, Run or scientific Revision. Requirements
must be offered reviewed consumers with distinct aliases and one association per
selected member. A complete association cannot masquerade as missing roles.

Pending state pins origin turn, latest accepted binding turn, tool, required roles
and resource-policy digest, within the original Revision/generation snapshot.
The original submission retains ordered registration IDs/digests and the exact
InputSet ID/definition digest. Partial associations also appear in admitted group
metadata, so a lone persisted-role mutation rejects during Session decoding.
No partial dataset aliases become executable inputs.

`answer_roles` uses the offered `@roles` handle, exact consumer/source pairs and
optional existing grounded arguments. Agent merges only compatible associations;
previously admitted roles cannot be reassigned. Partial answers preserve the task
and ask for the remaining consumers. A complete answer materializes the existing
group binding and calls the ordinary Planner, compiler, preflight and executor.
The original utterance remains the scientific request, even through a subsequent
parameter clarification. Files need neither reupload nor reselection.

Selected files offered again must equal the captured ordered group and resource
choice. Changed selection, replacement, registration or choice cannot redirect the
pending task. Latest-turn and generation checks reject competing or stale replies.
A new independent task uses ordinary admission without inherited roles or values.

## F. Scientific qualification

The Interpreter interprets intent, explicit source associations and declarations.
Agent validates offered identities and state transitions. Scientific owners still
validate graphs, exact ordered cells, metadata columns, provenance, representation
and resource compatibility. Conversational claims of qualification, corrected
barcodes, compatible checkpoints or sufficient replication issue no proof.

The current ordered-H5AD selection supports the five reviewed consumers from
UA3.6.3a. This is not arbitrary multi-input fanout or new biological qualification.
Missing source-specific raw InputSets, library/reference/QC contracts, model
resources and annotation/adaptation specifications remain operator/owner work.
No filename, role answer or receipt substitutes for those contracts.

## G. Real scientific witness

The primary fixture uses 32 ordered Unicode cell IDs, a 32 × 512 float32 synthetic
representation and a previously prepared compact neighbor graph as an explicit
current scientific input. Its second uploaded H5AD holds the exact label column
`Cell 类型 Δ`. The original objective explicitly asks to cluster one selected
dataset and evaluate the resulting fixed clustering against the other. Its typed
question requests `cluster_cells.analysis` and
`evaluate_cell_clustering.ground_truth_dataset`; no workflow is invented from an
otherwise ambiguous operation.

The ordinary reply is: “The first dataset is the fixed analysis input, and the
second is ground truth.” The scripted Interpreter proposes the exact captured
positions. All downstream Planner/compiler admission, registered Leiden/evaluation
bodies, verification, persistence, evidence and report generation are real.
The accepted plan uses the first registered graph for clustering, its intra-plan
`StepOutputRef` for evaluation, and the second registration for ground truth.
Both source bytes/order, original objective, label, InputSet token and original
Interaction remain exact. The accepted output is not promoted into newly invented
historical clustering authority.

Observed primary counts: initial Interpreter 1, Planner/science 0; continuation
Interpreter/Planner/compiler 1 each, preflight 2, Leiden and evaluation production
1 each, ordinary clustering/evaluation verification 15 each, evaluation source
checks/metric calculations 16: one initial calculation plus 15 normal verification
calculations. Verification is counted as scientific work. Covered neighbors/EpiZoo
production during the turns is zero. NMI, ARI,
AMI and homogeneity are 1.0 for this deliberately separated synthetic fixture;
this supplies no biological performance claim.

The additional genuine chain leaves the label absent initially, completes roles,
receives the existing compiler-authored `label_key` prerequisite, then supplies the
exact Unicode string. It retains original objective and source guards without
reselection and completes the same real owners once. Duplicate completion adds no
provider or science work. The final direct witness is
`/tmp/agent-ua363c-app-web-final-uwfuru2q/pytest-temp/test_uploaded_role_answer_exec0/role-continuation-scientific-witness.json`.
It records exact source IDs, locators, arguments, identities and measured calls.

| Final representative identity | Actual value |
| --- | --- |
| First selected graph resource | `local-c7b78f00ec6111540ef792cc44bfb4322f387fcde7501ec6bfed0959389e65ea` |
| First registration SHA-256 | `ee920710c6dad75c6b10e82eaf58394b40e07b785ed08ef8b7c751720c7ab29a` |
| First H5AD SHA-256 | `e323249ac80635ea5d243ce9ca302a1f74d20415fa0b38ec694b4209100a5a47` |
| Second selected truth resource | `local-6bcd2f9c74240c2160319964e5c8edff2433107aa4f664def566a97e4131471c` |
| Second registration SHA-256 | `ef3b1774b7800722ad7304a8d8609c65e607428012e2f8713b51074110f22c4a` |
| Second H5AD SHA-256 | `340d6406c1d1bcc76d386baa3e2e344bf88782d0ddf7be6746394fc8008acbc4` |
| InputSet `captured` definition SHA-256 | `feae4932c54d76740c611b2d912577ed7d101dcbbb324ac54a3183d647f3e3c2` |
| Accepted Run | `turn-d2b975444e9efcf81064a4835f54b68c1af739a5ed8333ed27dcc42fff87c73f:run` |
| Evaluation accepted-step SHA-256 | `c159ab4bae485216b346ddf26aa80a46435facb5b3774d588145c76d026dd7c9` |
| Run-result SHA-256 | `4206835ddd8cfc761a12c4c1f106602fdc560df4887de3c4ebef6c36182469f7` |
| Accepted Revision, generation 1 | `1b7fbf4e12fe03c1b97e6ff284bbde437ca59720039561272149941d481a34a6` |

## H. Cross-family generality

Reference/query fixtures use the same `request_roles` / `answer_roles` and pending
schema, with `transfer_cell_labels.reference_dataset` and `.query_dataset`.
Recorded companions and resource-choice identities remain exact. Partial answers,
actual process restart and both no-reselection and exact-original-reselection
role-to-column chains pass through normal compiler/preflight. The executor
deliberately withholds scientific production: these are admission/generalization
witnesses, not label-transfer scientific acceptance. The final direct secondary
record is
`/tmp/agent-ua363c-app-web-final-uwfuru2q/pytest-temp/test_reference_query_continuat0/secondary-role-continuation-witness.json`.

No target-specific role-answer branch or keyword parser exists. Scientific meaning
stays with registered semantic ports and owners. Exact literal/span admission reuses
the existing scalar/string binder; no additional declaration type is introduced.

## I. Negative and recovery coverage

| Invalid condition | Observed boundary |
| --- | --- |
| Wrong/no pending, changed ordered selection or resource choice | `invalid_prerequisite`; no execution/Revision. |
| Unoffered source, duplicate source or unauthorized consumer | `MULTI_INPUT_ROLE_INVALID`. |
| Compiled source/embedding substitution | `UNAUTHORIZED_REQUEST_INPUT`; wrong port `UNKNOWN_TARGET_PORT`. |
| Repeated declaration consumer | `ambiguous_parameter`. |
| Changed registered bytes | `LOCAL_RESOURCE_INTEGRITY_INVALID`. |
| Invented exact string or wrong character span | `ungrounded_operand`; unsupported declaration `unsupported_intent`. |
| Changed generation | `INTERACTIVE_GENERATION_CONFLICT`. |
| Duplicate turn with changed meaning | `INTERACTIVE_TURN_CONFLICT`. |
| Initial role request classified by the existing candidate-reference veto | `unsupported_intent`; no pending executable task, Run or science. |
| Persisted initial/partial role mutation or missing chained source metadata | Session decoding rejects the invalid binding. |

For competing different-turn answers, the second sees no offered pending while
the first is interpreting; the first then loses its latest-turn anchor. Both
reject with zero plans/science/Revisions. No Session-wide lock or winner policy is
invented. Contradictory reassignment of an admitted role rejects. A complete swap
proposed before any role was admitted to a truly ambiguous request remains model
semantic quality; structural admission cannot read intent from prose again.

Existing owner negatives reject missing graph provenance and wrong ordered ground
truth. Owner entry that rejects is counted honestly; these cases create no accepted
Revision. Archived-source historical views, duplicate submission and fresh facade
reads add zero provider/scientific calls. Pending reference/query recovery is also
tested in an actual new Python process. Legacy species and parameter decoding and
source-specific qualification remain covered by the affected domain suite.

## J. Declaration readiness

This is a targeted assessment of the four scenarios and named experimental-design
examples, not a 23-tool audit. Existing mechanisms suffice for the demonstrated gap.

| Scenario/declaration | Current boundary / remaining prerequisite |
| --- | --- |
| Raw QC/selection numerical thresholds | Existing grounded scalar/pending continuation. Species/assembly/layout/library/barcode/reference/QC facts remain exact structured source-specific InputSets and qualified owner resources. No corrected-barcode or statistical-calling inference. |
| MAESTRO tissue/context and grouping | These live in pinned `scatac-annotation-inputs.v1` with source/group TSV/provenance and gene/signature resources. The public annotation tool exposes matrix/specification identity pairs, not a free-standing tissue argument. A reply cannot author or rewrite that owner specification. |
| EpiZoo species, downstream parameters and labels | Human/mouse reviewed categorical choice and existing species pending; numeric neighbors/Leiden/UMAP parameters and three reviewed reference/ground-truth label columns use current scalar/exact-string contracts. Checkpoint compatibility and embedding provenance remain owner/operator qualification. |
| Adaptation strategy | Registry accepts `de_novo` / `mapped_reference` through structured inputs. Its strategy metadata does not currently offer shared conversational choice admission. Qualified source/SEAM/reference/specification/profile and any mapping direction/runtime remain prerequisites. This stage does not broaden that contract. |
| Replicates, conditions and directed contrasts | Pseudobulk `replicate_key`, `group_key`, `condition_key`, `group_source`; DA `group_value`, `condition_key`, numerator/denominator and `design_type` are explicit structured Registry inputs, not reviewed conversational column/choice declarations. Ordered `covariate_keys` / `covariates` lists lack a shared conversational type. Owners validate actual design and replication. |

Exact assessment/owner anchors are retained in
`/tmp/agent-ua363c-targeted-readiness.md`. Relevant implementations are the existing
annotation Registry/contract, species-adaptation Registry and adaptation strategy
contract, plus pseudobulk/DA specifications in `orchestration/registry.py`.
Scenario integration must qualify specification authoring or narrowly reviewed
additional declarations where needed. No unrestricted JSON/list language, inferred
replicate design or per-tool prerequisite engine is added here.

## K. Prompt/schema and live validation

Final measurements use actual ordinary Session projections and no provider or
science for measurement. Payloads expose safe labels, handles and reviewed roles;
source paths/digests, resource-choice tokens, credentials and private operator
configuration remain absent.

| Interpreter request | Prompt + schema UTF-8 bytes |
| --- | ---: |
| Ordinary context-free | 27,033 |
| Two-source initial role question | 41,365 |
| Captured pending role answer | 44,891 |
| Eight sources, maximum 160-character Unicode labels | 47,495 |
| Representative parameter continuation | 28,886 |

The unchanged frozen scoped-v4 fixture is exactly 33,998 bytes
(21,013 prompt + 12,985 schema), below its 34,000-byte fixture limit. Dynamic
Interpreter payloads have a separate measured footprint; that fixture is not a
universal request limit. No limits are increased or scientific context truncated.
Measurement artifacts are under `/tmp/agent-ua363c-prompt-final-w04qktjl`.
`final-code-projection-record.json` pins final source hashes and explains that the
last candidate-veto fix runs after Interpreter payload generation; the measured
payloads remain unchanged.

Five fixed Interpreter-only live probes are authorized after deterministic
acceptance: explicit roles, missing roles, exact pending answer, contradictory
answer and absent model resources. The accepted OpenRouter development profile
and existing strict structured-output/spending controls are reused. Before calls,
the schedule is locked to one completion per case, five total, with a cumulative
conservative reservation ceiling of US$0.75, 4,096 output-token ceiling, exact
model/prices, no SDK retry, no fallback and no repeated attempts. Live pending
context is captured by scripted Sessions; live probes do not run a Planner or
scientific owner and cannot establish a complete live scientific workflow.
The exact accepted profile is `f1-openrouter-gpt-5-6-sol`, model
`openai/gpt-5.6-sol`. Discovery confirms its exact price and a 1,050,000-token
context / 128,000-token completion capacity. The worst conservative guarded input
bound is 47,752 tokens plus at most 4,096 output tokens, within the existing
150,000 input bound and discovered capacity. No supported hard limit is exceeded.

| Fixed live case | Model-authored outcome |
| --- | --- |
| A1 explicit roles | `Execute` preserves the label but adds a redundant direct evaluation-analysis association to the already selected clustering source. Semantic failure; offline exact binding rejects `MULTI_INPUT_ROLE_INVALID`. |
| A2 missing roles | `PROVIDER_CONNECTION_FAILED`, `UNEXPECTED_EOF_WHILE_READING`; no authored decision, unverified. |
| A3 pending answer | Exact `RoleAnswer`, `@roles`, captured source/consumer pairs, no added declarations; pass. |
| B1 contradictory reply | `Clarify / ambiguous_subject`, no inferred roles/declarations; pass. |
| B2 missing checkpoint/model resources | `Execute / epizoo_embed_cells` with no invented declarations or roles; semantic-restraint pass only. Deterministic resource eligibility remains a separate boundary. |

Five attempts, one per case, reserve US$0.634098 against US$0.75. The four reported
responses use 26,877 prompt and 736 completion tokens, observed cost US$0.0745465;
failed-request usage/cost is unreported. There is no retry, fallback, tuning or
subsequent paid call. Offline A1 binding runs zero providers/Planner/science and
creates no Revision. Three semantic passes, one semantic failure and one transport
failure do not qualify a complete live workflow. Records and exact command are
under `/tmp/agent-ua363c-live-semantic-gi07khgr`, including `observed-summary.json`,
`command.json`, `semantic.log`, `results/results.json`, five call captures and
`offline-a1-admission.json`.

## L. Test results and preservation

Development acceptance initially passes 37 new typed/replay cases, 30 application
role-continuation cases and six genuine Web cases. The latter passes in 20.50 s
with three existing dependency warnings. A pre-final compatibility selection
passes 161 tests in 126.73 s. These selections overlap the final domains and are
not unique totals. Development records are under
`/tmp/agent-ua363c-role-application-dev-xcq2wusz` and
`/tmp/agent-ua363c-role-web-dev-fxd6c4zk`; reconstructed observed summaries are
labelled explicitly, not represented as verbatim pytest logs.

Final affected-domain commands use conda `agent` Python, `-B`, `PYTHONPATH=src:.`,
`PYTHONDONTWRITEBYTECODE=1`, all `RUN_*` gates unset, private caches,
`-q -p no:cacheprovider` and isolated basetemp. Exact invocations, logs and results:

* `/tmp/agent-ua363c-app-web-final-uwfuru2q/{command.txt,pytest.log,result.json}`:
  all `tests/application` and `tests/web`.
* `/tmp/agent-ua363c-contracts-authority-final-ujb9k0gf/{command.txt,pytest.log,result.json}`:
  Registry/semantic metadata/compiler/prompt/v4 wire/prior-output planning/dispatcher,
  existing prior/processed/captured/raw conversational authority and continuation,
  application/intent/scientific-dialogue authority and response contracts. The
  command record names all 16 exact selections.
* `/tmp/agent-ua363c-role-guard-final-_94mi94x/{command.txt,pytest.log,result.json}`:
  all three new modules plus existing guidance-selection, scientific-guidance,
  turn-decision and dialogue-provider contracts, after the final candidate-veto
  repair. Exact seven selections are listed in the command record.

The completed Application/Web selection passes **2,534 tests, nine skipped,
three warnings in 854.21 s**, exit 0 (runner elapsed 857.42 s). The skipped cases
are the existing explicit Chromap gates. The warnings are existing
`pkg_resources`, `louvain` and TBB/Numba dependency notices. The
Registry/compiler/authority selection passes **443 tests, three existing warnings
in 2,407.59 s**, exit 0 (runner elapsed 2,410.09 s); it has no skipped cases.

Final review exposed the new initial role-request decision bypassing the existing
generic M17.2 candidate veto, which checked only `Execute`. A meaningful seam test
observes the owned positive predicate being skipped. The fix includes
`RequestRoles` in that existing initial veto; it adds no language rule and does not
reinterpret role-answer text. The regression's initial assertion expected no
SessionTurn, but ordinary terminal clarification already records a metadata-only
turn without a Run. Its assertion now checks that exact existing behavior. The
corrected contract module passes 38 cases in 3.66 s. The full Application/Web
selection above precedes this one-line repair; the focused final seven-module
selection and fresh native browser below validate the repaired branch without
rerunning the whole domain. Earlier guard-development logs retain their observed
assertion failures and are not substituted for the final passing record.
The final seven-module guard selection passes **258 tests, three existing warnings
in 34.50 s**, exit 0 (runner elapsed 35.88 s). Its counts overlap Application/Web;
the selections are not summed as unique tests. All three final command records
listed above exit 0.

Native Firefox passes 21 checks (12 initial + nine fresh-server) with both original
uploads retained, visible role clarification, ordinary composer reply, accepted
evidence/report/result, pending refresh, duplicate reply and exact fresh-server
restore. Initial pending state has one Interpreter call and zero Runs, SessionTurns
or science. Successful reply has the genuine owner/verification counts in G; total
scripted completions are five. Fresh-server providers and science are zero; exact
generation is one and Revision is
`e73052d9c2fa9f125db65ddb9e60dde96d373377c755c90ad571cc41ed4f04ed`.
The direct native record is
`/tmp/agent-ua363c-native-firefox-final-guard-2xe5udu8/browser-summary.json`, with command
arrays and reviewed screenshots. The new Firefox profile is
`/home/likeyi/agent-ua363c-firefox-i8i_jl62/profile`. All browser/server processes
are stopped; application JavaScript errors are zero. Internal Firefox notices are
recorded separately from application errors (two per phase). Seven relevant source
files remain unchanged throughout this final browser acceptance, as recorded by
`final-source-start.json` / `final-source-comparison.json` in its workspace.

No full 7,000-plus-test regression, foundation
inference, adaptation training or real biological preprocessing is performed.

The repeated post-test preservation comparison finds **zero mismatches across all
72,781 protected records**. Evaluation inventory remains exactly 1,274 files and
the original output inventory exactly 198 files. Existing inputs, outputs,
Run/Session/Revision envelopes, scientific authority, operator configuration and
qualified resources retain their recorded bytes/metadata. Git index bytes and
`HEAD` / `main` / `origin/main` remain unchanged; a final `git ls-remote` again
confirms remote `main` at the baseline commit. Of 626 baseline tracked files,
619 remain byte-identical; the seven intentional edits are the six production
files and existing response-contract test in D. Three tests and this document are
new. Existing evaluations and acceptance documents are unchanged.

Preservation files are `protected-before.json`, `tracked-before.json`,
`refs-before.json`, `index-before`, `preservation-final.json` and
`inventory-final.json` in the admission directory in A. The initial 524-file
source/test/benchmark freeze matches through the broad Application/Web run. The
final candidate repair intentionally changes only `turns.py` and its new contract
test; `final-guard-repair.json` records that boundary. The final 524-file freeze
`implementation-final-freeze.json` matches **all files** after the passing final
commands; `post-final-tests-freeze.json` records zero mismatches and all three exit
codes. Python-only inventory is 506 files. `git diff --check` passes. External
EpiZoo remains clean at `029cd631d0f9806a646c4a3b42ce10b958f2b67f`.

## M. Next-stage readiness

UA3.6.4 should qualify the four core scientific scenarios using this shared
interaction path. Raw preprocessing still needs source-specific qualified operator
resources; MAESTRO needs accepted matrix/grouping/context/specification resources;
EpiZoo needs compatible checkpoint/catalog inputs and live/scientific scenario
qualification; adaptation needs owner-qualified source/SEAM/reference/strategy and
mapping specifications. Structured design and the unoffered adaptation choice
remain precisely recorded declaration boundaries. No next-stage implementation is
included. Changes remain unstaged and uncommitted for review.

UA3.6.3c implementation and bounded software/scientific interaction acceptance are
complete. Live semantic qualification remains partial as described in K. The next
stage is UA3.6.4 scenario integration, not another conversation implementation.
