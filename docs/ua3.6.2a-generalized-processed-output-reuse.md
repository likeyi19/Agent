# UA3.6.2a — Generalized trusted scientific output reuse

Implemented on 2026-10-12 against
`86a30590a8eed8cc2aa0e254845d0c1f488d5b73` on `main`. This implements the
first reusable processed-artifact boundary: a genuine registered neighbors
publication consumed by a separate registered Leiden execution. Full
conversational dependency-sensitive recomputation remains UA3.6.2b work.

## A. Baseline and preservation

Admission found `HEAD`, local `main` and existing `origin/main` equal to the
expected UA3.6.1 commit. The tracked worktree and index were clean; only
`?? evals/` was untracked. No fetch, pull, reset, clean, staging, commit or push
was performed. UA3.6.1 string declaration binding is unchanged.

The before/after preservation record is held under
`/tmp/agent-ua362a-preservation-o6qlhgce`. It inventories 1,274 existing evaluation
files, 198 output files, 208 preserved real-state files, the operator configuration,
23 UA3.5.5 scientific-state files, seven preparatory files, two human input files,
four auxiliary files and the existing checkpoint. Ordinary files are compared by
SHA-256, size, modification time and mode. The 5,231,645,507-byte checkpoint is
checked by metadata only; it is neither loaded nor rehashed. New scientific
fixtures, Runs, Sessions, reports and logs live under isolated `/tmp` workspaces.
The final comparison finds **zero mismatches across all 1,718 protected records**;
all 1,274 evaluation files remain unchanged. Index bytes and all three Git refs
also remain unchanged. Of 498 baseline source/test/benchmark files, 490 remain
byte-identical; the eight intentional tracked changes are listed in section C.

## B. Existing architecture reused

The accepted [M15.3 historical-output mechanism](m15.3-prior-output-binding.md)
remains the executable path. A captured Session Revision supplies exact accepted
output locators; the active-context projection offers opaque handles. The scoped
semantic compiler lowers the selected handle to the existing `PriorOutputRef`.
Preflight checks terminal successful source Run, accepted step, schema-2 authority,
producer port, complete semantic membership and the registered consumer channel.
The executor resolves exact persisted values and imports the accepted authority
closure. No path search, first match, language rule or upstream recomputation
substitutes for these checks.

`VerificationContext` remains operation-local. Its proofs are issued after owner
verification and stable input/publication snapshots; accepted proofs are imported
only from RunStore. File integrity, receipt/resource identity, dependency closure
and accepted-step anchors remain checked during consumption. Session records locate
science; they do not issue scientific authority. Reports and UI presentation do
not grant execution rights.

The relevant prior boundary records are
[UA3.5 closeout](ua3.5-closeout.md),
[capability readiness](ua3.5.4d-practical-capability-readiness.md),
[human EpiZoo acceptance](ua3.5.5-human-epizoo-user-workflow.md) and
[accepted matrix delivery](ua3.3.2-accepted-matrix-delivery.md). Their historical
processed-reuse limitation is extended only for the qualified neighbors owner here.

## C. Minimal implementation

| File | Purpose |
| --- | --- |
| `src/agent/orchestration/registry.py` | Immutable private `ScientificAuthorityHooks`, declared qualified producer ports and constructor validation; register the neighbors owner's callbacks. |
| `src/agent/tools/scientific_authority.py` | Shared owner dispatch; delegate existing raw/matrix/annotation/adaptation contracts to their unchanged adapter. Reject unknown, duplicate and shadowing owner kinds. |
| `src/agent/tools/analysis/neighbors_publication.py` | Owner-specific publication, source-alignment proof, receipt and exact schema-2 authority description. |
| `src/agent/tools/analysis/embedding_analysis.py` | Add source capture and bounded manifest/receipt publication around unchanged neighbors production. Four optional lightweight result fields identify the publication. |
| `src/agent/tools/data/authority_context.py` | Dispatch descriptions, proof metadata, runtime compatibility and initial source-snapshot policy through the shared owner interface. |
| `src/agent/orchestration/prior_outputs.py` | Require an explicitly owner-qualified producer port; use shared owner-kind lookup in active dependency validation. |
| `src/agent/orchestration/verification_authority.py` | Load registered owner authority through the existing exact accepted-context path; preserve the raw-only explicit legacy qualification boundary. |
| `src/agent/orchestration/verifier.py` | Delegate new publication validation to its scientific owner; retain ordinary legacy graph verification; issue authority through shared dispatch. |
| `src/agent/orchestration/executor.py` | Register exact durable owner execution before production; issue current authority after proof; revalidate already-recorded owner authority during resume. |
| `src/agent/orchestration/runtime.py` | Supply durable identity to checkpointed owner-hook plans and preserve accepted owner context during existing recovery paths. |

Three focused test modules cover the owner, generic callback/Registry contracts,
and real historical execution in `tests/tools/analysis/test_neighbors_publication.py`,
`tests/unit/orchestration/test_scientific_authority_dispatch.py` and
`tests/authority_dag/test_processed_prior_outputs.py`.

Registry inventory remains **23 tools and 172 public arguments**. The hooks contain
repository-owned callbacks, are not serialized into prompts, and add no provider
inputs. No compiler, Session schema, Web, string binder, EpiZoo implementation or
checkpoint changes are needed.

## D. Scientific ownership

The neighbors owner owns the 512-axis float32 representation, compact H5AD
structure, exact source embedding values and ordered cell IDs, sparse graph
validity, Scanpy parameters, scientific provenance and publication profile. Its
initial independent review validates these facts without rerunning graph
construction. The existing structural validator is reused; there is no second
neighbors algorithm.

Agent owns execution identity, accepted-step and authority digests, Registry
qualification, exact semantic channel/member selection, RunStore admission,
historical closure, source-policy enforcement and generation-checked Revision
publication. Compatibility stays with the registered semantic channel and the
scientific owner contract. The unchanged Leiden owner validates and consumes the
graph through its normal public API.

## E. Qualified publication and authority

The additive owner contract is `cell-neighbors-publication.v1`, artifact type
`agent.cell-neighbors`, authority kind `cell_neighbors`, and verifier
`agent.cell-neighbors-source-alignment` compatibility `1`. Profile SHA-256:

```text
ed5a7b28794d9d91ed09b2aead75dc56cc9a54f673bd5737bfe3b96f18675994
```

For the existing `<embedding-stem>.neighbors.h5ad`, the owner writes
`<embedding-stem>.neighbors.manifest.json` and
`<embedding-stem>.neighbors.receipt.json`. Each JSON file is bounded at 65,536
bytes, parsed with closed fields and duplicate/nonfinite rejection, written by
atomic replacement and fsynced. The original H5AD publication stays unchanged.
The manifest pins exact H5AD SHA-256/size, source embedding and cell-ID SHA-256/size,
ordered-cell digest, declared arguments with existing defaults, producing
execution, owner profile and reviewed lightweight summary. Original caller source
aliases and initially resolved source locators remain distinct recorded facts.

The receipt binds arguments, producing identity, contract/profile token and
manifest digest. Shared schema-2 authority includes manifest/H5AD/receipt file
closure, actual execution identity, argument/receipt digests, initial input proof
identity, owner profile/verifier, and the existing exact resource keys
`scientific_proof_sha256`, `manifest_identity`, `result_metadata`.
The reviewed proof metadata must equal the owner-qualified registered result;
altered metadata is rejected even if the manifest/result remain intact.
This is `scientific_correctness.v1` at original acceptance; later reuse establishes
`artifact_integrity_lineage_historical_sources.v1`.

The executable producer port is exactly `neighbors`, semantic type
`cell_neighbors.v1`, with member `analysis_path`. The existing Registry channel
admits `cluster_cells.analysis`; it does not admit a direct neighbors-to-UMAP
substitution. The `PriorOutputBinding` separately pins this port/type, source
Run/step/tool, entire accepted-step digest, manifest digest and authority digest.
These combined records establish the exact semantic publication; no filename or
receipt alone grants trust.

Input embeddings are explicit scientific inputs in this scope. Their hashes and
alignment prove which representation was used, not that EpiZoo inference or a
particular foundation checkpoint produced them. No EpiZoo historical authority is
invented. Existing prior-output dependencies, if present, are still imported
recursively from exact persisted bindings through the shared DAG.

Historical policy accepts the originally verified input identities without
opening their current files. Changed, archived or replaced original sources do
not alter an intact accepted publication. Explicit
`current_source_freshness.v1` additionally hashes the historical source locators
and rejects absent, unsafe or changed files. Initial verification checks both
exact source bytes and stable snapshots.

Missing/legacy proof remains ineligible. Ordinary reads, navigation, retries and
resume do not upgrade it; the existing explicit raw qualification API remains
restricted to its original owner list. Partial new result metadata fails closed.
Standalone publications carry no accepted execution identity and cannot be adopted
as Run authority merely by supplying matching paths.

No durable neighbors publication-recovery hook is introduced. Incomplete
manifest/receipt publication fails closed; a crash before accepted step persistence
does not grant reusable authority. Automatic recovery of such an in-flight
neighbors publication is outside this acceptance. An already accepted neighbors
step can supply its proof when a pending same-plan consumer resumes.

## F. Genuine separate executions

The deterministic fixture uses 32 explicitly ordered Unicode cell IDs and a
32 × 512 float32 matrix generated with seed 17, with two separated groups. No
biological matrix, foundation model or checkpoint is required. A scripted scoped-v4
model supplies typed plans only; all registered neighbors and Leiden bodies,
compiler, preflight, authority, persistence and scientific verification are real.

The original `neighbors-original:run`, step `neighbors`, completes through normal
`sessions.run`, selects `analysis_path` and activates its first Revision. A separate
application instance plans `later-clustering:run`, step `clustering`, from the
captured opaque graph handle. Its compiled argument is the exact existing
`PriorOutputRef`; only `cluster_cells` executes. Default resolution 1.0 yields two
clusters. Cell IDs, embedding values and both sparse graphs remain exactly equal.
The first Run/publication/input bytes are unchanged, and the new child Revision
preserves the old Revision. The child selects its newly verified clustering output;
it does not automatically retain or qualify additional outputs.

The new clustering result is accepted for this execution but has no reusable
cluster-owner authority. Therefore this witness establishes graph-to-Leiden reuse,
not another cross-run processed capability. Accepted evidence records the precise
historical binding. Model prompts contain no workspace paths, Run IDs or authority
digests. Duplicate turn submission and explicit back/forward navigation perform
zero additional planning, scientific production, proof reconstruction or integrity
work in their observed operation.

The representative acceptance workspace is
`/tmp/agent-ua362a-processed-prior-final/test_real_separate_runs_compil0`.
`cross-run-witness.json` records full paths, identities and invocation counts.

| Producer identity | Actual value |
| --- | --- |
| H5AD SHA-256 | `ee67ff809894d384022a0ab5753811fadc7f6fd4d8e0e375703bc65ab2952930` |
| Manifest SHA-256 | `98138b3594c8ee4350490370f4b8b2e6f21b65f034c156acbbbb2aeaa5c0e128` |
| Execution identity | `e33e88bfa75054d45fbbfcc6c3bb81b067cfe7a0ef4e33ca6d3193570d7add18` |
| Accepted-step SHA-256 | `4afcb23b31bd036fe5677b1d7088d515743af5d7974c9e911e2baccda8036526` |
| Authority SHA-256 | `7f2086d40ab8dc9194bf340ee57bdd6be861bdf83d43872d79e7264171c26e6e` |
| Parent Revision | `96cc49224e84266a38b069d75a9eb86229a88b313b029f6f8e0e1cca65370453` |
| New Revision | `a92e6d7db929dedc2f2ce7a2a65fc3ec54bbf13bf2a7eb818d30ace2eb45ec44` |

| Actual observed body | Original producer operation | Separate consumer operation |
| --- | ---: | ---: |
| Scanpy neighbors production | 1 | 0 |
| Leiden production | 0 | 1 |
| Independent neighbors owner `_verify` | 1 | 0 |
| Normal neighbors step verifier | 8 | 0 |
| Normal clustering step verifier | 0 | 8 |
| Source capture / source check | 1 / 1 | 0 / 0 |
| Embedding / cell-ID source loads | 2 / 3 | 0 / 0 |
| Shared `file_closure` | 10 | 15 |
| Graph structural validation | 3 | 19 |
| Analysis H5AD reads | 2 | 18 |
| Semantic compilation / preflight | 1 / 2 | 1 / 2 |

The normal verification counts include the existing repeated execution,
evidence/presentation and acceptance checks. File closure and graph structure
validation remain necessary; neither is neighbors production or reconstruction.

Fresh-process terminal consumer recovery, with both source files archived,
observes zero production/owner/source calls, six file closures, four clustering
verifications and eight graph checks/analysis reads. An interrupted same-plan Run
after accepted neighbors, also with archived sources, resumes with Leiden once,
neighbors/owner/source zero, six file closures, eight neighbors and eight clustering
verifications, and 19 graph checks. Its accepted graph/result/authority are
preserved. Legacy terminal and unfinished Runs retain `artifact_authority=None`;
the unfinished case runs only its pending Leiden once. Neither becomes bindable.
Those legacy fixtures run the genuine numerical owner with its original result
keys and a fixture Registry lacking the new authority hook. They simulate absent
historical qualification; no preserved real legacy Run is migrated or modified.

## G. Fail-closed coverage

The new owner suite exercises real payload/source/order/provenance/parameter
mutations, forged metadata, exact-source mismatch, changed inputs during production
or review, source aliases, and standalone execution identity substitution. A
checksum pin by itself cannot qualify a scientifically invalid representation.

The application suite rejects missing or unsupported legacy authority, incomplete
resource bindings, failed source Run or unaccepted step, changed source Run/step
and accepted arguments, forged binding pins, incorrect producer port/member,
incompatible consumer channel, invalid dependency identity, modified manifest/H5AD,
missing or changed scientific provenance/order, mutation after context capture,
unoffered handles and foreign Session/Revision scope. Explicit current freshness
rejects changed ordered IDs. Rejections run no implicit neighbors production or
owner reconstruction and create no successful scientific Revision.

Representative actual rejections include:

| Invalid contract | Observed rejection |
| --- | --- |
| Missing source Run / step | `RunNotFoundError` / `Ambiguous accepted step.` |
| Missing authority or unaccepted step | `No recorded scientific authority; legacy qualification is not implicit.` |
| Failed source Run | `Historical source run is not terminal successful.` |
| Wrong producer port | `Historical output port lacks owner-qualified scientific authority.` |
| Forged binding pins | `Historical output anchor, publication, or authority changed.` |
| Changed accepted arguments | `Neighbor publication lacks exact producing execution provenance.` |
| Changed verifier/dependency binding | `Persisted authority compatibility, lineage or resource mismatch.` |
| Changed proof metadata | `Neighbor proof metadata differs from its owner-qualified result.` |
| Missing semantic payload member | `Historical output is missing a semantic member.` |
| Wrong mapped member / consumer channel | `Historical semantic member mapping changed.` / `Historical output is incompatible with the consumer port.` |
| Unsupported schema-1 authority | `Unsupported fragments producer contract.` (existing v1 admission rejects this owner) |
| Explicit freshness mismatch | `Current source bytes differ from historical verified identity.` |
| Source mutation during initial review | `FRAGMENTS_SOURCE_CHANGED_DURING_EXECUTION` (shared snapshot guard) |

Artifact corruption is rejected by owner/accepted file closure before historical
science. Initial source-alignment negatives additionally reject reordered cells,
changed embedding values, graph parameters and provenance even when their new
file hash is supplied to standalone verification.

## H. Generality and extension

Historical binding and accepted-context loading contain no
`build_cell_neighbors` special case. Scientific facts live in its owner module;
the Registry's explicit hook registration names that module. The unchanged raw
adapter retains its original scientific responsibilities and qualification policy.
The shared dispatcher rejects owner shadowing and duplicate kinds during Registry
construction, so no first-match owner selection occurs.

Two distinct future-owner fixtures demonstrate callback dispatch and identity
separation, qualified versus unqualified ports, missing proof rejection, and raw
fallback. These are interface tests, not scientific qualification of another owner.

To qualify a future clustered-analysis owner, that owner would define its own
publication/profile, scientific verifier, exact accepted upstream graph dependency
and compatibility rules; provide description, publication-record, proof-metadata,
reuse-result and optional runtime callbacks; declare its qualified producer port
in the default immutable Registry; and earn genuine execution/mutation acceptance.
The existing compiler, `PriorOutputRef`, accepted RunStore loader, source-policy
checks and Session locators then remain shared. Current persisted-owner admission
uses the default Registry; arbitrary runtime plugins/custom Registry owners are
not qualified by this change. Embedding bundles additionally require an atomic
embedding-plus-ordered-ID owner contract. Neither owner is implemented here.

## I. Validation and preservation

All commands use conda `agent` Python,
`PYTHONPATH=src:.`, disabled `RUN_*` gates and `/tmp` cache/output paths. No full
7,000-plus-test regression, paid model call, foundation inference, adaptation
training or real raw biological preprocessing is run.

| Focused/domain command selection | Result |
| --- | --- |
| Registry, executor, recovery-policy, semantic compiler, prior-output planning, existing embedding analysis, new owner and dispatcher (final metadata-check rerun) | 247 passed; three existing warnings; 14.98 s. |
| Dispatcher, Registry and semantic metadata | 105 passed; 1.46 s. |
| Runtime, recovery policy and cancellation | 84 passed; 1.52 s. |
| New real cross-run, recovery, legacy and negative acceptance (final metadata-check rerun) | 37 passed; three existing warnings; 50.30 s. |
| Existing raw/historical authority, qualification, prior-output and application checks | 211 passed; 1,132.91 s. |
| External/neutral/canonical matrix domains | 202 passed, one gated stress skip; 53.86 s. |
| Annotation and adaptation authority/lifecycle | 60 passed; 61.73 s. |
| Sixteen affected Application/Web modules, after Runtime authority opt-in | 495 passed; three existing warnings; 145.97 s. |

These command selections overlap; totals must not be presented as unique test
counts. Owner-only acceptance also passes 21 cases. Existing numerical code is
compared against the baseline by ten AST checks: complete Leiden/UMAP bodies,
neighbors signature/defaults and Scanpy call, and six unchanged scientific/atomic
publication helpers. EpiZoo and other scientific-owner files remain unchanged.

Development acceptance exposed a missing Runtime durable-identity opt-in for an
owner with scientific authority but no recovery hook. The fresh neighbors Run
initially completed without reusable authority; binding correctly rejected it.
Adding the explicit owner-hook opt-in repairs that generic execution boundary.
The final real witness and affected Application/Web regression run after the fix.
An owner test's symlink exception expectation was corrected to the actual shared
snapshot error; the stable focused suite passes. No scientific algorithm repair
or changed default was needed.
Final review also demonstrated an inconsistent new-owner `result_metadata`
being accepted by the description callback. The owner now requires equality to
its reviewed manifest/result. A dedicated corruption case observes the exact
rejection above; all 37 real integration cases and the 247 contract cases pass
again after this change. The original accepted witness remains valid and unchanged.

The core command is:

```bash
PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 \
/home/likeyi/anaconda3/envs/agent/bin/python -B -m pytest \
tests/authority_dag/test_processed_prior_outputs.py -q -s -x \
-p no:cacheprovider --basetemp=/tmp/agent-ua362a-processed-prior-metadata-recheck --tb=long
```

The 247-case command selects exactly:

```text
tests/unit/orchestration/test_registry.py
tests/unit/orchestration/test_executor.py
tests/unit/orchestration/test_recovery_policy.py
tests/unit/orchestration/test_semantic_compiler.py
tests/unit/orchestration/test_prior_output_planning.py
tests/tools/analysis/test_embedding_analysis.py
tests/tools/analysis/test_neighbors_publication.py
tests/unit/orchestration/test_scientific_authority_dispatch.py
```

Its invocation is the same Python/environment with `-B -m pytest -q
-p no:cacheprovider` and those eight paths. Logs are
`/tmp/agent-ua362a-focused-contracts-final.log`,
`/tmp/ua362a-authority-registry.log`, and
`/tmp/ua362a-runtime-recovery-cancellation.log`.
Annotation/adaptation selects
`tests/annotation_integration/test_annotation.py`,
`tests/species_adaptation_integration/test_public.py` and
`tests/epizoo_adaptation/test_adaptation_lifecycle.py` with `-m pytest -q`;
log `/tmp/ua362a-annotation-adaptation-regression.log`.
Matrix invocation and complete log are
`/tmp/agent-ua362a-matrix-domain-92me40tv/{command.txt,pytest.log}`;
Application/Web invocation and complete log are
`/tmp/agent-ua362a-app-web-authority-4wpefj6w/{command.txt,pytest.log}`.
Both command records list all exact selections, cache paths and isolated basetemp.
All listed completed commands exit 0. The existing raw/historical-authority domain
command uses `-m pytest -q` and these seven exact paths:

```text
tests/authority_dag/test_context.py
tests/authority_dag/test_propagation.py
tests/authority_dag/test_qualification.py
tests/authority_dag/test_application.py
tests/authority_dag/test_prior_outputs.py
tests/chromap/test_authority_dag.py
tests/chromap/test_verification_authority.py
```

Its log is `/tmp/agent-ua362a-raw-authority-regression.log`. The matrix,
annotation/adaptation, raw/authority and Application/Web selections together cover
968 passed affected domain tests plus one gated skip; the additional focused/unit
selections overlap and are reported separately. The three repeated warnings are the existing
`pkg_resources`, `louvain` and TBB/Numba dependency notices. The sole skipped case
is the explicit synthetic matrix stress gate.

`processed-acceptance-result.json` in the preservation directory records the exact
36-case command, observed counters, exceptions and recovery results. It is labeled
an observed summary reconstructed from tool output, not a verbatim pytest log.
The final 37-case stdout/stderr log, including the additional metadata negative,
is `processed-prior-metadata-recheck.log` in that same directory. The earlier
36-case run passed in 50.88 s before the final equality check.
The representative `cross-run-witness.json` is the directly emitted test artifact.
`/tmp/ua362a-provider-footprint-comparison.json` and its companion replay script
compare baseline/current catalogs, v3/v4 prompts and schemas, compiler contract,
result planning fields and planning-scope fingerprint: all eight projection
groups match exactly. The existing frozen v4 fixture remains 33,998 bytes; the
complete embedding-analysis scoped fixture remains 10,435 bytes. These are exact
fixture measurements, not universal prompt-size claims.

`post-review-preservation.json` records the zero-mismatch final historical check
and unchanged refs/index after the metadata repair;
`implementation-final-freeze.json` pins all 503 current source/test/benchmark
files, including the five new files. Earlier preservation/freeze records remain
available. `git diff --check` passes. No old evaluations or acceptance documents
are rewritten.

## J. Remaining work

UA3.6.2b still needs owner-qualified embedding/ordered-ID publication and accepted
upstream closure, clustered/UMAP publication authority where reusable results are
needed, and scoped language/planning acceptance for explicit resolution or
neighbors changes. The existing natural-language interpreter must select intent
and references; no phrase rules or automatic downstream completion are introduced.
New Leiden/UMAP plans must name their required outputs and dependencies, then use
the existing generation-checked Revision mechanism. This stage already demonstrates
that mechanism for separate neighbors/Leiden Runs.

Processed H5AD/embedding downloads remain UA3.6.2c work with explicit authority,
payload, privacy, companion and transport contracts, following accepted matrix
delivery. No Web change or download eligibility is added. Human EpiZoo resources,
raw resource provisioning, MAESTRO interaction, adaptation interaction, experimental
design binding, broad Planner redesign and new profiles remain outside this stage.

UA3.6.2a implementation and focused acceptance are complete within this bounded
scope. Changes remain unstaged and uncommitted for review. No next-stage work,
commit or push is performed.
