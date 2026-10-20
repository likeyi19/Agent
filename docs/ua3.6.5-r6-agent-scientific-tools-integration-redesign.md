# UA3.6.5-R6 — Agent–Scientific Tools Integration Redesign

R6 implements a simpler scientific input, execution and continuation boundary.
The LLM selects registered science and offered inputs; Agent binds identities,
resources and execution state; scientific owners parse their inputs and compute;
Agent presents accepted facts and downloadable results. R5 planning, cited
answers and semantic review remain the accepted architecture.

The implementation and cross-tool follow-up have passed final engineering review.
Annotation acceptance evidence is outside Git at `/home/likeyi/agent-ua365-r6-msezvdc1`.
The subsequent cross-tool review and acceptance are at
`/home/likeyi/agent-ua365-r6-cross-sy9ei_vl`; the final section below supersedes
the earlier annotation-only file inventory and disposition.
The scope is bounded engineering and product acceptance, not annotation accuracy
or universal scientific-format qualification.

## Baseline and protection

Before editing, HEAD, local `main`, `origin/main` and remote `main` agreed at
`266aa828cb46764b6671588c7c8796b8f756e262`
(`Complete UA3.6.5-R5 LLM-Agent interface and scientific interaction redesign`).
Tracked worktree and index were clean; only historical `evals/` was untracked.
README, AGENTS, R1–R5 records and the current implementation/consumers were read.
An archived baseline source tree supports isolated before/after checks without
changing the working branch.

`baseline.json` inventories 261,016 protected files. SHA-256 is recorded for
253,308 files, including the operator configuration and historical eval files;
remaining large historical files use size, nanosecond mtime and mode. This is a
bounded inventory, not a claim that all historical checkpoint/data bytes were
rehash-read. Selected scientific inputs/resources and delivered artifacts also
receive their normal exact integrity checks.

The normal operator configuration, scientific library, web workspace, accepted
datasets/checkpoints and historical acceptance directories are protected.
Acceptance uses a new workspace, upload directory, operator-config copy and
Firefox profile. The copied config points to the existing scientific library and
configured OpenRouter model. No new reference download, library installation or
production-default replacement is needed. Data and evidence remain outside Git.

## Initial call graph and owner audit

The actual direct-annotation path was:

1. Web upload → `LocalResourceAdmission` immutable private registration.
2. `turns`/typed interpreter → `raw_handoff` observation, declarations and resource
   resolution → internally generated specification and bound inputs.
3. Registry semantic planning → compiler source/parameter binding → runtime
   preflight → `annotate_scATAC_cell_types`.
4. Annotation `_build`: qualify source, native MAESTRO, load source for derivative,
   stage artifacts and manifest → `verify_annotation`.
5. The owner verifier qualified inputs again, reran the identical native backend,
   compared intermediate science and exact derivative, then issued operation proof.
6. Atomic publication/receipt → public result verifier → orchestration verifier →
   accepted scientific authority → Session/Revision.
7. Evidence/report/download consumers and terminal resume reused accepted authority.
   Calls into the verifier wrapper were not necessarily calls into its body.
   Standalone completed-publication recovery without an imported proof replayed
   native annotation again.

The audit followed consumers rather than treating historical tests as requirements.

| Boundary and current consumer | Invalid condition / origin | Decision and resulting owner |
| --- | --- | --- |
| Local resource registration, `input_sets`, `sessions`, prior outputs | Wrong Session, changed bytes, unsupported role or stale generation; integrity | **KEEP.** Agent owns registration and exact binding. Users do not construct internal hashes, manifests or Run identities. |
| `annotation_input.observe` and `peak_h5ad.inspect` encoding/attribute/top-level allowlists | Historically rejected valid AnnData encodings, producer attributes and numeric multidimensional metadata before native parsing | **REMOVE.** AnnData owns format interpretation. The separate observation-column limit of 64 is removed. |
| HDF5 external/soft links, repeated objects, virtual/external storage, plugin filters, size/depth/string bounds | Indirect file access, cycles, arbitrary filter loading, excessive allocation; execution safety | **KEEP / SIMPLIFY.** A storage walk remains before AnnData; it no longer defines a parallel AnnData encoding grammar. Resource limits can still reject large valid scientific files and are not biological requirements. |
| Private paths in supplied `obs`/`var`/`uns` | Download disclosure; previously also blocked scientific execution | **MOVE.** Accept ordinary private metadata for science. Delivery omits unsafe metadata fields, records omission count, preserves scientific values/axes and validates delivered bytes. |
| `external_marker_annotation` counts, coordinates, species/assembly, groups and references | Incompatible method measurements, malformed intervals, missing/misaligned cells; native science | **KEEP.** Human hg38 cut-site measurement history remains explicit; no silent binarization of the stored input, alias repair, normalization, dropping or relabeling. Native private working copies remain native science. |
| `count_slot` prerequisite | Mandatory declaration despite a usable native X default; historical Agent policy | **SIMPLIFY.** Omission reaches the owner, which uses X and records it. Explicit layers remain explicit. Old captured clarification records retain their original meaning. |
| Repeated source loads in qualifier/backend/derivative | Same immutable source loaded for adjacent owner phases | **SIMPLIFY.** `prepare` returns its qualified AnnData to the native call and derivative writer; completion loads a separate original once for exact preservation. No durable cache. |
| `verify_annotation` native replay | Duplicated identical RP/statistics/scoring; useful development qualification, not independent biological truth | **REMOVE from production.** Native completion, exact association/propagation, sparse result structure, sidecar/receipt integrity and source preservation remain. Explicit native oracle comparison supplies separate scientific qualification. |
| Scientific authority closure checks | Changed selected resources, artifacts or receipt | **SIMPLIFY.** Reuse the owner's checked closure; check the outer receipt once. Complete expected closure comparison and operation-local proof validation remain. No new authority framework. |
| Adapter source hashes in old annotation authority | Deployment implementation identity rather than changed accepted scientific bytes | **HISTORICAL READ ONLY.** Only the two retired annotation/HDF adapter pins use their recorded old entries after accepted schema-1 authority import. Source, native runtime, scientific resource and output identities remain checked. |
| `ScientificResourceLibrary`, `shared_resources`, `reference_supplementation` | Missing or incompatible reference/model; changed selected bytes | **KEEP.** Reuse installation qualification and selected-file integrity. No second catalog/promotion workflow; task data and generated groups remain private. |
| PCA owner and Registry/compiler | Invalid numeric X, dimensions, fit and ordered axes | **KEEP.** Sparse centered ARPACK, native effective parameters and independent fit checks; no EpiZoo vocabulary gate. This verifier checks mathematical fit rather than rerunning the same PCA algorithm. |
| EpiZoo handoff/native owner | Wrong checkpoint vocabulary, species/feature space or inference settings | **KEEP.** Exact model requirements remain; no error-driven PCA fallback. `tools/models/epizoo.py` is unchanged. |
| Neighbors/Leiden/UMAP and prior-output binding | Wrong representation, graph, dimensions, ordered cells or dependency | **KEEP.** Explicit accepted outputs and graph publication v1/v2 readers; no graph reconstruction on reads. |
| Accepted clustering → annotation | User otherwise had to export/reupload groups; native needs original counts plus ordered labels | **SIMPLIFY.** A small deterministic Session-private TSV projection follows exact accepted dependencies and requires the PCA input-file digest and ordered cells to match. |
| FASTQ/BAM/fragments handoffs and scientific owners | Assay/read roles, barcode history, Tn5/coordinate semantics, source conservation, scope and reference compatibility | **KEEP.** These define the measurement. Existing owners prepare internal manifests, bind independent uploads and reuse accepted output authority. No unrelated raw-science redesign or new real raw run. |
| Other Registry consumers of authority infrastructure | Incomplete publication, missing files, stale lineage | **KEEP.** Affected Registry/compiler/verifier, authority-DAG and embedding regressions cover shared changes; unaffected algorithms are not mechanically rewritten. |

## Implemented handoff and execution

`annotation_input` still supplies bounded metadata observations. It does not
claim scientific eligibility. The owner parses actual AnnData and validates
the chosen measurement slot, peak coordinates, exact groups and resources.
Nullable observation columns, more than 64 metadata columns, ordinary custom
attributes and numeric 3-D metadata now reach the owner. Scientific outputs
retain source matrices, ordered axes, layers, pairwise arrays, raw data and
metadata; only a separate download projection omits private metadata fields.

`external_marker_annotation.prepare` returns the already qualified matrix.
The native runner consumes it directly and the same object supplies the
derivative. Completion freshly reads the immutable original and checks the
derivative exactly. The native RP, marker testing, scoring and participation
algorithms and the pinned R scripts are unchanged.

New annotation manifests use schema 2 under the existing artifact contract.
Their profile records
`native_completion_exact_propagation_and_integrity.v2`, and their verifier is
`agent.cell-type-annotation-completion`, compatibility version 2. The outer
scientific authority schema and publication/recovery infrastructure are reused.
This explicitly distinguishes the guarantee from old replay-qualified results.

Completion checks source/resource association, ordered cell/group files,
group sizes, allowed status/label combinations, exact group-to-cell propagation,
finite nonnegative sparse RP shape with usable rows, every published sidecar
digest/size, summary counts, participation/provenance agreement and exact
annotated-source preservation. It does not independently endorse biological
labels or recompute RP, markers or candidate scores.

The direct native `count_slot` default is X. The owner’s normalized manifest
arguments record the effective value even when the plan contains no user
declaration for it. Resource resolution still selects the existing compatible
installed reference. Output directories and identity records remain Agent-owned.
Cancellation, temporary publication cleanup, atomic rename/fsync and receipts
remain with the existing owners.

## Accepted grouping continuation

`application/annotation_groups.py` is a narrow data projection, not a generic
converter. The captured Revision exposes opaque choices for its accepted
`cluster_cells` outputs or explicit clustering ancestors of UMAP outputs.
The LLM selects an offered handle. Agent validates exact accepted anchors,
follows `StepOutputRef`/`PriorOutputRef` dependencies to the neighbor graph and
representation, and checks the PCA source digest against the supplied count file.
The accepted cluster artifact must have exactly the same ordered cells.

The resulting ordered `cell_id/group` TSV is atomically written and registered
through the existing local resource owner in the current Session. It does not
enter the common library. The original captured input members remain bound.
Annotation receives original counts, not embedding coordinates. No clustering,
PCA or neighbor reconstruction is performed during this projection.

A valid result lacking exact source association yields a persisted prerequisite
clarification. A newly uploaded independent grouping can explicitly replace
that unusable grouping choice while preserving captured count inputs.
Wrong handles and corrupt accepted artifacts fail closed. Historical EpiZoo
publications lacking an exact source-file digest cannot establish this association;
equal cells, filenames or count totals are insufficient. They retain existing
downstream embedding analysis and explicit independent-group annotation routes.

## Removed machinery and migrated consumers

Removed the native replay temporary directory, repeated RP/marker/scoring
invocation and pairwise replay comparisons from the production owner verifier.
Removed obsolete HDF encoding/attribute/top-level/array-rank/fill-value vetoes
and the observation-column count veto. Moved private metadata checks from
scientific intake to exported-artifact handling. Reused adjacent owner input
loads and checked authority closures without introducing a persistent cache.

Migrated result validation, verifier wording, authority description and receipt
lookup, delivery provenance and profile dispatch, typed role continuation,
interpreter offered choices, and affected tests. Tests now require one native
production call and no recovery replay instead of preserving obsolete call
counts. R5's planner/compiler/answer design remains unchanged; the interpreter
schema/prompt extension only exposes the new accepted-group choice.

## Historical compatibility

Schema-1 manifests keep their exact old profile, receipt and verifier identities.
Version-aware publication-location readers resolve both current and old accepted
records. The historical proof loader uses recorded entries for the two retired
adapter implementation dependencies, only with imported accepted v1 authority.
It continues checking scientific files and the exact expected closure.

Old captured annotation observations without `interface_version` reproduce their
old mandatory-slot clarification, preserving immutable Session records. New
observations use interface version 2. Missing optional `groups_result` is omitted
from declaration serialization, preserving old declaration identities.

The obsolete replay implementation is not retained. An orphaned schema-1
publication without its accepted replay authority cannot be freshly requalified
by the schema-2 completion verifier. This explicit retired behavior avoids
silently assigning the old scientific guarantee to new checks.

Historical original artifacts are never sanitized in place. The accepted old
download stays byte-identical; new private-metadata omissions belong only to
schema-2 delivery. Essential scientific axes are never rewritten for privacy.
Unusual private metadata outside the supported omission fields fails delivery
safely rather than leaking or silently rewriting scientific identity.

## Optional comparison status

CellMarker corroboration exists as the reusable private
`marker_validation.corroborate_marker_annotation` component, with pinned
resource/context and exact candidate/gene mappings. Its tests remain applicable.
It is not a registered public tool and is not needed for primary MAESTRO.
No second comparison/evaluation framework or automatic label correction is added.

EpiAgent must not be described as nonexistent or as an ordinary registered Agent
backend. The early int64 diagnostic was superseded by the
[numeric/checkpoint qualification](m13.4c-numeric-checkpoint-follow-up.md) and
[full PBMC native qualification](m13.4c-full-pbmc-validation.md).
Preserved private scripts in the external M13.4c acceptance directories implement
the exact sparse int64≤2^24→float32 working-copy boundary and native EpiAgent-NT
inference. The pinned checkpoint is 3,020,477,066 bytes, SHA-256
`8d0bffe12b887205a77e9f7a2cdae09b2151ccd6ddcc95e0c2f36c1ec6ff797b`.
That accepted run processed 1,119 cells with batch size 4, maximum length 8,192,
seed 0 and about 3.53 GiB peak allocated GPU memory. It is not rerun in R6.

No current Registry function wraps those scripts. Public integration requires
explicit canonical vocabulary/model/resource ownership and output/comparison
contracts, beyond exposing an existing production function. Preserving them
without adding a new model backend fits this task. The historical 899/904
broad-lineage concordance is conditional agreement, not accuracy; the model has
no explicit NK class. Neither optional comparator gates or overwrites MAESTRO.

## Validation and real-product evidence

Software tests, simulated scientific fixtures and genuine native/provider
acceptance are reported separately. No all-milestone/full-suite rerun is used.
The final focused results and preservation/Git closure are recorded below.

| Evidence file | Scope and outcome |
| --- | --- |
| `affected-owners.log` | Registry, semantic compiler, orchestration verifier, authority DAG/propagation, PCA/EpiZoo continuation and embedding: **312 passed**, six warnings, 392.58 s. |
| `final-focused.log` | Annotation input/owner, conversation, uploaded references, workflow and initial R6 cases: **105 passed**, four warnings, 165.25 s. |
| `final-raw-regression.log` | BAM/FASTQ declarations/composition, Web handoffs, resource resolution/supplementation: **172 passed, one skipped**, two warnings, 94.06 s. |
| `final-handoffs.log` | Fragment declaration/upload/composition/Web paths plus R6: **128 passed**, one new fixture upload-type failure, three warnings, 120.49 s. The corrected R6 file passes in the next row; the fragment cases pass unchanged. |
| `final-r6.log` | Final new native-encoding/privacy, storage-safety, default/recovery and exact grouping tests: **7 passed**, three warnings, 23.08 s. Includes real Scanpy PCA/neighbors/Leiden/UMAP on simulated counts with simulated annotation, changed-source clarification and successful uploaded-group repair. |
| `baseline-source-counts.log`, `current-source-counts.log` | Instrumented direct production/terminal-resume test on isolated old/current source: **one passed each**, 2.23 / 2.12 s. |

These overlapping suites are not summed into a distinct-test total. The earlier
`affected-application.log` passed 358 tests with one failure in new test setup
(missing explicit Session context in a direct helper assertion); the corrected
R6 tests pass. Existing marker annotation and private marker-validation tests
were included there. The added recovery test initially used the legacy
`input_type=groups` on the Session upload route; the normal Session route uses
`attachment`. Correcting the fixture exercises the real route without changing
production upload policy. A diff-review error placing annotation fields in
`FragmentsDeclaration` was removed and the fragment regressions above cover it.
No observed regression failure is left unexplained or waived.

Warnings concern existing dependency deprecations/TBB and the deliberate
duplicate-cell negative fixture. The one skipped case is not counted as passed;
the quiet log did not collect its skip reason. The complete lightweight suite
was not rerun.

### Production and recovery accounting

An isolated R5 archive runs the original production/terminal-reuse and recovery
tests: 2 passed in 5.64 seconds. They demonstrate two native calls for one
production execution (one production plus one replay), zero additional calls
for accepted terminal resume, and one further native replay for standalone
completed-publication recovery. These use a simulated native fixture to isolate
call counting; they do not establish new biology.

Instrumented `ResearchAgentApplication.run` measures **six full source AnnData
loads before and two after**, and **two native calls before and one after**,
excluding the test's later comparison reads. The source-count JSON files record
each load and whether it was backed. This is direct owner production accounting,
not an assertion that Web metadata observation, group projection, upstream tools
or every later reader perform no HDF5 reads.

R6 production requires one native call. Standalone recovery of a current
publication performs completion/integrity checks with zero native calls; accepted
terminal recovery also performs zero native calls. Source qualification is
performed once in production and once in the fresh completion check. Proof-cache
wrapper entries must not be reported as repeated scientific computation.

### Explicit native scientific qualification

`native_oracle_check.py` performs one separate genuine native execution on the
preserved 1,004-cell, six-group PBMC case and compares it with the already
replay-qualified R5 oracle. It takes 26.35 seconds and matches every sparse RP
value/axis (1,004 × 28,307), marker table, group/cell assignment, candidate/coverage
record and feature-participation file exactly. There is no automatic replay.
`native-oracle-check.json` records baseline manifest
`6810c611494e06d25f06fd1d1dd06c3e03a96c359ddf7a80a6a0852223a05c69`
and current manifest
`98d6851485327b1618a64434fd841ad6f550ef36aa4a804b3025b6fd94b88e7c`.
This supports preservation of native science, not ground-truth cell labels.

### Real Firefox, OpenRouter and native continuation

The normal Web application uses native headless Firefox and configured profile
`openrouter-gpt-5-6-sol` (`openai/gpt-5.6-sol:floor`). Instrumentation records
entry counts and actual provider requests/responses without substituting decisions
or scientific results. Credentials are not recorded.

Session `24e9ae4c-a7fd-4c9e-b744-47d37bda421b` uploads the preserved real
1,004 × 82,579 PBMC peak-count H5AD, SHA-256
`c75bca64515710f6045a1c65d9f0dfb9fe9827102501856007b2bfcb58b33c81`.
The first ordinary request explicitly asks for PCA (five components), neighbors
and Leiden resolution 0.5, retaining supplied counts without normalization.
Native PCA, neighbors and Leiden each run once. Revision:
`b7d9c964691e0fe293de0fcd82f9b27afe04501a4b9f09723e7c89fc4bc5b06f`.

The follow-up requests MAESTRO on the original counts using the just-computed
Leiden groups and installed compatible human PBMC reference. The real interpreter
selects the exact offered group handle; the plan contains only annotation.
No group download/re-upload or clarification is required.
MAESTRO/RP runs once, source preparation twice, upstream native science zero.
Nine completion-wrapper entries reduce to one actual completion-body execution
and proof reuse. Revision:
`f27b700f902a333a4af5cc344967ddc933064a36381fa31af57516ac3e8ea42a`.
All 1,004 cells remain: 14 groups, 901 assigned, 103 unassigned, zero ambiguous.
These different labels/counts from the six-group oracle reflect a different
explicit grouping, not a claim that one grouping is biologically better.

A real follow-up answer uses accepted evidence and R5 generation/review to report
those counts and scientific limitations, with no new science or Revision.
Browser evidence includes `pca-clusters`, `maestro-continuation`,
`annotation-answer`, report text, screenshots and audit JSONL.

A separate real Session `c0719866-2508-4fb0-9337-f45ae9344e78` explicitly requests
EpiZoo on incompatible original peaks and prohibits substitution. It remains at
generation zero with no scientific execution/Revision, asking for a compatible
qualified checkpoint/vocabulary. This is a successful no-substitution boundary;
the provider's explanation is resource-oriented rather than a precise
feature-by-feature mismatch diagnosis.

A real browser continuation activates the earlier PCA/graph Revision and requests
Leiden resolution 0.8 plus UMAP, explicitly reusing the graph. The real provider
plans only Leiden and UMAP. Each native algorithm runs once; total native counts
across the whole Session remain PCA 1, neighbors 1, Leiden 2, UMAP 1, annotation 1.
The branch has 17 clusters; UMAP uses two components, `min_dist=0.5` and
`spread=1.0`. The original neighbor graph has 15 neighbors. Actual bound arguments,
results and artifact identities are captured in `real-run-summary.json`.
The new Revision
`c9a4ca61d4b7fc737ca5f434d55b29b5f034c068109feb6dbfe1ab74756014cb`
is a sibling of the annotation Revision with the same PCA/graph parent.
The earlier annotation remains accessible. `real-branch.json` contains an early
browser snapshot; `final-restart.json` and `real-branch-final.png` establish the
completed generation-4 state, three exact revisions and sibling identities.

### Historical and restart reads

Native browser inspection, reports, scientific downloads and refresh preserve
all acceptance-workspace files with zero additional provider/science calls.
A fresh server process repeats the read checks against the final implementation.
After branching, final fresh-process reads preserve all 77 private-workspace
files, with no provider, scientific computation or state writes.

The protected normal R5 Session
`289679ed-c548-493d-a8a1-14906870215e` reopens with its exact two revisions,
generation 2, stored responses and guidance. All 228 normal-workspace files
remain byte-identical. The old annotated download remains exactly 79,569,234
bytes with SHA-256
`c94c784c53ed55d99015fdcb25fc82fbeb295526b926ba0b455acb1998ef763f`.
Provider creation, runtime execution and owner verification are forbidden by
instrumentation during this read check. A separate accepted-authority load
restores one old annotation proof with native execution forbidden.

## Remaining limits and disposition

The direct annotation owner remains qualified for declared human hg38 Cell
Ranger ATAC cut-site measurements and compatible resources. This work does not
prove upload history, biological purity or annotation accuracy. Arbitrary-species
annotation, transformed measurements and arbitrary reference semantics are not
implicitly admitted. Native algorithms and all counts/axes remain unchanged.

Storage/metadata allocation limits remain (including 1 GiB source and 2 GiB
decoded bounds); not every AnnData encoding or arbitrarily large valid dataset
is qualified. Private metadata delivery has a bounded omission policy, not an
unrestricted sanitization engine. Native X default correctness is software-tested;
the real browser request explicitly selected X.

Accepted-group projection currently requires recorded PCA source-file identity.
Historical EpiZoo source correspondence cannot be manufactured. Older orphaned
annotation publications require their accepted authority; standalone v1 replay
is retired. Optional private comparisons remain private. Raw preprocessing is
audited and regression-covered, not newly qualified on real sequencing data.
Live provider behavior is demonstrated for this bounded flow, not guaranteed
for every paraphrase.

`protection-final.json` reports **261,016 files checked, 253,308 SHA-256 checked,
zero changed and zero missing**. All **1,274** historical `evals/` files retain
their exact file set and recorded bytes. Normal operator/library/defaults and
protected historical scientific state remain unchanged.

The diff contains the annotation/storage/delivery and authority changes,
the small accepted-groups handoff and typed consumers, six annotation test files,
and README/AGENTS/this record. No native scientific algorithm/model, biological
dataset, checkpoint, credential or external acceptance artifact is added to Git.
The exact final file inventory and ref/index checks are in `git-closeout.json`
outside Git. Source-test checkpoint differences are recorded separately; final
changes are covered by focused follow-up checks rather than falsely claiming
the entire working tree was frozen through all development tests.
There are **23 task files: 20 tracked modifications and three new files**;
the existing `evals/` remains untracked. `git diff --check` passes, the index is
empty, and all four Git refs still equal the verified R5 baseline. Native RP,
coordinate, count and grouping function ASTs match the archived baseline;
EpiZoo, primary marker scoring, private CellMarker and pinned R source bytes
are unchanged.

**Annotation checkpoint: complete for its bounded scope, ready for review.**
Real native science, live Web continuation/branching, necessary integrity and
historical continuity are demonstrated. The explicit limitations above remain;
they do not authorize guessed compatibility or a new scientific backend.
Nothing is staged, committed or pushed, and no subsequent milestone is begun.

## Cross-tool input and verification follow-up

The follow-up audits the complete current Registry (24 tools), with implementation
and focused acceptance where the audit identified unnecessary restrictions.
It does not claim that every native tool, species, format or real sequencing
workflow has been newly qualified. The earlier annotation work remains intact.
Evidence in this section is relative to
`/home/likeyi/agent-ua365-r6-cross-sy9ei_vl`.

### Shared responsibility and an actual application defect

Upload and `LocalResourceAdmission.register` retain byte streaming, quotas,
approved roots, regular-file/no-symlink requirements, Session permissions,
captured identities and integrity snapshots. Neither registration nor an
`input_type` hint qualifies scientific contents. `input_sets` and raw/resource
handoffs bind explicit roles, declarations and resources; their shallow type and
allowed-argument checks prevent ambiguous or unauthorized binding. Registry
allowlisting, actual signatures, compiler dependency/lineage checks and executor
preflight remain. Native input parsers and scientific owners interpret contents.

The first live test exposed `MULTI_INPUT_SOURCE_MISMATCH`: the interpreter
selected the uploaded dataset for PCA, and the planner correctly bound
inspection's `input_path` output to PCA. The application guard incorrectly
required a literal path at that port. The guard now follows that exact
identity-preserving inspection output, including cycle/unknown-output rejection,
and permits inspection of selected bytes. Actual resolved consumers still must
match the registered source, and their byte pins are checked at execution and
acceptance. Other transformations cannot stand in for the captured file.
No inspection step is inserted, no step order is inferred, and no scientific
tool-pair adapter or additional workflow mechanism is introduced. R5 interpreter,
planner, compiler and answer algorithms are unchanged.

### Capability audit and changes

| Registered capabilities | Admission decision and scientific requirements retained |
| --- | --- |
| `inspect_scATAC`, `pca_embed_cells` | Remove suffix-only gates; AnnData parses actual H5AD bytes. PCA retains finite real X, exact cells, bounded allocation and valid ARPACK dimensions; no checkpoint vocabulary, count normalization, scaling or filtering is imposed. |
| `epizoo_embed_cells`, `adapt_epizoo_species` | Keep qualified checkpoint/vocabulary, declared species/model compatibility and explicit training/reference semantics. These determine model meaning, not generic H5AD eligibility. Native EpiZoo source is unchanged; adaptation was audited, not newly trained. |
| `build_cell_neighbors`, `cluster_cells`, `compute_cell_umap` | Remove shared input suffix checks and the duplicate neighbor-publication suffix gate. Detect HDF representation from bytes, otherwise use the safe NPY parser. Keep representation dimensions, exact cell order, graph identity, native parameters and publication association. Historical neighbor v1 retains its original profile and NPY identity contract. |
| `evaluate_cell_clustering`, `evaluate_cell_annotation` | Remove H5AD suffix checks; retain exact cell correspondence, required label/cluster columns and metric eligibility. |
| `transfer_cell_labels` | Remove NPY/TXT/H5AD filename gates and migrate every helper caller. Keep actual parsers, aligned cells, compatible species/checkpoints and this method's EpiZoo-specific 512-dimensional float32 representation. This does not silently generalize label transfer to unrelated embeddings. |
| `validate_scATAC_feature_space`, `build_replicate_pseudobulk` | Remove sparse-only source storage and suffix gates. Dense sources are read in bounded row blocks and converted locally to CSR; existing CSR/CSC paths never densify. Remove the human/hg38–mouse/mm10 allowlist and Registry choices: aggregation uses no genome reference. Preserve exact supplied declarations without claiming reference qualification. Retain finite nonnegative exact integers, declared measurement, unique cell/feature IDs, explicit metadata and overflow checks. Distinct features may share coordinates; a one-base closed interval is valid. |
| `run_replicate_differential_accessibility` | Remove H5AD/CSR-only consumption and cell-level source/manifest reconstruction from input preparation. Validate the supplied aggregate, logical counts/digests, exact units, ordered features, libraries, covariates and experiment design. Keep eligible fragment/insertion counts, independent/paired replication, rank, residual degrees of freedom, contrast estimability, pinned R pipeline and memory/numeric bounds. |
| `inspect_raw_scATAC`, `prepare_scATAC_fragments` | Keep bounded discovery and explicit FASTQ member roles, read synchronization, assay/layout/whitelist/reference declarations and supported native alignment profile. Explicit declarations already bypass filename-based discovery; compression is inspected from bytes. Discovery suffix hints do not replace explicit declaration admission. No Chromap replay is added. |
| `prepare_scATAC_bam_fragments`, `import_scATAC_fragments` | Keep explicit barcode/coordinate/support history, exact references, primary-nuclear scope where applicable and source conservation. BAM indices/sorting are not blanket prerequisites; malformed records fail rather than being silently discarded. Public arbitrary-species BAM wiring is not introduced. |
| `compute_scATAC_qc`, `select_scATAC_cells` | Keep independently qualified TSS/reference semantics, exact barcode namespaces, explicit thresholds and definedness. No hidden cell calling, threshold defaults or alias repair is added. |
| `build_scATAC_cell_by_ccre`, `build_scATAC_cell_by_features`, `adopt_scATAC_cell_by_ccre`, `adopt_scATAC_cell_by_features` | Keep actual fragment/count semantics, exact selected cells/features, compatible reference/assembly/scope and independent count/order verification. These establish the measurement, not merely a preferred file layout. |
| `annotate_scATAC_cell_types` | Retain the earlier R6 native AnnData admission, one MAESTRO execution, exact measurement/resources and narrow accepted-Leiden group projection. No generic tool-pair conversion system is added. |

New feature-space manifests use **schema 2**. `source_sparse` reflects actual
storage; the existing `source_sparse_format` result field can be `dense`, `csr`
or `csc`. Biology records say `supplied_declarations` and
`reference_compatibility_assessed=false`, replacing the old allowlist claim.
The small schema-1 reader retains its old species/storage/coordinate constraints
and reconstructs the exact historical manifest. Historical source identity and
pseudobulk unit digests do not change. Equal coordinates are not merged and
one-based coordinates are not converted. The float-to-int64 boundary explicitly
rejects 2^63 rather than permitting an overflowing cast.

### Post-execution verification: measured change and retained work

| Path | Ordinary new publication | Decision and evidence |
| --- | --- | --- |
| MAESTRO | Native execution 1, formerly 2 | Earlier R6 completion checks and explicit independent native comparison remain applicable; no new annotation replay. |
| DA | Native edgeR 1; verifier edgeR 0 | `before-da.log` records native edgeR 1 + oracle edgeR 1, plus one upstream production SUM and one independent upstream SUM. `after-da.log` records only native edgeR 1; all three additional computations are absent. The same tiny simulated experiment and archived baseline source were used. |
| DA completion | Input/count digests, metadata/design association, native finite/range/TMM/BH checks, exact artifact/compact-result association and source/output integrity | New schema 2 explicitly records `native-result-integrity.v2` and no upstream aggregation reconstruction. This does not claim an independent refit. The independent pinned R oracle remains an explicit qualification test, outside ordinary runtime/recovery dispatch. |
| Pseudobulk | One native sparse aggregation plus one independent Python-integer exact SUM for a newly published aggregate | Retained to detect cell-to-unit mapping, axis or integer accumulation errors. DA no longer repeats either aggregation. Dense-source iteration is shared only as bounded input access, not the independent summation algorithm. |
| QC | One production result and one independent per-barcode reconstruction | Retained: production BEDTools overlap and independent bisect/SQLite TSS reconstruction check different implementations of the measurement. |
| Selection | One independent threshold/decision/identity check | Retained: exact rational threshold parsing, missing/undefined values, row conservation and selected-cell identity are substantive and bounded. No upstream QC recomputation. |
| cCRE/feature matrix | One independent interval-tree count reconstruction | Retained: checks the production BEDTools intersections, every sparse value and exact selected axes. No upstream producer/QC/selection replay. |
| BAM/external fragments | BAM sequential projection 2; external production scan 1 and independent source scan 1 | Retained for source-to-fragment conservation and support/filter semantics. BAM decoding is repeated, not alignment. `raw-owner-counts.log` also records producer, QC, selection and matrix owner verification exactly once each, and zero additional reconstruction when accepted authority is reused. |
| FASTQ | Input synchronization/provenance and canonical fragment/index validation | Retained byte/record scans; the verifier does not execute Chromap, barcode correction or alignment again. This is a code trace and retained regression scope, not new real-FASTQ acceptance. |

Generic embedding publication checks remain representation/axis/parameter and
artifact checks; PCA additionally checks its projection/eigenpair relationship
without fitting PCA again. Existing accepted-authority reuse continues to
separate source integrity from independent scientific reconstruction.

Retired code includes the DA preparer's raw feature-space/metadata reconstruction
and production aggregation, automatic edgeR oracle dispatch, filename-only
branches, the sparse-only source veto, and the current species allowlist.
All affected manifest/result readers, Registry descriptions, independent test
oracles and source iterators are migrated. A pre-existing two-call scripted
planner fixture in the feature tests was updated to the accepted R5 single-call
contract; no planning production change was made for that fixture.

### Focused acceptance and real product checks

The conda `agent` environment was used. No entire historical suite was run.

- `cross-regression-dev.log`: **336 passed in 137.91 s** across inspection,
  PCA, embedding/graph/Leiden/UMAP, EpiZoo boundaries, evaluation, Registry,
  PCA continuation and authority propagation.
- `admission-final.log`: **222 passed in 4.33 s**, including registration,
  attachment candidates, selected-role enforcement, the exact inspection
  handoff, substitution and unsupported-output rejection.
- `targeted-clean.log`: **43 passed in 15.60 s**, including extensionless
  native label/neighbor inputs and no-R recovery. The subsequent versioned
  acceptance below covers later feature/DA changes.
- `versioned-final.log`: **228 passed in 67.38 s**, with the explicit native
  edgeR gate enabled. This covers dense/CSR/CSC input, native H5AD parsing,
  declarations/coordinates, exact pseudobulk, versioned Registry contracts,
  native edgeR and its independent oracle, corruption, design incompatibility,
  cancellation, recovery and figureless application/report paths.
- `raw-owner-counts.log`: two instrumented tiny BAM/external-fragment chains
  pass; the log distinguishes source hashes, bounded inspection, projection,
  independent verification and accepted reuse.

These runs overlap and are not presented as a deduplicated test total.
Development failures are retained in the evidence directory: obsolete sparse
and historical-source expectations, an outdated planner fixture, test harness
argument/import mistakes, and the genuine live inspection-handoff defect were
resolved. There are no remaining failures in the final focused runs.

The live Firefox/OpenRouter flow uploads an unchanged real PBMC file under
`real-pbmc.data`: **26,869,587 bytes**, SHA-256
`c75bca64515710f6045a1c65d9f0dfb9fe9827102501856007b2bfcb58b33c81`.
The stored generic attachment has no `.h5ad` extension. The first attempt
fails before native science at the guard described above. After the fix, a
fresh Session `2f38d6ed-86fa-4439-af70-217cbecfc241` completes inspection,
10-component PCA, 15-neighbor graph and Leiden resolution 0.5 on all
**1,004 cells × 82,579 features**. Native counters show PCA 1, neighbors 1,
Leiden 1. The plan explicitly uses inspection's dataset output for PCA.
`real-browser-fixed.log`, provider/science audits and screenshots record the
actual product path, not scripted provider substitution.

Browser reports, scientific downloads, refresh and a fresh server reopen pass.
All 38 private-workspace files are unchanged by those reads, with no additional
provider or scientific calls (`browser-downloads.json`, `browser-restart.json`).
The real PBMC source is not assigned invented replicates: pseudobulk/DA and dense
aggregation qualification use explicitly simulated counts, separately from the
real native PCA acceptance.

Historical schema-1 feature manifests generated with the archived baseline
verify and feed a new pseudobulk without migration (`legacy-feature-read.json`).
Baseline schema-1 DA verifies without R (`legacy-da-read.json`). The preserved
R5 Session reopens with exact responses/revisions/downloads, all 228 normal
workspace files unchanged, and provider/runtime/owner verification forbidden
(`read-only-result.json`). The full protected comparison reports **261,016 files,
253,308 SHA-256 checks, zero changed/missing**, including all **1,274** historical
eval files (`protection-final.json`).

### Remaining limits and final disposition

No demonstrated unnecessary gate remains on the changed and exercised paths.
That is not universal native-format or tool qualification. Memory/storage limits,
exact axes and source permissions remain. EpiZoo and label transfer retain model
compatibility. Raw producer/reference profiles and annotation biological scope
remain bounded. Feature-space declarations do not prove genome compatibility,
experimental sample identity, biological purity or measurement history.

DA still consumes a pseudobulk publication carrying explicit count semantics
and experimental-unit metadata; arbitrary external aggregate H5AD import with
user-selected column mappings is not a new capability here. Removing cell-level
reconstruction does not certify such absent declarations. Native DA completion
does not independently refit every ordinary result; numerical method
qualification is provided by the retained independent oracle tests. Raw/QC/
matrix reconstruction remains a real first-publication cost for the stated
correctness reasons. Broad resource reclamation, unsupported species workflows,
new native backends and model retraining are outside this R6 follow-up.

**Disposition: R6 cross-tool review and justified simplifications have passed
final engineering review within the recorded scope.** No datasets, checkpoints,
credentials or acceptance artifacts enter Git. R7 has not begun.

### Final engineering review and Git closeout

The final review covers all **42 R6 files** against R5
`266aa828cb46764b6671588c7c8796b8f756e262`: 38 tracked modifications and four
new files, including the cross-tool follow-up. No substantive scientific,
execution-safety, integrity, architecture, historical-continuity or handoff
blocker was found. Agent retains identity, permission, binding and lifecycle
ownership; scientific owners retain method requirements and justified independent
checks. All 24 registered tools were included in the boundary audit, without
claiming universal format or biological qualification.

All **546 source/test Python files** matched the final acceptance snapshot at
review entry. Closeout updates documentation status and removes one extra blank
line at the end of `annotation_groups.py` caught by the staged whitespace check;
its Python syntax tree is unchanged. The other 545 Python files remain
byte-identical. No additional scientific execution or test rerun is needed.
The overlapping regression scopes above remain separate, and the documented
development failures have corresponding fixes and passing coverage.

The final review's protected-state comparison checks **261,016 entries**, with
**253,308 SHA-256 checks**, zero missing or changed entries, and all **1,274**
historical eval files unchanged with no additions or removals. The remaining
7,708 entries are checked by size, modification time and mode, not fully rehashed.

The user explicitly authorized committing and pushing the reviewed R6 changes
after this successful review. The final review evidence is outside Git at
`/home/likeyi/agent-ua365-r6-cross-sy9ei_vl/final-review`.
Its `git-closeout.json` records the resulting commit, exact file inventory,
independently checked remote reference, final index/worktree state and
post-push preservation check. Earlier no-commit statements describe their
historical acceptance checkpoints.
