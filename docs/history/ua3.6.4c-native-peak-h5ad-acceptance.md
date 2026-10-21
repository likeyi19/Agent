# UA3.6.4c — Native MAESTRO direct peak-H5AD implementation and acceptance

Assessment: **2026-10-14, Asia/Shanghai**. Approved Option B is implemented.
**Bounded software and genuine-data acceptance pass. Overall UA3.6.4c remains
PARTIAL:** the continuous live-model attempt stops at an unhelpful resource
question; universal uploaded-H5AD support and independent biological validation
are not claimed. No commit or push.

Evidence is outside Git at `/home/likeyi/program/ua364c-peaks-86vhohmg` (**R**).
The preceding [input-contract review](ua3.6.4c-peak-input-contract-review.md)
and [native simplification review](ua3.6.4c-maestro-native-simplification-review.md)
remain unchanged historical decision records. The user subsequently approved
count-weighted RP, direct external qualification and frozen legacy compatibility.

## 1. Baseline, scope and simplification

Starting and final branch is `main`, HEAD
`88cd19f11c0dd76d406746779a1b8092a238bef4`. The index was and remains empty.
The inherited 15 modified tracked files and non-evaluation drafts were inventoried
and copied into `R/starting-draft` before edits. No reset, clean, stage, commit,
push or production configuration change was performed. Registry remains 23.

The new route is:

```text
uploaded peak H5AD + explicit file roles / declarations / reviewed resources
  → existing Planner, compiler and preflight
  → annotation owner qualifies exact source, counts, coordinates and groups
  → pinned native weighted Enhanced RP → existing sparse normalization / native markers
  → existing conservative signature scoring and exact cell propagation
  → one owner replay → ordinary publication / authority / Run / Revision
  → accepted evidence, report and safe annotated H5AD export
```

The unfinished external-cCRE adoption loader/profile and mandatory adoption
Run dependency are removed. `matrix_adoption_registry.py` and
`verification_authority.py` are byte-identical to HEAD after removing this draft's
additions. The accepted adoption tools themselves are unchanged. The unaccepted
external runner is replaced with direct qualification and a count-preserving RP
adapter; it reuses historical signature, grouping and scoring helpers and the
pinned R adapter. There is no local RP or Wilcoxon reimplementation, peak-to-cCRE
projection, extra annotation registry, new Planner or workflow engine.

This simplifies the execution dependency graph, not every line count: the new
167-line direct adapter and 90-line bounded H5AD layout helper also handle ordinary
metadata/storage that the abandoned adoption layout could not preserve. The
layout walk is shared by source qualification and safe export. Scientific
qualification remains in the owner; upload registration is not scientific proof.

Useful group TSV uploads, atomic path/digest file roles and existing pending
continuation are retained. Count slot and obs group-column literals use existing
shared exact-string declaration admission. Method identity is internal reviewed
configuration, not a user-facing profile selector. The browser's explicit input
list now retains a single fixed source while a reviewed resource menu is selected;
its button/description admit grouping files. No layout redesign is introduced.

## 2. Supported direct-source contract

The new method is `maestro-enhanced-count-weighted-peaks.v1`, paired with
`scatac-annotation-peak-resources.v1` and owner input qualification
`external-peak-annotation-input.v1`. It is one active new-input route within the
existing public `annotate_scATAC_cell_types`; legacy persisted arguments retain
their original branch. Dispatch checks explicit validated method identity.

| Boundary | Qualified behavior |
| --- | --- |
| Measurement | Explicit `cellranger-atac-cut-site-counts.v1`, human/hg38. Upstream counting history is declared, not reconstructed from H5AD numbers. Conflicting recognized metadata is rejected. |
| Counts | Explicit `X` or `layers/name`; canonical sparse CSR/CSC, finite nonnegative integral values, each at most `2^53-1`, preserving exact float64 working values. No normalization, imputation or binary-to-count conversion. |
| Axes/coordinates | Unique exact cell/feature identities; column-aligned `chrom:start-end` names, declared zero-based half-open intervals, valid endpoints within qualified primary nuclear contigs. No alias guessing, reordering, merging or fixed feature vocabulary. |
| Scope/resources | Existing qualified human/hg38 QC-reference contig classification and catalog establish chromosome scope; MAESTRO refgenes and matching human immune signatures supply the science. No QC computation is run. |
| Groups | Explicit selected obs string/integer column or exact ordered `cell_id` / `group` TSV, every cell once, at least two groups, explicit derivation provenance. Numeric labels are opaque groups. |
| Integrity | Immutable registered source digest, source logical sparse digest, ordered axes, groups, declarations, resources and executed scientific code bind the annotation authority. There is no fabricated canonical matrix authority or upstream matrix Run. |
| Storage | Source at most 1 GiB; bounded ordinary AnnData encodings, 2 GiB decoded values, 4,096 objects, depth eight, strings at most 65,536 bytes; at most 100,000 cells, 2,000,000 features and 100,000,000 stored count entries. |
| Conservation | Preserve source X, layers, obs/var, raw, embeddings, pairwise arrays and admitted uns metadata. Add only annotation-owned columns/provenance. Unsupported encodings and private-locator metadata fail explicitly rather than being silently discarded. |

Owner qualification cannot infer species, assembly, raw-count history, tissue or
independent grouping from a filename or integer values. The reviewed resource
choice binds those explicit declarations and resources internally; the researcher
does not author manifest JSON, hashes or Run IDs. The isolated acceptance menu
contains this reviewed PBMC context, not an automatic specification builder for
all tissues and resources.

Raw fragment-overlap counts remain separately unqualified and are rejected by
this new measurement profile, as are declared binary/normalized/TF-IDF inputs.
Exact cCRE coordinates are an admissible feature set if the new count contract is
otherwise met; they are not required. Compatible accepted clustering outputs
still require explicit source/cell correspondence: no new automatic matching or
clustering is implemented. Coordinate fields separate from var_names, arbitrary
AnnData encodings and general conversion of uploaded representations are outside
this acceptance.

## 3. Scientific method and numerical checks

The pinned MAESTRO revision remains
`74f10babf4125b98ef005df4186b14ab02e13bfd`. The adapter executes the original
`calculate_RP_score`, `ExtractGeneInfo` and `RP_AddExonRemovePromoter` function
bodies from the hash-checked source. It supplies a private sparse float64
count-preserving transpose, Enhanced RP and decay 10,000. It does not binarize.
No cCRE reference or historical Run identity enters native RP.

The unchanged R adapter applies sparse `log1p(RP/cell_RP_sum*10000)` and native
`FindAllMarkersPresto` / Presto. Recorded filters remain raw p < .01,
absolute logFC > .25, rounded detection fraction > .1 in either group, then
adjusted p < 1e-5. Existing signed signature summation, duplicate-entry weighting,
coverage reporting, unique positive maximum, unresolved zero maximum and exact
positive-tie ambiguity are retained. Thresholds, candidate label merging and
optional validators are not retuned or added.

`numerical-check/result.json` compares the new adapter against the exact original
native function on a four-cell/two-region synthetic example. Outputs agree;
corresponding feature/column permutation conserves RP. `[9,1]` remains `[9,1]`
under weighted RP and becomes `[1,1]` under frozen historical binary RP. The
normalized outputs differ too. `zero-rp.json` records rejection of a zero-RP
cell. These are mechanism checks, not biological qualification or full native
Seurat-wrapper equivalence. Existing signed-marker/coverage/null/tie tests pass.

Runtime remains R 4.6.1, Matrix 1.7.6, dplyr 1.2.1 and Presto 1.1.0. No scientific
package installation or historical dependency replacement was necessary. No
Seurat, Signac, implicit clustering or dense complete-matrix conversion is added.

## 4. Historical M13 preservation and compatibility

Both historically bound files retain their exact paths and bytes:

- `marker_annotation.py`: `e69739983c6acc99177b66c63a84ab641031e87250afdbe5cf1cb59517135348`.
- `r/maestro_markers_v1.R`: `2b62353381ed385d7f6198e168dc3a9f68b610e1e0ffc2f56a3c0d40817f5ba4`.

The old binary profile, trusted source-store binding, resources, original data,
accepted authority, Session/Revisions and replay dependencies remain intact.
`historical-recovery.json` records a fresh-process terminal resume of accepted
Session `f2751a85-cd30-49a5-ad2b-8e204d64fb0f`, Revision
`35df30cc96723af0c05621a12278d756f66d75f5e34b8c2f0a87448fd5908918`.
Accepted proof import and terminal recovery succeed under guards against owner
replay, backend and provider construction. All 30 workspace files remain exact.
This is actual terminal recovery, not merely opening JSON, and runs zero science.

## 5. Genuinely external data and acquisition

The new data are the official [10x 500 PBMC Next GEM v1.1 dataset](https://www.10xgenomics.com/datasets/500-peripheral-blood-mononuclear-cells-pbm-cs-from-a-healthy-donor-next-gem-v-1-1-1-1-standard-2-0-0),
published 2021-05-03, Cell Ranger ATAC 2.0.0, human healthy-donor PBMC,
484 detected nuclei, CC BY 4.0. It is an external publisher peak matrix, not a
copy of Agent's 1,119-cell historical canonical matrix. The publisher notes that
these libraries share nuclei preparation with other PBMC loads; this is not an
independent-donor biological replication claim.

Downloaded 2026-10-14 from the publisher's
`cf.10xgenomics.com/samples/cell-atac/2.0.0/atac_pbmc_500_nextgem/` release:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `atac_pbmc_500_nextgem_filtered_peak_bc_matrix.h5` | 6,738,636 | `1bb08304e46c3aef81e4883c64a8ef37b61385732f2b8711112dc956e8367456` |
| `atac_pbmc_500_nextgem_analysis.tar.gz` | 47,232,884 | `4328527e2b5e7016ab3eaa0197880578c4e7c4b3a6f678b5e8c445a75b9d9aa3` |
| `atac_pbmc_500_nextgem_web_summary.html` | 2,591,125 | `f0ad5d87e40f55df9f62039bd677bccbf26e477c7d62e600c850dbe46cf9e47c` |

Exact URLs, dates, sizes and hashes are in `acquisition500.json`; no separate
publisher checksum was available in the inspected records. GRCh38 is explicit
in publisher feature metadata; the prepared declaration uses its hg38 assembly
identity. No chromosome aliases are rewritten. Cell Ranger's documented peak
matrix counts [fragment ends / cut sites](https://www.10xgenomics.com/support/software/cell-ranger-atac/2.1/algorithms-overview/algorithms-overview),
which are not interchangeable with fragment-overlap counts.

The first acquired 1.2.0 PBMC candidate was found to be hg19 and was excluded
before execution. Its acquisition record and files remain in R; it was never
relabeled as hg38. No protected scientific resource was overwritten.

`prepare_data.py` performs an explicit, checked format preparation. The complete
publisher matrix has 484 cells × 65,908 peaks. **X deliberately selects 65,887
primary nuclear peaks** using the qualified contig classification. The 21 excluded
scaffold features contain 3,171 counts. The complete original 65,908-feature
matrix, including those values, is retained exactly in `raw`, with its feature
metadata. This is disclosed feature selection, not a lossless claim about the X
feature set and not peak projection. The production owner never silently filters
an uploaded axis. Every original cell, count and feature survives in the prepared
file's complete raw; every selected X sparse value is compared exactly.

X contains 4,092,063 nonzeros, 11,660,406 total counts and maximum count 404.
`analysis/clustering/graphclust/clusters.csv` supplies publisher-derived groups
with exact barcode order, before MAESTRO signature inspection. No labels are
invented as biological ground truth and no grouping backend is rerun.

Prepared H5AD: **91,072,184 bytes**, SHA-256
`72e581c3f027cc9330098f655335b6cc9406d39c136c882d4e2e95768c2efe38`.
Group TSV: `583bb187c92e3646f6f3b566ab7afa8c1747c7fc6ec3a24c6507ffe96db50dec`.
Reviewed resource specification:
`9c34ca0c5d475ed0a02c608cd1f3b21283f1e332702c435b4894f59d89388b88`.
Existing MAESTRO refgenes/signatures and qualified hg38 contig scope are reused
with exact pins in `readiness-final.json`. No new reference/checkpoint is downloaded.

Before science, source qualification, all resource/code pins, free disk and an
estimate of 300 seconds, 4 GiB memory and 3 GiB new storage were recorded. The
actual instrumented main-process high-water observation at science entry is
1,415,968 KiB; it is not a complete child-process peak measurement. No GPU is used.

## 6. Native browser execution and accepted result

Native Firefox, controlled through Marionette, uploads the actual H5AD using the
file control. The researcher fixes it in the existing selected-input list using
**Add another input**, chooses the reviewed PBMC menu and asks:

> Please annotate the cell types in my uploaded human hg38 PBMC raw Cell Ranger ATAC cut-site peak-count H5AD using MAESTRO and the reviewed human immune resources. Use X. Ask me for the independent grouping file if needed, and explain unresolved groups.

The scripted Interpreter requests the missing `groups` file role. Firefox uploads
the ordered TSV and replies that the second file supplies publisher-derived
independent groups. Shared role continuation retains the original objective,
exact captured source, resources and Session generation. The resulting Run's
request equals the original request, not the supplementation reply. The researcher
does not enter internal specification paths or identifiers.

Normal planning, compilation, preflight, real owner execution, publication,
verification, evidence/report and Session completion succeed in **66.14 seconds**,
including presentation. These language decisions are **scripted**; the Python/R
science is genuine. Instrumentation records exactly **two native backend calls**,
one production and one within **one annotation-owner verifier entry**. No upstream
producer/verifier, historical binary backend, EpiZoo, CellMarker or EpiAgent runs.

- Session: `7d2d00b9-2f0f-49d0-87e8-c0a2bdf3ca0a`.
- Run: `turn-3820712072aade64d9e87c8c970cd6f03eb36ad0d583480ce4898ad0fef50566:run`.
- Revision: `4756408cf6052e2fecfe11875c16b8b5ed8315e38a2a4fe93e5bf3d7ef6da182`, generation 1.
- Authority: `fe61dc76787bc2775fb99b9766cc12b0b3937f5f773833fbe220f622c5e89b7a`.
- Manifest: `7b23736d40594d2833ed86abb39db48986d72f584211eca2968fffff48461878`.

The new authority has 242 file bindings and no upstream matrix authority.
The existing result envelope's `source_authority_sha256` field holds the explicit
owner input-qualification digest for this method; `source_manifest_sha256` holds
the external source H5AD digest. The new provenance identifies this source as
external and its counting history as declared, not an internal canonical Run.

RP has 28,307 genes and 6,366,239 nonzeros. Native marker statistics produce
113,228 rows; 2,320 survive the recorded signature-input filtering.

| Opaque group | Cells | Primary label | Status |
| --- | ---: | --- | --- |
| 1 | 207 | Monocytes | assigned |
| 2 | 114 | ActNK | assigned |
| 4 | 57 | MemoryBcells | assigned |
| 3 | 106 | NaiveCD4Tcells | assigned |

All 484 cells are assigned; unassigned and ambiguous counts are both zero.
These labels are method-derived candidates, not experimentally established
identities. All retain `validation_state=not_assessed`. Publisher clusters are
not ground-truth cell types; no subtype accuracy, purity, held-out accuracy,
independent reference corroboration or validation of every label is established.
Zero/unresolved and positive-tie policies are covered by synthetic tests; this
real dataset does not demonstrate a genuine unresolved group.

## 7. Conservation, evidence, downloads and recovery

The newly annotated original is **91,085,280 bytes**, SHA-256
`704ff53739959bfd3f7185daa8994fdc82b11bf12a3ced0959132634bcf2ea72`.
The safe export is **91,020,412 bytes**, SHA-256
`4380a81810f2eb8ddd1085e40b36ef610c816eddb993308150401ab229b933de`,
retained at `R/annotated-export.h5ad`. It omits only the annotation owner's private
`source_path` and adds a closed export provenance record; it does not assert
unchanged original file bytes or issue new scientific authority.

Firefox fetches and SHA-256 hashes the actual HTTP 200 download. A separate
bounded HTTP/file audit obtains the same bytes, compares X, full raw, complete
obs/var, annotations/nulls, remaining uns and every admitted ancillary slot, and
checks the original uploaded source digest. `scientific-acceptance.json` records
exact arguments, sidecars, result, authority and report/evidence locations.
Download plus file comparison and terminal audit take about 9.36 seconds; that
measurement includes adjacent verification rather than only network transfer.

Native evidence displays the assigned/unassigned counts, validation state and
annotation contract. The detailed rationale shows a chosen group's scores,
support and bounded omissions. The rendered safe report contains 3,070 characters.
An initial harness request used an invalid detail section (`groups`); it was
corrected to the existing `annotation_rationale` control with explicit group `3`.
The corrected read succeeds without scientific rerun.

A fresh-process result audit and terminal resume run under backend/provider guards.
An initial fresh-server Firefox reopening preserves 38 workspace files and the
four-message result conversation, Revision and generation, with zero scientific
entries or provider construction. A subsequent **scripted** result question
returns accepted evidence for 484 assigned, zero unassigned and `not_assessed`,
without science or a new Revision. A second fresh-server reopening preserves all
six messages and **46 workspace files**, again with zero science/providers.
These are terminal lifecycle checks, not crash-during-science recovery claims.

## 8. Continuous live-model attempt and spending

The user-specified existing `agent` Conda environment is activated. Credential
presence is checked only as a boolean; its value is neither printed nor copied
into configuration/evidence files. Groq is not called. The isolated OpenRouter
profile uses `openai/gpt-5.6-sol`; the production operator JSON is unchanged.

The copied acceptance-only ledger passes 11 guard tests. Its fixed limits are
12 total completions, six per turn, $3.00 conservative pre-dispatch reservations,
4,096 output tokens, zero SDK retries and no silent fallback. Reservations are
persisted before dispatch and are not refunded. This is the same established
accounting boundary, not new product budget infrastructure.

One continuous live attempt begins in Session
`b4eb6c57-43d1-4393-830b-62c65ae025c2`, through native Firefox upload and the same
reviewed menu and original scientific request. The actual Interpreter chooses
`request_resources`, target `annotate_scATAC_cell_types`, binding `source.0` to
`source`. The projected inputs already include the specification, method and
count slot; no additional resource choice is offered. The missing grouping file
was not requested through `request_roles`.

The UI displays its existing empty-resource clarification. Its generic statement
that an operator must provision resources is unhelpful here: the reviewed
resources are already selected. No plan, Run, Revision or science follows.
`provider/request-01.json`, `response-01.json` and `native-live.log` preserve the
exact prompt/schema, raw decision and rendered outcome. No semantic override,
manual plan replacement or favorable-attempt retry is made. This is an observed
live entry/clarification failure, not scientific incompatibility of these peaks.

Final spending: **one completion**, **$0.160868 reserved**, **$0.031691 reported
provider cost**, 9,637 prompt and 760 completion tokens, zero unfinished
reservations. The scripted result conversation is not combined with this failed
live attempt. Complete continuous live-model acceptance remains **BLOCKED in this
attempt**, and no later model improvement is inferred from software tests.

## 9. Focused validation and development failures

No full repository suite is run. The affected pre-closeout regression passes
125 tests in 179.09 seconds. Direct-entry tests initially pass 24 checks;
subsequent declaration/HTTP-column and cancellation coverage passes a final
81-test targeted batch in 46.98 seconds, with one intentional duplicate-cell
fixture warning. The relevant static batch passes 24 checks. The final combined
affected regression passes **239 tests in 195.29 seconds**, with that one expected
warning. All **503 frozen source/test/static files** remain unchanged through
this final verification; `git diff --check` passes.

Coverage includes source/digest substitution, incompatible count meaning,
nonfinite/fractional/negative/overflow counts, axes/coordinates, grouping alignment,
assembly/context, exact method identity, selected layers/obs columns, file-role
continuation, resource/session staleness, owner replay and tampering, metadata and
safe export, privacy, source conservation, marker/signature policies and cancellation.
Synthetic owner tests explicitly stub the native annotation backend; they do not
stand in for the real PBMC execution above. The tiny native RP probe executes the
actual pinned upstream functions. No production threshold is adjusted for a test.

Development failures remain in R: a stale v3 Planner fixture omitted nulls for
new optional ports; old static/declaration tests expected the old upload choices
or parameter count; harness configuration placed uploads outside the workspace;
initial audit scripts used the wrong import/export metadata key; and the first
browser detail request used an unsupported section. Fixtures/harnesses were
corrected without weakening owners or replaying successful science. Three stale
schema assertions after adding two exact-string declarations are corrected with
explicit new pairs, not removed. Split test logs are not presented as one early
clean run.

The only production change after the genuine science/code freeze was Registry
metadata exposing `count_slot` and `group_column` through the existing shared
exact-string mechanism. Native scientific files and numerical behavior remained
unchanged. The added HTTP test qualifies that final metadata change; fresh native
reads/recovery use it without reconstructing science. The failed live attempt
preceded that exposure; no new live success is claimed afterward.

## 10. Preservation, deployment and final scope

The bounded preservation audit checks **120,222 protected files**, including
**119,566 SHA-256 checks**, with no unexpected changes. Larger initially unhashed
files use size, mode and nanosecond mtime; this is not a fresh full-byte audit of
every large dataset/checkpoint. All **981** pre-existing Git-listed untracked eval
paths and their baseline content remain exact. Historical acceptance directories,
previous ledgers, sources, scientific resources and external operator configuration
remain unchanged. The two restored draft-only coupling files are separately
checked byte-for-byte against HEAD; their expected mtime changes are disclosed.

The final index remains empty. Seventeen tracked files are modified; new direct
adapter, layout helper, tests and this report remain untracked, alongside preserved
review documents and existing evals. Original accepted UA3.6.4c documentation is
unchanged. `R/baseline.json`, `final-code.json` and `preservation.json` carry exact
inventories. All work remains uncommitted; no next milestone is started.

Deployment is explicit: review `R/operator.json`, `resources.json` and `readiness.py`
before provisioning the reviewed resource menu and existing runtime/catalog
bindings. This task installs no production default and does not acquire new
reference or model packages. Other tissue/signature combinations require their
own reviewed declarations/resources; other measurement classes and species need
separate scientific qualification.

| Scope | Assessment |
| --- | --- |
| Removal of mandatory cCRE/Run dependency for scientifically compatible peaks | **PASS**; it was an inherited integration restriction, not a native RP requirement |
| Direct human/hg38 raw Cell Ranger cut-site H5AD owner and sparse weighted RP | **PASS, bounded**; new distinct method, genuine 10x execution and one owner replay |
| Uploaded TSV continuation, evidence/report, export and terminal Firefox recovery | **PASS, bounded**; scripted language decisions with genuine scientific execution |
| Explicit obs-column/count-layer entry | **PASS, software**; shared exact-string HTTP test with synthetic native backend |
| Continuous live OpenRouter request-to-result chain | **BLOCKED in the observed attempt**; empty-resource clarification, zero science |
| Raw fragment-overlap, arbitrary H5AD encodings, automatic grouping/correspondence | **Not qualified by this work** |
| Full native Seurat workflow equivalence / independent biological validation | **Not claimed** |
| Overall UA3.6.4c | **PARTIAL**, unchanged |

The remaining live issue is appropriate selection/handling of file versus resource
prerequisites when reviewed resources are already present, including clearer empty
resource guidance. It must not be solved by overriding provider intent, injecting
scientific facts or inventing authority. General uploaded-H5AD acceptance is not
complete. Stop for review; no commit, push, UA3.6.4d or UA3.6.5 work.
