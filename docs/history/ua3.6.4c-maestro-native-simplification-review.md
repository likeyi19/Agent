# UA3.6.4c — MAESTRO native integration simplification review

Assessment: **2026-10-14, Asia/Shanghai**. **Audit and design only.** Overall
UA3.6.4c remains **PARTIAL**; ordinary uploaded peak-H5AD acceptance is unqualified.

**Recommend Option B: a minimal existing-owner integration.** Keep the existing
native RP and marker calls, qualify direct external peak input and count-weighted
RP for future analyses, and retain the small normalization/scoring adapter where
the pinned upstream lacks a suitable standalone interface. Remove the unfinished
cCRE-adoption detour. Do not introduce two active annotation products or install
the full Seurat/Signac workflow merely to replace a few arithmetic expressions.
This recommendation requires the approvals in section 10; it is not implemented.

Starting branch `main`, HEAD `88cd19f11c0dd76d406746779a1b8092a238bef4`, empty index:
15 modified tracked files and three non-evaluation untracked draft files. All are
preserved. Evidence for this audit is outside Git at
`/home/likeyi/program/ua364c-native-review-zsvbjs80` (**R** below).

## 1. Current architecture and responsibilities

The committed M13 route, with the unfinished draft branches distinguished below:

```text
upload / reviewed inputs → typed prerequisites → Planner / compiler / preflight
  → scatac_annotation.execute_annotation → _build
    → source → bind_annotation_matrix (trusted accepted canonical matrix)
    → marker_annotation.annotate_cell_groups
      → load_matrix + exact_groups + checked gene/signature resources
      → enhanced_rp: private binary float64 copy; reversible coordinates
        → pinned calculate_RP_score / ExtractGeneInfo / RP_AddExonRemovePromoter
      → sparse RP NPZ + temporary MatrixMarket + exact axes/groups
      → r/maestro_markers_v1.R
        → sparse LogNormalize → native FindAllMarkersPresto → presto::wilcoxauc
        → native marker filters + explicit adjusted-p filter
      → score_candidates → group/cell assignments and evidence
    → unchanged-source annotated H5AD + manifest
    → verify_annotation: one backend replay + exact artifact/propagation checks
    → shared atomic publication / receipt → accepted authority → Run / Revision
  → accepted evidence / report / annotation_delivery safe H5AD derivative
```

**Agent responsibilities:** opaque input identity, typed declarations and roles,
durable questions, compilation, cancellation, publication, authority, lineage,
Session state and bounded private-path-safe delivery. Keep the shared machinery.

**Owner responsibilities:** matrix/resource eligibility, exact axes and groups,
lossless adaptation, pinned execution, output checks and source conservation.
These checks are not algorithms to delegate to an LLM or eliminate as duplication.

**Scientific computation:** RP geometry and transcript selection already execute
original upstream Python function bodies, loaded from the hash-checked source;
marker statistics and filtering already execute original upstream R plus Presto.
Local science consists principally of binary preprocessing, sparse normalization,
signature summation and primary-assignment policy. M13 uses Enhanced RP, decay
10,000, `log1p(RP/cell_sum*10000)`, raw p < .01, |logFC| > .25, rounded detection
fraction > .1 in either group, then adjusted p < 1e-5. These are recorded choices,
not interchangeable undocumented defaults.

The draft adds external **exact-cCRE adoption**, then calls the same binary RP
and scoring through `external_marker_annotation.py`; it is not ordinary peak
support and duplicates much of the existing execution envelope.

## 2. Native MAESTRO interfaces and deployment

All source inspection uses revision
`74f10babf4125b98ef005df4186b14ab02e13bfd` (DESCRIPTION version 1.5.1).
Eight source files, **41,737 bytes** total, were retrieved only into R for review;
URLs, sizes, hashes, publisher and retrieval date are in `upstream-sources.json`.
No package, dataset or reference was installed or replaced.

| Interface | Actual input, computation and output | Integration consequence |
| --- | --- | --- |
| Python `genescore` CLI / `calculate_RP_score` | Region-by-cell counts, coordinates and transcript/exon resource; native Enhanced RP and transcript-derived gene selection; gene-score output through its writer. | Already used below the CLI by M13. Retain this qualified source/function boundary; capture sparse output without the CLI's unrelated file conversions. |
| R `ATACCalculateGenescore` | Unnormalized peak-by-cell matrix; calls a **different** Python file, `inst/ATACCalculateGenescore.py`, through reticulate; chooses bundled human/mouse gene resources; returns sparse gene-by-cell scores. | Not an installed alternative here. Its non-`dgCMatrix` conversion passes through `as.matrix`; input and bundled-resource equivalence must be qualified before substitution. [Pinned R source](https://raw.githubusercontent.com/liulab-dfci/MAESTRO/74f10babf4125b98ef005df4186b14ab02e13bfd/R/ATACCalculateGenescore.R). |
| R `ATACRunSeurat` | Peak counts → Seurat filtering → LSI/TF-IDF/SVD or PCA → UMAP, neighbors, clusters and differential peaks; returns object plus peak markers and writes plots/tables. | Substantial extra science. Defaults remove features/cells; `orign.ident` is documented but unused in this body. Not needed when groups already exist. [Pinned source](https://raw.githubusercontent.com/liulab-dfci/MAESTRO/74f10babf4125b98ef005df4186b14ab02e13bfd/R/ATACRunSeurat.R). |
| R `ATACAttachGenescore` | Reindexes RP to object cells, creates ACTIVITY assay, finds variable genes, normalizes and scales. | M13 needs normalized assay data, not scaling/variable-feature selection. Avoid implicit cell subsetting and extra dense scaled intermediates. [Pinned source](https://raw.githubusercontent.com/liulab-dfci/MAESTRO/74f10babf4125b98ef005df4186b14ab02e13bfd/R/ATACAttachGenescore.R). |
| R `ATACAnnotateCelltype` | Clustered Seurat ACTIVITY assay; `FindAllMarkersMAESTRO`, adjusted-p filtering, then `RNAAnnotateCelltype`; returns annotated object. | Marker path is already delegated. `orig.ident` follows a supplied-identity branch, not a way to request fresh annotation of arbitrary groups. [Pinned source](https://raw.githubusercontent.com/liulab-dfci/MAESTRO/74f10babf4125b98ef005df4186b14ab02e13bfd/R/ATACAnnotateCelltype.R). |
| R `RNAAnnotateCelltype` | Embeds signed marker/signature summation, factor/integer cluster handling, maximum selection, optional named-CIBERSORT label merging and plotting. | No standalone score-return API. Default `min.score=0` selects the first maximum even at zero or a tie. Passing a signature data frame avoids named-resource label merging, but not the other coupling. [Pinned source](https://raw.githubusercontent.com/liulab-dfci/MAESTRO/74f10babf4125b98ef005df4186b14ab02e13bfd/R/RNAAnnotateCelltype.R). |

The two Python RP files are not byte-identical or signature-compatible. AST
comparison shows data-frame versus file input, strings versus bytes, return value
versus writer, and other differences; `RP_AddExonRemovePromoter` differs only in
logging in that comparison. This is source evidence, not full wrapper equivalence.
See `rp-source-comparison.json` and the
[R wrapper's Python helper](https://raw.githubusercontent.com/liulab-dfci/MAESTRO/74f10babf4125b98ef005df4186b14ab02e13bfd/inst/ATACCalculateGenescore.py).

The configured R executable/library has R **4.6.1**, Matrix **1.7-6**, dplyr
**1.2.1**, Presto **1.1.0**. MAESTRO, Seurat, SeuratObject, Signac, reticulate,
plyr, ggplot2 and Gmisc are absent from its library search path. Retained MAESTRO
sources are a source slice, not an installed R package. Full-package imports also
include substantial Bioconductor dependencies; installation is not proof of
compatibility with this old pinned wrapper. See `runtime-inventory.txt` and
[DESCRIPTION](https://raw.githubusercontent.com/liulab-dfci/MAESTRO/74f10babf4125b98ef005df4186b14ab02e13bfd/DESCRIPTION).

One small new probe evaluates the **original R scoring/selection expressions**
on two synthetic groups: unmatched signatures produce zero scores and label A;
an exact positive tie, including duplicate signature weighting, also selects A.
It stops before object assignment and plotting. `native_annotation_probe.R` and
its log establish this policy difference, not complete native annotation acceptance.
The earlier count-versus-binary RP experiment was reused, not repeated.

## 3. Exact redundancy and justified local processing

| Current code | Disposition after approval |
| --- | --- |
| `marker_annotation.enhanced_rp` | Keep native RP delegation. Omit binary conversion in the **new** count-weighted entry, without editing the historical pinned function. No local RP rewrite or peak projection. |
| `r/maestro_markers_v1.R` normalization | Retain the small sparse LogNormalize adapter. Upstream exposes it through Seurat; importing that stack plus scaling is a larger maintenance/memory cost. Qualify mathematical behavior against its stated definition. |
| Same R file: marker invocation | Retain native `FindAllMarkersPresto`, explicit thresholds and Presto. No duplicated Wilcoxon implementation exists to remove. |
| `marker_annotation.score_candidates` | Signature summation repeats the upstream formula, but the surrounding identity/coverage/null/tie evidence is necessary. Reuse this bounded helper; do not extract arbitrary expression fragments into a new production engine merely to eliminate one summation. Keep its divergence from native label selection explicit. |
| `external_marker_annotation.annotate_cell_groups` | Replace this unaccepted copied envelope with one direct-input adapter using existing helpers and owner publication. Remove its adoption-specific loader/profile. |
| `scatac_annotation.verify_annotation` | Keep one independently loaded backend replay and exact derivative checks. Shared proof reuse already suppresses repeated owner replay during publication/recovery; no redundant second verifier was demonstrated. |
| `annotation_delivery` | Keep accepted-authority and safe-export checks. Generalize supported metadata/storage carefully; the draft's stripped adoption layout cannot represent ordinary H5AD faithfully. |

The retained local policy assigns only a unique positive maximum, preserves nulls
and exact positive ties, and reports coverage. It is scientific output policy,
not a formatting detail. Calling native default assignment would change it.
Retaining a few bounded equations is justified here; claiming that every local
numerical line can disappear would misrepresent the available interfaces.

## 4. Alternatives and recommended target

| Alternative | Total cost and semantics | Decision |
| --- | --- | --- |
| A: full native workflow | New R/Python dependency stack, Seurat objects, filtering/clustering/UMAP/scaling and plotting; different grouping/label defaults; largest runtime/memory and qualification scope. Legacy replay still needed. | Reject for this bounded entry. |
| B: minimal existing owner | Existing sparse Python/R dependencies, native RP/markers, small explicit normalization/scoring policy; one new direct-input boundary; existing publication/replay/export. Removes adoption and binary preprocessing from new work. | **Recommend.** Least new science and integration machinery. |
| C: new native-function composition | Same native RP/marker building blocks are already composed. Delegating the remaining score requires Seurat or slicing an internal R expression block and intercepting results; the probe does not turn that block into a supported API. | No demonstrated simplification over B at this revision. |

Use one public `annotate_scATAC_cell_types` and one qualified count-weighted method
for new peak analyses. Explicit accepted inputs/resources determine eligibility;
the LLM interprets declarations and offered choices. No profile selector, new
Planner, annotation registry, automatic pipeline completion or per-tool language
parser. Exact cCRE coordinates remain a legitimate feature set, not a prerequisite.

Retain sparse matrix transpose, exact float64 working values, temporary sparse
MatrixMarket transfer, native marker execution and bounded sidecars. Do not route
through the CLI's plain-text integer conversion or add H5AD→10x→Seurat→H5AD
round-trips. RP weight construction and the existing replay still cost memory and
approximately two backend executions; there is no measured speedup claim.

## 5. Minimum ordinary peak-H5AD entry and grouping

The annotation owner should directly qualify the immutable upload; it should not
manufacture canonical matrix authority or require an adoption Run. Existing upload
registration establishes bytes and attribution, not biological eligibility.

| Requirement | What can be verified, and what remains declared |
| --- | --- |
| Matrix | Explicit `X` or named count layer; valid bounded sparse storage, unique ordered cells/features, finite nonnegative integral raw values. Private float64 adaptation must round-trip counts exactly; reject overflow and unsupported layouts. Never infer counts from integer-looking values alone. |
| Measurement | Preserve distinct raw cut-site and fragment-overlap declarations and source processing evidence. Both are candidates for the same native weighted operation, with separate acceptance coverage. Binary accessibility, TF-IDF, normalized and imputed values are not silently admitted as raw counts. |
| Coordinates | Exact column-aligned genomic intervals from an explicit field mapping or unambiguous coordinate IDs; declared coordinate convention, valid endpoints and qualified primary-nuclear contigs. No alias guessing, silent reordering, peak merging or mandatory full-reference axis. |
| Species/resources | Declared assembly/species matched to qualified gene annotations, namespace and tissue-appropriate signatures. File names/chromosome spelling cannot establish these facts. Start within the existing qualified human/hg38 scope; other resources need their own qualification. |
| Groups | Explicitly selected H5AD `obs` field, exact ordered TSV, or accepted clustering output with demonstrated cell/source correspondence; at least two groups. Record derivation and whether independent of marker/signature inspection. Opaque labels are not biological identities. |
| Result | Finite RP, positive per-cell RP totals, exact gene/cell order, marker and signature evidence, preserved source matrix/metadata, explicit source declarations and verification limits. Backend replay is computational verification, not cell-identity validation. |

Use typed declaration/role/resource questions to retain the original objective,
source digest and Session generation while missing context or a file arrives.
Bind reviewed resource choices internally; users supply scientific facts and
choose files/columns, not specification JSON, hashes or Run IDs. Conflicting or
ambiguous declarations remain unresolved rather than defaulted.

Prefer supplied groups for this bounded implementation. `ATACRunSeurat` is a real
clustering option but adds a new qualified workflow and filtering decisions; do
not run it implicitly. Existing EpiZoo clustering can be reused only when its
own input contract and exact correspondence qualify. Its cCRE requirements must
not migrate into the annotation input contract. General correspondence to a
separately uploaded clustering result is not already established by matching rows.

## 6. Historical integrity and minimal compatibility

The accepted UA3.6.4c authority has **264 file bindings**, including absolute paths
and hashes for `marker_annotation.py` and `r/maestro_markers_v1.R`:

- Python: `e69739983c6acc99177b66c63a84ab641031e87250afdbe5cf1cb59517135348`.
- R adapter: `2b62353381ed385d7f6198e168dc3a9f68b610e1e0ffc2f56a3c0d40817f5ba4`.

Fresh hashes match the actual accepted record (`accepted-code-closure.json`).
`scientific_authority.describe` also binds original resources, runtime executable,
native source files and package files, plus the upstream matrix dependency.
`_load_context` reconstructs and compares accepted authority and imports proofs;
standalone verification without those proofs performs backend replay. Merely
retaining documentation or moving old code would not preserve these contracts.

Keep those two files byte-for-byte at their recorded paths, their pinned resources,
runtime/libraries, source data, RunStores, Session/Revisions and historical method
identity. Retain the old source-binding and verifier branch for persisted legacy
arguments. New analyses receive a distinct recorded method/contract identity;
dispatch must use validated explicit identities, not the presence of `groups_path`
as the draft currently does. Do not migrate old authorities or relabel old science.

This is one frozen compatibility implementation plus one active new-input route,
not two configurable products. Reuse unchanged small helpers/R marker adapter
where valid. Do not monkeypatch the historical backend to inject new semantics.
Historical discovery and interpretation remain through existing revisions and
accepted projections; terminal recovery reuses exact accepted proofs, while
reproduction retains the original executable path. Later deletion requires an
explicit migration contract and is outside this work unit.

`historical-read-check.json` records a fresh read-only import of the real accepted
annotation authority with production/verification-entry guards. No Session resume
or replay is claimed by that check; the previous native reopen/resume acceptance
remains the recorded recovery evidence.

## 7. Primary annotation and optional corroboration

Reuse `marker_validation.corroborate_marker_annotation` for explicitly requested,
qualified CellMarker comparison: it consumes pinned primary markers and a frozen
resource/context/type mapping, returns separate associations, and cannot overwrite
labels or fill nulls. It reuses marker statistics rather than recalculating them.
Its current boundary is a private API, not a planner-visible public tool; exposing
it would need normal owner/authority/evidence integration, not direct provider calls.

M13.4c EpiAgent qualification is preserved external acceptance code, not an existing
registered validator. Its qualified input requires the exact EpiAgent vocabulary,
checkpoint and preprocessing. Ordinary peaks do not automatically qualify; no
peak projection or mandatory EpiAgent step belongs in this entry.

Present primary label/status, supporting or competing associations, vocabulary
limits and validation status separately. Shared markers, sample or prior knowledge
do not constitute independent trials; positive database membership alone does not
establish contradiction. Keep primary `validation_state=not_assessed`; do not call
backend replay, reference agreement or a second model experimental validation.
See [M13.4b](m13.4b-annotation-validation-semantics.md) and
[M13.4c](m13.4c-full-pbmc-validation.md).

## 8. File-by-file disposition of the existing draft

These are **recommendations, not edits made by this audit**. Paths below are under
`src/agent/` unless prefixed `tests/` or `docs/`.

| File | Recommended disposition |
| --- | --- |
| `application/annotation_delivery.py` | Revise: retain accepted/export plumbing; replace adoption-only empty-axis metadata assumptions with bounded direct-source preservation, including selected layers and explicitly supported ancillary slots. |
| `application/dialogue_evidence.py` | Retain intent, revise method recognition to validated contract identity; preserve legacy projections. |
| `application/local_resources.py` | Retain group uploads, atomic path/digest roles and single-file continuation; replace adoption source role with direct annotation source; retain argument/type validation for every bundle member. |
| `application/resource_continuation.py` | Retain exact role/resource continuation; qualify captured-source preservation and subsequent group supplementation. |
| `application/turn_decisions.py` | Retain typed roles on resource requests; test strict schema and backwards compatibility. |
| `application/uploads.py` | Retain bounded group TSV registration and truthful unverified attribution. Scientific validation remains with the owner. |
| `orchestration/annotation_registry.py` | Revise to direct H5AD/count-slot/group/declaration ports and one capability; remove mandatory external-cCRE producer coupling. Preserve persisted legacy arguments. |
| `orchestration/matrix_adoption_registry.py` | Revert this task's adoption-source metadata addition if no longer consumed; leave the accepted adoption tool intact. |
| `orchestration/verification_authority.py` | Discard the draft's mandatory external-adoption branch; preserve legacy trusted-store binding. Direct-source checks belong to annotation qualification. |
| `tools/analysis/annotation_contract.py` | Replace draft external-cCRE profile/specification and argument-presence dispatch with explicit new source/method contract; reuse resource descriptors and exact grouping. |
| `tools/analysis/scatac_annotation.py` | Retain publication, cancellation, replay and conservation; revise direct-source binding and supported H5AD metadata; keep legacy branch exact. |
| `tools/data/scientific_authority.py` | Revise new-entry closure to upload, declarations, groups, resources and executing code; no fictitious upstream matrix authority. Preserve legacy closure. |
| `web/app.py` | Retain bounded group upload and single selected resource list; no science in HTTP. |
| `web/static/app.js` | Retain group attachment/selection presentation after focused regression; no RP profile UI. |
| `web/static/index.html` | Retain group TSV option; no layout change. |
| `tools/analysis/external_marker_annotation.py` (new) | Replace the unaccepted exact-cCRE loader/copied runner with the bounded direct-peak adapter; reuse helpers rather than copying their science. |
| `tests/annotation_integration/test_external_entry.py` (new) | Revise adoption-chain fixtures and role names to direct entry; preserve mutation/continuation tests. Correct invalid planner wire. Do not keep cut-site rejection as a universal rule. |
| `docs/ua3.6.4c-peak-input-contract-review.md` (new) | Preserve as the prior decision record; this review supersedes its proposed active dual-profile direction. |

The earlier draft's failing planner fixtures and HTTP test remain unchanged and
unqualified. No clean draft regression or external-data acceptance is claimed.

## 9. Concrete implementation and acceptance plan after approval

1. Replace only the unaccepted external adapter with direct qualification and a
   count-preserving native RP call. Reuse the pinned marker R adapter and bounded
   scoring/evidence helpers. Keep legacy Python/R files immutable. Remove the
   adoption-only profile/loader/coupling listed above; do not create a backend framework.
2. Extend the existing annotation contract/Registry with direct source identity,
   declared measurement, coordinates/assembly, selected count slot and explicit
   grouping. Complete the existing shared declarations, file roles and resource
   continuation; construct bindings inside Agent. New authority must distinguish
   checked content from declared upstream history. Do not weaken generic schemas.
3. Preserve admitted original H5AD slots and metadata through annotation and safe
   export, retaining the original file unchanged. Unsupported encodings fail with
   an actionable limitation; never silently strip data to fit the adoption layout.
   Verify delivered scientific arrays/axes/annotations and declared safe omissions.
4. Run focused tests for direct peak/count contracts, exact dtype adaptation,
   malformed coordinates/CSR, assembly/resource mismatch, missing or substituted
   groups, zero-RP cells, signed markers, duplicate signatures, zero/tie outcomes,
   tampering, privacy and cancellation. Compare small weighted-RP outputs to the
   original function and scoring to the bounded native expressions; retain the
   existing numerical counterexample as a change-of-method regression.
5. Check legacy accepted-authority import and terminal Session recovery under
   zero-science/provider guards. New production should perform exactly one owner
   replay, with no upstream recomputation or replay on terminal reopen/download.
6. Qualify a **genuinely external ordinary peak-count dataset**, preferably a
   documented Cell Ranger cut-site matrix, with independent groups and pinned
   resource provenance. Freeze source/count semantics, acquisition checksums,
   expected compute and budget first. Qualify fragment-overlap input separately
   before claiming both measurement classes; reuploading the historical cCRE
   matrix is not external acceptance.
7. Native Firefox: upload → natural request → necessary declarations/group
   supplementation → real annotation → evidence/report → downloaded H5AD →
   refresh/restart. Then a separate bounded continuous live-model attempt with
   ledger and no semantic override. Keep synthetic, scripted, genuine-data and
   live-model outcomes separate; report remaining gaps honestly.

No full suite, live provider, genuine-data recomputation, GPU job or native browser
acceptance was run in this audit. The only new numerical work was the tiny R policy
probe. Reused scientific findings are in the
[prior peak review](ua3.6.4c-peak-input-contract-review.md).

## 10. Blocking approval decisions and audit closeout

Approve the following concrete scope **before implementation**:

- **New method:** count-weighted Enhanced RP for future direct peak inputs,
  retaining the current marker thresholds, normalization and explicit null/tie
  assignment policy. This changes binary preprocessing and is not full native
  Seurat/MAESTRO wrapper equivalence. No threshold retuning or label merging.
- **New input/authority contract:** owner-qualified external peaks and truthful
  measurement declarations, without historical canonical authority; cut-site and
  fragment-overlap classes require their stated qualification. Existing uploaded
  files with uncertain processing history must remain uncertain or unsupported.
- **Compatibility:** one active future route with frozen legacy replay/recovery,
  distinct recorded method identity and no modification of historical records.
  Keep optional corroboration outside the entry gate; do not add native clustering
  to this bounded implementation.

These approvals are required by the user's task section 13, which explicitly
withholds authorization for material method, contract or replay replacement.
There is no recommendation to approve a new framework or two permanent UI profiles.

The bounded preservation audit checks **120,226 protected files**, including
**119,570 SHA-256 checks**, with zero unexpected changes; large unhashed files use
size/mode/nanosecond-mtime checks. It is not a fresh full-byte audit of every large
dataset/checkpoint. All **981** pre-existing untracked evaluation paths and their
content remain exact. Current draft bytes are separately snapshotted. Only this
new report is added to the repository by this task; the index stays empty and
HEAD unchanged. No stage, commit, push, UA3.6.4d or UA3.6.5 work.

**Decision:** the exact-cCRE restriction was unnecessarily narrow for native RP;
the full high-level wrapper is not the smallest replacement in this deployment.
Proceed with the recommended bounded owner simplification only after review.
