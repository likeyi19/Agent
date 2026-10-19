# UA3.6.5-R3 — Native-tool-first scientific integration simplification

2026-10-19 (Asia/Shanghai). Scoped implementation validated; common deployment
acceptance remains **PARTIAL** for the reasons below.
Evidence root **R**: `/home/likeyi/agent-ua365-r3-chcvsvhp`, outside Git.

## Baseline and scope

HEAD, local main, origin/main and the read-only remote main query all matched
`6c15055623877f48f8ed7dedf69b1c783ebe5a68`. Branch main; empty index; `evals/`
was the only untracked entry. R/baseline.json captures tracked digests and 151,780
protected entries, including prior acceptance workspaces and evals. Protection
uses size/mtime/mode and SHA-256 for files at most 64 KiB and every eval file;
it is not a fresh full-byte audit of every dataset or checkpoint.

Reviewed README/AGENTS, R1/R2A/R2B, USI-4, native-peak acceptance and peak-contract
review, with the implementation of annotation, resource handoff, authority,
H5AD, native EpiZoo/PCA/Scanpy boundaries, raw input admission, compiler/preflight,
Session/Revision, evidence and delivery. Historical records describe the decisions
at their original checkpoints. This unit explicitly retires the annotation-only
primary-scope veto; it does not change primary-nuclear fragment/QC/matrix science.

## Native requirements and implemented changes

| Candidate | Decision and concrete reason |
| --- | --- |
| Every direct MAESTRO peak must be primary nuclear | **REMOVE.** Pinned native Enhanced RP uses exact contig strings and gene geometry, not QC classifications. Valid unmatched peaks receive zero weights. All supplied columns/counts now reach native RP. |
| QC resource required for annotation gene/signature handoff | **SIMPLIFY.** Native refgenes already supplies its transcript/exon geometry. A QC scope remains an optional, pinned source of known contig bounds for existing consumers; it is not required for a complete uploaded reference pair or a configured annotation resource. Unknown contig lengths are explicitly unassessed. |
| Refgenes must contain primary-classified transcripts | **REMOVE.** Nonempty valid transcript/exon rows and gene/signature correspondence are the relevant native requirements. New observations count all transcripts. A small reader retains the original `primary_transcripts` meaning for historical records. |
| Reopen/reinspect/rehash the just-loaded matrix within `qualify` | **REMOVE.** Shared in-memory axis validation and the already-calculated logical digest replace a redundant `load_matrix` call and second logical digest. Each independent verifier still loads its own exact inputs. |
| Canonical export branches conditional on external input after the external branch already returned | **REMOVE.** Unreachable external variants duplicated an existing complete external exporter. |
| Annotation reference qualification tied to Python implementation hashes | **REMOVE.** Changing an unrelated owner line forced registration/qualification again despite identical scientific bytes. New records pin resource bytes; the existing reader ignores only exact known implementation paths in old annotation records. Method/schema compatibility, data integrity and permission still apply. Scientific result authority and QC resource qualification are unchanged. |
| Native computation vs independent verification | **KEEP.** Production invokes pinned native RP/R/Presto; one owner replay checks the computation, assignments, exact source preservation and new participation evidence. Accepted authority supplies subsequent integrity/recovery without fresh science. |
| H5AD structure/privacy checks vs sparse numerical/axis validation | **KEEP.** Indirect storage/path privacy and scientific count semantics protect different boundaries. No full scATAC matrix is densified. |
| External FASTQ/BAM/fragments/H5AD intake | **KEEP.** Existing declared-source adoption/producer paths establish actual inputs without fabricated upstream Agent runs. Raw alignment/corrected-barcode and source-conservation claims still require their scientific evidence. |
| EpiZoo feature vocabulary | **KEEP.** A native checkpoint requirement; no changes to EpiZoo. PCA retains centered sparse-safe Scanpy ARPACK and shared downstream actual dimensions. |
| Planner/compiler/preflight and Session bindings | **KEEP.** Typed allowlisting, explicit roles/parameters, exact identities, lineage, generation and idempotency remain necessary. No language parser, implicit completion or new engine. |
| InputSet/catalog replacement; v3 planning and old native annotation retirement | **DEFER.** Actual configured inputs, API consumers and persisted records use them. Removing these would require a separate broader API/history decision; no new catalog or conversion framework. |
| Historic graph readers, accepted evidence vs report/export identities | **KEEP.** Real R1/R2A results use both graph versions. Source evidence, stored prose and privacy-safe derivative bytes serve distinct consumers. No old result is migrated or reinterpreted. |

The same direct method and numerical profile remain in use. The broadened
admission and new observation are identified by exact specification, adapter
identity and pinned sidecars. Existing publications without participation metadata
remain readable; new native publications pair an exact feature table with the
same compact observation in backend provenance, manifest and downloaded H5AD.
These are delivery/provenance copies checked against the single native observation,
not another scientific calculation or resource framework.

The owner observes `RP_AddExonRemovePromoter`'s actual sparse transcript-by-peak
weights without changing the returned weights. It records every supplied feature,
its exact integer count total, exact gene-contig-name presence and number of
nonzero transcript weights. This precedes native representative-transcript
selection: a nonzero transcript weight is not proof of influence on every final
gene score or on a cell label. Conversely, a zero column cannot contribute to RP.
No contig alias, coordinate, count, cell, feature or group is silently changed.

Intervals must retain exact `chrom:start-end` zero-based half-open syntax,
nonnegative start and positive width. The numerical bound preserves exact native
midpoint arithmetic. Supplied optional contig lengths are enforced for matching
names; no genome bounds are invented from gene coordinates. Species, assembly,
measurement, sparse finite nonnegative integral counts, unique exact axes, group
correspondence, resource pins/namespaces/context and nonzero per-cell RP remain
required. Syntactic validity and declared assembly are not independently proven
alignment history or genome-wide coordinate qualification.

## Scientific resource availability

The existing discovery → compatible selection → owner validation → consumption
path remains. Complete uploaded genes/signatures now work without a QC InputSet;
Agent supplies internal paths, digests and the existing specification. A selected
or captured reference is not silently replaced. Discovery does not reconstruct
reference science, and unchanged qualified references need no per-analysis
registration. Private Session scope, trusted promotion and pinned permission
review remain enforced. No new runtime installation was performed.

| Resource | Disposition |
| --- | --- |
| Pinned native MAESTRO and R/Presto runtime | Available through the existing normal `agent` Conda environment; unchanged. |
| Existing hg38 refgenes and CIBERSORT-derived signature | Computationally compatible in the established private research acceptance scope. Not promoted to common deployment. |
| Existing QC genome scope | Still available to QC/raw consumers and optional exact-bound checks; no longer an annotation prerequisite. |
| Common annotation signature | **Not yet approved.** No new applicable sharing grant for the exact retained CIBERSORT-derived bytes was established. |
| ScTypeDB alternative | Relevant human immune gene-marker candidate, but not qualified as a drop-in MAESTRO reference. Its positive/negative specificity scoring differs, and permission for the exact database bytes and upstream content was not independently established. |

R/resource-review.json and permission-sources.json record the new review. The
[ScType publisher repository](https://github.com/IanevskiAleksandr/sc-type) offers
GPL code; its [primary paper](https://www.nature.com/articles/s41467-022-28803-w)
describes a database compiled from CellMarker/PanglaoDB and a distinct scoring
method. Neither a convenient code license nor article availability was used to
assert a new third-party data grant or biological interchangeability. No marker
panel was substituted and no private reference became globally available.

Normal operator configuration and Conda runtime settings are preserved. Once applicable permission for the
exact retained signature and intended deployment, or a qualified permitted
replacement, exists, use the existing owner admission and trusted
`shared_resources.promote` path with pinned permission evidence. Compatible
ordinary Sessions then discover it automatically. This unit does not claim that
an isolated private reference fixes common deployment.

## Validation and product evidence

### Real Firefox, OpenRouter and native science

The ordinary request was “Annotate the cell types in this scATAC-seq dataset.
These are human PBMCs. Use an appropriate reference.” The actual configured
`openai/gpt-5.6-sol:floor` profile asked for assembly, count slot, measurement history
and independent groups. An ordinary scientific reply identified the original X
counts and the publisher's `cellranger_graphclust` obs column. The first reply
encountered an OpenRouter connection failure; the subsequent ordinary continuation
reused the original upload/task and selected/planned real native annotation.
No semantic repair, model fallback or scripted plan was applied. An earlier
browser navigation retried only because the isolated server was still starting;
that attempt submitted no turn.

- Session: `71cca8eb-7d66-4241-9707-0362887d7c03`.
- Run: `turn-3d747a32bb10b572be7b6bc3141be18a900ecc8ec527242363017764b6e3b261:run`.
- Revision: `9464a6efbf9d5fb44ae410963dfa6456ea5e6cfe8ed0d784090a8aa6a9820c27`.
- Source SHA-256: `c75bca64515710f6045a1c65d9f0dfb9fe9827102501856007b2bfcb58b33c81`.

This is the unchanged, full, publisher-derived R2B H5AD: **1,004 cells × 82,579
peaks**, int32 CSR, 8,164,121 nonzeros and **22,279,127 supplied counts**. No
primary-only subset or altered chromosome names was prepared. The private
acceptance specification references the same pinned gene/signature bytes, without
a QC scope dependency. Native production and one independent native replay run
once each; source qualification runs twice, once per independent invocation.
Subsequent questions, downloads and recovery run zero additional science.

| Native observation | Features | Supplied counts |
| --- | ---: | ---: |
| Exact contig name absent from the selected gene table | 31 | 6,062 |
| Zero weight across all native transcripts, including the above | 3,286 | 238,421 |

The cells × genes RP is 1,004 × 28,307 with 12,960,672 nonzeros. It matches the
preserved R2B full-input native diagnostic at every sparse value, with zero
difference nonzeros. R2B's separately qualified alias diagnostic remains relevant:
26 of the 31 extra peaks receive weights under different documented names. R3
neither denies that potential contribution nor applies the alternate names.
Accepting the exact-name computation is not a claim that those peaks are
biologically irrelevant or that all supplied peaks informed the labels.

All six publisher groups receive unique positive candidate assignments under the
existing scoring rule; 1,004 assigned cells, zero unassigned/ambiguous cells.
Group labels are MemoryBcells (106), Monocytes (359), ActNK (132), CD8Tex (114),
RestMemCD4Tcells (115), and NaiveCD4Tcells (178). These are **method outputs**,
not independently confirmed PBMC identities; validation remains `not_assessed`.

Real Firefox opens the report and annotation detail, displays the participation
limits, and downloads the H5AD. Export SHA-256 is
`a1a78ed3ece83a359c03239924e576ace622878ee8a6801cce2220a7a0fd827b`.
R/science-check.json verifies exact original sparse arrays/dtype, cells/features,
obs/var and original uns, plus every cell's group/assignment propagation. Export
provenance includes the participation counts and limitation text; private source
locators remain omitted. Source artifact and exported-content digests remain
distinct. R/feature evidence is in the accepted run's `primary/feature-participation.tsv`;
it records the complete ordered feature axis without recomputing RP.

“Did all of the supplied peaks contribute…?” receives an accurate live explanation
using the newly accepted participation facts and method limits. “Which groups
could not be identified?” returns the correct zero-unassigned conclusion, but its
prose also incorrectly cites `groups_omitted` (summary truncation) as support for
that conclusion. The exact independently accepted `unassigned_cells=0` supports
the conclusion; the omitted-group citation does not. This provider-authored
semantic attribution defect is recorded, not counted as fully qualified answer
reasoning or repaired with a tool-specific language rule. It remains a separate
shared answer-quality follow-up.

The current dual-pane UI remains unchanged structurally. The existing limitation
panel and annotation details now expose the native matching caveat. Source-format
guidance, assignment summary, report, download and refresh work in native Firefox.
Fresh server restoration preserves all ten conversation texts, generation 1,
exact Revision and all **43** workspace files, with zero provider/scientific calls.

### Actual normal deployment and historical access

Using the unchanged normal operator and ordinary Conda activation, fresh Session
`f34ace91-6f96-4312-8adf-f8347dbe199d` captures the same upload and scientific
clarifications, then reports the missing compatible reference. No annotation runs,
no Revision is created and generation remains zero. Restart restores all four
texts and preserves all **103** normal workspace files. The operator remains SHA
`f59312491c2719ae93bc3366724335550a713e5c76cf6bd84ddbc01a0558477e`.
Only ordinary new Session/upload records are added; old records/configuration are
not overwritten. This is **blocked deployment acceptance**, not normal annotation
success. The successful private trial above is explicitly separate.

Across the private and normal journeys, R/audit-summary.json records **12 provider
attempts over seven live turns**, including one connection failure; all use the
configured model. These are bounded observations, not a reliability estimate.
Read-only historical USI-4 and R2B checks preserve **38** and **71** workspace files
respectively and construct no providers or science. The old normal R2B pending
Session also reads successfully with its original stored clarification. The
reader accepts its exact retired scope-related message without rewriting it.
Historic error-code presentation is retained; the retired scope exception is no
longer raised by new direct annotation.

### Software and scientific validation

R/native-tiny.json uses explicitly simulated inputs with the real pinned RP code:
a contributing feature, a distant zero-weight feature and an unmatched name.
Observation-enabled and unobserved RP are identical. This is bounded numerical
software evidence, not biological qualification. The genuine full-data comparison
and native/R/Presto execution above are separate.
R/native-negative.json separately confirms that actual pinned RP rejects simulated
cells with zero evidence when all supplied features are unmatched.

The final focused run passes **40 tests in 114.40 seconds**, with one intentional
duplicate-axis warning. It covers changed direct inputs, malformed intervals and
counts, source/group/assembly integrity, independent replay, participation tampering,
export, reannotation/history and complete reference upload without a QC catalog.
The earlier development module run had 27 passes and a stale cancellation stub
failure; the stub now supplies the observation seam. A nonfrozen development
annotation run had 126 passes and two export failures while implementation modules
were being edited. The consistent-module focused rerun passes both cases; neither
failure is labelled a baseline failure. The final frozen adjacent regression is
reported below, rather than combining these overlapping runs.

The frozen adjacent regression reports **363 passed, one failed, one skipped,
eight warnings in 810.42 seconds**, exit 1. All **557** frozen source/test files
remain unchanged throughout. The sole failure is the new resource-code-update
test asserting an internal explanation against the public `invalid_reference`
exception string; the owner correctly rejected the mutated scientific bytes.
After that run, only the test assertion was corrected to check the typed error
code and its explanation. No production changes followed the frozen regression.
The full shared-resource module rerun is recorded below.

The regression includes all annotation integration tests,
shared-resource privacy/promotion/permission/integrity and code-update reuse,
scientific detail/dialogue, Web artifacts, FASTQ/BAM/fragments handoff, native marker
policies, PCA, shared graph publication and PCA authority continuation. No
`RUN_*` opt-ins or test exclusions are used. The one skip is the explicitly gated
real Chromap FASTQ qualification; unchanged native raw science is not rerun here.

The final full shared-resource module passes **nine tests in 42.59 seconds**,
exit 0. This covers the corrected assertion plus existing private/shared visibility,
permission review, promotion, dependency integrity and configured-root behavior.
These overlapping runs are not summed into a fictitious single passing suite.

The final preservation check finds **zero changes across 151,780 protected
entries**, including all 1,274 historical eval files, under the bounded inventory
method above. EpiZoo, retained scientific inputs/references/checkpoints and normal
operator configuration remain unchanged. `git diff --check` passes; the index is
empty. The complete review-file list follows; no commit or push was performed.

## Remaining work and disposition

Common-reference permission remains a product blocker. Recommended next exploration:
qualify the permission/science of a concrete human-PBMC signature, deploy it through
the existing normal resource path, and test annotation from a fresh ordinary
Session plus accepted-clustering-to-original-count continuation. Neither code
simplification nor computational label production establishes biological accuracy.
UA3.6.5 remains open. No commit or push is authorized or performed by this unit.

**Commit recommendation:** the bounded implementation is ready for review and a
scoped commit after user approval. Real full-input native execution, independent
replay, exact downloaded data, historical access, browser recovery and the affected
software checks support the simplification. This recommendation does not approve
common-reference deployment or claim complete live-answer quality. HEAD, main,
origin/main and remote main still match the intake baseline; R/final-git.json
records the final file list, hashes, empty index and checks.

## Review files

The implementation changes are limited to:

- `src/agent/tools/analysis/{external_marker_annotation,annotation_contract,annotation_input,scatac_annotation}.py`
- `src/agent/tools/data/scientific_authority.py`
- `src/agent/application/{annotation_delivery,dialogue_evidence,raw_handoff,shared_resources}.py`
- `src/agent/orchestration/annotation_registry.py`
- `src/agent/report/analysis_report.py`
- `tests/annotation_integration/{test_external_entry,test_shared_resources,test_uploaded_resources}.py`
- `README.md`, `AGENTS.md`, and this report.

The pre-existing untracked `evals/` tree is outside this change. Browser captures,
audit logs, resource review evidence, downloads and acceptance scripts remain
under R, outside Git. No credentials, scientific datasets or reference binaries
are included in the review files.
