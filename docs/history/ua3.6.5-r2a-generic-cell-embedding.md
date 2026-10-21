# UA3.6.5-R2A — PCA and shared cell embedding analysis

Date: 2026-10-19. Closeout status: **COMPLETE within the bounded R2A scope**.
This adds the requested scientific capability and acceptance evidence; it does
not declare the broader UA3.6.5 product exploration fully accepted.

R1 was committed and pushed separately at
`d46e132b27fa0ad80c5859bedcf67fd6639dd47d`
(`Complete UA3.6.5-R1 scientific input continuation fixes`). Before committing,
all 15 authorized file hashes and all 553 verification hashes matched the
reviewed snapshot; the index was empty and local/remote main matched the stated
baseline. Only those 15 paths were staged. After the push, HEAD, local main,
origin/main and queried remote main agreed; only the protected untracked `evals/`
remained. The initial R2A checkpoint below was uncommitted. The later architecture
cleanup section records the subsequent explicit commit/push authorization and
final validation; it supersedes that checkpoint’s Git status.

New isolated evidence root **R2** is
`/home/likeyi/agent-ua365-r2a-vc9lfps6`. Historical R1 inputs and evidence are
read-only. The new Web workspace, operator configuration, browser profile,
provider/scientific call audit and test outputs are under R2, outside Git.

## Initial R2A checkpoint — scientific capability and applied defaults

The Registry adds `pca_embed_cells`, bringing the executable inventory to 24.
It operates on the supplied H5AD `X` values and preserves ordered cells and
feature positions. It performs no normalization, log transform, scaling,
whitening, HVG selection, cell filtering, feature filtering or genome conversion.
Duplicate feature names do not trigger alignment or removal; positions and exact
source identity remain authoritative. Cell IDs must remain unique and representable
in the existing ordered text sidecar. No genome/model resource is required.

Production calls native `scanpy.pp.pca`, with centered PCA and explicit ARPACK.
The default component count follows the library's ordinary policy:
`min(50, min(n_cells, n_features) - 1)`. Optional explicit component count and
random seed are supported; seed defaults to 0. Computation uses float64 and
published scores use float32; both are recorded. Sparse X is never densified.
The sparse source/copy estimate and solver-workspace estimate each have a 2 GiB
execution bound. These are engineering bounds, not biological qualification or
limits on all Agent uploads. Constant/nonfinite/non-numeric measurements fail.

ARPACK avoids an implicit switch to a feature-covariance solver on high-dimensional
peak matrices. This is centered PCA, not uncentered TruncatedSVD or TF-IDF/SVD
(LSI). Native semantics are documented in
[Scanpy PCA](https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.pca.html)
and [scikit-learn PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html).
The audited environment uses Scanpy 1.11.5 and scikit-learn 1.9.0; online sklearn
documentation may describe a newer patch release.

The compact H5AD records `X_pca`, actual dimensions, applied parameters, versions,
source SHA-256, exact ordered-axis digests and fit state (loadings, means, variance
and explained-variance ratios). Its explicit publication locator binds the
PCA owner's manifest. The independent verifier checks exact source bytes/axes,
centered source projection, orthogonality, variance and eigenpair residuals. For
a truncated fit, SciPy's native largest-eigenvalue solver on the centered omitted
subspace rejects a fit that silently substitutes lower-variance eigenvectors.
It builds no dense feature covariance. Small dense-reference tests independently
compare the principal subspace and explained variance.

The PCA owner uses existing operation-local proofs and schema-2 scientific
authority. Publication/source integrity and accepted lineage are checked by the
existing owners. No second Planner, resource manager, graph engine or authority
framework was introduced. Later report construction and accepted reuse can reuse
the scientific proof; standalone unqualified consumption remains source-aware.

## Shared downstream contract and compatibility

The existing embedding/cell-ID port now also accepts `cell_embedding_bundle.v1`.
PCA supplies a self-describing H5AD and ordered IDs. The same neighbor tool reads
the verified producer representation and shape and explicitly passes `X_pca`
to Scanpy. User strings and Scanpy's automatic representation choice do not select
the representation. New graph publications use `cell-neighbors-publication.v2`.

The existing `.npy` EpiZoo channel, 512-dimensional `X_epizoo` artifacts,
publication v1 profile, receipt identities and legacy reader remain exact. Native
EpiZoo validation and model code are unchanged. PCA is never padded, truncated or
relabeled to satisfy EpiZoo. EpiZoo-specific label transfer is not broadened to
independently fitted PCA coordinate systems.

Leiden and UMAP reuse the same algorithms and compact-artifact validation. They
preserve the accepted embedding identity, source parameters and graph lineage.
Both graph-only and clustered UMAP remain supported. Safe downloads retain the
actual representation and cells; PCA fit arrays and private source locators are
omitted from the compact public exports. Reports/evidence identify PCA and its
applied parameters, including in downstream-only continuation runs.

## Typed method selection

Current offered H5AD bindings can project bounded scientific-owner suitability
observations to the interpreter. These validate supplied X and compare available
qualified EpiZoo resources using the existing native vocabulary/value validators.
They do not run PCA or EpiZoo, select a workflow, infer biological sample identity,
or confer execution authority. Accepted-result continuations with no new files
use existing context instead of rereading offered historical raw inputs.

The LLM interprets generic versus explicit method requests through existing typed
admission/planning. The Registry/compiler and scientific owners validate actual
execution. Explicit EpiZoo requests have no exception-triggered PCA fallback.
Generic clustering may use PCA when the available EpiZoo vocabulary is incompatible.

## Real Web/OpenRouter acceptance

All rows below used native headless Firefox/Marionette, the ordinary upload and
chat controls, and actual OpenRouter `openai/gpt-5.6-sol:floor` responses through
the configured GPT-5.6 Sol profile. Provider responses were observed, never
scripted or repaired. Acceptance-only call instrumentation observes the unchanged
scientific functions. It is separate from the small scripted application tests.

| Case | Observed outcome |
| --- | --- |
| Untouched Chen 2019, “Cluster these cells and show me a UMAP.” | Real model selected PCA → existing neighbors → Leiden → UMAP. Accepted 9,190 cells × 241,757 supplied features, 50-dimensional `X_pca`, 10 clusters at resolution 1 and a 2D UMAP. No species/assembly/PCA-setting question and no EpiZoo execution. |
| Follow-up, “Change the Leiden resolution to 0.5 and show the updated UMAP, reusing the current neighbor graph.” | New child Revision: 6 clusters, resolution 0.5, updated UMAP. Exactly one Leiden and one UMAP call; zero PCA/neighbor production or independent upstream reconstruction. Exact prior graph path and authority were reused. |
| Current-result question about method and dimensions | Real scientific answer cites accepted downstream evidence: PCA and 50 components. No science or new Revision. |
| Preserved compatible 32-cell human control, “Cluster these human cells and show me a UMAP.” | Real model selected qualified EpiZoo and its pinned resources. Native CUDA inference, 512-dimensional `X_epizoo`, shared neighbors, 4 Leiden clusters and UMAP all succeeded. No PCA execution. |
| Untouched Chen, explicit EpiZoo request | Clarification; zero scientific execution and no Revision, with no PCA substitution. The model chose the existing EpiZoo role handoff; its conservative H5AD intake rejected an unsupported group encoding. This establishes no fallback, not a successful EpiZoo-incompatibility explanation. |
| Untouched Chen, “Use PCA with 5 components to embed these cells.” | Real model bound the explicit component count and selected only PCA. Accepted actual 5-dimensional embeddings for all 9,190 cells. |
| Clearly labelled simulated 5 × 8 constant matrix, explicit PCA | Model selected PCA despite the negative suitability observation; the scientific owner rejected zero centered variance. Failed Run, no accepted artifact or Revision. This is software rejection evidence, not biological qualification. |

The external source remains exactly 60,786,319 bytes with SHA-256
`cc0ac62fd36a0be400e7d3bf2f7e32d5d03a191c332cd829e590e437048019f3`.
No feature/cell metadata or measurement was rewritten to accommodate Agent.
The preserved human control remains 80,541,640 bytes with SHA-256
`0c601bf017053f82fa34273156f647148c3ccddecf6646fcd9fcb4cc4aabe264`.
Its 32-cell bounded scope is not a new biological validation of EpiZoo.

External Session: `c8cffb81-2e3a-4b5b-82c3-c4cd23a8185e`.
Initial Revision:
`fd63236ce02f5403bd60bb6aea41f80b85f0728ae662399f47c6269501d8bb84`.
Resolution-follow-up Revision:
`94d8d84dadd86ed0166894a0f96266da2d5c3f5b2803b835c08b2025a353b46b`.
Compatible-control Session: `830f5704-a325-4310-ad59-461d8a028d3c`.
Explicit-PCA Session: `e3c56e3f-8999-4d52-88ab-4961620cecab`.

Browser report access identifies PCA and `X_pca`; the native UMAP figure renders.
Three native downloads (`embeddings.h5ad`, `clusters.h5ad`, `analysis.h5ad`) match
their delivered response digests. Independent reads confirm every ordered cell,
PCA value, retained graph value, label and coordinate matches the pinned source.
Public exports exclude private paths and PCA fit arrays. Original artifact hashes
and delivered-export hashes are deliberately distinct. Graph-only delivery is
covered by the shared software checks, not counted as a fourth native download.

After the first accepted result, a fresh server and Firefox restored the Session,
report, figure and downloads with all 28 workspace files byte-identical, zero new
provider calls and zero science. After continuation, all original scientific
files remain identical; only the Session envelope changes through ordinary chat
append/activation. Two-Revision historical preview/refresh passes without changing the active Revision.
The compatible-control browser report projects its selected 32-cell, 4-cluster
results; the original accepted full report and run record bind EpiZoo/512. The
legacy client-safe projection does not automatically add upstream EpiZoo fields.

Evidence under R2 includes `provider-audit.jsonl`, `science-audit.jsonl`, browser
snapshots/screenshots and timeline, `run-summary.json`, `native-downloads.json`,
`download-science-verification.json`, `restart-verified.json`, and
`original-artifacts-after-continuation.json`. A second fresh server using the
final production code restores both historical/current revisions and both real
Sessions with **all 100 workspace files unchanged**, zero new providers and zero
science (`final-recovery-verified.json`). The isolated server and Firefox were
stopped after acceptance. First-run cold compilation and
acceptance tracing inflated neighbor latency; these runs are not performance
benchmarks.

## Scientific verification and regression

The small numerical tests compare centered PCA to an independent dense SVD
reference and forbid sparse-to-dense conversion in production. They cover native
small-sample component counts, explicit counts, exact input preservation, repeated
feature names/HVG metadata without filtering, invalid inputs, altered scores/axes/
variance/loadings/parameters/source and replacement with lower principal components.
Shared neighbors/Leiden/UMAP, clustering-evaluation validation, public exports,
registered semantic channels, publication authority and exact graph continuation
are exercised. Scripted application continuation forbids PCA/neighbor production
and independent upstream reconstruction and still succeeds.

A separate labelled simulated 32 × 512 EpiZoo-style input was run through the
preserved baseline and current shared neighbors/Leiden/UMAP implementations at the
same isolated paths. All public results and all five generated artifact/manifest/
receipt files match byte-for-byte (`legacy-compatibility/{baseline,current}.json`).
Native EpiZoo model source remains unchanged.

Interim focused checks passed 149 tests in 22.20 seconds after correcting registry
expectations and a report-projection regression. The broader neighboring regression completed **1,642 passed, 48 skipped,
4 failed, 10 warnings in 601.70 seconds**, with all 556 frozen source/test/resource
files unchanged. All `RUN_*` gates were removed and no test exclusions were used.
The four failures were assertions: two assumed the first five Registry entries
were the legacy EpiZoo plan; two carried stale EpiZoo schema fields/full-v3 size
budgets. Both latter failures reproduce against an isolated archive of the pushed
R1 source (`baseline-planner.log`). The explicit expected EpiZoo chain and current
qualified target-model fields are now tested. Full v3 measures 16,769 schema bytes
and 27,607 prompt bytes; bounded budgets are 17,000, 28,000 and 45,000 combined.
The scoped v4 planner contract and its budgets are unchanged.

Code review also found and corrected a new case-sensitive suffix dispatch that
would reject previously valid uppercase `.NPY` inputs. A real legacy publication
regression now verifies exact source values, v1 profile and owner qualification.
This does not change lowercase legacy publication bytes. Final correction scope passes **674 tests, 48 skipped, 10 warnings in 119.99
seconds**, exit 0. It includes all analysis-tool and report tests, both affected
Planner modules, the scientific verifier, PCA authority continuation and UMAP
independence. All 556 frozen source/test/resource files remain identical during
execution; all `RUN_*` gates are disabled and no exclusions are used. Counts
across scopes overlap and must not be summed. Exact commands, logs and frozen
hashes are in `final-regression.json`, `closeout-regression.json`, their matching
logs and `closeout-source-freeze.json`. Warnings concern installed dependency
versions and deliberately repeated feature names; gated scientific evaluations
were not enabled or rerun. No unresolved failure remains in these checked scopes.

## Limitations and preservation

Centered PCA is a numerical operation on supplied X. This acceptance does not
establish that unnormalized accessibility counts yield optimal biological clusters,
calibrate resolution, qualify cell purity, or establish correspondence between
independently fitted PCA spaces. Label transfer remains within its existing
EpiZoo scientific contract. No LSI capability or silent preprocessing is added.

Suitability observations inform the LLM; they do not guarantee provider intent
quality or override a typed decision. The invalid constant fixture demonstrates
that the owner still rejects an unsuitable plan. Existing generic error messages
for invalid PCA and EpiZoo intake remain less specific than the owner diagnostics.
The explicit-EpiZoo case exposes an existing conservative H5AD handoff limitation;
it is not a claim that every external H5AD receives an ideal compatibility message.

Read-only preservation checks match 107,531 protected historical files, including
107,453 small-file SHA-256 checks; 1,274 eval files; all 90 R1 workspace files; both
consumed real inputs; and native EpiZoo source. Large historical files retain their
recorded size/mtime/mode; this is not a fresh hash of every large historical byte.
No new public resource download, resource-catalog replacement, production-default
change, historical rewrite or biological dataset/checkpoint Git addition occurred.
R2A changes remain unstaged and uncommitted for review.

## Review state

R2A implements one PCA producer, reuses the existing graph/clustering/UMAP path,
and passes the bounded real browser and scientific verification checks above.
The remaining message-quality and scientific-generalization limits are explicit.
`git diff --check` passes. The index is empty; the 30 R2A source/test/documentation
paths are available for review. No R2A commit or push was performed. HEAD, local
main, origin/main and remote main remain the pushed R1 SHA recorded above.
The pre-existing untracked `evals/` directory remains intact and excluded.

## Architecture cleanup — decision record (2026-10-19)

The follow-up explicitly authorizes a validated R2A/cleanup commit and normal push.
Its starting branch is main at the R1 SHA above, with an empty index and exactly
30 R2A paths matching the previous closeout hashes. Initial diff, source copies,
tracked hashes and previous R2 evidence hashes are captured outside Git under
`/home/likeyi/agent-ua365-r2a-cleanup-sx5ki_7v` (**C**). The earlier uncommitted
review state above describes the preceding checkpoint, not the new authorization.

The inventory covers scientific owners/publications, application source/role and
resource admission, planner/Registry/compiler/preflight, Session/Revision,
authority, report projections and Web/download delivery. Decisions precede edits:

| Candidate | Decision and evidence | Intended change / verification |
| --- | --- | --- |
| Extension-selected v1/v2 new neighbor publication and receipt identity | **SIMPLIFY**. `build_cell_neighbors` is the sole publisher caller; both formats already share production and owner verification. | All new graphs use existing v2 with producer metadata. One version-aware identity reader preserves old v1 profile/receipt bytes. Test both producers, actual old graphs, authority reuse and delivery. |
| v1 interpretation / old graph representation | **KEEP, bounded reader**. Four accepted real steps in R1, R2 and USI-6 include three v1 EpiZoo graphs and one v2 PCA graph (`C/historical-neighbors.json`). | No rewrite or new legacy writer. Verify their pinned authority, exact historical result access and downloads; inspect original source when explicitly qualifying standalone. |
| EpiZoo NPY/ID reader | **KEEP**. Current native EpiZoo producer and label transfer expose/consume it; compact embedding H5AD is a pinned download, not a replacement execution-authority contract. | Keep native dimensions/vocabulary and label-transfer scope. Generic graph consumers use the trusted producer representation. |
| Extra embedding read in graph publisher | **REMOVE**. Producer already has exact metadata; independent graph verifier separately checks source correspondence. | Pass captured metadata to publisher; retain initial independent verification and accepted-proof reuse. |
| Whole peak-annotation H5AD validation in EpiZoo vocabulary observation and reference feature extraction | **SIMPLIFY**. `epizoo_inputs.observe_source` and `reference_supplementation._feature_names` duplicate an entire-file annotation gate before reading only the feature index. This rejects the untouched Chen file's unrelated encoding. | Shared bounded feature-index reader at the data owner; preserve exact vocabulary, pin checks, indirect-storage rejection for consumed objects and later model/reference qualification. No annotation-owner changes. |
| PCA projection/eigenpair/leading-subspace checks | **KEEP**. Projection alone accepts lower components; corruption tests demonstrate the invariant. | Keep initial independent numeric checks. Existing operation/accepted authority replaces covered deep reconstruction; no new cache. |
| InputSet, single/group/FASTQ collection bindings | **KEEP / DEFER migration**. Real R1/R2 operator configurations and captured turns use them. Groups bind explicit semantic roles; collections preserve ordered raw-library declarations and source provenance. | No forced conversion of immutable captured submissions. Preserve generation, retries, exact binding and attachment continuation checks. |
| Resource qualification vs discovery vs consumption | **KEEP**. `shared_resources`, `shared_genome`, reference supplementation and owner checks separate scientific qualification, Session/private scope, deployment review and current selected-byte integrity. | Retain those responsibilities and resource pins. General catalog consolidation is deferred; no new resource framework or MAESTRO work. |
| v3 planning / deterministic offline planner | **DEFER retirement / KEEP explicit offline use**. CLI/API still offer explicit v3 and offline modes; default interactive v4 has no automatic fallback. Removing them requires a separately declared API retirement, not scientific-history migration alone. | Preserve compiler/preflight and current explicit consumers. Reconcile actual registry-derived assertions; no budget increase merely to suppress defects. |
| Unpublished historical graph verifier | **KEEP**. It interprets pre-publication result fields separately from owner-qualified authority; it grants no schema-2 proof. | Do not silently promote old results or replace recorded verification meanings. Shared graph structural validators remain reused. |
| Evidence, original report, client-safe projection, public export | **KEEP distinct responsibilities**. Accepted report identity, bounded user facts and safe downloads have different content/privacy guarantees. | No UI redesign or collapsing a derivative into original scientific evidence. Test accepted report/figure/download reads without science. |

Two additional confirmed delivery defects were classified **SIMPLIFY / CORRECT**:
`InteractiveAgentApplication._accepted_neighbors` still admitted only v1 despite
the generic owner/export reader, so the earlier real PCA graph was absent from
Web downloads. Application eligibility and snapshot delivery now use the owner's
small contract/profile map and preserve each stored contract identity. PCA compact
export also overwrote the source-artifact digest with the raw-input digest; new
derivatives retain `source_sha256` for the artifact and `input_sha256` separately.
Existing downloaded bytes are preserved, not relabeled or silently repaired.

### Implemented cleanup

All newly produced EpiZoo and PCA graphs now use **cell-neighbors-publication.v2**.
The suffix-selected v1 writer/profile choice is removed. The one existing graph
verifier, publication builder and receipt checker take the persisted version only
when interpreting history. Three real v1 graphs justify the small v1 profile and
legacy missing-descriptor interpretation. New EpiZoo descriptors explicitly name
`X_epizoo`, float32 and 512 dimensions. No native EpiZoo model code, vocabulary,
label transfer or graph/clustering/UMAP algorithm changed. New publication bytes
intentionally differ from R1; the earlier byte-equivalence experiment above is a
record of the pre-cleanup checkpoint, not a claim about the new v2 writer.

The graph publisher receives metadata already read by its producer. It no longer
reopens the embedding simply to obtain the same descriptor. Initial independent
source-alignment verification remains; reuse/downloads do not repeat PCA fitting,
leading-subspace verification, EpiZoo inference or neighbor construction.

`scatac.read_feature_names` replaces both duplicated full annotation-format gates.
It supports the actual ordered AnnData 0.1.0 and 0.2.0 feature indexes, bounds
file/count/string decoding and rejects indirect consumed storage. It reads neither
X nor unrelated annotations. This observes a vocabulary, not matrix validity,
biological eligibility, reference compatibility or execution authority. Existing
source pins and scientific-owner qualification remain mandatory. The strict
peak-annotation reader is unchanged for its actual annotation consumer.

New EpiZoo downstream evidence records method/representation without inventing
PCA parameters. Safe graph exports retain reviewed method/dimension fields while
omitting the full producer descriptor, which can contain private PCA locators.
Tests now assert that explicit safe projection rather than the old assumption
that every non-path top-level provenance field must be copied verbatim.

### Historical compatibility evidence

`C/historical-verified.json` checks all four preserved accepted graphs through
normal accepted-authority loading, actual Revision/artifact inventory, and
application download preparation. This covers R1 EpiZoo v1, R2 EpiZoo v1, R2 PCA
v2 (9,190 cells), and USI6 EpiZoo v1. Each downloaded graph preserves every cell,
embedding and sparse graph value and identifies its original stored contract.
All four also produce a new resolution-0.5 clustering outside historical storage.
Upstream production, deep PCA/graph verification and embedding-source loading
are forbidden during these operations. **257 historical workspace files remain
byte-identical**. `C/historical-presentation.json` additionally records 12 accepted
report/figure reads. No historical receipt, evidence, Session or Revision is upgraded.


A final Registry-consumer scan classifies distributed milestone-count assertions
as **REMOVE**: eleven tests asserted a permanent total of 23, although their real
contracts concern a specific tool/port/authority boundary. Exact allowlist coverage
already belongs to `test_registry`; local tests retain their scientific assertions.
One older integration assertion also equated the first five catalog entries with
an EpiZoo workflow; it will name that explicit workflow instead. No executable
planner behavior is changed. The resource-choice test also includes a null
`fastq_whitelist` extension in an older digest expectation; this reproduces on R1.
The actual persisted definition deliberately omits that absent optional extension.
Keep the production legacy-digest rule and correct the test's explicit expectation;
do not migrate captured choices or change their identity.

### Final validation and product acceptance

The first cleanup-focused pass exposed two consumer assumptions: the export test
expected the full new embedding descriptor, and the report required PCA parameters
for EpiZoo. The privacy-safe projection and method-specific facts were corrected.
The next focused run passed **171 tests in 46.99 s**. The broad neighboring run
passed **2,023, skipped 48, failed 2 in 963.48 s**; all 557 frozen files were unchanged.
Its two failures were the obsolete species-adaptation Registry count and the
R1-reproduced absent-whitelist digest expectation described above. There was no
scientific algorithm failure. The final correction scope passes **208 tests,
6 warnings in 69.38 s**, exit 0, with all 557 frozen files unchanged. This includes
both failed tests, every affected distributed Registry assertion, the full older
Milestone-6 downstream integration module, feature-index/reference handoff,
PCA numerics/corruption, v1/v2 graph verification/export, actual application PCA
graph download eligibility, authority continuation and standard Web handoff.

The broad run also covers all analysis/report/orchestration tests, species
adaptation integration, accepted processed-prior outputs, resource identity/privacy,
scientific dialogue, typed roles, attachment reuse and UMAP independence. Exact
commands/logs/frozen inventories are `C/final-regression.*`, `C/corrections.*`,
`C/corrections-source-freeze.json` and `C/baseline-choice.log`. All `RUN_*` gates
are disabled; no exclusions or increased budgets hide failures. Scopes overlap
and counts must not be added. The broad scope was not rerun wholesale
after test-only corrections and removal of an unused delivery constant; final
focused validation covers every subsequent change. This is not the entire
repository's historical scientific evaluation suite.

Fresh native Firefox/OpenRouter GPT-5.6 Sol journeys use the ordinary product
controls and a new isolated C workspace. They record 14 real provider calls:

| Journey | Final observation |
| --- | --- |
| Unmodified Chen input, generic suitable embedding/Leiden/UMAP | PCA 50; 9,190 × 241,757 supplied measurements; v2 graph; 10 clusters; 2D UMAP. No EpiZoo execution. |
| Resolution 0.5 follow-up | Child Revision with 6 clusters and updated UMAP; exact accepted graph retained; zero upstream production or deep verification. |
| Compatible real human 32-cell input, generic clustering | Native EpiZoo 512, v2 graph, 4 clusters and UMAP; no PCA. |
| Explicit EpiZoo on Chen | Existing typed prerequisite asks for genome assembly and measurement semantics. The former unrelated H5AD-encoding rejection is gone. Original files/objective remain captured; zero science and no PCA substitution. This does not qualify Chen for EpiZoo. |
| Current-result method/dimension question | Provider authored `clarify / ambiguous_subject`; no semantic retry, override, science or new Revision. Previous successful R2 answer remains historical evidence, not a guarantee of current provider behavior. |

Native downloads now include all four PCA files: embeddings, neighbors, clusters
and UMAP analysis. Browser response and downloaded hashes match; independent reads
confirm exact axes/coordinates/graphs/labels, distinct source-artifact/input
hashes and absence of private paths/fit arrays. A fifth native download verifies
the new EpiZoo v2 graph and all its values. Reports, figures, two-Revision preview
and refresh work. Fresh-server read-only recovery preserves **all 72 C workspace
files**, invokes **zero providers and zero science**, and restores both scientific
Sessions and historical/current report views. The isolated acceptance services
are stopped after verification.

Scientific call accounting is PCA production 1, independent PCA verification 1,
EpiZoo production 1, graph production/initial alignment verification 2 each,
Leiden 3 and UMAP 3; three input suitability observations perform no decomposition.
Evidence includes `C/provider-audit.jsonl`, `science-counts.json`, `run-summary.json`,
`native-downloads.json`, `download-science-verification.json`,
`epizoo-native-download.json`, browser timelines/screenshots, and
`restart-verified.json`. Initial Chen Session is
`cd498444-2a80-448e-b5fd-b11ea2fb516f`; initial and child Revisions are
`ef03571fbe97669a9749d029cd657d1257727c9f1bee2d44404f973cadeffe59` and
`70fa452746c0d036e3fd69ca8d1e5af74fc8d901cb1e5444a905c31a582e7dd5`.

### Preservation and Git closeout

`C/preservation.json` confirms 107,531 historical file records unchanged (107,453
small-file SHA checks), 1,274 eval-file hashes unchanged, all 90 R1 workspace files
unchanged, both consumed real input hashes unchanged, native EpiZoo source unchanged,
and **11,126 prior R2 evidence hashes unchanged**. Large-file preservation retains
the stated size/mtime/mode limitation above. No datasets, checkpoints, credentials,
production configuration, resource defaults or historical evidence enter Git.

Final review covers the original 30 R2A paths and justified cleanup: 49 source,
test and documentation paths in total. Exact staged paths/diff and pre-commit
hashes are recorded in C. `git diff --check` passes. The authorized cohesive commit
is titled `Complete UA3.6.5-R2A PCA integration and architecture simplification`.
The immutable SHA and post-push HEAD/main/origin-main/queried-remote observations
are recorded in `C/git-closeout.json`, avoiding a self-referential commit hash in
this document. Normal push is conditional on unchanged remote R1 and the completed
checks; no force push or historical migration is used. Only preserved untracked
`evals/` should remain after closeout.

R2A and this selective cleanup are complete within the recorded scope. No
unresolved scientific correctness, artifact-access or delivery blocker remains in
the checked paths. Provider subject resolution remains variable, and explicit
EpiZoo on incompatible external data still needs scientifically supported
prerequisites/resources. The next highest-value work is making real-data resource
readiness and compatibility guidance clearer, including the later MAESTRO work
unit when separately authorized. No MAESTRO implementation, new scientific
capability beyond R2A, UI redesign or large resource acquisition began here;
the broader UA3.6.5 exploration is not declared complete.
