# Unified Scientific Admission and Validation Simplification

The premature external-fragments admission barrier is corrected. A registered
source can now reach ordinary interpretation and clarification without a complete
operator companion. Actual import still requires the existing declarations,
compiler bindings, exact source/resource consumption, and scientific owner checks.
FASTQ, BAM and fragments share the same explicit-or-unique-default selection policy.

This is a bounded UA3.6 integration correction. It does not qualify a new biological
workflow. [UA3.6.4a](ua3.6.4a-scientific-dialogue-closeout.md) remains **PARTIAL**;
[UA3.6.4b](ua3.6.4b-raw-preprocessing-acceptance.md) remains **BLOCKED**. Historical
records are unchanged. Evidence for this task is retained separately at
`/tmp/agent-admission-simplify-fmf7bqsw`.

## A. Exact baseline and preservation

HEAD, main, origin/main and remote main were all
`8d7ecf8842834758b56df3893a17f696e929e889`. Remote main was read with
`git ls-remote`, without fetching. The initial index and tracked worktree were
clean. Existing untracked content comprised the UA3.6.4b report and `evals/`:
981 nonignored evaluation files, 1,274 evaluation files including ignored content.

The refreshed preservation snapshot covers 86,123 existing entries. Of these,
85,501 have byte hashes; remaining large files retain size, mode and nanosecond
mtime checks. This is bounded invariance evidence, not a new universal hash of
all scientific datasets and checkpoints. The eight intended source/test edits
are the only changed existing entries. Original acceptance, Sessions, Revisions,
RunStore, resources, operator configuration and evaluations remain unchanged.
The UA3.6.4b report retains SHA-256
`a5b2ff85b4375f6c05f27fa3dbe37029121610801d246e06e2eb34137b9efc0a`.

File sets under the acceptance root (6,486 files), production configuration root
(one file), and evaluations (1,274 files) are also reconciled. EpiZoo remains clean
at `029cd631d0f9806a646c4a3b42ce10b958f2b67f`. The new report is untracked;
implementation and tests are unstaged. No staging, commit, push, fetch, pull,
reset or cleanup was performed.

## B. Validation architecture findings

The executable inventory was read from `build_default_tool_registry()`, including
required arguments, request-source bundles, semantic ports and scientific authority
hooks. `registry-contracts.json` retains the inventory of all **23** tools. The
following table identifies their contract responsibilities; it does not claim
that every tool was scientifically executed in this review.

| Registered capability | Authoritative scientific responsibility and necessary execution boundary |
| --- | --- |
| `inspect_scATAC` | `data/scatac`: bounded H5AD shape, axes and sparse metadata; no prior Agent authority or embedding resource required. |
| `epizoo_embed_cells` | `analysis/epizoo_embedding` and existing EpiZoo backend: declared supported species, exact vocabulary/model resources, sparse preprocessing and captured inference settings. |
| `build_cell_neighbors` | `analysis/embedding_analysis`: exact ordered embedding/cell identity and qualified neighbor graph; paths or matching dimensions alone are insufficient. |
| `cluster_cells` | Same owner: compatible graph, weighted Leiden parameters, exact population and publication identity. |
| `compute_cell_umap` | Same owner: compatible graph, fixed settings and exact cell order; no inferred correspondence. |
| `evaluate_cell_clustering` | `analysis/clustering_evaluation`: exact evaluation labels/cell alignment and fixed clustering; labels remain evaluation-only. |
| `transfer_cell_labels` | `analysis/label_transfer`: explicit reference/query roles, same-species and checkpoint compatibility, exact ordered labels/embeddings. |
| `evaluate_cell_annotation` | `analysis/annotation_evaluation`: exact annotation and declared ground truth alignment; no inferred role or label column. |
| `validate_scATAC_feature_space` | `analysis/replicate_pseudobulk`: raw sparse count/value, coordinate, species/assembly and feature provenance declarations plus actual data checks. |
| `build_replicate_pseudobulk` | Same owner: exact group/replicate/condition assignments and sparse SUM; no inferred replicates. |
| `run_replicate_differential_accessibility` | Existing DA owners and pinned edgeR backend: supported design/contrast, replicate adequacy, independent output verification. |
| `inspect_raw_scATAC` | `data/raw_scatac`: bounded FASTQ/BAM observations and separately recorded declarations; inspection readiness does not authorize production. |
| `prepare_scATAC_fragments` | FASTQ producer/verifier: exact intake, library/read attribution, whitelist/barcode policy, reference/Chromap resources and full producer qualification. |
| `import_scATAC_fragments` | External adoption/verifier: declared supported source profile, namespace/reference, complete records/index/conservation; upstream correction history remains declared. |
| `prepare_scATAC_bam_fragments` | BAM producer/verifier: exact source/intake/context/reference, corrected-CB declaration and paired-record reconstruction; no inference of barcode history. |
| `compute_scATAC_qc` | QC owner: qualified fragments and exact QC/TSS resource; complete barcode metrics with independent intersection checks. |
| `select_scATAC_cells` | Selection owner: exact QC parent, explicit thresholds and every-barcode decisions; statistical cell calling remains unassessed. |
| `build_scATAC_cell_by_ccre` | Matrix owner: exact fragments, selected cells, reference and ordered int64 CSR counts, independently reconstructed. |
| `adopt_scATAC_cell_by_ccre` | Matrix adoption/verification: declared external value semantics, exact reference/axes, all-value source conservation; no invented upstream processing. |
| `annotate_scATAC_cell_types` | Annotation owner: exact accepted canonical matrix authority, group identities, declared context and qualified MAESTRO resources; pinned replay verifies assignments. |
| `adopt_scATAC_cell_by_features` | Neutral adoption/shared matrix owner: explicit species/assembly/reference and external value semantics, exact sparse conservation. |
| `adapt_epizoo_species` | Existing species-adaptation owner: compatible matrix semantics, explicit strategy/profile/mapping, qualified model bundles and structural checkpoint/forward verification; no universal biological-quality proof. |
| `build_scATAC_cell_by_features` | Neutral matrix/shared counting owner: exact reference, primary scope where applicable, explicit cells or QC selection, complete ordered sparse reconstruction. |

The cross-layer findings are:

| Classification | Actual boundary and finding | Decision |
| --- | --- | --- |
| Necessary scientific check | Producer, QC, matrix, annotation and adaptation owners inspect scientific payloads and verify their own distinct facts. Registry/compiler checks alone cannot establish those facts. | Retain. |
| Necessary check at the wrong stage | Web `_fragments_selection` and application `_fragments_inputs` required complete import declarations before knowing whether the turn requested execution. | Move completeness to execution/first acceptance; retain supplied-value typing and source integrity before new consumption. |
| Unnecessary configuration prerequisite | No fragments companion or multiple defaults prevented even interpretation, while FASTQ/BAM/H5AD already supported limited interaction. | Permit empty context, persist ambiguity, never choose the first candidate. |
| Necessary application check | Registration pins bytes; captured InputSet identity and final plan/result checks prevent replacing the selected source or resource after model calls. | Retain metadata/identity guards. They confer no scientific authority. |
| Necessary orchestration check | Registry allowlist, semantic bundles, compiler/preflight, exact `PriorOutputRef` and accepted execution anchors constrain actual execution. | Retain. A provider-authored Execute is not permission to omit resources. |
| Redundant revalidation | Repeated verification-wrapper entry does not imply repeated owner reconstruction. `VerificationContext.verify` reuses compatible operation-local/accepted proofs while rechecking closure and execution anchors. | No newly demonstrated redundant scientific reconstruction to remove. |
| Genuine missing capability | Raw resource/provenance choice continuation, source-bound library-context provisioning, ordinary custom BED qualification, and several scenario specification composers are incomplete Web interfaces. | Document separately; do not remove owner contracts or invent authority. |

Registration, declaration, scientific qualification and accepted authority remain
four different claims. In particular, a fresh external source does **not** need a
historical Agent Run before import. The existing importer can qualify its actual
records under declared source semantics and reviewed reference resources. It
cannot independently reconstruct historical barcode correction or alignment.
Likewise, new H5AD inspection and raw FASTQ/BAM inspection do not require an old
scientific Run. MAESTRO's exact accepted canonical matrix requirement is specific
to that consumer and its counting semantics, not a universal new-file prerequisite.

The inspected owners include `external_fragment_manifest`, `_bam_fragment_io`,
`scatac_reference`, `regulatory_feature_reference`, `annotation_contract`,
`scatac_annotation`, `epizoo_adaptation/resources` and `publication`. The existing
[M11.8 record](m11.8-biological-closeout.md) qualifies the preserved human FASTQ
workflow only; it explicitly leaves real BAM/external/mouse generalization open.
[M14.9](m14.9-neutral-bam-fragments.md) qualifies software/synthetic neutral BAM
behavior. [M14.8](m14.8-real-macaque-fragment-matrix.md) and
[M14.11](m14.11-real-macaque-qc.md) provide real macaque evidence at their exact
resource/source locations. None transfers qualification automatically to an upload.

## C. User-facing bottlenecks

Before this correction, uploading fragments and asking what was needed to process
them returned HTTP 400 `FRAGMENTS_RESOURCE_REQUIRED` without constructing the
Interpreter. An incomplete explicit companion also failed before interpretation,
even when the model could simply clarify the request. Two defaults caused the
same premature rejection under the ambiguity code.

Those requests now return the ordinary asynchronous receipt and reach the existing
typed Interpreter. Its clarification is persisted with the original utterance,
registered source/index identity and supplied context. If it chooses import,
missing required ports still fail before scientific execution. Closed diagnostics
present the compiler's actual missing namespace/profile/reference; incomplete
path/hash bundles retain their existing compiler errors. No provider error text
is parsed to derive a declaration or fix a plan.

This does not yet provide a generic “choose a new reference and resume” turn.
Existing typed pending continuation supports reviewed scientific parameters,
species and source roles; its origin/generation/source/resource guards stay intact.
A generic model clarification is stored conversation, not automatically an
executable pending task. In particular, raw profile/namespace and arbitrary
resource-file choices are not silently added to the reviewed parameter schema.
The remaining integration must explicitly represent those choices before it can
safely resume a resource-incomplete task. Existing supported parameter/species/role
continuations still preserve the original objective through clarification and restart.

## D. Minimal implementation

Four production files change:

- `web/app.py`: one shared `_raw_selection` policy replaces three duplicated
  selectors. Explicit family-compatible context wins; one marked default is
  captured exactly; absent defaults yield empty context; multiple defaults retain
  a closed ambiguity marker. Selection does not assert scientific compatibility.
- `application/uploads.py`: forwards the fragments ambiguity marker through the
  existing registered-input composition boundary.
- `application/local_resources.py`: composition validates supplied declarations
  and exact source bytes without demanding a complete import. The required-input
  set still comes from Registry, checked at actual plan consumption and first
  result acceptance. Even matching absent fields cannot pass the consumption
  guard. Complete historical attribution envelopes remain unchanged.
- `application/dialogue_execution.py`: consolidates the existing BAM/FASTQ
  missing-port presentation branch and adds external import to it. Only closed
  compiler codes and exact tool/port identities are used. It neither repairs a
  plan nor grants a pending resource authorization.

Four test files cover the changed behavior and adversarial consumption cases.
No scientific owner, tool contract, provider prompt/schema, pending-state schema,
reference builder or authority format changes. Source mutation, typed-value
rejection, wrong-family selection and exact resource-choice checks remain enforced.

## E. Reference and raw-data usability

Ordinary users can upload supported fragments and describe a task before a complete
scientific companion exists. A configured, uniquely marked raw default is offered
without exposing its paths/hashes. Compatibility is still established by actual
contracts at execution, not inferred from the label. An explicit incomplete choice
is never silently filled from another default. Multiple defaults remain ambiguous.
Raw defaults are scoped by input family; they are not an assembly-aware reference
search catalog. Language alone cannot resolve multiple raw reference candidates
through the present typed choice interface.

H5AD retains limited inspection without model resources, explicit species
clarification, and selection from the existing qualified EpiZoo catalog. FASTQ/BAM
retain bounded inspection before producer qualification. FASTQ still requires a
complete explicitly attributed collection; an individual read file cannot be
silently treated as a library. These distinctions reflect real source semantics.

The existing `scatac-reference-bundle.v1` builder binds declared human/hg38 or
mouse/mm10 reference content, ordered contigs and BED vocabulary. Its bounded
loader checks manifest identity without re-reading all resource payloads; resource
reinspection is a separate owner operation. BED semantics are zero-based half-open,
with exact `chrom:start-end` identity and preserved order. This does not certify
an arbitrary BED as EpiZoo's canonical vocabulary or authenticate declared assembly.

The neutral `regulatory-feature-reference.v1` builder additionally supports exact
species/taxonomy/assembly declarations and `peak_set`, `curated_ccre` or
`regulatory_regions`; it checks FASTA/FAI correspondence and the ordered BED.
Declarations remain declarations, overlapping features remain separate, and there
is no coordinate repair, sorting, deduplication or inferred liftOver.

Ordinary Web upload currently admits H5AD, FASTQ, BAM and external fragments,
not a BED-plus-genome reference qualification workflow. The missing boundary is
an explicit resource-admission interface collecting the BED, assembly/coordinate
semantics and intended feature identity, binding an already reviewed genome, then
invoking the existing builder/validator and returning an opaque usable choice.
A new reference registry or relaxed BED validator is unnecessary. That interface
is outside this bounded correction; no custom BED acceptance is claimed.

The preserved real-resource readiness remains as reported in UA3.6.4b: human PBMC
FASTQ has qualified original resources, but new uploaded locations need exact
intake/library-context binding and six files exceed the current 256 MiB deployment
limit. Qualified real BAM source/provenance is unavailable. Real macaque fragments
are present, but neutral raw import is a data-layer capability outside the current
public raw importer; exact existing neutral artifacts can already feed downstream
QC/selection/matrix. Removing conversational rejection solves none of those
scientific/deployment gaps by itself.

## F. Tests and measured behavior

Five added application cases failed against the old implementation: four incomplete
contexts were rejected during composition and ambiguity could not be represented.
The initial broader focused run had 67 passes and four test failures: the old
scripted Planner referenced unavailable inputs, producing the correct existing
compiler errors. The fixture now offers only actual input bindings, with separate
partial-bundle expectations. A subsequent targeted run passes nine cases in
3.23 seconds; all diagnostic development logs are retained.

The completed 32-file directly affected regression records **884 passed,
6 skipped, 3 warnings and 1 failed in 1,474.04 seconds**, exit 1. Its only failure
is the stale fragments-selection assertion described below, not a scientific
execution failure. Exact command, environment overrides and selected files are
in `regression-command.json`; full output is in `regression.log`. The warnings
concern existing louvain/pkg_resources deprecations and the unavailable TBB
threading layer. Skipped cases are not counted as qualified coverage.

The one regression failure was an unchanged test in
`test_qualified_resource_choices.py` still requiring fragments ambiguity to raise
before interpretation. It is updated to the same captured ambiguity assertion as
BAM/FASTQ. The complete affected file then passes **7 tests in 2.45 seconds**,
exit 0 (`choices-final.log`). No production change followed the regression run.
The already passing scientific fixtures were not repeated merely for this test
expectation update; this is a completed regression plus a focused correction,
not a claim that the original regression exited successfully.

The directly affected regression covers raw/H5AD composition and Web execution,
explicit/default/ambiguous choices, source/index/reference tampering, typed scalar
and role continuation, species/resource handoff, exact source consumption, accepted
matrix/table delivery, prior-output authority and read-only evidence/dialogue.
Only provider decisions are scripted where required. Small owner fixtures perform
real production and independent verification; EpiZoo fixtures are not a new GPU or
biological model qualification. All `RUN_*` gates are disabled. The full repository
suite was not run.

A separate real-resource HTTP/facade check uploaded the exact M14.8 prepared source
(121,673,085 bytes) and TBI (813,220 bytes) into a new isolated workspace, without
companion declarations. The receipt reached one scripted Interpreter clarification;
retry and a fresh application instance preserved the exact response and source
identity without another completion. Scientific-entry guards recorded **zero**
executions/reconstructions, no scientific Run or Revision, generation zero, and
**zero paid calls**. Source SHA-256 remains
`bde857546f431bb0bb22fe3088f597dfce363bda45f3b7278f5cca56625cc76c`;
TBI SHA-256 remains
`6ebcb3d004f10c54acda8f97d5701b26b19903c3ba94cbd6153582e7f0383342`.
This is real-byte transport/interaction evidence with scripted semantics, not
new-upload scientific qualification or live-model language acceptance. The first
harness attempt failed before upload because its approved directory did not yet
exist; the corrected isolated harness creates it before application construction.

The current explicit-choice and unique-default synthetic HTTP chain tests each
assert exactly one production and one independent owner body for fragments, QC,
selection and matrix. They also check all four schema-2 authorities and exact
matrix/QC/selection downloads, then assert no extra science on retry. These are
current instrumented software checks, not the preserved real macaque experiment.

Cost review distinguishes byte hashing, bounded metadata checks and scientific
reconstruction. Registration/new consumption hashes source pins; checks around
provider calls defend against changed files. Producer publication independently
verifies source conservation, QC verifies metrics, selection verifies decisions,
and matrix verification reconstructs counts. These checks protect different facts.
Accepted proof reuse checks physical/resource closure, runtime compatibility and
accepted execution anchors; standalone verification without such proof remains deep.

Existing measured [M15.6](m15.6-real-multiturn-closeout.md) acceptance records zero
upstream fragment/QC production or owner reconstruction across continuation, with
one owner verification for each new selection/matrix. [M18.5](m18.5-real-interactive-acceptance.md)
records five selection productions and five owner verifications, zero covered
upstream bodies, and no scientific reconstruction on read-only restoration.
Those historical measurements are reused, not rerun or presented as new timings.
Current source inspection confirms `VerificationContext.verify` and
`verification_authority._load_context` implement this behavior. Presentation and
download handling validate exact accepted metadata/pinned bytes; a download is not
a fresh scientific verification. No scientific runtime saving or removal of
independent validation is claimed in this patch.

## G. Architecture review

There remains one Interpreter/Planner path, one existing Session/pending mechanism,
one immutable 23-tool Registry and the existing scientific authority/proof owners.
No new execution engine, resource registry, cache manager, validation service or
scientific implementation is introduced. Arbitrary source claims cannot become
accepted authority. LLMs still select semantics and write evidence-grounded prose;
Agent still controls exact identities, admissibility and evidence binding.

The useful general rule is now consistent across the reviewed input paths:
resource completeness belongs to the requested operation. Receiving and clarifying
a task does not establish its scientific feasibility. Qualified facts and supported
limited operations remain usable while stronger unqualified interpretations stay
explicitly limited.

## H. Next recommendation

The smallest remaining integration is to connect explicit raw context/resource
choices to the existing pending continuation contract, preserving the originating
objective, source/index identity, generation and captured choice. This needs a
reviewed typed choice boundary for reference/profile/namespace and source-bound
library-context construction through existing owners. It must not allow user prose
to fabricate historical correction or accepted source qualification.

For subsequent UA3.6.4b real raw acceptance, the preserved human PBMC FASTQ route
has the strongest existing production evidence. Reuse its real files and qualified
reference/tool resources; admit the complete relocated library under its actual
intake and library binding before paying for models or running science. Deployment
upload-limit adjustment is also required for that route. Existing accepted artifacts
can support cheaper downstream acceptance, but cannot satisfy new raw-to-matrix
acceptance by substitution. Public neutral raw import is a separate contract-wiring
gap, not a reason to change species declarations. No new download is needed here.

MAESTRO can later reuse the same explicit-choice interaction boundary while composing
its existing exact matrix/group/context specification. Adaptation similarly needs
safe selection of existing strategy/profile/resource specifications. Both retain
their owner-specific scientific checks; a generic workflow or qualification engine
would add no demonstrated value. Custom BED admission is an optional separate
interface extension, not a blocker to using existing qualified references.

This bounded simplification and its verification are complete. The live dialogue
compliance gap in UA3.6.4a and new real raw workflow gap in UA3.6.4b remain unchanged.
