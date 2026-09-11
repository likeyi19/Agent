# Agent

Agent is an autonomous AI agent for single-cell epigenomic / scATAC-seq
analysis. It turns natural-language requests and structured scientific inputs
into validated workflows, reproducible artifacts, and verified reports.
Existing foundation models such as EpiZoo and EpiAgent are scientific backends
to reuse; the Agent does not reimplement them. EpiZoo is the currently validated
embedding backend.

Milestones 1–9 and Post-M9 Planner Interface Hardening are complete, including
static target-port/Groq compatibility, feature-space semantic-v4 parity, and
the migration to v4 as the default LLM planning wire.
Guarded real-data differential-accessibility acceptance remains outstanding.
[AGENTS.md](AGENTS.md) contains the detailed engineering rules, scientific
contracts, milestone history, acceptance results, and deferred work.

**Milestone 10 — Raw scATAC Data Intake & Preflight is complete.**
Natural-language requests now support raw FASTQ/BAM intake → deterministic
preflight → verified manifest/evidence → figureless deterministic report.
M10 identifies the supplied input, whether its structure/provenance is
interpretable, and whether it satisfies the intake contract for its intended
preprocessing route.

Phase II began with raw scATAC intake capabilities. Milestone 10.1 established
the versioned raw-scATAC intake domain contract
and manifest infrastructure. Raw-input readiness is distinct from execution
success: the contract separates required information, preparation/repair
requirements, and normal downstream prerequisites.

Milestone 10.2 is complete: a deterministic, read-only FASTQ intake vertical
slice built on M10.1. It inspects local plain/gzip FASTQ with bounded parsing,
accepting `.fastq`, `.fastq.gz`, `.fq`, and `.fq.gz` for Agent inspection.
Discovery supports explicit files and non-recursive directories. Reviewed
Illumina/10x-style names support deterministic sample-token/lane/chunk/read-role
grouping without establishing biological sample identity.

The first supported declared assay family is 10x / Cell Ranger ATAC-compatible
FASTQ, with two layouts:

| Layout | Genomic reads | Raw cell-barcode read | Optional sample index |
| --- | --- | --- | --- |
| A | R1, R3 | R2 | I1 |
| B | R1, R2 | I2 | I1 |

Assay identity is not inferred from filenames or read lengths alone. Barcode
provenance requires supported assay/layout evidence; generic or ambiguous FASTQ
remains inspectable without inferred barcode semantics.

Milestone 10.3 is complete: deterministic, read-only BAM intake supports explicit
local `.bam` files and non-recursive directories. It uses mature `pysam`/htslib
parsing, with no custom BAM decoder. The tested optional dependency is
`pysam==0.24.1`, declared in [requirements-bam.txt](requirements-bam.txt) and
lazy-loaded for BAM inspection; existing FASTQ/H5AD workflows do not require it.
Bounded sequential reading works without an index and collects compact
header/reference, species/source-assembly, sort/index, mapping/pairing, and
barcode-tag evidence. Missing indexes alone do not make inputs invalid.

Usable sampled `CB` may establish a BAM cell identifier; `CR` supplies raw
cellular-barcode evidence and `CY` its quality evidence. Generic CB is not
assumed corrected. `BC`/`QT` describe sample/library barcodes and qualities,
never cell identity. Barcode values, read sequences, qualities, and per-read
coordinates are not persisted.

Human model-oriented preprocessing targets `hg38`; mouse targets `mm10`.
Source BAM assembly remains distinct from target assembly and is never inferred
merely from species or chromosome naming. A mismatch does not establish a
supported conversion/realignment route. FASTQ has no source coordinate assembly;
FASTQ alignment targets the appropriate reference directly. Inspection is
sample-scoped unless sequential EOF observation permits complete coverage and
does not certify uninspected content.

Milestone 10.4 is complete: M10.1–M10.3 intake is now exposed through one public
capability, `inspect_raw_scATAC`. Natural-language LLM planning selects this tool;
the compiler binds structured request values without putting their filesystem
paths in the model prompt. Private FASTQ/BAM helpers are not Planner tools.
Agent code owns format dispatch, FASTQ layout and BAM tag/header interpretation,
source/target assembly semantics, manifest publication, readiness, and verification.

Raw sequencing has planning semantics distinct from processed-H5AD `RAW_SCATAC`.
The result is a versioned authoritative manifest reference and compact summary.
It cannot feed EpiZoo directly: preprocessing must first construct the required
cell-by-cCRE/H5AD input. Execution success is separate from readiness (`READY`,
`READY_WITH_REPAIRS`, `NEEDS_USER_INPUT`, `UNSUPPORTED`, or `INVALID`). Dedicated
verification independently reloads the manifest and reobserves current sources.

PLAN_ONLY performs zero raw inspection and writes no manifest; BAM planning
does not load pysam. Verified Runtime EXECUTE and durable resume are supported.

Milestone 10.5 is complete: FASTQ and BAM intake now support the full Application
path from natural-language planning → `inspect_raw_scATAC` → source-aware
verification → evidence → deterministic verified Markdown report. Raw-intake
reports are figureless by design; no scientific visualization is generated.
Reports distinguish successful Agent execution, successful source verification,
and preprocessing readiness. A truthful non-ready or `INVALID` finding can still
be a successful Application execution.

Reports summarize input kind, readiness, file/group counts, species, separate
source/target assemblies and compatibility, harmonization requirements, barcode
provenance/scope, structured issue codes, required information, preparations,
normal downstream prerequisites, and inspection coverage. Bounded/sample-scoped
inspection is distinguished from complete sequential inspection and is not
whole-file certification. Evidence freshly verifies sources before projection;
verified manifest provenance and SHA-256 identify the authoritative intake artifact.
Evidence/reports contain no whole raw manifest, reads, read names, barcode values,
qualities, complete BAM reference dictionaries, or raw sequencing payloads.

M10 provides intake/preflight; M11.2 adds FASTQ and M11.3c adds qualified BAM
preprocessing below. Sort/index observations alone do not authorize transformations.
M11.5c now adds cell-by-cCRE construction. Realignment/liftOver, automatic
statistical cell calling and model inference from raw sequencing remain deferred.

## Milestone 11 status

**Milestone 11.6 — Production Reference & Runtime Resource Provisioning is complete.**
The accepted slices are [M11.6a — production resource specification and operator
workflow](docs/m11.6a-production-resource-specification.md), [M11.6b — production
parent/QC resources and qualification](docs/m11.6b-production-qc-resources.md), and
[M11.6c — full-reference Chromap resources and runtime
handoff](docs/m11.6c-production-chromap-resources.md).

Authenticated human/hg38 and mouse/mm10 parents retain full ordered cCRE
vocabularies of 1,355,445 and 1,341,077, respectively. Production-qualified GENCODE
49/human and M25/mouse QC resources share an immutable two-species catalog.
Qualified QC and matrix BEDTools runtimes, the durable qualified patched Chromap
executable, and both full-reference Chromap indexes complete the existing handoff:
`AGENT_QC_RESOURCE_CATALOG`, `AGENT_QC_BEDTOOLS`, `AGENT_MATRIX_BEDTOOLS`,
`AGENT_CHROMAP_BIN`, and `AGENT_CHROMAP_INDEX_ROOT`.

The accepted GTF-only 4 GiB annotation-source guard changes only an engineering
bound; other QC resources/sidecars retain 2 GiB and QC scientific semantics are
unchanged. Full lightweight acceptance passed 3,579 tests, with 83 skipped and
7 existing warnings. Both indexes were constructed on the current lab host:
human native size ≈12.55 GB, mouse ≈11.78 GB; human child RSS was observed at
≈26.1 GiB. Mouse RSS was not isolated. These are operator observations, not
universal hardware requirements or throughput qualification.

M11.6 establishes production resource readiness and runtime qualification, not
real biological raw-data acceptance, biological preprocessing correctness,
throughput qualification, automatic statistical cell calling, doublet detection,
FRiP, peak calling or biological model-readiness. Selection retains
`cell_call_method=none`, `cell_call_state=not_assessed`, and
`selection_method=explicit_qc_thresholds.v1`. Registry remains 18; Planner,
semantic compiler, v3/v4 wire schemas, Application, fragment science, M11.4
QC/selection science and M11.5 matrix science are unchanged. Provisioning adds
no Planner-visible scientific tool.

Next: **M11.7 — Real-Data End-to-End Production Acceptance** tests the accepted
software/resources on real biological data at real scale. **M11.8 — Biological
Validation & Raw Preprocessing Closeout** retains biological validation. Neither
milestone is implemented by this closeout.

**M11.1 is complete: reference identity and library/barcode processing contracts.**
`ScATACReferenceBundle` identifies the genome and full ordered cCRE vocabulary:
1,355,445 features for human/hg38 and 1,341,077 for mouse/mm10.
`LibraryProcessingContext` binds selected M10 intake groups to explicit processing
libraries, barcode namespaces, interpretation/correction policies, and whitelist
identities. These artifacts are intentionally separate: library/barcode decisions
do not select a genome reference.

M11.5a explicitly supersedes M11.1's proposed support-weighted matrix semantics.
Human and mouse use canonical fragment-record counts: every canonical fragments-v2
record contributes one to every distinct cCRE with positive-base overlap, ignoring
support and strand. Repeated contributions sum without binarization or new
deduplication. Support has producer-specific meanings and is not a universal
biological weight. Full reference order and zero columns remain present; selection
owns exact rows, including zero rows and empty selection. EpiZoo filtering stays
downstream, retaining 700,460 human or 814,020 mouse features.

**M11.5a freezes matrix science/artifacts and narrow BEDTools qualification.**
At that milestone no matrix-building tool or Planner/Application integration was
added; the registry remained at 17 tools. The project-authoritative mouse full vocabulary is provisioned
deterministically from the exact Fang2021 H5AD ordered feature names into the
unchanged reference interface, with a hash-bound derivation receipt. See the
[M11.5a contract and acceptance](docs/m11.5a-matrix-contract.md).

**M11.5b adds bounded data-layer matrix construction and complete independent
verification.** Exact verified fragments, selection and reference produce one
ordered int64 CSR H5AD. SQLite-backed BEDTools incidence counting and a separate
augmented interval-tree verifier preserve full axes and zero rows/columns. This
data-layer milestone did not add a registered tool. M11.5c now supplies
Planner/Application, reporting and durable lifecycle integration. See [implementation and evidence](docs/m11.5b-matrix-construction.md).

**M11.5c integrates the full matrix into Agent.** The eighteenth registered tool,
`build_scATAC_cell_by_ccre`, consumes exact fragments-v2, selection-v1 and reference
manifest path/SHA pairs. Existing artifacts and same-plan upstream outputs compose
through reviewed semantic ports:

```text
raw input → verified fragments → QC → selected cells → full cell-by-cCRE matrix
```

The tool reuses the frozen constructor and independent verifier, publishes an
immutable H5AD with a durable receipt, and supports resume, cancellation, bounded
evidence and figureless reports. Empty selections and zero rows remain valid.
`AGENT_MATRIX_BEDTOOLS` configures the explicitly qualified executable; existing
QC runtime/resource qualification remains mandatory. PLAN_ONLY reads no scientific
payloads and performs no matrix work. See [the API and closeout](docs/m11.5c-agent-integration.md).

The full mouse 1,341,077-column component acceptance remains valid. A complete
production mm10 QC → selection → full matrix run remains operational acceptance
work. M11.6b now supplies its operator-qualified mm10 QC bundle/catalog dependency.
Automatic statistical cell calling, doublets, FRiP filtering, peak calling,
raw-derived EpiZoo preprocessing/inference and biological throughput qualification
are separate future work.

**M11.2 is complete: FASTQ → verified canonical fragments.**
`prepare_scATAC_fragments` consumes a validated intake, library-processing context,
and reference bundle. Human FASTQ targets hg38; mouse FASTQ targets mm10. The
patched/pinned Chromap backend preserves exact duplicate-group support. Output is
canonical per-library v2 BGZF fragments with tabix indexes and an independently
verified manifest. Identity remains `(namespace, barcode)`; observed fragment
barcodes are not called cells.

Natural-language Application execution produces freshly verified evidence and a
deterministic figureless report. PLAN_ONLY needs no preprocessing executables;
resume and exact publication recovery avoid another alignment. New execution
requires an explicitly configured qualified backend (`AGENT_CHROMAP_BIN`) and
matching index catalog (`AGENT_CHROMAP_INDEX_ROOT`); production full-genome index
provisioning and biological FASTQ acceptance remain deferred. No automatic index
build or backend fallback occurs.

**M11.3d converges all three fragment routes on v2.** FASTQ, qualified BAM,
and adopted external fragments publish `agent.scatac-fragments`, schema 2,
`scatac-fragments.v2`. The existing FASTQ tool retains the qualified Chromap
science and complete M11.2 provenance in a bound producer record. Its recovery
policy is now `prepare-scatac-fragments-fastq-v2`.

The common freshly verified v2 streaming interface preserves reference,
namespace/barcode identity, exact integer support, optional strand and producer
provenance. Generic content/resource verification does not qualify producer
science; each producer retains its own verification and support meaning.
There is no downstream format branch by producer origin.

This is an intentional pre-release compatibility break. Fragments v1 artifacts
and their persisted runs/receipts are unsupported by the current runtime;
there is no automatic upgrade or dual recovery path. Git history and the
historical M11.2/M11.3a–c records preserve the former v1 behavior. See the
[M11.3d contract and M11.4 input boundary](docs/m11.3d-fragment-convergence.md).

**M11.4a freezes QC science and resources.** Immutable QC reference bundles,
deterministic transcript-TSS construction, independent source reinspection, and
narrow local pysam/BEDTools qualification are available as data-layer APIs.
These APIs add no tools. Production hg38/mm10 annotation provisioning remains
deferred. See [the frozen QC contract](docs/m11.4a-qc-resources.md).

**M11.4b adds verified per-barcode QC.** `compute_scATAC_qc` publishes
`scatac-barcode-qc.v1` for every observed `(namespace, barcode_identifier)`,
in exact ASCII tuple order. All three qualified fragment producers use the same
unweighted canonical-record depth, fixed endpoint/TSS incidence, and fragment-length
metrics. Exact rational values and stable undefined reasons remain in a compressed
table; a bounded whole-artifact length histogram accompanies it. No cell calling
or selection is performed by this QC step.

Production uses the explicitly configured qualified `AGENT_QC_BEDTOOLS` runtime;
an independent verifier reconstructs every row without the BEDTools intersection.
An independently provisioned operator qualification catalog
(`AGENT_QC_RESOURCE_CATALOG`) must approve the exact production QC resource.
Tiny synthetic bundles require explicit test opt-in (`AGENT_QC_ALLOW_SYNTHETIC=1`).
No canonical biological hg38/mm10 QC bundle is supplied by this milestone.
Durable recovery and figureless Application evidence/reporting freshly verify QC.
See [the M11.4b contract and API](docs/m11.4b-barcode-qc.md).

**M11.4c adds explicit QC selection.** `select_scATAC_cells` consumes one exact
verified QC artifact and caller-supplied thresholds, publishing
`scatac-cell-selection.v1`. Every observed barcode gets a decision and all applicable
failure reasons. Exact rational comparisons, stable undefined-metric behavior,
reversible rendered IDs and ordered selected identities provide the M11.5 row
handoff. Empty selection is valid. Selected barcodes are **QC-selected candidate
cells**; statistical background-versus-cell calling is **not assessed**.

M11.4a, M11.4b and M11.4c together complete M11.4 under the explicit-QC-selection
scope: fragments v2 → barcode QC → optional explicit QC selection → ordered candidate
set. M11.4c brought the registry to 17 tools; M11.5c adds matrix construction as
tool 18. Metrics-only QC remains valid; selection is never inserted automatically.
Automatic calling, doublets, FRiP and biological cell-calling validation remain deferred.
See [the M11.4c contract, API and M11.5 handoff](docs/m11.4c-explicit-selection.md).


**M11.3b is complete: verified external-fragment adoption.** The new
`import_scATAC_fragments` tool accepts the explicitly selected
`10x-atac-fragments.v1` profile: five-column fragments without strand or
six-column fragments with preserved strand, in plain text, gzip or BGZF.
Source SHA-256, reference bundle and processing namespace are explicit; an
optional source TBI must be explicitly bound and verified. Adoption preserves
coordinates, identifiers and exact read-pair support while sorting and publishing
canonical v2 BGZF/TBI artifacts. Independent verification reparses the complete
source and proves record conservation; historical alignment, Tn5 adjustment,
deduplication and barcode correction remain declared, not reconstructed.
External adoption has reviewed semantic planning metadata and
durable recovery and verified figureless Application reporting. Synthetic
acceptance passed; biological acceptance remains deferred to M11.8. See the
[M11.3b contract](docs/m11.3b-external-fragments.md).

**M11.3c is complete at synthetic acceptance: qualified BAM → fragments v2.**
`prepare_scATAC_bam_fragments` supports only `agent-cb-paired-atac.v1`: one BAM,
selected group, processing library and namespace, explicit corrected `CB`, exact
human/hg38 or mouse/mm10 reference, and caller-declared unshifted coordinates and
retained duplicate pairs. The pinned pysam/htslib adapter requires neither source
coordinate sorting nor an index. Both mates require MAPQ ≥30 excluding 255;
Agent applies +4/−5 once and sums individually eligible pairs only at identical
namespace/barcode/endpoint keys, including duplicate-marked pairs. All exact FAI
contigs remain eligible; strand is absent. This is not Cell Ranger reproduction.

Complete source identity, independent BAM transformation verification, v2 output
integrity, exact durable recovery and figureless Application reporting are
supported. Historical alignment/correction and source-history declarations are
not independently proven. All 18 scientific tools have semantic metadata;
PLAN_ONLY performs no BAM IO. See the [M11.3c contract](docs/m11.3c-bam-fragments.md).
Biological acceptance remains deferred to M11.8.

CRAM/SAM support, automatic statistical cell calling and raw-derived EpiZoo
inference remain unimplemented. See the
[M11.3a contract](docs/m11.3a-fragment-boundary.md) and detailed
[M11.2 contract](AGENTS.md#milestone-112--fastq-to-canonical-fragments-and-joint-closeout).

## Architecture

```text
Planning → Orchestration → Scientific Tools → Foundation Models → Verified Output
```

These five functional areas share persistence, recovery, cancellation,
execution traces, provenance, verification, and auditability. Planning produces
an `AgentPlan`; `AgentRuntime` and `PlanExecutor` preflight and sequentially
execute only tools in the immutable `ToolRegistry`. Tools reuse scientific
backends where needed and return lightweight results bound to artifacts.
Providers generate plans only and receive no scientific Python callables,
filesystem access, or run-store access. Arbitrary Python and shell execution
are prohibited in Agent plans.

For supported reporting workflows, the application composes fresh verified evidence,
supported deterministic visualizations, and a deterministic scientific report.
These are post-run services, outside `AgentPlan` and the scientific registry.
The same runtime remains the only scientific execution engine.

## Planning: choices, interfaces, and execution

**LLM owns choices; Agent owns facts and deterministic consequences of choices.**
The LLM/user owns intent, tool selection, workflow/DAG composition, scientific
choices, and genuinely ambiguous source, producer, source-port, or parameter-
scope decisions. Agent code owns registered interface facts, legal semantic
ports, authorized deterministic request binding, exact argument/result mappings,
`StepOutputRef` construction, induced dependencies, defaults, canonicalization,
validation, and whole-plan preflight.

The default LLM semantic wire-v4 path is:

```text
AgentRequest
→ registry-driven semantic planning catalog/prompt
→ provider-neutral PlanningModel
→ semantic wire v4
→ strict parser
→ SemanticPlanCandidate
→ registry-derived deterministic semantic compiler
→ AgentPlan
→ whole-plan preflight/runtime
```

The compiler deterministically lowers semantic choices into executable plan
contracts. It derives only unique mappings explicitly authorized by reviewed
`ToolSpec.semantic_planning` metadata. Zero or multiple legitimate semantic
choices fail closed when a choice is required. It never fills in a workflow
using step names/order, first-match behavior, generic input fanout, hidden
scientific inference, or automatic workflow completion. Scientifically valid
noncanonical DAGs remain allowed. Executable values come from structured
`AgentRequest.inputs` or verified upstream references; the LLM cannot invent
paths, parameters, or executable literals. Structured input values are excluded
from the planning catalog/prompt; input names and basic types may be exposed,
and the natural-language request itself is sent to the model.

Semantic **v4 is the default for LLM planning**, removing model-authored execution
argument dictionaries, raw result keys, references, and redundant dependency
serialization. Wire **v3 remains an explicit compatibility mode** through
`--wire-mode v3` or `PlanningWireMode.V3` in Python. Its registry-derived schema
retains exact keyed argument bindings. There is no automatic schema switching,
version detection, combined schema, or hidden v4-to-v3 fallback.

The static target-port follow-up projects each tool's legal target ports from
`ToolSpec.semantic_planning.consumer_ports` into the v4 provider schema:

```json
{
  "target": "dataset",
  "source": {"kind": "input", "input": "input_path"}
}
```

The closed outer object constrains the selected tool's target; inner source
variants discriminate only on `kind` (`input`, `step`, or `step_port`). This
resolves Groq's `discriminator_multiple_candidates` rejection of the earlier
flat target/kind alternatives. The parser validates targets early and still
accepts historical flat v4 sources through the same strict checks. The compiler
retains authoritative `UNKNOWN_TARGET_PORT` defense. The registry remains the
single semantic authority; this fix adds no workflow inference or planner layer.

`PlanningModelProfile` and adapter-only `PlanningModelFactoryRegistry` separate
configuration from planning. OpenAI, Gemini, Groq, and custom `PlanningModel`
injection are supported; no production model is hard-coded. Application/CLI new
runs require an explicit primary LLM profile unless deterministic planning or
another Planner is explicitly selected. Missing configuration fails clearly.
Low-level `AgentRuntime()` retains its deterministic offline default; that
runtime construction is separate from normal application LLM planning. The
deterministic planner is not a semantic oracle for LLM output.

## Scientific capabilities

The current inventory below comes from
[`build_default_tool_registry()`](src/agent/orchestration/registry.py).
Planner-visible coverage is registry-derived, not a permanent tool-count limit.

| Workflow | Registered tools and accepted behavior |
| --- | --- |
| Raw sequencing intake | `inspect_raw_scATAC`: bounded FASTQ/BAM inspection, authoritative intake manifest, source-aware verification, and full Application execution with a verified figureless report |
| FASTQ preprocessing | `prepare_scATAC_fragments`: canonical per-library BGZF/tabix fragments, exact support, independent verification, durable recovery, and figureless Application reporting |
| External fragment adoption | `import_scATAC_fragments`: explicit 10x fragment semantics, complete source validation and conservation, canonical v2 artifacts, durable recovery, and figureless Application reporting |
| Qualified BAM fragments | `prepare_scATAC_bam_fragments`: explicit corrected-CB profile, complete BAM identity, independently recomputed exact-key support, strand-absent v2, durable recovery and figureless reporting |
| Inspect and embed | `inspect_scATAC`, `epizoo_embed_cells`: safe H5AD inspection, validated sparse preprocessing, process-local EpiZoo model reuse, 512-dimensional embeddings plus ordered cell IDs |
| Downstream embedding analysis | `build_cell_neighbors`, `cluster_cells`, `compute_cell_umap`: compact copy-on-write H5ADs with sparse graphs, weighted Leiden labels, and 2D UMAP |
| Clustering evaluation | `evaluate_cell_clustering`: NMI, ARI, AMI, and Homogeneity for fixed clustering; arithmetic averaging for NMI/AMI |
| Cell annotation | `transfer_cell_labels`: exact deterministic CPU kNN transfer directly between within-species reference/query EpiZoo embeddings using the same canonical checkpoint |
| Annotation evaluation | `evaluate_cell_annotation`: fixed-prediction assignment rate, overall/assigned accuracy, macro-F1, per-class diagnostics, rectangular confusion counts, and descriptive confidence medians |
| Regulatory feature foundation | `validate_scATAC_feature_space`, `build_replicate_pseudobulk`: explicit raw sparse feature provenance and exact SUM by `(group, replicate, condition)` |
| Differential accessibility | `run_replicate_differential_accessibility`: biological-replicate DA with pinned edgeR v4 quasi-likelihood fitting/testing and independent verification |

EpiZoo embedding requires its species-specific raw feature layout: 1,355,445
features for human or 1,341,077 for mouse, with retained cCRE names/order matching
the local EpiZoo frequency/filter resources. Input must contain compatible
sparse, finite, nonnegative count-like values. Arbitrary peak matrices are not
automatically projected or converted. The checkpoint and local resources must
be available; detailed prerequisites are in [AGENTS.md](AGENTS.md).

The registered embedding tool defaults to `device="cuda:0"` and permits explicit
checkpoint/device selection. It fixes batch size 4, maximum sequence length
8192, truncation seed 0, random sampling, and requested AMP; these settings are
not planner-configurable. The validated GPU path is the RTX 4090 acceptance
described below; unavailable CUDA does not trigger a CPU fallback.

Neighbors use all 512 EpiZoo dimensions, `n_neighbors=15`, Euclidean distance,
and seed 0. Leiden defaults to weighted igraph flavor, resolution 1.0, seed 0;
UMAP uses two dimensions, `min_dist=0.5`, `spread=1.0`, spectral initialization,
and seed 0. Compact downstream artifacts never contain the original raw scATAC
feature matrix, and source inputs are never modified.

Transfer defaults to exact Euclidean kNN (`k=20`), uniform plurality voting,
and confidence threshold 0.0. Distance ties use reference row order; tied top
votes remain structurally unassigned with confidence retained. No approximate
neighbors, automatic k reduction, batch correction, clustering, or UMAP enters
transfer. Query ground truth is unavailable to the production transfer path.

Evaluation ground truth is used only after clustering or annotation is fixed;
it never tunes parameters, selects a workflow, or triggers upstream reruns.
Exact cell identity and order are required without intersection/reordering.
Annotation evaluation counts unassigned cells as incorrect for overall accuracy,
uses ground-truth classes for macro-F1, and preserves undefined assigned accuracy
or confidence summaries as `null`. It does not optimize confidence thresholds.

Regulatory analysis returns to raw sparse `X` or an explicitly named sparse
layer. Feature validation requires declared fragment/insertion counts or binary
accessibility; normalized/continuous input is ineligible. Supported assemblies
are human/hg38 and mouse/mm10. Coordinates are optional and never inferred.
Pseudobulk preserves original features and exact integer SUMs without filtering
or normalization; groups come from raw metadata or an exactly aligned fixed
annotation with every cell assigned. Independent verification recomputes SUMs
with a distinct Python row-map algorithm without whole-matrix densification.

DA accepts verified SUM-count pseudobulk and never treats cells as replicates
or silently uses binary accessibility in a count model. Independent designs
need at least two disjoint biological replicates per condition (warning at two);
paired designs need three complete pairs. The fixed numerator-minus-denominator
contrast supports ordered additive categorical/numeric covariates. The isolated
`agent-edger` runtime pins R 4.6.1 / Bioconductor 3.23 / edgeR 4.10.4 and performs
condition-based `filterByExpr`, library-size recalculation, TMM, robust v4
quasi-likelihood testing, and BH correction. Statistical settings and R scripts
are repository-controlled. A separate pinned R verifier independently checks
statistics, preparation, provenance, and package compatibility.

DA execution requires `AGENT_EDGER_RSCRIPT` set to an absolute path resolving
to the executable `Rscript` in the intended pinned `agent-edger` environment.
This is runtime configuration, not an `AgentRequest` scientific parameter;
there is no automatic environment discovery.

## Reliability and verified output

Whole-plan preflight occurs before scientific side effects. Each returned
result passes verification before downstream use. PLAN_ONLY executes **zero
scientific tools**, including across restart/resume/cancellation, and the
application creates no evidence, figures, or report for it.

Verification coverage is specific to each tool/artifact. Standalone embedding
verification checks result metadata and artifact existence/non-emptiness; it
does not reload or hash the embedding and ordered-ID contents. Later tools add
their defined content checks. Fresh verification does not establish universal
content integrity; [AGENTS.md](AGENTS.md) records the exact boundaries.

`FileRunStore` persists versioned canonical JSON with SHA-256 integrity, plan
fingerprints, optimistic revisions, atomic fsynced replacement, a short state
lock, and a separate execution lease. Verified successes are checkpointed before
downstream execution. Durability is opt-in for low-level `AgentRuntime` and
owned by `ResearchAgentApplication`. Nonterminal resume is planner-free and
revalidates completed work before restoring references. Terminal runtime resume
returns the immutable stored result; evidence/application composition freshly
verifies artifacts afterward. Unknown stale RUNNING work becomes INTERRUPTED
without automatic rerun. Valid terminal legacy v1/v2 records remain readable;
legacy nonterminal EXECUTE work without authoritative recovery provenance cannot
start new science. Current run-state schema is v3.

Cancellation intent uses a separate durable sidecar without taking the execution
lease or changing the main revision. Running calls finish, verify, and checkpoint;
once cancellation is observed no new attempt starts. Duplicate cancellation is
idempotent, terminal states are immutable, and prior failure evidence is retained.

Scientific same-step retry stays in `PlanExecutor`: each attempt gets a fresh
canonical-equivalent argument copy, with no changed scientific settings.
`AgentError.recoverable` means static retry eligibility only. Versioned immutable
recovery-policy provenance blocks incompatible resume. Unknown error codes and
verification failures fail closed; raw exception prose is sanitized. The
downstream M6/M8 tools have no automatically retryable scientific codes.

Planning recovery separately permits one initial provider call, either one
same-profile transport retry or complete plan repair, and at most one explicitly
configured final-profile failover: three logical calls maximum, with built-in
SDK retries disabled. Interrupted planning is not replayed; only the final
preflight-passing plan is durable. HTTP 413 is terminal
`PROVIDER_REQUEST_TOO_LARGE`, distinct from retryable HTTP 429.
Sanitized planning diagnostics use schema v3 for wire v3 and schema v4 for wire
v4. The existing planner benchmark remains explicitly pinned to wire v3; its
report schema v4 preserves historical scoring. These diagnostics and reports expose
attempt provenance and distinguish hard semantic correctness from canonical
workflow conformance and first-attempt versus recovered success.

Post-run schema-v1 evidence contains whitelisted verified facts and an
authoritative evidence-file SHA, explicitly distinguishing source digest
protection from structural/provenance/content verification. Supported PNGs are
Leiden UMAP, the four clustering metrics, and raw annotation confusion counts.
Deterministic Markdown reports use attributed frozen facts, preserve exact and
nullable values, and optionally copy verified PNG bytes unchanged. No LLM
narrative, invented analysis stage, visual interpretation, or qualitative
biological claim is generated. Inspection, pseudobulk, and DA can produce valid
figureless reports; failure of an expected visualization remains fatal.

## Usage

Run from the repository using the configured Python environment. Packaging and
an installed console script remain deferred. API keys stay in the environment.
For explicit offline inspection planning:

```bash
PYTHONPATH=src python -m agent run \
  --request-id inspect-demo --request "Inspect this scATAC dataset" \
  --workspace /path/to/workspace --input /path/to/cells.h5ad \
  --planner deterministic --plan-only
```

For default semantic v4 LLM planning (the model shown is a prior
Groq acceptance configuration, not an automatically selected default):

```bash
PYTHONPATH=src python -m agent run \
  --request-id inspect-v4 --request "Inspect this scATAC dataset" \
  --workspace /path/to/workspace --input /path/to/cells.h5ad \
  --provider groq --model openai/gpt-oss-120b --plan-only
```

Omit `--wire-mode` for v4, or select `--wire-mode v3` for compatibility. Explicit
`--wire-mode v4` is also supported. `LLMPlanner(model)` defaults to v4, as does
application-owned LLM planning when `planning_wire_mode` is omitted. Optional
`--secondary-provider` and `--secondary-model` configure the single final
failover. `--provider deterministic` remains a compatibility alias for explicit
deterministic planning; deterministic mode rejects LLM wire/model settings.
Use `--inputs-json /path/to/inputs.json` for additional structured scientific
inputs; embedding also accepts `--species`, `--checkpoint`, and `--device`.
Remove `--plan-only` to execute and compose supported verified outputs.

Resume or cancel using the `run_id` returned in the compact CLI JSON:

```bash
PYTHONPATH=src python -m agent resume --workspace /path/to/workspace --run-id RUN_ID
PYTHONPATH=src python -m agent cancel --workspace /path/to/workspace --run-id RUN_ID
```

Neither operation needs planning/provider configuration. The Python service,
`ResearchAgentApplication(workspace_root, ...)`, exposes the same `run(request)`,
`resume(run_id)`, and `cancel(run_id)` operations with typed configuration and
compact `ArtifactReference` results. Full signatures and CLI exit codes are in
[AGENTS.md](AGENTS.md).

Managed workspaces contain `run_state/` and `runs/<full-sha256-of-run-id>/`, with
`composition.lock`, `scientific/`, `evidence/`, `visualizations/`, and `report/`.
Output roots are application-owned; raw IDs never form paths, managed symlinks
are rejected, and a per-run composition lock protects postprocessing. Resume
verifies/reuses valid outputs, builds missing stages, and rejects tampering or
partial/conflicting outputs without silent overwrite or repair. The workspace
is trusted and local; full hostile-filesystem-race protection is not claimed.

The canonical application demo is inspection → EpiZoo → neighbors → Leiden →
UMAP → verified evidence → Leiden UMAP figure → deterministic report. The richer
reference/query annotation workflow is also available.

## Validation and current boundaries

Real Fang2021 acceptance established exact manual EpiZoo parity for 2,000 cells,
`(2000, 512)` embeddings, fixed-seed reproducibility, batch size 4, and about
10.9 GiB peak GPU allocation on an RTX 4090. Downstream analysis and held-out
annotation/evaluation were validated with unchanged sources and isolated
evaluation-only labels. Development assumes at most 24 GB GPU memory and never
densifies a complete raw scATAC matrix.

Feature-space semantic-v4 parity completed six optional mappings: `layer_key`,
`feature_chrom_key`, `feature_start_key`, `feature_end_key`, `coordinate_system`,
and `semantics_metadata_key`. Its acceptance recorded 378 focused and 1119
broader passes, plus gated DA PLAN_ONLY acceptance. Live Groq explicit-v4
checks passed inspection, the five-tool downstream workflow, and the
parameter-heavy feature-space case.

The subsequent v4-default migration recorded 821 focused and 1162 broad passes
with 3 skips, plus a separate gated DA PLAN_ONLY pass. Groq
`openai/gpt-oss-120b` passed the same three PLAN_ONLY cases while omitting
`--wire-mode`; diagnostics confirmed v4. All these live PLAN_ONLY attempts
executed zero scientific tools. Explicit v3 reached Groq with an unchanged
request contract but received HTTP 413 / `PROVIDER_REQUEST_TOO_LARGE`: an
observed provider limitation, not a default-switch regression. V3 remains
explicit compatibility mode with offline acceptance.

These are recorded acceptance results, not a single current full-suite total.
Earlier static target-port/Groq acceptance counts and live checks remain in
[AGENTS.md](AGENTS.md) as historical records.

The mechanical serialization burden, static target-port generation, and Groq
discriminator issues have been addressed. Genuine hosted-model source/source-port
variability, incomplete or explicit unsupported decisions, and provider
availability/rate-limit/authentication issues remain possible;
remaining variability must not be hidden by deterministic workflow guessing.

Guarded real-data DA acceptance needs eligible human/mouse raw counts with true
replicated conditions. Local Fang2021/PBMC lack the design, BMMC is normalized
mixed-modality data, and replicated rice heat-shock data is outside scope; no
metadata was fabricated or external data downloaded. Broader statistical models,
regulatory interpretation, genomic/peak-to-gene annotation, motifs/pathways,
volcano/MA plots, perturbation, and mutation analysis remain deferred. So do
reporting-stage cancellation, transferred-label UMAP, richer figures/exports,
LLM scientific interpretation, retrieval/RAG, multi-agent architecture, and
browser or multi-turn UI. The detailed acceptance gates, environment warnings,
and nonblocking engineering follow-ups are preserved in [AGENTS.md](AGENTS.md).
