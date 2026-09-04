# Agent

An autonomous AI agent for single-cell epigenomic analysis powered by epigenomic foundation models.

Milestone 2 provides safe scATAC inspection, artifact-based EpiZoo cell embedding, and process-local model reuse.

## Current status

Milestones 1–5, Milestones 6.1–6.4, Milestones 7.1–7.4, and Milestones 8.1–8.2
are complete. Milestone 9 is complete through M9.5.

- Milestone 1: validated EpiZoo scientific backend
- Milestone 2: reusable scientific tool layer
- Milestone 3: safe Agent orchestration core
- Milestone 4: provider-neutral natural-language planning
- Milestone 5.1: durable run state and planner-free resume
- Milestone 5.2: cooperative cancellation and durable run lifecycle
- Milestone 5.3: production error classification and deterministic recovery policy
- Milestone 6.1: EpiZoo embedding analysis with neighbors, Leiden clustering,
  and UMAP
- Milestone 6.2: quantitative evaluation of fixed cell clustering with NMI,
  ARI, AMI, and Homogeneity
- Milestone 6.3: within-species reference-to-query biological cell annotation
  through direct EpiZoo embedding-space label transfer
- Milestone 6.4: supervised evaluation and confidence diagnostics for fixed
  cell-annotation artifacts
- Milestone 7.1: deterministic verified analysis evidence for successful runs
- Milestone 7.2: deterministic verified scientific visualization from accepted
  analysis evidence
- Milestone 7.3: deterministic verified scientific reports from accepted
  evidence and optional verified visualizations
- Milestone 7.4: end-to-end research application with a Python service API,
  managed workspaces, verified post-run composition, and a one-shot CLI
- Milestone 8.1: explicit raw regulatory feature-space provenance and exact
  sparse replicate-aware pseudobulk SUM aggregation
- Milestone 8.2: independently verified replicate-aware differential
  accessibility through a pinned edgeR v4 quasi-likelihood workflow
- Milestone 9.1: deterministic Planner robustness benchmark with calibrated
  hard-semantic scoring, now extended through recovery-aware report schema v4
- Milestone 9.2: structured, sanitized planning diagnostics, now at schema v3
- Milestone 9.2.5: provider/model abstraction through immutable
  `PlanningModelProfile` and adapter-only `PlanningModelFactoryRegistry`
- Milestone 9.3: planning wire schema v3, registry-derived tool-discriminated
  schemas, and composable planning semantics
- Milestone 9.4: bounded Planning Recovery with transport retry, complete Plan
  repair, and one explicitly configured final profile failover
- Milestone 9.4.5: LLM-first user-facing new-run policy with explicit
  deterministic mode and provider-free resume/cancel
- Milestone 9.5: final LLM Planner robustness acceptance and Milestone 9
  closeout

The current Agent can construct and execute validated scientific workflows
through an explicit tool registry, with structured planning, verification,
error handling, dependency resolution, and execution tracing.

## LLM planning architecture

The Agent supports provider-neutral LLM planning. The LLM is an interchangeable
candidate-plan generator that owns natural-language intent understanding, tool
selection, and workflow composition. `LLMPlanner` converts its structured
decision into the existing `AgentPlan`, while deterministic code validates the
schema, tool allowlist, bindings, references, provenance, and complete plan
before execution.

`PlanningModelProfile` and `PlanningModelFactoryRegistry` decouple deployment
configuration and provider-adapter construction from planning behavior. OpenAI,
Gemini, Groq, and custom `PlanningModel` injection remain supported; no
production model is hard-coded as the Agent's intelligence. User-facing new
runs are LLM-first when supplied an explicit primary profile; missing LLM
configuration fails clearly instead of silently selecting deterministic
planning. `DeterministicPlanner` remains available through explicit application
selection and as the unchanged default of low-level `AgentRuntime`; it is not a
semantic oracle for LLM plans.

Planning wire schema v3 uses registry-derived, tool-discriminated structured
output: the selected tool fixes its exact keyed argument contract, input and
`StepOutputRef` bindings are distinct, request bindings are restricted to
available input names, and executable literals remain prohibited. Sanitized
registry metadata supplies artifact, data-flow, provenance, and scientific-
parameter preservation guidance. Scientifically valid noncanonical DAGs remain
allowed.

The deterministic offline Planner benchmark separates hard semantic correctness
from canonical workflow conformance and executes in PLAN_ONLY mode with zero
scientific calls. Recovery-aware report schema v4 distinguishes first-attempt,
transport-recovered, repair-recovered, and configured-failover outcomes.
Diagnostic schema v3 records sanitized attempt ordering and provider/model
provenance without raw request values, provider responses, or credentials.

Planning recovery is bounded to one initial call, either one same-profile
transport retry or one complete same-profile Plan repair, and—only when
explicitly configured—one final secondary-profile failover. Retry and repair
are mutually exclusive, failover is always the last call, and the hard ceiling
is three logical provider calls. Built-in provider adapters disable SDK retries.
Only the final preflight-passing Plan is persisted; interrupted planning is not
automatically replayed on resume. Application-owned LLM construction uses this
recovery path automatically, including an optional explicitly configured
secondary profile. Deterministic fallback, automatic model routing/ranking,
tool filtering, and keyword/regex routing remain unimplemented.

Milestone 9 is complete. Its final accepted planning path is:

```text
User Request
→ LLM-first Planner
→ exact tool/schema/data-flow planning interface
→ candidate AgentPlan
→ deterministic validation
→ bounded retry / repair / configured failover
→ final AgentPlan
→ authoritative preflight
→ scientific execution
```

The Planner core remains provider/model independent, deterministic planning is
explicit offline infrastructure, and no automatic deterministic fallback is
implemented. Planning Recovery permits at most three logical provider calls.
PLAN_ONLY executes zero scientific tools, while resume and cancel require no
provider, model, SDK, or credential. Final acceptance retains benchmark report
schema v4 and sanitized planning diagnostic schema v3.

Post-Milestone-9 provider compatibility hardening keeps planning wire schema v3
strict and tool-discriminated while using reusable closed `$defs`/`$ref`
binding schemas. Optional bindings use one flat input/ref/null union, and the
prompt carries a compact semantic catalog while still exposing all eleven
tools. Reliable HTTP 413 failures are reported terminally as sanitized
`PROVIDER_REQUEST_TOO_LARGE`; they are not retried, repaired, or treated as 429
rate limiting.

Post-M9 planner-interface hardening has established an experimental,
registry-driven semantic planning foundation across all currently
planner-visible tools. Semantic tool-interface metadata generates deterministic
compiler authority while keeping LLM reasoning and source selection separate
from binding, dependency construction, and serialization. Current tools are not
treated as a permanent closed set. An experimental, disconnected semantic wire
schema v4 and strict parser now substantially reduce planner serialization
burden. Production `LLMPlanner` continues to use planning wire schema v3; v4 is
not production-enabled or integrated with providers.

Real-provider PLAN_ONLY validation passed with Groq and guarded scientific
tools. LLM providers generate plans only: they receive no Python tool callables
and cannot directly execute scientific tools.

## Durable execution

Durability is opt-in through `AgentRuntime(..., run_store=...)`; without a run
store, execution remains in memory. `FileRunStore` provides local, versioned
persistence for the request, validated plan and lifecycle, step results,
errors, verification, and execution trace/provenance. Verified successful
steps are checkpointed before downstream execution.

`AgentRuntime.resume(run_id)` reuses the persisted plan without replanning.
Previously successful steps are revalidated through the existing argument
resolver, `ToolRegistry`, and verifier, so `StepOutputRef` dependencies continue
to work across process restart. PLAN_ONLY remains zero-execution on resume, and
unknown in-flight scientific work left RUNNING by a genuine interruption is
conservatively marked INTERRUPTED rather than rerun.

Run-state updates use atomic replacement, revision checks, SHA-256 integrity,
and stable lock files; a separate execution lease prevents concurrent runtimes
from interfering with an active run. Scientific tools, providers, planners,
registry, verifier, and retry semantics are unchanged. Providers receive no
filesystem or `RunStore` access, and core durability tests remain deterministic
and offline without network, GPU, model checkpoint, or provider configuration.

## Cooperative cancellation

`AgentRuntime.cancel(run_id)` requests cancellation of a durable run without
acquiring or stealing its execution lease. Cancellation is cooperative:
already-running scientific calls are not force-killed, and verified completed
work is checkpointed and preserved. Once cancellation is observed at a safe
checkpoint, no new attempt or downstream step starts; unstarted work is marked
SKIPPED and the run becomes CANCELLED. PLAN_ONLY remains zero scientific-tool
execution, and terminal CANCELLED resume invokes zero planners and zero tools.

Cancellation intent is stored separately from revisioned run state, so a
request can be recorded while another runtime owns execution without
invalidating active checkpoints. Duplicate requests are idempotent, terminal
runs remain immutable, and stale RUNNING work with an unknown outcome remains
INTERRUPTED. Core cancellation behavior remains deterministic and offline and
does not change scientific tools, planners, providers, registry, verifier, or
retry policy.

## Production error and recovery policy

Milestone 5.3 keeps the existing bounded `PlanExecutor` retry loop authoritative
and adds explicit recovery semantics rather than another retry engine.
`RecoveryDisposition` distinguishes no automatic recovery, same-step retry
eligibility, compatible-runtime resume, required user action, and manual
reconciliation. `AgentError.recoverable` now means only static same-step retry
eligibility, not general resumability or user fixability; unknown codes fail
closed to safe nonautomatic recovery.

Tool, provider, runtime, and verifier failure messages are sanitized before
persistence, so arbitrary raw exception strings are not exposed as persisted
`AgentError` messages. Reliable provider failures use provider-neutral codes,
and resource failures such as CUDA out-of-memory are classified explicitly.
Resource failures never silently change scientific settings such as batch size,
device, dtype, truncation, model, or overwrite behavior. Verification failures
remain conservative and do not blindly rerun scientific tools.

Every retry receives a fresh canonical-equivalent copy of the validated
arguments, so mutation in one attempt cannot affect the next. Exhaustion keeps
the underlying error and records attempts, the configured bound, and policy
provenance. New durable EXECUTE runs persist an immutable recovery-policy
snapshot; resume rejects changes to maximum attempts, retryable codes, or tool
recovery/classifier versions before scientific execution.

Run-state schema v3 keeps valid terminal v1/v2 records readable and historical
PLAN_ONLY runs zero-tool. Nonterminal legacy EXECUTE runs are rejected when
their historical recovery policy cannot be proven, and stale RUNNING work
continues to require manual reconciliation. Milestone 5.2 cancellation behavior
is unchanged. Core Milestone 5.3 tests require no network, provider credentials,
GPU, model checkpoint, or biological dataset.

## Downstream embedding analysis

Milestone 6.1 extends the validated scientific workflow from EpiZoo cell
embeddings through neighbor-graph construction, Leiden clustering, and 2D UMAP.
Each stage writes a compact, copy-on-write AnnData artifact containing ordered
cell IDs, the 512-dimensional EpiZoo representation, sparse graph data, analysis
outputs, and versioned provenance—never the original million-dimensional scATAC
feature matrix.

The real production path was validated end to end on 2,000 Fang2021 cells with
an RTX 4090. All five registered scientific steps succeeded and verified, cell
order was preserved, the input file remained unchanged, and durable terminal
resume revalidated the artifacts without rerunning scientific tools.

Milestone 6.2 adds artifact-based quantitative evaluation of fixed clustering.
Reference annotations are used only after unsupervised clustering and never to
tune neighbors, Leiden, UMAP, parameters, or cluster selection. Real Fang2021
evaluation successfully validated NMI, ARI, AMI, and Homogeneity while
preserving exact cell identity and order.

Milestone 6.3 adds exact deterministic label transfer directly between
reference and query EpiZoo embeddings. Persisted predictions contain a
biological label when assigned, confidence, and a separate assigned/unassigned
state. Real held-out Fang2021 validation succeeded without exposing query
ground-truth labels to the request, planner, embedding tools, transfer tool, or
production verifier.

Milestone 6.4 evaluates those fixed annotations with assignment coverage,
overall and assigned-only accuracy, macro-F1, deterministic per-class
diagnostics, a rectangular confusion summary, and descriptive confidence
medians. Ground truth remains evaluation-only and cannot tune or rerun label
transfer. Real held-out Fang2021 evaluation reproduced the frozen Milestone 6.3
metrics and confidence summaries.

## Replicate-aware regulatory foundation

Milestone 8.1 returns to the immutable raw scATAC H5AD for regulatory count
analysis. It adds two registered scientific tools:

```python
validate_scATAC_feature_space(
    input_path,
    output_dir,
    *,
    matrix_source,
    matrix_semantics,
    species,
    genome_assembly,
    coordinate_source,
    layer_key=None,
    feature_chrom_key=None,
    feature_start_key=None,
    feature_end_key=None,
    coordinate_system=None,
    semantics_metadata_key=None,
    overwrite=False,
)

build_replicate_pseudobulk(
    feature_space_path,
    replicate_key,
    group_key,
    condition_key,
    output_dir,
    *,
    group_source,
    group_annotation_path=None,
    covariate_keys=(),
    overwrite=False,
)
```

The feature-space tool accepts only sparse `X` or an explicitly named sparse
layer, preserves the declared fragment-count, insertion-count, or binary
accessibility semantics, and rejects normalized/continuous input. Species and
assembly are restricted to human/hg38 and mouse/mm10. Coordinates are optional
and explicitly provenance-recorded; they are never inferred.

The pseudobulk unit is exactly `(group, replicate, condition)`, ordered by first
occurrence in source cell order. Replicate identity may span conditions.
Replicate, condition, and covariates come from raw `.obs`; group comes either
from raw `.obs` or the fixed `predicted_label` of an accepted Milestone 6.3
annotation with exact cell identity/order and no unassigned cells. Aggregation
is exact sparse SUM with no normalization, feature filtering, cell filtering,
intersection, reordering, remapping, or coordinate inference.

Feature validation writes a compact schema-v1 canonical JSON manifest.
Pseudobulk writes one sparse CSR/int64 schema-v1 H5AD with original ordered
features, pseudobulk metadata in `.obs`, optional exact coordinates in `.var`,
and versioned provenance in `uns["agent_milestone8_pseudobulk"]`. Deterministic
pseudobulk row IDs are domain-separated hashes scoped to the authoritative
feature-space identity and the complete unit tuple.

Independent verification never invokes either scientific callable. It
reconstructs source and feature identity, metadata, unit order and IDs,
covariates, coordinates, library sizes, and every integer SUM with a Python
row-map algorithm distinct from production sparse matrix multiplication.
Counts are compared exactly and complete matrices are never densified.

Planning schema v2 and RunStore schema v3 remain unchanged. The production
registry now contains exactly eleven tools. Both M8.1 recovery identities are
versioned and have no automatically retryable scientific codes. Evidence and
deterministic reports remain schema v1, and M8.1 visualization is intentionally
figureless. edgeR, TMM, differential accessibility, genomic annotation, and
motif analysis are not part of Milestone 8.1.

## Replicate-aware differential accessibility

Milestone 8.2 adds the eleventh registered scientific tool:

```python
run_replicate_differential_accessibility(
    pseudobulk_path,
    group_value,
    condition_key,
    numerator_condition,
    denominator_condition,
    design_type,
    output_dir,
    *,
    covariates=(),
    overwrite=False,
)
```

The authoritative input is an accepted Milestone 8.1 SUM-count pseudobulk;
individual cells are never treated as DA replicates. Independent designs
require at least two disjoint biological replicates per condition, with an
explicit low-replication warning at two. Paired designs require at least three
complete biological pairs. The fixed two-condition contrast is numerator minus
denominator, with optional ordered additive categorical or numeric covariates.

The statistical path uses edgeR 4.10.4 on R 4.6.1 / Bioconductor 3.23 in the
isolated `agent-edger` runtime: condition-based `filterByExpr`, library-size
recalculation, TMM normalization, robust edgeR v4 quasi-likelihood fitting and
testing, and Benjamini–Hochberg correction. Statistical parameters are
repository-controlled and cannot be supplied by users or planners.

Verification is a separate execution path. It independently revalidates and
recomputes the M8.1 source, reconstructs selection/design/contrast/digests in
Python without calling production preparation or DA code, and invokes the
separately pinned `edger_ql_verify_v1.R` script. Source, artifact, preparation,
filter, normalization, statistics, effect directions, provenance, scripts, and
the exact R package stack are checked before success or durable reuse.

Planning schema v2 and RunStore schema v3 are unchanged. The DA recovery
identity is `run-replicate-differential-accessibility-edger-ql-v1`; the tool has
one actual attempt and no automatically retryable scientific/backend code.
AnalysisEvidence and deterministic reports remain schema v1 and compact, and
the application produces a verified figureless DA report without feature-level
statistics or biological interpretation.

Guarded real-data acceptance remains outstanding: local Fang2021 and PBMC
count data lack a genuine two-condition replicate design, the local BMMC
candidate is normalized continuous mixed-modality data rather than eligible raw
accessibility counts, and the replicated rice heat-shock dataset is outside the
supported human/mouse contract. No metadata was fabricated and no external data
was downloaded.

Deferred scope includes binary-accessibility inference, alternative or mixed
models, multi-condition/interacting/time-course designs, effect-size shrinkage,
adaptive filtering, genomic or peak-to-gene annotation, motifs, pathways,
regulatory interpretation, volcano/MA plots, perturbation analysis, and mutation
analysis.

## Verified analysis evidence

Milestone 7.1 adds deterministic post-run evidence generation:

successful `AgentRunResult`
→ fresh existing `verify_run()` and `verify_step()` verification
→ compact schema-v1 `AnalysisEvidence` projection
→ atomic `analysis_evidence.json`

The projection uses explicit whitelists for the eleven existing scientific tools
and has an authoritative evidence-file SHA-256. It records whether source
artifacts have authoritative cryptographic digests or are instead protected by
existing structural, provenance, and content verification. Embeddings, cell
vectors, UMAP coordinates, labels, confidence arrays, AnnData objects, and raw
scATAC matrices are never copied into evidence.

`AnalysisEvidence` is downstream of orchestration: it is neither a
`ToolRegistry` scientific tool nor part of `AgentPlan`. The production registry
contains exactly eleven tools, while planning schema v2 and the RunStore schema are
unchanged, and evidence build and verification never invoke registered
scientific callables. Terminal-resume results still undergo fresh artifact
verification before evidence is accepted. Visualization and narrative report
generation are separate downstream concerns.

## Verified scientific visualization

Milestone 7.2 adds a presentation-only flow downstream of orchestration:

successful `AgentRunResult`
→ verified `AnalysisEvidence`
→ fresh Milestone 7.1 verification
→ explicit presentation-only reads from verified artifacts
→ deterministic plotting-data projection
→ PNG figures and `visualization_manifest.json`

Version 1 produces exactly a Leiden-colored UMAP, a fixed NMI/ARI/AMI/
Homogeneity clustering-metric bar chart, and an annotation-evaluation raw
confusion matrix. Transferred-label UMAP, per-class F1 and confidence plots,
SVG, narrative reporting, and interactive UI remain deferred.

Visualization is neither a `ToolRegistry` scientific tool nor part of
`AgentPlan`; the production registry contains exactly eleven tools and planning
schema v2 remains unchanged. It adds no RunStore schema or recovery-policy
change. Build and verification invoke zero registered scientific callables,
never rerun or tune scientific analysis, never access raw scATAC `.X`, and use
only artifact paths explicitly bound by verified evidence rather than directory
discovery. Terminal-resume sources undergo fresh evidence and source
verification.

## Verified deterministic scientific report

Milestone 7.3 adds a fully deterministic, user-readable reporting flow:

successful `AgentRunResult`
→ verified `AnalysisEvidence`
→ optional verified `AnalysisVisualizations`
→ frozen report-fact projection
→ deterministic Markdown and optional copied PNGs
→ `report_manifest.json`

Report sections appear only when supported by verified workflow evidence, so
inspection-only reports are valid and absent analysis stages are never
fabricated. Scientific values come from a frozen whitelisted projection with
stable fact IDs for machine-readable attribution. Exact numeric values and
nullable values are preserved; qualitative claims such as excellent
performance, reliable annotation, or well-separated clusters are intentionally
not generated.

Visualization is optional. When supplied, every verified Milestone 7.2 PNG is
copied in its original order with byte-for-byte and SHA-256 equality. Figures
are neither redrawn nor visually interpreted. Markdown, fact attribution,
section bindings, copied-figure bindings, and the canonical manifest are
deterministic and exactly verifiable.

Milestone 7.3 v1 contains no LLM-generated narrative. Future constrained LLM
interpretation remains separate from this deterministic reporting boundary.

## End-to-end research application

Milestone 7.4 provides the first coherent user-facing flow:

```text
natural-language scientific request
→ constrained planning
→ verified durable execution
→ verified evidence
→ verified visualization when supported
→ verified deterministic scientific report
→ compact user-facing application result
```

The public service boundary is:

```python
ResearchAgentApplication(
    workspace_root,
    *,
    planner=None,
    primary_planning_profile=None,
    recovery_planning_profile=None,
    planning_model_factory_registry=None,
    planning_recovery_policy=None,
    registry=None,
    executor=None,
)
```

It exposes `run(request)`, `resume(run_id)`, and `cancel(run_id)`. The
application owns its `FileRunStore`, while all scientific execution continues
to occur exclusively through `AgentRuntime`. An explicitly injected Planner is
authoritative. Otherwise a new run requires an explicit primary model profile,
or explicit deterministic selection; optional secondary-profile failover uses
the same factory registry and M9.4 recovery implementation. Resume and cancel
require none of those planning settings. The application composes the existing
evidence, visualization, and deterministic-report APIs rather than duplicating
their logic.

The CLI defaults new runs to LLM mode and requires explicit `--provider` and
`--model` values. `--planner deterministic` selects offline deterministic
planning; `--provider deterministic` remains a compatibility alias.

After successful execution, evidence is built or verified and reused, supported
visualization kinds are queried explicitly, applicable figures are built or
reused, and the deterministic report undergoes final verification. An empty
visualization capability is a normal figureless-report workflow; failure of an
expected visualization remains fatal. PLAN_ONLY returns its validated plan and
preflight result with zero scientific execution and creates no evidence,
visualization, or report.

Resume remains planner-free. Valid post-run artifacts are verified and reused,
missing stages may be built, and tampered, mismatched, partial, or conflicting
outputs fail closed without silent repair or overwrite. Cancellation delegates
to `AgentRuntime.cancel()`; it remains cooperative, and reporting-stage
cancellation is deferred.

Managed output uses full SHA-256 run identities:

```text
<workspace>/
├── run_state/
└── runs/<full-sha256-of-run-id>/
    ├── composition.lock
    ├── scientific/
    ├── evidence/
    ├── visualizations/
    └── report/
```

Raw request/run IDs are not path components. Output roots are application-owned,
managed symlinks and wrong-type paths are rejected, and the application uses
exact artifact paths rather than directory discovery. The workspace is assumed
to be trusted and local; complete hostile-filesystem-race protection is not
claimed. A nonblocking per-run composition lock serializes evidence through
final report verification without changing runtime execution leases or adding
another durable state machine.

The compact JSON-safe application result retains the authoritative
`AgentRunResult` and uses `ArtifactReference` values containing only artifact
type, path, and SHA-256. It never returns embeddings, matrices, coordinates, or
AnnData objects. Application-local errors are sanitized and stage-aware, while
runtime/planner/scientific `AgentError` values remain authoritative.

The first CLI uses standard-library `argparse` and prints compact JSON:

```text
PYTHONPATH=src python -m agent run ...
PYTHONPATH=src python -m agent resume ...
PYTHONPATH=src python -m agent cancel ...
```

Deterministic offline planning is the default. Existing OpenAI, Gemini, and
Groq planning adapters can be selected when configured; API keys remain in the
environment. There is no installed console script yet because packaging
metadata is deferred. Streamlit, Gradio, a persistent REPL, multi-turn state,
LLM report narrative, and browser UI are also deferred.

The canonical first application demo is:

```text
inspect_scATAC
→ epizoo_embed_cells
→ build_cell_neighbors
→ cluster_cells
→ compute_cell_umap
→ verified evidence
→ Leiden UMAP
→ deterministic report
```

The richer reference/query annotation workflow remains available through the
scientific-tool layer but is not required for this primary demo. Milestone 7.4
did not run the optional real-provider plus real-EpiZoo acceptance.
