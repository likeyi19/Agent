# Agent

Agent is an autonomous AI agent for single-cell epigenomic / scATAC-seq
analysis. It turns natural-language requests and structured scientific inputs
into validated workflows, reproducible artifacts, and verified reports.
Existing foundation models such as EpiZoo and EpiAgent are scientific backends
to reuse; the Agent does not reimplement them. EpiZoo is the currently validated
embedding backend.

Milestones 1–9 and Post-M9 Planner Interface Hardening are complete, including
the latest static target-port projection and Groq schema compatibility fix.
Guarded real-data differential-accessibility acceptance remains outstanding.
[AGENTS.md](AGENTS.md) contains the detailed engineering rules, scientific
contracts, milestone history, acceptance results, and deferred work.

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

After successful execution, the application composes fresh verified evidence,
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

The opt-in semantic wire-v4 path is:

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

Wire **v3 remains the default**. Its registry-derived, tool-discriminated schema
requires exact keyed argument bindings, with reusable closed `$defs`/`$ref`
schemas and flat optional input/ref/null unions. Semantic **v4 is explicit
opt-in**, removing model-authored execution argument dictionaries, raw result
keys, references, and redundant dependency serialization. There is no automatic
schema switching, version detection, combined schema, or hidden v4-to-v3 fallback.

The latest follow-up projects legal target ports directly from each tool's
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
planner is not a semantic oracle for LLM output.

## Scientific capabilities

The current inventory below comes from
[`build_default_tool_registry()`](src/agent/orchestration/registry.py).
Planner-visible coverage is registry-derived, not a permanent tool-count limit.

| Workflow | Registered tools and accepted behavior |
| --- | --- |
| Inspect and embed | `inspect_scATAC`, `epizoo_embed_cells`: safe H5AD inspection, validated sparse preprocessing, process-local EpiZoo model reuse, 512-dimensional embeddings plus ordered cell IDs |
| Downstream embedding analysis | `build_cell_neighbors`, `cluster_cells`, `compute_cell_umap`: compact copy-on-write H5ADs with sparse graphs, weighted Leiden labels, and 2D UMAP |
| Clustering evaluation | `evaluate_cell_clustering`: NMI, ARI, AMI, and Homogeneity for fixed clustering; arithmetic averaging for NMI/AMI |
| Cell annotation | `transfer_cell_labels`: exact deterministic CPU kNN transfer directly between within-species reference/query EpiZoo embeddings using the same canonical checkpoint |
| Annotation evaluation | `evaluate_cell_annotation`: fixed-prediction assignment rate, overall/assigned accuracy, macro-F1, per-class diagnostics, rectangular confusion counts, and descriptive confidence medians |
| Regulatory feature foundation | `validate_scATAC_feature_space`, `build_replicate_pseudobulk`: explicit raw sparse feature provenance and exact SUM by `(group, replicate, condition)` |
| Differential accessibility | `run_replicate_differential_accessibility`: biological-replicate DA with pinned edgeR v4 quasi-likelihood fitting/testing and independent verification |

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

## Reliability and verified output

Whole-plan preflight occurs before scientific side effects. Each returned
result passes verification before downstream use. PLAN_ONLY executes **zero
scientific tools**, including across restart/resume/cancellation, and the
application creates no evidence, figures, or report for it.

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
Sanitized diagnostic schema v3 and offline benchmark report schema v4 expose
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

For LLM planning with explicit semantic v4 opt-in (the model shown is a prior
Groq acceptance configuration, not an automatically selected default):

```bash
PYTHONPATH=src python -m agent run \
  --request-id inspect-v4 --request "Inspect this scATAC dataset" \
  --workspace /path/to/workspace --input /path/to/cells.h5ad \
  --provider groq --model openai/gpt-oss-120b --wire-mode v4 --plan-only
```

Omit `--wire-mode` or use `--wire-mode v3` for the default wire contract. Optional
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

The latest pre-push Planner follow-up acceptance record reports 203 focused
passes, 1006 orchestration/provider/benchmark passes, 1429 lightweight passes
with 54 skips, and 7648 independent JSON Schema payload checks. These are
historical acceptance totals, not tests rerun for this documentation pass.
Separate live Groq checks accepted the strict schema, application v4 inspection,
and the complete five-tool downstream PLAN_ONLY DAG with zero scientific
execution and no target-port failure.

The mechanical serialization burden, static target-port generation, and Groq
discriminator issues have been addressed. Genuine hosted-model source/source-port
variability, incomplete or explicit unsupported decisions, and provider
availability/rate-limit/authentication issues remain possible. V4 stays opt-in;
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
