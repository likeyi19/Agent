# Agent

Agent is an AI system for autonomous single-cell epigenomic analysis.

## Completed milestones

### Milestone 1 — EpiZoo cell embedding backend

The first validated vertical slice is complete:

raw scATAC-seq AnnData
→ validated sparse preprocessing
→ EpiZoo
→ reproducible cell embeddings

Validated on Fang2021:
- 2,000 mouse scATAC-seq cells
- output shape: `(2000, 512)`
- exact scientific parity with the manual EpiZoo pipeline
- deterministic inference with fixed truncation seed
- RTX 4090 peak GPU memory: ~10.9 GiB
- validated default batch size: 4

The validated implementation is:

`src/agent/tools/models/epizoo.py`

Do not modify this backend unless required to fix a verified bug or to support a clearly defined new capability.

### Milestone 2 — Standard scientific tool layer

Milestone 2 is complete and validated on Fang2021.

Validated tools:

1. `inspect_scATAC`
2. `epizoo_embed_cells`
3. process-local EpiZoo model caching

The validated flow is:

user/file input
→ structured scientific tool
→ validated backend
→ structured lightweight result

Generalized LLM planning remains future work.

Do not implement annotation, clustering, UMAP, RAG, literature retrieval, reports, or multi-agent orchestration in this milestone.

### Milestone 3 — Agent orchestration core

Milestone 3 is complete. The Agent now supports natural request
representation, deterministic bootstrap planning, typed executable plans, an
explicit immutable tool registry, safe sequential execution, dependency/output
references, orchestration verification, structured errors, bounded
same-argument retry, execution traces, and PLAN_ONLY mode.

The production tool vocabulary is limited to `inspect_scATAC` and
`epizoo_embed_cells`; arbitrary Python and shell execution are prohibited. Real
end-to-end acceptance uses `inspect_scATAC` through AgentRuntime. Generalized
LLM planning remains future work.

### Milestone 4 — Natural-language planning

Milestone 4 is complete. The validated planning flow is:

Natural-language request
→ `LLMPlanner`
→ provider-neutral `PlanningModel`
→ strict versioned planning wire schema
→ existing `AgentPlan`
→ existing `AgentRuntime`
→ full-plan preflight
→ executor/verifier

Optional OpenAI, Gemini, and Groq provider adapters generate plans only. They
receive no Python tool callables and cannot directly execute scientific tools.
`ToolRegistry` remains the executable allowlist, and full-plan validation occurs
before side effects. PLAN_ONLY executes zero tools. Executable argument values
come only from structured `AgentRequest.inputs` or an existing `StepOutputRef`,
not arbitrary LLM literals. The default runtime remains deterministic and
offline; external providers require explicit injection and configuration.

Real-provider acceptance passed with Groq using `openai/gpt-oss-20b`: a strict
schema v2 plan passed through `LLMPlanner`, `AgentPlan`, and AgentRuntime
PLAN_ONLY preflight while guarded scientific tool callables confirmed zero
execution.

### Milestone 5.1 — Durable run state and resume

Milestone 5.1 is complete and accepted. Durability is opt-in through
`AgentRuntime(..., run_store=...)`; the default runtime remains in-memory.
`PersistedRunState` records the durability-specific PLANNING, VALIDATED,
RUNNING, PLANNED, SUCCEEDED, FAILED, and INTERRUPTED lifecycle states together
with the request, plan, step results, errors, verification, and
trace/provenance. `RunStore` is the persistence boundary, and `FileRunStore`
stores versioned canonical JSON with SHA-256 integrity, a plan fingerprint,
optimistic revision checks, and atomic `fsync` plus `os.replace` updates.

Each run has a stable state-update lock and a separate full-lifecycle execution
lease. Verified successful steps are durably checkpointed before downstream
execution. `AgentRuntime.resume(run_id)` is planner-free and reuses the persisted
plan through the existing `PlanExecutor`, `ToolRegistry`, argument resolver, and
verifier; there is no duplicate execution engine. Persisted successes are
revalidated before reuse, including `StepOutputRef` restoration across restart.

PLAN_ONLY remains zero-execution across restart, terminal resume is idempotent,
and stale RUNNING scientific work is conservatively marked INTERRUPTED with no
automatic rerun when its outcome is unknown. Scientific tools, providers,
planners, registry, verifier, and retry semantics were not changed. Providers
receive neither filesystem nor persisted `RunStore` access.

Accepted validation:

- durability/resume: 29 passed
- canonical orchestration regression: 222 passed
- complete lightweight regression: 376 passed, 6 skipped

Deferred non-blocking follow-ups:

- type-exact canonical comparison for restored resolved arguments
- stricter persisted attempt-count provenance validation
- optional progress-phase enum cleanup
- unused timestamp-helper cleanup
- canonical JSON helper consolidation
- post-replace fsync/chmod ambiguity documentation
- stale lock-file cleanup
- stronger trusted-directory/symlink hardening if the store root becomes untrusted

### Milestone 5.2 — Cooperative cancellation and run lifecycle

Milestone 5.2 is complete and accepted. The public cancellation contract adds
`RunStatus.CANCELLED`, `RunLifecycleStatus.CANCELLED`,
`ErrorCategory.CANCELLATION`, `CancellationReceipt`, and
`AgentRuntime.cancel(run_id)`.

Durable cancellation intent is stored separately as
`<sha256(run_id)>.cancel.json` with its own schema version, canonical JSON,
SHA-256 corruption detection, and atomic temporary write, `fsync`,
`os.replace`, and directory `fsync`. Cancellation uses the short per-run state
lock and never acquires the execution lease or increments the main run-state
revision. Duplicate requests preserve the original request timestamp;
malformed, corrupt, or unsupported sidecars fail closed.

Cancellation is cooperative and does not force-kill processes, threads, GPU
kernels, or tools. An already-started scientific call finishes, its returned
result follows normal `verify_step()` validation, and verified success is
durably checkpointed before cancellation takes effect. After cancellation is
observed, no next scientific attempt starts, cancellation before retry prevents
the retry, and pending downstream steps become SKIPPED. Existing failure
evidence is preserved, while a stale RUNNING step with an unknown outcome
remains INTERRUPTED.

Run and resume retain exclusive ownership of the execution lease, while cancel
may record intent during that lease. The separate sidecar avoids main-revision
conflicts, and the short state lock arbitrates cancellation against terminal
commit. Cancellation wins over a normal PLANNED, SUCCEEDED, or FAILED commit if
its intent linearized first; if terminal commit linearized first, cancel returns
ALREADY_TERMINAL. Terminal states remain immutable, and terminal CANCELLED
resume invokes neither planner nor scientific tools.

Cancellation never routes PLAN_ONLY through scientific execution. PLAN_ONLY
remains zero-tool across run, cancel, restart, and resume.

Persisted run-state schema version is now 2. Valid Milestone 5.1 version-1
records remain readable and are checked against the committed Milestone 5.1
lifecycle, error-category, and trace-event vocabularies. Loading v1 does not
rewrite it; the next legitimate update persists v2. A v1 record cannot contain
v2-only cancellation semantics.

Accepted validation:

- cancellation-focused: 46 passed
- canonical orchestration regression: 268 passed
- complete lightweight regression: 422 passed, 6 skipped

Deferred non-blocking follow-ups:

- tighter boundary-specific validation for legal CANCELLED state shapes
- direct PLAN_ONLY cancellation-requested resume test
- simultaneous no-owner cancel plus resume test
- synchronized concurrent cancel/terminal-commit race test
- execution-level unreadable-sidecar I/O simulation
- stronger single-test end-to-end cancellation trace coverage
- optional multiprocess cancellation race coverage
- future atomic JSON helper consolidation
- stronger trusted-directory/symlink hardening if the store root becomes untrusted
- post-`os.replace`/`fsync` ambiguity documentation

### Milestone 5.3 — Production error classification and recovery policy

Milestone 5.3 is complete and accepted with non-blocking follow-ups. It keeps
the existing `PlanExecutor` bounded retry loop as the only same-step retry
engine and adds explicit, versioned error and recovery semantics around it.

#### Public recovery semantics

`RecoveryDisposition` provides:

- `NO_AUTOMATIC_RECOVERY`
- `SAME_STEP_RETRY_ELIGIBLE`
- `RESUME_WITH_COMPATIBLE_RUNTIME`
- `USER_ACTION_REQUIRED`
- `MANUAL_RECONCILIATION`

`AgentError.recoverable` means static same-step retry eligibility only. It does
not mean general resumability, nonterminality, or user fixability. Dynamic retry
decisions remain separate and are recorded in RECOVERY trace events.

#### Error classification

The broad `ErrorCategory` set remains unchanged, while stable codes distinguish
planning, resource, environment, persistence, and execution failures. Unknown
codes fail closed. Generic `TypeError` and `ValueError` are not automatically
treated as user mistakes. CUDA out-of-memory and other reliably identifiable
resource failures are classified explicitly, and provider failures use
sanitized provider-neutral codes where structured information permits.

Arbitrary raw tool or provider exception messages are not persisted. Verifier
exception presentation is sanitized without changing scientific verification
logic.

#### Retry behavior

Retry preserves identical validated scientific values. Each attempt receives a
fresh canonical-equivalent argument copy, so nested mutation by one attempt
cannot affect another. Retry exhaustion preserves the underlying error and
records `retry_exhausted`, `attempts`, `max_attempts`, and the policy
fingerprint. Verification failures are not automatically retried, and
cancellation observed before a retry continues to suppress that retry.

#### Recovery-policy provenance

New durable EXECUTE runs persist an immutable schema-v3 policy snapshot before
scientific execution. It contains the catalog version, effective maximum
attempts, planned tool identities, tool recovery/classifier versions, sorted
retryable error-code sets, and a canonical fingerprint.

Resume reconstructs the current effective policy and rejects semantic drift
before scientific execution. An incompatible resume invokes no new scientific
tools, does not call the planner, and does not mutate or terminalize the stored
run; it requires a compatible runtime.

#### Legacy and persistence behavior

Persisted run-state schema is now version 3. Valid historical v1/v2 terminal
states remain readable, and historical PLAN_ONLY remains zero-tool. Nonterminal
legacy v1/v2 EXECUTE records without authoritative recovery provenance are
rejected before new scientific execution; current registry state is never used
to fabricate historical policy. Historical decoding remains distinct from new
current-schema state creation.

Current schema-v3 `AgentError` records require `recoverable` and
`recovery_disposition` to be semantically consistent. Contradictory persisted
fields are corruption rather than being silently normalized. New states cannot
spoof a historical `source_schema_version` to bypass v3 invariants, and
`FileRunStore` must not successfully create a record it cannot read back.

#### Safety invariants

- arbitrary Python and shell execution remain prohibited
- LLM providers plan only and cannot directly execute tools
- `ToolRegistry` remains the executable allowlist
- batch size, device, dtype, truncation, model, and overwrite settings are never automatically changed
- stale RUNNING scientific work is never automatically rerun
- required checkpoint failure still stops downstream execution
- PLAN_ONLY remains zero scientific execution
- Milestone 5.2 cancellation semantics remain unchanged

Accepted validation:

- focused Milestone 5.3: 38 passed
- orchestration/provider unit tests: 408 passed
- canonical orchestration regression: 306 passed
- complete lightweight regression: 460 passed, 6 skipped

Deferred non-blocking follow-ups:

- interrupted-before-plan recovery-disposition inconsistency
- direct catalog-version drift regression
- explicit `StepOutputRef` plus mutable-list retry regression
- minor remaining `RunStore` I/O normalization gaps
- stronger catalog call-site enumeration coverage

### Milestone 6.1 — Downstream EpiZoo embedding analysis

Milestone 6.1 is complete and accepted. The production `ToolRegistry` now
contains exactly five scientific tools:

1. `inspect_scATAC`
2. `epizoo_embed_cells`
3. `build_cell_neighbors`
4. `cluster_cells`
5. `compute_cell_umap`

The accepted downstream artifact flow is:

EpiZoo embeddings `.npy` plus ordered cell IDs
→ `*.neighbors.h5ad`
→ `*.neighbors.clustered.h5ad`
→ `*.neighbors.clustered.umap.h5ad`

These compact, copy-on-write artifacts preserve exact cell order and versioned
provenance. They contain `obsm["X_epizoo"]`, sparse neighbor graphs, Leiden
labels, and 2D UMAP coordinates, but not the original million-dimensional
scATAC feature matrix. User input and upstream artifacts are never modified.

Accepted scientific defaults:

- neighbors use all 512 EpiZoo dimensions through `use_rep="X_epizoo"`,
  `n_neighbors=15`, Euclidean distance, Scanpy UMAP-style connectivity, and
  `random_seed=0`
- clustering uses Leiden only, resolution `1.0`, igraph flavor, the weighted
  graph, no resolution sweep or label-informed selection, and `random_seed=0`
- UMAP uses two dimensions, `min_dist=0.5`, `spread=1.0`, spectral
  initialization, and `random_seed=0`

Milestone 6.1 preserves the existing orchestration and lifecycle contracts.
Only registered scientific tools execute; arbitrary Python and shell remain
prohibited. Executable planner arguments still come only from
`AgentRequest.inputs` or `StepOutputRef`, and the LLM planner cannot invent
executable literals. Planning schema v2 and whole-plan preflight remain
authoritative, PLAN_ONLY executes zero tools, and downstream tools are
nonretryable by default. Durable resume revalidates artifacts before restoring
downstream references, cancellation behavior is unchanged, and raw scATAC
matrices are never densified.

The new recovery-policy identities are:

- `build-cell-neighbors-v1`
- `cluster-cells-v1`
- `compute-cell-umap-v1`

Existing recovery identities and the global error-policy catalog version are
unchanged.

Accepted validation:

- focused Milestone 6.1: 182 passed
- canonical orchestration regression: 318 passed
- complete lightweight regression: 506 passed, 6 skipped

Real production-path acceptance ran the complete five-tool Fang2021 workflow
for 2,000 cells on an RTX 4090. All steps succeeded and verified in about 64.4
seconds with about 10.9 GiB peak allocated GPU memory. The final EpiZoo
representation was `(2000, 512)`, the UMAP was `(2000, 2)`, Leiden produced 21
clusters, cell order was exact, the input file was unchanged, and terminal
resume revalidated every artifact without rerunning scientific tools.

Non-blocking environment notes from acceptance: the installed Louvain package
emits a `pkg_resources` deprecation warning, Scanpy notes that Louvain is
superseded by Leiden, and the installed TBB version disables Numba's TBB
threading layer. Milestone 6.1 uses Leiden, and these warnings did not affect
acceptance.

### Milestone 6.2 — Quantitative clustering evaluation

Milestone 6.2 is complete and accepted. Its public scientific API is:

```python
evaluate_cell_clustering(
    analysis_path,
    reference_h5ad_path,
    label_key,
    output_dir,
    *,
    cluster_key="leiden",
    overwrite=False,
)
```

The production `ToolRegistry` now contains exactly six scientific tools:

1. `inspect_scATAC`
2. `epizoo_embed_cells`
3. `build_cell_neighbors`
4. `cluster_cells`
5. `compute_cell_umap`
6. `evaluate_cell_clustering`

The accepted scientific direction is strictly:

unsupervised clustering
→ supervised evaluation

Ground-truth annotations are used only during evaluation. They never affect
neighbors, Leiden resolution, clustering, UMAP, parameter optimization,
cluster selection, or upstream reruns. Milestone 6.2 performs no resolution
sweep, label-informed clustering, metric-guided optimization, or automatic
best-clustering selection.

Evaluation accepts valid Milestone 6.1 clustering- or UMAP-stage artifacts and
opens the reference AnnData backed and read-only. It reads the exact ordered
cell IDs and selected `obs[label_key]` only; the raw scATAC `.X` matrix is never
materialized or densified. Cell count, identity, uniqueness, and order must
match exactly. There is no cell intersection, reordering, sorting, subset
alignment, or silent dropping. Reference annotations require at least two
classes as a v1 scientific-validity rule, while a one-cluster prediction
remains valid.

The tool calculates exactly:

- Normalized Mutual Information (NMI)
- Adjusted Rand Index (ARI)
- Adjusted Mutual Information (AMI)
- Homogeneity

NMI and AMI explicitly use `average_method="arithmetic"`. No additional
evaluation metric is part of Milestone 6.2.

The persisted artifact is strict, atomic, overwrite-protected JSON named
`<analysis-stem>.clustering_metrics.json`. It records resolved source paths,
label and cluster keys, cell/class/cluster counts, the four metrics, sklearn
backend/version, arithmetic averaging, cell-order validation, and SHA-256
digests of the scientifically relevant ordered cells, normalized ordered
reference labels, normalized ordered predicted labels, and canonical
Milestone 6 analysis provenance. It does not hash the complete reference
`.h5ad` and does not contain complete cell-ID or label vectors, matrices,
embeddings, graphs, UMAP coordinates, or AnnData objects.

The deterministic evaluation workflow is:

inspect
→ embed
→ neighbors
→ cluster
→ evaluate

UMAP is intentionally omitted because it is unnecessary for clustering
metrics. Planning schema v2, whole-plan preflight, the registered-tool
allowlist, arbitrary Python/shell prohibition, PLAN_ONLY zero-tool behavior,
and cancellation semantics remain authoritative. Executable values still
come only from `AgentRequest.inputs` or `StepOutputRef`; the LLM planner cannot
invent literals. When omitted, `cluster_key="leiden"` comes from the Python API
default rather than a planner-generated literal.

Evaluation verification strictly reopens the JSON report, rereads the compact
analysis artifact and selected backed reference annotation, revalidates exact
cells/order, recomputes all relevant fingerprints, independently recomputes all
four sklearn metrics, and checks report/result/recomputed consistency using
`rel_tol=1e-12` and `abs_tol=1e-12`.

On nonterminal durable resume, a completed evaluation is revalidated before
reuse. Changed reference labels, predicted clusters, or relevant analysis
provenance and missing/corrupt reports or source artifacts are detected. A
successfully verified result is restored without invoking the evaluation tool
again. Existing terminal-resume behavior is unchanged.

The new recovery identity is `evaluate-cell-clustering-v1`, with no retryable
error codes. Existing recovery identities and the global error-policy catalog
version are unchanged.

Accepted validation:

- Milestone 6.2 focused: 194 passed
- new direct/integration/lifecycle tests: 45 passed
- canonical orchestration regression: 329 passed
- complete lightweight regression: 556 passed, 6 skipped

Real acceptance reused the fixed Milestone 6.1 clustering for 2,000 Fang2021
cells with `label_key="celltype"`; no EpiZoo/CUDA rerun or label-informed
parameter tuning occurred. Evaluation found 20 reference classes and 21 Leiden
clusters: NMI `0.8642463249536162`, ARI `0.746014277040041`, AMI
`0.8591719263671213`, and Homogeneity `0.854796248075491`, using scikit-learn
1.9.0 and arithmetic averaging. Independent sklearn recomputation reproduced
all values, exact cell identity/order passed, durable nonterminal resume reused
the verified evaluation without rerunning it, and both source files remained
byte-identical.

Milestone 6.2 introduced no new environment warning. The non-blocking
Louvain/`pkg_resources`, Scanpy Louvain deprecation, and TBB/Numba notes from
Milestone 6.1 remain unchanged and did not affect acceptance.

## Development environment

- Linux server
- NVIDIA RTX 4090
- 24 GB VRAM
- VS Code Remote SSH
- Python
- PyTorch
- Scanpy / AnnData

## Scientific backends

EpiAgent and EpiZoo are existing scientific foundation models.

They should be treated as scientific backends and reusable tools,
rather than being reimplemented as part of the agent.

Existing validated scientific logic should be reused whenever possible.

## Engineering rules

- Never densify a complete scATAC-seq matrix.
- Never assume more than 24 GB GPU memory.
- Always test on small datasets before full-scale execution.
- Do not modify validated EpiZoo scientific logic unless necessary.
- Every scientific capability should be exposed through a clean reusable tool.
- Every tool should have explicit inputs and outputs.
- Every tool should validate its inputs and provide informative errors.
- Keep analyses reproducible.
- Record model checkpoints and execution parameters.
- Do not commit biological datasets or model checkpoints to Git.

## Completed development task

The first standard Agent tool layer exposes validated scientific backends.

Validated capabilities:
- inspect a scATAC-seq `.h5ad` file safely
- expose EpiZoo embedding through a file/path-based tool interface
- return structured results suitable for later LLM tool calling

## Do not implement yet

- multi-agent architecture
- literature retrieval
- ENCODE retrieval
- RAG
- cCRE perturbation
- variant interpretation
- web UI
- automatic scientific reports
