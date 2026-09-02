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

### Milestone 6.3 — Reference-to-query cell-label transfer

Milestone 6.3 is complete and accepted. Its public scientific API is:

```python
transfer_cell_labels(
    reference_embedding_path,
    reference_cell_ids_path,
    reference_h5ad_path,
    reference_label_key,
    query_embedding_path,
    query_cell_ids_path,
    query_h5ad_path,
    output_dir,
    *,
    reference_species,
    query_species,
    reference_checkpoint_path,
    query_checkpoint_path,
    n_neighbors=20,
    metric="euclidean",
    min_confidence=0.0,
    overwrite=False,
)
```

The production `ToolRegistry` now contains exactly seven scientific tools:

1. `inspect_scATAC`
2. `epizoo_embed_cells`
3. `build_cell_neighbors`
4. `cluster_cells`
5. `compute_cell_umap`
6. `evaluate_cell_clustering`
7. `transfer_cell_labels`

The accepted label-transfer workflow is:

```text
annotated reference scATAC
→ inspect reference
→ reference EpiZoo embedding ┐
                               ├→ reference-to-query label transfer
query scATAC                  │
→ inspect query            │
→ query EpiZoo embedding ──────┘
```

Transfer operates directly in the original validated 512-dimensional EpiZoo
embedding space. It does not use PCA, UMAP, Leiden, clustering, centering,
standardization, batch correction, learned projections, approximate neighbors,
or reference subsampling.

The exact CPU backend uses scikit-learn chunked pairwise distances with bounded
working memory, `n_jobs=1`, and no backend auto-switching or random scientific
stage. Accepted defaults are `n_neighbors=20`, Euclidean distance, uniform
plurality voting, and `min_confidence=0.0`. Neighbor ordering is deterministic:
distance ascending, then reference row index ascending, including ties at the
kth boundary. There is no automatic reduction of k.

Confidence is the winning vote count divided by `n_neighbors`. A prediction is
assigned only when one label has a unique plurality and confidence is at least
`min_confidence`. Exact top-vote ties remain unassigned, with a missing
`predicted_label` and retained confidence. Assignment state is stored separately
in `prediction_status`; `"unassigned"` is not a reserved biological label.
There is no distance weighting, lexicographic tie break, class-frequency
correction, or label-guided parameter tuning.

Reference annotations are read only from the selected reference `.obs` column.
They must be nonmissing, nonblank text or categorical text labels without
leading or trailing whitespace and must contain at least two classes. Numeric,
boolean, and arbitrary-object labels are rejected, while accepted biological
label text is preserved exactly. Query ground-truth labels are neither required
nor available to the production transfer path.

Milestone 6.3 v1 is within-species only. Both species must be supported by
EpiZoo and equal. Reference and query checkpoint paths are canonicalized, must
exist, and must resolve to exactly the same file. The approximately 5.2 GB
checkpoint is not fully hashed. Within the Agent workflow, verified embedding
results and `StepOutputRef` bindings provide species and checkpoint provenance;
externally supplied embeddings do not gain cryptographic historical proof that
the stated checkpoint produced them.

Canonical digests protect both embedding contents, both ordered cell-ID
sidecars, ordered reference labels, and the fixed EpiZoo model configuration.
Embedding digests cover a versioned schema, shape, dtype, and row-major float32
contents through chunked memory-mapped reads. The raw reference and query h5ad
files are not fully hashed, and transfer neither accesses nor densifies their
raw scATAC `.X` matrices.

The compact, atomic annotation artifact is:

```text
<query-stem>.label_transfer.h5ad

n_obs = query cells
n_vars = 0
X = None

obs_names = exact ordered query cell IDs
obs["predicted_label"]
obs["prediction_confidence"]
obs["prediction_status"]
uns["agent_milestone6_label_transfer"]
```

It contains no raw scATAC matrix, source embedding matrix, neighbor list,
distance matrix, complete reference-label vector, Leiden label, UMAP coordinate,
or graph matrix. Writing is overwrite-protected, temporary-file validated,
atomically installed, and fsynced. The final annotation-file SHA-256 is stored
in the lightweight tool/durable step result, not inside the same H5AD artifact,
where it would be self-referential.

The deterministic planner generates:

```text
inspect_reference → embed_reference ┐
                                      ├→ transfer
inspect_query     → embed_query     ┘
```

Both embedding steps receive the same structured species and checkpoint
configuration. Transfer receives actual upstream species, checkpoint path,
embedding path, and ID-sidecar path through `StepOutputRef`. Optional scientific
arguments are omitted unless present in structured request inputs. Planning
schema v2, whole-plan preflight, the executable allowlist, and PLAN_ONLY
zero-tool behavior remain authoritative. The LLM cannot invent executable
paths, species, checkpoint, label key, k, metric, confidence threshold, or
defaults.

The verifier intentionally does not rerun the full kNN calculation. It checks
the final annotation SHA, compact structure, exact query order,
label/status/confidence consistency, assignment counts and rate, current
reference vocabulary, all source and model digests, species and canonical
checkpoint compatibility, and scientific parameters/backend/provenance. On
nonterminal resume, changed embeddings, IDs, reference labels, or missing or
corrupt annotations fail revalidation. A valid completed transfer is restored
without rerunning kNN; terminal resume behavior is unchanged.

The new recovery identity is `transfer-cell-labels-v1`, with no retryable error
codes. All previous recovery identities and the global error-policy catalog
version remain unchanged.

Accepted validation:

- Milestone 6.3 focused: 205 passed
- all orchestration unit tests: 345 passed
- canonical orchestration regression: 350 passed
- complete lightweight regression: 612 passed, 6 skipped

Real acceptance used a seed-0, label-independent split of 2,000 Fang2021 cells:
1,400 annotated reference cells and 600 disjoint query cells, with each subset
returned to original source order. The reference label key was `celltype`; the
production query h5ad contained no `celltype` column. Held-out query labels
existed only in the acceptance harness and were unavailable to `AgentRequest`,
the planner, both embedding tools, transfer, and the production verifier.

All five production steps succeeded on their first attempt and passed
verification in about 61.65 seconds. Transfer itself took about 0.90 seconds,
durable revalidation took about 0.92 seconds without reinvoking any tool, and
peak allocated GPU memory was about 10.8 GiB. No Leiden, UMAP, clustering
evaluation, or parameter tuning occurred.

Using the unchanged defaults, 596 of 600 query cells were assigned, for an
assignment fraction of `0.9933333333`. Held-out descriptive evaluation performed
only after artifact finalization and production verification gave overall
accuracy `0.905`, assigned-only accuracy `0.9110738255`, and macro-F1
`0.8615679910` across 20 true query classes and 19 assigned predicted classes.
Median confidence was `1.0`, with median confidence `1.0` for correct assigned
predictions and `0.6` for incorrect assigned predictions. These metrics did not
choose the split, reference, k, metric, confidence threshold, or any rerun.

A second transfer using the same persisted embeddings and defaults produced
identical biological predictions, assignment statuses, confidence values, and
scientific provenance. SHA-256 before/after checks confirmed that the reference
and query subsets, both embeddings, both ID sidecars, and original Fang2021
source were unchanged.

Milestone 6.3 introduced no new blocking environment issue. The existing
non-blocking Louvain/`pkg_resources`, Scanpy Louvain deprecation, and TBB/Numba
warnings remain unchanged and did not affect acceptance.

### Milestone 6.4 — Annotation evaluation and confidence diagnostics

Milestone 6.4 is complete and accepted. Its public scientific API is:

```python
evaluate_cell_annotation(
    annotation_path,
    ground_truth_h5ad_path,
    ground_truth_label_key,
    output_dir,
    *,
    overwrite=False,
)
```

The production `ToolRegistry` now contains exactly eight scientific tools:

1. `inspect_scATAC`
2. `epizoo_embed_cells`
3. `build_cell_neighbors`
4. `cluster_cells`
5. `compute_cell_umap`
6. `evaluate_cell_clustering`
7. `transfer_cell_labels`
8. `evaluate_cell_annotation`

Milestone 6.4 evaluates an already-fixed valid Milestone 6.3 annotation.
Ground truth is evaluation-only: it never affects EpiZoo embedding, reference
selection, k, metric, confidence threshold, voting, assigned/unassigned state,
transfer reruns, clustering, or UMAP.

The accepted metrics are assignment rate (`assigned_count / n_cells`), overall
accuracy (`correct_assigned_count / n_cells`, with unassigned cells counted as
incorrect), assigned-only accuracy (`correct_assigned_count / assigned_count`),
and macro-F1. Assigned-only accuracy is JSON `null` when no cell is assigned;
undefined confidence subsets also use `null`, never NaN or a fabricated zero.
Ground-truth biological classes define the fixed macro-F1 class set,
`zero_division=0`, and structural unassigned predictions count as errors without
becoming a biological class. Assigned predicted classes absent from query
ground truth remain valid predictions and count as errors where appropriate.

Milestone 6.4 v1 reports only descriptive confidence medians across all cells,
assigned cells, correct assigned predictions, and incorrect assigned
predictions. It performs no threshold optimization, ECE, calibration fitting,
ROC thresholding, or alternate assignment generation. The persisted report also
contains deterministic per-ground-truth-class support, true positives,
precision, recall, and F1 in first-occurrence ground-truth order.

The confusion summary is rectangular. Rows are ground-truth biological classes;
columns are observed predicted ground-truth classes in ground-truth order,
observed external predicted biological classes in first-prediction order, and a
final structural-unassigned column. Structural unassigned uses a null-labeled
descriptor and cannot collide with a legitimate biological label such as
`"unassigned"`.

Ground-truth AnnData is opened backed and read-only. Evaluation reads only exact
ordered `obs_names` and the selected `obs[ground_truth_label_key]`; raw `.X` is
never accessed or densified. Cell identity and order must match exactly, with no
intersection, sorting, reordering, subset alignment, silent dropping, or label
normalization. Ground-truth labels must be valid biological text, and a single
ground-truth class is valid.

Input annotations must retain the accepted Milestone 6.3 schema, type, stage,
and provenance, with `n_vars = 0`, `X = None`, and exactly
`predicted_label`, `prediction_confidence`, and `prediction_status` in `obs`.
The strict, deterministic, atomic, fsynced, overwrite-protected report is:

```text
<annotation-stem>.annotation_evaluation.json
artifact type: agent.cell-annotation-evaluation
schema version: 1
```

It records counts, metrics, confidence and per-class diagnostics, rectangular
confusion counts, validation/provenance, and software/backend metadata. It does
not contain complete cell-ID, ground-truth, prediction, status, or confidence
vectors, embeddings, or raw matrices. Serialization rejects NaN, infinity, and
duplicate object keys, and the temporary report is strictly validated before
atomic publication.

The direct scientific dependency boundary is the fixed M6.3 annotation plus
ordered ground-truth labels. Provenance protects the complete annotation-file
SHA-256, canonical M6.3 annotation provenance, ordered query IDs, ordered ground
truth and predicted labels, prediction statuses, and ordered confidence values.
Milestone 6.4 intentionally does not rehash reference/query EpiZoo embeddings or
checkpoint contents, which remain M6.3 responsibilities, and it does not hash
the complete ground-truth H5AD.

The deterministic planner supports standalone fixed-annotation evaluation and:

```text
inspect_reference → embed_reference ┐
                                      ├→ transfer → evaluate_annotation
inspect_query     → embed_query     ┘
```

The chained evaluation receives `transfer.annotation_path` through
`StepOutputRef`. Ground-truth path and key occur only in the evaluation step and
never reach inspection, embedding, or transfer. Planning schema v2 remains
authoritative, PLAN_ONLY executes zero scientific tools, and the LLM receives
only sanitized metadata and cannot invent executable metric settings.

For a direct transfer dependency, the verifier requires both annotation path
and annotation SHA-256 to equal the verified transfer result. Standalone
evaluation validates the current M6.3 artifact but cannot prove its historical
EpiZoo/checkpoint origin.

Evaluation is inexpensive enough for the verifier to reopen both sources and
fully recompute cell/order validation, counts, assignment rate, overall and
nullable assigned accuracy, macro-F1, confidence and per-class diagnostics,
rectangular confusion counts, and all M6.4 provenance digests. Report, result,
and recomputed floats use `rel_tol=1e-12` and `abs_tol=1e-12`.

On nonterminal durable resume, completed evaluation is fully revalidated.
Changed predictions, statuses, confidence values, annotation provenance, ground
truth, or missing/corrupt reports and sources are rejected. Valid evaluation is
restored without invoking `evaluate_cell_annotation` again; terminal-resume
semantics remain unchanged.

The new recovery identity is `evaluate-cell-annotation-v1`, with no retryable
error codes. All previous identities and the global error-policy catalog version
remain unchanged.

Accepted validation:

- Milestone 6.4 focused: 219 passed
- full orchestration unit suite: 370 passed
- canonical orchestration regression: 375 passed
- complete lightweight regression: 673 passed, 6 skipped

Real acceptance reused the frozen accepted Milestone 6.3 annotation without
EpiZoo inference, CUDA, transfer rerun, or tuning. Exact seed-0 query order and
the annotation SHA were verified for 600 held-out Fang2021 cells. Evaluation
found 20 ground-truth classes and 19 assigned predicted classes: 596 cells were
assigned and 4 unassigned, with 543 correct and 53 incorrect assigned
predictions. Assignment rate was `0.9933333333333333`, overall accuracy `0.905`,
assigned accuracy `0.9110738255033557`, and macro-F1 `0.8615679910000722`.
Median all-cell, assigned, and correct-assigned confidence were `1.0`; median
incorrect-assigned confidence was `0.6`. These reproduced the frozen M6.3
descriptive oracle and did not tune transfer behavior. Production verification
passed, and durable nonterminal resume preserved one evaluation invocation with
zero planner calls.

Milestone 6.4 introduced no new blocking environment issue. Existing
Louvain/`pkg_resources`, Scanpy Louvain deprecation, and TBB/Numba warnings
remain non-blocking; deliberate duplicate-ID negative tests also emit expected
AnnData warnings.

### Milestone 7.1 — Verified Analysis Evidence

Milestone 7.1 is complete and accepted. Its public API is:

```python
build_analysis_evidence(
    run_result: AgentRunResult,
    output_dir: str | Path,
    *,
    registry: ToolRegistry,
    overwrite: bool = False,
) -> AnalysisEvidenceResult
```

```python
verify_analysis_evidence(
    run_result: AgentRunResult,
    evidence_path: str | Path | AnalysisEvidenceResult,
    *,
    registry: ToolRegistry,
) -> VerificationResult
```

Only successful, non-PLAN_ONLY `AgentRunResult` objects are eligible. Evidence
construction and verification both require a fresh `verify_run()` and a fresh
topological `verify_step()` for every step. Existing `StepOutputRef` and
resolved-argument bindings are checked as part of that boundary. Scientific
callables are never invoked by evidence construction or verification, and a
terminal resume does not bypass fresh artifact verification.

Schema v1 contains compact, explicit whitelisted projections for all eight
current production tools. Unsupported future tools fail closed, and arbitrary
future result fields are not automatically exposed. The evidence layer does not
scan arbitrary output directories and excludes embeddings, cell-ID and label
vectors, UMAP coordinates, confidence arrays, AnnData objects, raw scATAC
matrices, and other large scientific payloads.

The persisted artifact is:

```text
analysis_evidence.json
artifact_type: agent.analysis-evidence
schema_version: 1
```

Persistence uses canonical deterministic JSON, rejects duplicate keys and
nonfinite numbers, validates the temporary artifact, fsyncs the file, installs
it with atomic `os.replace`, fsyncs the directory, protects against accidental
overwrite, and returns the final evidence-file SHA-256. This digest is
authoritative for the evidence file itself. Evidence does not imply universal
whole-file cryptographic hashing of every scientific artifact: some existing
artifacts have authoritative digests, while others are protected by existing
structural, provenance, and content verifier logic. Schema v1 records that
distinction explicitly.

Milestone 7.1 is downstream of orchestration and introduces no scientific-tool
registration. The production `ToolRegistry` remains exactly eight tools. There
is no new recovery identity, recovery-policy change, planning-schema change,
RunStore-schema change, orchestration change, provider change, or EpiZoo change.
It introduces no visualization, narrative generation, or LLM exposure of
scientific payloads. Visualization is provided separately by Milestone 7.2,
while narrative/report-model integration remains later work.

Accepted validation:

- focused Milestone 7.1: 30 passed
- canonical orchestration regression: 375 passed
- complete lightweight regression: 703 passed, 6 skipped

The lightweight integration path is:

```text
tiny sparse H5AD
→ AgentRuntime
→ inspect_scATAC
→ successful AgentRunResult
→ build_analysis_evidence
→ verify_analysis_evidence
```

It requires no network, API key, GPU, checkpoint, or EpiZoo inference. Guarded
scientific callables also proved that evidence construction and verification
execute zero scientific tools.

### Milestone 7.2 — Verified Scientific Visualization

Milestone 7.2 is complete and accepted. Its public API is:

```python
build_analysis_visualizations(
    run_result,
    evidence,
    output_dir,
    *,
    registry,
    overwrite=False,
)
```

```python
verify_analysis_visualizations(
    run_result,
    evidence,
    visualization,
    *,
    registry,
)
```

Both a successful `AgentRunResult` and its `AnalysisEvidence` are required.
The accepted trust boundary is:

1. freshly verify `AnalysisEvidence`;
2. strictly load the verified evidence;
3. derive the exact supported figure set;
4. read only artifact paths explicitly bound by evidence;
5. derive deterministic plotting data and domain-separated digests;
6. render or verify the visualization bundle.

No arbitrary artifact or directory scanning is permitted. Build narrows source
races with a second fresh evidence verification and source projection before
publication. Verification freshly rederives the expected figures and plotting
metadata without rerendering PNG bytes.

#### Exact v1 figures

Milestone 7.2 v1 produces exactly:

1. UMAP by Leiden cluster;
2. an NMI / ARI / AMI / Homogeneity bar chart;
3. an annotation-evaluation raw confusion matrix.

Transferred-label UMAP, per-class F1 figures, confidence figures, SVG,
narrative reporting, and interactive UI are deferred. Transferred-label UMAP
requires a future explicit provenance binding between the query UMAP and the
exact query embedding/cell-ID source used by label transfer; matching cell IDs
alone is intentionally insufficient.

The UMAP presentation reads only ordered `obs_names`, `obsm["X_umap"]`, and
`obs["leiden"]` from the verified compact UMAP H5AD. It does not access `.X`,
`obsm["X_epizoo"]`, neighbor graphs, or raw scATAC. Coordinates and cell order
are unchanged: there is no jitter, subsampling, or coordinate transformation.
Leiden categories use first-occurrence order, a fixed versioned palette with a
deterministic extension rule, deterministic cell-count-based point sizing, and
fixed presentation parameters.

Clustering metrics come directly from verified Milestone 7.1 evidence in the
fixed order NMI, ARI, AMI, and Homogeneity. The chart uses a fixed `[-1, 1]`
axis and zero reference line and performs no ranking, parameter comparison, or
selection.

The annotation confusion figure strictly presentation-reads the exact
Milestone 6.4 report referenced by evidence. It uses persisted raw counts,
preserves exact row and column order, and retains structural unassigned as the
final column. It performs no prediction, threshold, calibration, normalization,
or scientific optimization.

#### Visualization bundle and rendering

The persisted bundle is:

```text
analysis_visualizations/
├── figures/
│   └── NNN_<figure-kind>_<step-hash>.png
└── visualization_manifest.json
```

The manifest has artifact type `agent.analysis-visualizations` and schema
version 1. It binds run/request/plan identity, the authoritative evidence path
and SHA-256, the exact expected figure set, producing scientific steps,
explicit evidence artifact bindings, plotting-data digests, plotting-spec
version, PNG SHA-256 values, dimensions and DPI, and the
Matplotlib/NumPy/Agg/font renderer contract. It does not persist UMAP coordinate
arrays, complete label vectors, duplicated confusion matrices, embeddings,
AnnData, or raw matrices.

Rendering uses direct Matplotlib `Figure` plus `FigureCanvasAgg`, with no global
pyplot dependency, Scanpy plotting wrapper, seaborn, Plotly, or Pillow. Version
1 is PNG-only and uses fixed plotting parameters and timestamp-free metadata.
Actual PNG bytes are SHA-256 hashed. Rendering is deterministic within the
recorded renderer, font, and software contract; universal byte-identical output
across arbitrary Matplotlib or font environments is not claimed.

`verify_analysis_visualizations()` freshly verifies evidence, checks manifest
schema/type/status and source identities, reopens only explicit verified source
artifacts, rederives the exact figure set and plotting-data/specification
digests, and validates PNG SHA-256, signature, dimensions, names, and exact set.
Missing, extra, renamed, tampered, and source-drifted figures fail closed.
Neither build nor verification invokes a registered scientific callable.

Publication uses a staged bundle, file and directory `fsync`, and the manifest
as completion marker. `overwrite=False` protects existing results. Failed new
publication leaves no completed bundle; overwrite uses conservative
backup/rollback behavior, failed replacement preserves the previous valid
bundle, and rollback failure fails closed. Nonempty-directory replacement is
not claimed to be universally atomic.

Milestone 7.2 remains downstream of orchestration and introduces no
`ToolRegistry` entry, recovery identity, planning-schema change, RunStore
change, provider change, scientific-tool change, EpiZoo change, or dependency.
The production registry remains exactly eight tools and planning schema v2 is
unchanged. Milestone 7.1 evidence remains the authoritative trust boundary.

Accepted validation:

- focused Milestone 7.2: 27 passed
- combined Milestone 7.1 + 7.2 report tests: 57 passed
- canonical orchestration regression: 375 passed
- complete lightweight regression: 730 passed, 6 skipped

Default validation required no network, API key, provider, GPU, checkpoint, or
EpiZoo inference. Guarded registry callables proved that visualization build
and verification execute zero scientific tools.

Deterministic scientific reporting is provided separately by Milestone 7.3.
Transferred-label UMAP remains deferred until explicit query-artifact
provenance binding exists, and LLM interpretation and an interactive Agent UI
or demo remain later work.

### Milestone 7.3 — Verified Deterministic Scientific Report

Milestone 7.3 is complete and accepted. Its public API is:

```python
build_analysis_report(
    run_result,
    evidence,
    output_dir,
    *,
    registry,
    visualization=None,
    overwrite=False,
)
```

```python
verify_analysis_report(
    run_result,
    evidence,
    report,
    *,
    registry,
    visualization=None,
)
```

`AgentRunResult` and `AnalysisEvidence` are required, visualization is
optional, and `ToolRegistry` is explicitly caller supplied. Reporting remains
post-run and outside `AgentPlan`.

The accepted trust boundary is:

1. freshly call `verify_analysis_evidence()`;
2. strictly load the verified evidence;
3. optionally freshly call `verify_analysis_visualizations()`;
4. require exact run, request, plan, and evidence identity binding;
5. construct a frozen report-fact projection;
6. generate deterministic Markdown;
7. optionally copy every verified PNG byte-for-byte;
8. persist a strict report manifest;
9. verify the report through exact regeneration.

The report layer executes no scientific tools, browses no arbitrary files,
reopens no arbitrary H5AD or scientific evaluation artifact, and does not
recompute metrics, inspect UMAP geometry or image pixels, tune or rerun an
analysis, or infer a missing scientific stage.

#### Report artifacts and frozen facts

The persisted bundle is:

```text
analysis_report/
├── analysis_report.md
├── report_manifest.json
└── figures/              # only when visualization is supplied
```

The manifest identity is:

```text
artifact_type: agent.analysis-report
schema_version: 1
report_spec_version: 1
```

It binds run/request/plan identity, the evidence path and authoritative
SHA-256, optional visualization-manifest identity and SHA-256, ordered sections
and fact records, the fact-projection SHA-256, section-to-fact bindings, figure
bindings, Markdown SHA-256, and report generator/spec identity. It contains no
embeddings, coordinates, label vectors, matrices, AnnData, or duplicated
confusion arrays.

Each frozen fact has a stable identifier such as `F0001`, assigned in evidence
topological step order and frozen per-tool field order. Every record contains
its source step, tool, field, and exact value. Unknown future evidence fields
do not silently appear, and every rendered scientific value originates from
this frozen projection. These compact attributed facts form the safe substrate
for a future constrained `ReportModel`.

#### Conditional sections and scientific wording

The fixed conditional section order is:

1. Analysis Summary
2. Dataset
3. EpiZoo Representation
4. Clustering and UMAP
5. Clustering Evaluation
6. Cell Annotation
7. Annotation Evaluation
8. Figures
9. Methods / Analysis Parameters
10. Provenance and Reproducibility

Sections appear only when their verified source exists. Inspection-only reports
are valid; clustering without evaluation has no clustering-evaluation claims;
annotation without evaluation has no accuracy claims; and no visualization
input means no Figures section or figure claims. Multiple same-kind steps
remain in deterministic workflow order.

Version 1 reports verified facts only, including cell and feature counts,
embedding dimensions, cluster counts and resolution, NMI/ARI/AMI/Homogeneity,
assignment counts and rates, accuracy, macro-F1, verified figure identities,
analysis parameters, and provenance. It does not claim that results are well
separated, excellent, reliable, biologically meaningful, or demonstrate
conservation, and introduces no arbitrary performance thresholds. Numeric
values retain their exact representation; nullable values such as
`assigned_accuracy` remain undefined/`null`, never zero.

#### Visualization, verification, and persistence

Visualization is optional. When supplied, every verified Milestone 7.2 figure
is copied in exact figure order. Source and copied PNG bytes and SHA-256 values
must match, captions use fixed factual templates, and figures are never redrawn
or interpreted. Evidence/visualization identity mismatch fails closed.

`verify_analysis_report()` freshly verifies evidence and any required
visualization, checks exact source identities, and regenerates the frozen fact
projection, section order, captions, figure bindings, exact UTF-8 Markdown, and
canonical manifest. It validates Markdown SHA-256 and byte equality, copied PNG
SHA-256 and equality with the verified source PNG, and rejects missing, extra,
renamed, or modified artifacts. Verification invokes zero scientific
callables. Unlike PNG rendering, Markdown is verified through exact byte
regeneration rather than heuristic review.

Publication uses a staging directory; fsyncs report, figure, and directory
contents; and treats the manifest as the completion marker. `overwrite=False`
protects existing results. `overwrite=True` uses conservative backup and
replacement, restores a previous valid report when replacement fails where
possible, and reports rollback failure explicitly and fail-closed. Universal
atomic replacement of a nonempty directory is not claimed.

Milestone 7.3 introduces no `ToolRegistry` entry, `AgentPlan` integration,
planning-schema change, RunStore change, recovery identity, provider change,
scientific-tool change, Milestone 7.1/7.2 semantic change, EpiZoo change, or
dependency. The production registry remains exactly eight tools, planning
schema remains v2, and RunStore remains v3.

Milestone 7.3 v1 contains no LLM-generated narrative. Future scientific
interpretation should use a separate constrained `ReportModel` that consumes
only compact report facts with stable fact IDs. Other brief future directions
are application/UI composition, an interactive Agent demo, and richer export
formats if later justified.

Accepted validation:

- focused Milestone 7.3: 29 passed
- combined Milestone 7.1–7.3 report tests: 89 passed
- Milestone 7.3 offline integration: 3 passed
- canonical orchestration regression: 375 passed
- complete lightweight regression: 762 passed, 6 skipped

Default acceptance requires no network, API key, provider, GPU, checkpoint, or
EpiZoo inference. Guarded registry callables proved that Milestone 7.1–7.3
post-run reporting invokes zero scientific tools.

### Milestone 7.4 — End-to-End Research Agent Application

Milestone 7.4 is complete and accepted. It closes Milestone 7 / Phase II with
the first coherent application flow:

```text
natural-language scientific request
→ constrained planning
→ verified durable execution
→ verified evidence
→ verified visualization when supported
→ verified deterministic scientific report
→ compact user-facing application result
```

The completed reporting and application stack is:

```text
M7.1 — Verified Analysis Evidence
M7.2 — Verified Scientific Visualization
M7.3 — Verified Deterministic Scientific Report
M7.4 — End-to-End Research Agent Application
```

#### Application API and result contracts

The public service is:

```python
ResearchAgentApplication(
    workspace_root,
    *,
    planner=None,
    registry=None,
    executor=None,
)
```

It exposes `run(request)`, `resume(run_id)`, and `cancel(run_id)`. The
application constructs and owns its `FileRunStore` beneath the approved
workspace. Scientific execution remains exclusively in the existing
`AgentRuntime`; the application composes accepted public APIs and contains no
second planner, executor, verifier, scientific pipeline, or cancellation state
system.

The immutable JSON-safe application schemas are:

- `ApplicationStatus`: PLANNED, SUCCEEDED, FAILED, CANCELLED
- `ApplicationStage`: RUNTIME, EVIDENCE, VISUALIZATION, REPORT, COMPLETE
- `ArtifactReference`: artifact type, path, and SHA-256
- `ApplicationError`: sanitized stable code, message, and stage
- `ApplicationResult`: application/run identity, application and runtime
  statuses, workspace, authoritative `AgentRunResult`, compact artifact
  references, and optional application error

Artifact references never contain embeddings, matrices, UMAP coordinates,
cell vectors, AnnData, or loaded report contents. `AgentRunResult` remains the
authoritative source of planner, preflight, scientific, verification, recovery,
and cancellation errors. The application error layer is deliberately thin;
stable codes include `APP_REQUEST_INVALID`, `APP_WORKSPACE_INVALID`,
`APP_OUTPUT_CONFLICT`, `APP_COMPOSITION_ACTIVE`, `APP_EVIDENCE_FAILED`,
`APP_VISUALIZATION_FAILED`, and `APP_REPORT_FAILED`. Arbitrary raw exception
strings are not exposed through JSON-facing `ApplicationError` values.

#### Workspace and configuration contract

The managed layout is:

```text
<workspace>/
├── run_state/
└── runs/
    └── <full-sha256-of-run-id>/
        ├── composition.lock
        ├── scientific/
        ├── evidence/
        ├── visualizations/
        └── report/
```

Raw request and run IDs never become filesystem path components. Managed names
are fixed, each run uses the complete SHA-256 of its run ID, canonical paths
must remain beneath the approved root, and managed symlinks and unexpected file
types are rejected. The application consumes only exact artifact paths returned
by accepted APIs; it does not scan directories to discover scientific results.
The implementation assumes a trusted local workspace and does not claim full
protection against a hostile actor concurrently changing the filesystem.

Scientific values—input/reference/query H5AD paths, species, checkpoint,
device, labels, and explicit scientific parameters—remain structured
`AgentRequest.inputs`. The application owns run-state and scientific,
evidence, visualization, and report output roots. Before planning it creates a
new effective `AgentRequest` containing the trusted scientific `output_dir`;
the caller's immutable request is not modified. Reserved output-root fields and
application-level `overwrite=True` are rejected. This preserves the existing
argument-provenance boundary and prevents an LLM from inventing output paths.

Post-run composition has a per-run, local, nonblocking `flock` covering:

```text
evidence
→ visualization capability/build
→ deterministic report
→ final verification
```

Concurrent composition for the same run fails safely with
`APP_COMPOSITION_ACTIVE`. This lock is neither distributed nor a new durable
state machine and does not modify `AgentRuntime` execution-lease semantics.

#### Composition, PLAN_ONLY, resume, and cancellation

Only a successful executed `AgentRunResult` enters post-run composition:

1. M7.1 evidence is built or freshly verified and reused.
2. supported visualization kinds are queried explicitly.
3. M7.2 visualizations are built or verified and reused when applicable.
4. an empty capability result proceeds normally to a figureless M7.3 report.
5. the M7.3 report is built or verified and reused.
6. final report verification must pass before application success.

If a visualization kind is supported, visualization build or verification
failure is fatal; it is never silently downgraded to a figureless report.

PLAN_ONLY returns `ApplicationStatus.PLANNED` with the validated `AgentPlan`
and preflight verification preserved in `AgentRunResult`. It executes zero
scientific tools and creates no evidence, visualization, or report.

`ResearchAgentApplication.resume(run_id)` calls the planner-free
`AgentRuntime.resume(run_id)` and deterministically recovers the same hashed
workspace. Terminal successful scientific steps are not rerun. Post-run reuse
is exact:

```text
valid existing artifact → verify and reuse
missing artifact → build
tampered/mismatched artifact → fail closed
partial/conflicting destination → output conflict
```

There is no silent overwrite, hidden repair, in-place mutation, application
manifest, or reporting cache. A new request/workspace is the v1 regeneration
path.

`cancel(run_id)` delegates directly to `AgentRuntime.cancel()`. Existing
cooperative semantics remain authoritative: active scientific calls are not
force-killed, and a cancelled scientific run creates no evidence,
visualization, or report. Reporting-stage cooperative cancellation remains
deferred.

#### Visualization capability and CLI

The new public M7.2 query is:

```python
get_supported_visualization_kinds(
    run_result,
    evidence,
    *,
    registry,
)
```

It freshly verifies the accepted evidence boundary, uses the same authoritative
figure mapping as M7.2 construction, returns deterministic workflow-ordered
kinds, and returns an empty tuple for a legitimate figureless workflow. It
invokes zero scientific callables and performs no artifact discovery.
`build_analysis_visualizations()` remains strict and still rejects calls with
no supported figure.

The standard-library `argparse` CLI is:

```text
PYTHONPATH=src python -m agent run ...
PYTHONPATH=src python -m agent resume ...
PYTHONPATH=src python -m agent cancel ...
```

It emits compact deterministic JSON. Exit code 0 means success or planned; 2
means invalid CLI/application/provider configuration; 3 means runtime or
durable-state failure; 4 means a cancelled application result; and 5 means
postprocessing failure. Deterministic/offline planning is the default. The CLI
may select the existing OpenAI, Gemini, or Groq adapters when configured, but
provider secrets remain environment-only and never become CLI arguments,
request inputs, durable state, application results, or output. No provider
factory was added to the application service, and provider adapters were not
changed. Installable console-script packaging remains deferred.

#### Safety invariants, demo, and acceptance

Natural-language planning retains all accepted boundaries. Executable values
come only from `AgentRequest.inputs` or `StepOutputRef`; providers receive
sanitized tool/schema data and no Python callables; planning schema v2 and
whole-plan preflight remain authoritative; and `ToolRegistry` remains the
executable allowlist. “Generate a report” names application postprocessing, not
a scientific tool. Arbitrary Python and shell execution remain prohibited.

Milestone 7.4 introduced no scientific tool, registry identity, planning-schema
change, RunStore-schema change, recovery identity, executor/verifier semantic
change, runtime semantic change, provider semantic change, EpiZoo change,
dependency, ReportModel, or LLM-generated scientific narrative. The production
registry remains exactly eight tools, planning schema remains v2, and RunStore
remains v3.

The canonical first demo is:

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
scientific-tool layer but is not required for this primary application demo.

Accepted validation:

- focused application service: 19 passed
- workspace: 8 passed
- CLI: 7 passed
- Milestone 7.4 integration: 3 passed
- combined Milestone 7.1–7.4 reporting/application: 128 passed
- canonical orchestration regression: 375 passed
- complete lightweight regression: 801 passed, 6 skipped

Default acceptance required no network, provider API key, GPU, checkpoint, or
real EpiZoo inference. The lightweight integration exercised the real
application, runtime, RunStore, downstream CPU tools, reporting composition,
and verification. Downstream scientific steps ran once; reporting plus terminal
resume did not increase their invocation counts. Optional real-provider plus
real-EpiZoo Fang2021 acceptance remains guarded/deferred and was not required.

Nonblocking future work is reporting-stage cooperative cancellation, optional
real-provider/EpiZoo acceptance, installable console-script packaging, stronger
hostile-filesystem-race hardening, browser or multi-turn UI, and a separately
scoped constrained `ReportModel`. None is part of Milestone 7.4.

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
- LLM-generated scientific interpretation
