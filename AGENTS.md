# Agent

Agent is an autonomous AI agent for single-cell epigenomic / scATAC-seq
analysis. Existing EpiZoo and EpiAgent foundation models are scientific
backends to reuse, not models to reimplement inside the Agent. EpiZoo is the
currently validated embedding backend; mentioning EpiAgent does not imply a
registered EpiAgent execution tool.

This is the detailed engineering contract and acceptance record. Start with
[README.md](README.md) for the overview and CLI examples. Current contracts
below take precedence over historical milestone scope and version statements.

## Architecture and engineering rules

```text
Planning → Orchestration → Scientific Tools → Foundation Models → Verified Output
```

These are functional areas, not five separate execution engines. Planning
produces an `AgentPlan`; orchestration owns validation and safe sequential
execution; registered tools wrap scientific algorithms and foundation models
where needed. Verification checks scientific results and artifacts. Persistence,
recovery, cancellation, trace, provenance, and auditability span this path.
Evidence, visualization, and deterministic reports compose verified results
post-run, outside `AgentPlan` and the scientific tool registry.

The development environment is a Linux server with an NVIDIA RTX 4090, 24 GB
VRAM, VS Code Remote SSH, Python, PyTorch, and Scanpy / AnnData.

- Never densify a complete scATAC-seq matrix or assume more than 24 GB GPU memory.
- Test on small datasets before full-scale execution.
- Reuse validated scientific logic. Modify
  `src/agent/tools/models/epizoo.py` only for a verified bug or a clearly defined
  new capability.
- Expose each scientific capability as a reusable tool with explicit inputs,
  outputs, validation, lightweight results, and informative errors.
- Keep analyses reproducible; record checkpoints, execution parameters, and
  provenance. Never silently change batch size, device, dtype, truncation,
  model, or overwrite settings for recovery.
- Do not commit biological datasets or model checkpoints to Git.
- Fail closed on invalid contracts, ambiguity, corrupt artifacts, or unsupported
  provenance. Preserve exact cell/feature order where required; never silently
  align, drop, or relabel scientific inputs.
- The immutable `ToolRegistry` is the executable allowlist. Providers receive no
  Python callables, filesystem access, or `RunStore` access. Arbitrary Python,
  shell, and user/planner-supplied R execution are prohibited in Agent plans;
  the registered DA tool uses repository-controlled pinned R scripts.
- Do not add workflow heuristics, first-match behavior, step-name inference,
  hidden step-order inference, generic input fanout, hidden scientific inference,
  or automatic workflow completion to resolve LLM ambiguity.

## Current scientific tool inventory

The following inventory is derived from `build_default_tool_registry()` in
`src/agent/orchestration/registry.py`. All twelve currently registered tools
are planner-visible and have authoritative semantic metadata. This is a
snapshot, not a permanent tool-count constraint: future coverage must be
derived from the registry.

| Registered tool | Scientific purpose / artifact |
| --- | --- |
| `inspect_scATAC` | Safe scATAC H5AD inspection and lightweight metadata |
| `epizoo_embed_cells` | Sparse preprocessing, cached EpiZoo inference, `.npy` embeddings and ordered cell IDs |
| `build_cell_neighbors` | Compact neighbor-graph H5AD in the 512-dimensional EpiZoo space |
| `cluster_cells` | Fixed-seed weighted Leiden clustering in a new compact H5AD |
| `compute_cell_umap` | Deterministic 2D UMAP in a new compact H5AD |
| `evaluate_cell_clustering` | Fixed clustering versus evaluation-only labels; NMI, ARI, AMI, Homogeneity JSON |
| `transfer_cell_labels` | Exact within-species reference-to-query kNN annotation H5AD |
| `evaluate_cell_annotation` | Fixed annotation metrics, confusion counts, and confidence diagnostics JSON |
| `validate_scATAC_feature_space` | Explicit raw sparse regulatory-count provenance manifest |
| `build_replicate_pseudobulk` | Exact sparse SUM by `(group, replicate, condition)` |
| `run_replicate_differential_accessibility` | Replicate-aware, independently verified pinned edgeR v4 quasi-likelihood DA |
| `inspect_raw_scATAC` | Bounded FASTQ/BAM intake, authoritative manifest, and independent source-aware verification |

Detailed scientific contracts, recovery identities, artifact formats, public
APIs, and accepted scientific results remain in the milestone references below.
Evidence/report projections support the existing processed-H5AD workflows;
raw-intake projection remains deferred to M10.5. Unsupported projections fail
closed; arbitrary new result fields never become report facts.

## Current scientific runtime and verification contracts

### EpiZoo input, inference, and artifacts

The registered `epizoo_embed_cells` tool supports human and mouse only. Its
input H5AD must have the EpiZoo raw feature dimension: 1,355,445 for human or
1,341,077 for mouse. At the configured filter indices, `.var_names` must
exactly match the ordered retained cCRE names in the species resource
(700,460 human or 814,020 mouse retained features). The tool does not project,
convert, align, or reorder arbitrary peak matrices into this feature space.

The tool loads sparse `X` into memory without densification; the backend does
not accept backed AnnData directly. Values must be real, finite, nonnegative,
and integer-like within absolute tolerance `1e-6`. Cells must have unique,
nonempty IDs and positive counts, with retained cCREs remaining after filtering.
The text sidecar cannot represent IDs containing newlines or carriage returns.

The installed EpiZoo backend and local species resources are required:
`cCRE_frequencies_<species>.npy` and `cCRE_filter_idx_<species>.csv`, currently
under `/home/likeyi/program/EpiZoo/data`. Resource dimensions, indices, and
retained names are validated. The lower-level backend accepts `resources_dir`;
the registered tool does not expose that parameter. Its checkpoint default is
`/home/likeyi/program/model_checkpoints/EpiZoo/pretrained_EpiZoo.pth`, with an
explicit `checkpoint_path` override supported. Loading requires an existing
checkpoint compatible with the fixed model architecture and strict state-dict
validation; only the defined deterministic loss buffers may be synthesized.
The loader computes the checkpoint SHA-256, and backend metadata includes
checkpoint and resource hashes.

The registered tool defaults to `device="cuda:0"`, loads float32 model weights,
and fixes `batch_size=4`, `max_length=8192`, `random_sample=True`,
`random_seed=0`, `use_amp=True`, `num_workers=0`, and `show_progress=False`.
AMP is enabled for CUDA. Species, checkpoint, device, and overwrite are tool
inputs; batch size, dtype, truncation, seed, and resource directory are not
registered planning parameters. Unavailable requested CUDA fails without an
automatic CPU fallback. Historical RTX 4090 acceptance is not a guarantee of
full inference support or identical resource use on every selectable device.

The process-local model cache is keyed by resolved checkpoint path, normalized
device, and dtype. A lock serializes cache misses; only successful loads are
cached. Cache identity does not include checkpoint contents or modification
time, so replacing a file at the same path does not invalidate an existing
entry. Cache clearing drops cached references without forcing CUDA allocator
cleanup.

Outputs are `<input-stem>.epizoo_embeddings.npy` with finite float32 shape
`(n_cells, 512)` and `<input-stem>.epizoo_obs_names.txt` in exact input cell
order. The tool checks these values before writing and defaults to
`overwrite=False`. Its lightweight result, persisted for durable runs, records
source/artifact paths, dimensions/dtype, finite/order flags, backend, species,
checkpoint path, and device. It does not persist the backend's full metadata,
including checkpoint/resource hashes and the fixed inference settings, or
content hashes of the two output files. Those fixed settings remain defined
by the implementation; provenance must not be described as richer than the
actual persisted result.

### Verification coverage by tool and artifact

Fresh verification applies each tool's defined checks; it is not universal
content-integrity checking. Standalone EpiZoo orchestration verification checks
result metadata, applicable inspection-dependency consistency, and artifact
existence/non-emptiness. It does not reload or hash `.npy` or ordered-ID
contents. The tool's checks before writing do not imply later content checking
on resume or evidence construction; evidence explicitly records the
`existence_and_nonempty` protection mode for these artifacts.

Transfer adds embedding/ID/reference-label content digests and annotation-file
integrity checks, but does not rerun kNN or fully hash checkpoint contents.
Clustering evaluation and annotation evaluation recompute their defined metrics.
Pseudobulk verification independently recomputes exact SUMs. DA independently
reconstructs preparation and reruns the frozen statistical contract through its
separate SHA-pinned R verifier. Reporting calls no registered scientific tool,
but fresh evidence verification can still perform these computations. Detailed
artifact-specific guarantees remain in the M6–M8 contracts below.

### Differential-accessibility runtime configuration

DA execution and independent DA verification require `AGENT_EDGER_RSCRIPT` in
the process environment. Its value must be an absolute path resolving to an
existing executable file named `Rscript`. The intended runtime is the isolated
`agent-edger` environment with the exact R/Bioconductor/edgeR and supporting
package versions listed in M8.2; an executable path alone does not satisfy the
package-compatibility checks. Missing or invalid executable-path configuration
fails with `RSCRIPT_UNAVAILABLE`. There is no automatic Conda-environment or
PATH discovery.
This is runtime/environment configuration, never an `AgentRequest` scientific
parameter or a planner-selected executable. Repository-controlled production
and verification scripts retain their existing execution boundaries.

## Current Planner contract

### Responsibility and registry authority

**LLM owns choices; Agent owns facts and deterministic consequences of choices.**
The LLM/user owns natural-language intent, tool selection, workflow/DAG
composition, genuinely ambiguous producer/source or source-port selection,
reference/query/ground-truth roles, scientific choices, and ambiguous optional
parameter scope. Scientifically valid noncanonical DAGs remain admissible.

Agent owns registered interface facts, allowlisting, legal semantic ports,
reviewed unique request-source binding, grouped semantic-port/channel expansion,
exact argument/result mappings, `StepOutputRef` construction, induced
dependencies, defaults, canonicalization, strict `AgentPlan` construction,
validation, diagnostics, and whole-plan preflight. Lowering constructs the
existing `AgentPlan`, `PlanStep`, and `StepOutputRef` types and canonicalizes
mechanically redundant graph representation. The semantic compiler
**deterministically lowers semantic choices into executable plan contracts**.
It is not merely a formatting layer.

> Unique, explicitly authorized deterministic mappings may be derived by Agent
> code; zero or multiple legitimate semantic choices must not be silently guessed.

A missing optional choice may preserve a reviewed tool default. Explicit `False`
and legal explicit `None` remain distinct from omission. Generic optional
selectors with multiple destinations require explicit scope. An explicitly
scoped structured selector may be compiler-bound without redundant LLM output
only when it has one reviewed destination and no competing source. Reviewed
embedding, transfer, and downstream parameters use this generic rule.

`ToolSpec.planning` and argument/result planning guidance describe roles,
artifact flow, source eligibility, provenance, parameter preservation, and
bindability. They do not independently authorize compiler behavior or create a
runtime semantic oracle. Reviewed `ToolSpec.semantic_planning` is authoritative:
consumer/producer ports, logical members, grouped request sources, exact
argument/result mappings, upstream permissions, and lineage constraints.
Catalog, schema projection, and compiler derive from that single authority;
there is no central tool-name-specific mapping catalog.

A normal future tool should ordinarily require implementation, execution/result
contracts, `ToolSpec` registration, reviewed semantic metadata, focused tests,
and benchmarks—not provider-specific rendering, workflow templates, or changes
to the generic compiler, executor, runtime, persistence, or recovery. Missing
reviewed metadata for required scientific/request parameters fails closed;
never infer, ignore, or silently default such parameters. Production must not
import the benchmark-only semantic oracle or use `DeterministicPlanner` as an
oracle for LLM output. Keyword/regex routing, workflow classifiers/tables,
string-similarity guesses, and reference/query positional guesses are prohibited
in LLM planning, semantic lowering/compiler behavior, planning recovery/fallback,
and automatic workflow inference. The existing regex/intent routing in the
offline `DeterministicPlanner` remains supported when that planner is selected,
including the low-level `AgentRuntime()` compatibility default; it must not
become an automatic fallback or semantic oracle for LLM output.

### Planning modes, v3/v4, and compatibility

Application/CLI new runs are LLM-first with an explicit primary
`PlanningModelProfile`; missing configuration fails before durable state
creation. An explicitly injected Planner is authoritative. Select
`DeterministicPlanner` explicitly for application offline workflows; low-level
`AgentRuntime()` retains its deterministic default for compatibility. Resume
and cancel require no planner, provider, model, SDK, or credential configuration.

Wire v4 is the default in `LLMPlanner` and application-owned LLM construction.
Wire v3 remains available as an explicit compatibility mode.
Its closed, registry-derived, tool-discriminated schema fixes the exact keyed
argument contract, with distinct request-input and upstream reference bindings.
Input names are restricted to the request; executable LLM literals are forbidden.
Reusable closed `$defs`/`$ref` schemas and a flat optional input/ref/null union
reduce repetition; the compact prompt retains the complete semantic catalog.
Schema-v2 output is not silently reinterpreted as v3.

Omitting `wire_mode` in `LLMPlanner(model)`, `planning_wire_mode` in
`ResearchAgentApplication`, or `--wire-mode` in LLM CLI runs selects v4 through
one shared default. Explicit `PlanningWireMode.V3` / `--wire-mode v3` selects
compatibility mode; explicit v4 remains supported. The application's `None`
sentinel distinguishes omission from conflicting explicit configuration.
Deterministic CLI mode rejects this LLM-specific option. Low-level
`AgentRuntime()` still defaults to `DeterministicPlanner`; this separate offline
construction behavior does not change normal application LLM planning.
There is no schema auto-detection, combined schema, automatic version switching,
v4-to-v3 hidden fallback, or cross-version recovery fallback. Failover retains
the selected wire mode. Explicit v3 prompt/parser/planner and plan identity
behavior remain compatible with the accepted v3 path.

```text
AgentRequest
→ registry-driven semantic planning catalog/prompt
→ provider-neutral PlanningModel.complete()
→ semantic wire v4
→ strict parser
→ SemanticPlanCandidate
→ registry-derived deterministic semantic compiler
→ existing strict AgentPlan
→ whole-plan preflight / AgentRuntime / PlanExecutor / verifier
```

The semantic catalog/prompt presents tool purpose, consumer/producer ports,
request-source selectors, accepted upstream semantic types, lineage, scientific
choices/defaults/constraints, and request input names/basic types. Structured
input values are excluded (including paths, labels, conditions, checkpoints,
output roots, arrays, and nested values); the natural-language request itself
is passed to the model. The interface omits raw Python argument inventories,
execution member/result-field names, binding objects, `StepOutputRef`, and
reference-induced dependency serialization. The full scientific context is
retained, including feature-space layer, coordinate, and semantics-metadata
parameters. This can make the v4 prompt alone larger than v3; focused acceptance
compares combined prompt/schema size instead. V4 is the default LLM wire.

Wire v4 contains only a plan/unsupported decision, step identities, selected
tools, semantic sources, and explicit control-only dependencies. The parser
retains closed shapes, duplicate-key/nonfinite rejection, and response/tree/
step/source safety limits. Names and interface projections are registry- and
request-derived, with no permanent hard-coded tool set. Semantic candidates do
not grant execution authority: compiler legality and whole-plan preflight are
still required.

### Static target-port projection and Groq compatibility

For each selected tool, provider-facing v4 projects legal target enums directly
from `ToolSpec.semantic_planning.consumer_ports`. The final source representation
is a closed outer object with target legality separated from the shared inner
source-kind discriminator:

```json
{
  "target": "dataset",
  "source": {"kind": "input", "input": "input_path"}
}
```

Inner variants are `{"kind":"input","input":"<request selector>"}`,
`{"kind":"step","step":"<producer step ID>"}`, or
`{"kind":"step_port","step":"<producer step ID>","source_port":"<port>"}`.
Input selectors are constrained to available request names, never values.
The selected producer's source-port compatibility remains a compiler check;
projecting target legality does not solve genuinely ambiguous source choices.
Tools with zero consumer ports accept only empty source arrays.

The first target-enum implementation put both `kind` and `target` into the
flat source alternatives inside Groq `anyOf`. Groq rejected the schema with
`discriminator_multiple_candidates`. Probes of outer-property factoring, enum
wrappers, explicit discriminator hints, and exact-pattern alternatives also
failed. The accepted nesting separates tool-specific target legality from the
inner `kind` discriminator, with shared closed definitions, no regex constraints,
and no provider-specific schema fork or semantic adapter behavior.

Schema version remains 4. The parser accepts historical flat v4 source payloads
as well as the nested provider shape, applying the same strict validation;
mixed or malformed shapes are rejected. Unknown targets fail early in the
parser, and the compiler retains authoritative `UNKNOWN_TARGET_PORT` defense
for directly constructed candidates. No new planner layer, workflow heuristic,
automatic workflow completion, or additional semantic inference was introduced.

### Provider abstraction, recovery, diagnostics, and benchmark

Immutable `PlanningModelProfile` describes deployment/model configuration;
`PlanningModelFactoryRegistry` constructs adapters only. OpenAI, Gemini, Groq,
and custom `PlanningModel` injection are supported. The factory does not inspect
intent, route, retry, repair, rank models, or execute science. No production
model is hard-coded, and credentials remain environment/provider concerns,
never profile, request, diagnostic, or benchmark data.

V3 and v4 use the existing `PlanningRecoveryCoordinator`: one initial call,
either one same-profile transport retry for explicit transient provider failure
or one complete same-profile Plan repair for an objectively invalid candidate,
and at most one explicitly configured secondary-profile failover as the final
call. Retry and repair are mutually exclusive; failover cannot recover again;
the ceiling is three logical `PlanningModel.complete()` calls. Built-in SDK
retries are disabled; custom models are responsible for hidden internal behavior.
Application-owned LLM construction enables this bounded path automatically.
Reliable HTTP 413 is terminal `PROVIDER_REQUEST_TOO_LARGE`, not retried or
repaired or treated as retryable HTTP 429 rate limiting.

Recovery is cancellation-aware and checkpoints sanitized diagnostics/decisions
before further calls. Only the final authoritative-preflight-passing plan is
durable. Failed candidates and raw responses are never the stored plan;
interrupted planning is not automatically replayed, and resume after plan
persistence is planner-free. This is separate from scientific same-step retry:
`AgentError.recoverable` retains its M5.3 static eligibility meaning.

Planning diagnostics use schema v3 for wire v3 and schema v4 for wire v4.
Both record sanitized attempt order, profile/provider and safe
model provenance/digests. V4 parser/compiler diagnostics can identify safe step,
producer step, target/source port, tool, and input names. Persisted diagnostics
exclude raw prompts, structured input values/paths, provider responses,
exception prose, HTTP bodies/headers, request IDs, credentials, and tokens.
Run-state schema remains v3; v3 diagnostic and recovery behavior is unchanged.

The existing `benchmarks/planner/` harness is explicitly pinned to wire v3,
including replay, binding scoring, and diagnostics; it is not a v4 benchmark.
V4 correctness is covered separately by semantic-v4 acceptance tests.
The deterministic offline harness uses synthetic requests
and PLAN_ONLY with zero scientific calls. Report schema v4 separates hard
semantic correctness from canonical workflow conformance using structural,
nonpositional matching, and distinguishes first-attempt, transport-recovered,
repair-recovered, and configured-failover success. Unsafe provenance swaps,
invented bindings, lost parameters, broken artifact flow, and unsupported
substitutions fail even if preflight accepts their shape. Alternative valid
DAGs remain accepted. Live-provider availability is optional evidence, not an
offline acceptance dependency. Deterministic fallback, automatic model
routing/ranking, prompt-based routing, and tool filtering are not implemented.

## Current accepted validation and limitations

The v4 default-switch acceptance passed 821 focused tests, 1162 broad tests
with 3 skipped, and the separately gated DA PLAN_ONLY test. Live Groq
`openai/gpt-oss-120b` passed inspection, the canonical five-tool downstream
workflow, and feature-space planning with all six optional parameters while
omitting `--wire-mode`; diagnostics confirmed wire v4. All live attempts
executed zero scientific tools. Explicit `--wire-mode v3` reached Groq but
returned HTTP 413 / `PROVIDER_REQUEST_TOO_LARGE`. Its prompt/schema matched
the starting HEAD `9ac802593f2abd837cabf3f3efc1bc33fa3b4738` exactly, and offline
v3 compatibility passed. Live v3 success was unavailable under the current
provider request-size limit; this is a non-blocking provider limitation, not
a regression introduced by the default switch. Closeout conclusion:
`READY_TO_COMMIT_V4_DEFAULT_SWITCH`.

The accepted static target-port implementation baseline is
`c053c6543f839f72c11f36bcf2236a8fe822eb9a` (static target-port follow-up).
The following pre-push validation totals come from the completed follow-up's
acceptance record supplied at the documentation-consolidation checkpoint;
they are historical results for that follow-up. The committed code
and tests confirm the nested schema, early target validation, historical flat
compatibility, provider transport, and the then-default v3 behavior.

| Automated acceptance | Result |
| --- | --- |
| Five focused suites | 203 passed |
| Orchestration / providers / benchmarks | 1006 passed |
| Full lightweight regression | 1429 passed, 54 skipped |
| Independent JSON Schema payload checks | 7648 passed |
| `git diff --check` | Passed |

Separate live Groq smoke acceptance accepted a direct strict-schema probe,
application-level v4 inspection (`PLANNED`, zero scientific execution), and a
complete five-tool downstream PLAN_ONLY plan:
`inspect_scATAC` → `epizoo_embed_cells` → `build_cell_neighbors` →
`cluster_cells` → `compute_cell_umap`. No target-port failure occurred in that
accepted live verification. Earlier closeout used Groq `openai/gpt-oss-120b`
for inspection, complete downstream DAGs, canonical and optional-heavy transfer,
paired DA with covariates, grouped channels, scoped parameters, preflight, and
zero scientific execution. OpenAI/Gemini transported the same interface in
mocked tests; live checks were not run without credentials/configuration.

The subsequent feature-space semantic parity follow-up adds optional,
request-only mappings for `layer_key`, `feature_chrom_key`, `feature_start_key`,
`feature_end_key`, `coordinate_system`, and `semantics_metadata_key`. The existing
scientific validator retains conditional layer, coordinate, and metadata checks;
no compiler, scientific backend, or default wire-mode change was needed.
Offline acceptance passed 378 focused tests, 1119 broader orchestration/provider/
benchmark/application tests, and the separately enabled gated DA PLAN_ONLY test.
The final unchanged-tree acceptance pass also passed 283 focused checks.

Final live acceptance for this parity patch was completed manually in the server
terminal and reported by the user at closeout. Groq `openai/gpt-oss-120b`, with
explicit `--wire-mode v4`, passed inspection, the complete five-tool downstream
workflow above, and parameter-heavy feature-space validation using an existing
synthetic fixture with all six newly mapped parameters. The feature-space case
returned `status=PLANNED`, `run_status=PLANNED`, `error=null`, and only
`validate_scATAC_feature_space` in `tool_names`. All three checks were PLAN_ONLY
with zero scientific execution. This records live provider planning acceptance,
not biological-data or DA statistical acceptance. V3 was still the default at
that parity-patch checkpoint; the current default is v4.

Resolved interface issues are v3's model-authored mechanical binding burden
(addressed by v4), statically invalid target generation (constrained by schema
projection and rejected by the parser/compiler), and Groq's discriminator
incompatibility (resolved by nesting). Hosted models can still choose wrong
sources/source ports or emit incomplete or unsupported candidates; successful
smokes do not establish perfect repeated consistency. Explicit scoped inputs
resolve deterministic optional-scope ambiguity without solving every semantic
choice. V4 is the default LLM wire. These limitations are not executor/preflight,
allowlist, persistence, or recovery defects and do not justify workflow guessing.
Future hardening should follow new empirical failures or requirements, not a
pursuit of perfect hosted-model consistency. Provider availability, rate limits,
and authentication remain environment concerns.

M8.2 guarded real-data acceptance remains outstanding for lack of eligible local
human/mouse raw counts with genuine replicated conditions; the detailed audit
and deferred statistical scope are preserved below. M8.1's original guarded
real-data acceptance and M7.4's combined real-provider/EpiZoo demo were not run.
Previously accepted Fang2021 scientific runs and later live PLAN_ONLY checks
must not be conflated with those gates.

Still deferred: reporting-stage cancellation, installable console-script
packaging, stronger hostile-filesystem-race hardening, browser/web or multi-turn
UI (including Streamlit, Gradio, and a persistent REPL), interactive demo, richer
exports, transferred-label UMAP with explicit query provenance, per-class F1 or
confidence figures, SVG, and a separately constrained `ReportModel`/LLM-generated
scientific interpretation. Do not implement multi-agent architecture,
literature/ENCODE retrieval, RAG, cCRE perturbation, or variant interpretation
without a separately defined capability. Detailed nonblocking lifecycle
follow-ups remain beside their M5 contracts; their historical acceptance does
not imply those follow-ups were implemented.

## Milestone and Post-M9 development history

Milestones 1–9 and the Post-M9 interface-hardening cycle are complete, subject
to the explicitly guarded scientific acceptance gates above. This history
records capability introduction; the current contract supersedes interim
scope exclusions, registry sizes, and planning-version defaults.

| Milestone | Capability established |
| --- | --- |
| M1 | Validated EpiZoo backend with exact manual-pipeline parity |
| M2 | Reusable inspection/embedding tool layer and process-local model caching |
| M3 | Typed plans, immutable allowlist, safe sequential executor, references, verification, errors, bounded retries, traces, PLAN_ONLY |
| M4 | Provider-neutral natural-language `LLMPlanner`; Groq wire-v2 PLAN_ONLY acceptance |
| M5.1 | Opt-in durable state, execution lease, fingerprints, checkpointing, planner-free resume |
| M5.2 | Cooperative cancellation and separately persisted intent; run-state v2 |
| M5.3 | Error catalog, recovery dispositions, immutable policy provenance, run-state v3 |
| M6.1 | Neighbors, Leiden, UMAP from compact EpiZoo artifacts |
| M6.2 | Evaluation-only clustering metrics |
| M6.3 | Within-species reference-to-query label transfer |
| M6.4 | Fixed-annotation evaluation and confidence diagnostics |
| M7.1 | Verified compact AnalysisEvidence |
| M7.2 | Verified deterministic PNG visualization |
| M7.3 | Verified deterministic Markdown reports and attributed frozen facts |
| M7.4 | Python application, managed workspace, post-run composition, one-shot CLI; Phase II closeout |
| M8.1 | Raw feature provenance and exact replicate-aware sparse pseudobulk |
| M8.2 | Pinned edgeR DA, independent verification, compact figureless evidence/report |
| M9.1 | Calibrated hard-semantic offline Planner robustness benchmark |
| M9.2 | Structured sanitized planning diagnostics (v3 for wire v3; v4 for wire v4) |
| M9.2.5 | Immutable model profiles and adapter-only factory registry |
| M9.3 | Tool-discriminated planning wire v3 and composable semantic guidance |
| M9.4 | Bounded transport retry / full-plan repair / configured final failover |
| M9.4.5 | LLM-first new-run policy; explicit deterministic mode; provider-free resume/cancel |
| M9.5 | Final LLM robustness acceptance and Milestone 9 closeout |
| Post-M9 compatibility | Reusable closed v3 schemas, compact complete prompt, terminal HTTP 413 classification |
| Post-M9.1/M9.2 | Initially disconnected semantic candidate/compiler for authorized binding and dependency lowering |
| Post-M9.3 | Registry-attached authoritative metadata replaced central tool-specific compiler mappings |
| Post-M9.3.3 | Initially disconnected semantic wire-v4 schema/parser foundation |
| Post-M9.3.4 | Initially disconnected registry-derived semantic catalog/prompt and privacy tests |
| Post-M9.4.1 | Explicit typed opt-in v4 integration in `LLMPlanner`, preserving failover wire mode |
| Final interface closeout | Scoped optional binding, semantic recovery/diagnostics, generic provider transport, offline/live acceptance |
| CLI wire selection | Application/CLI v4 opt-in exposed; omission stays v3; deterministic conflicts rejected |
| Static target-port follow-up | Tool-specific target-enum projection and early parser defense |
| Groq compatibility correction | Nested target/source shape resolves competing discriminators; historical flat-v4 parsing retained |
| Feature-space semantic-v4 parity | Six optional feature-space mappings completed through reviewed registry metadata; offline and explicit-v4 Groq PLAN_ONLY acceptance |
| Semantic-v4 default migration | Shared v4 default for LLM/application/CLI planning; explicit v3 compatibility and v3 benchmark retained; omitted-wire Groq PLAN_ONLY acceptance |
| M10.1 | Phase II raw scATAC preprocessing foundation: versioned intake domain contract and manifest infrastructure |
| M10.2 | Read-only bounded FASTQ intake, declared 10x ATAC layouts, source reinspection, and M10.1 manifest population |
| M10.3 | Read-only bounded BAM intake through optional pysam/htslib, conservative assembly/barcode evidence, and source reinspection |
| M10.4 | Public raw-intake registry/semantic planning integration, deterministic publication, independent verification, and durable Runtime execution |

M9 final offline acceptance covered inspection, embedding, downstream analysis,
clustering evaluation, label transfer/evaluation, pseudobulk, and both fixed-
artifact and raw-to-pseudobulk DA. It checked request/ref provenance,
dependencies, scientific parameter preservation, reference/query separation,
evaluation-only truth, alternative valid DAGs, terminal rejection, and PLAN_ONLY
zero execution. Benchmark report v4, diagnostic v3, and run-state v3 were kept.

Post-M9 acceptance checkpoints below preserve earlier validation provenance;
they are not competing current totals. V4 foundation measurements found
representative schemas about 80% smaller than v3 and complex downstream,
transfer, and DA responses about 61–73% smaller. V4 removed model-authored raw
argument dictionaries, binding discriminators, result-field names,
`StepOutputRef`, duplicated dependency/reference structures, and large nullable
optional inventories. The original mechanical failure class did not recur in
closeout live checks. Earlier omitted/incorrect sources were reduced by scoped-
input hardening; observed unknown targets motivated the later schema projection.

| Checkpoint | Accepted automated validation |
| --- | --- |
| Post-M9.1/M9.2 compiler | Focused 42; orchestration/providers/benchmarks 785; lightweight 1201 passed, 54 skipped; existing v3 planner 77 |
| Post-M9.3 registry | Focused 75; orchestration/providers/benchmarks 818; v3 planner 77; lightweight 1234 passed, 54 skipped |
| Post-M9.3.3 wire foundation | Wire 53; compiler/registry 75; v3 planner 77; orchestration/providers/benchmarks 871; lightweight 1287 passed, 54 skipped |
| Post-M9.3.4 prompt | Prompt 19; wire 53; compiler/registry 75; v3 planner 77; orchestration/providers/benchmarks 890; lightweight 1306 passed, 54 skipped |
| Post-M9.4.1 integration | V4 integration 26; v3 planner 77; semantic suites 147; orchestration/providers/benchmarks 916; lightweight 1332 passed, 54 skipped; production-runtime PLAN_ONLY zero scientific calls |
| Final interface closeout | Registry 23; compiler 57; prompt 21; wire 54; v4 planner 26; v4 recovery 11; v4 transport 7; v4 benchmark acceptance 30; v3 planner 77; M9 recovery 117; all benchmarks 69; lightweight 1388 passed, 54 skipped |

The early disconnected/compiler-only and no-live-acceptance descriptions applied
only at those checkpoints. V4 integration, semantic repair/diagnostics, and live
Groq acceptance are now complete; the static target-port/Groq correction was
followed by feature-space parity and the v4-default migration recorded above.
No scientific tool, strict internal plan, executor, runtime, persistence/resume,
cancellation, verification, recovery budget, or provider-specific semantic
engine was added by that correction.

## Detailed accepted contracts and scientific acceptance

The sections below retain precise APIs, scientific rules, artifact trust
boundaries, validation counts, environment observations, and deferred items.
Unless stated otherwise, a milestone's test counts describe its acceptance
checkpoint, not the latest regression total. Historical registry expansion was
2 tools in M2–M5, 5/6/7/8 in M6.1/6.2/6.3/6.4, 8 through M7, 10 in M8.1, and
11 in M8.2; the current inventory above is authoritative. Planning wire v2 was
the M4–M8 contract and was superseded by the current v4/default and v3/compatibility
interface without replacing `AgentPlan` or scientific execution contracts.

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

M2 deliberately scoped implementation to inspection and embedding. Natural-language
planning, annotation, clustering, UMAP, and reports were added in later milestones;
retrieval, RAG, and multi-agent orchestration remain outside current scope.

### Milestone 3 — Agent orchestration core

Milestone 3 is complete. The Agent now supports natural request
representation, deterministic bootstrap planning, typed executable plans, an
explicit immutable tool registry, safe sequential execution, dependency/output
references, orchestration verification, structured errors, bounded
same-argument retry, execution traces, and PLAN_ONLY mode.

M3 acceptance used the original inspection/embedding vocabulary; real
end-to-end acceptance exercised `inspect_scATAC` through `AgentRuntime`.
M4 subsequently supplied natural-language planning. The prohibition on arbitrary
Python and shell execution remains authoritative.

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

Terminal `AgentRuntime.resume()` returns the immutable stored terminal result;
it does not itself rerun artifact verification. Nonterminal resume revalidates
successes before reuse, and downstream evidence/application composition freshly
verifies artifacts even after terminal resume.

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

M5.2 introduced persisted run-state schema version 2 (superseded by M5.3 v3).
Valid Milestone 5.1 version-1
records remain readable and are checked against the committed Milestone 5.1
lifecycle, error-category, and trace-event vocabularies. Loading v1 does not
rewrite it; the next legitimate update wrote v2 at M5.2 and now writes v3.
A v1 record cannot contain
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

Milestone 6.1 is complete and accepted.

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
executable literals. Whole-plan preflight remains authoritative, PLAN_ONLY
executes zero tools, and downstream tools are
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
resume plus acceptance artifact revalidation reran no scientific tools.

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
metrics. Whole-plan preflight, the registered-tool allowlist, arbitrary
Python/shell prohibition, PLAN_ONLY zero-tool behavior,
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
arguments are omitted unless present in structured request inputs.
Whole-plan preflight, the executable allowlist, and PLAN_ONLY
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
never reach inspection, embedding, or transfer. PLAN_ONLY executes zero
scientific tools, and the LLM receives
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

Schema v1 began with compact, explicit whitelisted projections for the eight
M7 tools; M8 added projections for feature validation, pseudobulk, and DA.
Unsupported future tools fail closed, and arbitrary
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
registration. M7.1 retained the then-existing eight-tool registry. There
is no new recovery identity, recovery-policy change, planning-schema change,
RunStore-schema change, orchestration change, provider change, or EpiZoo change.
It introduces no visualization, narrative generation, or LLM exposure of
scientific payloads. Visualization is provided separately by Milestone 7.2,
while LLM narrative / `ReportModel` integration remains later work and
deterministic reports are supplied by M7.3.

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
LLM narrative reporting, and interactive UI are deferred; deterministic
reporting is supplied by M7.3. Transferred-label UMAP
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
M7.2 retained the then-existing registry and planning wire v2. Milestone 7.1
evidence remains the authoritative trust boundary.

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

`AgentRunResult` and `AnalysisEvidence` are required; `AnalysisVisualizations`
is optional, and `ToolRegistry` is explicitly caller supplied. Reporting remains
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

The current fixed conditional section order extends M7.3 with the M8 sections:

1. Analysis Summary
2. Dataset
3. EpiZoo Representation
4. Clustering and UMAP
5. Clustering Evaluation
6. Cell Annotation
7. Annotation Evaluation
8. Regulatory Feature Space (added by M8.1)
9. Replicate-aware Pseudobulk (added by M8.1)
10. Replicate-aware Differential Accessibility (added by M8.2)
11. Figures
12. Methods / Analysis Parameters
13. Provenance and Reproducibility

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
dependency. At this milestone the registry had eight tools and planning wire
was v2; RunStore remains v3.

Milestone 7.3 v1 contains no LLM-generated narrative. Future scientific
interpretation should use a separate constrained `ReportModel` that consumes
only compact report facts with stable fact IDs. Application composition was
completed in M7.4; interactive UI/demo and richer
export formats remain future directions if justified.

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
    primary_planning_profile=None,
    recovery_planning_profile=None,
    planning_model_factory_registry=None,
    planning_recovery_policy=None,
    planning_wire_mode=None,
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

Application-owned new runs require an explicit primary LLM profile unless an
explicit Planner, including `DeterministicPlanner`, is injected. Missing LLM
configuration is rejected before durable run-state creation. Resume and cancel
do not require planning configuration.

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
postprocessing failure. Since M9.4.5, new runs default to LLM planning with
explicit provider/model configuration; deterministic/offline planning requires
explicit selection. OpenAI, Gemini, and Groq adapters are supported, but
provider secrets remain environment-only and never become CLI arguments,
request inputs, durable state, application results, or output. M9 subsequently
added profile/factory-based application construction while keeping provider
adapters separate from semantic planning. Installable console-script packaging
remains deferred.

#### Safety invariants, demo, and acceptance

Natural-language planning retains all accepted boundaries. Executable values
come only from `AgentRequest.inputs` or `StepOutputRef`; providers receive
sanitized tool/schema data and no Python callables; whole-plan preflight
remains authoritative; and `ToolRegistry` remains the
executable allowlist. “Generate a report” names application postprocessing, not
a scientific tool. Arbitrary Python and shell execution remain prohibited.

Milestone 7.4 introduced no scientific tool, registry identity, planning-schema
change, RunStore-schema change, recovery identity, executor/verifier semantic
change, runtime semantic change, provider semantic change, EpiZoo change,
dependency, ReportModel, or LLM-generated scientific narrative. At M7.4 the
registry had eight tools and planning wire was v2; RunStore remains v3.

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

### Milestone 8.1 — Feature-space and replicate-aware pseudobulk foundation

Milestone 8.1 is complete and accepted. Regulatory accessibility analysis now
returns to the immutable raw scATAC H5AD rather than treating compact M6/M7
artifacts as a regulatory feature matrix.

Its public scientific APIs are:

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

Feature validation accepts only sparse CSR/CSC `X` or an explicitly named
sparse layer. The structured request must assert one of fragment counts,
insertion counts, binary accessibility, or normalized/continuous signal;
normalized/continuous input is explicitly ineligible for pseudobulk. Fragment
and insertion identity is never inferred from values. An optional configured
raw `uns` field may corroborate the declaration and must agree exactly. Binary
accessibility is additionally content-validated as exact zero/one values. All
accepted matrices must be finite, nonnegative, and integer-valued.

M8.1 v1 supports only human/hg38 and mouse/mm10. Coordinates are optional:
`coordinate_source="none"` is valid and explicitly recorded. When coordinates
come from named `.var` columns, column identities, coordinate system, values,
and ordered digest are strictly validated and provenance-bound. Coordinates
are never parsed from feature names or otherwise inferred.

The canonical schema-v1 feature-space JSON manifest records the complete source
H5AD SHA-256, resolved matrix source/layer, declared semantics and assertion
source, species/assembly, exact dimensions/storage/dtype/nnz, ordered cell and
feature digests, canonical sparse-matrix digest, optional coordinate digest,
software versions, and a domain-separated feature-space identity. It contains
no complete cell, feature, coordinate, or matrix vectors. The raw file is
hashed before and after validation and is never modified.

Pseudobulk metadata semantics are strict. `replicate_key` is biological
replicate/subject identity and may span conditions. The exact unit is
`(group, replicate, condition)`. Replicate, condition, and covariates always
come from raw `.obs`. Group comes only from raw `.obs` or the fixed
`predicted_label` of an accepted M6.3 annotation. Verified annotation groups
require exact raw cell identity/order and every cell assigned; arbitrary H5AD,
CSV, Leiden, intersection, sorting, reordering, and silent dropping are
prohibited. Covariates are preserved only when constant within each
replicate-condition pair. M8.1 imposes no later DA design/rank/replication rule.

Aggregation is exact sparse SUM only. Units are ordered by first occurrence in
the authoritative source cell order, no low-cell unit is removed, and every
cell is assigned once. Pseudobulk IDs use a domain-separated canonical digest
of the authoritative feature-space identity plus group, replicate, and
condition. Production aggregation uses chunked sparse membership-matrix
multiplication with checked int64 accumulation and a checked overflow fallback.
It performs no normalization, cell or feature filtering, feature intersection,
sorting, reindexing, remapping, liftOver, or coordinate inference.

The schema-v1 pseudobulk artifact is:

```text
<source-stem>.replicate_pseudobulk.h5ad

rows = first-occurrence-ordered (group, replicate, condition) units
columns = exact original ordered regulatory features
X = sparse CSR int64 SUM counts
layers/obsm/obsp/varm/varp/raw = empty

obs:
  group, replicate, condition
  n_cells, first_cell_index, library_size
  covariate_000... in requested order

var:
  exact feature IDs
  optional exact chrom, start, end only when supplied

uns:
  agent_milestone8_pseudobulk = schema-v1 provenance
```

The artifact provenance binds the feature manifest and identity, raw source,
matrix semantics and assertion source, species/assembly, metadata keys and
optional M6.3 annotation digest, ordered source metadata and unit assignments,
feature identity/order, optional coordinates, exact pseudobulk matrix, counts,
library sizes, aggregation settings, validation flags, and software versions.
The complete artifact SHA-256 remains in the lightweight result/durable step
result to avoid a self-referential file digest.

Verification never invokes either new scientific callable. It independently
rehashes and reconstructs the raw feature space and metadata, deterministic
unit order and IDs, cell counts, first-cell positions, covariates, features,
coordinates, and library sizes. Every pseudobulk SUM is recomputed with Python
integer row maps, an algorithm distinct from production sparse matrix
multiplication, and compared exactly row by row as canonical CSR. Source and
artifact files are checked for mutation across verification. No complete
matrix densification is permitted.

The deterministic planner produces:

```text
validate_feature_space
→ build_pseudobulk
```

The second step receives `feature_space_path` through `StepOutputRef`. All
other executable values come from structured `AgentRequest.inputs`; the LLM
planner receives descriptions and schemas but no executable Python callable.
Whole-plan preflight, RunStore schema v3, durability, cancellation, and bounded
recovery semantics remain authoritative. PLAN_ONLY
executes zero scientific tools.

The recovery identities are `validate-scatac-feature-space-v1` and
`build-replicate-pseudobulk-v1`, both with no automatically retryable
scientific codes. Durable resume independently revalidates a completed feature
manifest/raw source before restoring its reference; changed or missing evidence
blocks downstream pseudobulk. Cancellation observed after feature validation
prevents pseudobulk invocation, and semantic recovery-policy drift is rejected
before new scientific execution.

AnalysisEvidence and deterministic reports remain schema v1 with explicit
additive support for both tools. Existing envelope/grammar and M1–M7 behavior
remain unchanged. M8.1 has no supported visualization kind, so the application
correctly produces a verified figureless report. The complete M7.4 application
path, terminal resume, and post-run artifact reuse are supported.

Accepted validation:

- focused M8.1 and adjacent integration/regression selection: 239 passed
- canonical orchestration/provider/lifecycle regression: 405 passed
- complete lightweight regression: 832 passed, 6 skipped
- realistic backed-sparse acceptance: 1,024 cells by 50,000 features, backed
  CSC raw input, 64 pseudobulks, exact production plus independent verification,
  with CSR/CSC densification methods guarded against invocation

The realistic acceptance required no network, provider API key, GPU, model
checkpoint, or biological dataset. No local Fang2021/raw scATAC source was
available, so guarded real-data M8.1 acceptance was not performed. Existing
nonblocking Louvain/`pkg_resources`, Scanpy Louvain deprecation, TBB/Numba, and
duplicate-test-ID warnings remain unchanged and do not affect M8.1.

M8.1 does not implement edgeR, TMM, differential accessibility, DA feature
filtering, genomic annotation, motif analysis, regulatory interpretation, or
any later Milestone 8 capability. Future coordinate-dependent interpretation
must fail closed when coordinates are absent, and binary-accessibility
pseudobulk must not silently enter a sequencing-count DA model.

### Milestone 8.2 — Replicate-aware differential accessibility

Milestone 8.2 is complete with guarded real-data acceptance outstanding because
no scientifically eligible local dataset was available. Its public scientific
API is:

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

The production `ToolRegistry` contains exactly eleven scientific tools. The new
recovery identity is
`run-replicate-differential-accessibility-edger-ql-v1`; execution has one actual
attempt and no M8.2 scientific/backend error is automatically retryable.
M8.2 did not change planning wire v2; current planning versions are documented
above. RunStore remains schema v3.

The authoritative scientific input is a verified Milestone 8.1 sparse int64
SUM-count pseudobulk. DA never uses individual cells as replicates. Independent
designs require at least two disjoint biological replicates per condition and
emit `DA_LOW_REPLICATION` when either side has exactly two. Paired designs
require at least three complete biological pairs. One-cell pseudobulk units are
retained and emit `DA_ONE_CELL_PSEUDOBULK`. Selection, exclusion reasons,
ordered categorical/numeric covariates, condition coding, numeric design,
contrast, rank, estimability, and residual-DF checks are fixed by M8.2-A.

The two-condition contrast is numerator minus denominator, with optional
ordered additive categorical or numeric covariates.

The fixed M8.2-B backend uses R 4.6.1, Bioconductor 3.23, edgeR 4.10.4,
BiocManager 1.30.27, limma 3.68.5, locfit 1.5.9.12, statmod 1.5.2, and lattice
0.23.1 in the isolated `agent-edger` runtime. It applies condition-based
`filterByExpr`, subsets with recalculated library sizes, TMM normalization,
robust edgeR v4 `glmQLFit`/`glmQLFTest`, and BH correction. Users and planners
cannot provide R code, formulas, shell strings, backend paths, or statistical
parameters. The compact DA H5AD is figureless and contains no count matrix,
duplicated counts, graph, embedding, or large result table outside `.var`.

Authoritative verification is independent of production M8.2 code. It first
rehashes and independently verifies the M8.1 feature-space/raw-source binding
and exact pseudobulk SUMs. A separate Python implementation reconstructs all
M8.2 preparation identities and digests. The separately SHA-pinned
`src/agent/orchestration/r/edger_ql_verify_v1.R` script independently reruns the
same frozen edgeR contract without sourcing or invoking the production R
script. Exact structural/discrete/digest comparisons, tolerance-bounded edgeR
numeric comparisons, independent Python BH recomputation, exact package-stack
compatibility, and before/after source/artifact hashes are required.

Deterministic planning supports both a fixed verified pseudobulk → DA plan and
raw scATAC → feature validation → pseudobulk → DA. The chained DA path uses a
`StepOutputRef`; mixed raw and fixed-pseudobulk sources are rejected. At M8.2
the LLM planner used wire v2; current v3/v4 planning preserves the request/ref
executable-value boundary. PLAN_ONLY
starts neither Python science nor R.

Verified DA success is durably checkpointed. Nonterminal resume independently
revalidates the source and artifact and reuses the result without reinvoking
production DA. Drift or an incompatible R stack blocks reuse; stale RUNNING
work requires manual reconciliation. Cooperative cancellation before DA starts
prevents it; cancellation during R lets the current call finish, verify, and
checkpoint before cancellation becomes authoritative.

AnalysisEvidence schema v1 adds only compact verified comparison, design,
warning, filtering/normalization/backend/version, artifact, and digest facts.
It excludes feature statistics, sample vectors, designs, normalization vectors,
and peak lists. The deterministic schema-v1 report adds a factual
"Replicate-aware Differential Accessibility" section. No M8.2 visualization,
significant-peak selection, interpretation, or causal claim is generated; the
application path is a verified figureless report.

Accepted validation:

- focused M8.2-A/B/C plus registry/planner selection: 244 passed
- adjacent reporting/application regression: 132 passed
- adjacent M8.1 regression: 27 passed
- canonical orchestration/provider/lifecycle regression: 503 passed
- complete lightweight regression: 903 passed, 54 skipped

The guarded local-data audit did not identify a valid supported dataset. Local
Fang2021 and PBMC count data lack a genuine two-condition replicate design; the
local BMMC candidate has real donors and conditions but stores normalized
continuous mixed GEX/ATAC values rather than eligible raw accessibility counts;
the replicated rice heat-shock dataset is outside the human/mouse contract. No
metadata was fabricated, no external data was downloaded, and real-data M8.2
acceptance remains an explicit review gate.

Deferred scope includes binary-accessibility inference, DESeq2, limma-voom,
mixed models, multi-condition contrasts, interactions, time courses,
effect-size shrinkage, adaptive filtering, user-configurable edgeR parameters,
peak-to-gene or genomic annotation, motifs, pathways, regulatory networks,
volcano/MA plots, biological interpretation, perturbation analysis, and
mutation analysis.

### Milestone 10.1 — Raw scATAC intake contract and common infrastructure

M10.1 is complete and accepted. Phase II has started toward raw scATAC
preprocessing capabilities with a standalone intake domain contract and manifest
infrastructure in `src/agent/tools/data/raw_scatac_manifest.py`.

| Versioned identity | Accepted value |
| --- | --- |
| Manifest artifact | `agent.raw-scatac-intake`, schema version `1` |
| Intake contract | `raw-scatac-intake.v1` |
| Future FASTQ route | `fastq-to-cell-by-ccre.v1` |
| Future BAM route | `bam-to-cell-by-ccre.v1` |

These route identities describe future execution contracts; they do not imply
implemented processing. Readiness is a scientific intake state, separate from
run/step success. A future inspection may succeed while truthfully reporting
an input as unready or invalid. Group and aggregate readiness are derived
deterministically from validated findings, with precedence:

`INVALID` > `UNSUPPORTED` > `NEEDS_USER_INPUT` > `READY_WITH_REPAIRS` > `READY`.

The manifest preserves findings even when a higher-priority state wins.
Structured issue severity and readiness effect are separate fields; severity
alone does not determine readiness. Required information, preparation
requirements, and normal downstream prerequisites are separate concepts.
Unresolved required information blocks readiness. Supported preparation
requirements can yield `READY_WITH_REPAIRS`; ordinary downstream work such as
alignment and target-reference provision does not itself downgrade `READY`.

Source genome assembly and target genome assembly are separate scientific
concepts. Supported model-oriented targets are `hg38` for human and `mm10` for
mouse. A source BAM assembly must not be inferred from species, filename, or
chromosome naming alone. FASTQ has no source genomic coordinate assembly by
default and may target hg38/mm10 directly through future alignment. An assembly
mismatch establishes a harmonization requirement, not proof that a supported
harmonization route exists. Mismatch alone does not imply `READY_WITH_REPAIRS`:
unresolved route admissibility requires user input, supported admissibility
requires explicit route-matched evidence, and an evidenced unsupported route
yields `UNSUPPORTED`, subject to readiness precedence. M10.1 validates supplied
contract evidence; it does not establish route feasibility by processing files.

Barcode provenance, barcode identity scope, and optional namespace metadata are
separate. Deterministic group-local barcode identity scope is allowed; absent
namespace metadata alone does not prevent readiness. Identical barcode strings
across independent groups do not imply identical cells. Unresolved identity
scope remains a required-information condition; namespace labels do not silently
merge groups or resolve ambiguous scope.

Inspection coverage explicitly records bounded/sample scope, limits, and stop
conditions. Sample observations and observed-region digests must not be described
as whole-file verification. The infrastructure validates supplied records and
coverage claims without parsing or verifying the raw inputs themselves.

Canonical manifest serialization is deterministic and strict: stable ordering
and content-derived identities, closed field/type/enum contracts, and rejection
of duplicate JSON keys, nonfinite values, inconsistent derived values, and invalid
references. Manifest loading validates the contract. Publication uses canonical
bytes, atomic replacement, and a manifest SHA-256; this small-artifact digest
does not establish raw-input content integrity.

M10.1 added no FASTQ/BAM parser, `pysam`/`samtools` dependency, public
raw-inspection tool, planner/compiler/registry/verifier/report integration,
alignment, liftOver, or cell-by-cCRE execution. Existing scientific execution
and planning contracts remain unchanged.

Accepted validation, including the contract-hardening pass:

| Acceptance suite | Result |
| --- | --- |
| M10.1 focused | 194 passed |
| Relevant regression | 75 passed |
| Full lightweight regression | 1702 passed, 54 skipped, 7 warnings |

### Milestone 10.2 — FASTQ data-layer vertical slice

M10.2 is complete and accepted, including the suffix/content and missing-role
hardening pass. `src/agent/tools/data/_raw_fastq.py` provides deterministic,
read-only FASTQ discovery and inspection through `inspect_fastq_inputs()`.
It produces the accepted M10.1 `RawIntakeManifest`: artifact
`agent.raw-scatac-intake`, schema version `1`, intake contract
`raw-scatac-intake.v1`, and route `fastq-to-cell-by-ccre.v1` remain unchanged.
FASTQ-specific facts use bounded canonical evidence under `raw-fastq.v1`.
This is a data-layer API, not a ToolRegistry/public Agent capability.

Agent inspection accepts `.fastq`, `.fastq.gz`, `.fq`, and `.fq.gz`; this does
not establish the naming conventions accepted by external Cell Ranger tools.
Selection supports explicit local regular files and non-recursive local
directories, with deterministic lexical discovery and bounded candidate/entry
counts. Remote URLs and recursive discovery are unsupported. Explicit file
symlinks resolve to canonical paths and duplicate canonical aliases collapse;
conflicting hard-link aliases and directory-discovered candidate symlinks are
rejected. Source directories are not mutated.

| Repository-owned inspection bound | Limit |
| --- | --- |
| Candidate files / selections | 128 each |
| Directory entries per directory | 16,384 |
| Records inspected per file | 256 |
| Decoded bytes per file | 2 MiB |
| Line size, including terminator | 16 KiB |
| Encoded gzip input | 4 MiB |

The reviewed filename grammar is
`[sample]_S[number][_L[lane]]_[role]_[chunk]` followed by an accepted suffix.
Roles are `R1`, `R2`, `R3`, `I1`, and `I2`; lanes and chunks have three digits.
Parsing proceeds from the right so sample tokens may contain underscores.
Missing lane remains missing. Grouping uses the canonical leaf location, sample
token, sample number, lane state/value, and chunk. Matching sample names across
independent roots are never merged. Filename grouping is syntactic evidence
only: it establishes neither biological sample/replicate/condition nor shared
barcode identity. Unrecognized names remain independent unresolved groups.

The first supported assay family is **10x / Cell Ranger ATAC-compatible**,
requiring explicit `FastqAssay.TENX_ATAC` authority before filename roles receive
a supported ATAC interpretation. Neither filenames nor read lengths establish
assay identity; generic R1/R2 paired-end input is not automatically 10x ATAC.

| Layout | Identity | Genomic read 1 | Raw cell-barcode / i5 read | Genomic read 2 | Optional sample index |
| --- | --- | --- | --- | --- | --- |
| A | `tenx-atac-r1-r2-r3.v1` | R1 | R2 | R3 | I1 |
| B | `tenx-atac-r1-i2-r2.v1` | R1 | I2 | R2 | I1 |

Supported assay/layout evidence permits `FASTQ_READ` barcode provenance with
locator `R2` for A or `I2` for B. I1 is never the cell-barcode source. No
corrected/canonical barcode identity is claimed, and barcode values are never
persisted. Group-local barcode identity scope remains M10.1-derived; identical
strings in independent groups do not establish the same cell.

A declared or uniquely determined candidate layout missing a required role
produces `FASTQ_REQUIRED_ROLE_MISSING`, with exact absent roles in canonical
evidence. Missing required input yields `NEEDS_USER_INPUT`, subject to other
findings and M10.1 precedence: an additional required artifact must be supplied,
without automatically declaring the experiment `INVALID`. Missing optional I1
is acceptable. Duplicate-role, incompatible-role, unresolved-layout, and
unresolved-assay findings remain distinct. No file is invented or repaired.

Caller guidance reserves optional `declared_layout` for incomplete, explicitly
declared 10x assay inputs that cannot otherwise distinguish A from B. It accepts
only the supported A/B identities, applies to all selected groups, and never
replaces deterministic filename/content inspection or assay authority. Without
such a declaration, R1/R2 preserves both conditional alternatives (A missing R3,
B missing I2), rather than guessing one. Evidence distinguishes declared-layout,
unique-candidate, and alternative-candidate bases. An unresolved assay or
unresolved grouping does not support a precise missing-role assertion.

The bounded parser checks four LF/CRLF-terminated FASTQ lines, an `@` header,
a `+` separator, equal sequence/quality lengths, and bounded printable ASCII
sequence/quality content without restricting sequence to A/C/G/T. Optional
identifier text on the separator must exactly repeat the header text after `@`.
Line/component and decoded-byte limits bound record allocation; limit stops
are distinguished from malformed/truncated records. This is a narrow parser
contract, not unrestricted support for every FASTQ representation.

Actual compression is determined from content/magic bytes, independently of the
filename suffix hint. Valid plain and gzip FASTQ are supported. A misleading
suffix produces advisory `FASTQ_COMPRESSION_SUFFIX_MISMATCH`; mismatch alone
does not downgrade readiness. Inputs are never renamed or recompressed. Gzip
corruption encountered during inspection is invalid, including trailer/CRC
errors encountered while reaching EOF. Malformed decoded FASTQ remains invalid.
Uninspected gzip tails are not certified; naming preparation for a future
external executable is not missing scientific information in M10.2.

Multi-read synchronization compares sampled first whitespace-delimited header
tokens after removing `@` and normalizing only terminal `/1` or `/2`. There is
no fuzzy matching, reordering, or resynchronization. Observed read-ID mismatch
is blocking/invalid. When all compared files reach EOF, unequal complete record
counts are blocking; an observed early EOF against another file's longer checked
prefix is also blocking. Budget termination is not reported as premature EOF,
and sampled agreement is not a claim of complete read-count equality.

Coverage explicitly distinguishes bounded samples from complete coverage;
complete requires sequential EOF observation under the parser contract. It
records inspected records/decoded bytes, declared limits, EOF, and stop reason.
The observed-region SHA-256 covers only inspected decoded content, including
observed partial components; it is not a whole-file hash. No read IDs, sequences,
qualities, or barcode vectors are persisted.

`observe_fastq_sources()` is the reusable low-level source-reinspection primitive.
It reopens recorded FASTQ inputs and reproduces bounded structural observations,
observed-region digests, and synchronization summaries without trusting
serialized readiness. Recorded size/mtime and before/after snapshot checks guard
against materially changing sources; inspection preserves source bytes/mtimes.
The primitive supports later independent orchestration verification and is not
yet connected to `verify_step()`. Operational selection/read/snapshot failures
use sanitized `RAW_FASTQ_*` errors; safely observed scientific defects remain
manifest findings, preserving other groups where inspection remains trustworthy.

Species is never inferred from FASTQ filenames or contents. Explicit human maps
to target `hg38`; explicit mouse maps to `mm10`, through M10.1. FASTQ source
coordinate assembly is not applicable. Alignment to the target reference is a
normal downstream prerequisite and does not itself downgrade readiness. No
liftOver or reference conversion occurs.

Accepted representative behavior below assumes other common M10.1 gates are
satisfied; human is declared where hg38 is shown, and species is deliberately
absent in the final row:

| FASTQ input | Readiness | Target | Barcode locator |
| --- | --- | --- | --- |
| Declared human Layout A | `READY` | hg38 | R2 |
| Declared mouse Layout B | `READY` | mm10 | I2 |
| Matching names without assay declaration | `NEEDS_USER_INPUT` | hg38 | Unknown |
| Generic R1/R2 | `NEEDS_USER_INPUT` | hg38 | Unknown |
| Declared supported layout missing required role | `NEEDS_USER_INPUT` | hg38 | Unknown |
| Malformed FASTQ | `INVALID` | hg38 | Unknown |
| Sampled read-ID mismatch | `INVALID` | hg38 | Unknown |
| Supported Layout A with unknown species | `NEEDS_USER_INPUT` | Unresolved | R2 |

M10.2 introduced no BAM support, `pysam`, `samtools`, alignment, realignment,
liftOver, barcode correction, fragments, cell calling, cell-by-cCRE construction,
ToolRegistry capability, `ArtifactSemanticKind` change, planner/compiler change,
orchestration verifier dispatch, or reporting integration.

Accepted validation, including contract hardening:

| Acceptance suite | Result |
| --- | --- |
| M10.2 focused | 177 passed |
| M10.1 regression | 194 passed |
| Relevant data/artifact regression | 75 passed |
| Full lightweight regression | 1879 passed, 54 skipped, 7 warnings |

### Milestone 10.3 — BAM data-layer vertical slice

M10.3 is complete and accepted. `src/agent/tools/data/_raw_bam.py` provides
deterministic, strictly read-only BAM intake through `inspect_bam_inputs()`.
It populates the existing M10.1 `RawIntakeManifest`: artifact
`agent.raw-scatac-intake`, schema version `1`, intake contract
`raw-scatac-intake.v1`, and route `bam-to-cell-by-ccre.v1` remain unchanged.
BAM-specific compact observations use `raw-bam.v1` evidence. M10.2 FASTQ
implementation and behavior remain unchanged. This is a data-layer API, not a
public ToolRegistry/Planner capability; readiness remains separate from execution
success and follows M10.1 deterministic derivation and precedence.

The tested optional backend is `pysam==0.24.1` (accepted environment: htslib
`1.24`), declared in `requirements-bam.txt`. It is lazy-loaded only on BAM
execution paths. Agent import, M10.1, M10.2 FASTQ inspection, and existing H5AD
workflows remain functional without pysam. A BAM attempt without it raises
sanitized `RAW_BAM_DEPENDENCY_UNAVAILABLE`; untested pysam versions are rejected
with `RAW_BAM_BACKEND_VERSION_UNSUPPORTED`. No external `samtools` executable
is required. Agent owns scientific interpretation; pysam/htslib owns BAM format
access, with no custom BAM binary decoder.

Selection accepts explicit local `.bam` files and non-recursive local
directories, ordered lexically and deterministically. One BAM forms one
independent raw-input group; filename similarity never merges BAMs. `.bai` and
`.csi` are sidecar observations only. Canonical duplicate aliases collapse;
conflicting hardlink aliases and directory-discovered candidate symlinks are
rejected. SAM, CRAM, remote URLs, and recursive discovery remain unsupported.
Production inspection never modifies source BAMs or source directories.

| Repository-owned inspection bound | Limit |
| --- | --- |
| Selections / candidate files | 128 each |
| Directory entries per directory | 16,384 |
| Sequential alignment records per BAM | 256 |
| Header projection | 1 MiB |
| Reference dictionary records | 4,096 |
| Read groups | 256 |
| Programs | 128 |
| Distinct normalized metadata values | 4 |

Lazy `pysam.AlignmentFile` opens BAM in binary read mode; deterministic prefix
inspection uses `fetch(until_eof=True)` without an index. A separate explicit
index probe avoids letting corrupt sidecars determine sequential readability.
Reaching EOF before the record bound may establish complete sequential coverage;
stopping at the bound remains sample-scoped, without certifying uninspected
tails. `decoded_bytes_inspected` and `decoded_byte_limit` are `None`. Header
limits apply after backend decoding. Neither the record bound nor the header
projection limits claim hard decoded-byte/pre-allocation memory bounds within
pysam/htslib. Normalized observation/source digests are compact summary identities,
not whole-file BAM hashes or hashes of read/barcode vectors.

Header evidence is restricted to reviewed compact facts. `@HD` presence,
version, and `SO` declaration are observed; absence of optional `@HD` alone is
not invalid. Ordered `@SQ` names/lengths and present `M5` reference-sequence
digests are compactly fingerprinted. Reviewed `AS` and `SP` fields contribute
metadata assertions. `@RG` supplies compact read-group/library/sample counts;
exactly one authoritative, consistent library identity may populate M10.1
library provenance. Multiple libraries are not collapsed; RG/SM never establish
experimental condition or replicate. `@PG` supplies narrow identity/version
provenance. Arbitrary reference `UR` fields and complete `@PG CL` command lines
are not persisted.

Species requires explicit declaration or complete reviewed authoritative SQ
evidence. Reviewed header aliases include Homo sapiens/human → `human` and
Mus musculus/mouse → `mouse`; no species inference uses BAM filenames, read
sequences, or chromosome naming. Assembly normalization supports GRCh38/hg38 →
`hg38` and GRCm38/mm10 → `mm10`; other accepted normalized source identifiers
remain distinct. Incomplete reviewed SQ assertions remain observations rather
than complete header authority, and contradictory assertions remain conflicts.
Neither species nor chromosome names establish exact source assembly.

Human target assembly remains `hg38`; mouse remains `mm10`. Source and target
are independent, with MATCH/MISMATCH/UNKNOWN/CONFLICT semantics owned by M10.1.
Human source hg38 and mouse source mm10 match their respective targets. Human
source hg19 mismatches target hg38 and requires harmonization, but route
admissibility remains unresolved; this is not automatically `READY_WITH_REPAIRS`.
Human with unknown source retains target hg38 and requires user input.
M10.3 detects mismatch without establishing that reads/cell identity can safely
be recovered, that a supported realignment route exists, or that preparation
is admissible. It implements no BAM coordinate liftOver. Transformation-route
decisions belong to future M11 work, not this accepted inspection slice.

Sort declaration and observed ordering remain separate. `@HD SO` is declared
metadata; sampled primary mapped records permit local coordinate comparisons.
Sampled consistency does not prove whole-file sorting. An observed inversion
preserves a local finding, including contradiction of a coordinate declaration.
Sort findings are currently advisory and do not automatically create preparation
requirements; sorting is deferred to later preprocessing routes.

Index observations distinguish absent, openable, unusable, and competing local
BAI/CSI candidates. No competing candidate is silently preferred. Sequential
inspection needs no index and creates none. An openable index does not prove
universal index correctness. Missing index alone is neither `INVALID` nor
automatically `READY_WITH_REPAIRS`; unusable/competing index findings are advisory.

Sampled record counts include primary, secondary, supplementary, mapped/unmapped,
paired/unpaired, read1/read2, proper-pair, duplicate, and QC-fail flags. No
biological QC thresholds are applied. Individual read names, sequences,
qualities, CIGAR vectors, and coordinates are not persisted.

| Standard tag | Accepted interpretation |
| --- | --- |
| `CB` | Usable sampled cell identifier → `BAM_CELL_IDENTIFIER`; generic CB is not universally corrected |
| `CR` | Raw cellular barcode sequence → `BAM_RAW_SEQUENCE` when usable CB is unavailable; future barcode processing remains a normal downstream prerequisite |
| `CY` | Raw cellular barcode-quality evidence associated with CR |
| `BC` / `QT` | Sample/library barcode and quality evidence, never cell identity |
| `RG` | Read-group provenance, never cell identity |

Usable CB takes precedence over CR while preserving CR/CY observations. Reviewed
tag types, nonempty values, and duplicate tags are checked without coercion.
Prevalence records explicit all-sampled, primary, and mapped-primary denominators
and present/usable/invalid counts. Zero observations mean not observed in the
inspected sample, not absent from the complete BAM. Barcode values are never
persisted, suffixes are not stripped, and no minimum-prevalence threshold is
invented. Generic tags retain general SAM semantics. Exact reviewed Cell Ranger
ATAC program provenance may establish ATAC assay and its specific corrected/
known-good CB, raw CR, and quality CY interpretation; these semantics are never
universalized to arbitrary CB. Generic aligner provenance alone does not establish
ATAC assay; otherwise an explicit supported assay declaration is required.

Group-local identity is normally sufficient for sampled CB. Independent BAM
groups never merge identical barcode values automatically. CR with demonstrably
multiple independent libraries leaves identity scope unresolved. An explicit
decorative namespace is not required when group-local scope is already safe.

`observe_bam_source()` independently reconstructs source observations, compact
header/reference identity, bounded record counts, tag prevalence, sort/index
observations, and the normalized observation digest without trusting serialized
readiness. Recorded size/mtime and before/after source/sidecar snapshot checks
reject material changes; source bytes/mtimes are preserved. This primitive is
intended for later independent orchestration verification and is not wired into
`verify_step()`. Operational dependency/access/stability failures raise sanitized
errors. Observed malformed/truncated content produces scientific `INVALID`
findings, preserving other valid groups where trustworthy inspection is possible.

Accepted representative outcomes assume other common gates are satisfied:

| BAM input | Readiness / scientific consequence |
| --- | --- |
| Human hg38 + CB | `READY`; target hg38, MATCH |
| Mouse mm10 + CB | `READY`; target mm10, MATCH |
| Human hg19 | `NEEDS_USER_INPUT`; target hg38, MISMATCH, harmonization admissibility unresolved |
| Human + unknown source | `NEEDS_USER_INPUT`; target hg38, source remains unknown |
| CB absent, CR present, safe group-local scope | `READY` with future barcode-processing prerequisite |
| BC/QT only | `NEEDS_USER_INPUT`; cell barcode source unknown |
| Missing index | Sequential inspection succeeds and may remain `READY` |
| Observed malformed/truncated BAM | `INVALID` |
| Multi-library CR ambiguity | `NEEDS_USER_INPUT`; identity scope unresolved |

M10.3 introduced no production BAM sorting, index creation, alignment,
realignment, FASTQ recovery, liftOver, barcode correction, whitelist processing,
fragment construction, cell calling, cell-by-cCRE, ToolRegistry/public
raw-inspection capability, `ArtifactSemanticKind` change, planner/compiler change,
orchestration verifier dispatch, or reporting integration. At the M10.3 checkpoint,
M10.4 and M11 remained future work.

Accepted validation used synthetic temporary BAM/index fixtures with pysam enabled:

| Acceptance suite | Result |
| --- | --- |
| M10.3 BAM focused | 121 passed |
| M10.1 manifest | 194 passed |
| M10.2 FASTQ | 177 passed |
| Relevant regression | 75 passed |
| Full lightweight regression | 2000 passed, 54 skipped, 7 warnings |

### Milestone 10.4 — Raw intake registry, Planner, and verification integration

M10.4 is complete and accepted. `inspect_raw_scATAC` is the only registered
high-level raw-sequencing intake capability, with planning role `INSPECTION` and
recovery identity `inspect-raw-scatac-v1`. Private FASTQ/BAM inspectors remain
internal helpers, not Planner tools. The public API in
`src/agent/tools/data/raw_scatac.py` is:

```python
inspect_raw_scATAC(
    raw_input_paths: Sequence[str | Path] | str | Path,
    output_dir: str | Path,
    *,
    species: str | None = None,
    raw_assay: str | None = None,
    source_genome_assembly: str | None = None,
    fastq_layout: str | None = None,
) -> RawScATACInspection
```

Inputs may be one local path or a JSON-frozen tuple/list of local paths,
including non-recursive directories. Bounded deterministic discovery dispatches
FASTQ-only selections to M10.2 and BAM-only selections to M10.3. Mixed selections
fail closed with `RAW_INPUT_KIND_MIXED`; empty/invalid selections produce stable
sanitized input errors. Dispatch does not run both parsers to guess a format.
FASTQ execution and verification do not import pysam; BAM retains its accepted
optional lazy dependency. There are no public arguments for input kind, barcode
read/tag, inspection budgets, target assembly, output filename, or overwrite.
Those details remain deterministic Agent concerns under the accepted contracts.

Declarations come from structured request inputs, not model-invented facts:

| Declaration | Accepted meaning |
| --- | --- |
| `species` | Explicit normalized lowercase declaration; other declared species can truthfully yield `UNSUPPORTED` |
| `raw_assay` | `TENX_ATAC` for FASTQ/BAM; `SCATAC` for BAM only |
| `source_genome_assembly` | Explicit source coordinates for BAM only; never inferred from species |
| `fastq_layout` | `tenx-atac-r1-r2-r3.v1` or `tenx-atac-r1-i2-r2.v1`, requiring FASTQ/TENX_ATAC context |

Incompatible declarations fail closed rather than supplying guessed assay or
coordinate authority. Human target remains hg38; mouse remains mm10. M10.1 owns
source/target compatibility, preparation admissibility, and readiness derivation.

Publication uses canonical M10.1 bytes and the deterministic filename
`raw-scatac-intake-<full-sha256>.json` inside managed `output_dir`, through the
accepted atomic publication helper. Strict reload confirms actual bytes/digest.
An identical existing artifact is reused idempotently; conflicting content is
preserved and rejected. No overwrite parameter, random ID, or publication-time
timestamp enters artifact identity. Recorded source mtimes remain source evidence.

The exact lightweight `RawScATACInspection` result fields are:

```text
status, manifest_path, manifest_sha256, artifact_type,
artifact_schema_version, intake_contract_version, input_kind, readiness,
n_files, n_groups, n_issues, n_required_information, n_repairs, n_prerequisites
```

`status="success"` means inspection execution succeeded, not that data are
scientifically `READY`. `input_kind` is `fastq` or `bam`. The authoritative
artifact remains `agent.raw-scatac-intake`, schema `1`, contract
`raw-scatac-intake.v1`. The full manifest is never embedded in the tool result.
The strict registry result contract checks exact fields, identities, digest shape,
readiness vocabulary, and bounded/coherent counts without reinspecting sources.

Planning adds `RAW_SCATAC_SEQUENCING = "raw_scatac_sequencing"` and
`RAW_SCATAC_INTAKE_MANIFEST = "raw_scatac_intake_manifest"`. Existing
`RAW_SCATAC` and `raw_scatac_dataset.v1` retain processed-H5AD/matrix semantics.
Authoritative consumer ports authorize these request sources:

| Consumer port | Request source |
| --- | --- |
| `raw_input` | `raw_input_paths` |
| `output_dir` | `output_dir` |
| `species` | `species` |
| `raw_assay` | `raw_assay` |
| `source_genome_assembly` | `source_genome_assembly` |
| `fastq_layout` | `fastq_layout` |

Existing H5AD `input_path` is not authorized for `raw_input`. Application-managed
`output_dir` is compiler-bound, never model-authored. The sole producer port is
`intake_manifest`, semantic type `raw_scatac_intake_manifest.v1`, with members
`manifest_path` and `manifest_sha256` only. Readiness, target assembly, barcode
source, input kind, and inferred species are not planner-composable output
members. No authorized semantic channel connects this manifest directly to
EpiZoo/H5AD consumers; future preprocessing must construct the required processed
cell-by-cCRE/H5AD input first.

Scripted wire-v4 acceptance used the request: "Inspect these raw mouse scATAC
FASTQs and tell me whether they are ready for preprocessing." The model selected
`inspect_raw_scATAC` and the semantic source `raw_input_paths`; the compiler bound
actual raw paths, managed output directory, explicit species/assay, and applicable
declarations from the request. Structured path values are absent from model
context. The natural-language prompt itself retains the existing pass-through
contract. The model does not author FASTQ/BAM kind, read roles, BAM tags, target
assembly, inspection budgets, or output paths. Unique authorized bindings need
no redundant model selections. Explicit wire-v3 compatibility remains accepted;
the historical M9 benchmark retains its processed-H5AD corpus.

`verify_step()` explicitly dispatches to
`src/agent/orchestration/raw_scatac_verifier.py`. Verification:

1. Validates resolved arguments and the normal registry result contract.
2. Requires the manifest to belong to resolved managed `output_dir`, with the
   expected content-addressed filename.
3. Strictly loads actual bytes and checks SHA-256, artifact/schema/contract identity.
4. Reconstructs current raw-source inspection through the accepted private
   M10.2/M10.3 implementations with the same resolved explicit declarations.
5. Compares canonical reconstructed/stored manifests and the actual byte digest.
6. Derives the lightweight summary from the verified manifest and requires exact
   agreement with the returned result.

This independently checks source inventory and observations; it never calls
public `inspect_raw_scATAC` or publishes another manifest. Tests detect source,
BAM-index observation, declaration, digest, content, and summary drift, including
self-consistent forged manifests and noncanonical bytes. Source checking retains
M10.2/M10.3 bounded coverage: it does not certify uninspected tails or turn BAM
summary digests into whole-file hashes.

A truthful `NEEDS_USER_INPUT`, `UNSUPPORTED`, or `INVALID` manifest can execute
successfully and pass verification. Verification establishes the authenticity
and source binding of the finding, not scientific readiness. In particular,
truthful `INVALID` means successful inspection with verified invalid-input
findings, not an automatic runtime exception.

PLAN_ONLY performs zero public raw-tool execution, zero private FASTQ/BAM
inspection, zero publication, and zero source reinspection. It requires no
pysam import and returns `PLANNED` with no step results. Subprocess acceptance
blocks pysam imports for Runtime/Application PLAN_ONLY and for FASTQ execution
plus verification. Application-managed output binding uses the existing workspace.

Verified FASTQ and BAM Runtime EXECUTE and durable completed-step persistence
are accepted. Terminal resume preserves immutable-return behavior, with no
planner rerun or fresh source check. Nonterminal resume revalidates completed
raw work and reuses its production result; pending work may execute normally.
Source drift at that fresh verification boundary blocks further production.
The two-step acceptance recorded one production call before interruption and
one additional call for the pending step after successful resume; the completed
step was not rerun. Content-addressed reuse produces no duplicate artifact.
Existing cooperative cancellation behavior remains unchanged.

Errors distinguish user/input failures, environment/dependency failures,
verification/source-integrity failures, and output/resource failures from
scientific readiness. Stable raw-intake/FASTQ/BAM codes and sanitized policy
messages are preserved. No automatic retries were introduced for deterministic
input/scientific failures. Adding the registry capability does not change
unrelated historical plan recovery identities.

M10.4 required no production changes to M10.1 manifest implementation, M10.2
FASTQ, M10.3 BAM, Runtime, Executor, RunStore, Application service, semantic
compiler, semantic prompt, semantic wire-v4, LLMPlanner, or evidence/report code.
Existing export/inventory test snapshots were updated for the added capability.
The accepted path is AgentRequest → planning → Runtime execution → independent
step verification → persistence. Full `ResearchAgentApplication.run(EXECUTE)`
raw-intake evidence/report composition remains M10.5 work.

M10.4 introduced no alignment, sorting/index creation, realignment, FASTQ
recovery, liftOver, barcode correction, whitelist processing, fragment generation,
cell calling, TSS/FRiP/QC metrics, cell-by-cCRE, EpiZoo inference, evidence/report
integration, CLI convenience flags, or M11 preprocessing. M10.5 and M11 remain
unimplemented.

Accepted validation:

| Acceptance suite | Result |
| --- | --- |
| Public tool | 64 passed |
| Verifier | 27 passed |
| Semantic/v3/v4 | 12 passed |
| Lifecycle/no-pysam | 12 passed |
| M10.1 | 194 passed |
| M10.2 | 177 passed |
| M10.3 | 121 passed |
| Relevant orchestration regression | 1124 passed, 3 warnings |
| Full lightweight regression | 2115 passed, 54 skipped, 7 warnings |
