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
