# Agent

An autonomous AI agent for single-cell epigenomic analysis powered by epigenomic foundation models.

Milestone 2 provides safe scATAC inspection, artifact-based EpiZoo cell embedding, and process-local model reuse.

## Current status

Milestones 1–5 and Milestones 6.1–6.3 are complete.

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

The current Agent can construct and execute validated scientific workflows
through an explicit tool registry, with structured planning, verification,
error handling, dependency resolution, and execution tracing. `LLMPlanner`
uses an injected `PlanningModel` and a strict versioned planning schema to
produce the existing `AgentPlan`; Milestone 3 preflight, execution, and
verification remain unchanged. Optional OpenAI, Gemini, and Groq adapters are
available, while the default runtime remains deterministic and offline.

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
