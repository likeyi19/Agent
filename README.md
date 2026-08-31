# Agent

An autonomous AI agent for single-cell epigenomic analysis powered by epigenomic foundation models.

Milestone 2 provides safe scATAC inspection, artifact-based EpiZoo cell embedding, and process-local model reuse.

## Current status

Milestones 1–4 and Milestones 5.1–5.2 are complete.

- Milestone 1: validated EpiZoo scientific backend
- Milestone 2: reusable scientific tool layer
- Milestone 3: safe Agent orchestration core
- Milestone 4: provider-neutral natural-language planning
- Milestone 5.1: durable run state and planner-free resume
- Milestone 5.2: cooperative cancellation and durable run lifecycle

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
