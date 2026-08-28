# Agent

An autonomous AI agent for single-cell epigenomic analysis powered by epigenomic foundation models.

Milestone 2 provides safe scATAC inspection, artifact-based EpiZoo cell embedding, and process-local model reuse.

## Current status

Milestones 1–4 are complete.

- Milestone 1: validated EpiZoo scientific backend
- Milestone 2: reusable scientific tool layer
- Milestone 3: safe Agent orchestration core
- Milestone 4: provider-neutral natural-language planning

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
