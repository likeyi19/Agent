# Interactive model qualification

This benchmark-local instrument reuses Agent's interpreter, scoped-v4 Planner,
compiler, preflight, scientific answer and Guidance contracts. Production code,
prompts, scientific semantics and Web provider choices are unchanged.

`scenarios.py` defines fourteen immutable scenarios. `fixtures.py` builds fresh
disposable scripted accepted metadata through real Session/evidence readers.
These fixtures test dialogue contracts; they establish neither biological
acceptance nor reusable scientific authority. Real accepted-artifact closure
qualification remains a separate, explicitly provisioned exercise.

`harness.run_attempt(scenario, candidate, model=...)` requires an explicit model.
Interpreter, Stage A, fixed-scope Stage B, scientific Answer and Guidance are
independent. Layer 2 uses actual Session interpretation/admission, intercepts
only the execution handoff, and delegates unchanged planning/recovery to the
application runtime with `PLAN_ONLY`. It preserves the interpreter target check.
Actual-plan retained-output selection and scientific Revision activation are
outside that preflight stop.

Every attempt starts with fresh observation, Session and planning state. Attempt
IDs are diagnostic metadata and do not enter prompts. Guards prohibit scientific
Registry calls, execution entries and step results; read-only surfaces also
prohibit planning, owner reconstruction and evidence/report regeneration.

Results separate operational, contract, semantic and deterministic-admission
failures. Direct Stage B makes one call; Layer 2 retains production retry/repair
semantics, records initial and final outcomes, and never repairs from the oracle.
Typed unsupported is contract-valid and terminal. Answer/Guidance retain bounded
claim/reference validation but require human semantic review. Full prompts/raw
responses and provider exception prose are never persisted by default.

Candidate descriptors record exact provider/model/profile/transport identity.
An exact account model lookup must succeed before generation. Groq, Gemini and
DashScope reuse production adapters. Mistral uses a generic benchmark-local
strict-schema Chat Completions adapter with the unchanged Agent prompt/schema.
DashScope reuses the explicitly configured historical `DASHSCOPE_BASE_URL`.
Credential values are passed directly to SDK construction and never recorded.

OpenRouter admission reuses the generic strict-schema Chat Completions adapter
with exact `:free` catalog identities, required-parameter routing, zero-price
limits, no provider fallback, and healing/compression/web plugins disabled.
Its admission wrapper permits one completion. `run_attempt` accepts an explicit
model; future Q2 attempt/runtime construction remains separate work.

Q0 through Q1e are closed. The first Q2 cohort is frozen to exactly:

- Groq `openai/gpt-oss-120b`;
- Groq `qwen/qwen3.8-27b`;
- OpenRouter `nvidia/nemotron-3-super-120b-a12b:free`.

Stop candidate expansion. Agent prompts, schemas, scientific contracts and
planning/admission semantics remain frozen. Q2 has not been run. Deferred and
unavailable candidates retain their local historical evidence without retesting;
all live outputs under `evals/` remain uncommitted.

Run focused offline acceptance in the `agent` environment:

```bash
PYTHONPATH=src:. python -B -m pytest -q tests/benchmarks/test_interactive_*.py
```

The historical admission smoke validates exact account model availability, then
makes at most one interpreter completion per available candidate. It never
substitutes models, repeats scenarios, tunes prompts or ranks candidates:

```bash
PYTHONPATH=src:. python -B -m benchmarks.interactive.run_smoke --live \
  --output evals/interactive_model_qualification_q1_NEW_RUN
```

Model-discovery requests and completion calls are counted separately. Results
are transport/contract readiness evidence, never a claim that a model is
qualified. Existing output directories fail closed to prevent overwrites.
The historical `run_smoke` default includes deferred candidates; neither it nor
`run_openrouter_smoke` selects the frozen Q2 cohort or runs formal qualification.
Running admission again requires a separate explicit request.
