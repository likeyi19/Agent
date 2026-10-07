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
responses and unsanitized provider exception prose are not persisted by default;
operational diagnostics retain only bounded, redacted fields.

Candidate descriptors record exact provider/model/profile/transport identity.
An exact account model lookup must succeed before generation. Groq, Gemini and
DashScope reuse production adapters. Mistral uses a generic benchmark-local
strict-schema Chat Completions adapter with the unchanged Agent prompt/schema.
DashScope reuses the explicitly configured historical `DASHSCOPE_BASE_URL`.
Credential values are passed directly to SDK construction and never recorded.

OpenRouter admission reuses the generic strict-schema Chat Completions adapter
with exact `:free` catalog identities, required-parameter routing, zero-price
limits, no provider fallback, and healing/compression/web plugins disabled.
Its admission wrapper permits one completion. OpenRouter breadth calls use a
fresh admission wrapper per production call so bounded Layer-2 recovery remains
visible. `run_attempt` accepts an explicit model.

Q2 free-cohort qualification is closed for exactly:

- Groq `openai/gpt-oss-120b`;
- Groq `qwen/qwen3.8-27b`;
- OpenRouter `nvidia/nemotron-3-super-120b-a12b:free`.

GPT-OSS showed the strongest coverage, but repeated supported-inspection failures
and unstable scientific Answer contracts prevented acceptance as development
primary. Qwen remains a diagnostic comparator; Nemotron is removed from broad
primary-model qualification. This cohort has no accepted development-primary
LLM. These are development-screening findings, not production reliability or
biological qualification.

The next comparison should use one substantially stronger frontier model against
the frozen architecture; its exact provider/model has not been selected or added.
I12 null-subject wording and I13 Guidance context/capacity work remain deferred.
Agent prompts, schemas, scientific contracts and planning/admission semantics
remain unchanged. All live evidence under `evals/` stays local and uncommitted.

Run focused offline acceptance in the `agent` environment:

```bash
PYTHONPATH=src:. python -B -m pytest -q tests/benchmarks/test_interactive_*.py
```

The Q2 entry points validate frozen inputs and separate discovery, completion,
recovery and failure accounting. Breadth covers the fourteen canonical scenarios;
resume preserves the fifteen completed Q2.1 cells and completes its twenty-seven
missing Groq cells. Stability adds two GPT-OSS attempts for each of I01, I10, I08
and I14. Resume/stability require compatible preserved local ledgers, and stability
also requires the separate historical I08 operator-review disposition.
The experiment identity remains the original Q1 commit; runners report the current
infrastructure commit separately and reject changes to frozen Agent, Planner or
original qualification Python contracts. Offline checks allow pending infrastructure
edits; live runs require a clean tracked tree and index. Stability accepts an explicit
`--human-review` path for the separate historical disposition.

Use dry validation before a separately authorized live run:

```bash
PYTHONPATH=src:. python -B -m benchmarks.interactive.run_breadth --dry-run
PYTHONPATH=src:. python -B -m benchmarks.interactive.run_breadth --live \
  --output evals/interactive_model_qualification_q2_1_NEW_RUN
PYTHONPATH=src:. python -B -m benchmarks.interactive.run_breadth --live \
  --resume evals/interactive_model_qualification_q2_1_2026-10-07/results.json \
  --output evals/interactive_model_qualification_q2_1b_NEW_RUN
PYTHONPATH=src:. python -B -m benchmarks.interactive.run_stability --live \
  --original evals/interactive_model_qualification_q2_1b_2026-10-07/merged_results.json \
  --output evals/interactive_model_qualification_q2_2_NEW_RUN
```

Admitted Answer/Guidance prose stays `HUMAN_REVIEW_PENDING` until externally
reviewed. Historical classifications remain intact; later human dispositions are
separate records. Benchmark retries and automatic semantic prose grading are absent.

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
