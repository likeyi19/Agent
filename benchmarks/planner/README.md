# LLM Planner Robustness Benchmark

This benchmark began as the Milestone 9.1 baseline for production
`LLMPlanner`. Its semantic workflow oracle exists only in
`benchmarks.planner.benchmark`; production orchestration never imports it.

The corpus contains synthetic prompts, paths, and metadata only. Expectations
describe outcomes, tool roles, binding origins, dependency edges, and
`StepOutputRef` relationships. Provider descriptions, rejection wording, and
provider-generated step IDs are ignored.

## Offline replay

The normal deterministic path uses scripted responses and requires no network,
credentials, provider SDK call, GPU, checkpoint, or dataset:

```bash
PYTHONPATH=src:. python benchmarks/planner/run_benchmark.py
```

The replay fixture intentionally contains a few invalid outputs so the baseline
demonstrates detection of malformed JSON, a hallucinated tool, an incorrect
result reference, unsupported false acceptance, and a preflight-valid but
semantically wrong plan.

## Live provider benchmark

Live execution requires the explicit `--live` flag and the existing provider
environment configuration. It always constructs `PLAN_ONLY` requests and wraps
all eleven registry callables with failing guards.

```bash
PYTHONPATH=src:. python benchmarks/planner/run_benchmark.py \
  --live --provider groq --model openai/gpt-oss-20b --repeat 3
```

Use `--case-id CASE_ID` repeatedly to select cases and `--output REPORT.json`
to persist the JSON report. The runner applies no quality threshold, so provider
nondeterminism is reported rather than converted into a normal test failure.

## Metric denominators

- Planning success, executable-plan rate, exact sequence accuracy, and false
  unsupported rate use expected-plan requests.
- Argument-binding accuracy compares the union of expected and emitted argument
  slots, so missing and extra bindings are both errors.
- Dependency/reference accuracy compares the union of dependency edges and
  reference producer/output relationships.
- Hallucinated-tool rate uses all emitted steps.
- Unsupported rejection and false acceptance use expected-unsupported cases.
- Semantic-wrong-but-preflight-valid rate uses all preflight-valid emitted plans.
- First-attempt and final semantic success use all requests. Schema-v4 reports
  transport-recovered and repair-recovered successes separately from initial
  attempt successes.
- Repair-attempt and repair-success rates report bounded complete regeneration.
  Failover-attempt and failover-success rates report use of the one explicitly
  configured secondary profile. Fallback remains JSON `null` because
  deterministic fallback is not implemented.
- Provider calls count `PlanningModel.complete()` invocations, not opaque
  SDK-internal behavior. Built-in adapters disable SDK automatic retries; a
  custom injected model remains responsible for any behavior hidden inside one
  `complete()` call.
