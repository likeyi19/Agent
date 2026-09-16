# LLM Planner Quality Benchmarks

The existing runner has two explicit tracks. `--track v3` (the CLI default)
preserves the historical benchmark below. `--track scoped-v4` evaluates the
current two-stage Planner using the same registry guards and benchmark-only
role/semantic-policy scorer. Neither track supplies production routing or repair
feedback. Production planning, compiler authority, scientific tools, and recovery
are unchanged.

## Historical v3 track

This benchmark began as the Milestone 9.1 baseline for production
`LLMPlanner`. Its semantic workflow oracle exists only in
`benchmarks.planner.benchmark`; production orchestration never imports it.

The harness explicitly selects `PlanningWireMode.V3`, including for hosted-model
runs and recovery. Replay, binding scoring, and diagnostic interpretation retain
their historical v3 semantics despite the production LLM default becoming v4.
Semantic-v4 correctness remains covered by the separate v4 acceptance tests.

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
all currently registered scientific callables with failing guards.

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

## Current scoped-v4 track

Stage A capability selection → `PlanningScope` → Stage B semantic-v4 planning
→ compiler / preflight evaluation.

Run all 22 request instances offline, with no credentials or provider calls:

```bash
PYTHONPATH=src:. python benchmarks/planner/run_benchmark.py --track scoped-v4
```

`scoped_cases.json` uses corpus schema **2**, wrapping the existing case and
semantic policy with intent, wording, input-profile and scope expectations.
Scoped reports use schema **5**. Historical corpus schema 1, report schema 4,
wire v3 and their interpretation remain unchanged. Report version, diagnostic
version and planning-wire version are separate contracts.

The corpus contains **18 core intents + 4 wording variants = 22 scoped-v4
request instances**. The core corpus has 14 positive and four negative intents:

| ID | Intent |
| --- | --- |
| C01 | Processed inspection only |
| C02 | EpiZoo neighbors, clustering and UMAP; explicit device/resolution |
| C03 | Clustering and evaluation against fixed labels |
| C04 | Separate reference/query embeddings and label transfer |
| C05 | Evaluate an existing annotation |
| C06 | Transfer and evaluate query predictions |
| C07 | Declared counts → validation → replicate pseudobulk → paired DA |
| C08 | DA from fixed pseudobulk |
| C09 | FASTQ → fragments → QC → explicit selection → canonical matrix |
| C10 | Qualified corrected-CB BAM → fragments only |
| C11 | External fragments → QC only |
| C12 | Exact external canonical matrix adoption |
| C13 | Primary marker annotation of an already accepted canonical matrix |
| C14 | Independent processed and raw inspections |
| C15 | Missing indispensable QC-selection thresholds |
| C16 | Unresolved optional device scope across repeated embeddings |
| C17 | Unsupported ordinary-peak projection/adoption/annotation composition |
| C18 | Unsupported RNA-only work |

C02, C04, C07 and C13 each have a `-p1` paraphrase with identical inputs and
scientific constraints. Fixtures use synthetic declarations and paths; they
establish planning expressibility, not biological artifact validity. Both
planning prompts expose input names/types, not input values or file contents.
No scorer expectation requires the model to inspect those hidden contents.

### Observations and scoring

Stage A is observed independently. Each case declares required coverage,
alternative complete coverage sets, plausible optional families and unrelated
families. Alternatives support overlapping tool membership. Required tool
visibility is checked against registry-derived family membership. Missing
coverage fails; plausible extras do not. Unrelated extras are reported as an
efficiency finding. Stage-A `unsupported` is correct only when no family is
relevant: C15–C17 should select relevant families and fail closed at Stage B.

The observer parses each returned response and independently invokes the existing
pure compiler/preflight for measurement. It does not alter the response or feed
oracle results to the Planner. Semantic candidate summaries retain ports and
branch references. Compiled summaries retain authorized request-binding origins,
grouped-channel expansion and dependencies, without values. Role matching ignores
step IDs and accepts declared binding alternatives, independent branch ordering,
auxiliary inspection and default-equivalent omissions. Control-only dependencies
are retained in summaries but do not create scientific dataflow for terminal-role
checks. Canonical sequence imitation is not a success criterion.

Each positive fixture has an expressible calibration witness checked before any
provider invocation. That witness is one valid realization, not the scoring
definition. Negative cases accept an unsupported decision or the specifically
declared missing/ambiguous-source diagnostic and target; arbitrary failures are
not safe-refusal successes. A preflight-valid plan that violates intent fails the
oracle without triggering extra recovery.

Reports keep separate phase, production diagnostic stage/code/reason, evaluation
failure kind and root-cause assessment. Failure kinds cover transport, scope,
wire/schema, tool/binding/dependency/source, intent, missing information,
unsupported handling, compiler/preflight and repair failures. A compiler rejection
is a detection boundary, not evidence that the compiler is defective.

### Denominators and recovery

Every rate contains `numerator`, `denominator` and `rate`; an empty denominator
has `rate: null`. There is no composite score.

- Family recall uses observed required coverage opportunities; complete-scope
  rate excludes no-relevant-family cases. Multi-family completeness is separate.
- Initial parse/schema, compiler and preflight rates use attempts observable at
  that boundary. Not-reached boundaries are null, never silently failed.
- First-candidate plan correctness uses expected-plan cases. Initial outcome
  correctness also includes expected refusals. Final correctness is reported both
  for all outcomes and for positive plans, with separate refusal metrics.
- Conditional Stage-B metrics include only sufficient scopes. Scope omissions
  do not create additional independently attributed Stage-B hard-error counts;
  the actual rejection diagnostics remain available.
- Hard-error category counts count affected sessions, including initial errors
  retained after successful repair. They exclude transport and scope-overselection
  findings and are not an additive overall score.
- Transport-only failures have unobserved quality. Operational plan acceptance
  still includes transport-blocked positive sessions in its denominator.
- Operational repair yield uses all invoked repairs. Semantic repair success
  uses returned, parse/schema-valid evaluable decisions (including valid refusals).
  Parse/schema-invalid repairs and transport-blocked repairs are separately visible;
  neither disappears from operational yield.
- Wording-pair consistency requires both variants to satisfy their shared intent.
  Two wrong plans do not count as consistency; transport-incomplete pairs are null.

Normal scoped success is **two calls, no recovery**. Production diagnostics
schema 5 identifies phase and attempt kind. Existing recovery is unchanged:
one Stage-A call without recovery; Stage B can use one transport retry OR one
repair, then at most one explicitly configured final failover; scope stays frozen
and the global ceiling is four. No secondary profile is configured by default.
The Python API can observe an explicitly supplied recovery profile/factory for
offline calibration or a separately designed comparison.

Reports retain the initial failure, sanitized correction identifiers, correction
fingerprint, repair outcome and new-error observations. Feedback sufficiency is
unassessed until audited. Original-error correction is true on successful repair,
false for an identical rejected candidate/error, and otherwise unknown; a different
error alone does not prove the original problem was fixed.

### Root-cause adjudication and privacy

`assess_root_cause` records explicit evidence: returned candidate, fixture validity,
interface reconstruction, expressibility, information clarity and demonstrated
interface defect. Default root cause is `unresolved`. A proven interface defect
can support `agent_interface`; a clear, expressible option violated by a returned
candidate can support `llm_reasoning`. Shared failures across models do not
automatically establish an interface defect. Invalid fixtures/harness observations
are excluded separately. Invalid positive witnesses stop evaluation before calls.
Adjudication is report-only; it never influences production planning or recovery.

`s2_calibration()` records only the accepted rejected edge and diagnostics:
`embed.dataset → neighbors.embedding`, `WRONG_SOURCE_PORT`,
`producer_channel_incompatible`, root cause `llm_reasoning`. Its full candidate
summary and rate-limited repair quality remain unknown.

Provider responses stay transient. Reports contain normalized local step IDs,
known identifiers, outcomes, byte counts and fingerprints, never raw responses,
reasoning, prompt text, request paths/values, HTTP bodies or exception prose.
The comparison manifest binds commit, working production/evaluator source,
registry/capabilities, corpus, installed dependency fingerprint, wire mode and
recovery policy. Profiles retain provider/model/timeout; built-in adapters have
zero SDK retries and no explicit temperature/seed/top-p controls. Custom models'
hidden calls are not observable. Live generation is not claimed deterministic.

### Six-case live smoke — partial evidence (2026-09-16)

The agreed smoke is C02, C04, C13, C14, C15 and C18, one repetition each. Rehearse it
offline with repeated `--case-id` arguments. Only after separate authorization,
add `--live --provider PROVIDER --model MODEL`. Scoped-v4 live CLI defaults to
these six IDs when no selection is supplied; offline defaults to all 22 instances.
Explicit case IDs allow smaller batches. No automatic secondary model is used.

Normal smoke cost is approximately 11 logical calls. On a rate-limited session,
the runner lets existing bounded recovery finish and stops launching new cases.
The report lists pending request IDs/repetitions for a later quota window. It
does not change the five-second production retry cap or rerun cases until success.
Freeze comparison-manifest identities and input cases when comparing models.
Fixed-scope Stage-B comparison is deferred and cannot replace end-to-end failures.

The first real smoke used Groq `openai/gpt-oss-120b` and stopped after HTTP 429.
Only C02 produced an evaluable semantic candidate: required-family recall and
complete scope were both 1/1, with zero unrelated selections or false unsupported
outcomes. Its initial candidate used `neighbors.analysis → cluster.analysis`,
while the offered producer port was `neighbors`. The compiler correctly rejected
it with `WRONG_SOURCE_PORT`. An offline audit reproduced the exact interface
fingerprints, confirmed preserved scoped semantics, and passed a correct candidate
through parser/compiler/preflight/scoring; the initial error is `llm_reasoning`,
not `agent_interface`. This is a different edge from the earlier S2 error.

The permitted repair was transport-blocked: two provider calls returned, the
third received HTTP 429, and no repair candidate was observed. C04, C13, C14,
C15 and C18 remain pending/unobserved, not failed. Scientific executions and
execution-entry attempts were both zero. Sanitized reports and offline audit
remain local under `evals/scoped_v4_quality_smoke_2026-09-16_r1/`, outside Git.
No new Agent-side Planner defect was demonstrated. Broader model-quality
measurement remains incomplete because of provider rate limiting; one evaluable
case does not establish broad accuracy or require a stronger model.

### Offline acceptance

Calibration covers all 22 fixtures, valid DAG variations, scientific mutations,
scope omissions, false refusals, transport separation, original-error retention,
four-call failover, privacy, execution guards and S2 classification. The existing
v3 tests remain part of the benchmark regression. These are evaluator-readiness
results, not measured live LLM quality.

Accepted offline validation: **46 scoped-v4 calibration tests**, **115 complete
benchmark regression tests**, and **1,204 broader benchmark/orchestration/provider/
Application tests** passed, with three existing warnings in the broader run.
The six-case offline rehearsal produced six correct outcomes in 11 scripted
provider calls, with zero recovery calls and zero scientific calls. These results
were retained for documentation-only closeout; no additional live calls were made.
