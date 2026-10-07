# Interaction / Model Layer Closeout

The interaction/model phase is complete as of 2026-10-07. This record describes
the accepted current state; earlier qualification and milestone records retain
their original outcomes.

## Baseline and preservation

At entry, `HEAD`, local `main` and local `origin/main` were aligned at
`8d79f6927ed1fca68e4b8e3439c6a58b61023ab2` (accepted IR2). The tracked tree
and index were clean; `git status --short` contained only `?? evals/`.
All 705 existing local evaluation entries remain byte/symlink-identical.
They are untracked operator evidence, not files to stage or commit.

This closeout changes only this document and the stale current IR2 status in
[the interactive benchmark README](../benchmarks/interactive/README.md).
No production, test, runtime/default configuration or historical record changes;
no new tests, live calls, scientific execution, staging, commit or push.

## Development-primary qualification

OpenRouter `openai/gpt-5.6-sol` is the accepted **current development-primary
LLM**, recorded by frontier closeout commit `e29f81d` and the
[qualification record](../benchmarks/interactive/README.md).

| Surface | Accepted observations |
| --- | --- |
| I01 supported inspection planning | PASS / PASS / PASS |
| I10 current-result referent resolution | PASS / PASS / PASS |
| I08 evidence-grounded scientific Answer | PASS / PASS / PASS, externally reviewed |
| I14 genuine Clarification | PASS / PASS / PASS |

The stronger model improved prior supported-planning and Answer instability on
the existing contracts without adding Agent semantic heuristics. This supports
the model-reliability explanation for those failures, without establishing
absolute causality or universal reliability. Further model search, keyword
routing or provider-specific semantic repair is not justified by this phase.

This is a development qualification decision, not a permanent production model,
exclusive supported provider or provider lock-in. Release-time qualification
remains future work. Historical failed and pending attempts remain valid evidence.

Runtime profiles/defaults remain operator-configured. The UI selects an admitted
profile; the backend supplies its model through the existing `PlanningModel`
abstraction. Development-primary acceptance does not change Web defaults or add
a built-in OpenRouter Web factory. Profile metadata is not scientific Revision
state. No model/profile capacity-admission framework was added; benchmark price
and output bounds remain qualification controls.

## IR1: Interpreter null-subject clarification

Accepted commit `5c37dac8afa962cf5c76fc35843e65db0502ab9a` changes wording only:
`subject=null` means the LLM asserts no new subject for this turn. It neither
clears a subject retained through the captured referent nor states that Agent
has no effective subject. The model selects an offered referent; Agent resolves
and retains exact authoritative identity without asking the model to reconstruct it.

Typed decisions, referent resolution, subject retention and Session/Revision
behavior are unchanged. Three fully recorded I12 trials pass: two select
`@previous_turn_result`, one selects `@focus`; all emit `subject=null` and admit
the exact retained annotation subject. The earlier collection-failed trial stays
unqualified. No semantic fallback or provider-specific behavior was introduced.
See the [Interpreter](../src/agent/application/turn_decisions.py),
[resolution owner](../src/agent/application/scientific_dialogue.py) and
[provider-contract tests](../tests/application/test_dialogue_provider_contract.py).

## IR2: clarity-first Guidance

Accepted commit `8d79f6927ed1fca68e4b8e3439c6a58b61023ab2` prioritizes semantic
correctness, model-facing clarity, true duplication removal, provider-neutral
deterministic representation and then request size. The mechanically lossless
encoding raised live stability concerns; unseeded A/B evidence did not prove
a causal regression.
Those experiments remain historical evidence, not the accepted interface.

The [Guidance renderer](../src/agent/application/scientific_guidance.py) sends the
complete direct Registry catalog and evidence. Aliases/shared-value dictionaries
are removed. Only identical registration, advisory status, execution-input scope
and applicable evidence handles are factored globally; capability-specific
requirements remain explicit. The inherited Registry record format is retained.
Full authoritative context/catalog, public readiness, candidate identity and
scientific semantics are unchanged. Existing prose restrictions were clarified
positively; validator semantics were not weakened or changed.

Independent equivalence, determinism, immutability, digest, schema, public
readiness and fixed-offer checks pass. The accepted focused suite reports
**313 passed in 87.71 seconds**; `git diff --check` passed before commit.

| Final targeted Guidance case | Contract, exact binding and external human review |
| --- | --- |
| Canonical I13 | PASS / PASS / PASS |
| Fixed/singleton follow-up | PASS |
| Larger selection case | PASS |

All five completed with `stop` below the unchanged 4,096-token output limit,
with zero retries, prompt/output repair or scientific execution. Requests are
10.36–10.43% smaller than verbose; I13 prompt plus schema is 44,099 → 39,499
bytes. Size is secondary and has no percentage acceptance target. This is bounded
development qualification, not biological execution or statistical reliability.

## Responsibility and architecture audit

| Owner | Responsibility |
| --- | --- |
| LLM | Language, semantic intent/referents, scientific/workflow reasoning, tool selection, explanation and Guidance semantics. |
| Agent | Typed contracts, structured inputs, Session/Revision, exact identity/binding, evidence, authority, lineage, Registry, deterministic validation, compiler, preflight, runtime and fail-closed correctness. |
| UI | Presentation, interaction and selection/input controls; scientific state remains backend-owned. |

Interpreter admission resolves offered captured references. Planner semantics stay
model-owned, followed by Registry-based compilation and preflight. Scientific
Answer uses accepted evidence and Agent-rendered complete claims, with integrity
and context rechecks. Guidance selects capabilities through the model and attaches
authoritative advisory readiness independently. Profiles/providers remain generic;
interactive submissions use the selected per-turn model without moving scientific
authority into UI or profile state. No unresolved interface defect was found.

Across qualification, IR1 and IR2, the only production-source changes are
Interpreter wording and Guidance presentation. Scientific tools/EpiZoo, Registry
semantics, compiler/preflight/runtime, evidence/authority/lineage, Session/Revision
semantics and provider-neutral ownership are unchanged. Existing bounded Planner
recovery remains unchanged; the no-retry statement above applies to accepted IR2
Guidance calls.

Preserve one authoritative owner per fact, minimal unified architecture and
fail-closed downstream authority/integrity/lineage validation. Integration and
performance do not justify changing science or reconstructing upstream science.
No keyword routing, workflow recipes, fuzzy semantic repair, Agent-side scientific
interpretation, model-specific semantic branches, output rewriting, new capacity
machinery or multi-model routing was introduced.

## Evidence and documentation scope

Existing local acceptance records were read in place and are not Git dependencies:

- `evals/ir1_i12_null_subject_2026-10-07/final_audit.json` and `results.json`;
- `evals/ir2_clarity_first_2026-10-07/closeout.json`, `focused_tests.json` and
  `size_measurements.json`;
- its `stage1/results.json`, `stage2/results.json`,
  `stage1_human_semantic_review.json` and `stage2_human_semantic_review.json`.

The committed Guidance source matches the accepted source, and separate human
reviews match exact response and ledger digests. The stale benchmark statement
that IR2 had not begun is replaced by the current acceptance/link. Historical
Groq/DeepSeek failures, earlier compression results, collection failures and
pending verbose V1/V2 reviews remain untouched; no verbose-baseline semantic
quality conclusion is inferred. Current architecture/evidence/link checks and
`git diff --check` pass. No regression suite was repeated for this documentation-only
closeout.

## Deferred next phase

**Next major phase: User Upload / Resource Admission.** It will cover user-provided
supported scientific inputs, beginning with H5AD, FASTQ, BAM and fragments, and
additional resources according to existing capability/resource contracts.
Upload UI, transfer/storage, admission, lifecycle, security, quotas and new
resource contracts are neither designed nor implemented by this closeout.

**INTERACTION/MODEL LAYER COMPLETE — ready for User Upload / Resource Admission planning.**
