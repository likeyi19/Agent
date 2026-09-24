# Semantic-minimum interaction decisions

Starting HEAD: `be1a3c4213935c9d2361b87a30cb24335ec9735e` (M17.1).
This is a bounded interface cleanup, not a new milestone or provider adaptation.

The LLM authors semantic choices; Agent code derives facts uniquely determined
by the chosen branch. The provider-facing decision schema now omits:

| Branch | Removed model-authored fields | Agent-derived internal values |
| --- | --- | --- |
| `answer_scientific` | `intent` | `intent='scientific'` |
| `execute_plan` | `operation`, `base`, `delta` | `operation='plan'`, `base='current'`, `delta=None` |

`turn_decisions.parse_decision` normalizes these branches into the existing
`Answer`/`ScientificQuestion` and `Execute` types. `scientific_dialogue.admit`,
`dialogue_execution.admit` and the existing execution/grounding paths retain
their responsibilities. No utterance-based guessing supplies the removed fields.
The same derivation is correct if every currently tested provider disappears:
the branch already determines each removed value, independently of model behavior.

The LLM still chooses discussion versus execution, the registered tool, exact
result/subject and comparison references, and discussion focus. M15 intent deltas
retain their explicit parameter/operator/literal/evidence checks. Guidance keeps
semantic evidence/capability selection and conditional rationale; Agent owns
stable references and incomplete readiness. Planner scope, symbolic sources,
semantic target choices and genuine control ordering remain model-authored.
Protocol versions and discriminators remain for strict parsing and dispatch.

## Compatibility and fail-closed admission

Previously emitted explicit tagged forms and the existing legacy `answer` and
`execute` forms are used by scripted callers/tests and the public decision parser.
The new schema emits only the minimal tagged forms. The parser also accepts a
complete historical tagged form when its fixed assertions agree exactly. Partial
execution assertion groups, conflicting values, extra fields and missing semantic
fields fail closed. Historical tagged execution with a non-current base or non-null
delta was never executable under `dialogue_execution.admit`; it now fails earlier
in parsing. Legacy forms retain their existing admission checks.

Persisted interactions store admitted canonical semantics, not raw provider wire
responses. Typed decisions and persisted formats are unchanged; no migration or
schema-version change is needed. Unknown tools/results, unauthorized revisions,
ambiguous subjects, incompatible evidence and unsupported effects still require
the same deterministic admission. A minimal execution branch does not authorize
guidance-candidate handoff.

Only two prompt sentences change: the scientific instruction stops asking for
`intent`; the execution instruction stops asking for `operation/base/delta` and
states their Agent-derived current-revision semantics. No other prompt rewrite.

## Non-goals and validation

No changes to Planner schemas, scoping, catalog compression, recovery, providers,
Registry semantics, scientific contracts, compiler/preflight, evidence authority,
or M15/M16/M17 architecture. No live-model testing and no M17.2 implementation.

Final focused command (86 passed in 12.30 seconds):

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
/home/likeyi/anaconda3/envs/agent/bin/python -B -m pytest -q \
  tests/application/test_dialogue_provider_contract.py \
  tests/application/test_scientific_dialogue.py \
  tests/application/test_scientific_guidance.py
```

The broader regression uses the exact focused file list recorded in
[M17.1 validation](m17.1-read-only-scientific-guidance.md#validation), including
application, dialogue, guidance, authority and Planner/compiler/preflight tests.
Its expanded command and output are recorded in ignored
`outputs/semantic-minimum-cleanup/command.json` and `regression.log`.
Result: **517 passed in 105.69 seconds**, exit 0, no failures, skips or warnings.
All inherited `RUN_*` gates were removed. The command was executed through:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/likeyi/anaconda3/envs/agent/bin/python -B \
  outputs/semantic-minimum-cleanup/regression.py
```

Final `git diff --check` passed. Registry remains 23; EpiZoo remains clean at
`029cd631d0f9806a646c4a3b42ce10b958f2b67f`.
The full-project suite is deferred for review because only two interaction wire
branches change; the historical M17.1 full-suite result is not a run on this diff.

Nothing is staged, committed or pushed.
