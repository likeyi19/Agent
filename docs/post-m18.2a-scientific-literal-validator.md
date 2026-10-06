# Post-M18.2a — Scientific Literal Validator Correction

Baseline: `72a5ab0c64fff3dd0be7f793db95ed925529c215`
(`Simplify typed turn admission`). This is a narrow conversational validator fix.

## Correction and ownership

The accepted single-call qualification returned valid structured JSON, seven
offered inspection claim references and reviewed meaning `m0`. Rendering rejected
`zero` inside `non‑zero entries` (U+2011), mistaking the technical metric name for
an unbound numerical assertion.

The literal validator now recognizes complete, word-bounded `non-zero`,
`non‐zero` (U+2010) and `non‑zero` (U+2011) lexemes within each original text part,
case-insensitively. It folds only their internal hyphen in a validation copy;
`nonzero` already needs no folding. The existing concatenated literal regex is
unchanged. Recognition cannot span adjacent text, claim or meaning parts.
Categorical checks and rendering still use the original prose.

Standalone number words, digits, braces, paths/URLs, supplied categorical values,
unknown references and invalid comparison bindings retain their rejection.
No sentence whitelist, general compound-word exemption or semantic repair is
introduced. Agent retains accepted values, identities, provenance and binding;
the LLM retains claim/reference selection, support judgment and explanation.
Evidence semantics, prompts, schemas, Planner/compiler/runtime, scientific tools
and all deferred interaction/UI/input/upload work remain unchanged.

## Validation

- **152 focused tests pass in 13.80 seconds** across scientific dialogue,
  closeout, detail, provider-contract and guidance suites. The 29 new cases cover
  five positive spellings/case variants, fifteen unbound-literal negatives and
  nine split-part attacks. Positives bind `nnz` explicitly and preserve its exact
  accepted value/source and original prose. Negatives include standalone zero,
  ninety nine, digits, other hyphenated number words, paths/URLs and an alphabetic
  categorical value.
- The exact saved 662-byte Groq response passes production generation, strict
  parsing and rendering offline with unchanged prompt/schema, all seven claim
  values/sources, `m0` and original U+2011 wording. No provider call is made.
- One completed relevant regression across `tests/application` and the six
  existing dialogue/intent/guidance authority suites passes **521 tests in
  582.37 seconds**, exit 0. All `RUN_*` gates are disabled; no tests are excluded.
  All 403 frozen source/test/resource files remain unchanged during this run.
  An earlier partial regression was stopped for the split-part correction.
  The full repository suite is not run.
- `git diff --check` passes. All 125 pre-existing nonignored untracked eval files
  and the two ignored eval files retain their content manifest.

The optional native Firefox follow-up used the existing accepted inspection and
normal browser/HTTP/application submission once with `groq-gpt-oss-120b`. Its sole
live interpreter call chose operational `intent="matrix"`; that route returned
`I cannot establish that from the current verified state.` Scientific-answer
generation and this validator were not reached. This separate routing result
does not qualify live browser scientific-answer success; no retry or scope
expansion follows. An initial Firefox profile startup failure made zero calls.

Only one ordinary failed interaction and its empty lease file are appended. The
active Revision, generation 1, all prior interactions and scientific Run/evidence/
report bytes remain unchanged. No new science, planning or execution occurs;
guards record zero forbidden attempts. Temporary browser/server processes stop,
their ports close and the temporary Firefox profile is removed. Credentials stay
environment-only. External records are under
`/tmp/agent-literal-validator-correction.en3ww9ic/`.

Post-M18.2a software acceptance passes and the change is ready for review with
the optional browser limitation above. Changes remain unstaged; no commit or
push is performed.
