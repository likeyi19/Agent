# UA3.6.3b — Unified qualified scientific resource connection

Implemented on 2026-10-12 against committed UA3.6.3a
`bcb24ea6df80e3d5562cea7cb62651213165767a`. Existing operator InputSets now retain
their exact choice identity through registered-source submission, and existing
Registry resource members are checked against actual compiled/resolved consumers.
One genuine four-owner raw workflow uses unchanged production-qualified mouse
resources; an independent BAM witness qualifies binding and owner compatibility.
This stage does not provision the current lab deployment's missing raw InputSets.

## A. Baseline and preservation

Admission found HEAD, local `main`, local `origin/main` and independently read
remote `main` equal to the commit above. The tracked worktree/index were clean;
only the existing `?? evals/` was untracked. The UA3.6.3a record was present.
Remote inspection used `git ls-remote origin refs/heads/main`. No reset, clean,
fetch, pull, staging, commit or push is performed.

`/tmp/agent-ua363b-preservation-wc4x5wbk` captures all **622 tracked baseline
files before editing**, index bytes and refs. Its `protected-before.json` covers
**45,062 protected files**: the previous 23,624 protected records, 21,425
UA3.6.3a final acceptance/browser/prompt files and 13 additional production
resource/runtime paths. These include all 1,274 existing evaluation files, 198
outputs, original operator configuration, preserved real scientific/application
state and the large checkpoint. Ordinary records pin SHA-256, size, mtime and
mode. The 5,231,645,507-byte EpiZoo checkpoint is metadata-only and is neither
loaded nor rehashed. The existing large mouse FASTA and annotation are also
preserved by metadata; the authorized scientific owners perform their necessary
integrity and source checks during the bounded witness.

All new fixtures, isolated configuration, resource declarations, Runs, Sessions,
reports, caches and logs use fresh `/tmp` workspaces. Native Snap Firefox uses
only a fresh home-directory profile. No original operator configuration or
credentials are changed. `implementation-final-freeze.json` pins all **521**
source/test/benchmark files before the final domain regressions.
All 521 remain byte-identical with unchanged inventory after both domains pass.
The final preservation comparison finds **zero mismatches across all 45,062
protected records**. The complete evaluation/output inventories remain unchanged;
index bytes, all three local refs and final remote `main` remain exact. Of 622
tracked baseline files, 614 remain byte-identical; the eight intentional changes
are the six production files and two existing tests in section D. Three new test
modules and this document remain unstaged. Final test, preservation, freeze and
witness pins are collected in `completion-validation.json` in that directory.

## B. Existing mechanism reused

The existing path already works:

```text
registered source / completed FASTQ library + approved ScientificInputSet
  → existing LocalResourceAdmission composition
  → ordinary Interpreter / scoped Planner
  → Registry/compiler/preflight
  → scientific owners and independent verification
  → RunStore / Session Revision / accepted result delivery
```

Fragments composition already accepts import, QC, explicit selection and canonical
matrix companions. BAM/FASTQ composition already supports inspection without
producer-only resources, exact intake/library context and source-specific producer
requirements. H5AD/EpiZoo resources, ordered multi-input roles, exact-string/scalar
binding, same-plan dependencies, accepted historical authority and downloads are
reused. Explicit/default resource selection already has a Web control and policy.

The relevant records are [fragments upload](ua3.4.1-external-fragments-web-upload.md),
[BAM upload](ua3.4.2-bam-web-upload.md), [FASTQ upload](ua3.4.3-fastq-web-upload.md),
[capability readiness](ua3.5.4d-practical-capability-readiness.md),
[multi-input roles](ua3.6.3a-unified-multi-input-role-binding.md),
[production Chromap resources](m11.6c-production-chromap-resources.md),
[production QC resources](m11.6b-production-qc-resources.md),
[raw biological closeout](m11.8-biological-closeout.md) and
[trusted result delivery](ua3.6.2c-unified-scientific-result-delivery.md).

## C. Gap classification and current readiness

Two common implementation gaps are confirmed. Web extracted `.inputs()` and
dropped `ScientificInputSet.input_set_id`, so different approved choices with
identical values produced indistinguishable submissions/retries. Local resource
admission pinned the raw producer arguments but did not independently compare
captured downstream QC/reference resource members after compilation. The normal
scoped compiler already binds exact aliases; the added guard also rejects direct
compiled/resolved member substitution.

The safe current operator audit finds **two InputSets and zero raw companions**:
one ordinary accepted H5AD input set and one mouse species companion. There are
two reviewed EpiZoo entries/defaults, for human and mouse; the historical human
mapping gap in UA3.5.4d no longer describes current state. No private configuration,
credentials or resource payload is printed by this audit.

| Raw family | Existing handoff / qualification owner | Current missing prerequisite category |
| --- | --- | --- |
| External fragments | Exact registration, source profile/namespace/reference, then QC/selection/matrix owners; full reviewed mouse resources are available and used below. | Operator provisioning: no configured fragments InputSet. Resource/source compatibility still requires owner admission. |
| BAM | Exact uploaded BAM source, existing intake/sidecar checks, declared assay/species/assembly and corrected-CB library context; bare inspection remains supported. | Operator provisioning and source-specific eligibility: no configured BAM InputSet or context qualified for a newly uploaded source. |
| FASTQ library | Explicit completed membership/layout, accepted intake, library/barcode context, whitelist and qualified reference/index/runtime. | Operator provisioning/source-specific qualification: no configured FASTQ InputSet or library context for a new upload. Historical qualified resources do not confer new library history. |

Missing provisioning is not repaired in external configuration. Incompatible or
absent source history is not a value the researcher can qualify by assertion.
No filename/species-name matching, resource discovery or compatibility fallback
is added. The UI knows approved candidate families; owners establish applicability.

## D. Minimal implementation

| Changed production file | Purpose |
| --- | --- |
| `src/agent/application/local_resources.py` | Optional immutable choice token on existing single/collection/group bindings; closed reconstruction; exact Registry resource-member consumption checks. |
| `src/agent/application/uploads.py` | Forward the optional token through existing upload resolution. |
| `src/agent/application/dialogue_execution.py` | Preserve choice capture through role/scalar replay and continuation; validate original attribution; retain supported explicit species-resource completion. |
| `src/agent/web/config.py` | `ScientificInputSet.resource_choice()` derives exact configured definition identity. |
| `src/agent/web/app.py` | One shared choice-capture helper following the unchanged explicit/default policy; forward the selected token. |
| `src/agent/web/static/app.js` | Filter fragments choices by the existing family flag and describe configured BAM/FASTQ candidates accurately. |

New focused modules are `tests/application/test_resource_choice_identity.py`,
`tests/web/test_qualified_resource_choices.py` and
`tests/web/test_qualified_resource_handoff.py`. Existing `test_species_handoff.py`
adds captured-choice persistence assertions; `test_upload_static.py` updates the
two labels and adds one filtering/truthfulness check.

No Registry, compiler, scientific algorithm/owner, authority schema, operator
configuration, provider transport or public scientific tool changes. Inventory
remains **23 tools and 172 arguments**. No extra catalog, parser, state manager,
workflow engine, resource form or upload service is created.

## E. Ownership

The Interpreter owns task intent and explicit scientific declarations; the Planner
proposes registered steps/dependencies. Neither issues resource qualification.
The operator supplies approved immutable InputSets and genuine owner-qualified
resources. Agent captures exact source/choice identities and enforces typed
bindings, generation/retry identity and compiled consumption. Registry/compiler
retain semantic admission; scientific owners retain source history, reference
compatibility, integrity, numerical methods and independent correctness checks.
Web transports the existing selection and displays accepted results.

## F. Exact resource-choice handoff

The additive inline private field is:

```text
resource_choice = {input_set_id, definition_sha256}
```

The digest uses the existing canonical digest over all immutable InputSet fields,
including identifier, label, execution inputs and companion/default policy. The
token is frozen and closed; partial/extra/malformed forms and explicit null
attribution reject. Absence preserves legacy attribution bytes. Exact paths and
scientific resource digests remain trusted typed execution values, never model
inputs or browser arguments. The HTTP schema accepts only configured InputSet IDs;
the server authors the token. A different choice ID or definition conflicts on
retry even when its scientific values are identical. Explicit/default use of the
same configured choice remains equivalent.

The token records operator configuration; it is neither scientific authority nor
authentication of arbitrary trusted-Python-supplied digest tokens. Scientific
resource files still require the existing pinned owner contracts. No definition
database or second proof layer is introduced.

For bindings carrying the token, existing Registry artifact kinds for reference
bundles, QC references and library contexts identify request-only resource ports.
Complete captured request-source member groups must equal the actual compiled and
resolved consumer members. Substitution rejects with
`LOCAL_RESOURCE_CONSUMPTION_MISMATCH` before execution or first Session acceptance.
This check reads no resource payload. Ports with accepted upstream resources retain
the existing compiler/dependency contracts; legacy bindings retain their policy.

The single existing resource selector is kept. It filters raw families using
configured flags, uses one explicitly designated default where the existing policy
allows it, and requires explicit choice for other candidates. Configured choices
are described as approved candidates. No browser claim of scientific qualification
or resource matching by label is added.

## G. Genuine production-resource scientific witness

An isolated configuration supplies the explicitly selected `reviewed-mm10` InputSet.
The new source has three synthetic external fragment records, explicit namespace,
`10x-atac-fragments.v1` and truthful `source_selection=unknown`. Existing reviewed
full mm10 parent/GENCODE M25 QC resources and the unchanged immutable catalog are
used; `AGENT_QC_ALLOW_SYNTHETIC` is **unset**. Actual qualification is
**`operator_qualified`**. No resource is shortened, rebuilt, self-attested or
promoted from a synthetic bundle.

| Exact resource identity | SHA-256 |
| --- | --- |
| Original QC catalog | `9d9ea0340640fbab74f5f591831592ad3badf5973a09fcb988a7e2cba10e5eae` |
| Mouse parent manifest | `95c2962bbfa1bfbadc5b3e24ad23d5dd6f9203371c3d6071484c69f1f1979408` |
| Mouse QC manifest | `3b3214e03dcb549cdef9c82f07038be74232d38163a4e70c32ae934da5ad5038` |
| Explicit InputSet definition | `4c79b7dc7fa4598fc56478985f909e7187d957a6ea4bc9aae70ada9df2244b1e` |

The ordinary uploaded-source request is:

> Import these external fragments with the selected reviewed reference, compute barcode QC, select candidates using the explicitly provided minimum depth 1 and minimum TSS enrichment 1, and construct their cell-by-cCRE matrix.

Only model decisions are scripted. Actual upload/registration, Interpreter,
Planner, scoped-v4 compiler, preflight, import/QC/selection/matrix owners,
independent verification, RunStore and Session publication execute. Import retains
three records with diagnostic support 4. QC observes two barcodes; explicit
selection keeps one and rejects one. The sparse int64 matrix is **1 × 1,341,077**,
nnz 1, total count 1, on the complete ordered canonical feature axis. It is never
densified. The thresholds are fixture-only explicit policy; calling remains
`none/not_assessed`. This synthetic source establishes software/resource execution,
not biological accuracy, calibrated thresholds or mouse biological generalization.

| Actual observed body | Original execution | Evidence/delivery, duplicate and fresh-instance delta |
| --- | ---: | ---: |
| Import / QC / selection / matrix production | 1 each | 0 |
| Independent import / QC / selection / matrix owner | 1 each | 0 |
| External source / matrix reconstruction | 1 / 1 | 0 |
| QC-resource source reconstruction | 2 | 0 |
| Parent-reference source reinspection | 6 | 0 |
| Compiler / preflight | 1 / 2 | 0 |
| Normal step verifier | 8 per tool | 0 |
| Shared file closure | 67 | 0 |
| Scripted model completions | 4 | 0 |

The repeated normal checks are existing lifecycle/presentation/acceptance work.
Counts come from actual Python code-body profiling; no numerical producer or
independent verifier is replaced. Five eligible downloads pass byte count and
SHA checks: the 161,573,536-byte matrix H5AD and four bounded gzip tables for
barcode QC, lengths, decisions and selected cells. Accepted evidence and the
existing safe readable report are available. The uploaded source is archived
after acceptance; completed duplicate and forbidden-provider fresh-instance
reads preserve the one Revision/generation 1 without new science or providers.

The directly emitted witness is
`/tmp/agent-ua363b-production-qualified-8thjqs98/scientific/production-resource-witness.json`.
Run is `turn-d143591821e63323d72e9ede4fb3d5150b72b6a0f0a0ad64037dd58bf1b69d77:run`;
Revision is `a50086f05adc2d9d3970746f40a35d9c522f2cfaad6921535b8b824a20bd8434`.
Acceptance takes 430.09 s; complete execution/delivery/fresh-instance checks take
494.58 s, exit 0. Exact command/environment, full log and exit record are
`command.json`, `stdout-stderr.log`, `result.json` in that acceptance directory.

This production process loaded the choice-capture implementation before the final
pure resource-member guard was added. A fresh process subsequently checks exact
persisted compiled/resolved members and four mutated QC/matrix members against
the final guard, with zero observed science/provider/integrity work. Its record is
`scientific/final-resource-admission.json`. This is exact resource-consumption
revalidation, not new scientific qualification, authority issuance or first
Session acceptance. Two genuine tiny four-owner executions in the focused suite
exercise the final implementation end to end under the explicitly supported
`synthetic_only` test qualification. No second costly production run is claimed.

## H. Independent raw-family witness

The existing BAM fixture uploads two byte-identical files with distinct registered
locators. Genuine bounded intake and explicit corrected-CB library declarations
qualify context against the first actual uploaded source. The exact bounded intake
is explicitly supplied through the existing semantic intake request port; the new
witness compiles one prepare step, not same-plan inspection. Existing same-plan
handoff regression remains separate. The common choice binding,
compiler/preflight and existing owner binder accept the exact first association.
The second upload cannot reuse the first intake/context despite matching bytes:
`LOCAL_RESOURCE_BINDING_INVALID` and `BAM_FRAGMENTS_CONTEXT_MISMATCH` reject.

This witness qualifies shared binding/compiler/owner-compatibility behavior,
with zero BAM producer executions, Runs or Revisions. It is not a second executed
raw-production workflow. The direct artifact is
`/tmp/agent-ua363b-final-focused-i9g7sssb/pytest-temp/test_bam_existing_owner_reject0/bam-resource-binding-witness.json`.
All three raw default/explicit/no/ambiguous/wrong-family policies are also checked.

## I. Negative coverage

Focused acceptance rejects changed/stripped choices, same-ID changed definitions,
conflicting retries, malformed attribution, absent/ambiguous/wrong-family choices,
source replacement, altered producer source/path/hash/profile/namespace/reference
arguments, and changed downstream QC/reference path or digest members. Existing
scientific owners retain ordered/source/reference/history validation; no Web
scientific validator or guessed fallback appears.

The actual QC owner rejects an otherwise configured tiny resource when both
production attestation and synthetic opt-in are absent: `QC_RESOURCE_UNQUALIFIED`.
The already completed import does not become full preprocessing success; QC,
selection and matrix production remain zero and no successful Revision activates.
An initial import reference mutation retains existing `FRAGMENTS_SOURCE_MISMATCH`;
downstream member mutation uses the new exact-consumption error. One test expectation
was corrected to preserve this existing error distinction; no science was repaired.

## J. Multi-turn, retry and historical behavior

Selection is immutable captured execution input. Group role binding, scalar/string
replay, source identity and pending parameter completion retain the choice token.
An altered captured choice/source cannot redirect a pending raw parameter answer.
The established species completion may explicitly choose a compatible current
InputSet for the same original H5AD; its new turn captures that choice while the
original request remains immutable and original scientific values remain checked.
This preserves the existing valid mouse/model completion rather than imposing a
new global ban on resource choices in follow-up turns.

Completed views, navigation, refresh and duplicate retries do not reopen original
sources or requalify resources. Missing producer context remains different from
an answerable scalar declaration; BAM/FASTQ inspection remains available under
its existing contracts. No new pending store, automatic resource provisioning,
unfinished-science recovery path or source-history adoption is implemented.
Standalone source-aware unfinished group-run resume remains outside the prior
UA3.6.3a qualification. Existing authority and eligible delivery remain unchanged.

## K. Validation and model footprint

All commands use conda `agent` Python, `-B`, `PYTHONPATH=src:.`, removed `RUN_*`
gates, private caches/basetemp and `-p no:cacheprovider`. No full regression,
paid/live provider, foundation inference, adaptation training or real PBMC
reprocessing is run. The two final domain commands have independent isolated
workspaces; their counts overlap focused selections.

| Final selection | Result |
| --- | --- |
| Three new modules, species continuation, upload static and multi-input role admission | 98 passed, three existing warnings, 71.28 s; exit 0. |
| Complete Application/Web domain | 2,460 passed, nine gated skips, three existing warnings, 837.74 s; exit 0. |
| Orchestration, raw external/BAM, QC/library resources and conversational authority | 1,597 passed, 14 gated skips, three existing warnings, 682.93 s; exit 0. |
| Genuine production-resource execution/delivery/reopen | Passed; 494.58 s; exit 0. |
| Native Firefox | 25 checks passed: 14 initial, 11 fresh-server; exit 0. |

Earlier supporting selections pass 412 resource/contract checks and 14 scientific
handoff cases. They overlap the final selections and are not unique totals. The
four-case species recheck passes after preserving legitimate explicit compatible
selection during a new species continuation; the earlier overstrict draft caused
two failures and was corrected before final focused acceptance.
All nine Application/Web skips and all 14 raw/orchestration skips require the
existing explicit `RUN_CHROMAP_QUALIFICATION=1` gate in `tests/chromap/conftest.py`.
The raw command's 14 are its selected backend-qualification cases; Application/Web
has three Application and six Web FASTQ cases. No selected tests are excluded.
The three warnings are the existing `pkg_resources`, `louvain` and TBB/Numba notices.

The focused command selects exactly:

```text
tests/application/test_resource_choice_identity.py
tests/web/test_qualified_resource_choices.py
tests/web/test_qualified_resource_handoff.py
tests/web/test_species_handoff.py
tests/web/test_upload_static.py
tests/application/test_multi_input_role_admission.py
```

Its exact command/log/result are
`/tmp/agent-ua363b-final-focused-i9g7sssb/{command.txt,pytest.log,result.json}`.
The complete Application/Web command selects `tests/application tests/web`;
records are `/tmp/agent-ua363b-app-web-final-bf6axpbd/{command.txt,pytest.log,result.json}`.
The raw/orchestration command selects exactly:

```text
tests/unit/orchestration
tests/external_fragments
tests/bam_fragments
tests/tools/data/test_scatac_qc_reference.py
tests/tools/data/test_scatac_library_context.py
tests/barcode_qc/test_qualification.py
tests/chromap/test_backend_qualification.py
tests/authority_dag/test_conversational_raw_context.py
tests/authority_dag/test_conversational_processed_continuation.py
tests/authority_dag/test_captured_planning_context.py
```

Its exact command/log/result are
`/tmp/agent-ua363b-raw-orchestration-final-4d0ve99q/{command.txt,pytest.log,result.json}`.
All use `/home/likeyi/anaconda3/envs/agent/bin/python -B -m pytest` with `-q
-p no:cacheprovider` and the named workspace's `pytest-temp` as `--basetemp`.
The replay runner is `/tmp/agent-ua363b-run-selection.py`.

Frozen representative scoped-v4 prompt/schema remain byte-identical to admitted
HEAD: **21,013 + 12,985 = 33,998 UTF-8 bytes**, below the unchanged 34,000 bound.
Replay is `/tmp/agent-ua363b-prompt-measurement-epp977i4/{replay.py,result.json}`.
Actual Interpreter comparison with metadata absent, one explicit InputSet, the
same unique default and maximum 64 configured raw choices with maximum Unicode
labels is byte-identical: **21,016 + 11,146 = 32,162 bytes**. Resource-choice
metadata adds **zero model bytes**. The clarification-only fixture performs no
planning, science, Runs or Revisions; it is not scientific resource qualification.
Records are `/tmp/agent-ua363b-resource-prompt-dun4etf4`.

The actual production request's variable Interpreter prompt/schema total is
**37,830** bytes; scope/detailed/output selection totals are **3,031 / 15,086 /
3,012** bytes using its recorded JSON schema encoding. This is existing variable
context, not a new resource-option projection or universal 34,000-byte claim.
`scientific/production-prompt-privacy.json` records source/resource/choice digests
and private locators absent from all four prompts. No budget increase, provider
special case, transport change or declaration truncation occurs. Live semantic
role/resource interpretation accuracy remains unqualified.

Native evidence is `/tmp/agent-ua363b-native-firefox-TqAVQ9/browser-summary.json`,
with exact commands in `commands.json`, retained logs and reviewed screenshots.
Only fresh Snap profile `/home/likeyi/agent-ua363b-firefox-4qjfysmo/profile` is used.
The browser selects an explicit override through the existing selector, preserves
it across pre-execution refresh, submits the ordinary composer, runs four genuine
tiny owners, views evidence/report and verifies pinned matrix delivery bytes
through a same-origin browser fetch and SHA check. No native filesystem-save
operation is claimed.
After source archival, a forbidden-provider fresh server restores exact generation
and Revision and accepts the completed browser retry with zero provider/science.
There are zero application JavaScript errors. Three internal Firefox telemetry/
login-manager notices are retained separately in each phase's browser record;
browser/server processes are stopped.
This native fixture is explicitly synthetic-only, separate from the production
resource witness. No API-only or live-provider acceptance is claimed.

## L. Operational resource readiness

Software handoff and genuine fixture execution are qualified. The current lab
operator configuration still has no raw InputSets, and is not modified here.
Real use requires the operator to configure a raw-family InputSet containing the
exact approved parent/reference and downstream QC path-and-digest pairs. The
existing qualified catalog/runtime environment is provisioned separately: for
this fragments chain, `AGENT_QC_RESOURCE_CATALOG`, `AGENT_QC_BEDTOOLS` and
`AGENT_MATRIX_BEDTOOLS`. These are operator settings, not arbitrary InputSet or
Planner arguments. External fragments also need explicit source profile and
namespace, and explicit supported selection thresholds where selection is requested.

BAM additionally requires exact accepted intake and corrected-barcode/library
context for the actual registered source and any supplied index. FASTQ requires
the completed exact library membership/layout and source-bound library/barcode/
whitelist/reference/index/runtime qualification. An old context is not reusable
merely because a new upload has matching bytes/species. Existing owner APIs must
establish that history; the conversation cannot manufacture it. Operator-approved
choice availability and owner qualification remain distinct.

## M. Remaining work

UA3.6.3c should address only actual shared declaration/clarification gaps:
explicit role-clarification continuation and supported categorical/list/structured
scientific declarations where existing owner/Registry contracts require them.
Unavailable resources and insufficient source history remain provisioning or
qualification blockers, not guessed clarification answers.

The four later scenarios still require: source-specific approved raw resources;
qualified canonical groups/context and MAESTRO tissue/marker resources;
scientifically compatible EpiZoo inputs and approved resource provenance; and
adaptation's explicit strategy/target/source/SEAM qualification. Live semantic
accuracy and integrated researcher acceptance remain later qualification.
No new species, MAESTRO resources, adaptation science, automatic reference
finder, general processed authority or additional download type is added.

Implementation, representative acceptance, affected domain regression and
preservation checks are complete within this bounded scope. Changes are unstaged/uncommitted;
no next-stage work, staging, commit or push is performed.
