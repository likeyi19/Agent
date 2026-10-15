# USI-3B — Practical scientific file and reference handoff

2026-10-15 (Asia/Shanghai). **PARTIAL at regression closeout.** All three requested
functional workflows are implemented and scientifically exercised; one
non-reproducing application-planning interruption remains unexplained. No commit or push is authorized or
performed by this work unit.

## A. Baseline and preservation

Branch `main`, HEAD, local `origin/main` and independently queried remote
`refs/heads/main` all initially resolved to
`0c01fd27deec43103305ffd52ce668d695f788a5`. USI-3A was present and committed.
Tracked files and the index were clean; `evals/` was already untracked.
README, AGENTS, the USI-2/USI-3A records and relevant owner/handoff contracts were
reviewed. This work does not reopen their acceptance or change their conclusions.

Evidence root **R**: `/home/likeyi/agent-usi3b-dd7o09u8/`, outside Git and
historical acceptance directories. `baseline.json` records initially tracked
file hashes, the existing eval-file size/mtime inventory and the previously
recorded operator inventory. Final preservation checks compare those inventories
and unchanged tracked files. These are bounded source hashes and protected-file
metadata checks, not a claim to rehash all historical biological data.

Final preservation passes: all **1,274** pre-existing eval files retain size/mtime,
with no additions/removals in that inventory; the operator configuration retains
its size/mtime and SHA-256
`f59312491c2719ae93bc3366724335550a713e5c76cf6bd84ddbc01a0558477e`.
All initially tracked files outside the declared changes retain their original
hashes. HEAD is unchanged and the index is empty. `preservation-final.json` records
all 18 modified tracked files and three new files listed below.

All new scientific inputs and outputs are explicitly simulated, isolated fixtures
under R. There are no public scientific downloads, configuration activations,
new dependencies, checkpoint loading or GPU execution. Upload limits, production
defaults and historical scientific artifacts are unchanged. Free space was checked
(approximately 15 TiB available); fixture resource use is bounded.

## B. Architectural audit and changes at existing owners

| Existing boundary | Finding and resolution |
| --- | --- |
| Trusted uploads and `raw_handoff` | Existing candidate capture, exact append, declarations, resource choice and durable prerequisites cover the task. Add a fragments declaration variant here; do not introduce another workflow or resource catalog. |
| External-fragment owner | Already reads independent plain/gzip/BGZF sources, optional explicitly supplied TBI, and canonical or regulatory-feature references. It independently verifies exact source conservation. Public execution/recovery unnecessarily forced canonical-reference loading; use its existing closed manifest discriminator and admit the existing neutral species result representation. |
| Generic region reference owner | Already qualifies FASTA/FAI and BED, including ordered interval identities, bounds and duplicates. Application code binds trusted registrations and calls it; it does not parse BED. |
| Reference supplementation | Extract the existing H5AD htslib FAI preparation into one shared helper. Reuse it for explicit plain FASTA plus BED. Existing supplied-FAI validation and H5AD behavior remain covered. |
| Fragment-feature matrix binding | Previously rejected canonical external fragments and canonical QC solely because their references used the older contract. Accept these exact qualified sources when species, assembly, genome identities and selection/QC lineage agree. Counting and independent reconstruction are unchanged. |
| Planner metadata | Generic matrix fragments were request-only, and the tool was visible only under species adaptation. Admit explicit v2 upstream edges and expose the same registered tool under raw preprocessing. No inferred steps, fanout, plan completion or new tool is added. |
| Application consumption guard | Check that either matrix tool consumes the exact captured target reference. An explicitly selected custom BED cannot be replaced by configured cCREs. |
| Session, authority, reports and delivery | Reuse existing owners and persistence. The new prerequisite discriminator is part of the existing Interaction contract. No new Session, authority schema or execution engine exists. |

The registry still contains 23 tools. EpiZoo, MAESTRO, fragment parsing,
conservation, QC, selection thresholds, sparse counting and independent numerical
verification are not reimplemented. Public external import gains an explicit
reference-contract extension; old string species values, canonical manifests,
producer profiles, receipt identities and recovery policies retain their meaning.
The data-layer neutral import entry points remain bounded readers/wrappers for
accepted identities and explicit primary preparation.

## C. Conversational workflows and continuation

The interpreter selects only opaque captured attachment handles. Its fragments
variant declares source, optional source index, single-library namespace,
human/mouse species, hg38/mm10 assembly and the supported external source profile.
The latter declares adjusted zero-based half-open ATAC/ARC intervals, corrected
barcode identifiers and read-pair support. File names, columns and barcode strings
do not establish this processing history. Missing declarations remain questions.

Independent fragments need no FASTQ, BAM, previous Agent Run, library-context
receipt or researcher-authored InputSet. Existing compatible reference choices
are resolved through USI-3A matching. Explicit or captured choices are retained;
ambiguity and invalid pins never trigger first-match selection or silent fallback.

For custom peaks or regions, the interpreter also binds the BED and its declared
category (`peak_set` or `regulatory_regions`) in the same assembly and coordinate
convention. Agent derives a private regulatory-feature reference against the
selected exact genome. The source import keeps its source reference; the matrix
receives the exact derived target reference. Arbitrary peaks are never relabelled
as curated cCREs or made eligible for EpiZoo's fixed vocabulary.

Two genuine supplementation paths are exercised:

1. **Missing BED:** preserve the original source, objective, thresholds and chosen
   reference; after restart, append the BED, qualify it and execute the original
   import → QC → selection → custom-region matrix request.
2. **Missing genome:** with no configured reference, capture fragments and BED,
   request the declared species/assembly's plain FASTA, then accept it through the
   ordinary conversation upload. Prepare a managed FAI if absent, or validate the
   explicitly supplied FAI. The existing reference owner qualifies the private
   reference, and the original external-import operation executes and verifies.
   This witness requests import only; it does not fabricate TSS resources or QC.

The second path builds the already-supported regulatory-feature reference from
explicit source files. It is not a general genome-reference builder, canonical
FASTQ/BAM bundle generator, downloader or annotation qualification service.
A selected/configured reference cannot be silently replaced by an uploaded genome;
when both are supplied, their exact genome identities must agree. Private
resources are not promoted into the operator catalog.

All paths preserve source order and identity, original task text and exact numeric
parameters. Only explicitly bound candidates are consumed; unrelated notes stay
staged. Original fragments, BED and FASTA bytes remain at their uploaded locations.
Task-local format registration shares those bytes without a renamed file copy.
New publication consists of existing owner manifests and, when needed, an FAI.

## D. Qualification, authority and privacy

The reference owner checks BED3-or-more input, coordinate bounds, exact contig
names, unique interval identities and caller order. Features use the existing
`chrom:start-end` identity convention; arbitrary extra BED columns do not become
new feature identities. The owner also checks streaming FASTA/FAI consistency.
When reusing a genome, FASTA hash, FAI hash, ordered contig hash and contig count
must match the selected reference exactly.

The matrix owner requires exact species/taxonomy and assembly, identical genome
identities, source-qualified fragments and exact selected-cell/QC lineage. Mapping
canonical `human`/`mouse` contracts to their defined taxonomy identities is an
explicit contract mapping, not biological inference from a file. Existing
independent fragment conservation and matrix reconstruction remain mandatory.

The private reference receipt is task binding, not scientific authority. Normal
registered execution issues schema-2 authority for the accepted fragments, QC,
selection and matrix. Existing evidence/report owners expose the actual feature
category, provenance, ordered feature digest and upstream identities. Downloads
use the normal pinned delivery path.

Source registration, optional indexes and supplemental files retain exact hashes
and Session scope. Foreign attachments, unknown handles, out-of-range roles,
changed sources and incompatible reference identities fail closed. Historical
replay validates captured records without reopening scientific resources or
constructing a provider. Completed reads also pass after source and BED removal.

External import conserves the supplied source; it does not assert primary-nuclear
filtering, alignment history reconstruction, barcode correction or cell purity.
The existing explicit primary-scope preparation remains the owner for claims of
that filtering. A BED or uploaded FASTA never supplies contig-classification proof.

## E. Tests and acceptance evidence

All provider decisions are scripted. Upload, application admission, Planner,
compiler, runtime, scientific production, independent verification, reports and
HTTP delivery execute normally. No live-provider quality or native-browser
acceptance is claimed.

The main synthetic fixture imports three records with total support six. With
explicit test thresholds of zero fragment records and zero defined TSS enrichment,
the accepted selection contains barcode A; undefined TSS for barcode B is not
silently treated as zero. The custom BED has three features, replacing the
configured two-feature cCRE vocabulary. The produced int64 sparse matrix is
**1 × 3**, ordered exactly as:

| Feature | Count for A |
| --- | ---: |
| `chr2:3000-3010` | 1 |
| `chr2:1000-1001` | 1 |
| `chr1:500-600` | 0 |

The count remains one for the record with support two: measurements count
canonical fragment records, not support weights. The chosen BED path/hash,
category, feature order, fragment and selection identities are asserted. Each
of the four production and independent owner bodies executes once. The generated
report and all five scientific downloads succeed; every scientific download
matches its advertised SHA-256. Retries/restart perform no new science or provider
construction. A substituted canonical target reference is rejected by the
application's acceptance guard.

Additional positives cover BGZF with and without TBI, and actual no-catalog
FASTA supplementation with generated and supplied FAI. The supplied-FAI case
forbids index generation. Both genome cases yield independently verified external
adoption, schema-2 authority and a downloadable report, with no shared resource
promotion. The neutral data-layer fixture also verifies public import and durable
recovery for its existing qualified reference contract.

Negatives cover invalid fragment support, out-of-bounds/duplicate/unknown-contig
BED, malformed FAI, compressed FASTA outside the supported plain-FASTA preparation,
changed source/reference pins, foreign supplements, ambiguous or missing source
roles/history, conflicting declarations, arbitrary paths and custom-reference
substitution. Existing affected suites cover broader upload privacy, append
identity, ordinary parameter continuation, FASTQ/BAM and resource reuse.

Development results are retained, including failures:

- `dev1.log`: 2 failures identified the missing generic-matrix capability visibility.
- `dev2.log`: both complete scientific executions succeeded; 2 test assertions
  incorrectly expected a numeric threshold as a string.
- `dev3.log`: 16 passed, 4 failed. Negative-test assumptions about terminal turn
  records and foreign-resource HTTP status were corrected; owner qualification
  failures now use the existing invalid-reference response.
- `contracts.log`: 11 declaration checks passed.
- `focused.log`: 214 passed, 1 failed in 259.32 s. The indexed-import script
  accidentally assigned its index as a BED before clearing that role; corrected.
- `genome-dev.log`: 4 selected BGZF/genome cases passed in 17.92 s; 20 tests were
  intentionally deselected for this development check only.
- `final-focused.log`: **58 passed**, 2 dependency warnings in **118.80 s**,
  covering final handoff tests, declarations, neutral boundaries and existing
  reference supplementation before the final declaration-conflict refinement.
- `qualified-fastq.log`: **1 passed in 21.85 s** with real pinned Chromap and
  simulated input; all **539 frozen files** unchanged.

The affected-domain closeout selects 39 test paths, including complete external
fragment, fragment-feature owner/integration domains, relevant generic QC and
reference checks, application admission/continuation/privacy, web handoffs,
FASTQ/BAM, resource reuse and semantic compiler/registry/scope tests. No cases
are excluded within those paths. All `RUN_*` gates are unset in that run; the
qualified FASTQ positive is the separate explicitly enabled test above.

**Closeout result: 1,305 passed, 1 skipped, 4 failed, 2 warnings in 844.93 s**,
exit 1. All **539 frozen files** are unchanged. The skipped qualified FASTQ case
passes in the separate enabled run above. The suite is not claimed green.
Exact commands/environment, JUnit XML, logs and frozen-file results are under R (`scoped-command.json`, `scoped.xml`,
`scoped.log`, `scoped-result.json`, `frozen.json`).

The two previously recorded unrelated failures were independently reproduced
against an archive of this unit's exact starting HEAD: **2 failed in 1.95 s**
(`baseline-failures-command.json`, `baseline-failures.log`). They are the unknown
singleton-resource 404-versus-422 assertion and the identity expectation that
includes absent `fastq_whitelist=None`. Neither is repaired.

The expanded owner regression also exposes the old complete-catalog size limit:
`test_v4_complete_catalog_size_and_explicit_null_index` already fails at baseline,
**35,656 > 34,000 bytes**, independently reproduced in **0.63 s**
(`baseline-catalog-command.json`, `baseline-catalog.log`). This is distinct from
actual scoped production planning and is not changed merely to make counts green.

The fourth failure is **not classified as a baseline failure**:
`test_registered_fragments_preserve_scientific_failures[reference-REFERENCE_DIGEST_MISMATCH]`.
The first aggregate run stored an application error while the Run remained in
planning, before any scientific step. It did not produce or accept scientific
output. Its exception traceback was not retained by that run's normal sanitized
presentation boundary, so the cause is not established.

Follow-up observations, without a speculative production repair:

- The exact test passes alone on current code (**1.25 s**) and on the archived
  baseline (**2.61 s**); see `diagnostic.log` and `baseline-reference.log`.
- Its entire integration module passes **40 tests in 74.54 s** with a diagnostic
  hook that prints exceptions and returns the unchanged existing error.
- A diagnostic replay of the exact preceding 12 test paths through that module
  completes **719 passed, 1 failed in 389.48 s**. The sole failure is the already
  reproduced catalog assertion, now **35,712 > 34,000 bytes**. All **539 final
  frozen files** remain unchanged (`order-command.json`, `order.xml`, `order.log`,
  `order-result.json`, `final-frozen.json`). This is a targeted order diagnostic
  for the unresolved failure, not a second full closeout regression.

Before that last replay, one test-only witness refinement makes the missing-BED
initial utterance explicitly say the BED will arrive later and identifies notes
as unrelated. It now agrees exactly with its captured roles and actual uploads.
The replay covers these final witnesses. All production files are identical to
the frozen broad regression and qualified FASTQ run. The original aggregate
failure remains evidence; later passes do not erase it or establish its cause.


Relevant command forms, all from the repository unless the recorded `cwd` says
otherwise:

- `git status --short --branch`, `git rev-parse HEAD`,
  `git ls-remote origin refs/heads/main`, `git diff --check`, scoped diff review.
- `/home/likeyi/anaconda3/envs/agent/bin/python -B -m pytest -q ...` with
  `PYTHONPATH=src`, `PYTHONDONTWRITEBYTECODE=1` and isolated `--basetemp=R/...`.
  Named development logs correspond to the modules described above.
- `python -B R/closeout.py` records exact final argv/environment. Its FASTQ call
  enables only `RUN_CHROMAP_QUALIFICATION=1` and uses the unchanged
  `/tmp/m116c-test-build-identity.json`; its scoped call disables all `RUN_*` gates.
- `git archive HEAD | tar -x -C R/baseline-source` plus the recorded baseline
  pytest commands reproduce unrelated assertions without modifying the worktree.
- `python -B R/recheck_order.py` records/runs the targeted order diagnostic;
  `debug_plugin.py` prints otherwise sanitized exceptions without changing outcomes.
- `python -B R/preservation.py` verifies protected inventories, unrelated source
  hashes, HEAD and index state. All final commands use the recorded interpreter.

## F. Assessment, limits and complete change inventory

**Overall assessment: PARTIAL at regression closeout; requested functional
implementation and bounded scientific witnesses are complete.** Independent
fragments, actual custom-BED matrix construction, missing-BED continuation and
missing-FASTA/optional-FAI continuation all execute through existing owners.
Three aggregate failures are independently proven baseline assertions. The fourth
planning interruption does not reproduce alone, in its full module, or after its
original test-order prefix; its cause remains unknown. It is not relabelled a
baseline issue or silently removed from the results. No demonstrated production
fix is justified by the available evidence.

**USI-3C has no demonstrated prerequisite architecture gap at this handoff
boundary.** Defaults and scale work can build on these interfaces. This report
does not assert a clean regression closeout; investigating the unexplained
aggregate planning interruption remains the acceptance limitation.

The demonstrated scientific scope is small simulated external fragments and
regions, principally human/hg38 chat fixtures, with the existing neutral owner
fixtures retained. Typed chat declarations remain bounded to human/hg38 and
mouse/mm10. This is not live-provider qualification, biological qualification,
large-genome performance, calibrated QC thresholds, statistical cell calling,
arbitrary-species raw chat, or production deployment readiness.

FASTA plus BED supplies a qualified genome/region reference, not independently
qualified transcripts/TSS. A requested QC path still needs its existing compatible
qualified QC resource. No generic annotation parser or TSS construction is added.
Missing or incompatible configured resources preserve a question; private upload
qualification is not an implicit correction of an already captured resource.
Canonical FASTQ/BAM provisioning and alignment-index preparation retain their
existing boundaries. No USI-3C/default/scale work is implemented here.

Changed files:

- `src/agent/application/raw_handoff.py`: fragments declarations, shared resource
  matching, exact optional roles, private reference preparation and pure replay.
- `src/agent/application/turn_decisions.py`: typed request/answer schema and
  interpreter instructions for independent fragments, BED and genome supplements.
- `src/agent/application/turns.py`: existing prerequisite routing and safe offers.
- `src/agent/application/dialogue_execution.py`: existing continuation capture and
  replay routing for the fragments discriminator.
- `src/agent/application/session_state.py`: admit that discriminator in the existing
  prerequisite record.
- `src/agent/application/responses.py`: render its preserved-task questions.
- `src/agent/application/interactive.py`: display exactly bound versus staged files.
- `src/agent/application/local_resources.py`: source/index byte-sharing registration,
  generic matrix companion inputs and exact target-reference consumption checks.
- `src/agent/application/reference_supplementation.py`: shared existing FAI helper
  and owner-qualified private BED/genome reference coordination.
- `src/agent/orchestration/external_fragments_registry.py`: admit existing neutral
  reference results with explicit validated species representation.
- `src/agent/orchestration/fragment_feature_registry.py`: explicit v2 source edges
  and visibility under raw preprocessing.
- `src/agent/orchestration/planning_scope.py`: describe that existing capability.
- `src/agent/tools/data/scatac_fragment_import.py`: closed reference dispatch for
  execution/recovery and the existing neutral result representation.
- `src/agent/tools/data/_fragment_feature_binding.py`: compatible canonical external
  source/QC admission with exact genome and lineage requirements retained.
- `src/agent/tools/data/scatac_reference.py`: explicit taxonomy identity mapping
  for the two existing canonical species contracts.
- `src/agent/tools/data/neutral_fragment_import.py`: update the boundary description;
  retain its historical and primary-preparation entry points.
- `src/agent/web/static/app.js`: recognize a pending fragments-role answer in the
  existing attachment composer.
- `tests/application/test_fragments_declaration_contract.py`: bounded declarations,
  missing information, immutable roles and conflicting-argument rejection.
- `tests/web/test_fragments_chat_handoff.py`: actual scientific HTTP acceptance,
  supplementary files, privacy, corruption, restart, provenance and delivery.
- `tests/fragment_features/test_neutral_fragment_boundary.py`: replace the retired
  blanket public-neutral rejection with exact public import/recovery acceptance.
- `docs/usi-3b-practical-file-reference-handoff.md`: this record.

## G. Bounded stability follow-up — 2026-10-15

**PARTIAL — Functional implementation complete; stability observation unresolved.**
This follow-up preserves the original regression and acceptance results above.
New evidence **S**: `/home/likeyi/agent-usi3b-stability-a28_nhoc/`.

### A. Investigation

Reviewed R's `scoped.log`, `scoped.xml`, `scoped-result.json`, `diagnostic.log`,
`order.log`, `order.xml`, `order-result.json`, diagnostic hook and recorded commands,
plus the original affected Run/Session JSON. The reference-digest case failed in
0.527 s with `INTERACTIVE_APPLICATION_FAILED`. Its persisted Run is revision 2,
`PLANNING`, with no plan, steps or errors; the Session has generation zero and no
scientific Revisions. The last persisted diagnostic is `PLANNING_SCOPE_ACCEPTED`.
No detailed-provider-start diagnostic is persisted. This locates the last durable
checkpoint, **not necessarily the exact exception site**. The original traceback
is unavailable; neither the XML nor the stored presentation recovers it.

Inspected the exact fixture and scripted provider, admission wrapper, synchronous
submission/error boundary, scope/detailed planner, recovery diagnostic callbacks,
RunStore updates and trace validation. Ordinary planner exceptions are normally
converted into a persisted terminal failure by `AgentRuntime._run_durable`.
The surviving planning record alone therefore cannot distinguish an initial
planning exception from an error while recording diagnostics or terminal state.
There is no original exception or system-event evidence establishing an I/O,
clock, resource-exhaustion or other environmental cause.

The test calls `submit_turn` synchronously; no HTTP executor or asynchronous worker
is on this path. Its model, call list, source-factory counter and temporary workspace
are created per test. Relevant patches use pytest monkeypatch cleanup; active
planning context resets its ContextVar token in `finally`. No concrete leaking
fixture or shared-state mutation was identified. The test, source fixture,
runtime, RunStore, recovery coordinator, LLM planner, semantic prompt/schema,
active-context module and persisted-state schema are byte-identical to starting
HEAD (`S/path-baseline-comparison.json`). Changed USI-3B metadata still participates
in planning, so this comparison does **not** classify the anomaly as pre-existing.

One additional observation run selected the immediately preceding navigation/branch
case, invalid-record case and affected invalid-reference case, in their original
relative order. Temporary `S/stability_probe.py` observes provider phases, context
cleanup and state updates, and would capture runtime or application exceptions
before sanitization. It forwards original calls, results and exceptions unchanged;
it records no provider prompt or scientific file contents. No faults were injected.

### B. Root-cause assessment

**Cause cannot be established with available evidence (category 5).** Neither a
USI-3B regression, a pre-existing defect, fixture contamination nor a transient
environment condition is demonstrated. Earlier isolated/current/baseline, module
and order-prefix passes remain observations, not proof of a fix. The additional
probe also passed, with no underlying traceback obtained. Its 62 state updates
succeeded, all observed calls ran on `MainThread`, persisted trace timestamps were
ordered, and planning context was empty before and after each test. These facts
apply to this probe; they cannot reconstruct the original execution environment.

### C. Changes

Only this documentation section is added in the repository. No production or test
code, expectations, scientific contracts, error sanitization or retries change.
The temporary hook and diagnostic artifacts stay outside Git under S. No new
framework, dependency or abstraction is introduced. No specific defect justifies
a repair; further speculative repetitions are stopped.

### D. Verification and preservation

From `/home/likeyi/program/Agent`, with all `RUN_*` variables unset:

```bash
PYTHONPATH=src:/home/likeyi/agent-usi3b-stability-a28_nhoc PYTHONDONTWRITEBYTECODE=1 \
/home/likeyi/anaconda3/envs/agent/bin/python -B -m pytest -q -p stability_probe \
  tests/application/test_registered_resource_integration.py::test_registered_fragments_existing_session_navigation_and_branch \
  'tests/application/test_registered_resource_integration.py::test_registered_fragments_preserve_scientific_failures[record-EXTERNAL_FRAGMENTS_RECORD_INVALID]' \
  'tests/application/test_registered_resource_integration.py::test_registered_fragments_preserve_scientific_failures[reference-REFERENCE_DIGEST_MISMATCH]' \
  --basetemp=/home/likeyi/agent-usi3b-stability-a28_nhoc/pytest-temp \
  --junitxml=/home/likeyi/agent-usi3b-stability-a28_nhoc/probe.xml \
  -o cache_dir=/home/likeyi/agent-usi3b-stability-a28_nhoc/pytest-cache
```

**3 passed in 11.49 s**, exit 0 (12.30 s process wall time). The unchanged reference
case asserts the expected `REFERENCE_DIGEST_MISMATCH`, zero production and independent
reconstruction calls, no accepted Revision/authority, and stable retry/reopen.
Exact argv, environment and dependency versions are in `S/probe-command.json`;
results, XML and observations are in `probe-result.json`, `probe.log`, `probe.xml`
and `events.jsonl`. Python 3.11.16, pytest 8.4.2; `PYTEST_ADDOPTS` and
`PYTHONHASHSEED` were unset. No full regression, module repetition, baseline rerun
or additional scientific qualification was performed in this follow-up.

The original **1,305 passed / 1 skipped / 4 failed** result remains unchanged.
Its three independently reproduced baseline failures (catalog size, absent
whitelist identity and singleton-resource HTTP status) remain outside this unit.
Existing functional acceptance is retained without expanding its synthetic scope.

`S/preservation.json` compares entry/exit source hashes and protected inventories:
all existing repository files except this report retain their hashes; the original
report is an exact prefix of the appended version. All 1,274 eval files and R's
existing evidence retain their size/mtime inventories. Operator configuration
retains its recorded SHA-256. HEAD and the empty index remain unchanged; the set
of modified/untracked repository paths is unchanged. No historical scientific
resources, checkpoints or Session/Revision stores were written. These are bounded
hash/metadata checks, not a full rehash of historical biological resources.

### E. Final disposition

**PARTIAL — Functional implementation complete; stability observation unresolved.**
The observed interrupted planning turn remains a residual execution-stability risk;
its incidence and cause are unknown. There is no evidence of scientific execution
or accepted-result corruption in that event, and no demonstrated defect requiring
BLOCKED status. A later pass is not described as a fix.

**USI-3C may proceed without additional prerequisite architecture work.** No such
gap has been demonstrated. Preserve this observation for any future recurrence;
an original exception captured at recurrence would supply the missing diagnostic
evidence. Nothing is staged, committed or pushed.
