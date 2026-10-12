# UA3.6.2c — Unified scientific result presentation and trusted delivery

Implemented on 2026-10-12 against
`6e651b80d72f409a723e44a9896b44adcfd5a0e2`, the committed UA3.6.2b baseline.
This stage adds readable accepted-result presentation and one qualified processed
download role through the existing scientific-file delivery boundary. It does
not qualify every registered scientific output for download or historical reuse.

## A. Baseline and preservation

Admission found HEAD, local `main`, existing `origin/main`, and remote `main`
equal to the exact commit above. Remote inspection used `git ls-remote origin
refs/heads/main`; no fetch was needed. The tracked worktree and index were clean,
only `?? evals/` was untracked, and the UA3.6.2b document was present. No reset,
clean, fetch, pull, staging, commit or push is performed.

`/tmp/agent-ua362c-preservation-wpl8zuyr/protected-before.json` pins **4,371
protected files**: 1,274 evaluation records, 198 outputs, 208 real application
state files, one operator configuration, 23 UA3.5.5 scientific-state files,
seven preparatory files, two human inputs, four EpiZoo auxiliary files,
1,164 UA3.6.2a acceptance files, 1,489 UA3.6.2b acceptance files and one
checkpoint. Ordinary files use SHA-256, size, modification time and mode.
The 5,231,645,507-byte checkpoint uses metadata only; it is never loaded or
rehashed. The record also pins all 609 tracked baseline files, exact index bytes,
and all three local refs.

All new scientific fixtures, Runs, Sessions, reports, evidence, caches and
acceptance logs use isolated `/tmp` workspaces. Snap Firefox requires a new
browser profile/download directory under `/home/likeyi`; that directory contains
only new browser state and delivered test files. No existing browser profile or
scientific state is used.

The final comparison after all validation finds **zero mismatches across all
4,371 protected records**. All 1,274 evaluation files and their inventory remain
unchanged, as does the complete 198-file output inventory. Index bytes and all
three local refs are unchanged; remote `main` still matches the admitted commit.
Of the 609 tracked baseline files, 600 remain byte-identical; the nine intentional
changes comprise eight existing production files and one Web test double. The
new owner delivery module, four new test modules and this document are untracked.
`preservation-final.json` records the complete comparison.

`implementation-final-freeze.json` pins all 512 current source/test/benchmark
files, including new implementation and tests. The final focused, browser and
domain checks leave every frozen file and the complete inventory unchanged.
The external EpiZoo checkout remains clean at its admitted
`029cd631d0f9806a646c4a3b42ce10b958f2b67f` commit. The checkpoint remains checked
only by metadata. `completion-validation.json` records the final inventory,
freeze, preservation and completed validation results.

## B. Existing shared boundaries and changes

The authoritative path remains:

```text
ordinary conversational execution → accepted Run/step → SessionTurn/Revision
  → existing evidence/report/PNG projection → interactive facade → static Web
  → reviewed scientific-file descriptor → shared bounded snapshot → HTTP stream
```

Session/Revision records, typed decisions, Planner, Registry, compiler/preflight,
scientific authority and the UA3.6.2b context/retention mechanism are unchanged.
The directly reused contracts are [accepted matrix/table delivery](ua3.3.2-accepted-matrix-delivery.md),
[human EpiZoo acceptance](ua3.5.5-human-epizoo-user-workflow.md),
[qualified graph reuse](ua3.6.2a-generalized-processed-output-reuse.md) and
[ordinary conversational continuation](ua3.6.2b-unified-conversational-state-planning.md).

| Production file | Change |
| --- | --- |
| `application/interactive.py` | Select the qualified processed export source using exact accepted identity and managed owner metadata; add existing reviewed report labels to safe evidence. |
| `application/interactive_schemas.py` | One closed scientific artifact descriptor and immutable, bounded evidence field labels. |
| `application/matrix_delivery.py` | Dispatch the owner export after the same source copy/hash; reserve both temporary files under the existing slot/disk limits; keep source/content digests distinct. |
| `tools/analysis/neighbors_delivery.py` | Closed owner format/privacy inspection and fresh logical HDF5 copying for the explicit export role. |
| `tools/analysis/neighbors_publication.py` | Extract the existing bounded JSON/manifest parsers for secure descriptor-based metadata reads; file APIs retain their behavior. |
| `web/app.py` | Use the prepared original-source digest in its existing source header. |
| `web/static/app.js`, `index.html`, `styles.css` | Readable statuses, recorded operations, originating turn, shared structured evidence, historical-use/retention labels and the reviewed download control. |

New tests are `tests/tools/analysis/test_neighbors_delivery.py`,
`tests/application/test_processed_delivery_stream.py`,
`tests/application/test_result_presentation_metadata.py` and
`tests/web/test_unified_scientific_result_delivery.py`. The existing Web snapshot
test double adds its explicit source digest to match the prepared-resource
contract. There is no second exporter service, snapshot engine, artifact store,
per-tool renderer or download controller. Registry remains 23 tools and 172
public arguments. EpiZoo and all numerical scientific production are unchanged.

## C. Scientific result presentation

The Web presents the existing Revision status and persisted `StepView` operations,
selected outputs and explicitly retained outputs. Viewing an earlier result uses
its exact Revision URL and does not activate it. Full identities remain in
technical details; status and originating turn are readable outside those details.
The existing “View result” action opens evidence and artifact details.

Evidence remains the existing `DialogueEvidence` projection. The facade reuses
reviewed report field labels; the frontend displays available values as bounded
structured facts rather than a raw JSON block. No statistics or biological
conclusions are calculated in JavaScript. Existing whole-field omission of paths
and private configuration remains in force. Labels are immutable, limited to
offered fields and covered by the existing client byte limit.

Effective resolution/random seed, execution-argument presence, disabled/default
selection thresholds and utterance-bound user declarations come from the
existing evidence owner. `explicit_execution_argument` is explicitly distinguished
from evidence of a user instruction. `existing_owner_default` comes from the
accepted operation projection, never from comparing against today's defaults.
Unsupported/missing origins retain their existing unavailable status.

The exact `EvidenceSource.prior_outputs` records explain historical input
consumption. Retained outputs are labeled separately. Accepted results without
recorded reusable authority receive an explicit boundary note; reporting success
does not imply historical execution eligibility. Existing software versions and
safe resource facts remain readable, with their accepted provenance available in
details. No new LLM summary or model-facing contract is introduced.

Accepted report projections and pinned PNGs use the existing endpoints and checks.
Unavailable figures receive a separate availability note, not a scientific failure
label. Reports remain safe projections rather than disclosure of raw private
Markdown. The existing conversation remains the primary interface.

## D. Qualified processed role and ownership

The single new role is `neighbors_export`, source contract
`cell-neighbors-publication.v1`, client artifact type `scientific_analysis`,
filename **`neighbors.h5ad`**, label **“Cell neighbors — safe export”**.
The derivative declares `cell-neighbors-export.v1` in
`uns/agent_scientific_export`. This is download eligibility, not new schema-2
scientific authority or another historical producer port.

The accepted original compact graph includes absolute `source_embedding_path`
and `source_cell_ids_path` under `uns/agent_milestone6`. Its original bytes remain
ineligible. The existing matrix H5AD privacy policy is unchanged and still rejects
this unrelated layout. No manifest, receipt, original embedding/ID source or
arbitrary scientific-directory file is exposed.

Agent selects only a terminal successful accepted producing step with qualified
neighbors port, exact execution/accepted-step identity, schema-2 original
scientific-correctness authority, owner profile/verifier, arguments, manifest,
receipt, proof metadata and complete three-file closure. Metadata is read through
the existing no-follow managed-path opener and the owner's bounded closed parsers.
The handle binds Session, Revision, originating turn, accepted step, authority,
manifest, original payload digest/size and export contract. It cannot select a
client path or adopt a standalone publication.

Current-request membership follows the already accepted matrix/table rule: the
exact successful run and explicitly planned accepted owner step may supply its
reviewed file even when a selected output is a summary. A retained historical
source additionally requires its exact selected qualified semantic member and
original run/step anchor. A `PriorOutputRef` consumed by the child alone does not
authorize a child download. A retained graph is eligible in child Revision details;
it is not presented as newly produced in that turn's chat attachment inventory.

The scientific owner controls the compact format and the deliberate derivative.
It checks a closed ordinary uncompressed HDF5 layout, exact owner metadata and
graph parameters, identifier privacy, the 512-column float32 representation layout
and axis/CSR storage shapes against accepted metadata. Exact ordered identity
comes from the accepted original hash and unchanged logical values. This is
format/privacy inspection, not reconstruction of the scientific order proof,
graph construction or initial source-alignment verification.

The export uses a **fresh file**, copying the inspected ordered cell identifiers,
embedding values, both sparse graphs, parameters and software versions. It omits
only the two exact recorded source-locator fields and adds explicit export
provenance: original graph SHA/size, manifest SHA, historical embedding/ID SHA/size
and ordered-cell digest. Fresh creation prevents deleted HDF5 free-space bytes
from carrying removed paths. Unknown metadata fails closed; it is not silently
redacted. Original scientific files and accepted authority are unchanged.

No original input file is opened. Embeddings in this fixture are explicit
scientific inputs; their identity is not an invented EpiZoo inference/checkpoint
claim. The derivative grants no new execution rights and is not a replacement
owner publication. Legacy graphs without accepted authority remain ineligible;
UA3.5.5 history is not upgraded. Clustering and UMAP retain accepted reporting
status without newly invented reusable authority or scientific-file eligibility.

## E. Shared snapshot, privacy and transport

The existing `MatrixDownloadPreparer` and `_ScientificFileResponse` remain the
only transport engine for matrix, QC/selection tables and the graph export.
Preparation uses the original accepted SHA/size, managed no-follow regular-file
opening, bounded 1 MiB copying, inode/time stability checks and an anonymous
private `TemporaryFile`. The owner receives that verified snapshot and a fresh
anonymous destination. It never writes a derivative into RunStore.

The existing maximum is 1 GiB per source/delivered file and two active transfers
per facade process. For the export, disk admission conservatively reserves
`source_size + min(source_size + 65,536, 1 GiB)` while both snapshots exist,
under the same transfer slot and 64 MiB free-space margin. Thus two processed
preparations reserve at most 4 GiB of payload; matrix/table reservations retain
their original sizes. Destination writes and truncation cannot exceed their
bound. Numeric copying is at most 1 MiB per read, strings use fixed-width bounded
chunks, and no complete scATAC matrix is densified.

Closed format checks reject unknown fields/attrs, unexpected private path forms
and sensitive metadata, unsafe identifiers, soft/external links, hard-link
aliases/cycles, virtual datasets, external raw storage, filters, custom fill
values, userblocks and malformed or unsupported structures. Only the two known
source locators may contain paths, and they must equal accepted owner metadata
before omission. The completed derivative receives the same closed export check
before headers or bytes are sent.

Source SHA/size in the opaque descriptor continue to identify the accepted
original. HTTP `Content-Length` and `X-Artifact-Content-SHA256` describe the
prepared derivative; `X-Artifact-Source-SHA256` identifies its original. Matrix
and table source/content digests remain equal. The frontend labels the export
explicitly. Browser anchors use the existing route and attachment streaming;
JavaScript does not buffer scientific H5AD bytes.

Every failure closes both temporary descriptors and releases its reservation.
Response cancellation before iteration, disconnect/send failure and interruption
retain existing explicit response-owned cleanup. Original replacement after
successful preparation cannot alter transmitted bytes. Limits return existing
safe 404/413/503 errors without scientific paths. These remain trusted lab-host
contracts: opaque handles do not create authenticated account ownership, and
capacity is per process rather than a host-wide quota.

## F. Genuine multi-turn and cross-family acceptance

The processed witness starts with the real registered neighbors owner over 32
ordered Unicode cell IDs and explicitly supplied seed-17 32 × 512 float32 inputs.
The ordinary Web/application request is:

> Change the clustering resolution to 0.5 and update the UMAP.

The unchanged UA3.6.2b interpreter/planning/context path compiles historical
neighbors → new Leiden → new UMAP and explicitly retains the graph. The new
child Revision activates; its immutable parent remains exact. Actual accepted
resolution is 0.5. Evidence shows the effective value, explicit execution origin
and exact utterance-bound user declaration. Existing accepted PNG and report
access remain available. The new clustering/UMAP results gain no reusable owner
authority. No foundation model/checkpoint or live provider is needed.

The second family uses genuine tiny synthetic-reference fragments → QC →
selection → matrix. Its ordinary depth-1 follow-up runs selection alone through
existing conversational execution. Current and historical evidence use the same
projection; all existing five file roles are downloaded through the same route.
The old matrix remains accessible from its originating Revision and is rejected
as an unselected child attachment. These permissive fixture thresholds are not
calibrated biological quality defaults or cell-calling claims.

API acceptance covers parent/child views, explicit retention, no-retention input
consumption, exact historical handles, refresh/reopen, a fresh facade with a
forbidden provider factory, generation-checked navigation and immutable scientific
bytes. Reads/downloads create no Revision or Session generation change. Navigation
uses its existing explicit state change and creates no scientific Revision.

Actual counters and native browser results are recorded in the final validation
section. Numerical equality checks cover every ordered ID, embedding element and
both sparse graph payloads, rather than checksum-only scientific claims.

## G. Fail-closed acceptance

New format, resolver and stream tests exercise missing/corrupt authority,
failed/incomplete source execution, wrong step/output identity, forged/stale or
foreign Session/Revision handles, unretained historical inputs, modified source
or manifest bytes, unsafe symlinks, private identifiers/metadata, unsupported
HDF5 links/storage/layout, oversize and disk/slot exhaustion, mutation during
copying, source replacement after preparation and interrupted downloads.
The original privacy-unsafe neighbors H5AD remains rejected by original matrix
delivery. Existing matrix/table format and transport negatives are retained.

Export errors close both private files; all approved roles compete for the same
two slots. A demonstrated owner write-helper issue was fixed during review:
implicit `truncate()` after an excessive seek must enforce the same bound as
explicit truncation/write. This is format transport hardening, not scientific
algorithm repair.

Development also corrected fixture expectations: the existing user-origin value
is a mapping, delivery integrity need not call authority `file_closure`, a fully
consumed snapshot needs rewinding before an interruption test, and current-run
summary selections legitimately admit that run's reviewed owner step. The wrong
step/member negatives now use internally consistent Session selection records.
Superseded exploratory runs are not presented as final passing acceptance.

## H. Validation records

Every command uses conda `agent` Python,
`PYTHONPATH=src:.`, `-B`, disabled `RUN_*` gates, `-p no:cacheprovider`, isolated
basetemp and `/tmp` Numba/Matplotlib/XDG caches. No full 7,000-plus-test suite,
paid provider call, EpiZoo inference, adaptation training or real PBMC raw
preprocessing is run.

| Final selection | Result |
| --- | --- |
| New processed transport, client metadata, genuine Web acceptance, new owner format and existing owner publication | **100 passed**, three existing warnings, 92.89 s; exit 0. |
| Existing Web/static contracts, after the final originating-request display refinement | **24 passed**, 0.04 s; exit 0. |
| Full affected Application/Web domain | **2,314 passed**, nine existing skips, three existing warnings, 799.73 s; exit 0. |
| Processed/raw continuation, captured context, authority dispatch, prior planning and Registry | **174 passed**, three existing warnings, 420.52 s; exit 0. |

Selections overlap and must not be added as unique test totals. The new selection
contains 47 format/privacy, 21 existing owner, 10 processed stream, six presentation
metadata and 16 Web acceptance cases. The three warnings are the existing
`pkg_resources`, `louvain` and TBB/Numba dependency notices.
All nine skips are the existing explicit `RUN_CHROMAP_QUALIFICATION=1` gate
in `tests/chromap/conftest.py`: three registered FASTQ Application cases and six
Web FASTQ execution cases. No selected test is excluded.

The exact focused command is:

```bash
PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 \
/home/likeyi/anaconda3/envs/agent/bin/python -B -m pytest \
  tests/application/test_processed_delivery_stream.py \
  tests/application/test_result_presentation_metadata.py \
  tests/web/test_unified_scientific_result_delivery.py \
  tests/tools/analysis/test_neighbors_delivery.py \
  tests/tools/analysis/test_neighbors_publication.py \
  -q -p no:cacheprovider \
  --basetemp=/tmp/agent-ua362c-new-acceptance-final-1pgk9oh6/pytest-temp
```

All `RUN_*` variables are unset and cache paths are isolated by
`/tmp/agent-ua362c-run-selection.py`. The exact invocation and full stdout/stderr
are `/tmp/agent-ua362c-new-acceptance-final-1pgk9oh6/{command.txt,pytest.log}`;
`result.json` records exit status and runner elapsed time separately from pytest.
The final static command selects `tests/web/test_static.py`,
`tests/web/test_rich_static.py` and `tests/web/test_scientific_artifact_static.py`
with the same Python/environment/flags and
`--basetemp=/tmp/agent-ua362c-web-static-frozen`; its exact command is also recorded
in the browser summary.

The complete affected Application/Web command is:

```bash
PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 \
/home/likeyi/anaconda3/envs/agent/bin/python -B -m pytest \
  tests/application tests/web -q -p no:cacheprovider \
  --basetemp=/tmp/agent-ua362c-app-web-final-c4ezvjsc/pytest-temp
```

Its exact environment/command, full stdout/stderr and exit record are in
`/tmp/agent-ua362c-app-web-final-c4ezvjsc/{command.txt,pytest.log,result.json}`.
The authority/context regression starts only after that command exits 0:

```bash
PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 \
/home/likeyi/anaconda3/envs/agent/bin/python -B -m pytest \
  tests/authority_dag/test_processed_prior_outputs.py \
  tests/authority_dag/test_conversational_processed_continuation.py \
  tests/authority_dag/test_captured_planning_context.py \
  tests/authority_dag/test_conversational_raw_context.py \
  tests/unit/orchestration/test_scientific_authority_dispatch.py \
  tests/unit/orchestration/test_prior_output_planning.py \
  tests/unit/orchestration/test_registry.py \
  -q -p no:cacheprovider \
  --basetemp=/tmp/agent-ua362c-authority-final-o2z4vmdm/pytest-temp
```

Its corresponding records are in
`/tmp/agent-ua362c-authority-final-o2z4vmdm/{command.txt,pytest.log,result.json}`.
These stable final runs cover existing matrix/table delivery, Session/navigation,
UA3.6.1 declaration binding, UA3.6.2b ordinary continuation and raw/processed
authority boundaries. Both use the same isolated runner environment as the
focused command. `git diff --check` passes.

The directly emitted API scientific witness is
`/tmp/agent-ua362c-new-acceptance-final-1pgk9oh6/pytest-temp/test_ordinary_processed_result0/unified-result-delivery-witness.json`.
Its actual observed bodies are:

| Body | Original graph execution | Ordinary continuation | Viewing/download/restart/navigation reads |
| --- | ---: | ---: | ---: |
| Scanpy neighbors production / independent neighbors owner review | 1 / 1 | 0 / 0 | 0 / 0 |
| Leiden / UMAP production | 0 / 0 | 1 / 1 | 0 / 0 |
| Interpreter / Planner detailed body | Not instrumented in seed | 1 / 1 | 0 / 0 |
| Source capture / source check | 1 / 1 | 0 / 0 | 0 / 0 |
| Source embedding / ordered-ID loads | 2 / 3 | 0 / 0 | 0 / 0 |
| Compiler / preflight | 1 / 2 | 1 / 2 | 0 / 0 |
| Shared authority `file_closure` | 10 | 24 | 0 |
| Normal clustering / UMAP step verifier | 0 / 0 | 15 / 15 | 0 / 0 |
| Existing graph structure / analysis H5AD reads | 3 / 2 | 66 / 64 | 0 / 0 |
| Shared download preparation / owner export | 0 / 0 | 0 / 0 | 4 / 4 |
| Closed export layout inspection / safe managed source opening | 0 / 0 | 0 / 0 | 8 / 28 |

The continuation uses four scripted completions; reads add zero completions and
construct no provider. EpiZoo inference is zero throughout. Normal existing
verification counts include repeated acceptance/presentation checks during
execution; they are not repeated production. Download format inspection and
bounded copying are counted separately from scientific reconstruction.
The accepted graph source SHA is
`0abebb98cfb513a117674ac1c9e1188aafe74e1f82e7cecf6f0c421130684a52`;
the delivered export SHA is
`d9db1990502d9f41c94dcbc4ec1d6afe56d7abd4735dad6f5f7daf5d345354b3`.

The raw witness, `test_raw_results_use_same_evid0/unified-raw-result-delivery-witness.json`
in the same basetemp, records one original fragments/QC/selection/matrix producer
and independent owner invocation each (plus one generic fragments verification).
The follow-up increments only selection production and its independent owner
once each. All subsequent evidence, download and fresh-facade reads preserve those
counts and provider completion totals. Historical matrix and current/historical
table payloads retain their original byte identity.

### Native Firefox

**45 native workflow checks pass:** 13 processed execution/read checks, 12
processed fresh-server/history checks, 11 raw execution/read checks and nine raw
fresh-server/history checks. Repeated exploratory/browser probes are reported
separately rather than added to that total. These are actual native Firefox
composer submissions, DOM result/evidence viewing, pinned PNG display and browser
attachment downloads; API-only tests are not substituted for browser acceptance.

The browser workspace is `/tmp/agent-ua362c-native-firefox-xxxwvvg5` and the fresh
Snap-visible profile/download root is `/home/likeyi/agent-ua362c-firefox-iit15q5s`.
Launch is `/usr/bin/firefox --headless --no-remote --marionette
--remote-allow-system-access --profile
/home/likeyi/agent-ua362c-firefox-iit15q5s/profile`. The adapted native Marionette
helpers, isolated server script and browser driver remain in that workspace.
`browser-summary.json` contains authoritative phase results, actual counters,
common-body probes, safe headers and saved-byte comparisons.

Processed browser execution observes Leiden=1 and UMAP=1, with neighbors,
neighbors-owner review, source loads and EpiZoo inference zero; four scripted
completions are used. Raw browser continuation increments only selection and its
owner once, with three scripted completions. Fresh servers use forbidden provider
factories and create zero provider clients/completions or scientific Revisions.
Parent/child switching changes generation from 2 to 4 through the existing
navigation contract, while both immutable Revisions remain exact.

Final common-body probes explicitly monitor `turn_decisions.interpret` and the
Planner's shared `_plan_detailed` body, covering the normal recovery-capable
planning entry too. Both are zero, alongside provider, production, source loading
and owner reconstruction. The processed probe observes 11 safe source opens,
one shared preparation/export and two closed format inspections. The raw probe
adds 15 safe source opens, one preparation and one matrix format inspection.
All snapshots are closed and active transfer counts return to zero.

The browser saves the processed export as 121,136 bytes with SHA
`f5e3f0c08548659235095031fd56bc5c2b6c5a55862d2baae4d0e7268ef97943`;
its original source SHA is
`22ebcaf33541f9af73faedd27f8acbce774608e3fd7aba34a9d685cc44ded932`.
HTTP delivered digest, actual saved bytes and repeated original/child downloads
agree. Every ordered ID, float32 embedding value and sparse graph value is equal
to the accepted original; private source locators are absent. Direct existing
raw table/matrix browser downloads also match their accepted size/digest.

There are **zero application JavaScript errors**. Firefox's internal Telemetry
and LoginManager notices are retained and separately classified in the summary.
Readability and the actual accepted UMAP were visually checked in
`processed-parameter-viewport.png` and `processed-umap-viewport.png`.
Refresh/reopen, exact old links, fresh-server restoration and navigation pass.
The acceptance server and Firefox processes are stopped after qualification.
This qualifies bounded software/scientific-fixture behavior, not live language
quality, biological interpretation or universal export coverage.

## I. Remaining boundaries and next step

Embedding NPY/ordered-ID pairs, clustering/UMAP originals, annotated analysis
H5AD, pseudobulk/DA artifacts, evaluation JSON, BGZF/TBI companions and adapted
models remain outside newly qualified scientific-file delivery. Their existing
supported evidence/report/PNG behavior remains. Additional owner contracts and
format/privacy/companion review are required where those files are needed; this
stage does not automatically upgrade or export them.

There is no processed-role blocker within the accepted neighbors scope. Broader
scientific input and operator-resource availability remain the practical
integration work. The recommended next stage is **UA3.6.3 — Unified Scientific
Input and Resource Composition**, reusing Registry/Application role binding,
qualified resources, user declarations and ordinary clarification. No next-stage
implementation, staging, commit or push is performed here.
