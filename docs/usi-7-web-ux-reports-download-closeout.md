# USI-7 — Web UX, scientific reports & download closeout

2026-10-16 (Asia/Shanghai). **USI-7 COMPLETE** within the validated Web usability
and delivery scope below. UA3.6.5 live-provider scientific qualification remains
separate and has not begun.

## A. Baseline and preservation

Initial `main`, HEAD, local `origin/main`, and independently queried remote
`refs/heads/main` all identify
`0666b43490de783a24cc0c3068ac3db6fe494917`,
`Complete USI-6 standard analysis and cross-revision comparison`.
The tracked tree and index were clean; initial status was solely `?? evals/`.
The previous USI-5 checkpoint is `89385ffaffcb3d24ff17a0a0f5d367d07b570fa2`.
Nothing is staged, committed or pushed. No production configuration is edited.

Evidence root **R** is `/home/likeyi/agent-usi7-qhtee_1k`, outside Git and all
protected historical workspaces. `R/baseline.json` freezes tracked hashes and
92,862 protected file entries: the inherited USI-6 preservation scope plus the
USI-6 evidence tree. Every protected entry has size, nanosecond mtime and mode;
files at most 2 MiB also have SHA-256. This is a bounded preservation audit,
not a new full-content hash of every biological dataset or checkpoint.
`R/preservation.py`, `preservation.json` and `preservation-final.log` record the final check.

The browser reads the real accepted USI-6 Session `test` from
`/home/likeyi/agent-usi6-07lldlo1/final/workspace` through a separate GET-only
acceptance server. Provider and registered scientific execution functions raise
if called. No navigation writes are allowed there. New Sessions, uploads,
simulated tiny H5AD input and real inspection executions use only `R/fixture`.
Model-choice inspection copies the existing configured profile declarations into
an isolated workspace; it does not submit provider requests. All screenshot,
profile, download, test and server evidence is under R. All owned acceptance
servers and the isolated Firefox process were stopped after validation.

## B. Original usability problems

The audit read the actual HTML, CSS, JavaScript, HTTP routes, public projections,
upload owner/configuration, Web tests, README and USI-1–6/M18 contracts. The
baseline was already single-column and had working multi-file attachment and
safe-delivery foundations. It did not need another frontend framework.

Concrete shortcomings, also visible in `R/before-1366.png`:

- First use required creating a Session before typing or attaching. Empty-state
  copy emphasized administrative setup rather than the scientific task.
- Advanced inputs had both a button and a disclosure for the same controls.
  Configured resources repeated this pattern. Empty input-selection, upload and
  readiness notices competed with the composer even when nothing was selected.
- Conversation growth pushed the composer down the whole document. Polling
  rebuilt messages without preserving the reader's scroll position.
- Attachments repeated long qualification wording, lacked size display, and sat
  outside the associated user bubble after submission. Upload completion and
  server registration had no distinct end-of-transfer wording.
- The upload-size limit was not offered to the browser. README incorrectly said
  the maximum was 1 GiB; the existing configuration and upload owner support 8 GiB.
- Revision navigation exposed shortened internal IDs. There was no prominent
  result entry point or simple result selector.
- Accepted numerical facts were available through evidence, but not presented
  as a concise result summary. Scientific artifact sizes were not displayed.
- Keyboard submission had no documented shortcut. Initialization could write
  empty Session conveniences while asynchronous restoration was still pending.

A further defect was reproduced by downloading real USI-6 artifacts in Firefox:
embedding and Leiden descriptors offered `embeddings.h5ad` and `clusters.h5ad`,
but HTTP `Content-Disposition` named both `analysis.h5ad`. The original headers
are retained in `R/browser-accepted-verified.log`; the browser actually saved
`R/downloads/analysis.h5ad`. A frontend `download` attribute cannot reliably
supersede the server's filename. This justified the small delivery-layer fix.

## C. Design and user interaction

The page now has a compact identity/model header, closed Session settings, one
conversation surface, its composer, and vertically arranged result details.
The palette uses a light background and one muted teal accent, restrained borders,
readable text and visible focus. No external assets, framework or dependency
were added.

A first visit creates an ordinary empty Session, without providers or science.
An existing saved Session reopens instead. Draft/model conveniences are not
written until initialization finishes. Explicit **New conversation** clears the
old composer and input selections so the next Session does not inherit an
unnoticed source choice. Historical Session records remain server-owned.

The composer offers **Attach files** and **Send**. Repeated native file selections
append to the same ordered draft. Rows show the filename, local display size,
transfer/registration state, retry and remove actions. Removing a draft attachment
returns keyboard focus to **Attach files** and never deletes accepted server bytes.
A failed transfer blocks submission while retaining other files. Completed draft
references are checked against the Session attachment catalog after refresh.
Attachments captured by the server clear from the draft and appear inside their
user message, with the server's consumed/offered/staged distinction.

Enter inserts a newline. Ctrl/Command + Enter sends only outside IME composition;
`isComposing` and key code 229 prevent accidental composition submission. Duplicate
submission, exact retry, polling, cancellation and stale-response guards retain
the existing Session/turn identities. Actual backend checkpoints drive status;
only byte transfer uses a measured percentage.

The conversation has a bounded, keyboard-scrollable message area. Refreshing
message content follows the newest result only when the reader was already near
the bottom; older reading positions and exact disclosure/focus identities remain.
Results are below the composer, not in a persistent side panel. Small screens
wrap filenames/actions and use the same controls without horizontal scrolling.

The duplicate disclosure buttons and empty normal-path input notices are removed.
**Advanced inputs** retains the established explicitly declared upload/library,
paired-index and registered-source interfaces. These remain useful for existing
operator/legacy input selections and are not silently reinterpreted as neutral
attachments. **Configured scientific resources** appears only when the existing
catalog offers applicable controls. Ordinary attachment tasks continue through
USI-1–6 typed clarification; the frontend supplies no scientific roles or defaults.

## D. Implementation inventory

| File | Responsibility |
| --- | --- |
| `src/agent/web/static/index.html` | Simplified controls, composer hints, result entry point/selector, summary and limitations placement, accessible labels. |
| `src/agent/web/static/styles.css` | Single-column visual hierarchy, bounded conversation, attachment/result surfaces, responsive wrapping, focus and reduced-motion support. |
| `src/agent/web/static/app.js` | Empty-Session startup, safe convenience persistence, attachment display/limit precheck, IME-safe shortcut, scroll/disclosure preservation, friendly result labels, accepted quantity summaries and direct artifact links. |
| `src/agent/web/app.py` | Add optional `upload_max_bytes` to the existing enabled `/resources` response, read directly from the upload owner. Disabled response remains unchanged. |
| `src/agent/application/analysis_delivery.py` | Share the existing three accepted analysis-stage filenames. Descriptor identities, export data and scientific checks are unchanged. |
| `src/agent/application/matrix_delivery.py` | Use that closed filename mapping for compact-analysis snapshots; legacy absent-stage UMAP remains `analysis.h5ad`. |
| `tests/web/test_upload_api.py` | Actual advertised limits, bounded oversized-header rejection, cleanup, unchanged provider/science counts. |
| `tests/web/test_fastq_selection.py`, `test_uploaded_multiturn.py` | Retain exact catalog-response assertions including the new actual upload-limit field. |
| `tests/web/test_standard_analysis_handoff.py` | Assert download response filenames agree with all four independently offered descriptors. |
| `tests/tools/analysis/test_compact_analysis_delivery.py` | Exercise embedding/Leiden snapshot names, valid exported bytes, digest and capacity release. |
| `tests/web/test_upload_static.py`, `test_scientific_artifact_static.py` | Update assertions for consolidated controls and explicit server-source size display. Existing scientific/privacy assertions remain. |
| `README.md` | Current upload range, explicit 2 GiB deployment configuration and conversational UI instructions. |
| This report | Audit, evidence, validation scope and disposition. |

There is no new application/scientific schema, workflow engine, state manager,
resource resolver, model integration, report engine or comparison engine.
Registry, Planner/compiler/preflight, Session machinery, scientific authority,
all algorithms and `src/agent/tools/models/epizoo.py` remain unchanged.

## E. Actual rendered-browser acceptance

The installed Firefox runs headless with its native Marionette protocol, using
existing outside-Git browser helpers; no frontend testing package was installed.
Native file inputs and buttons, normal HTTP routes, actual durable application
state and real tiny H5AD inspection are exercised. Language decisions are
scripted, not live-provider qualification. Existing USI-6 scientific results are
read separately without rerunning their science.

Desktop and tablet checks use 1366 and 768 CSS-pixel window widths. Firefox's
headless window has a 500-pixel minimum; browser zoom produces an actual **392
CSS-pixel** narrow viewport (382-pixel document content width with scrollbar).
The timeline records measured `innerWidth`, `clientWidth` and `scrollWidth`, not
just the requested 390-pixel window size. This checks real responsive browser
layout, not physical-phone touch/virtual-keyboard behavior.

| Evidence | Verified behavior |
| --- | --- |
| `R/browser-final.log`, `browser_acceptance.py` | Coherent upload → clarification → supplementary H5AD → real inspection → evidence/report → second result → older/current navigation → scientific answer → refresh workflow. Two inspection calls; model sequence alpha, beta, beta, beta. |
| Same | Native repeated file selection/removal; real invalid-filename rejection; deliberately aborted XHR and exact retry; 2 GiB+1 sparse engineering file rejected before transfer; accepted earlier attachments retained. IME/Enter do not submit; Ctrl+Enter does; repeated Send creates no duplicate turn. |
| Same | Another client changes Session generation; stale composer submission fails visibly, refreshes context and performs no extra provider/science call. Detached 5 GiB checkpoint descriptor builds a direct link with zero fetch calls. This is a renderer test, not a 5 GiB transfer. |
| `R/browser-restart-prepare.log`, `browser-restart.log`, `browser_restart.py` | Pending task, draft and unsent supplemental upload survive a fresh server; reads make zero provider/science calls. Reply continues the original objective with one inspection. Earlier completed history/generation/revisions restore exactly without additional calls. Long Chinese/ASCII filenames wrap on the narrow viewport. |
| `R/browser-accepted-closeout.log`, `browser_accepted.py` | Read real USI-6 four-revision history and persisted comparisons. Select exact older/newer Leiden outputs: 32 cells, 3 clusters/resolution 0.5 versus 4 clusters/resolution 1.0. Preview accepted report and PNG. Deliver four actual H5AD exports with matching HTTP filenames. Native browser saves `embeddings.h5ad`; its delivered hash and 32 × 512 embedding shape are independently checked in `R/native-saved-download.json`. Unavailable artifact produces a visible safe error. Zero forbidden provider/science calls. |
| `R/browser-models.log`, `browser_models.py` | Actual operator-admitted Groq and OpenRouter profile choices, selection and refresh retention. Native Tab produces visible 3px focus; visible form controls have accessible labels. Provider submission is disabled in this separate metadata acceptance server. |

Configured choices are `groq-gpt-oss-120b` / `openai/gpt-oss-120b` and
`openrouter-gpt-5-6-sol` / `openai/gpt-5.6-sol:floor`, with the operator's actual
display labels. Scripted beta-turn execution separately proves that the selected
next-turn profile is transmitted. Neither test claims the OpenRouter provider's
known HTTP 404 is resolved.

Useful rendered screenshots, visually inspected during the work:

- Before: `R/before-1366.png`; `R/before-390.png` actually has a 500-pixel viewport.
- Initial: `R/initial-1366.png`, `initial-768.png`, `initial-390.png`.
- Attachments/errors: `R/upload-error-390.png`, `long-attachment-mobile.png`.
- Clarification: `R/clarification-1366.png`, `clarification-390.png`.
- Results: `R/accepted-results-1366.png`, `accepted-results-768.png`,
  `accepted-results-390.png`, `inspection-result-1366.png`.
- Actual accepted figures/reports: `R/accepted-figure-1366.png`,
  `accepted-figure-390.png`, `accepted-report.png`.
- Actual model catalog: `R/configured-models-desktop.png`.

Main text, muted text and primary-action contrast ratios are respectively 12.74,
5.19 and 6.24 (`R/review-checks.json`). Keyboard focus, labels, status/error roles,
IME guarding and narrow layouts are checked. No human screen-reader evaluation,
physical mobile device test or cross-browser compatibility claim is made.

Early acceptance-driver failures are retained: one expected three USI-6 revisions
but the preserved final Session has four; another changed localStorage during
asynchronous startup instead of reopening via the Session control. Corrected
checks use exact server inventory and ordinary UI reopening. Neither changed
historical evidence. The startup convenience write guard also prevents a rapid
refresh from overwriting the saved Session before restoration completes.

## F. Bounded uploads and deployment

The default remains **256 MiB per file**. The existing supported range is one byte
through **8 GiB**. A practical explicit configuration for the previously accepted
large dataset and FASTA workflows is:

```json
{
  "upload_root": "uploads",
  "upload_max_bytes": 2147483648,
  "upload_max_concurrent": 2
}
```

These are fields to add to the existing operator JSON, not a complete replacement
configuration. Restart the Web server. No production default/configuration was
changed here. The browser reads the current limit from `/api/v1/resources` and
rejects oversized selected files with removal/operator/retry guidance; HTTP
header and actual streamed-byte checks remain authoritative.

Uploads still stream binary files via XHR and the existing bounded owner.
Server writes/hash updates remain at most 1 MiB per slice. Capacity, filesystem
leases, staging, fsync, incomplete-upload cleanup, disk exhaustion handling and
the 128-file bound are unchanged. The existing 1–4 configurable concurrent limit
is unchanged. No aggregate disk quota or resumable/chunk protocol is added.
Operators must provision disk for retained inputs, results and temporary files,
and configure any reverse proxy's request-size/timeout policy consistently.

This unit tests the actual configuration/header/stream-rejection boundaries and
native oversized-file precheck. It does not repeat the previous approximately
836 MiB dataset and 1.71 GB FASTA transfer benchmarks. A successful transfer does
not establish a scientific role, compatibility or qualification.

## G. Scientific summaries, reports and downloads

The frontend displays exact persisted responses and accepted step statuses.
The result selector opens an exact server Revision; browsing does not activate
it. **Use as current result** and **Continue from here** retain generation-checked
activation. Result numbers describe the available ordered history; bounded
histories are labelled “Recent result.” Full IDs and parent relationships remain
in expandable details. Comparisons are existing evidence-grounded chat answers.

A summary's output selector uses `revision.evidence_outputs`; numeric values
come only from available accepted evidence facts for that explicit output.
Field labels and limitations come from the existing evidence projection. There
is no computation of new metrics, matching across outputs, filename inference,
scientific qualification or biological conclusion in JavaScript. Non-numeric
annotation detail, method semantics and training limitations remain in the
accepted response/report/evidence, not fabricated overview cards.

Scientific links use exact server-offered handles and closed reviewed descriptors.
The UI labels descriptor sizes **Source size**, because a privacy-safe derivative
can be larger or smaller. The native browser streams the download; JavaScript
does not buffer scientific H5AD/checkpoint bytes. Bounded text-report and PNG
preview continue using the existing formats and delivery paths. Errors open in
a separate browser context for direct downloads, preserving the conversation;
preview/inventory failures show local recovery guidance.

Real tested exports are `embeddings.h5ad`, `neighbors.h5ad`, `clusters.h5ad` and
`analysis.h5ad`, including independent UMAP without Leiden. Final headers,
delivered sizes and hashes are in `R/accepted-downloads.json`. Earlier and final
browser logs preserve the filename defect and its correction. Snapshot creation,
source/content digests, integrity checks, privacy stripping, capacity, cleanup
and export bytes are not changed by the filename fix.

Reports remain explicitly **client-safe report projections**. Original Markdown
can contain private server paths and remains withheld by the existing contract;
no unsafe original-report link is invented. PNG bytes are existing accepted
figures, not regenerated by the UI. The existing checkpoint-only **5 GiB** bound,
metadata-specific bound and separate matrix/analysis/table limits are unchanged.
No global 5 GiB download cap is introduced.

## H. Regression and preservation

| Run | Result |
| --- | --- |
| Initial focused Web/upload/configuration | 214 passed in 17.29 s (`R/focused.log`, XML). |
| Final compact export/HTTP/static | 64 passed, three warnings in 32.09 s (`R/final-delivery.log`, XML). |
| Final matrix/table/model delivery boundaries | 170 passed, two warnings in 62.60 s (`R/delivery-boundaries.log`, XML). |
| Final Web/API/Session focused regression | 308 passed in 107.74 s (`R/final-web.log`, XML). |
| Broader whole-Web/application regression | 1,041 passed, 7 skipped, 9 failed, four warnings in 1,468.36 s (`R/regression.log`, XML). Reconciliation below. |
| Exact starting-HEAD failed-node reproduction | 6 failed, 1 passed, three warnings in 129.41 s (`R/baseline-failures.log`, XML). |
| Final catalog modules and failed-node reconciliation | 48 passed, the same 6 baseline failures, three warnings in 137.07 s (`R/final-reconciliation.log`, XML). No new failures. |

The broad run began during implementation, before the final presentation and
filename corrections. The final focused runs above cover those changed areas;
the broad run is not misrepresented as an immutable final-tree execution.
Two broad failures were exact catalog-shape expectations in FASTQ selection and
uploaded multi-turn tests. They are updated to assert `upload_max_bytes` alongside
all original fields, without relaxing scientific assertions. All 47 tests in the
two complete catalog modules pass in the final reconciliation.

The five reproduced baseline assertion failures are:

- `test_leiden_resolution.py::test_invalid_explicit_declaration_never_reaches_scientific_production[0.7-changes8]`.
- `test_multi_source_selection_transport.py::test_selection_transport_rejects_unbounded_ambiguous_or_path_inputs[changes1]`.
- `test_unified_scientific_result_delivery.py::test_ordinary_processed_result_presentation_downloads_history_and_fresh_service`.
- `test_unified_scientific_result_delivery.py::test_historical_dependency_alone_does_not_authorize_child_download_attachment`.
- `test_interactive_rich.py::test_existing_supported_detail_owner_is_reused_without_science[False-annotation_rationale-3]`.

The two unified-delivery expectations omit the already accepted USI-6 independent
Leiden download. The other assertions reproduce exactly at starting HEAD. They
are not changed by this UX work.

`test_rich_api.py::test_first_threshold_execution_retains_exact_qc_for_followup_guidance`
also exceeds its existing 20-second helper deadline in both the baseline and
final isolated runs, with scientific selection succeeded and presentation still
finalizing. `test_guidance_uses_normal_planner_and_navigation_creates_sibling_branch`
had the same timeout category in the broad run but passes on both isolated trees.
This is recorded as timing-sensitive existing test coverage, not a green full
suite or a newly repaired scientific lifecycle. No timeout or validation is relaxed.

`R/regression-reconciliation.json` compares the exact final six failed nodes with
the independently archived baseline: **no new failure nodes**, with all six failure
messages identical after normalizing run/revision digests. The final 551
source/test/static file hashes in `R/final-code-hashes.json` remain unchanged
during final reconciliation. Commands and XML preserve all assertions and skips.

All regression runs use isolated basetemps under R and disable pytest's cache
provider. The main regression removes every `RUN_*` environment gate and runs
all `tests/web` plus relevant upload, interactive boundary/rich/core/schema,
Session, local-resource, standard-comparison and processed-delivery tests.
`R/regression-command.json`, log/XML and result JSON preserve the exact invocation.
No historical eval collection, live-provider call, full biological analysis or
checkpoint training is run. Existing assertions are not deleted or weakened to
mask failures.

The original USI-6 1,815-pass / 6-skip / 12-baseline-failure record remains its
own scoped evidence. This differently scoped regression is compared by exact
failed node against an isolated `git archive` of starting HEAD, not by count.
No baseline assertion or wait timeout is changed to obtain a green result.

The preservation audit passes for all 92,862 recorded files. File-set checks show
no additions/removals in the recorded USI-4 resource workspace (40,487), USI-5
(20,938), USI-6 (28,954) and `evals/` (1,274). The audit retains the inherited
symlink scope: fifteen recorded USI-5 file links are included; historical links
excluded from earlier inventories are not retrospectively claimed as hashed.
The first file-set checker excluded those fifteen links and was corrected to
compare the same recorded scope. All original recorded content/stat checks
passed. EpiZoo's separate working tree is clean. `git diff --check` passes.

## I. Remaining limitations

- The browser remains a trusted local/lab application with the established
  Session-ID reopening mechanism; it gains no account system or Session listing.
- Draft persistence depends on browser storage. Interrupted unsaved transfers
  need reselection after refresh; completed bytes are not retransmitted blindly.
  Large-file throughput, aggregate disk quotas and proxy deployment sizing remain
  operator concerns.
- Scientific eligibility, annotation meaning, raw-processing declarations,
  resource availability and model reuse restrictions remain USI-1–6 contracts.
  UI simplification does not imply arbitrary H5AD conversion, biological validation
  of group-derived labels, or improved biology from reduced training loss.
- Native validation uses Firefox at desktop/tablet/narrow CSS viewports. Physical
  mobile keyboards, touch behavior, other browsers and human screen-reader use
  are not comprehensively qualified.
- Language decisions in the executed tiny workflow are scripted. Actual
  configured profile display does not prove provider availability. OpenRouter's
  previous HTTP 404 and comprehensive four-scenario live-provider/native-browser
  scientific qualification remain UA3.6.5; that work has not begun.

## J. Final disposition

**USI-7 COMPLETE.** The ordinary path is a unified conversation with integrated
multi-file attachments, visible next-turn model choice, necessary typed
clarifications, accepted quantity summaries, report/figure previews, direct
scientific downloads and friendly result navigation. Advanced scientific choices
remain available without dominating the page. Fifteen files are changed: six
production files, seven test files, README and this new report; their ownership
is listed in section D.

Actual Firefox acceptance covers the coherent tiny-data workflow and fresh-process
continuation, plus read-only real USI-6 evidence, comparisons, PNG/report viewing
and four independent H5AD exports. Final focused runs pass 308 Web/API/Session,
64 export/HTTP/static and 170 delivery-boundary checks (overlapping scopes, not
an additive unique-test total). Final reconciliation passes 48 checks with six
independently reproduced baseline failures. No new correctness, security or
scientific regression is known within this scope. All 92,862 recorded protected
files remain unchanged.

The UI is ready for UA3.6.5's separately scoped live-provider qualification.
Physical-device/cross-browser testing, existing timing-sensitive tests, biological
scope limits, provider availability and explicit upload/proxy deployment settings
remain as described above. Production settings and scientific defaults are
unchanged. No commit, stage or push was performed.

Final `git status --short` (the pre-existing `evals/` entry is preserved):

```text
 M README.md
 M src/agent/application/analysis_delivery.py
 M src/agent/application/matrix_delivery.py
 M src/agent/web/app.py
 M src/agent/web/static/app.js
 M src/agent/web/static/index.html
 M src/agent/web/static/styles.css
 M tests/tools/analysis/test_compact_analysis_delivery.py
 M tests/web/test_fastq_selection.py
 M tests/web/test_scientific_artifact_static.py
 M tests/web/test_standard_analysis_handoff.py
 M tests/web/test_upload_api.py
 M tests/web/test_upload_static.py
 M tests/web/test_uploaded_multiturn.py
?? docs/usi-7-web-ux-reports-download-closeout.md
?? evals/
```

## K. Final UX Polish and Human Review Closeout

2026-10-16. This continues the uncommitted USI-7 implementation above. Sections
A–J remain the initial acceptance record; this section records the subsequent
changes requested after the user's visual review. It does not claim that the
user has approved the final screenshots.

Final-polish evidence root **P** is
`/home/likeyi/agent-usi7-polish-2ukynsts`. Original evidence root **R** remains
`/home/likeyi/agent-usi7-qhtee_1k`. Both are outside Git. All original USI-7 edits
are retained, including the embedding/Leiden HTTP filename correction.

### A. Initial findings

The reviewed initial desktop, result, figure and report screenshots showed a
large welcome area and redundant composer labels; execution steps, output lists
and origin text appeared before the figure. Reports were labelled scientific
reports although their delivered content was a privacy-safe verification
projection. Artifact identities and repeated export notices competed with file
names and actions. The clarification screenshot referred to scientific roles and
selection positions without clearly inviting a missing file. Historical browsing,
activation and continuation lacked a concise explanation of their differences.
The narrow and configured-model screenshots were also inspected.

### B. Changes and ownership

This follow-up changes only four production files:

- `src/agent/web/static/index.html`: summary, preview, downloads, visible
  limitations, and expandable methods/evidence; compact composer; explanatory
  result navigation; expandable technical report record.
- `src/agent/web/static/styles.css`: shorter empty state, integrated composer
  actions, quieter file rows, readable answer width, bounded original figure
  display, and responsive layout. The final narrow layout explicitly keeps Send
  inside its panel; document-wide overflow checks alone had missed that defect.
- `src/agent/web/static/app.js`: concise accepted quantities and method label,
  existing safe figure preview on result load, descriptive download names,
  closed chat download disclosures, report labelling and secondary metadata.
- `src/agent/application/responses.py`: only the deterministic text of
  `missing_input_roles`. The existing choices are required tool/port roles,
  **not qualified dataset candidates**. The new question asks which file to use,
  invites an attachment if absent, and preserves the original task/files. The
  role labels, typed decision, admission and continuation are unchanged. This
  fixes the reproduced text-only-attachment ambiguity without a new resolver.

Four existing tests are updated for intentional presentation changes:
`tests/web/test_static.py`, `test_rich_static.py`,
`test_scientific_artifact_static.py`, and
`tests/application/test_response_contracts.py`. This report is the ninth file
changed by this follow-up. Across both USI-7 passes there are nineteen changed
files: seven production files, ten test files, README and this report.

The ordinary empty composer no longer repeats Ready, Ask Agent and a saved-state
caption visually; its accessible label remains. Busy, failed, pending and upload
states remain visible. Existing advanced input capabilities stay available.
Operator model labels remain exact, including routing text; a wider selector and
full option title improve readability without altering profile IDs or routing.
Chat answers retain the complete stored text, including repetitive scripted
comparison qualifications. Chat downloads remain attached to their exact turn,
inside one closed disclosure rather than another expanded result panel.

### C. Scientific summary and technical report

The output selector still selects an exact offered `revision.evidence_outputs`
name. The method label comes from `EvidenceView.source.tool_name`; quantities
come from available numeric `facts`, with existing `field_labels`. At most four
are shown, in evidence order, excluding seed fields from the short summary.
Seeds and every other offered field remain in the full evidence view. No values
are combined across outputs and no metrics are computed in JavaScript.
Selecting the accepted initial Leiden output shows **32 cells, 3 clusters,
resolution 0.5**; the second result shows **32 cells, 4 clusters, resolution 1**.
All offered limitations for the selected output are visible below downloads.
Missing quantities are explicitly reported rather than supplied by inference.

Accepted PNGs use the original bounded preview endpoint and original bytes.
The first offered PNG is displayed in inventory order, purely as a presentation
choice. Multiple PNGs are numbered, with all preview actions available. The
existing descriptor has no semantic figure title or primary-figure designation;
the UI does not invent one or infer a figure's scientific meaning from filenames.
The caption explains that changing the summary selection does not change the
figure. No figure or placeholder appears for a figureless result. This is not a
new association between Leiden summaries and independently computed UMAPs.

The delivered plain-text projection is now labelled **Technical report**,
explained as execution/verification evidence rather than biological interpretation.
Its complete original safe text is available under **Full execution & verification
record**, collapsed initially. Native-browser validation compares expanded
`textContent` with the exact delivered text. No Markdown parser, report generator,
private original-report access, path recovery or additional LLM call is added.
The readable scientific summary remains separate and above the technical record.

### D. Download presentation and boundaries

Names are derived from closed reviewed descriptors already accepted by
`supportedScientificFile`: Cell embeddings (H5AD), Neighbor graph (H5AD), Leiden
clusters (H5AD), UMAP analysis (H5AD), and Annotated cells (H5AD). Existing matrix,
QC/selection table and model labels remain available. Report/figure names use
trusted artifact kinds, with numbering when necessary. Unsupported kinds use a
neutral label; arbitrary uploaded filenames never establish scientific identity.

Each scientific row retains the exact offered filename, source size and direct
native download link. Export-size/privacy guidance is shared in the main file
area rather than repeated per analysis file. Model reuse requirements remain
explicit. Report/PNG descriptors have no size or descriptive scientific-title
field; the UI does not invent those values. Their identity and source digest
remain under File provenance. Normal links preserve exact server handles,
authorization, pinned digests, privacy filtering, capacity and cleanup. Large
scientific payloads are never fetched into browser JavaScript. The checkpoint
5 GiB bound and all other delivery bounds remain unchanged.

`P/accepted-downloads.json` and `download-comparison.json` verify all four real
USI-6 H5AD exports have exactly the same delivered size and SHA-256 as initial
USI-7. All four HTTP filenames match their descriptors. Firefox again saves
`embeddings.h5ad` through a native link; its saved digest matches the response.

### E. Actual before/after browser evidence

Firefox uses the existing native Marionette helper, not a new framework. The
accepted USI-6 workspace is served GET-only with provider/science functions
replaced by failing counters. The tiny simulated upload/inspection fixture and
configured-model metadata server remain isolated under P. The latter performs
no provider submission. Browser scripts, timelines and logs are retained.

| View | Before | After |
| --- | --- | --- |
| Initial desktop/narrow | `R/initial-1366.png`, `R/initial-390.png` | `P/initial-1366.png`, `P/initial-390.png` |
| Attached composer | `R/upload-error-390.png` | `P/attached-files-1366.png`, `P/attached-files-390.png` |
| Clarification | `R/clarification-1366.png`, `R/clarification-390.png` | `P/clarification-1366.png`, `P/clarification-390.png` |
| Accepted result and figure | `R/accepted-results-1366.png`, `R/accepted-figure-1366.png` | `P/accepted-results-1366.png`, `P/accepted-figure-1366.png` |
| Narrow result | `R/accepted-results-390.png` | `P/accepted-results-390.png`, `P/accepted-figure-390.png` |
| Report | `R/accepted-report.png` | `P/accepted-report.png`, `P/accepted-technical-record.png` |
| Models | `R/configured-models-desktop.png` | `P/configured-models-desktop.png` |

For a controlled comparison, `P/before/` freezes the actual pre-polish static
files. `browser_comparison.py` serves these through a separate read-only server
and renders the same exact historical result and PNG at equivalent widths.
`before-comparison-{1366,768,390}.png` and matching `after-comparison-*` images
are full-page captures. `layout-comparison.json` records actual geometry:

| CSS viewport | Figure offset from result-panel top, before → after |
| --- | --- |
| 1366 px | 1723 → 577 px |
| 768 px | 1854 → 577 px |
| 392 px (requested 390) | 2289 → 714 px |

The same accepted plot is reached about two-thirds sooner. Narrow width is
measured, using the existing Firefox zoom method; all three document widths have
no horizontal overflow. Final composer checks also verify the Send button's
bounds against its containing panel. Actual screenshots were visually inspected.
This is native desktop-browser responsive validation, not physical-phone testing.

### F. Validation

`browser-interactions.log` exercises multiple attachments, interrupted-transfer
retry, real upload rejection, removal, oversize precheck, Ctrl/Command+Enter/IME
safety, duplicate submission, clarification supplementation, selected scripted
model execution, real tiny inspection, historical browsing/activation/continuation,
scientific answers, stale-generation rejection and refresh. A detached 5 GiB
checkpoint descriptor constructs a direct link with zero payload fetches.
`browser-restart-final.log` proves pending task, draft and supplementary attachment
recovery after a fresh server process; historical reads invoke no science/model,
then one explicit continuation performs the expected tiny inspection.

`browser-accepted-final.log` checks the preserved four-result USI-6 history,
comparison text, two exact Leiden summaries, real PNG and safe report, all four
H5AD deliveries, unavailable-artifact error, and zero forbidden calls.
`browser-polish-checks.log` additionally checks figureless presentation, closed
chat disclosures, disclosure/scroll preservation and a text-only attachment
clarification with no scientific execution. `browser-models.log` verifies actual
Groq/OpenRouter metadata, retained selection, native Tab focus and visible-control
labels. `browser-composer-final.log` verifies final responsive controls and
attachment removal with restored focus and no additional science/model call.

Focused presentation/response tests pass **69 tests in 1.55 seconds**. The first
24-module related regression passed **572 tests with one failure in 202.63 s**:
`test_execution_and_result_verification_failure_do_not_activate[verification]`
raised `updated_at cannot precede created_at` in unchanged run-state validation.
The exact starting-HEAD isolated node passes (1 test, 2.50 s); this is **not**
claimed as a reproduced baseline failure. No clock, runtime, Session code or test
assertion is changed to mask it. Final stabilized regression results are recorded
below. The initial run also preceded the final narrow-composer CSS adjustment
and HTML whitespace cleanup; the final run uses the frozen final source tree.

The earlier broad USI-7 baseline failures and reconciliation in section H are
preserved as historical evidence, not counted as failures in this focused
selection. No full historical evaluation collection, EpiZoo training or expanded
scientific-domain regression is run.

### G. Preserved boundaries and remaining limitations

Scientific algorithms, Registry, Planner/compiler/preflight, Session/Revision
semantics, scientific authority and input qualification remain unchanged. The
only new backend change is clarification presentation text. No new endpoint,
schema, resolver, state manager, framework or report engine is introduced.
Browsing remains read-only; Make current activates the viewed result, and Continue
also opens a draft. Neither starts scientific execution before Send.

All 92,862 previously protected file entries pass the same bounded stat/hash
check. All **14,586 files in the initial USI-7 evidence root** also retain their
recorded sizes, mtimes and hashes where bounded, with no additions or removals.
See `P/preservation.json`, `prior-evidence-preservation.json`, and the final log.
Operator configuration, evals, accepted data, checkpoints and prior workspaces
are preserved. Source/test/static hashes are frozen in `P/final-code-hashes.json`.

The UI cannot supply absent semantic figure titles, figure sizes or a fully
authored biological report from the current descriptors. Full technical records
can still be long when explicitly expanded. Scientific limits remain those of
the accepted evidence; concise cards are not complete results or scientific
interpretation. Scripted answer wording and historical stored clarifications are
not rewritten. Live-provider availability, including the earlier OpenRouter 404,
is not qualified or claimed fixed. The production upload limit remains operator
configured; no configuration changes are made. Unsupported scientific operations
remain unsupported. UA3.6.5 has not begun.

### H. Final regression and disposition

The final stabilized 24-module Web/API/result-delivery/Session/response selection
passes **573 tests, three existing dependency warnings, in 204.81 seconds**,
exit 0. See `P/regression-final-command.json`, `regression-final.log` and XML.
It includes the previously passing 308/64/170 selections as a deduplicated union
plus the affected response and attachment-continuation tests; these counts must
not be added together. The earlier timestamp failure does not recur in the full
final selection. Its initial traceback and passing isolated baseline check are
retained, without asserting a cause beyond the observed timestamp ordering.
Final source/test/static hashes match the frozen set (`P/final-code-check.json`).

Early harness issues are retained transparently: the first focused invocation
omitted `PYTHONPATH=src` and failed collection; one legacy static assertion
expected summary/download disclosures closed and was updated to the requested
new hierarchy; one restart driver navigated before the new server was ready and
passed after checking readiness. None required a scientific-contract change.

**USI-7 FINAL UX POLISH COMPLETE.** The rendered result prioritizes accepted
quantities and an original figure, has clear scientific download names, and
separates readable summaries from technical records. The coherent interaction,
restart and read-only accepted-result paths pass within the scopes above.
The final screenshots and remaining limitations are recorded in this section.
All owned acceptance servers and Firefox were stopped. No staging, commit,
push or UA3.6.5 work was performed.

Final `git status --short`:

```text
 M README.md
 M src/agent/application/analysis_delivery.py
 M src/agent/application/matrix_delivery.py
 M src/agent/application/responses.py
 M src/agent/web/app.py
 M src/agent/web/static/app.js
 M src/agent/web/static/index.html
 M src/agent/web/static/styles.css
 M tests/application/test_response_contracts.py
 M tests/tools/analysis/test_compact_analysis_delivery.py
 M tests/web/test_fastq_selection.py
 M tests/web/test_rich_static.py
 M tests/web/test_scientific_artifact_static.py
 M tests/web/test_standard_analysis_handoff.py
 M tests/web/test_static.py
 M tests/web/test_upload_api.py
 M tests/web/test_upload_static.py
 M tests/web/test_uploaded_multiturn.py
?? docs/usi-7-web-ux-reports-download-closeout.md
?? evals/
```

## Final Two-Panel Results & Conversation Layout

### A. Baseline and bounded scope

The requested left-results/right-conversation presentation supersedes the earlier
single-column preference. Long scientific results previously separated the
composer from the result being discussed. This refinement keeps both available
within the same desktop viewport.

The actual starting HEAD was `1e3cded096a9db821fbf2eb289033da11ade3d77`
(`Complete USI-7 web UX scientific reports and downloads`). Earlier USI-7 work
was already committed; the initial tracked tree and index were clean, with only
pre-existing `?? evals/`. The preceding sections retain their historical Git
status and acceptance records. No earlier work was reverted.

New isolated evidence root **L**:
`/home/likeyi/agent-usi7-layout-s9ekz13p`.
`L/git-baseline.json` records the starting state; `L/before/` preserves its three
static files. R and P, defined above, remain protected historical evidence.
This change touches only HTML/CSS/JavaScript, presentation tests and documentation.
No backend contract change was necessary.

### B. Layout and interaction

At desktop widths the workspace uses approximately 60% for results and 40% for
conversation, with a 400 px minimum conversation column and an 1800 px maximum
overall width. The compact shared header retains the conversational model
selector, New conversation and expandable Session settings. Scientific checkpoint
selection retains its separate existing input control.

The results pane and message history scroll independently. The composer remains
at the bottom of the conversation pane; long attachments or expanded input
settings can scroll within its bounded area. Chronological messages, full accepted
answers and caveats remain intact. Existing disclosure, focus and message-scroll
restoration continues to follow new messages only when the reader was near the
bottom. The chat retains concise result messages and its existing optional links.

Below 1100 px, the combined ~600 px results / 400 px chat content requirement no
longer fits comfortably with padding and gutters. Conversation and Results buttons
then select one pane. The inactive pane retains its geometry while being hidden
from interaction, preserving both scroll positions even when history updates.
Buttons expose their selected state; switching focuses the results region or
composer without scrolling the whole document. No routing framework is added.

Before the first accepted result, a compact centered conversation is prominent.
Desktop adds a small explanatory results empty state; narrow screens omit that
card and disable the Results switch. Upload, message entry, model choice and Send
are visible without a large empty scientific workspace.

The left hierarchy is a grounded method title and compact result selector,
accepted status/version, selected-output summary quantities, the original accepted
figure when available, and download/report rows. Figureless results omit the
viewer. Existing limitations remain readily visible; methods, full evidence,
resource provenance and execution/lineage details remain expandable. Technical
reports retain their explicit label and safe delivered projection. File names,
formats, source-size wording, export-size distinction and native streaming links
are unchanged. No scientific figure is redrawn or reinterpreted.

### C. Accepted-result selection

The existing Session envelope owns the accepted inventory and active Revision.
When the displayed result was the previously active result, an updated envelope
selects the new server-active result automatically. This covers the first result
and subsequent successful publications without requiring a View result click.
No arbitrary last historical result is substituted when there is no active result.

A deliberately selected historical Revision remains displayed when new results
arrive. A compact banner identifies the preview and the active conversation result,
with **View current result** as an explicit return action. Browsing and this return
action do not activate a Revision. Make current and Continue retain their existing
generation-checked activation behavior; Continue then focuses a draft in the chat.
Science still requires Send.

When the exact displayed Revision is unchanged, Session refresh updates only its
navigation/context labels. It does not reload the immutable result content. Thus
an open report, selected evidence output, figure, downloads and results scroll
position survive answer-only turns, clarification, failure and a new publication
elsewhere in history. When selection changes, the existing request-epoch guards
clear stale content and bind the viewer, summary and downloads to that exact
Revision before rendering it.

Local convenience storage adds only the selected panel and a deliberate historical
preview ID. The preview must be offered by the restored Session before use. The
previous active result is not persisted as a forced preview, allowing a fresh load
to follow the server's current active result. Scientific facts, artifacts and
activation authority are not cached in convenience storage.

### D. Actual Firefox acceptance

Native Firefox/Marionette uses the existing outside-Git helper and isolated
servers. Rendered CSS widths are 1600, 1366, 768 and approximately 392 px. The
narrow view uses desktop Firefox zoom/window simulation, not a physical mobile
device. All measured document widths equal their scroll widths.

| Scope / evidence | Observed result |
| --- | --- |
| `L/browser-interactions.log` | Multiple attachments, removal, real upload rejection, aborted-XHR retry, local oversized-file rejection, IME/Enter protection, keyboard submission, duplicate-submit suppression, clarification supplementation, model choice, historical activation/continuation and stale-generation rejection pass. A detached large-checkpoint descriptor remains a native link with no frontend fetch/buffering; this is not a real 5 GiB transfer. |
| `L/browser-follow-verified.log` | A simulated two-cell inspection publishes the first result automatically, then follows a second. While previewing the first result, an answer and third publication preserve the exact report text, disclosure, scroll and links. Refresh retains the deliberate preview; View current result returns to the active third result. An actual corrupt-H5AD tool failure and subsequent clarification leave that accepted result intact. |
| `L/browser-accepted-layout.log` | GET-only preserved USI-6 results show 32 cells / 3 clusters / resolution 0.5 and 32 cells / 4 clusters / resolution 1.0 from their respective evidence. Browsing leaves active Result 4 unchanged. Actual PNG/report previews, exact report text, four H5AD response names/digests and visible unavailable-artifact errors pass. Provider/science forbidden-call counters remain empty. |
| `L/browser-panels-final.log` | Desktop results scrolling leaves composer geometry unchanged. Older chat scroll survives history rendering. At 768/392 px, switching retains both scroll positions and exact historical identity, including history updates while chat is hidden. Focus and Send bounds pass; refresh restores the Results pane and preview. All four actual native download links save matching exports. |
| `L/browser-pending-prepare.log`, `browser-pending-restart.log` | A notes-only request clarifies while retaining the old accepted result. After a fresh server starts, history, active/preview identities, model selection, draft and uploaded supplement restore with zero model/science calls. Sending the supplement resumes the original task with one model call and one `inspect_scATAC` call, creates one accepted Revision and retains the deliberate old preview. |
| `L/browser-models.log` | Actual operator-admitted Groq/OpenRouter profile metadata and selection restore without provider submission. Native Tab traversal has visible 3 px focus; visible controls have accessible labels. |

The successful auto-follow Session is `2ae8fd9c-31ca-4e63-a85b-0eeb4dafb05e`.
Its first phase has three successful inspections and one deliberately failed
inspection; the later restart continuation adds one successful inspection.
Earlier harness attempts use separate Sessions and are not folded into these
counts. Simulated tiny data and scripted providers establish bounded application
behavior, not biological or live-provider qualification. The preserved USI-6
science is read only; no EpiZoo or MAESTRO computation is repeated.

`L/accepted-downloads.json`, `export-comparison.json` and `native-downloads.json`
record `embeddings.h5ad`, `neighbors.h5ad`, `clusters.h5ad` and `analysis.h5ad`.
All delivered sizes/digests and accepted source digests match P. Native saved bytes
match the delivered export digests; exports are not falsely equated with source
H5AD bytes. Repeated clicks receive Firefox's normal filename suffixes.

Actual before/after captures (all outside Git):

| View | Before, stacked UI | After, two-panel UI |
| --- | --- | --- |
| Initial desktop/narrow | `P/initial-1366.png`, `P/initial-390.png` | `L/initial-1366.png`, `L/initial-390.png` |
| Accepted desktop result | `P/accepted-results-1366.png` | `L/results-1366.png`, `L/results-1600.png` |
| Original scientific figure | `P/accepted-figure-1366.png`, `P/accepted-figure-390.png` | `L/accepted-figure-1366.png`, `L/accepted-figure-390.png` |
| Tablet/narrow panes | `P/accepted-results-390.png` | `L/results-768.png`, `L/results-390.png`, `L/conversation-768.png`, `L/conversation-390.png` |
| Download rows | `P/accepted-results-1366.png` | `L/downloads-1366.png`, `L/downloads-390.png` |
| Publication/history | Prior stacked result above | `L/publication-first.png`, `L/publication-second.png`, `L/historical-preview-new-result.png` |
| Pending task | `P/clarification-1366.png` | `L/pending-supplement-desktop.png` |

The screenshots compare preserved previous acceptance to the new rendered UI;
`L/before/` also freezes the exact starting static source. Final layout captures
are logged in `browser-layout-verified.log`. Early implementation screenshots
remain evidence of intermediate states, not all final-layout claims.

### E. Tests and preservation

Focused static/result-display tests pass **59 tests in 0.54 s**. Eleven new cases
execute the production display-selection/refresh functions through the existing
JavaScriptCore helper: accepted active follow, historical retention, cross-Session
isolation, unavailable references, no-active state and unchanged-result content
across answer/clarification/running/failed/cancelled turn refreshes. Existing tests
continue to cover exact artifact handles, safe delivery and Session contracts.
No new browser framework or scientific test infrastructure is introduced.

The final 24-module Web/API/artifact/Session/response regression passes
**584 tests, three dependency warnings, in 199.59 s**, exit 0. This is the previous
573-test selected scope plus eleven new cases, not an additional full historical
suite. See `L/regression-final-command.json`, log and XML. All `RUN_*` gates are
unset. All **540 frozen source/test/static/resource files** remain unchanged during
this final run (`L/final-code-hashes.json`, `final-code-check.json`). A preliminary
584-test run also passed, before the final empty-state CSS adjustment. Earlier
historical failure records above remain intact.

Harness corrections are retained transparently in L: an initial follow driver
tried reading lexical JS state from Marionette's sandbox; another clicked before
startup readiness; a layout driver checked an old same-ID DOM value before a
revision finished loading; and a download driver counted both chat and results
links. Drivers now wait on visible loaded state and scope download selectors to
the inventory. A rapid sequence of native downloads saved only two files; spacing
native clicks by two seconds delivered all four, without changing browser download
policy or production links. These checks do not establish a product defect.

The bounded preservation audit confirms all **92,862 protected file entries**,
**14,586 R evidence files**, and **5,363 P evidence files** retain recorded stat/hash
identities, with no additions/removals in the protected sets. Operator configuration,
accepted scientific workspaces, datasets, checkpoints and pre-existing evals remain
unchanged. See `L/preservation.json`, `prior-evidence-preservation.json`,
`polish-preservation.json` and logs. Scientific algorithms, Registry,
Planner/compiler/preflight, qualification, RunStore, authority publication,
Session/Revision semantics, artifact limits and privacy remain unchanged.

### F. Final disposition

**Final two-panel layout refinement is complete and ready for UA3.6.5 product
acceptance within this bounded UI scope.** Researchers can inspect the accepted
result and compose the next instruction together on desktop, switch usable panes
on narrow screens, automatically see new accepted results, or keep an intentional
historical preview without mixing result artifacts.

UA3.6.5 and live-provider scientific acceptance have not begun. Existing provider
availability and scientific qualification limits remain as recorded above. Mobile
keyboard/device behavior beyond desktop viewport simulation is not qualified.
Full technical records can still be long when explicitly expanded; semantic figure
titles and biological interpretations are limited to existing accepted descriptors.
All owned acceptance servers and Firefox were stopped.

Eight files changed: README, this closeout report, three static frontend files,
and three Web presentation test files. HEAD remains
`1e3cded096a9db821fbf2eb289033da11ade3d77`. Nothing is staged, committed or pushed.
Final `git status --short`:

```text
 M README.md
 M docs/usi-7-web-ux-reports-download-closeout.md
 M src/agent/web/static/app.js
 M src/agent/web/static/index.html
 M src/agent/web/static/styles.css
 M tests/web/test_rich_static.py
 M tests/web/test_scientific_artifact_static.py
 M tests/web/test_static.py
?? evals/
```
