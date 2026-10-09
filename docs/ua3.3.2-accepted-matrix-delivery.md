# UA3.3.2 — Accepted matrix H5AD delivery

Implemented and validated on 2026-10-09. Sections A–H record the original
unstaged UA3.3.2 checkpoint, when no commit or push had occurred. The UA3.3.3
closeout below records the combined accepted scope and supersedes its Git and
next-stage status. Original scientific files use the existing interactive
Application and Web interfaces.

## A. Repository baseline and preservation

Started on `main` at `16247822ca0f9dae5612f813015302647b68be49`;
tracked worktree and index were clean. The only untracked files were existing
`evals/` entries. No competing editing task was found; the visible Codex processes
were persistent VS Code services. No reset, clean, fetch or pull occurred.
HEAD, local `main` and local `origin/main` still identify that commit.

Content, size, modification time and mode checks preserve all 705 eval files,
198 output files, and 208 files across the real PBMC application and existing
Web workspace. Inventories have no additions or removals. Operator configuration
is unchanged (SHA-256
`dd86afd663fc2345b542bd78fa4bf97a5ebbfaf6d2557e84736bcddb4a6d4389`).
The original 5,231,645,507-byte pretrained EpiZoo checkpoint is unchanged
(SHA-256 `6b2d13fdbd54a9b0d56efa5afa81bc4832b813f4eac8662e4d93cc08c9d9b39a`).
Preservation records are outside Git at `/tmp/agent-ua332-yy78ki0e/`.

## B. Implementation

Modified files:

- `src/agent/application/interactive.py`
- `src/agent/application/interactive_schemas.py`
- `src/agent/web/app.py`
- `src/agent/web/static/app.js`
- `src/agent/web/static/styles.css`
- `tests/web/rich_helpers.py`

New files:

- `src/agent/application/matrix_delivery.py`
- `tests/application/test_scientific_matrix_artifacts.py`
- `tests/application/test_matrix_delivery_privacy.py`
- `tests/application/test_matrix_delivery_stream.py`
- `tests/web/test_scientific_artifacts.py`
- `tests/web/test_scientific_artifact_static.py`
- This record.

The allowlist covers exactly these existing contracts:

| Tool | Accepted contract |
| --- | --- |
| `build_scATAC_cell_by_ccre` | `scatac-cell-by-ccre.v1` |
| `adopt_scATAC_cell_by_ccre` | `scatac-cell-by-ccre.external.v1` |
| `build_scATAC_cell_by_features` | `scatac-cell-by-features.v1` |
| `build_scATAC_cell_by_features` | `scatac-cell-by-features.qc-selected.v1` |
| `adopt_scATAC_cell_by_features` | `scatac-cell-by-features.external.v1` |

Resolution follows the exact Session/Revision output locator, its actual
`output.run_id` and accepted step, reviewed `matrix`/`dataset` semantic port,
owner-recognized manifest, and existing schema-2 scientific authority. It checks
the accepted step digest, complete owner result summary, manifest digest/profile,
authority manifest identity, proof metadata, and payload/manifest closure entries.
The manifest SHA and matrix payload SHA remain distinct.

Descriptors contain only a server-issued opaque handle, safe filename/label,
size, digest and existing Revision/turn identities. The deterministic handle
binds Session, Revision, selected output locator, accepted authority identity,
artifact role, manifest and payload identities. There is no persistent artifact
index. Forged or mismatched handles fail closed.

`scientific_artifact_handles` reads only bounded accepted metadata and the
64 KiB maximum manifest, through component-safe regular-file opening. It never
opens, copies or hashes H5AD payloads. Inventory eligibility does not promise
current byte integrity or privacy eligibility; preparation checks those later.

Web routes under `/api/v1/sessions/{session_id}` are:

- `GET /turns/{turn_id}/scientific-artifacts`
- `GET /revisions/{revision_id}/scientific-artifacts`
- `GET /revisions/{revision_id}/scientific-artifacts/{handle}`

## C. Chat result integration

The existing assistant bubble includes a visible
**matrix.h5ad — Original accepted matrix** download row. Users can download
directly from that result. Existing scientific details, reports and figures
remain available; details also expose the same eligible files.

Conversation inventory requires the exact immutable successful execute
presentation, admitted request identity and created Revision. Only explicitly
selected new outputs from that execution are offered. Retained/upstream matrices,
inspection, embedding/UMAP, answers, navigation, failures and unbound legacy
presentations expose no new chat file control. Historical links use their own
Revision identities and never the active Revision pointer.

The browser caches metadata by exact Session/turn/Revision, fetching once per
eligible displayed result. Refresh/reopen clears the cache and resolves those
same persisted bindings. Unrelated conversation polling does no H5AD inspection.
Download anchors use ordinary browser attachment transport; JavaScript never
loads an H5AD Blob.

## D. Integrity, privacy and streaming

Preparation re-resolves the handle and accepted metadata. Source opening walks
every managed path component with `O_NOFOLLOW`, uses `O_NONBLOCK`, and requires
a regular file. Copying pins that descriptor, reads at most 1 MiB per chunk and
hashes an anonymous/unlinked private `TemporaryFile`. Accepted size and SHA-256
must match; source descriptor size/inode/time state is checked before and after
copying. No hard link or permanent duplicate store is used.

The same snapshot receives closed H5AD eligibility inspection before HTTP
headers or scientific bytes are sent. The policy covers the exact normalized
creation/adoption layouts, attrs, axis columns and scalar metadata. It rejects
extra metadata, absolute private path forms, oversized identifier strings,
external/soft links, hard-link aliases/cycles, virtual datasets, external raw
storage, unsupported filters, userblocks and custom fill values. It preserves
reviewed scientific identifiers and metadata. It reads identifier batches in
bounded fixed-width chunks and never reads sparse matrix count values or
reconstructs science. An ineligible original is rejected without modification.

Successful responses stream only the validated snapshot, with binary media type,
attachment filename `matrix.h5ad`, verified Content-Length and SHA-256 headers.
Existing no-store/security headers remain. Source replacement after preparation
cannot change transferred bytes. Preparation failure and response construction,
disconnect, cancellation before iteration, interruption and send failure release
the snapshot and capacity. The ASGI response owns explicit final cleanup; a
defensive snapshot destructor covers lost response ownership.

Limits are 1 GiB per file and two active preparations/transfers per facade
process, giving at most 2 GiB of snapshot payload. Disk admission conservatively
includes active reservations and a 64 MiB free-space margin. Copy and stream
chunks are at most 1 MiB; HDF5/runtime caches use additional memory. Safe failures
return JSON with 404 (unavailable/ineligible), 413 (size) or 503 (capacity/disk).

This remains a trusted lab-host application. Session IDs and opaque handles do
not establish authenticated account ownership. Multi-process deployment would
have these capacity limits separately in each process; no account system or
host-wide quota manager was added.

## E. Tests and real artifact acceptance

Focused runs pass: resolver **34**, privacy **43**, snapshots **12**, and
Web/static delivery **16** checks. Resolver fixtures exercise all five contracts,
both semantic ports, originating run identity, exact historical bindings,
conflicting authority/result/manifest metadata and unsafe manifest replacement.
Privacy/snapshot/ASGI tests cover unsupported structures, modified/missing bytes,
unsafe paths, resource limits, bounded reads, mutation during preparation,
replacement after validation and cleanup before/during streaming.

The single affected domain regression command was:

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 \
  /home/likeyi/anaconda3/envs/agent/bin/python -m pytest \
  tests/application tests/web -q -p no:cacheprovider \
  --basetemp=/tmp/agent-ua332-yy78ki0e/domain-temp
```

Result: **1,276 passed, 3 skipped, 3 warnings in 292.39 seconds**, exit 0.
This includes UA3.1 upload, UA3.2 H5AD composition and existing report/PNG
regressions. Warnings concern existing louvain/pkg_resources and TBB dependencies.
The full repository regression was not run. Afterward, the resolver's retained
and unrelated-output tests were strengthened with valid execute presentations;
the focused resolver suite again passed **34 tests**, with no production change.

Eleven native Firefox checks pass in an isolated workspace over two tiny
accepted matrix publications with deterministic stored conversation displays.
They verify bubble placement, distinct historical links, refresh, fresh server,
generation-checked navigation, omitted ineligible messages, working details,
direct browser attachment bytes, unchanged originals and no JavaScript errors.
Providers and scientific work remain zero during those native interactions.
The browser attachment SHA-256 is
`d16781fef05873bbac2562ab15aaf5eb52e5038b267d168611930ce7941b377f`.
Records and screenshot are at
`/home/likeyi/agent-ua332-browser-nM3OPY9V/native-checks.json` and `matrix-chat.png`.
These are bounded fixture/software results, with no live-provider qualification.

The original PBMC H5AD also passed actual localhost HTTP streaming through the
new route. Accepted identity came from Session `m156-real-pbmc-r1`, Revision
`e138f2b02de686b1921330f688c9e043b0dd557852d3581e94d0ce786b0f06a9`, and
originating run `m117b-pbmc1k-nextgem-r1:run`:

- 1,119 × 1,355,445; 11,535,823 nonzeros; int64 CSR.
- 191,009,521 delivered bytes.
- Accepted and delivered SHA-256:
  `5525e68a1902d5fdd465542e7a2627862c15a95b8f3e3766ca84f2126244dca9`.
- Original bytes unchanged; nine Revisions and generation 13 unchanged.
- No provider or registered scientific execution; zero active snapshots afterward.
- Client reads bounded to 1 MiB; preparation plus delivery took 49.04 seconds.

Observed combined server/client process peak RSS rose from 674,572 to 783,072
KiB. This is a host observation, not a fixed total-memory guarantee. The
historical `initial` turn has no captured chat presentation, so its chat inventory
is empty; no association was fabricated. Its exact Revision download is verified.
The real check script/result and domain log are in
`/tmp/agent-ua332-yy78ki0e/real_delivery.py`, `real-delivery.json`, and
`domain-regression.log`. No PBMC regeneration, EpiZoo inference or live provider
call was required; no dependency was installed.

## F. Architecture preservation

Scientific owners, algorithms, Interpreter, Planner, Registry inventory,
compiler semantics, Session/Revision persistence schemas and scientific
authority semantics are unchanged. No workflow engine, file manager, second
RunStore, artifact database, export/sanitization engine or scientific
verification pipeline was introduced. Downloads are deterministic reads of
accepted results and create no scientific Revision or Session generation change.

## G. Repository status

Six existing files are modified and seven implementation/test/documentation
files are new. All are unstaged. The index is clean, HEAD is unchanged, and
existing `evals/` remains untracked and untouched. Whitespace validation passes.
The implementation is ready for review; no commit or push is authorized here.

## H. Next stage

The smallest useful UA3.3.3 expansion is accepted barcode-QC metrics and
cell-selection decision tables. Reuse exact accepted authority/output binding,
opaque handles, private snapshot integrity and bounded transport, adding only
the reviewed table roles and their closed privacy/layout checks. Keep paired
artifacts, ZIPs and other H5AD classes deferred. UA3.3.3 is not implemented.

## UA3.3.3 — QC/selection delivery and scoped UA3.3 closeout

**ACCEPTED — UA3.3 scoped scientific result delivery complete.** Validated on
2026-10-09 against the same committed baseline, with the original UA3.3.2 work
preserved. This is a bounded initial scientific-file delivery milestone, not
scientific-file coverage for all 23 registered tools.

### Supported original files

| Accepted source | Manifest role | Original filename | Chat label |
| --- | --- | --- | --- |
| Five matrix contracts above | `matrix` | `matrix.h5ad` | Original accepted matrix |
| `compute_scATAC_qc` / `scatac-barcode-qc.v1` | `table` | `barcodes.tsv.gz` | Barcode QC |
| Same QC contract | `histogram` | `lengths.tsv.gz` | Fragment length distribution |
| `select_scATAC_cells` / `scatac-cell-selection.v1` | `decisions` | `decisions.tsv.gz` | Cell-selection decisions |
| Same selection contract | `selected` | `selected.tsv.gz` | Selected cells |

Existing safe report projections and supported PNGs remain available. Manifests,
receipts, private paths and unrelated sidecars are never scientific attachments.
QC observed barcodes and explicitly QC-selected candidates retain their existing
semantics; statistical cell calling remains `none/not_assessed`.

### Identity, membership and presentation

QC/selection resolution reuses the existing facade and scientific-file API. It
checks accepted step identity, schema-2 scientific authority, owner manifest
identity/profile, normalized persisted arguments, exact publication token,
owner result summary, and the manifest plus both subordinate payload records.
Every payload has its own accepted SHA-256 and size. QC/selection authority's
`result_metadata` is the existing empty mapping; no matrix-style proof fields
were invented for these owners. Inventory reads bounded metadata only, never
the compressed table bytes.

The accepted Revision's run-result digest anchors the originating request,
plan and all scientific step results. Alongside exact selected outputs, the
same successful originating run can therefore expose files from its explicitly
planned, accepted, allowlisted steps. A QC → selection → matrix request shows
all five eligible files even when its selected output is only the matrix.
Current QC/selection summary-only selections likewise retain their files.
No additional Revision output, history record, implicit scientific completion
or step-order inference is introduced.

Historical input references do not become current-request attachments. Retained
historical outputs require their exact reviewed semantic port and original
`output.run_id`; retained statistics cannot invent artifact bindings. Legacy
turns without a successful captured execute presentation keep empty chat
inventory. These existing display limitations remain explicit; no historical
message association is fabricated.

Table handles bind exact Session, Revision, originating run/step and accepted
step digest, authority, manifest, role and payload identity. Equivalent cooutputs
of one accepted source deduplicate by identical handle and complete server-record
equality. Independent steps remain distinct even when their bytes match.
The focused review caught and corrected duplicate table rows for
`manifest_path`/`manifest_sha256` aliases. Existing matrix handle identities are
unchanged. Inventory is re-anchored after metadata resolution.

Ineligible optional table sources omit their controls independently, so an
unmanaged or unavailable legacy table cannot suppress a separately eligible
matrix. Their old or guessed handles still fail closed. Changed table bytes
are detected at download time. Scientific details, evidence, report and PNG
delivery continue to work. Chat file rows use exact captured Revision URLs and
remain associated after later turns, refresh/reopen, fresh server and navigation.

### One shared preparation/streaming boundary

The existing `matrix_delivery.py` retains the single copy/hash/snapshot/capacity
implementation; its historical internal class names remain. The minimal
extension dispatches matrix eligibility or the separate closed table checker
and adds approved filename/content type to the prepared resource. There is no
second download service, worker, storage system or generalized exporter.
All file types share the same two active slots, 1 GiB per-file limit, private
temporary storage, bounded chunks, source descriptor checks and response cleanup.

`table_delivery.py` checks original owner gzip encoding: one member, no filename,
comment or extra headers, mtime zero and the reviewed level-6 header. zlib checks
CRC/trailer; concatenated members and trailing metadata fail. Compressed reads
are at most 64 KiB, decompression calls at most 4,097 bytes, and rows at most the
owner's 4,096-byte limit. Headers, column shapes, closed literals, scientific
identifier encoding and accepted row counts are checked without recalculating
QC counts, enrichment, thresholds or selection decisions. `lengths.tsv.gz` is
headerless with 1,001 existing bins; an empty `selected.tsv.gz` is valid when
the accepted selected count is zero. Unmistakable private filesystem identifiers
and unreviewed metadata are rejected; legitimate opaque identifiers survive.

Integrity and eligibility complete on the same private snapshot before HTTP
headers/body. Tables use `application/gzip`; matrix response semantics remain
unchanged. Browser anchors download originals without buffering scientific
bytes in JavaScript. No provider, Interpreter, Planner, execution/recovery or
new scientific Revision is involved. Existing trusted lab-host restrictions
apply; opaque handles are not authenticated account ownership and per-process
limits are not a host-wide quota system.

### Focused validation

The final focused Application command was:

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 \
  /home/likeyi/anaconda3/envs/agent/bin/python -m pytest \
  tests/application/test_scientific_matrix_artifacts.py \
  tests/application/test_scientific_table_artifacts.py \
  tests/application/test_matrix_delivery_privacy.py \
  tests/application/test_table_delivery_privacy.py \
  tests/application/test_matrix_delivery_stream.py \
  tests/application/test_interactive_schemas.py \
  tests/application/test_interactive_rich.py \
  -q -p no:cacheprovider \
  --basetemp=/tmp/agent-ua333-mq6gdwc1/application-final
```

Result: **272 passed in 5.44 seconds**, exit 0, no skips or warnings. This includes
34 matrix resolver, 41 table resolver, 43 HDF5 eligibility, 80 table eligibility,
22 shared snapshot, and 52 existing rich-interface/schema checks. Adversarial
cases include substituted/missing payloads, unsafe paths, closed manifest roles,
conflicting authority, scalar/cooutput bindings, gzip metadata/framing,
resource limits, source mutation and response cleanup.

Focused Web commands used the same Python/environment and pytest flags:

```bash
python -m pytest tests/web/test_scientific_artifacts.py \
  tests/web/test_scientific_artifact_static.py tests/web/test_static.py \
  -q -p no:cacheprovider
python -m pytest tests/web/test_upload_api.py \
  tests/web/test_h5ad_composition_api.py tests/web/test_uploaded_multiturn.py \
  tests/web/test_rich_api.py tests/web/test_rich_static.py \
  tests/web/test_upload_static.py -q -p no:cacheprovider \
  --basetemp=/tmp/agent-ua333-web-preserved
```

Results: **40 passed in 71.41 seconds** and **64 passed in 132.59 seconds**, no
skips or warnings. These cover original matrix delivery, all four gzip roles,
multi-step message inventory, headers and safe pre-response failures, ASGI
interruption cleanup, UA3.1 uploads, UA3.2 input composition, and existing
report/PNG/evidence/navigation behavior. The prior 1,276-test regression and
191 MB PBMC H5AD transfer were not repeated; matrix-specific eligibility is
unchanged and current tiny matrix HTTP/native checks cover shared delivery.

Seventeen native Firefox checks pass over four accepted tiny fixture revisions:
one multi-step request displays five files, individual QC and selection display
two each, and a matrix-only request shows one without historical input tables.
All five direct attachment downloads match original accepted bytes. Ten exact
historical links survive refresh, fresh facade/server and generation-checked
navigation; originals remain unchanged, no JavaScript error appears, and
providers/scientific production/owner reconstruction remain zero during native
reads, downloads and navigation. The final run reused accepted fixture artifacts
without rerunning setup science. Records are at
`/home/likeyi/agent-ua333-browser-4wvZuU5u/native-checks.json` and
`scientific-chat.png`. This is software/UI acceptance, not live language quality.

All four preserved real PBMC tables also pass actual localhost HTTP through
the same endpoints, without regenerating fragments, QC or selection:

| File | Delivered bytes | Accepted and delivered SHA-256 |
| --- | ---: | --- |
| `barcodes.tsv.gz` | 2,356,867 | `48242f05f58e3d8a727eeff56e33ae328a281776c91dc8bb187c85b0620a756a` |
| `lengths.tsv.gz` | 4,220 | `124a013c5bb3f2d688d10909cf1b90fcebf3cbaa209453a5c7e93599baa6be93` |
| `decisions.tsv.gz` | 2,584,095 | `cd5125108fd5a7653556369f415e49313ac6bf6e08b6f324dd144502886f9c88` |
| `selected.tsv.gz` | 18,272 | `a82502579d206ec77f529d939a7666f9ee7d2ece18b18c7d5976cb27f6a0ba0d` |

The accepted Session/Revision remain the PBMC identities above, with nine
Revisions and generation 13. Originals are unchanged, all temporary snapshots
are closed, and instrumented registered tool/production/owner/provider calls
are zero. Its matrix handle and accepted identity are exactly the UA3.3.2 values.
The legacy initial message still has no chat presentation and was not rewritten.
Scripts/results and the final Application log are in
`/tmp/agent-ua333-mq6gdwc1/real_tables.py`, `real-tables.json`, and
`application-final.log`. No dependency installation, GPU or live-provider
execution was needed. The full repository or full Application/Web suite was
not run in UA3.3.3.

### Preservation, review and Git finalization

The targeted UA3.3.2 review found its accepted identity, snapshot/hash, HDF5
privacy, limits/cleanup and chat placement sound. UA3.3.3 added no scientific
algorithms, Planner, Interpreter, Registry entry, compiler semantics, RunStore,
scientific authority or Session/Revision schema change. The new table checker
reuses owner format definitions rather than introducing new scientific schemas.

Against the last committed baseline, the intended combined diff contains six
modified files and ten new files. The three additions beyond UA3.3.2 are
`src/agent/application/table_delivery.py`,
`tests/application/test_scientific_table_artifacts.py`, and
`tests/application/test_table_delivery_privacy.py`; other changes extend the
previously listed facade/schema/shared snapshot/Web/tests/record files.
All 705 evals, 198 outputs, 208 real application/Web files, external operator
configuration and pretrained checkpoint preserve their inventories and exact
content/size/mtime/mode identities. No biological data, credentials, checkpoint,
temporary snapshot or unrelated file belongs in the commit.

This record captures acceptance before Git finalization. The user authorized
one coherent commit, `Complete UA3.3 accepted scientific result delivery`, and
a normal fast-forward push after independent remote-tip inspection. Final
commit/push refs are reported in the completion response. No fetch, pull,
reset or force push is part of this closeout.

### Deferrals and UA3.4 recommendation

BGZF fragments/TBI, embedding NPY/cell-ID pairs, compact analysis or annotated
H5AD, marker tables, pseudobulk/DA, evaluation JSON and adapted models remain
deferred with their authority, privacy and companion/export requirements.
Existing evidence/reports remain available for those tools. No unresolved
same-request multi-step limitation exists for the implemented eligible roles;
unsupported/unmanaged files and unbound historical messages remain ineligible.

The smallest useful UA3.4 slice is one external fragments upload/selection,
an optional explicitly paired TBI, and operator-admitted reference, namespace
and producer-profile choices through the existing fragment admission tool.
Reuse current byte admission and structured resource selection; add no filename
inference or new preprocessing engine. FASTQ library/read grouping and BAM
library/reference contexts can follow as separate input slices. UA3.4 has not
been implemented.
