# UA3.4.1 — External fragments Web upload and resource selection

Implemented on 2026-10-09; unstaged and ready for review. No commit or push.
This records the original stage checkpoint. The [UA3.4.3 closeout](ua3.4.3-fastq-web-upload.md)
supersedes its repository and Git status.

## A. Baseline and preservation

Started on `main` at `c885a77ed26e720e93cb41f0acba9ddb355688c3`.
HEAD, local `main` and local `origin/main` identify that commit. The tracked
worktree and index were clean; only existing `evals/` was untracked. No reset,
clean, fetch, pull or external configuration change occurred.

Complete SHA-256, size, mtime and mode checks against the prior accepted
inventories preserve 705 eval files, 198 scientific output files and 208 files
across the original PBMC application and existing Web workspace. The eval
inventory has no additions/removals. Operator configuration retains SHA-256
`dd86afd663fc2345b542bd78fa4bf97a5ebbfaf6d2557e84736bcddb4a6d4389`.
The 5,231,645,507-byte pretrained EpiZoo checkpoint retains SHA-256
`6b2d13fdbd54a9b0d56efa5afa81bc4832b813f4eac8662e4d93cc08c9d9b39a`.
Preservation records and logs are retained outside Git.

## B. Implementation and files

Modified:

- `src/agent/application/uploads.py`
- `src/agent/application/local_resources.py`
- `src/agent/web/app.py`
- `src/agent/web/config.py`
- `src/agent/web/static/app.js`
- `src/agent/web/static/index.html`
- `tests/web/test_config.py`
- `tests/web/test_static.py`
- `tests/web/test_rich_static.py`
- `tests/web/test_upload_static.py`
- `README.md`

Added:

- `tests/application/test_fragments_uploads.py`
- `tests/application/test_fragments_composition.py`
- `tests/web/test_external_fragments_execution.py`
- `tests/web/test_fragments_selection.py`
- This record.

The existing upload owner, approved root, staging, capacity, locks and source
catalog serve both H5AD and external fragments. Its historical class name
`H5ADUploadAdmission` remains compatible. Existing H5AD storage/registration IDs,
unpaired record serialization and binding digests are unchanged. Optional
`source_index = {path, sha256, size_bytes}` extends the existing immutable record;
no additional source store, scientific tool or upload service is introduced.

## C. Actual input contract

The unchanged public `import_scATAC_fragments` accepts consistent five/six-column
TAB-separated UTF-8/LF records: chromosome, start, end, barcode, read-pair support,
and optional strand. Bounded leading comments are supported. Plain TSV, gzip
and BGZF are recognized from bytes. Filename extensions establish no validity.
The declared `10x-atac-fragments.v1` profile preserves already adjusted,
0-based half-open intervals and corrected opaque barcode identities. The public
reference boundary retains qualified human/mouse scope. Neutral arbitrary-species
producers remain data-layer-only.

A supplied index must be a supported TBI for that exact BGZF source. Plain/gzip
sources with an index fail through the owner. Omitting the index makes no index
claim; no sidecar discovery occurs. Source and index become one selectable
registration, with a safe `has_source_index` flag only for paired records.

An explicitly designated operator InputSet (`fragments_companion: true`) supplies
the existing required `reference_bundle_path`, `reference_bundle_sha256`,
`source_profile` and `namespace`. Species/assembly are declared by that qualified
reference; there is no redundant species field or filename inference.
Optional `source_selection` accepts `full_export`, `subset_export` or `unknown`;
omission retains the scientific owner's `unknown` default. It declares export
coverage, without independently qualifying historical processing or called cells.
Source/index paths and hashes cannot appear in Web companions.

Explicit selection wins. Otherwise exactly one InputSet explicitly marked
`fragments_default: true` may be used. No default gives `FRAGMENTS_RESOURCE_REQUIRED`;
multiple defaults give `FRAGMENTS_RESOURCE_AMBIGUOUS`. There is no species-based
guess or fallback to EpiZoo resources. Existing operator configuration has not
been activated or edited to add a fragments default.

## D. User experience and normal execution

Choose **External fragments** in the composer, choose one source, optionally
attach its TBI and upload. Choose an approved **Scientific inputs and declarations**
set, then describe the requested analysis in chat. The completed source remains
available after refresh/restart. Changing the pending source clears its old index;
changing the explicit pair creates a new transfer identity.

The browser sends only opaque resource/InputSet IDs at submission. Approved
InputSets hold the typed reference, profile, namespace, export history and optional
QC/selection/matrix parameters; the browser sees safe labels and role/default
flags. Declarations and thresholds supplied only in prose are not collected by
this integration. Conversational collection remains UA3.5 work.

Registry-validated focused composition permits explicitly configured import,
QC, selection and canonical-matrix inputs. It generates no steps, bindings or
defaults. Ordinary Interpreter/Planner decisions, compiler/preflight and runtime
select and execute the operations. Unsupported typed intent remains clarification.

Missing source, reference, profile and namespace produce reviewed specific safe
errors. Scientific-owner errors retain their codes for invalid records, mismatched
references and incompatible indexes, without private paths. Changed bytes fail
existing integrity admission before new providers/science. No exception prose is
parsed into scientific findings.

Accepted results use existing Session/Revision, evidence/report and chat delivery.
The tiny downstream acceptance offers matrix H5AD and all four QC/selection tables
through unchanged UA3.3 handles. Uploaded sources and scientific fragments/TBI
outputs create no download entries.

## E. Integrity and transfer

The existing binary PUT defaults to H5AD. `input_type=external_fragments` selects
the declared upload role. A pair additionally supplies `index_filename` and a
positive `source_size` together; body bytes are source followed by index.
Content-Length, when present, counts both. Browser `Blob([source, index])` streams
file references without reading the files into JavaScript memory.

Both components are staged privately in the existing directory, hashed and
written in at most 1 MiB chunks. The configured 256 MiB default / 1 GiB maximum
applies per file; a pair has at most twice that aggregate bound. Shared concurrency
and the 128-file bound count both components. Generated names never use filenames.
Empty, interrupted, oversized or malformed transfers publish no registration.
Exclusive publication never overwrites files; a second-link failure rolls back
newly linked components. The single record publishes only after both complete.
Complete crash-orphan bytes remain undiscoverable until a matching retry finishes
registration. Registration failures retain complete bytes under the existing rule.

Accepted retries compare source label, pair presence and both byte identities.
An index filename is transient safe transport metadata; renaming it with identical
bytes does not create another scientific identity. Deleted components can only be
repopulated by an identical retry. Source and index snapshots are rechecked before
record publication and new consumption. Plan and accepted resolved arguments must
match the selected source, index and import declarations. Historical accepted
completion/retry checks metadata without reopening deleted or changed sources.

## F. Validation and limitations

Focused checks passed:

- Existing H5AD upload + HTTP boundary: 58 passed, 2.33 s.
- Source/pair transport plus existing uploads: 55 passed, 1.51 s.
- Binding/config/selection/local-resource/H5AD checks: 335 passed, 5.05 s,
  before 11 additional HTTP cases; final selection/HTTP suite: 24 passed, 1.76 s.
- New ordinary Web execution: 14 passed, 29.14 s, no warnings/skips.
- Existing UA2.2 fragments integration: 15 passed, 25 deselected, 26.60 s.
- Static UI boundaries: 21 passed, 0.02 s.

The affected-domain run covered 835 checks: 834 passed with three existing
louvain/Numba dependency warnings in 284.70 s; one static assertion still expected
the old convenience-field list. It was updated to allow only the new opaque
InputSet ID and require catalog rechecking. The final 34 affected static checks
pass in 0.04 s. No production changes followed this run and the 834 passing checks
were not unnecessarily repeated. The final static run and every focused command
above exit 0; the initial domain command exited 1 for that corrected assertion.

Reproduce the affected scope from the repository with `AGENT_PYTHON` set to the
existing agent environment's interpreter:

```sh
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 "$AGENT_PYTHON" -m pytest \
  tests/web tests/application/test_local_resources.py tests/application/test_uploads.py \
  tests/application/test_fragments_uploads.py tests/application/test_h5ad_composition.py \
  tests/application/test_fragments_composition.py tests/application/test_registered_resource_integration.py \
  tests/application/test_scientific_matrix_artifacts.py tests/application/test_matrix_delivery_privacy.py \
  tests/application/test_matrix_delivery_stream.py tests/application/test_scientific_table_artifacts.py \
  tests/application/test_table_delivery_privacy.py -q -p no:cacheprovider
```

Focused targets use the same prefix: `tests/web/test_external_fragments_execution.py`
(14 passes); `tests/application/test_fragments_uploads.py tests/application/test_uploads.py`
(55); and `tests/web/test_rich_static.py tests/web/test_static.py tests/web/test_upload_static.py
tests/web/test_scientific_artifact_static.py` (34). No full-repository regression ran.

Sixteen native Firefox checks pass. One two-record import executes actual production
and independent reconstruction once each through scripted semantic providers.
Upload, refresh/restart and result reads add no providers/science. Checks include
real paired File/XHR transfer, missing context, submitted namespace persistence
after selection changes, H5AD refresh, evidence/report and absence of JS errors.
The native `native-checks.json` and screenshot evidence are retained outside Git.

Normal execution checks exercise plain/gzip/BGZF, exact selected versus neighboring
source bytes, an explicit BGZF/TBI pair and valid but wrong-source TBI rejection.
A three-record synthetic downstream run executes import/QC/selection/matrix once
and verifies all five delivered files against accepted size/SHA. These use tiny
test-only references, explicit permissive thresholds and existing synthetic QC
qualification; they do not establish biological quality or calibrated defaults.
One initial unindexed-BGZF worker failure did not reproduce in affected/final
reruns; test-only traceback capture supports diagnosis if it recurs. No scientific
implementation was changed on that evidence.

No reviewed biological external source was found. Preserved PBMC fragments have
FASTQ production provenance; macaque needs neutral data-layer admission. The
available EpiAgent sample lacks reviewed exact external profile/assembly history.
None was relabeled, trimmed, downloaded or executed as biological external acceptance.
No live providers, GPU inference, dependency installs or reference downloads ran.

## G–I. Architecture, status and next slice

All 161 checked protected source files match the baseline, including scientific
owners, orchestration/Registry/compiler, providers, schemas, interactive facade,
Session state and UA3.3 delivery owners. Registry remains 23 with unchanged
semantics. Scientific authority and RunStore/Revision contracts are unchanged.
All changes listed above remain unstaged; the index is empty. Baseline refs and
preserved data remain unchanged. UA3.4.1 is ready for review; no commit or push.

The smallest UA3.4.2 slice is one explicitly declared corrected-CB BAM upload
through this same transport/catalog, with operator-approved BAM library/reference
context and the existing public inspection/preparation binding. Inspect actual
index requirements first and support only the existing qualified producer contract.
Reuse normal planning and safe prerequisite errors; defer arbitrary chemistries,
multi-library grouping and neutral producer exposure. BAM, FASTQ grouping and
arbitrary companion/reference upload are not implemented here.
