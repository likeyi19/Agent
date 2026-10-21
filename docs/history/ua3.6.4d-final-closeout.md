# UA3.6.4d — Native defaults and accepted model delivery closeout

Date: 2026-10-14. Implementation is unstaged; no commit or push. This final
bounded follow-up addresses native default omission and model download. The
[preceding post-training record](ua3.6.4d-posttraining-workflow-acceptance.md),
[Work A](ua3.6.4d-reference-supplementation-acceptance.md), and earlier workflow
report remain unchanged historical evidence. No new training or live-provider
request was performed.

## Baseline and evidence

Starting branch `main`, HEAD `ce5b53d7f716b777fc5333ea84e4d868ccf0da71`.
All inherited uncommitted UA3.6.4d work was preserved. Before editing, 1,649
repository files were hashed and the inherited diff captured. Evidence for this
follow-up is outside Git at:

`/home/likeyi/program/agent-acceptance/ua3.6.4d-final` (**C**).

The preceding genuine result remains under
`/home/likeyi/program/agent-acceptance/ua3.6.4d-posttraining` (**B**).
C contains the new baseline, test logs, isolated read-only server and native
Firefox harness, downloads, screenshots and preservation records. It does not
replace B's configuration, provider ledger, RunStore, reports or checkpoints.

## Native defaults, explicit overrides and resource admission

`native_posttraining.resolve` instantiates the pinned native
`EpiZooXPostTrainConfig`. Omitted `max_steps` now retains **500,000**; omitted
learning rate retains **5e-5**. Omitted training batch uses the existing qualified
loader default **4**. Explicit steps, learning rate or batch replace only that
field. The supported batch range remains 1–4; invalid numbers/types remain
rejected. There is no mandatory step-count clarification or automatic replacement
with ten steps. Other trainer fields come from the native dataclass.

The existing native-v2 method identity is retained. Explicit configurations that
were already valid keep their exact resolved profile and method hash. The
original method descriptor's admission label is preserved as historical identity
metadata; it is no longer the active omission rule. `requested_overrides` records
`null` for omitted values, while `trainer` records effective values. Thus omission
and an explicit equal value remain distinguishable in the execution arguments,
manifest, receipt and authority. Historical M14 fixed profiles are unchanged.

The pinned adapter files are part of qualified source/SEAM bundles. Changing this
adapter therefore requires the operator to requalify those existing resources
with the existing owner before a new production execution on this code. This
follow-up does not rewrite old bundles, migrate authority or change production
configuration. Delivery of an already accepted checkpoint does not require a new
training-bundle qualification.

The existing scientific entry checked CUDA availability but had no wall-clock or
monetary training quota. A small native admission check now uses the native model
on the **meta device**, without allocating weights, to calculate necessary model,
gradient/Adam-state and retained-checkpoint storage bounds. It checks free
`cuda:0` capacity and free disk before dispatching the adaptation owner. Failure
has a safe actionable `EPIZOO_TRAINING_CAPACITY_UNAVAILABLE` message and requires
user/operator action; settings are never reduced or retried automatically. There
is no new scheduler, budget database or universal confirmation requirement.

A running native adaptation step shows its resolved attempts, learning rate,
batch and these lower bounds in the existing step-status area. This is display
information from exact resolved inputs, not a new execution authority. Missing
or invalid pinned metadata does not produce a fabricated estimate. Completed
results retain the existing accepted effective-configuration evidence/report.

For the accepted 17,922-feature architecture, the default schedule's estimates
are **4,937,594,808 bytes** for model/optimizer state and **8,840,871,340 bytes**
for checkpoint storage under native retention, including bounded serialization
allowance. Activations and preprocessing need additional memory/storage. Runtime
and monetary cost are uncalibrated; these numbers are necessary lower bounds,
not proof that every workload fits 24 GiB or that 500,000 steps will converge.
The admission tests simulate both available and insufficient CUDA/disk capacity.
The controlled normal-planning test reaches the native owner with 500,000 steps
and stops before scientific execution; **500,000 steps were not run**.

## Accepted model inventory and safe delivery

Two entries are added to the existing scientific-result inventory:

- **EpiZooX model checkpoint** — `epizoox-checkpoint.pth`, the original accepted
  state-dict bytes.
- **Model configuration and reuse requirements** — `epizoox-model.json`, a closed
  projection of accepted metadata, not the private publication manifest.

The application resolves an exact successful `adapt_epizoo_species` execution
through the requesting Session's accepted Revision/current-run or explicitly
retained output. It checks schema-2 scientific authority, accepted step identity,
publication location, manifest, receipt, method, result and pinned file records.
No browser-supplied path, species lookup or global model catalog grants access.
Unretained historical dependencies are not new model attachments. Model artifacts
remain private task outputs within the existing operator-managed access boundary;
this is not authenticated multi-tenant isolation.

Inventory reads bounded metadata rather than hashing a large checkpoint on every
poll. Download preparation opens all source path components without symlinks,
checks the accepted size, copies and hashes **1 MiB chunks** into an anonymous
private temporary snapshot, checks source stability and SHA-256, and only then
allows HTTP headers/bytes. It never calls `torch.load` or a scientific verifier.
The existing HTTP response closes the snapshot/reservation on success, exceptions
and disconnect. There is no permanent server-side checkpoint copy.

The actual checkpoint is **1,262,255,722 bytes** (about 1.18 GiB), exceeding the old
1 GiB bound. Only the exact accepted checkpoint role gains a **2 GiB** ceiling,
with headroom for this qualified architecture. Existing matrix/table/annotation/
analysis limits remain **1 GiB**. Model metadata input/projection is bounded to
8 MiB. All roles share the existing **two-transfer** capacity and disk reservations
plus a 64 MiB free-space margin. Oversized checkpoints return HTTP 413; insufficient
transfer/disk capacity returns the existing bounded busy response. This is not an
unbounded upload or download increase.

Metadata includes species/assembly, model and method identities, architecture,
fully resolved trainer settings, selected execution settings, source/SEAM identity,
ordered feature identity, training cell count, aggregate attempts/updates, numerical
runtime and exact hashes/sizes of the required missing inference companions. It
excludes absolute paths, RunStore coordinates, cell IDs, matrices, corpus tokens,
training logs and arbitrary provenance text. Path-like declarations fail closed.
Source-manifest and delivered-projection digests remain explicitly distinct in
HTTP headers, consistent with existing derivative exports.

The pinned inference consumer needs `features.txt` in exact order and
`document_frequency.npy`, the training cell count for TF-IDF, the native EpiZooX
architecture/strict loader and compatible code/dependencies, and an eligible
query matrix. The checkpoint includes learned model state but **these downloads
are not a standalone inference bundle**. The two corpus/vocabulary companions
are not exported here; obtain the exact matching files from the model owner.
The JSON lists their identities and the preprocessing/inference requirements.
No new packaging service or general model publication was introduced.

## Genuine native Firefox acceptance and recovery

The existing accepted scientific result was used unchanged:

- Session `1240e849-68a2-4a75-8a4b-d9d015a65650`.
- Revision `0a7e5d9feebfa0f60d4decd7baf68ddc91bd7de13a768c8cdc8b641765f7ca87`, generation 1.
- Model manifest SHA-256 `49aaac55da9189ce55039f21dcfbb214991005af786a4ec3672c0ee5a6ccbf77`.
- Checkpoint SHA-256 `716b80088fb457559401b7a7dde3711439e5a843a106b31ca32e9781ab9e5777`.

Native Firefox opened this Session on isolated port 8900, expanded the existing
scientific-results area and used native WebDriver element clicks on the visible
checkpoint and metadata **Download** controls. Firefox's ordinary download manager
saved the actual files. The checkpoint's exact size and streaming SHA-256 matched
the accepted identity. Metadata is 3,253 bytes, SHA-256
`63fcc393aa396c8d397b719587a8af323fe9638f2f42eef0c6d31774135e4776`.
The controls label model, compact analysis H5AD, matrix and graph separately and
warn that the model downloads are not standalone. Evidence:
`C/native-download.json`, `C/native-model-download.png` and `C/downloads/`.

An earlier harness check programmatically clicked the anchor while the outer
scientific panel was collapsed. Its preliminary logs/screenshot are retained;
the final visible native-control check above supersedes that interaction claim.
No scientific computation or model selection was repeated. Redundant preliminary
client download copies were removed after exact agreement; the final downloads
remain in C.

A fresh server process reopened the same Session and identical inventory, then
streamed the entire checkpoint again and checked its size/digest. Six conversation
texts, generation 1 and the current Revision remained correct. All **102 B
workspace files** retained identical hashes. Provider creation, training, SEAM,
mapping, corpus reconstruction, target verification and checkpoint deserialization
were guarded against execution and remained **zero**. Transfer count returned to
zero. `C/restart-result.json` records the result. The server's observed peak RSS
was below 1 GiB including scientific imports, below the checkpoint size; bounded
copy tests independently establish the chunk-read limit.

The prior `APP_EVIDENCE_FAILED` and subsequent recovery remain visible in history.
No old turn or audit record was erased, and no evidence recovery or science was
repeated here. The prior live source-bundle planning failure is still historical
fact and a UA3.6.5 priority. No new provider call, spending or credential/ledger
change occurred in this follow-up.

## Validation and preservation

Tests separate simulated software checks from genuine accepted-checkpoint
delivery. Neither set launches new training.

The completed affected selection ran without `RUN_*` gates or exclusions:
**695 passed, 3 skipped, 2 failed, 3 warnings in 317.40 s**. All **525 frozen
source/test/static files** stayed unchanged during that run. The exact command
and inventory are in `C/focused-final-result.json`.

One failure identified a compatibility issue in the new display field: ordinary
steps serialized an unnecessary `training_notice: null`. The field now uses the
existing `omit_none` mechanism and is accepted only for a running adaptation
step. Ordinary/historical step serialization remains unchanged; a new scoped
serialization test covers this behavior.

The other failure is
`test_interactive_rich.py::test_existing_supported_detail_owner_is_reused_without_science[False-annotation_rationale-3]`.
Its annotation fixture lacks `manifest['arguments']`, which the existing detail
reader requires to check the scientific profile. It reproduces on the isolated
committed HEAD `ce5b53d7f716b777fc5333ea84e4d868ccf0da71` in **1.83 s**.
The production annotation owner, detail reader and fixture are unchanged here.
No unrelated MAESTRO fix or weakened assertion was introduced. Baseline and
current isolated logs are preserved; this is not an all-green regression claim.

After the display compatibility fix, the supplementary schema/static-client,
embedding and neighbor-publication selection records **115 passed, 1 inherited
failure, 3 warnings in 9.19 s**. The same annotation case accounts for that failure.
Final model/default/transport/policy checks are recorded separately below.

On the final production code, the native/default, model-delivery, interactive
schema, HTTP artifact and error-policy selection passes **119 tests, 2 warnings
in 163.31 s**, exit 0 (`C/final-contract-tests.log`). The new display field is
omitted on ordinary steps and scoped to running adaptation; its compatibility
failure is resolved. No source/test edits occurred during that selection.
A subsequent explicit child-Revision test passes **1 test in 1.60 s**: a later
inspection that does not retain the model offers no model download, rejects the
old handle in the child context, and leaves the parent's accepted download valid
(`C/model-lineage-r3.log`). This uses simulated model bytes, not new training.

New tests use explicitly simulated model bytes with the real application,
publication, authority, Session and HTTP path. They cover valid/fresh-process
identity, tampering/missing/symlink/manifest/receipt failures, wrong Session,
Revision and output, failed/cancelled/running executions, missing authority,
private metadata, role and size limits, disk/capacity failures, bounded reads,
HTTP disconnects before headers and during body delivery, native default/partial
override resolution and resource admission. They are software acceptance, not
new scientific qualification. Genuine 1.26 GB delivery is separately demonstrated
above. Early test-harness setup mistakes and their failed logs are retained; they
do not count as passing acceptance. No full repository regression was run.

## Final change inventory

`C/final-preservation.json` records unchanged hashes for the other **1,638**
captured repository files, including all **981 evaluation files**, all **35 Work A
workspace files**, and **14 selected historical scientific/resource files**.
All **416 B files** and **436 preceding UA3.6.4d acceptance files** retain their
original size/mtime. The fresh-process read additionally preserves all **102 B
workspace byte hashes**. Earlier reports, operator configuration and provider
ledgers are unchanged. This is a bounded preservation audit, not a new byte audit
of every historical checkpoint. EpiZoo is clean at
`029cd631d0f9806a646c4a3b42ce10b958f2b67f`; Agent's `tools/models/epizoo.py` is
unchanged. The index is empty, branch/HEAD unchanged, and `git diff --check` passes.
The isolated server and Firefox are stopped; no GPU compute process remains.

Relative to C's captured starting tree, these existing files changed:

- `src/agent/tools/models/native_posttraining.py` — native omission, bounded
  resource estimates/admission and running-step explanation; sequence extraction
  and scientific algorithms are unchanged.
- `src/agent/tools/models/species_adaptation.py` — invoke native capacity admission
  before the scientific owner.
- `src/agent/orchestration/species_adaptation_registry.py` — accurate optional
  parameter description and classified resource error.
- `src/agent/orchestration/error_policy.py` — actionable safe resource message.
- `src/agent/application/interactive.py` — accepted model inventory and optional
  running native-training notice.
- `src/agent/application/interactive_schemas.py` — closed model descriptors and
  backward-compatible optional notice.
- `src/agent/application/matrix_delivery.py` — role-specific size policy and
  bounded metadata projection through existing snapshots.
- `src/agent/web/app.py` — checkpoint-specific HTTP 413 mapping.
- `src/agent/web/static/app.js` — existing scientific download labels/portability
  note and native step notice; no layout redesign.
- `tests/epizoo_adaptation/test_native_configuration.py` — default/override,
  estimate, resource-policy and notice checks.
- `tests/application/test_interactive_schemas.py` — optional-field compatibility.

Added files:

- `src/agent/application/model_delivery.py` — accepted authority/receipt selection
  and closed companion metadata projection, without training reconstruction.
- `tests/species_adaptation_integration/test_model_delivery.py` — delivery and
  controlled omitted-default integration checks.
- This report.

## Scope conclusion

| Scope | Assessment |
| --- | --- |
| Native defaults and explicit partial overrides | **PASS**, deterministic and controlled integration; no default-length training |
| Necessary capacity checks and effective-configuration display | **PASS** for stated lower bounds and existing local execution policy; runtime/full activation feasibility is not calibrated |
| Accepted checkpoint, safe metadata, role-specific large streaming and native Firefox download | **PASS** |
| Fresh-process accepted-result reads and delivery without science | **PASS** |
| Relevant regressions | **PARTIAL suite record** due to the reproduced inherited annotation fixture failure; new display serialization issue corrected |
| Standalone portable inference bundle | **NOT CLAIMED**; exact vocabulary and corpus-DF companions are listed but not exported |
| Continuous live-model workflow | **Still unqualified**, unchanged prior failure; UA3.6.5 priority |
| Biological convergence, new species qualification, broader H5AD conversion | **NOT RUN / outside this completion task** |

Both authorized functional changes are complete within this bounded scope.
UA3.6.4d is ready for bounded technical closeout and review, with the prior genuine
short training result preserved. The broader live scientific user workflow
retains its documented partial qualification. No ten-step result is recast as
biological convergence, and no default 500,000-step job was launched. Production
configuration, historical evidence, credentials and provider ledgers are not
changed. No commit or push is authorized or performed.
