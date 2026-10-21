# Post-M11.7b.3c — Preserved real-run qualification and closeout

Status: **COMPLETE**. The preserved run is explicitly qualified, application
closeout succeeds, and the single frozen full lightweight regression passes.

## Baseline and preserved identity

Started on clean `main`, aligned with `origin/main`, at
`bcb887943dde5dfb6b1e224c0aa1e076b1082366` (`Complete Post-M11.7b.3b legacy authority qualification`).
No source/test changes, scientific production, commit or push were performed.

The operator execution and safe-stop records identify:

- Root: `/home/likeyi/program/agent-acceptance/m11.7/pbmc-1k-nextgem-r1`.
- Run: `m117b-pbmc1k-nextgem-r1:run`.
- Workspace digest: `310a1e8c5448beb4d5b84f9c80cf361d66beebed4b9aa0d176f908c7efbaaf9c`.
- State: `application/run_state/<workspace-digest>.json`, schema 3,
  `SUCCEEDED`, revision 14, updated `2026-09-11T15:58:15.016734+00:00`.
- Plan: `m117b-pbmc1k-nextgem-r1-plan`; fingerprint
  `9f328f0d1593edb37bdea0f256151df88a509672a7148d99a70e51d63c39ade2`.
- Five successful single-attempt steps: intake → FASTQ fragments → QC →
  explicit selection → matrix (also directly dependent on fragments).

The authority foundation docs refer to the preserved M11.7b-2 run; its actual
paths and observations are retained in the operator records, particularly
`qualification/execution-procedure.py`, `execution-measurements.json` and
`verification-pass-audit-safe-stop-20260911T223349Z.json` under that root.
The historical application was interrupted during evidence verification after
scientific terminal success; it never finished a report.

## Read-only preflight

The state SHA-256 was exactly
`843f0f080ff5728b5670dad8d4101cb7ac2eb61267659e8b40b3f70b6923b866`,
matching the historical safe-stop assessment. Existing evidence SHA-256 was
`4a4f20f3ba6f98b185178ebeff7dbd0146a95acee78a97cc1affdd5624cc8671`,
also matching that assessment. All eight recorded FASTQ paths existed with
recorded file sizes. None of fragments/QC/selection/matrix had authority metadata;
there was no qualification sidecar. Scratch was empty.

`/tmp/agent-post-m117b3c/preflight.json` records the complete original state,
source availability and SHA-256/size inventory of 820 existing non-raw-path files
(15,093,683,925 bytes, including retained acquisition material). This is physical
hashing, not independent scientific reconstruction. All five scientific manifest
hashes matched the persisted results. The snapshot includes every scientific
payload, manifest, profile, producer record and receipt.

| Owner | Manifest SHA-256 |
| --- | --- |
| Fragments | `7360b733ad8d71e29b84d71310cd886365bcffaa49b80f6afc7c4641c59fc404` |
| QC | `951a63ae7caf3a98b9b0a9adde6486ba7a9c23ad27f69d2c8145399681e17b50` |
| Selection | `ecb6d267248bf7568ea1ba6dcbfc2078e5bd5c7ee896580cda1836b94eaa0fd7` |
| Matrix | `739a8fc2da6344160eb38ba0a3770422e67c0008fe00ddc05abd912e8d70d99e` |

The accepted result contains 20,276,156 fragment records (support 45,521,966),
263,747 observed barcodes, 1,119 explicitly selected candidates, and an int64 CSR
matrix of 1,119 × 1,355,445, with 11,535,823 nonzeros and total count 15,114,730.
Matrix file SHA-256 is
`5525e68a1902d5fdd465542e7a2627862c15a95b8f3e3766ca84f2126244dca9`;
logical matrix identity is
`db74d29f9695925e6fb18be0a9bef77ab126093003882de878681f076c40c0ce`.
These are existing results, not newly generated science or biological validation.

## Explicit qualification and presentation handling

The operator harness invokes the accepted public API:

```python
qualify_legacy_authorities(
    FileRunStore(root / 'application/run_state'),
    'm117b-pbmc1k-nextgem-r1:run',
)
```

It uses the original recorded production runtime/resource environment, disables
all `RUN_*` gates, blocks `PlanExecutor.execute`, and counts actual independent
owner callbacks through `VerificationContext.verify`. No authority is manually
constructed. Scientific owner and integrity algorithms remain unchanged.

Existing evidence contains the retired `fresh_independent_*` labels. The user
explicitly authorized byte-preserving archival of this evidence and rebuilding
presentation through the existing application, without changing scientific
payloads or execution/publication records. The archive is
`qualification/post-m11.7b.3c/historical-analysis_evidence.json`. This is an
explicit operator action, not automatic evidence migration or changed matching
semantics. Normal application behavior first rejected the old evidence, then regenerated
and verified presentation after explicit archival, as recorded below.

Qualification passed in **2,087.51 seconds**, exit 0. The actual owner callback
counts were FASTQ 1, canonical-v2 fragments 1, QC 1, Selection 1, Matrix 1;
`PlanExecutor.execute` attempts were 0. Verifier entry counts were respectively
2, 3, 2, 2, 1: covered dependency calls reused proof without invoking their
independent bodies. This is an observed trust-boundary result, not an arbitrary
call-count optimization target or a throughput qualification.

All four records have schema 2, `scope=scientific_correctness.v1`,
`integrity_scope=artifact_integrity_lineage_historical_sources.v1`, and successful
completion. Each uses its existing independent verifier with compatibility
version 1. FASTQ retains `fastq_source_and_producer_record.v1`,
`kind=fastq_fragment_production` and `profile_id=chromap-atac-agent-support-v1`.
QC/Selection/Matrix have empty producer-qualification mappings; they consume the
common v2 boundary with exact qualified dependency lineage. Generic integrity
does not become a substitute for FASTQ qualification.

| Owner | Qualified authority identity SHA-256 |
| --- | --- |
| Fragments | `75f9d1680b4e85da834a39d26a0d638153006348d87aea4c7ea28d327ceaed96` |
| QC | `1359758b4e166674049fc623ca0504e56c39ef3c78cbc5d1a970dc3f0ba58240` |
| Selection | `13d1fd05c5dc53d5eea6175b34337472745dbbfa8e04604f539ed0dedd67113c` |
| Matrix | `6999b907cae91e56f1788bb42cf9403bdf99ee9e08b54dd7e7b35fe24ed8167f` |

The normal loader validated exact execution, arguments, receipt, dependency,
resource, profile and verifier bindings before the API atomically published
`application/run_state/<workspace-digest>.authority.json`. It binds the original
state, not a rewritten terminal record. The preserved human reference manifest
SHA is `20681078b31446df31c689992191d842d4110435004a5e547efea9696b59d7c4`;
QC manifest SHA is `e9037c7a80964495e5b3aaea4aa6d634035d05cf7aaf5a36d8cff27bb21afbd5`.
The operator environment retains its exact catalog, BEDTools and qualified
Chromap/index paths; no runtime setting was substituted.

Immediately after qualification, all **820 pre-existing files** reproduced their
preflight SHA-256 and size. The **only addition** was the authority sidecar.
Fragments, QC, Selection, Matrix, all receipts, terminal state and step results
were byte-identical. The comparison performed no scientific reconstruction.

## Domain regression

No source/test change required focused tests. The relevant domain suite ran once:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
NUMBA_CACHE_DIR=/tmp/agent-numba-cache MPLCONFIGDIR=/tmp/agent-matplotlib-cache \
conda run --no-capture-output -n agent python -c \
'import os,sys,pytest; [os.environ.pop(k) for k in tuple(os.environ) if k.startswith("RUN_")]; sys.exit(pytest.main(sys.argv[1:]))' \
-q tests/authority_dag tests/chromap/test_authority_dag.py \
tests/chromap/test_verification_authority.py tests/application tests/report \
tests/unit/orchestration tests/integration/test_milestone7_application.py \
tests/integration/test_milestone7_analysis_evidence.py \
tests/integration/test_milestone7_analysis_report.py \
tests/integration/test_milestone7_analysis_visualization.py \
tests/raw_intake/reporting tests/chromap/test_fragments_evidence.py \
tests/chromap/test_fragments_report.py tests/chromap/test_fragments_application.py \
tests/chromap/test_fragments_orchestration.py tests/bam_fragments/test_integration.py \
tests/external_fragments/test_integration.py tests/barcode_qc/test_integration.py \
tests/cell_selection/test_integration.py tests/matrix_integration \
> /tmp/agent-post-m117b3c/domain.log 2>&1
```

Result: **1,468 passed, 3 skipped, 3 existing warnings, 1,241.21 seconds, exit 0**.
Warnings are the existing pkg_resources/Louvain deprecations and unavailable TBB
threading-layer version. Strict standalone/recovery, authority compatibility,
corruption, qualification atomicity and application reuse tests remain intact.

## Application reuse and terminal lifecycle

The unchanged application path was exercised with the accepted .3a authority
scope. Instrumentation used the existing tests' `sys.setprofile` approach to
count actual undecorated scientific-owner bodies, including any strict call
outside a context. Presentation function bodies were counted separately.
Production execution was guarded throughout.

```text
ResearchAgentApplication.resume
  → terminal AgentRuntime.resume (persisted result; no recovery/production)
  → composition lease + accepted_authorities
  → artifact/receipt/resource/execution/dependency/verifier compatibility
  → evidence build/verification → visualization discovery → report/verification
  → final authority-anchor validation
```

| Operation | Result | Deep scientific owner calls | Presentation calls |
| --- | --- | --- | --- |
| Original evidence at its preserved path | `FAILED / APP_EVIDENCE_FAILED`; runtime `SUCCEEDED` | 0 | evidence verification 1 |
| Explicitly authorized archival, then normal resume | `SUCCEEDED`, 29.00 s | 0 | evidence build 1, verification 6; visualization discovery 1; report build 1, verification 2 |
| Repeat terminal resume of new presentation | `SUCCEEDED`, 20.56 s | 0 | evidence verification 4; visualization discovery 1; report verification 2 |

No verifier matching was weakened to accept retired evidence labels. The old
22,625-byte evidence was moved under the existing composition lease to
`qualification/post-m11.7b.3c/historical-analysis_evidence.json`, retaining its
exact historical SHA-256. Existing application behavior then produced the new
presentation. No scientific artifact or immutable execution/publication record
was archived, regenerated or rewritten.

The two successful application results were identical. The new evidence, report
manifest and report Markdown were byte-identical across repeat resume. No
visualization bundle was required or created. Presentation SHA-256 identities:

- Evidence: `76de83952ec2f384744e24b5bd3c71d65f2cd300f2fbed9bc2503a59c26eb3cc`.
- Report manifest: `0f26ed65039e329bb2ce5e0df8215bce9dca1a2597734b4cc1b5fe5f73e54a77`.
- Report Markdown: `4aaa43239a036a092b03439bb48d611b7f52f5f8edba45a3b5863e69da2b260c`.
- Authority sidecar: `c72d7ef875ec173391d99c86ace4bafd8d82a30d347d30d1351ae6ad824a7437`.

A second comparison covered all 57 pre-existing application/qualification files
(420,267,576 bytes), mapping old evidence to its archive. Every hash and size
matched. The original state remained revision 14 with its historical SHA-256;
all scientific results and receipt bytes remained unchanged. The authority
sidecar did not change during application consumption. Eight existing scientific,
state, execution and composition locks were independently acquired nonblocking
and released after completion. Scratch contained no files; no active private
publication or presentation staging remained. Lock files remain as normal durable
coordination files, not held leases.

No pending or interrupted scientific execution was fabricated to test recovery.
Terminal resume used the existing immutable-return branch. Standalone/recovery
strictness is established by the unchanged implementation and passing domain
fixtures, including deep reconstruction and fail-closed corruption cases; the
real run was not put through another standalone deep audit for reassurance.

## Operator procedures and retained records

Commands were run with the same `PYTHONDONTWRITEBYTECODE`, `PYTHONPATH`, Numba and
Matplotlib settings as the regression command. The qualification and presentation
scripts restore the exact operator environment from the preserved
`qualification/execution-measurements.json`; no historical procedure was rerun.

```bash
conda run --no-capture-output -n agent python /tmp/agent-post-m117b3c/preflight.py
conda run --no-capture-output -n agent python /tmp/agent-post-m117b3c/qualify.py
conda run --no-capture-output -n agent python /tmp/agent-post-m117b3c/invariance.py qualification
conda run --no-capture-output -n agent python /tmp/agent-post-m117b3c/presentation.py
conda run --no-capture-output -n agent python /tmp/agent-post-m117b3c/invariance.py presentation
```

The initial preflight helper attempt used the host's older Python, which lacks
`hashlib.file_digest`; it failed before writing a snapshot or modifying the run.
Running it in the accepted `agent` environment succeeded. Qualification and
presentation then each ran once; no scientific verification retry was needed.

The operator harnesses and detailed JSON/log records are retained outside Git at
`qualification/post-m11.7b.3c/` under the preserved run root. These include the
preflight inventory, qualification authority/counter result, both invariance
comparisons, application/lifecycle results, domain/full logs, frozen source/test
hashes and final freeze audit. `retention-index.json` records their hashes/sizes.
They are acceptance observations, not new Agent APIs or manually issued authority.

## Final lightweight freeze

After successful qualification, invariance, application/lifecycle checks and the
single domain regression, `git diff --check` and `git diff --exit-code -- src tests`
passed. All 275 tracked source/test files were hashed at implementation freeze,
with HEAD unchanged at `bcb887943dde5dfb6b1e224c0aa1e076b1082366`.
Exactly one fresh full lightweight regression was then started:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
NUMBA_CACHE_DIR=/tmp/agent-numba-cache MPLCONFIGDIR=/tmp/agent-matplotlib-cache \
conda run --no-capture-output -n agent python -c \
'import os,sys,pytest; [os.environ.pop(k) for k in tuple(os.environ) if k.startswith("RUN_")]; sys.exit(pytest.main(sys.argv[1:]))' \
-q tests > /tmp/agent-post-m117b3c/full-lightweight.log 2>&1
```

Result: **3,752 passed, 83 skipped, 7 existing warnings in 1,950.37 seconds
(32m 30s), exit 0**. All normal `RUN_*` gates were disabled, with no exclusions.
This was the only full regression invocation. The warnings are the existing
third-party deprecation/threading warnings and deliberate nonunique-observation
fixtures. All **275 source/test hashes** and HEAD matched the freeze after the
pass. No source/test changes or additional tests followed; only documentation and
retention of acceptance records were finalized.

## Deferred and review scope

Non-blocking hardening remains deferred: physical hashing/I/O performance work,
broader historical-schema migration support, and stronger hostile-filesystem
protection outside the existing trusted local operator boundary. None is needed
for this preserved-run acceptance; none was implemented. This closeout does not
extend biological validation, cell calling or foundation-model readiness claims.

Repository changes are documentation only: this record plus current README and
AGENTS pointers. No scientific data, checkpoint, operator sidecar or runtime
resource is added to Git. No commit or push was performed.

Final review state: `main`, aligned with `origin/main`; HEAD remains `bcb8879`.
`git diff --check` passed (empty output, exit 0). Nothing was staged.

```text
 M AGENTS.md
 M README.md
?? docs/post-m11.7b.3c-preserved-real-run-closeout.md

 AGENTS.md | 24 +++++++++++++++++++++---
 README.md | 20 ++++++++++++++------
 2 files changed, 35 insertions(+), 9 deletions(-)
```

The stat covers the two tracked documentation edits; this new closeout record is
untracked and therefore additional to that stat. No commit or push was made.
