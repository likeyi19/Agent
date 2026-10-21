# Post-M11.7b.3b — Explicit legacy authority qualification

Baseline: clean, pushed `main` at
`ea15bff5df74a53e65469a5694bb51c3c0b09a04`.
This extends the existing [scientific authority machinery](post-m11.7b.2-scientific-authority-propagation.md)
and [application reuse boundary](post-m11.7b.3a-application-authority-reuse.md).
The preserved real M11.7b run is not accessed or qualified here.

## Audit and bounded gap

The current owner verifiers already provide every needed independent proof:
`verify_fragments`, `verify_bam_fragments`, `verify_external_fragments`,
`verify_barcode_qc`, `verify_cell_selection`, and `verify_cell_by_ccre`. Canonical
fragment verification remains a separate scope beneath qualified producers.
`verify_step(..., authority_execution_identity=...)` already uses those verifiers
and `scientific_authority.issue`; issuance requires a successful operation-local
proof and validates exact publication/receipt identity.

The missing capability was an explicit way to invoke that path against an old
accepted run and persist newly qualified authority. Existing authority was attached
to successful step verification during durable execution. Normal loading never
issued persistent authority, and terminal execution state correctly rejected updates.
No new scientific verifier, algorithm, producer contract or authority family is needed.

## Supported legacy boundary and API

```python
from agent.orchestration import FileRunStore
from agent.orchestration.verification_authority import qualify_legacy_authorities

authorities = qualify_legacy_authorities(FileRunStore('/operator/run_state'), run_id)
# dict[step_id, VerifiedArtifactAuthority], using the existing schema version 2
```

This is one explicit operator Python API, not a Planner tool, CLI migration command,
or automatic loader fallback. It qualifies the supported scientific steps in the
selected run as one operation; there is no partial-success publication.

Here, **legacy** means a successful persisted schema-3 raw-processing run whose
scientific steps lack schema-2 authority, including supported FASTQ authority-v1
records. The existing scientific artifact schemas, publication receipts, ordered
inputs, execution identity and recovery-policy snapshot must still satisfy the
current contracts. Qualification does not migrate old run-state schemas or retired
fragments-v1 payloads. Missing recovery provenance, incompatible policies, unsupported
artifact schemas and non-successful runs fail closed. Runs may contain the existing
`inspect_raw_scATAC` predecessor; unrelated scientific workflows are outside this API.

The implementation follows the persisted plan's stable topological order. It uses
the existing executor argument resolver for exact `StepOutputRef`/literal bindings,
`verify_run` for run consistency, and `verify_step` for each step's defined checks.
No workflow completion, source inference or new dependency-order heuristic is added.

```text
explicit qualification + execution lease
  → load unchanged successful run and validate current recovery provenance
  → validate any already-recorded authority with the existing loader
  → persisted topological steps and exact resolved arguments
    → reuse compatible current authority, if present
    → otherwise register the ORIGINAL execution identity in VerificationContext
      → strict owner verification of EXISTING outputs
      → existing authority issuance and exact publication/receipt checks
      → pending schema-2 authority available to later verified dependencies
  → validate the complete candidate set through the normal accepted loader
  → recheck accepted anchors and cancellation
  → atomically publish qualification sidecar under the existing state lock
  → ordinary accepted_authorities / load_fragment_authority / application reuse
```

For the usual complete legacy DAG, the producer, canonical fragments, QC, Selection
and Matrix independent owner verifiers each run once. A dependency without its own
recorded authority takes the existing independent deep path; that does not manufacture
a durable execution anchor for an unrecorded upstream artifact. A missing or
incompatible dependency cannot be silently aligned, rewritten or guessed. Existing
current authority is never rewritten to fit different newly proposed lineage.

## Persistence and atomicity

The only new stored object is `<hashed-run-id>.authority.json` beside the existing
run state. It is a checksummed, closed version-1 storage envelope containing the run
ID, canonical original-state SHA-256 and a step-ID mapping of the **existing**
`VerifiedArtifactAuthority` schema-2 records. It is not a second authority model.
The extra file is necessary to preserve terminal-state immutability: no historical
step verification, trace, plan, timestamps, scientific manifest, receipt, source,
QC table, selection decision or matrix is rewritten.

FileRunStore reuses its existing fsynced temporary-file/atomic-replace writer and
state lock. The explicit operation holds the existing execution lease. The entire
qualification is verified before publication; partial owner success remains local.
Caught publication failures remove the new sidecar while readers are still excluded
by the state lock. Cancellation is checked between owners and before the atomic store transaction,
using the existing cooperative cancellation scope. Callbacks execute outside the
state lock so they can safely read RunStore cancellation state. No new lifecycle or
cancellation subsystem is added. A process death around atomic publication can leave
no sidecar or the complete, already-verified sidecar; no partially qualified mapping
is published. Stronger hostile-filesystem or storage-failure guarantees are not claimed.

Normal authority loading is read-only. It checks the sidecar envelope and original
execution-state binding, then uses the existing exact authority validation, including
manifest/payload/sidecar/resource bytes, receipt/execution/argument identity, dependency
lineage, scientific profile, producer scope and verifier compatibility. Malformed,
corrupt or incompatible existing authority is an error, never a reason to issue new
trust. A qualified v1 authority is additive; the original v1 record remains intact.

Repeated qualification validates and reuses compatible current authority without
reconstructing covered science or rewriting the sidecar. A run already carrying
current authority needs no qualification sidecar. A missing sidecar/authority does
not cause any loader or application consumer to persist a replacement.

## Producer and source semantics

One authority model remains downstream of all three producer contracts:

| Producer | Independent qualification scope |
| --- | --- |
| FASTQ | `fastq_source_and_producer_record.v1` |
| BAM | `bam_read_pair_transformation.v1` |
| External | `external_source_conservation.v1` |

Qualification invokes each route's existing verifier. Generic canonical integrity
never substitutes for producer qualification, and producer scopes never substitute
for one another. QC, Selection and Matrix retain their original owner algorithms.
There is no alignment, fragment production, QC production, selection production,
matrix production or replacement scientific artifact.

Missing producer authority requires the current producer verifier's source evidence.
For these existing producer scopes that includes original source bytes matching their
recorded identities. Unavailable/changed sources therefore prevent initial strict
qualification; no weaker provenance-only substitute is introduced. Existing v1
records also retain their original validation requirements while being qualified.
This is a consequence of the current scientific scope, not a new blanket freshness
requirement for every consumer or an automatic source search.

Once current authority has been established, ordinary reuse retains
`historical_verified_sources.v1` and can operate with archived producer sources.
It does not claim continuing current-source freshness. Explicit
`current_source_freshness.v1` audits still check present bytes and reject unavailable
or changed sources. An explicit raw-intake step retains its own current bounded
source-aware verification. Standalone and runtime recovery behavior is unchanged.

## Validation

Tiny BAM/external fixtures persist valid scientific outputs without the current
authority field. The synthetic FASTQ backend covers both a complete legacy DAG with
a nonempty matrix and historical authority-v1 qualification. Tests count actual
owner execution through VerificationContext, guard production execution, compare
scientific/state bytes, check idempotence, archive raw sources before trusted reuse,
and reject cross-producer substitution and explicit freshness of missing sources.

Negative cases cover scientific payloads, receipts, QC/Selection manifests,
references, producer sources, runtime configuration, original execution/arguments,
missing accepted verification or recovery provenance, sidecar checksum/state binding,
scientific scope/profile/verifier/upstream qualification incompatibility, interrupted
owner verification, cancellation before publication, and publication failure.

Commands run from the repository root with:

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src
export NUMBA_CACHE_DIR=/tmp/agent-numba-cache
export MPLCONFIGDIR=/tmp/agent-matplotlib-cache
```

Primary focused command (all RUN_* gates absent):

```bash
conda run --no-capture-output -n agent python -m pytest -q \
  tests/authority_dag/test_qualification.py \
  tests/chromap/test_authority_dag.py \
  tests/chromap/test_verification_authority.py \
  -k 'qualification or test_current_authority_is_reused_without_creating_sidecar or test_interruption_discards_all_pending_authorities or test_corrupt_qualified_authority_cannot_be_requalified_or_reused or test_missing_or_mismatched_execution_anchor_fails_closed or test_failed_publication_after_write_is_not_reusable or test_cancellation_before_publication_discards_pending_proofs or test_required_producer_evidence_and_runtime_cannot_be_skipped' \
  > /tmp/agent-post-m117b3b-focused.log 2>&1
```

**51 passed, 32 deselected in 245.43 seconds (4m 5s), exit 0.** This includes
48 new BAM/external cases, two new FASTQ cases, and one existing qualification
compatibility parameter case selected by the filter. An initial development attempt
failed only because the test expected AuthorityError for an archived-source freshness
audit; the existing verifier correctly raised FileNotFoundError. The corrected test
accepts that established failure semantics. This primary focused result supersedes earlier
development batches; the storage follow-up below covers the final checkpoint fix.

The bounded domain command removes all RUN_* gates explicitly and excludes the two
new FASTQ tests already accepted above. The new BAM/external test file is not repeated:

```bash
conda run --no-capture-output -n agent python -c \
  'import os, sys, pytest; [os.environ.pop(k) for k in tuple(os.environ) if k.startswith("RUN_")]; sys.exit(pytest.main(sys.argv[1:]))' \
  -q tests/authority_dag/test_context.py \
  tests/authority_dag/test_propagation.py \
  tests/authority_dag/test_application.py \
  tests/chromap/test_verification_authority.py \
  tests/chromap/test_authority_dag.py \
  tests/chromap/test_fragments_orchestration.py \
  tests/bam_fragments/test_integration.py \
  tests/external_fragments/test_integration.py \
  tests/barcode_qc tests/cell_selection tests/cell_by_ccre tests/matrix_integration \
  tests/unit/orchestration/test_run_store.py \
  tests/unit/orchestration/test_resume.py \
  tests/unit/orchestration/test_cancellation.py \
  tests/unit/orchestration/test_recovery_policy.py \
  tests/unit/orchestration/test_executor.py \
  tests/unit/orchestration/test_verifier.py \
  --deselect tests/chromap/test_authority_dag.py::test_explicit_legacy_fastq_qualification \
  --deselect tests/chromap/test_verification_authority.py::test_explicit_v1_qualification_preserves_historical_record \
  > /tmp/agent-post-m117b3b-domain.log 2>&1
```

Domain result: **681 passed, 1 skipped, 2 deselected, 3 existing warnings in
1,316.40 seconds (21m 56s), exit 0.** The skip is the guarded full-reference Matrix
component test. The warnings are the existing louvain/pkg_resources deprecations
and Numba/TBB version warning. No full repository regression was run.

During domain regression, review found that a cancellation callback could read
RunStore while the qualification publisher held the state lock. The checkpoint
was moved before the locked atomic transaction, and a nonblocking lock-probe
regression now exercises a callback that reads cancellation state. The existing
loader and scientific paths were unchanged, so the domain run continued; the entire
FileRunStore test file and affected qualification failure/publication paths were
rechecked on the final code:

```bash
conda run --no-capture-output -n agent python -m pytest -q \
  tests/authority_dag/test_qualification.py \
  tests/unit/orchestration/test_run_store.py \
  -k 'test_run_store or cancellation or publication_after or interruption' \
  > /tmp/agent-post-m117b3b-storage-followup.log 2>&1
```

**24 passed, 40 deselected in 51.29 seconds, exit 0.** This includes the two new
store-reading cancellation callback cases. No source/test changes followed this
correction and its follow-up. No expensive passing qualification batch or whole
repository suite was repeated.

## Files changed

| File | Reason |
| --- | --- |
| `src/agent/orchestration/verification_authority.py` | Explicit qualification and additive sidecar consumption through the existing authority loader. |
| `src/agent/orchestration/run_store.py` | Immutable-state-bound sidecar reading and atomic publication using existing locking/writer mechanics. |
| `tests/authority_dag/test_qualification.py` | Tiny BAM/external legacy qualification, reuse, integrity, scope and failure/cancellation coverage. |
| `tests/chromap/test_authority_dag.py` | Complete synthetic legacy FASTQ DAG qualification without producer replay. |
| `tests/chromap/test_verification_authority.py` | Explicit historical authority-v1 qualification with original record preservation. |
| `README.md` | Link the explicit qualification API and remaining deferred work. |
| `AGENTS.md` | Record the current engineering and compatibility boundary. |
| `docs/post-m11.7b.3b-legacy-authority-qualification.md` | Audit, API, trust transition and acceptance evidence. |

## Deferred

Actual preserved real-run qualification, preserved-run application/report closeout,
real FASTQ/BAM producer workloads, biological validation, old artifact/run-state
schema migration, application/report redesign and unrelated hardening remain deferred.
Whole-run authority loading and the existing filesystem isolation limits remain;
no performance optimization or migration framework is introduced. No full repository
regression, commit or push is part of this step.


## Final repository state

`main` remains at `ea15bff5df74a53e65469a5694bb51c3c0b09a04`, aligned with
`origin/main`. `git status` lists six modified tracked files and two new files
(the qualification test module and this record), with nothing staged.
`git diff --check` passes. `git diff --stat` covers the tracked changes; the two
new files are separately identified by status and the file inventory above.
No commit or push, real producer workload, preserved-run access or qualification,
or real-run application closeout was performed.
