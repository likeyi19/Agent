# Post-M11.7b.2 — Scientific authority propagation

Baseline: `3c4d45674076a2fa5a10a2066017dc55f374c547` on clean, aligned `main`.
This extends the accepted [Post-M11.7b.1 foundation](post-m11.7b.1-verification-authority.md).
It changes verification dependency reuse, not scientific algorithms or artifact
schemas. The authority record has a new closed schema version 2 to make its
historical-source semantics explicit. Schema version 1 retains its original
current-source validation behavior; existing records are never upgraded.

## Scientific boundary and capabilities

Canonical `scatac-fragments.v2` integrity and producer qualification remain separate.
One authority architecture supports all three existing producer contracts:

- FASTQ: `fastq_fragment_production`, scope `fastq_source_and_producer_record.v1`,
  exact FASTQ/Chromap provenance;
- BAM: `bam_fragment_production`, scope `bam_read_pair_transformation.v1`,
  corrected-CB read-pair/transformation provenance;
- External: `external_fragment_adoption`, scope `external_source_conservation.v1`,
  source conservation and import provenance.

Each qualification retains the .1 contract's distinct profile, scope and verifier
compatibility identity. Generic integrity never grants any producer qualification.
No producer qualification substitutes for another. The implication from a
successfully qualified producer to canonical integrity is one-way.

Normal **new durable scientific execution** now establishes this chain:

```text
producer independent verification → fragments authority
  → independent QC reconstruction → QC authority
    → independent Selection reconstruction → Selection authority
      → independent Matrix reconstruction → Matrix authority
```

QC requires the exact producer-qualified fragments dependency. Selection requires
the exact QC dependency. Matrix requires exact fragments and Selection dependencies,
plus the complete reference/cCRE identity. Transitive scientific lineage is retained
without rerunning its science. Existing producer-specific dispatch remains at the
producer qualification boundary; no new downstream scientific routing is added.

Each authority binds the artifact and manifest, physical payload/sidecar closure,
receipt, execution identity, normalized arguments, current reference/resources,
immediate dependency authority identities, science profile, verifier compatibility,
and successful scientific completion. The bound manifest additionally preserves:

- QC: resource/catalog and runtime qualification, table/histogram identities and order;
- Selection: thresholds, decisions/reasons, selected membership and ordered identity;
- Matrix: logical sparse identity, exact rows and full ordered cCRE columns, counts,
  overlap diagnostic, reference and backend profile.

Every new layer still runs its existing independent scientific verifier. Selection
remains `explicit_qc_thresholds.v1`, with `cell_call_method=none` and
`cell_call_state=not_assessed`. Matrix remains canonical fragment-record overlap
counts, with independent reconstruction of counts and both axes.

## Source policy

Version 2 records use `historical_verified_sources.v1` and the distinct integrity
scope `artifact_integrity_lineage_historical_sources.v1`. Raw source path/SHA/size
identities are established by the original producer verifier and stored in
`historical_sources`, disjoint from the current physical artifact/resource closure.
Ordinary downstream consumption neither scans nor hashes these raw files. The
historical source may have changed or become unavailable; this does not alter the
previously verified producer identity and does not establish current freshness.

Operator code may explicitly request `current_source_freshness.v1` when loading
accepted authorities. This hashes every recorded physical source against its
historical identity, deduplicates unchanged files within the operation, and checks
for changes again during consumption and at scope exit. Changed or missing sources
fail closed. This policy checks the recorded files; it does not rediscover source
directories or claim freshness for additional files outside the accepted closure.
It does not rerun producer science.

Current artifact payloads, receipts, lineage and resources are always checked.
The reduction in raw-source reads does not remove resource or payload hashing.
The trust model remains an operator-owned local RunStore and local filesystem;
checksums do not authenticate a malicious replacement of the trusted store.

## Operation context and publication

`VerificationContext` uses the authority vocabulary and only successful independent
proofs or validated accepted records. It is scoped to one new durable executor
operation or an explicit `accepted_authorities(store, run_id, source_policy=...)`
scientific operation. No planner parameter, skip flag, process-global cache or
manifest-only adoption API is introduced.

Proof identity includes scientific content, execution, exact dependencies, profile,
verifier/scope, qualification, physical closure and source policy. File hashes are
reused only for an unchanged device/inode/size/mtime/ctime snapshot, never a path
alone. Changed accepted content fails closed. Failed or interrupted verification
creates no successful proof. Verified metadata is immutable.

Private-stage verification creates only operation-local proof. Publication may
reuse it after validating that the atomic move changed only reviewed owned
locators. Existing fsync order and post-publication cancellation behavior are
preserved. Executor authority hashing adds no new cancellation checkpoints inside
scientific tool work; the original tool checkpoints retain control. Explicit
authority loading outside that executor scope remains cancellable. Durable authority is attached only after exact receipt/publication
verification, and becomes an accepted downstream anchor after the executor's
successful persistence checkpoint. Separate new producer/QC/Selection/Matrix executions receive separate owned-science
proof identities, even for otherwise identical content. Generic canonical integrity
claims no producer or downstream execution identity. Unrecorded dependencies do not
inherit unrelated execution identities when output roots are shared.

Persisted reuse loads only successful authority-bearing steps through the trusted
RunStore, checks the accepted execution anchor, and compares current descriptions
against the entire recorded authority. Arbitrary deserialized authority, manifest
or receipt bytes confer no trust. A new scientific dependency without recorded
authority takes the existing deep path; its freshly established proof can be used
within that operation but is not an implicit upgrade of a legacy record.

The current explicit loader conservatively validates all version-2 authorities in
the selected run. It does not prune validation to only one requested dependency.
Custom Matrix verification limits retain the deep verification path instead of
reusing a result obtained under different operational limits.

## Compatibility and exclusions

Standalone verifiers, execution without a durable authority context, and resumed
execution with completed steps retain deep verification. Existing receipt recovery,
leases, PLAN_ONLY, cancellation and terminal-state rules are unchanged. No Registry,
Planner, semantic compiler, scientific public arguments or algorithms are changed.

Evidence projection/application verification, visualization discovery, report
source snapshots and report verification receive no authority integration here.
Their existing deep verification can still repeat scientific work. This is **not**
application-wide `70 → 2` completion.

The preserved legacy M11.7b real run is untouched. Legacy authority qualification
and preserved-run application closeout remain future work. Tests use tiny synthetic
fixtures; this milestone establishes no biological acceptance or real-workload
throughput result.

## Validation

Tests instrument actual owner verifiers, complete source scans/projections, full
source hashing and bounded intake/encoding inspections. They also exercise persisted
producer-only reuse for a new QC reconstruction, cross-producer scope rejection,
all four publication/payload boundaries, same-size replacements, incompatible
profiles/verifiers/scopes, lineage/resource mismatch, explicit freshness, failed or
interrupted verification, and publication cancellation ordering.

Measured normal tiny scientific DAG counts (fixture setup excluded):

| Producer | Complete source passes | Full source-file hashes | Bounded inspection calls | Additional downstream raw scans/hashes |
| --- | --- | --- | --- | --- |
| FASTQ | 2 structural scans: input validation + independent provenance | 8 (twice for each of four role files) | 2 intake reinspections | 0 / 0 |
| BAM | 2 projections: production + independent reconstruction | 4 | 2 intake reinspections | 0 / 0 |
| External | 1 production scan + 1 independent conservation scan | 2 | 2 encoding inspections | 0 / 0 |

For every route, independent producer, generic canonical, QC, Selection and Matrix
verifiers each run once in the normal new durable DAG. FASTQ also produces a
nonempty matrix whose independently verified total count is 1. Bounded inspections
are counted separately from full passes; they are not described as full scans.
Persisted Matrix consumption adds no deep/source calls. Persisted producer-only
consumption performs new QC reconstruction without generic or producer replay.

Earlier checks exposed an added FASTQ cancellation checkpoint and an execution
ownership mismatch for shared output roots; both were corrected without weakening
the existing cancellation test. The final manifest-SHA correction then completed
before closeout validation. Earlier interrupted full runs are not acceptance runs.
The final focused, domain and frozen-code full results below supersede the earlier
intermediate results.

## Closeout scope and deferred follow-up

The manifest-SHA correction binds each verified location to both its semantic
scientific identity and its exact manifest digest. JSON formatting changes at an
existing publication therefore invalidate reuse even without a caller-supplied
expected digest. Reviewed private-to-public moves establish the destination's exact
digest separately. The closeout authority/corruption batch passes all 100 tests
(256.21 seconds), including all three producers' manifest-byte checks.

The following are deferred, not additional requirements or implementations here:

- Narrowing persisted validation from all version-2 records in a run to a requested
  dependency subset.
- Further deduplication of runtime qualification and resource reinspection costs.
- Stronger whole-operation filesystem snapshot isolation beyond the existing
  trusted-local-filesystem contract.

The presentation and legacy-run work listed above remains deferred. Closeout adds
no new authority architecture or scientific capability.

## Closeout domain regression

The final manifest-SHA code passes 1,862 tests, with 28 skipped and 3 warnings
in 1,303.13 seconds. All `RUN_*` gates were disabled.

| Domain | Passed | Skipped |
| --- | ---: | ---: |
| FASTQ / Chromap contracts | 286 | 27 |
| BAM fragments | 161 | 0 |
| External fragments | 113 | 0 |
| Barcode QC | 88 | 0 |
| Selection | 121 | 0 |
| Matrix data layer | 25 | 1 |
| Matrix integration | 45 | 0 |
| Raw intake | 172 | 0 |
| Orchestration, recovery and lifecycle | 851 | 0 |

## Acceptance criteria confirmation

The following criteria are checked against the implementation, the 100-test
closeout authority/corruption batch, and the completed closeout domain regressions.
Full lightweight acceptance is recorded separately after the frozen-code run.

| # | Criterion | Evidence / confirmation |
| --- | --- | --- |
| 1 | Unified producer authority | One versioned schema, context, persisted loader and reviewed adapter set serve FASTQ, BAM and external fragments. |
| 2 | Explicit isolated producer scopes | The three accepted `.1` qualification/profile/verifier contracts remain distinct. |
| 3 | No substitution | All six cross-producer directions and generic integrity in place of qualification fail closed. |
| 4 | QC owns QC science | The original independent QC reconstruction runs once and successful new durable QC steps record authority. |
| 5 | Selection owns selection science | The original decisions/reasons/membership/order reconstruction runs once; thresholds and cell-calling status are unchanged. |
| 6 | Matrix owns matrix science | Independent count and axis reconstruction runs once; authority binds logical matrix, rows, full ordered cCREs and backend/reference. |
| 7 | Upstream scientific reuse | Local and persisted authority prevent covered producer/QC/Selection replay; producer-only persistence also supports a new QC reconstruction. |
| 8 | No repeated downstream raw scans/hashes | Instrumented FASTQ/BAM/external scientific DAGs add zero downstream raw scans, hashes or guarded raw-file opens. |
| 9 | Explicit current freshness | Historical identity and current-source hashing have separate policies; changed/missing current sources fail the explicit audit. |
| 10 | Strict standalone default | Without a trusted context, existing deep verifiers run; no legacy record is silently upgraded. |
| 11 | Fail-closed integrity/compatibility | Payloads, sidecars, manifests, receipts, lineage/resources, scope/profile/verifier mismatch and exact manifest-byte changes are rejected. Failed/interrupted verification grants no owner proof. |
| 12 | Lifecycle preservation | Existing publication/recovery, cancellation, terminal-state, lease and PLAN_ONLY tests pass; recovery retains deep verification and never replays production unsafely. |
| 13 | No algorithm changes | Scientific verifier bodies and production algorithms are unchanged; additions are authority wrappers and checked publication handoffs. |
| 14 | No planning changes | Registry interfaces, Planner, semantic compiler and wire schemas are unchanged; orchestration regressions pass. |
| 15 | Preserved real run untouched | No resume, mutation, qualification or deep re-verification of the preserved real M11.7b run was performed. Tests use tiny fixtures. |
| 16 | Presentation deferred | AnalysisEvidence, visualization, report snapshot/verification and legacy application closeout receive no reuse integration. |
| 17 | Intended review scope only | Repository diff/status review and source/test fingerprints delimit the uncommitted milestone changes; no commit or push. |

## Frozen final lightweight acceptance

After the 100-test authority/corruption batch and 1,862-test domain regression
passed, all 17 acceptance criteria above were confirmed and implementation was
frozen. Exactly one fresh full lightweight run then executed the complete `tests`
directory with all `RUN_*` environment gates removed and no exclusions:

**3,679 passed, 83 skipped, 7 warnings in 1,635.09 seconds (27m 15s), exit 0.**

The warnings are the existing third-party deprecation/threading and deliberate
nonunique-observation fixture warnings. SHA-256 fingerprints of all 273 source/test
files and the baseline HEAD matched after completion. No source or test changes
were made during or after this final regression. Only this documentation acceptance
record was finalized after the pass. No additional tests or development followed.

Review state: `main`, HEAD `3c4d45674076a2fa5a10a2066017dc55f374c547`, still aligned
with `origin/main`; intended Post-M11.7b.2 changes are uncommitted. No commit or push,
real producer workload, preserved-run access/resume or legacy authority injection
was performed. The milestone is ready for review under its accepted scientific-DAG
scope; application-wide verification reuse remains deferred.
