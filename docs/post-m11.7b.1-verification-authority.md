# Post-M11.7b.1 — Verification authority contract

This foundation introduces producer-neutral verification provenance and one
explicit fragments → QC verifier reuse boundary. It does not complete the
70→2 structural-scan optimization, change scientific algorithms, or qualify
the preserved M11.7b-2 run for reuse.

## Scopes and producer contracts

The closed authority schema is version 1. `VerificationScope` distinguishes:

| Scope | Meaning |
| --- | --- |
| `scientific_correctness.v1` | The named independent verifier completed its defined scientific checks |
| `artifact_integrity_lineage.v1` | Exact accepted bytes, publication, lineage and resource authorities match |
| `presentation_validation.v1` | Presentation corresponds to its accepted evidence; no scientific reconstruction implied |

Canonical `scatac-fragments.v2` integrity is a separate field from producer
qualification. The common `FragmentsAuthorityContract` requires canonical
integrity **and** the exact producer qualification, profile digest and verifier
compatibility. Its immutable contract table represents all three existing routes:

| Producer | Qualification scope | Independent verifier identity |
| --- | --- | --- |
| FASTQ | `fastq_source_and_producer_record.v1` | `agent.fastq-fragments-independent` |
| BAM | `bam_read_pair_transformation.v1` | `agent.bam-fragments-independent` |
| External import | `external_source_conservation.v1` | `agent.external-fragments-independent` |

Compatibility version is `1` for each semantic verifier contract. Exact matching
is required; there is no implicit compatibility across versions or producers.
The profile ID and SHA bind the existing frozen producer profile. Recovery policy
versions remain execution/recovery identities, not verifier compatibility.
FASTQ qualification still does not rerun alignment; BAM/external qualification
retains each existing verifier's declared upstream-history limits.

Generic integrity can be represented with an empty producer qualification, but
cannot satisfy any producer-qualified consumer. BAM and external authority
records can be represented and compatibility-checked without changing the schema
or consumer API. Their issuance/integrity adapters are not enabled yet and reuse
fails closed. This milestone runs no real BAM or external production workload.

## Record and trust anchor

`VerifiedArtifactAuthority` is an immutable, strictly decoded provenance record.
It binds artifact type/contract, final publication path, manifest digest, physical
file closure, durable execution identity, arguments digest, receipt digest,
upstream identities and accepted scopes, resources, science profile, producer
qualification, verifier identity/version, verification scope and completion.
Its identity is a canonical digest over the complete record.

The trust anchor is the accepted successful `StepExecutionResult` in the
operator-supplied `RunStore`, with authority under
`verification.artifact_authority`. New durable FASTQ fragment execution captures
the publication before executor verification, runs the existing independent
public verifier, validates the same bytes afterward, and records authority only
on successful verification. Cross-verification filesystem snapshots also reject
mutation followed by restoration. Failure/interruption cannot publish reusable
authority, even if the scientific publication itself is already durable.

The existing successful-step checkpoint, RunStore checksum envelope and revision
control persist the record; no second attestation database or new publication
receipt is created. This is an optional, self-versioned extension of verification
metadata within the current run-state envelope. Absence is omitted on serialization,
preserving historical record representations. Old readers fail closed on the new
field. A parsed record or arbitrary `passed` boolean is never a reuse capability.

`load_fragment_authority(store, run_id, step_id)` requires the accepted stored
step, exact plan-derived execution identity and compatible qualification. It
returns an immutable opaque handle bound to the whole stored step. Validation
reloads that anchor before and after checking publication integrity. It does not
write the store or upgrade historical records.

The operator-owned RunStore is inside the application's existing trust boundary.
Its checksum is not a signature: an actor who can coherently replace the entire
trusted store is outside this protection model. Coordinated replacement of local
payloads/manifests/receipts cannot replace the separately retained accepted
digests. User/planner input cannot supply a RunStore or arbitrary Python issuer.

Execution receipts retain their existing meaning: exact execution/argument
publication ownership. Their digest is one component of verification authority;
receipts alone never prove scientific correctness.

## Integrity and invalidation

The FASTQ adapter binds the final manifest, BGZF and tabix files, producer record,
profile, receipt, upstream intake/context/reference/index manifests, selected raw
sources, whitelists and bound physical reference/index resources. It retains
bounded intake inventory reinspection. QC's own reference, annotation, catalog
qualification and runtime checks remain fresh and are not covered by fragment
authority or cached.

Every reuse compares the complete bound-file SHA/size closure and semantic
identities against accepted provenance. It is not a path/stat-only cache.
Same-size replacements, altered sidecars, wrong publication/receipt, changed
upstream identities/scopes, reference/resource changes, unsupported profiles or
verifier contracts, and insufficient/cross-producer qualification all fail closed.
No invalid supplied authority silently falls back to accepting changed bytes.
New verification of changed bytes would require a new accepted authority.

This deliberately retains full physical hashing, including raw source files and
reference/index resources. Eliminating structural FASTQ reconstruction does not
mean eliminating raw I/O or establishing a throughput improvement. No persistent
verification-result cache or stat-based digest cache is introduced.

## First integration and default behavior

The explicit data-layer entry point is:

```python
from agent.orchestration.verification_authority import load_fragment_authority
from agent.tools.data.barcode_qc_verifier import verify_barcode_qc

authority = load_fragment_authority(store, run_id, fragment_step_id)
verified_qc = verify_barcode_qc(
    qc_manifest_path,
    expected_sha256=qc_manifest_sha256,
    fragments_authority=authority,
)
```

The common QC binding checks that the handle's producer qualification matches the
actual v2 producer. With compatible authority it validates integrity, retains
generic v2 validation/reading and reconstructs QC independently; it does not repeat
the covered producer-specific deep verifier. Completion checks validate authority
again. `handle.validate(...)` returns `AuthorityValidation` with integrity scope,
never a claim of fresh scientific reconstruction.

Without an authority, public QC verification follows its original deep path.
Registered QC production, orchestration verification, recovery, evidence,
visualization and report composition are not automatically wired to reuse it.
Recovery retains its existing fresh verification and no-production-replay rules;
it does not manufacture new authority for legacy/recovered results. A resumed
incomplete run may lose optional authority when its step is freshly reconstructed;
absence remains fail closed for explicit reuse.

## Validation and deferred integration

Tiny tests in `tests/chromap/test_verification_authority.py` cover issuance,
persistence, exact reuse/deep-call counters, all three producer contracts and
cross-producer rejection, corruption, same-size mutation, changed accepted anchors,
unsupported verifier/profile/scope, insufficient upstream scope, interruption,
mutation-and-restoration, and a rehashed QC metric forgery. A durable evidence test
confirms evidence still performs fresh verification and production is not replayed.
Existing corruption and recovery assertions are retained.

Validation on 2026-09-12: all 30 authority tests passed. The final focused run
(authority, fragment orchestration/recovery, RunStore and cancellation) passed
138 tests. Broader fragments/QC/selection/matrix, orchestration, Application and
report regressions passed 1,528 tests with 27 skipped and 3 warnings. All `RUN_*`
gates were disabled; only generated fixtures were used. No real acceptance run
was resumed, modified or deeply verified.

Deferred: BAM/external issuance adapters, Evidence projection integration,
figureless visualization integration, report snapshot/verification integration,
legacy M11.7b run qualification and preserved-run application closeout. Historical
`all_steps_freshly_verified` meaning is unchanged. The preserved real scientific
run and publications must not be retroactively labeled or modified by this work.
