"""Verified streaming views over the current v2 contract; never migrate or produce fragments.

Open performs fresh v2 content/resource verification. Iteration holds only
one decoded record plus BGZF buffers, checks artifact snapshots before/after,
and checks the complete stream digest on exhaustion. Consume to exhaustion (or
close) to run the final mutation check. This is trusted-local-filesystem
consistency, not a hostile-filesystem snapshot or a producer-history guarantee.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from ._fragments_common import canonical, fail
from . import scatac_fragments_v2 as v2
from .scatac_fragments_v2_verifier import FragmentVerification, FragmentVerificationRuntime, take_snapshots, check_snapshots


@dataclass(frozen=True)
class FragmentRecord:
    namespace: str
    contig: str
    start: int
    end: int
    barcode_identifier: str
    support: int
    strand: str | None

    @property
    def cell_identity(self):
        return self.namespace, self.barcode_identifier


@dataclass(frozen=True)
class FragmentLibrary:
    namespace: str
    n_fragment_records: int
    sum_support: int
    n_distinct_barcodes: int
    max_support: int
    strand_mode: str
    _provenance_bytes: bytes
    _support_bytes: bytes

    @property
    def provenance(self):
        return json.loads(self._provenance_bytes)

    @property
    def support_meaning(self):
        return json.loads(self._support_bytes)


def _record(line, namespace, present):
    """Decode already-verified canonical bytes, independently of verification."""
    values = line[:-1].decode('utf-8').split('\t')
    if len(values) != (6 if present else 5):
        fail('FRAGMENTS_READER_RECORD_INVALID')
    chrom, start, end, barcode, support = values[:5]
    return FragmentRecord(namespace, chrom, int(start), int(end), barcode,
                          int(support), values[5] if present else None)


@dataclass(frozen=True)
class VerifiedFragments:
    verification: FragmentVerification
    libraries: tuple[FragmentLibrary, ...]

    @property
    def manifest(self):
        """Copy of verified v2 metadata."""
        return self.verification.manifest

    @property
    def contract_version(self):
        return self.manifest['contract_version']

    @property
    def contigs(self):
        return self.verification.contigs

    def iter_fragments(self, namespace):
        """Iterate one explicitly selected library in its canonical order."""
        from ._fragment_io import _bgzf_lines
        entries = self.manifest['libraries']
        matches = [e for e in entries if e['namespace'] == namespace]
        if len(matches) != 1:
            fail('FRAGMENTS_READER_NAMESPACE_UNKNOWN')
        entry = matches[0]
        present = entry.get('strand', {}).get('mode') == 'present'
        path = Path(self.verification.manifest_path).parent / entry['bgzf']['path']
        self.verification.check_unchanged()
        digest = hashlib.sha256()
        try:
            for line in _bgzf_lines(path):
                digest.update(line)
                yield _record(line, namespace, present)
            if digest.hexdigest() != entry['canonical_record_stream_sha256']:
                fail('FRAGMENTS_READER_STREAM_CHANGED')
        finally:
            self.verification.check_unchanged()


def open_verified_fragments(manifest_path, *, expected_sha256, runtime):
    """Verify v2 content/resources without guessing or routing by producer origin.

    Generic verification does not qualify producer science. Producer-specific
    tools/verifiers establish that additional authority before downstream use.
    Historical v1 artifacts are rejected, never upgraded or rewritten.
    """
    from .scatac_fragments_v2_verifier import verify_fragments_v2
    initial = take_snapshots([manifest_path])
    _, value = v2.read_manifest_bytes(manifest_path, expected_sha256)
    if (type(value) is dict and value.get('artifact_type') == v2.ARTIFACT_TYPE
            and type(value.get('schema_version')) is int and value['schema_version'] == 1):
        fail('FRAGMENTS_CONTRACT_RETIRED')
    if (type(value) is not dict or value.get('artifact_type') != v2.ARTIFACT_TYPE
            or type(value.get('schema_version')) is not int or value['schema_version'] != 2):
        fail('FRAGMENTS_READER_CONTRACT_UNSUPPORTED')
    verified = verify_fragments_v2(manifest_path, expected_sha256=expected_sha256, runtime=runtime)
    check_snapshots(initial)
    libraries = []
    for entry in verified.manifest['libraries']:
        provenance = entry['provenance']
        libraries.append(FragmentLibrary(entry['namespace'], entry['n_fragment_records'], entry['sum_support'],
            entry['n_distinct_barcodes'], entry['max_support'], entry['strand']['mode'],
            canonical(provenance), canonical(provenance['support'])))
    return VerifiedFragments(verified, tuple(libraries))


def qualified_fragments(path,sha, *, fragments_authority=None):
    # The declared producer kind selects its reviewed verifier, never a QC algorithm.
    value=v2.load_fragments_manifest_v2(path,expected_sha256=sha)
    kinds={e['provenance']['kind'] for e in value['libraries']}
    if len(kinds)!=1: fail('FRAGMENTS_PRODUCER_UNQUALIFIED')
    kind=next(iter(kinds)); runtime=FragmentVerificationRuntime()
    if fragments_authority is not None:
        from .fragments_authority_contract import StoredFragmentAuthority
        from agent.schemas.verification_authority import AuthorityError
        if type(fragments_authority) is not StoredFragmentAuthority:
            raise AuthorityError('Fragments require stored producer-specific scientific authority.')
        fragments_authority.validate(path,sha,producer_kind=kind)
    if kind=='fastq_fragment_production':
        from .fastq_fragments_verifier import verify_fragments
        from .scatac_fragments import verification_runtime
        if fragments_authority is None:
            verify_fragments(path,expected_sha256=sha,runtime=verification_runtime())
        basis='qualified_fastq_source_and_producer_record';history='alignment_not_rerun'
    elif kind=='bam_fragment_production':
        from .bam_fragments_verifier import verify_bam_fragments
        if fragments_authority is None:
            verify_bam_fragments(path,expected_sha256=sha,runtime=runtime)
        basis='independent_bam_transformation';history='upstream_history_declared'
    elif kind=='external_fragment_adoption':
        from .external_fragments_verifier import verify_external_fragments
        if fragments_authority is None:
            verify_external_fragments(path,expected_sha256=sha,runtime=runtime)
        basis='independent_external_conservation';history='upstream_history_declared'
    else: fail('FRAGMENTS_PRODUCER_UNQUALIFIED')
    view=open_verified_fragments(path,expected_sha256=sha,runtime=runtime)
    if view.manifest!=value: fail('FRAGMENTS_SOURCE_CHANGED')
    return view,dict(kind=kind,profile_ids=sorted({e['provenance']['profile']['id'] for e in value['libraries']}),
        verification_basis=basis,history=history)
