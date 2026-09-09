"""Verified streaming views over v1/v2; never migrate or produce fragments.

Open performs the selected contract's fresh verification. Iteration holds only
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
from .scatac_fragments_v2_verifier import FragmentVerification, take_snapshots, check_snapshots


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


def _dispatch_manifest(path, expected_sha256):
    """Read bounded JSON for version dispatch, preserving v1's JSON encodings.

    The selected verifier still enforces its own encoding and full contract;
    in particular this does not relax v2's strict UTF-8 requirement.
    """
    v2.sha(expected_sha256)
    try:
        with Path(path).open('rb') as stream:
            payload = stream.read(v2.MAX_BYTES + 1)
        if len(payload) > v2.MAX_BYTES:
            v2.fail()
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            v2.fail('FRAGMENTS_V2_DIGEST_MISMATCH')
        return payload, json.loads(payload, object_pairs_hook=v2._pairs,
                                   parse_constant=lambda _: v2.fail())
    except v2.FragmentsV2Error:
        raise
    except (OSError, ValueError, TypeError, UnicodeError, RecursionError):
        v2.fail()


@dataclass(frozen=True)
class VerifiedFragments:
    verification: FragmentVerification
    libraries: tuple[FragmentLibrary, ...]

    @property
    def manifest(self):
        """Copy of the original version's metadata, never a converted manifest."""
        return self.verification.manifest

    @property
    def contract_version(self):
        return self.manifest['contract_version']

    @property
    def contigs(self):
        return self.verification.contigs

    def iter_fragments(self, namespace):
        """Iterate one explicitly selected library in its canonical order."""
        from .scatac_fragments_verifier import _bgzf_lines
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
    """Freshly verify one explicitly versioned artifact and expose its view.

    Runtime supplies the existing qualified packaging tools; no aligner field
    is needed. V1 still invokes its exact accepted verifier. No tool registration,
    execution configuration lookup, scientific producer, or format guessing.
    """
    initial = take_snapshots([manifest_path])
    payload, value = _dispatch_manifest(manifest_path, expected_sha256)
    if type(value) is not dict or value.get('artifact_type') != v2.ARTIFACT_TYPE:
        fail('FRAGMENTS_READER_CONTRACT_UNSUPPORTED')
    version = value.get('schema_version')
    if type(version) is not int:
        fail('FRAGMENTS_READER_CONTRACT_UNSUPPORTED')
    if version == 2:
        from .scatac_fragments_v2_verifier import verify_fragments_v2
        verified = verify_fragments_v2(manifest_path, expected_sha256=expected_sha256, runtime=runtime)
    elif version == 1:
        from .scatac_fragments_manifest import validate_fragments_manifest
        from .scatac_fragments_verifier import verify_fragments
        from . import scatac_reference as ref
        value = validate_fragments_manifest(value)
        _, bundle, _ = ref.load_scatac_reference_bundle(value['inputs']['reference_path'],
                                        expected_sha256=value['inputs']['reference_sha256'])
        paths = [manifest_path, value['inputs']['reference_path'], bundle.genome.fai.path]
        paths += [Path(manifest_path).parent / e[k]['path'] for e in value['libraries'] for k in ('bgzf', 'tabix')]
        before = take_snapshots(paths)
        value = verify_fragments(manifest_path, expected_sha256=expected_sha256, runtime=runtime)
        dictionary, fai_sha, ordered = ref._inspect_fai(Path(bundle.genome.fai.path),
                                                        Path(bundle.genome.fasta.path).stat().st_size)
        if fai_sha != bundle.genome.fai.sha256 or ordered != value['lineage']['ordered_contig_sha256']:
            fail('FRAGMENTS_READER_REFERENCE_MISMATCH')
        check_snapshots(before)
        verified = FragmentVerification(str(manifest_path), expected_sha256, payload,
            tuple(dictionary.items()), before, bound_resource_identities='legacy_v1_defined_checks',
            producer_profile_qualification='legacy_qualified_chromap_policy')
    else:
        fail('FRAGMENTS_READER_CONTRACT_UNSUPPORTED')
    check_snapshots(initial)
    value = verified.manifest
    libraries = []
    for entry in value['libraries']:
        if version == 1:
            # Preserve the actual legacy policy strings and full manifest access.
            provenance = {key: value[key] for key in ('backend', 'mapping_policy', 'inputs', 'lineage', 'semantics')}
            provenance['library'] = entry
            support = {'unit': 'paired_mappings', 'definition': value['semantics']['support']}
            mode = 'absent'
        else:
            provenance = entry['provenance']; support = provenance['support']; mode = entry['strand']['mode']
        libraries.append(FragmentLibrary(entry['namespace'], entry['n_fragment_records'], entry['sum_support'],
            entry['n_distinct_barcodes'], entry['max_support'], mode, canonical(provenance), canonical(support)))
    return VerifiedFragments(verified, tuple(libraries))
