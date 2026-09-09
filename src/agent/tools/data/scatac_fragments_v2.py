"""Producer-neutral fragments v2 domain; no fragment production or adoption.

Profile/source identities are bound claims, not execution authority. There are
no qualified v2 producers in M11.3a. A producer-specific verifier must establish
the scientific meaning/history of a future producer; generic verification checks
contents and resource bytes only. Existing Chromap fragments v1 is unchanged.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

from ._fragments_common import MAX_SUPPORT, MAX_TOTAL, canonical

ARTIFACT_TYPE = 'agent.scatac-fragments'
SCHEMA_VERSION = 2
CONTRACT_VERSION = 'scatac-fragments.v2'
MAX_BYTES = 8 * 1024 * 1024
MAX_LIBRARIES = 4096
MAX_SOURCES = 4096  # Across the whole manifest, not per library.
MAX_IDENTIFIER = 256
SEMANTICS = {
    'coordinates': '0-based-half-open',
    'order': 'fai-rank,start-numeric,end-numeric,barcode-ascii,strand-ascii.v2',
    'key': 'contig,start,end,barcode,strand-state;namespace-local.v2',
    'barcode': 'opaque-ascii-graph-1..256.v1',
    'cell_identity': 'namespace,barcode;not-called-cells.v1',
    'support': 'exact-positive-integer;meaning-in-library-provenance.v2',
    'individual_support_max': MAX_SUPPORT,
    'aggregate_support_max': MAX_TOTAL,
    'stream_identity': 'sha256(canonical-utf8-tab-fields-with-LF-in-order)',
    'encoding': 'BGZF;terminal-EOF;headerless;TBI-BED.v1',
}
SUMMARY_KEYS = ('canonical_record_stream_sha256', 'n_fragment_records',
                'sum_support', 'n_distinct_barcodes', 'max_support')
PROCESSING_KEYS = ('tn5', 'deduplication', 'mapq_filtering', 'barcode_correction')


class FragmentsV2Error(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def fail(code='FRAGMENTS_V2_CONTRACT_INVALID'):
    raise FragmentsV2Error(code) from None


def shape(value, keys):
    if type(value) is not dict or set(value) != set(keys):
        fail()


def sha(value):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
        fail('FRAGMENTS_V2_DIGEST_INVALID')


def number(value, low=1, high=MAX_TOTAL):
    if type(value) is not int or not low <= value <= high:
        fail('FRAGMENTS_V2_SUPPORT_INVALID')


def text(value, maximum=2048):
    if (type(value) is not str or not value or len(value.encode('utf-8')) > maximum
            or any(ord(c) < 32 or ord(c) == 127 for c in value)):
        fail()


def token(value):
    if type(value) is not str or re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', value) is None:
        fail()


def absolute_path(value):
    text(value, 4096)
    path = Path(value)
    if not path.is_absolute() or str(path) != value or '..' in path.parts:
        fail()


def resource(value):
    shape(value, ('path', 'sha256', 'size_bytes'))
    absolute_path(value['path']); sha(value['sha256'])
    number(value['size_bytes'], high=2**63 - 1)


def _provenance(value):
    shape(value, ('kind', 'contract_version', 'profile', 'sources', 'producer_record',
                  'support', 'processing', 'source_selection', 'verification_requirement'))
    kind = value['kind']
    if kind not in ('external_fragment_adoption', 'bam_fragment_production', 'fastq_fragment_production'):
        fail('FRAGMENTS_V2_PROVENANCE_INVALID')
    if (value['contract_version'] != 'fragment-producer-provenance.v1'
            or value['verification_requirement'] != 'producer-specific-verifier-required.v1'
            or value['source_selection'] not in ('all_source_records', 'producer_subset', 'unspecified')):
        fail('FRAGMENTS_V2_PROVENANCE_INVALID')
    shape(value['profile'], ('id', 'resource'))
    token(value['profile']['id']); resource(value['profile']['resource'])
    # An opaque, content-bound profile document is a policy identity, never
    # arbitrary executable configuration or an automatically accepted profile.
    sources = value['sources']
    if type(sources) is not list or not 1 <= len(sources) <= MAX_SOURCES:
        fail('FRAGMENTS_V2_PROVENANCE_INVALID')
    roles = []
    identities = []
    for source in sources:
        shape(source, ('role', 'resource'))
        resource(source['resource'])
        roles.append(source['role'])
        identities.append((source['role'], source['resource']['sha256'], source['resource']['path']))
    allowed = {
        'external_fragment_adoption': {'external_fragments'},
        'bam_fragment_production': {'bam', 'intake_manifest', 'library_context'},
        'fastq_fragment_production': {'fastq', 'intake_manifest', 'library_context'},
    }[kind]
    if (set(roles) != allowed or identities != sorted(set(identities))
            or len({p for _, _, p in identities}) != len(identities)):
        fail('FRAGMENTS_V2_PROVENANCE_INVALID')
    if kind != 'external_fragment_adoption' and any(roles.count(r) != 1 for r in ('intake_manifest', 'library_context')):
        fail('FRAGMENTS_V2_PROVENANCE_INVALID')
    # Reserve truthful future FASTQ publication without changing the v1 producer.
    # Rich backend/index/whitelist/decoded-source lineage belongs in a bound,
    # producer-specific record, whose interpretation requires its own verifier.
    if kind == 'fastq_fragment_production' and value['producer_record'] is None:
        fail('FRAGMENTS_V2_PROVENANCE_INVALID')
    if value['producer_record'] is not None:
        resource(value['producer_record'])
    support = value['support']
    shape(support, ('unit', 'definition'))
    if support['unit'] not in ('read_pairs', 'paired_mappings', 'source_records'):
        fail('FRAGMENTS_V2_PROVENANCE_INVALID')
    text(support['definition'])
    shape(value['processing'], PROCESSING_KEYS)
    for claim in value['processing'].values():
        shape(claim, ('status', 'description'))
        if claim['status'] == 'unspecified':
            if claim['description'] is not None:
                fail('FRAGMENTS_V2_PROVENANCE_INVALID')
        elif claim['status'] == 'declared':
            text(claim['description'])
        else:
            fail('FRAGMENTS_V2_PROVENANCE_INVALID')


def bound_provenance_resources(value):
    """Enumerate closed identity bindings; do not interpret profile documents."""
    for entry in value['libraries']:
        p = entry['provenance']
        yield p['profile']['resource']
        yield from (s['resource'] for s in p['sources'])
        if p['producer_record'] is not None:
            yield p['producer_record']


def _fragments_identity(value):
    """Logical identity excludes locators/physical outputs, retains all claims.

    Source bindings retain exact content identities, including profile bytes.
    The external manifest SHA separately protects every actual manifest byte.
    """
    value = deepcopy(value)
    value.pop('fragments_identity_sha256', None)
    value['reference'].pop('manifest_path', None)
    for entry in value['libraries']:
        entry.pop('bgzf', None); entry.pop('tabix', None)
    for item in bound_provenance_resources(value):
        item.pop('path', None)
    return hashlib.sha256(b'agent.scatac-fragments.v2\0' + canonical(value)).hexdigest()


def fragments_identity(value):
    """Compute the logical digest; malformed structures fail with a domain code."""
    try:
        return _fragments_identity(value)
    except (ValueError, TypeError, KeyError, IndexError, UnicodeError, RecursionError):
        fail()


def validate_fragments_manifest_v2(value):
    """Closed, bounded manifest validation with no scientific resource IO."""
    try:
        shape(value, ('artifact_type', 'schema_version', 'contract_version', 'reference',
                      'semantics', 'libraries', 'fragments_identity_sha256'))
        if (value['artifact_type'] != ARTIFACT_TYPE or type(value['schema_version']) is not int
                or value['schema_version'] != SCHEMA_VERSION or value['contract_version'] != CONTRACT_VERSION
                or canonical(value['semantics']) != canonical(SEMANTICS)):
            fail()
        r = value['reference']
        shape(r, ('manifest_path', 'manifest_sha256', 'reference_identity_sha256',
                  'species', 'assembly', 'ordered_contig_sha256'))
        absolute_path(r['manifest_path'])
        for key in ('manifest_sha256', 'reference_identity_sha256', 'ordered_contig_sha256'):
            sha(r[key])
        if r['species'] not in ('human', 'mouse') or r['assembly'] != {'human': 'hg38', 'mouse': 'mm10'}[r['species']]:
            fail()
        libraries = value['libraries']
        if type(libraries) is not list or not 1 <= len(libraries) <= MAX_LIBRARIES:
            fail()
        namespaces = []; total = records = source_count = 0
        for entry in libraries:
            shape(entry, ('namespace', 'strand', 'provenance', 'bgzf', 'tabix', *SUMMARY_KEYS))
            token(entry['namespace']); namespaces.append(entry['namespace'])
            shape(entry['strand'], ('mode', 'definition'))
            if entry['strand']['mode'] == 'absent':
                if entry['strand']['definition'] is not None:
                    fail()
            elif entry['strand']['mode'] == 'present':
                text(entry['strand']['definition'])
            else:
                fail()
            _provenance(entry['provenance'])
            source_count += len(entry['provenance']['sources'])
            if source_count > MAX_SOURCES:
                fail()
            sha(entry['canonical_record_stream_sha256'])
            for key in ('n_fragment_records', 'sum_support', 'n_distinct_barcodes'):
                number(entry[key])
            number(entry['max_support'], high=MAX_SUPPORT)
            n, s, maximum = entry['n_fragment_records'], entry['sum_support'], entry['max_support']
            if not (entry['n_distinct_barcodes'] <= n <= s and maximum + n - 1 <= s <= maximum * n):
                fail('FRAGMENTS_V2_SUPPORT_INVALID')
            total += s; records += n
            number(total); number(records)
            for kind, suffix in (('bgzf', 'fragments.tsv.gz'), ('tabix', 'fragments.tsv.gz.tbi')):
                artifact = entry[kind]
                shape(artifact, ('path', 'sha256', 'size_bytes'))
                if artifact['path'] != 'libraries/' + entry['namespace'] + '/' + suffix:
                    fail()
                sha(artifact['sha256']); number(artifact['size_bytes'], high=2**63 - 1)
        if namespaces != sorted(set(namespaces)):
            fail()
        sha(value['fragments_identity_sha256'])
        if fragments_identity(value) != value['fragments_identity_sha256']:
            fail('FRAGMENTS_V2_DIGEST_MISMATCH')
        if len(canonical(value)) > MAX_BYTES:
            fail()
        return deepcopy(value)
    except FragmentsV2Error:
        raise
    except (ValueError, TypeError, KeyError, IndexError, UnicodeError, RecursionError, OverflowError):
        fail()


def canonical_fragments_manifest_v2_bytes(value):
    return canonical(validate_fragments_manifest_v2(value))


def _pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            fail()
        value[key] = item
    return value


def read_manifest_bytes(path, expected_sha256):
    sha(expected_sha256)
    try:
        with Path(path).open('rb') as stream:
            payload = stream.read(MAX_BYTES + 1)
        if len(payload) > MAX_BYTES:
            fail()
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            fail('FRAGMENTS_V2_DIGEST_MISMATCH')
        value = json.loads(payload.decode('utf-8'), object_pairs_hook=_pairs,
                           parse_constant=lambda _: fail())
        return payload, value
    except FragmentsV2Error:
        raise
    except (OSError, ValueError, TypeError, UnicodeError, RecursionError):
        fail()


def load_fragments_manifest_v2(path, *, expected_sha256):
    """Expected SHA binds exact manifest bytes, including noncanonical JSON."""
    return validate_fragments_manifest_v2(read_manifest_bytes(path, expected_sha256)[1])


def publish_fragments_manifest_v2(value, path, *, runtime):
    """Verify existing canonical artifacts, then atomically publish JSON only.

    No BGZF generation, sorting, indexing, import, overwrite or bundle recovery.
    Future producers own their private directory staging/atomic bundle rename.
    """
    from .scatac_fragments_v2_verifier import verify_fragments_v2
    path = Path(path)
    payload = canonical_fragments_manifest_v2_bytes(value)
    digest = hashlib.sha256(payload).hexdigest()
    if not path.is_absolute() or path != path.resolve() or not path.parent.is_dir() or path.exists():
        fail('FRAGMENTS_V2_PUBLICATION_CONFLICT')
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix='.fragments-v2-', dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        verified = verify_fragments_v2(temporary, expected_sha256=digest, runtime=runtime)
        protected = {p for p, _ in verified.snapshots}
        if str(path) in protected:
            fail('FRAGMENTS_V2_PUBLICATION_CONFLICT')
        verified.check_unchanged()
        os.link(temporary, path)  # Atomic no-clobber, including competing writers.
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return {'manifest_path': str(path), 'manifest_sha256': digest, 'artifact_schema_version': 2}
    except FileExistsError:
        fail('FRAGMENTS_V2_PUBLICATION_CONFLICT')
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
