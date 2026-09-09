"""Tiny v2 resources; ordinary cases fake only external tabix boundaries."""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tracemalloc
import zlib

import pytest

from agent.tools.data import scatac_fragments_v2 as m
from agent.tools.data import scatac_fragments_v2_verifier as v
from agent.tools.data import scatac_fragment_reader as reader
from agent.tools.data import scatac_reference as ref
from agent.tools.data._fragments_common import MAX_SUPPORT, MAX_TOTAL, canonical
from agent.tools.data.scatac_fragments_verifier import BGZF_EOF


def sha(data):
    return hashlib.sha256(data).hexdigest()


def bgzf(data):
    output = bytearray()
    for start in range(0, len(data), 32000):
        block = data[start:start + 32000]
        compressor = zlib.compressobj(wbits=-15)
        packed = compressor.compress(block) + compressor.flush()
        output += bytes.fromhex('1f8b08040000000000ff060042430200')
        output += struct.pack('<H', len(packed) + 25) + packed
        output += struct.pack('<II', zlib.crc32(block), len(block))
    return bytes(output) + BGZF_EOF


def artifact(path):
    return dict(path=str(path), sha256=sha(path.read_bytes()), size_bytes=path.stat().st_size)


def seal(value):
    value['fragments_identity_sha256'] = m.fragments_identity(value)
    return value


def write_manifest(path, value):
    payload = m.canonical_fragments_manifest_v2_bytes(seal(value))
    path.write_bytes(payload)
    return sha(payload)


@pytest.fixture
def factory(tmp_path):
    # FAI order deliberately differs from lexical chromosome order.
    fasta = tmp_path / 'reference.fa'
    fasta.write_text('>chr2\n' + 'A' * 1000 + '\n>chr1\n' + 'C' * 1000 + '\n')
    fai = tmp_path / 'reference.fa.fai'
    fai.write_text('chr2\t1000\t6\t1000\t1001\nchr1\t1000\t1013\t1000\t1001\n')
    bed = tmp_path / 'ccre.bed'; bed.write_text('chr2\t0\t1000\nchr1\t0\t1000\n')
    bundle = ref.build_scatac_reference_bundle(species='human', target_assembly='hg38',
        fasta_path=fasta, fai_path=fai, ccre_bed_path=bed)
    pointer = ref.publish_scatac_reference_bundle(bundle, tmp_path / 'reference.json')
    profile = tmp_path / 'synthetic-profile.txt'
    profile.write_text('Synthetic test profile only; no production qualification. Count supplied read-pair support.')
    sources = {}
    for role in ('external_fragments', 'bam', 'fastq', 'intake_manifest', 'library_context'):
        p = tmp_path / (role + '.source')
        p.write_text('Synthetic bound identity only, not a parsed ' + role)
        sources[role] = artifact(p)

    def make(*, strand=False, rows=None, two=False, kind='external_fragment_adoption', real_index=False):
        root = tmp_path / ('artifact-' + str(len(list(tmp_path.glob('artifact-*')))))
        root.mkdir()
        if rows is None:
            suffix = '\t+' if strand else ''
            rows = [f'chr2\t0\t10\tACGT-1\t301{suffix}\n',
                    f'chr1\t990\t1000\tACGT-1\t1{suffix}\n']
        data = ''.join(rows).encode()
        libraries = []
        for namespace in (('a', 'b') if two else ('a',)):
            directory = root / 'libraries' / namespace; directory.mkdir(parents=True)
            path = directory / 'fragments.tsv.gz'; path.write_bytes(bgzf(data))
            index = Path(str(path) + '.tbi')
            if real_index:
                subprocess.run(['/usr/bin/tabix', '-p', 'bed', str(path)], check=True, capture_output=True)
            else:
                # Test double answers from its own frozen inventory, not BGZF.
                index.write_bytes(data)
            roles = {'external_fragment_adoption': ('external_fragments',),
                     'bam_fragment_production': ('bam', 'intake_manifest', 'library_context'),
                     'fastq_fragment_production': ('fastq', 'intake_manifest', 'library_context')}[kind]
            provenance = dict(kind=kind, contract_version='fragment-producer-provenance.v1',
                profile={'id': 'synthetic-test-only.v1', 'resource': artifact(profile)},
                sources=[{'role': role, 'resource': sources[role]} for role in roles],
                producer_record=None, support={'unit': 'read_pairs', 'definition': 'Supplied read-pair support in the synthetic fixture.'},
                processing={key: {'status': 'unspecified', 'description': None} for key in m.PROCESSING_KEYS},
                source_selection='unspecified', verification_requirement='producer-specific-verifier-required.v1')
            if kind == 'fastq_fragment_production':
                record = directory / 'synthetic-producer-record.txt'
                record.write_text('Synthetic identity only; no FASTQ v2 producer is implemented or qualified.')
                provenance['producer_record'] = artifact(record)
            values = [row.split('\t') for row in rows]
            supports = [int(row[4]) for row in values]
            entry = dict(namespace=namespace,
                strand={'mode': 'present' if strand else 'absent', 'definition': 'Declared source strand; dot means unknown.' if strand else None},
                provenance=provenance, canonical_record_stream_sha256=sha(data),
                n_fragment_records=len(rows), sum_support=sum(supports),
                n_distinct_barcodes=len({r[3] for r in values}), max_support=max(supports))
            for key, file in (('bgzf', path), ('tabix', index)):
                entry[key] = artifact(file) | {'path': str(file.relative_to(root))}
            libraries.append(entry)
        value = seal(dict(artifact_type=m.ARTIFACT_TYPE, schema_version=2, contract_version=m.CONTRACT_VERSION,
            reference=dict(manifest_path=pointer['manifest_path'], manifest_sha256=pointer['manifest_sha256'],
                reference_identity_sha256=bundle.reference_identity_sha256, species='human', assembly='hg38',
                ordered_contig_sha256=bundle.genome.ordered_contig_sha256),
            semantics=deepcopy(m.SEMANTICS), libraries=libraries))
        manifest = root / 'manifest.json'; digest = write_manifest(manifest, value)
        return manifest, digest, value
    return make


@pytest.fixture
def runtime(monkeypatch):
    monkeypatch.setattr(v, 'verify_packaging', lambda _: None)
    def records(path):
        return Path(str(path) + '.tbi').read_bytes().splitlines(keepends=True)
    def names(runtime, path):
        return list(dict.fromkeys(row.split(b'\t')[0].decode() for row in records(path)))
    def query(runtime, path, region):
        if ':' in region:
            chrom, span = region.split(':'); left, right = map(int, span.split('-')); left -= 1
        else:
            chrom = region; left = 0; right = 2**29
        return sha(b''.join(row for row in records(path) if row.split(b'\t')[0].decode() == chrom
            and int(row.split(b'\t')[1]) < right and int(row.split(b'\t')[2]) > left))
    monkeypatch.setattr(v, '_tabix_contigs', names); monkeypatch.setattr(v, '_query', query)
    return v.FragmentVerificationRuntime()


@pytest.mark.parametrize('strand', [False, True])
@pytest.mark.parametrize('kind', ['external_fragment_adoption', 'bam_fragment_production', 'fastq_fragment_production'])
def test_roundtrip_scopes_and_two_namespaces(factory, runtime, strand, kind):
    path, digest, value = factory(strand=strand, two=True, kind=kind)
    assert m.load_fragments_manifest_v2(path, expected_sha256=digest) == value
    verified = v.verify_fragments_v2(path, expected_sha256=digest, runtime=runtime)
    assert verified.artifact_content == verified.bound_resource_identities == 'verified'
    assert verified.producer_history == 'not_verified'
    assert verified.producer_profile_qualification == 'not_established'
    view = reader.open_verified_fragments(path, expected_sha256=digest, runtime=runtime)
    assert view.contract_version == 'scatac-fragments.v2'
    assert [lib.namespace for lib in view.libraries] == ['a', 'b']
    a, b = list(view.iter_fragments('a')), list(view.iter_fragments('b'))
    assert a[0].cell_identity == ('a', 'ACGT-1') and b[0].cell_identity == ('b', 'ACGT-1')
    assert a[0].support == 301 and a[0].strand == ('+' if strand else None)
    assert [r.contig for r in a] == ['chr2', 'chr1']
    assert view.manifest == value and path.read_bytes() == canonical(value)
    copy = view.libraries[0].provenance; copy['kind'] = 'forged'
    assert view.libraries[0].provenance['kind'] == kind
    with pytest.raises(ValueError, match='NAMESPACE_UNKNOWN'):
        list(view.iter_fragments('absent'))


def test_strand_is_a_preserved_key_state(factory, runtime):
    rows = [f'chr2\t0\t10\tbarcode-1\t{n}\t{s}\n' for n, s in enumerate(('+', '-', '.'), 1)]
    path, digest, _ = factory(strand=True, rows=rows)
    view = reader.open_verified_fragments(path, expected_sha256=digest, runtime=runtime)
    assert [r.strand for r in view.iter_fragments('a')] == ['+', '-', '.']
    assert view.libraries[0].n_fragment_records == 3 and view.libraries[0].sum_support == 6


def test_exact_uint64_and_wide_aggregate(factory, runtime):
    rows = [f'chr2\t{i}\t{i+1}\tB-1\t{MAX_SUPPORT}\n' for i in range(2)]
    path, digest, _ = factory(rows=rows)
    view = reader.open_verified_fragments(path, expected_sha256=digest, runtime=runtime)
    assert view.libraries[0].sum_support == 2 * MAX_SUPPORT
    assert all(type(row.support) is int and row.support == MAX_SUPPORT for row in view.iter_fragments('a'))


@pytest.mark.parametrize('value', [0, -1, True, 1.0, MAX_SUPPORT + 1])
def test_support_bounds(factory, value):
    _, _, manifest = factory()
    manifest['libraries'][0]['max_support'] = value
    with pytest.raises(m.FragmentsV2Error):
        m.validate_fragments_manifest_v2(seal(manifest))


def test_checked_cross_library_aggregate(factory):
    _, _, value = factory(two=True)
    for entry in value['libraries']:
        entry.update(n_fragment_records=2**64 + 1, sum_support=MAX_TOTAL,
                     max_support=MAX_SUPPORT)
    with pytest.raises(m.FragmentsV2Error, match='SUPPORT_INVALID'):
        m.validate_fragments_manifest_v2(seal(value))


def test_stream_support_aggregate_overflow_is_independently_checked(factory, runtime, monkeypatch):
    path, digest, _ = factory()
    monkeypatch.setattr(v, 'MAX_TOTAL', 301)
    with pytest.raises(m.FragmentsV2Error, match='SUPPORT_INVALID'):
        v.verify_fragments_v2(path, expected_sha256=digest, runtime=runtime)


@pytest.mark.parametrize('row,strand', [
    ('chr2\t0\t10\tACGT-1\t1\t?\n', True),
    ('chr2\t0\t10\tACGT-1\t1\n', True),
    ('chr2\t0\t10\tACGT-1\t1\t+\n', False),
    ('unknown\t0\t10\tACGT-1\t1\n', False),
    ('chr2\t-1\t10\tACGT-1\t1\n', False),
    ('chr2\t1\t1\tACGT-1\t1\n', False),
    ('chr2\t0\t1001\tACGT-1\t1\n', False),
    ('chr2\t00\t10\tACGT-1\t1\n', False),
    ('chr2\t0\t10\t\t1\n', False),
    ('chr2\t0\t10\twhite space\t1\n', False),
    ('chr2\t0\t10\tbad\rbarcode\t1\n', False),
    ('chr2\t0\t10\tnonASCIIé\t1\n', False),
    ('chr2\t0\t10\t' + 'a'*257 + '\t1\n', False),
    ('chr2\t0\t10\tB-1\t01\n', False),
    ('chr2\t0\t10\tB-1\t+1\n', False),
])
def test_bad_records(factory, runtime, row, strand):
    path, digest, _ = factory(rows=[row], strand=strand)
    with pytest.raises(m.FragmentsV2Error):
        v.verify_fragments_v2(path, expected_sha256=digest, runtime=runtime)


@pytest.mark.parametrize('rows', [
    ['chr2\t0\t10\tB\t1\n'] * 2,
    ['chr2\t10\t20\tB\t1\n', 'chr2\t1\t2\tB\t1\n'],
    ['chr1\t0\t10\tB\t1\n', 'chr2\t0\t10\tB\t1\n'],
    ['chr2\t0\t10\tZ\t1\n', 'chr2\t0\t10\tA\t1\n'],
])
def test_order_and_duplicate_keys(factory, runtime, rows):
    path, digest, _ = factory(rows=rows)
    with pytest.raises(m.FragmentsV2Error, match='RECORD_INVALID'):
        v.verify_fragments_v2(path, expected_sha256=digest, runtime=runtime)


@pytest.mark.parametrize('strands', [('+', '+'), ('-', '+')])
def test_stranded_duplicate_or_wrong_order(factory, runtime, strands):
    path, digest, _ = factory(strand=True, rows=[f'chr2\t0\t10\tB\t1\t{s}\n' for s in strands])
    with pytest.raises(m.FragmentsV2Error, match='RECORD_INVALID'):
        v.verify_fragments_v2(path, expected_sha256=digest, runtime=runtime)


@pytest.mark.parametrize('field', ['manifest_sha256', 'reference_identity_sha256', 'ordered_contig_sha256', 'assembly'])
def test_wrong_reference(factory, runtime, field):
    path, _, value = factory()
    value['reference'][field] = 'hg19' if field == 'assembly' else '0'*64
    with pytest.raises(m.FragmentsV2Error):
        digest = write_manifest(path, value)
        v.verify_fragments_v2(path, expected_sha256=digest, runtime=runtime)


@pytest.mark.parametrize('mutation', ['kind', 'unknown_field', 'status', 'support_unit', 'source_role', 'profile_missing', 'qualified_claim'])
def test_closed_provenance(factory, mutation):
    _, _, value = factory(); p = value['libraries'][0]['provenance']
    if mutation == 'kind': p['kind'] = 'sinto'
    elif mutation == 'unknown_field': p['arbitrary_json'] = {'science': 'guess'}
    elif mutation == 'status': p['processing']['tn5']['status'] = 'verified'
    elif mutation == 'support_unit': p['support']['unit'] = 'molecules'
    elif mutation == 'source_role': p['sources'][0]['role'] = 'fastq'
    elif mutation == 'profile_missing': p['profile'].pop('resource')
    else: p['verification_requirement'] = 'already-qualified'
    with pytest.raises(m.FragmentsV2Error):
        m.validate_fragments_manifest_v2(seal(value))


@pytest.mark.parametrize('mutation', ['missing_record', 'bam_role', 'missing_context', 'duplicate_intake'])
def test_fastq_provenance_cannot_omit_rich_record_or_mislabel_sources(factory, mutation):
    _, _, value = factory(kind='fastq_fragment_production')
    p = value['libraries'][0]['provenance']
    if mutation == 'missing_record': p['producer_record'] = None
    elif mutation == 'bam_role': p['sources'][0]['role'] = 'bam'
    elif mutation == 'missing_context': p['sources'].pop()
    else:
        duplicate = deepcopy(p['sources'][1])
        duplicate['resource']['path'] += '.duplicate'
        p['sources'].insert(2, duplicate)
    with pytest.raises(m.FragmentsV2Error, match='PROVENANCE_INVALID'):
        m.validate_fragments_manifest_v2(seal(value))


@pytest.mark.parametrize('mutation', ['namespace', 'duplicate_namespace', 'path_traversal', 'extra_root',
    'extra_library', 'extra_reference', 'extra_strand', 'extra_source', 'bad_claim', 'missing_claim',
    'bad_profile_hash', 'bad_resource_size', 'bad_count', 'bad_total', 'bad_strand_mode', 'source_duplicate'])
def test_nested_contract_failures(factory, mutation):
    _, _, value = factory(two=True); e = value['libraries'][0]; p = e['provenance']
    if mutation == 'namespace': e['namespace'] = '../unsafe'
    elif mutation == 'duplicate_namespace': value['libraries'][1]['namespace'] = 'a'
    elif mutation == 'path_traversal': e['bgzf']['path'] = '../elsewhere.gz'
    elif mutation == 'extra_root': value['unknown'] = 1
    elif mutation == 'extra_library': e['called_cells'] = 1
    elif mutation == 'extra_reference': value['reference']['chromap_index'] = '/made-up'
    elif mutation == 'extra_strand': e['strand']['infer'] = True
    elif mutation == 'extra_source': p['sources'][0]['extra'] = 'bad'
    elif mutation == 'bad_claim': p['processing']['tn5']['description'] = 'Unknown cannot have a declared value.'
    elif mutation == 'missing_claim': p['processing']['tn5'] = {'status': 'declared', 'description': None}
    elif mutation == 'bad_profile_hash': p['profile']['resource']['sha256'] = 'bad'
    elif mutation == 'bad_resource_size': p['profile']['resource']['size_bytes'] = True
    elif mutation == 'bad_count': e['n_fragment_records'] = 1.0
    elif mutation == 'bad_total': e['sum_support'] = MAX_TOTAL + 1
    elif mutation == 'bad_strand_mode': e['strand']['mode'] = 'auto'
    else: p['sources'] *= 2
    with pytest.raises(m.FragmentsV2Error):
        m.validate_fragments_manifest_v2(seal(value))


def test_explicit_processing_claims_are_retained_but_not_verified(factory, runtime):
    path, _, value = factory(); p = value['libraries'][0]['provenance']
    p['processing']['tn5'] = {'status': 'declared', 'description': 'Source declares +4/-5 adjustment; adoption preserves coordinates.'}
    p['source_selection'] = 'producer_subset'
    record = path.parent / 'producer-record.txt'; record.write_text('Synthetic identity, no execution proof.')
    p['producer_record'] = artifact(record)
    digest = write_manifest(path, value)
    result = v.verify_fragments_v2(path, expected_sha256=digest, runtime=runtime)
    assert result.manifest['libraries'][0]['provenance'] == p
    assert result.producer_history == 'not_verified'


@pytest.mark.parametrize('field', ['bgzf', 'tabix', 'profile', 'source'])
def test_bound_file_corruption(factory, runtime, field):
    path, digest, value = factory(); entry = value['libraries'][0]
    target = (path.parent / entry[field]['path'] if field in ('bgzf', 'tabix') else
              Path(entry['provenance']['profile']['resource']['path']) if field == 'profile' else
              Path(entry['provenance']['sources'][0]['resource']['path']))
    target.write_bytes(target.read_bytes() + b'corrupt')
    with pytest.raises(m.FragmentsV2Error):
        v.verify_fragments_v2(path, expected_sha256=digest, runtime=runtime)


@pytest.mark.parametrize('mode', ['gzip', 'eof', 'crc', 'partial_line', 'trailing', 'header'])
def test_rehashed_bgzf_corruption(factory, runtime, mode):
    path, _, value = factory(); e = value['libraries'][0]; target = path.parent / e['bgzf']['path']
    data = target.read_bytes(); raw = gzip.decompress(data)
    if mode == 'gzip': data = gzip.compress(raw)
    elif mode == 'eof': data = data[:-28]
    elif mode == 'crc': data = data[:-36] + bytes([data[-36] ^ 1]) + data[-35:]
    elif mode == 'partial_line': data = bgzf(raw[:-1])
    elif mode == 'trailing': data += b'extra'
    else: data = bgzf(b'#header\n' + raw)
    target.write_bytes(data); e['bgzf'] = artifact(target) | {'path': e['bgzf']['path']}
    digest = write_manifest(path, value)
    with pytest.raises(m.FragmentsV2Error):
        v.verify_fragments_v2(path, expected_sha256=digest, runtime=runtime)


def test_stale_rehashed_index_fails_functional_check(factory, runtime):
    path, _, value = factory(); e = value['libraries'][0]; index = path.parent / e['tabix']['path']
    index.write_bytes(index.read_bytes().replace(b'301', b'302'))
    e['tabix'] = artifact(index) | {'path': e['tabix']['path']}
    digest = write_manifest(path, value)
    with pytest.raises(m.FragmentsV2Error, match='INDEX_MISMATCH'):
        v.verify_fragments_v2(path, expected_sha256=digest, runtime=runtime)


def test_narrow_index_queries_are_checked_separately(factory, runtime, monkeypatch):
    path, digest, _ = factory(); query = v._query
    monkeypatch.setattr(v, '_query', lambda r, p, region: '0'*64 if ':' in region else query(r, p, region))
    with pytest.raises(m.FragmentsV2Error, match='INDEX_MISMATCH'):
        v.verify_fragments_v2(path, expected_sha256=digest, runtime=runtime)


def test_source_mutation_during_verification_fails(factory, runtime, monkeypatch):
    path, digest, value = factory(); original = v._verify_stream
    profile = Path(value['libraries'][0]['provenance']['profile']['resource']['path'])
    def changed(*args):
        result = original(*args)
        profile.write_bytes(profile.read_bytes())
        return result
    monkeypatch.setattr(v, '_verify_stream', changed)
    with pytest.raises(m.FragmentsV2Error, match='SOURCE_CHANGED'):
        v.verify_fragments_v2(path, expected_sha256=digest, runtime=runtime)


def test_changed_reference_resource_and_tbi_limit(factory, runtime, monkeypatch):
    path, digest, value = factory()
    monkeypatch.setattr(v, 'TBI_LIMIT', 999)
    with pytest.raises(m.FragmentsV2Error, match='INDEX_POLICY_UNSUPPORTED'):
        v.verify_fragments_v2(path, expected_sha256=digest, runtime=runtime)
    bundle = ref.load_scatac_reference_bundle(value['reference']['manifest_path'])[1]
    Path(bundle.ccre.bed.path).write_text('chr2\t0\t999\n')
    with pytest.raises(m.FragmentsV2Error, match='REFERENCE_MISMATCH'):
        v.verify_fragments_v2(path, expected_sha256=digest, runtime=runtime)


def test_independent_verifier_recounts_forged_summary(factory, runtime, monkeypatch):
    path, _, value = factory(); value['libraries'][0]['sum_support'] += 1
    digest = write_manifest(path, value)
    monkeypatch.setattr(reader, '_record', lambda *a: pytest.fail('Reader used as verifier'))
    with pytest.raises(m.FragmentsV2Error, match='STREAM_MISMATCH'):
        v.verify_fragments_v2(path, expected_sha256=digest, runtime=runtime)


@pytest.mark.parametrize('mode', ['digest', 'duplicate', 'nan', 'schema', 'utf16', 'oversize'])
def test_strict_manifest_loading(factory, mode):
    path, digest, _ = factory(); payload = path.read_bytes()
    if mode == 'digest': digest = '0'*64
    else:
        if mode == 'duplicate': payload = payload.replace(b'"schema_version":2', b'"schema_version":2,"schema_version":2')
        elif mode == 'nan': payload = payload.replace(b'"schema_version":2', b'"schema_version":NaN')
        elif mode == 'schema': payload = payload.replace(b'"schema_version":2', b'"schema_version":true')
        elif mode == 'utf16': payload = payload.decode().encode('utf-16')
        else: payload = b' ' * (m.MAX_BYTES + 1)
        path.write_bytes(payload); digest = sha(payload)
    with pytest.raises(m.FragmentsV2Error):
        m.load_fragments_manifest_v2(path, expected_sha256=digest)


def test_noncanonical_json_byte_identity_and_no_source_io(factory, runtime, monkeypatch):
    path, _, value = factory(); payload = json.dumps(value, indent=2).encode(); path.write_bytes(payload)
    view = reader.open_verified_fragments(path, expected_sha256=sha(payload), runtime=runtime)
    assert view.verification.manifest_bytes == payload
    assert sha(view.verification.manifest_bytes) == view.verification.manifest_sha256
    monkeypatch.setattr(ref, 'reinspect_scatac_reference_bundle_sources', lambda *a: pytest.fail('IO during load'))
    assert m.load_fragments_manifest_v2(path, expected_sha256=sha(payload)) == value
    assert m.canonical_fragments_manifest_v2_bytes(value) == canonical(value)


@pytest.mark.parametrize('encoding', ['utf-16', 'utf-8-sig'])
def test_reader_legacy_encoding_dispatch_does_not_relax_v2(factory, runtime, encoding):
    path, _, value = factory()
    payload = json.dumps(value).encode(encoding); path.write_bytes(payload)
    with pytest.raises(m.FragmentsV2Error):
        reader.open_verified_fragments(path, expected_sha256=sha(payload), runtime=runtime)


def test_identity_preserves_science_excludes_physical_encoding(factory):
    _, _, value = factory(); changed = deepcopy(value)
    changed['libraries'][0]['bgzf']['sha256'] = '0'*64
    changed['libraries'][0]['provenance']['profile']['resource']['path'] = '/relocated/profile'
    assert m.fragments_identity(changed) == value['fragments_identity_sha256']
    changed['libraries'][0]['provenance']['support']['definition'] = 'Different counted population.'
    assert m.fragments_identity(changed) != value['fragments_identity_sha256']


@pytest.mark.parametrize('when', ['before', 'during'])
def test_reader_rejects_drift(factory, runtime, when):
    path, digest, value = factory()
    view = reader.open_verified_fragments(path, expected_sha256=digest, runtime=runtime)
    stream = view.iter_fragments('a')
    if when == 'during': next(stream)
    target = path.parent / value['libraries'][0]['bgzf']['path']
    target.write_bytes(target.read_bytes())  # Same bytes still changes the trusted-local snapshot.
    with pytest.raises(m.FragmentsV2Error, match='SOURCE_CHANGED'):
        list(stream)


def test_streaming_path_has_bounded_record_storage(factory, runtime, monkeypatch):
    # Many records/barcodes; measured reader memory excludes fixture creation
    # and up-front verification. A materialized FragmentRecord list exceeds 10 MB.
    rows = [f'chr2\t0\t10\tB{i:06}\t1\n' for i in range(50000)]
    path, digest, _ = factory(rows=rows)
    view = reader.open_verified_fragments(path, expected_sha256=digest, runtime=runtime)
    original = reader._record; seen = 0
    def counted(*args):
        nonlocal seen
        seen += 1
        return original(*args)
    monkeypatch.setattr(reader, '_record', counted)
    tracemalloc.start()
    try:
        stream = view.iter_fragments('a')
        assert seen == 0
        assert next(stream).barcode_identifier == 'B000000' and seen == 1
        assert sum(1 for _ in stream) == 49999
        assert tracemalloc.get_traced_memory()[1] < 2 * 1024 * 1024
    finally:
        tracemalloc.stop()


def test_atomic_manifest_publication_and_conflict(factory, runtime):
    path, digest, value = factory(); original = path.read_bytes()
    target = path.parent / 'published.json'
    result = m.publish_fragments_manifest_v2(value, target, runtime=runtime)
    assert result['manifest_sha256'] == digest and target.read_bytes() == original
    with pytest.raises(m.FragmentsV2Error, match='PUBLICATION_CONFLICT'):
        m.publish_fragments_manifest_v2(value, target, runtime=runtime)
    assert path.read_bytes() == original


def test_publication_verification_failure_leaves_no_manifest(factory, runtime):
    path, _, value = factory(); target = path.parent / 'published.json'
    (path.parent / value['libraries'][0]['tabix']['path']).write_bytes(b'wrong')
    with pytest.raises(m.FragmentsV2Error):
        m.publish_fragments_manifest_v2(value, target, runtime=runtime)
    assert not target.exists() and not list(path.parent.glob('.fragments-v2-*'))


def test_publication_competing_writer_is_not_overwritten(factory, runtime, monkeypatch):
    path, _, value = factory(); target = path.parent / 'published.json'; link = m.os.link
    def compete(source, destination):
        Path(destination).write_bytes(b'other writer')
        link(source, destination)
    monkeypatch.setattr(m.os, 'link', compete)
    with pytest.raises(m.FragmentsV2Error, match='PUBLICATION_CONFLICT'):
        m.publish_fragments_manifest_v2(value, target, runtime=runtime)
    assert target.read_bytes() == b'other writer'
    assert not list(path.parent.glob('.fragments-v2-*'))


def test_reader_checks_drift_when_closed_early(factory, runtime):
    path, digest, _ = factory()
    view = reader.open_verified_fragments(path, expected_sha256=digest, runtime=runtime)
    stream = view.iter_fragments('a'); next(stream)
    path.write_bytes(path.read_bytes())
    with pytest.raises(m.FragmentsV2Error, match='SOURCE_CHANGED'):
        stream.close()


@pytest.mark.parametrize('version', [None, True, 3, '2'])
def test_common_reader_rejects_unknown_version_without_fallback(factory, runtime, version):
    path, _, value = factory(); value['schema_version'] = version
    payload = canonical(value); path.write_bytes(payload)
    with pytest.raises(ValueError, match='CONTRACT_UNSUPPORTED'):
        reader.open_verified_fragments(path, expected_sha256=sha(payload), runtime=runtime)


@pytest.mark.parametrize('strand', [False, True])
def test_real_qualified_tabix_and_corruption(factory, strand):
    if not Path('/usr/bin/tabix').is_file():
        pytest.skip('Qualified local packaging runtime unavailable')
    path, digest, value = factory(strand=strand, real_index=True)
    runtime = v.FragmentVerificationRuntime()
    view = reader.open_verified_fragments(path, expected_sha256=digest, runtime=runtime)
    assert sum(row.support for row in view.iter_fragments('a')) == 302
    index = path.parent / value['libraries'][0]['tabix']['path']
    index.write_bytes(b'not an index')
    value['libraries'][0]['tabix'] = artifact(index) | {'path': value['libraries'][0]['tabix']['path']}
    digest = write_manifest(path, value)
    with pytest.raises(m.FragmentsV2Error, match='INDEX_MISMATCH'):
        v.verify_fragments_v2(path, expected_sha256=digest, runtime=runtime)
