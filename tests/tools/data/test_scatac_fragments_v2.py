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
BGZF_EOF = bytes.fromhex('1f8b08040000000000ff0600424302001b0003000000000000000000')


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


def test_identity_preserves_science_excludes_physical_encoding(factory):
    _, _, value = factory(); changed = deepcopy(value)
    changed['libraries'][0]['bgzf']['sha256'] = '0'*64
    changed['libraries'][0]['provenance']['profile']['resource']['path'] = '/relocated/profile'
    assert m.fragments_identity(changed) == value['fragments_identity_sha256']
    changed['libraries'][0]['provenance']['support']['definition'] = 'Different counted population.'
    assert m.fragments_identity(changed) != value['fragments_identity_sha256']
