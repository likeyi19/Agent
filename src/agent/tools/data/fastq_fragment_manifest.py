"""Closed FASTQ/Chromap producer record for current fragments v2.

Retains qualified M11.2 scientific provenance; this is not a fragment artifact
or a validator/adapter for the retired fragments v1 contract.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

from . import _chromap as c
from ._fragments_binding import FragmentInputs
from ._fragments_common import (MAX_SUPPORT, MAX_TOTAL, PACKAGING_POLICY, canonical, fail)
from . import scatac_fragments_v2 as v2

ARTIFACT_TYPE = "agent.fastq-fragment-production"
CONTRACT_VERSION = "fastq-fragment-production.v1"
SEMANTICS = {
    'coordinates': '0-based-half-open',
    'order': 'fai-rank,start-numeric,end-numeric,barcode-ascii.v1',
    'support': c.SUPPORT_SEMANTICS,
    'tn5': 'Chromap-only:+4-start,-5-end;no-second-shift',
    'distinct_barcodes': 'observed-accepted-fragment-tokens;not-called-cells',
    'individual_support_max': MAX_SUPPORT,
    'aggregate_support_max': MAX_TOTAL,
    'stream_identity': 'sha256(canonical-utf8-five-tab-fields-with-LF-in-order)',
}

PROFILE = dict(artifact_type="agent.fastq-fragment-profile", schema_version=1,
    profile_id=c.BACKEND_POLICY, backend_policy=c.BACKEND_POLICY,
    executable_sha256=c.QUALIFIED_EXECUTABLE_SHA256, mapping_policy_sha256=c.policy_sha256(),
    index_policy=c.INDEX_POLICY, semantics=SEMANTICS, packaging=PACKAGING_POLICY)
PROFILE_BYTES = canonical(PROFILE)

MAX_BYTES = 8 * 1024 * 1024
BINDING_KEYS = 'namespace group_ids library_identity role_binding_sha256 whitelist_sha256 whitelist_set_sha256 barcode_length'.split()
SUMMARY_KEYS = 'canonical_record_stream_sha256 n_fragment_records sum_support n_distinct_barcodes max_support'.split()


def _shape(value, keys):
    if type(value) is not dict or set(value) != set(keys):
        fail('FRAGMENTS_VERIFICATION_MISMATCH')


def _hash(value):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
        fail('FRAGMENTS_VERIFICATION_MISMATCH')


def _number(value, low=1, high=MAX_TOTAL):
    if type(value) is not int or not low <= value <= high:
        fail('FRAGMENTS_SUPPORT_INVALID')


def _validate_record(value):
    _shape(value, 'artifact_type schema_version contract_version inputs lineage backend mapping_policy packaging_policy semantics libraries profile_sha256'.split())
    if (value['artifact_type'] != ARTIFACT_TYPE or type(value['schema_version']) is not int
            or value['schema_version'] != 1 or value['contract_version'] != CONTRACT_VERSION
            or value['profile_sha256'] != hashlib.sha256(PROFILE_BYTES).hexdigest()
            or canonical(value['semantics']) != canonical(SEMANTICS)
            or canonical(value['packaging_policy']) != canonical(PACKAGING_POLICY)
            or canonical(value['mapping_policy']) != canonical({'flags': c.FIXED_FLAGS,
                'implementation': c.IMPLEMENTATION_POLICY, 'upstream_tag': 'v0.3.2'})):
        fail('FRAGMENTS_VERIFICATION_MISMATCH')
    try:
        c.validate_backend(value['backend'])
    except ValueError:
        fail('FRAGMENTS_VERIFICATION_MISMATCH')
    _shape(value['inputs'], FragmentInputs.__dataclass_fields__)
    for key, item in value['inputs'].items():
        if key.endswith('_sha256'):
            _hash(item)
        elif (type(item) is not str or not Path(item).is_absolute()
                or any(ord(ch) < 32 or ord(ch) == 127 for ch in item)
                or '..' in Path(item).parts or str(Path(item)) != item):
            fail('FRAGMENTS_VERIFICATION_MISMATCH')
    lineage = value['lineage']
    _shape(lineage, 'intake_contract context_identity reference_identity species assembly ordered_contig_sha256 index_identity index_file_sha256 index_policy'.split())
    from .raw_scatac_manifest import RAW_INTAKE_CONTRACT_VERSION
    if (lineage['intake_contract'] != RAW_INTAKE_CONTRACT_VERSION
            or lineage['species'] not in ('human', 'mouse')
            or lineage['assembly'] != {'human': 'hg38', 'mouse': 'mm10'}[lineage['species']]
            or lineage['index_policy'] != c.INDEX_POLICY):
        fail('FRAGMENTS_VERIFICATION_MISMATCH')
    for key in ('context_identity', 'reference_identity', 'ordered_contig_sha256', 'index_identity', 'index_file_sha256'):
        _hash(lineage[key])
    libraries = value['libraries']
    if type(libraries) is not list or not 1 <= len(libraries) <= 4096:
        fail('FRAGMENTS_VERIFICATION_MISMATCH')
    namespaces = []; all_groups = []; total = 0
    for entry in libraries:
        _shape(entry, BINDING_KEYS + SUMMARY_KEYS + ['bgzf', 'tabix', 'scans', 'sources'])
        if type(entry['sources']) is not list or not entry['sources']:
            fail('FRAGMENTS_VERIFICATION_MISMATCH')
        for source in entry['sources']:
            v2.resource(source)
        namespace = entry['namespace']
        if type(namespace) is not str or re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', namespace) is None:
            fail('FRAGMENTS_VERIFICATION_MISMATCH')
        namespaces.append(namespace)
        groups = entry['group_ids']
        if type(groups) is not list or not 1 <= len(groups) <= 4096 or groups != sorted(set(groups)):
            fail('FRAGMENTS_GROUP_MISMATCH')
        for g in groups:
            if type(g) is not str or re.fullmatch('group:[0-9a-f]{64}', g) is None:
                fail('FRAGMENTS_GROUP_MISMATCH')
        all_groups.extend(groups)
        for key in ('library_identity', 'role_binding_sha256', 'whitelist_sha256', 'whitelist_set_sha256', 'canonical_record_stream_sha256'):
            _hash(entry[key])
        _number(entry['barcode_length'], high=32)
        for key in ('n_fragment_records', 'sum_support', 'n_distinct_barcodes'):
            _number(entry[key])
        _number(entry['max_support'], high=MAX_SUPPORT)
        if not entry['n_distinct_barcodes'] <= entry['n_fragment_records'] <= entry['sum_support'] or entry['max_support'] > entry['sum_support']:
            fail('FRAGMENTS_SUPPORT_INVALID')
        total += entry['sum_support']; _number(total)
        for key, suffix in (('bgzf', 'fragments.tsv.gz'), ('tabix', 'fragments.tsv.gz.tbi')):
            artifact = entry[key]
            _shape(artifact, ['path', 'sha256', 'size_bytes'])
            if artifact['path'] != 'libraries/' + namespace + '/' + suffix:
                fail('FRAGMENTS_VERIFICATION_MISMATCH')
            _hash(artifact['sha256']); _number(artifact['size_bytes'])
        _shape(entry['scans'], groups)
        for scan in entry['scans'].values():
            _shape(scan, ['record_count', 'normalized_id_sha256', 'decoded_role_sha256', 'identity_policy'])
            _number(scan['record_count']); _hash(scan['normalized_id_sha256'])
            if scan['identity_policy'] != 'decoded-record-bytes;sha256-concatenated-M10-id-hashes.v1':
                fail('FRAGMENTS_VERIFICATION_MISMATCH')
            if type(scan['decoded_role_sha256']) is not list or len(scan['decoded_role_sha256']) != 3:
                fail('FRAGMENTS_VERIFICATION_MISMATCH')
            for h in scan['decoded_role_sha256']:
                _hash(h)
    if namespaces != sorted(set(namespaces)) or len(all_groups) != len(set(all_groups)):
        fail('FRAGMENTS_GROUP_MISMATCH')
    return deepcopy(value)


def validate_record(value):
    from ._fragments_common import FragmentsError
    try:
        return _validate_record(value)
    except FragmentsError:
        raise
    except (TypeError, ValueError, KeyError, IndexError, UnicodeError, RecursionError):
        fail('FRAGMENTS_VERIFICATION_MISMATCH')


def _pairs(pairs):
    result = {}
    for k, v in pairs:
        if k in result:
            fail('FRAGMENTS_VERIFICATION_MISMATCH')
        result[k] = v
    return result


def load_record(path, *, expected_sha256=None):
    try:
        with Path(path).open('rb') as stream:
            payload = stream.read(MAX_BYTES + 1)
        if len(payload) > MAX_BYTES:
            fail('FRAGMENTS_VERIFICATION_MISMATCH')
        if expected_sha256 is not None:
            _hash(expected_sha256)
            if hashlib.sha256(payload).hexdigest() != expected_sha256:
                fail('FRAGMENTS_VERIFICATION_MISMATCH')
        value = json.loads(payload, object_pairs_hook=_pairs,
            parse_constant=lambda _: fail('FRAGMENTS_VERIFICATION_MISMATCH'))
        return validate_record(value)
    except (OSError, UnicodeError, RecursionError, TypeError, ValueError):
        fail('FRAGMENTS_VERIFICATION_MISMATCH')


def canonical_record_bytes(value):
    return canonical(validate_record(value))


def resource(path):
    """Exact physical resource identity; source snapshots surround callers."""
    path = Path(path)
    return dict(path=str(path), sha256=c.sha256(path), size_bytes=path.stat().st_size)


def source_resources(groups):
    return [resource(path) for path in sorted({p for group in groups for _, p in group.files})]


def reference(record):
    return dict(manifest_path=record['inputs']['reference_path'],
        manifest_sha256=record['inputs']['reference_sha256'],
        reference_identity_sha256=record['lineage']['reference_identity'],
        species=record['lineage']['species'], assembly=record['lineage']['assembly'],
        ordered_contig_sha256=record['lineage']['ordered_contig_sha256'])


def provenance(record, entry, resources):
    sources = [{'role': 'fastq', 'resource': source} for source in entry['sources']]
    sources += [{'role': role, 'resource': resources[key]} for role, key in
                (('intake_manifest', 'intake_path'), ('library_context', 'context_path'))]
    sources.sort(key=lambda item: (item['role'], item['resource']['sha256'], item['resource']['path']))
    return dict(kind='fastq_fragment_production', contract_version='fragment-producer-provenance.v1',
        profile={'id': c.BACKEND_POLICY, 'resource': resources['profile']},
        producer_record=resources['record'], sources=sources,
        support={'unit': 'paired_mappings', 'definition': c.SUPPORT_SEMANTICS},
        processing={key: {'status': 'declared', 'description': description} for key, description in (
            ('tn5', SEMANTICS['tn5']),
            ('deduplication', c.SUPPORT_SEMANTICS),
            ('mapq_filtering', 'Qualified fixed Chromap ATAC mapping policy; not the Agent BAM MAPQ policy.'),
            ('barcode_correction', 'Qualified Chromap correction with exact bound whitelist and library context.'))},
        source_selection='producer_subset', verification_requirement='producer-specific-verifier-required.v1')


def build_manifest(record, root):
    # The global producer record grows with libraries; hash shared resources once.
    resources = {key: resource(record['inputs'][key]) for key in ('intake_path', 'context_path')}
    resources.update(profile=resource(root / 'profile.json'), record=resource(root / 'production.json'))
    entries = []
    for entry in record['libraries']:
        entries.append({key: entry[key] for key in ('namespace', 'bgzf', 'tabix', *v2.SUMMARY_KEYS)} |
            {'strand': {'mode': 'absent', 'definition': None}, 'provenance': provenance(record, entry, resources)})
    value = dict(artifact_type=v2.ARTIFACT_TYPE, schema_version=2, contract_version=v2.CONTRACT_VERSION,
        reference=reference(record), semantics=v2.SEMANTICS, libraries=entries)
    value['fragments_identity_sha256'] = v2.fragments_identity(value)
    return v2.validate_fragments_manifest_v2(value)


def relocate_manifest(value, root):
    """Only relocate owned profile/record locators during atomic publication."""
    value = deepcopy(value)
    for entry in value['libraries']:
        p = entry['provenance']
        p['profile']['resource']['path'] = str(root / 'profile.json')
        p['producer_record']['path'] = str(root / 'production.json')
    return v2.canonical_fragments_manifest_v2_bytes(value)


def load_manifest(path, *, expected_sha256=None):
    """Current v2 FASTQ artifact only; historical v1 is never reinterpreted."""
    try:
        payload, value = v2.read_manifest_bytes(path, expected_sha256 or c.sha256(path))
        if type(value) is dict and value.get('artifact_type') == v2.ARTIFACT_TYPE and value.get('schema_version') == 1:
            fail('FRAGMENTS_CONTRACT_RETIRED')
        return v2.validate_fragments_manifest_v2(value)
    except v2.FragmentsV2Error:
        fail('FRAGMENTS_VERIFICATION_MISMATCH')


def record_for_manifest(value, root):
    if not value['libraries']:
        fail('FRAGMENTS_VERIFICATION_MISMATCH')
    profile = resource(root / 'profile.json'); producer = resource(root / 'production.json')
    for entry in value['libraries']:
        p = entry['provenance']
        if (p['kind'] != 'fastq_fragment_production' or p['profile']['id'] != c.BACKEND_POLICY
                or p['profile']['resource'] != profile
                or p['profile']['resource']['sha256'] != hashlib.sha256(PROFILE_BYTES).hexdigest()
                or p['producer_record'] != producer):
            fail('FRAGMENTS_VERIFICATION_MISMATCH')
    return load_record(root / 'production.json', expected_sha256=value['libraries'][0]['provenance']['producer_record']['sha256'])
