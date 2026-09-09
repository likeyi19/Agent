"""Closed, lightweight ScATACFragmentsManifest.v1 JSON domain.

Load performs no scientific source IO. Portable identity excludes input locator
paths and physical output encodings, but retains exact upstream artifact hashes;
M10's path-sensitive identity means this is not total relocation independence.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re

from . import _chromap as c
from ._fragments_binding import FragmentInputs
from ._fragments_common import (ARTIFACT_TYPE, CONTRACT_VERSION, MAX_SUPPORT, MAX_TOTAL,
    SEMANTICS, PACKAGING_POLICY, canonical, digest, fail)

MAX_BYTES = 8 * 1024 * 1024
BINDING_KEYS = 'namespace group_ids library_identity role_binding_sha256 whitelist_sha256 whitelist_set_sha256 barcode_length'.split()
SUMMARY_KEYS = 'canonical_record_stream_sha256 n_fragment_records sum_support n_distinct_barcodes max_support'.split()


def portable_identity(value):
    value = deepcopy(value)
    value.pop('fragments_identity_sha256', None)
    value['inputs'] = {k: v for k, v in value['inputs'].items() if not k.endswith('_path')}
    for entry in value['libraries']:
        entry.pop('bgzf', None); entry.pop('tabix', None)
    return hashlib.sha256(b'agent.scatac-fragments.v1\0' + canonical(value)).hexdigest()


def _shape(value, keys):
    if type(value) is not dict or set(value) != set(keys):
        fail('FRAGMENTS_VERIFICATION_MISMATCH')


def _hash(value):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
        fail('FRAGMENTS_VERIFICATION_MISMATCH')


def _number(value, low=1, high=MAX_TOTAL):
    if type(value) is not int or not low <= value <= high:
        fail('FRAGMENTS_SUPPORT_INVALID')


def _validate_fragments_manifest(value):
    _shape(value, 'artifact_type schema_version contract_version inputs lineage backend mapping_policy packaging_policy semantics libraries fragments_identity_sha256'.split())
    if (value['artifact_type'] != ARTIFACT_TYPE or type(value['schema_version']) is not int
            or value['schema_version'] != 1 or value['contract_version'] != CONTRACT_VERSION
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
        _shape(entry, BINDING_KEYS + SUMMARY_KEYS + ['bgzf', 'tabix', 'scans'])
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
    _hash(value['fragments_identity_sha256'])
    if portable_identity(value) != value['fragments_identity_sha256']:
        fail('FRAGMENTS_VERIFICATION_MISMATCH')
    return deepcopy(value)


def validate_fragments_manifest(value):
    from ._fragments_common import FragmentsError
    try:
        return _validate_fragments_manifest(value)
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


def load_fragments_manifest(path, *, expected_sha256=None):
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
        return validate_fragments_manifest(value)
    except (OSError, UnicodeError, RecursionError, TypeError, ValueError):
        fail('FRAGMENTS_VERIFICATION_MISMATCH')


def canonical_fragments_manifest_bytes(value):
    return canonical(validate_fragments_manifest(value))
