"""Closed external-adoption policy and record; no source inference or raw intake."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from . import scatac_fragments_v2 as v2
from ._fragments_common import canonical, MAX_SUPPORT, MAX_TOTAL

PROFILE_ID = '10x-atac-fragments.v1'
ADOPTION_CONTRACT = 'external-fragment-adoption.v1'
TRANSFORMATION = 'validate-sort-serialize-bgzf-tbi;no-biological-transformation.v1'
SUPPORT = {'unit': 'read_pairs', 'definition':
    'Read pairs associated with the supplied unique fragment, including the representative and duplicate read pairs, under the declared 10x ATAC source semantics.'}
STRAND_DEFINITION = 'Declared 10x adapter-origin strand: R1 +, R2 -; supplied dot is unknown, never inferred.'
MAX_LINE = 65536
MAX_HEADERS = 1024
MAX_HEADER_BYTES = 1024 * 1024
SELECTION = {'full_export': 'all_source_records', 'subset_export': 'producer_subset', 'unknown': 'unspecified'}
PROFILE = {
    'artifact_type': 'agent.external-fragment-profile', 'schema_version': 1,
    'profile_id': PROFILE_ID, 'coordinates': '0-based-half-open;already-Tn5-adjusted',
    'barcode': 'declared-corrected;opaque-v2-identifier;preserve-exactly',
    'support': SUPPORT, 'layouts': [5, 6], 'strand_definition': STRAND_DEFINITION,
    'headers': 'bounded-leading-#;opaque-provenance;no-fact-extraction',
    'line_policy': 'UTF-8;TAB;LF;canonical-unsigned-decimals',
    'max_line_bytes': MAX_LINE, 'max_header_bytes': MAX_HEADER_BYTES, 'max_headers': MAX_HEADERS,
    'transformation_policy': TRANSFORMATION,
    'semantic_references': [
        'https://www.10xgenomics.com/support/software/cell-ranger-atac/latest/analysis/outputs/fragments-file',
        'https://www.10xgenomics.com/support/software/cell-ranger-arc/latest/analysis/outputs/fragments-file'],
    'historical_processing': 'caller-declared;not-independently-reconstructed',
}
PROFILE_BYTES = canonical(PROFILE)
SOURCE_KEYS = ('resource', 'encoding', 'decoded_sha256', 'data_sha256', 'header_sha256',
               'header_bytes', 'header_records', 'columns', 'n_records', 'sum_support')


class ExternalFragmentsError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def fail(code='EXTERNAL_FRAGMENTS_CONTRACT_INVALID'):
    raise ExternalFragmentsError(code) from None


def sha_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def validate_arguments(arguments):
    """Pure shape/value checks; never opens source/reference/toolchain files."""
    required = ('source_path', 'source_sha256', 'source_profile', 'reference_bundle_path',
                'reference_bundle_sha256', 'namespace', 'output_dir')
    defaults = dict(source_selection='unknown', source_index_path=None, source_index_sha256=None)
    try:
        if not set(required) <= set(arguments) or set(arguments) - set(required) - set(defaults):
            fail()
        value = {**defaults, **dict(arguments)}
        for key in ('source_path', 'reference_bundle_path', 'output_dir'):
            value[key] = str(value[key]) if isinstance(value[key], Path) else value[key]
            v2.absolute_path(value[key])
        for key in ('source_sha256', 'reference_bundle_sha256'):
            v2.sha(value[key])
        v2.token(value['namespace'])
        if value['source_profile'] != PROFILE_ID:
            fail('EXTERNAL_FRAGMENTS_PROFILE_UNSUPPORTED')
        if value['source_selection'] not in SELECTION:
            fail()
        if (value['source_index_path'] is None) != (value['source_index_sha256'] is None):
            fail('EXTERNAL_FRAGMENTS_INDEX_BINDING_INVALID')
        if value['source_index_path'] is not None:
            value['source_index_path'] = str(value['source_index_path'])
            v2.absolute_path(value['source_index_path']); v2.sha(value['source_index_sha256'])
            if value['source_index_path'] in (value['source_path'], value['reference_bundle_path']):
                fail('EXTERNAL_FRAGMENTS_INDEX_BINDING_INVALID')
        return value
    except ExternalFragmentsError:
        raise
    except (TypeError, ValueError, KeyError):
        fail()


def validate_adoption_record(value):
    """Closed bounded producer-specific metadata. No arbitrary header facts."""
    try:
        v2.shape(value, ('artifact_type', 'schema_version', 'contract_version', 'profile_sha256',
                         'source', 'source_index', 'reference', 'namespace', 'source_selection',
                         'transformation_policy', 'canonical'))
        if (value['artifact_type'] != 'agent.external-fragment-adoption'
                or type(value['schema_version']) is not int or value['schema_version'] != 1
                or value['contract_version'] != ADOPTION_CONTRACT
                or value['profile_sha256'] != sha_bytes(PROFILE_BYTES)
                or value['transformation_policy'] != TRANSFORMATION):
            fail()
        v2.token(value['namespace'])
        if value['source_selection'] not in SELECTION:
            fail()
        src = value['source']; v2.shape(src, SOURCE_KEYS); v2.resource(src['resource'])
        if src['encoding'] not in ('plain', 'gzip', 'bgzf') or type(src['columns']) is not int or src['columns'] not in (5, 6):
            fail()
        for key in ('decoded_sha256', 'data_sha256', 'header_sha256'):
            v2.sha(src[key])
        v2.number(src['header_bytes'], low=0, high=MAX_HEADER_BYTES)
        v2.number(src['header_records'], low=0, high=MAX_HEADERS)
        if bool(src['header_bytes']) != bool(src['header_records']):
            fail()
        for key in ('n_records', 'sum_support'):
            v2.number(src[key])
        if value['source_index'] is not None:
            v2.resource(value['source_index'])
            if src['encoding'] != 'bgzf':
                fail('EXTERNAL_FRAGMENTS_INDEX_BINDING_INVALID')
        r = value['reference']
        v2.shape(r, ('manifest_path', 'manifest_sha256', 'reference_identity_sha256', 'species', 'assembly', 'ordered_contig_sha256'))
        v2.absolute_path(r['manifest_path'])
        for key in ('manifest_sha256', 'reference_identity_sha256', 'ordered_contig_sha256'):
            v2.sha(r[key])
        if r['species'] not in ('human', 'mouse') or r['assembly'] != {'human': 'hg38', 'mouse': 'mm10'}[r['species']]:
            fail()
        summary = value['canonical']; v2.shape(summary, v2.SUMMARY_KEYS)
        v2.sha(summary['canonical_record_stream_sha256'])
        for key in v2.SUMMARY_KEYS[1:]:
            v2.number(summary[key], high=MAX_SUPPORT if key == 'max_support' else MAX_TOTAL)
        if summary['n_fragment_records'] != src['n_records'] or summary['sum_support'] != src['sum_support']:
            fail('EXTERNAL_FRAGMENTS_CONSERVATION_MISMATCH')
        if len(canonical(value)) > 32768:
            fail()
        return deepcopy(value)
    except ExternalFragmentsError:
        raise
    except (ValueError, TypeError, KeyError, UnicodeError, RecursionError):
        fail()


def load_adoption_record(path, expected_sha256):
    try:
        with Path(path).open('rb') as source:
            payload = source.read(32769)
        if len(payload) > 32768 or sha_bytes(payload) != expected_sha256:
            fail('EXTERNAL_FRAGMENTS_RECORD_MISMATCH')
        return validate_adoption_record(json.loads(payload.decode('utf-8'), object_pairs_hook=v2._pairs,
                                                  parse_constant=lambda _: fail()))
    except ExternalFragmentsError:
        raise
    except (OSError, ValueError, TypeError, UnicodeError):
        fail('EXTERNAL_FRAGMENTS_RECORD_MISMATCH')


def provenance(record, profile_resource, record_resource):
    """Exact reviewed projection; no historical qualification assertion."""
    return dict(kind='external_fragment_adoption', contract_version='fragment-producer-provenance.v1',
        profile={'id': PROFILE_ID, 'resource': profile_resource},
        sources=[{'role': 'external_fragments', 'resource': record['source']['resource']}],
        producer_record=record_resource, support=deepcopy(SUPPORT),
        processing={
            'tn5': {'status': 'declared', 'description': 'Source intervals already adjusted under the selected 10x profile; adoption applies no shift.'},
            'deduplication': {'status': 'declared', 'description': 'Source declares unique fragments with duplicate read-pair support; adoption rejects duplicate keys without merging.'},
            'mapq_filtering': {'status': 'unspecified', 'description': None},
            'barcode_correction': {'status': 'declared', 'description': 'Source identifiers declared corrected; adoption preserves exact tokens without correction.'}},
        source_selection=SELECTION[record['source_selection']], verification_requirement='producer-specific-verifier-required.v1')
