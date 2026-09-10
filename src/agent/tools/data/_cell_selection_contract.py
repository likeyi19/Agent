"""Closed bounded selection artifacts and shared IO, not decision logic."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from . import _barcode_qc_contract as qc, scatac_fragments_v2 as v2
from .scatac_qc_profile import canonical
from .scatac_selection_profile import (METHOD, PROFILE, PROFILE_SHA256, REQUIRED, OPTIONAL,
    REASONS, thresholds, fail)
from agent.tools._cancellation import cancellation_checkpoint

ARTIFACT = 'agent.scatac-cell-selection'
CONTRACT = 'scatac-cell-selection.v1'
POLICY = 'select-scatac-cells-v1'
ORDER_DOMAIN = b'agent.ordered-selected-cells.v1\0'
ARGUMENTS = ('barcode_qc_manifest_path', 'barcode_qc_manifest_sha256', 'output_dir', *REQUIRED, *OPTIONAL)
DECISION_HEADER = b'namespace\tbarcode_identifier\trendered_cell_id\tcell_call_state\tqc_pass\tqc_selected\tfailure_reasons\n'
SELECTED_HEADER = b'matrix_row_index\tnamespace\tbarcode_identifier\trendered_cell_id\n'
FIELDS = ('artifact_type', 'schema_version', 'contract_version', 'policy', 'arguments',
    'qc_identity_sha256', 'qc_lineage', 'selection_profile', 'selection_profile_sha256',
    'thresholds', 'row_count', 'selected_count', 'rejected_count', 'cell_call_not_assessed_count',
    'reason_counts', 'readiness', 'ordered_selected_sha256', 'decisions', 'selected', 'identity_sha256')
LINEAGE_FIELDS = ('arguments', 'reference_identity_sha256', 'qc_resource_identity_sha256',
    'science_profile_sha256', 'resource_qualification', 'producer_authority', 'ordered_barcode_sha256')


def arguments(value):
    args = dict(value)
    for key in OPTIONAL: args.setdefault(key, None)
    v2.shape(args, ARGUMENTS)
    for key in ('barcode_qc_manifest_path', 'output_dir'):
        if isinstance(args[key], Path): args[key] = str(args[key])
        v2.absolute_path(args[key])
    v2.sha(args['barcode_qc_manifest_sha256'])
    thresholds(args)
    return args


def lineage(q):
    return {k:q[k] for k in LINEAGE_FIELDS}


def validate_lineage(value):
    v2.shape(value, LINEAGE_FIELDS)
    if canonical(value['arguments'])!=canonical(qc.validate_arguments(value['arguments'])):
        fail('SELECTION_QC_LINEAGE_INVALID')
    for key in ('reference_identity_sha256','qc_resource_identity_sha256','science_profile_sha256','ordered_barcode_sha256'):
        v2.sha(value[key])
    if value['science_profile_sha256']!=qc.PROFILE_SHA256: fail('SELECTION_QC_LINEAGE_INVALID')
    qualification=value['resource_qualification'];v2.shape(qualification,('mode','catalog_sha256','basis'));v2.token(qualification['basis'])
    if qualification['mode']=='synthetic_only':
        if qualification['catalog_sha256'] is not None: fail('SELECTION_QC_LINEAGE_INVALID')
    elif qualification['mode']=='operator_qualified':v2.sha(qualification['catalog_sha256'])
    else:fail('SELECTION_QC_LINEAGE_INVALID')
    producer=value['producer_authority'];v2.shape(producer,('kind','profile_ids','verification_basis','history'))
    if producer['kind'] not in ('fastq_fragment_production','bam_fragment_production','external_fragment_adoption'):
        fail('SELECTION_QC_LINEAGE_INVALID')
    if type(producer['profile_ids']) is not list or not 1<=len(producer['profile_ids'])<=3:fail('SELECTION_QC_LINEAGE_INVALID')
    for profile in producer['profile_ids']:v2.token(profile)
    for key in ('verification_basis','history'):v2.token(producer[key])


def identity(value):
    return hashlib.sha256(b'agent.cell-selection-identity.v1\0' + canonical(
        {k:v for k,v in value.items() if k != 'identity_sha256'})).hexdigest()


def validate(value):
    v2.shape(value, FIELDS)
    if len(canonical(value)) > qc.MAX_MANIFEST: fail('SELECTION_MANIFEST_LIMIT')
    if (value['artifact_type'] != ARTIFACT or type(value['schema_version']) is not int
        or value['schema_version'] != 1 or value['contract_version'] != CONTRACT or value['policy'] != POLICY
        or value['selection_profile'] != PROFILE or value['selection_profile_sha256'] != PROFILE_SHA256
        or qc.digest(value['selection_profile']) != PROFILE_SHA256):
        fail('SELECTION_PROFILE_INVALID')
    if canonical(value['arguments']) != canonical(arguments(value['arguments'])) or canonical(value['thresholds']) != canonical(thresholds(value['arguments'])):
        fail('SELECTION_THRESHOLD_MISMATCH')
    validate_lineage(value['qc_lineage'])
    for key in ('qc_identity_sha256', 'ordered_selected_sha256', 'identity_sha256'): v2.sha(value[key])
    for key in ('row_count', 'selected_count', 'rejected_count', 'cell_call_not_assessed_count'):
        if type(value[key]) is not int or not 0 <= value[key] <= qc.MAX_BARCODES: fail('SELECTION_SUMMARY_INVALID')
    n = value['row_count']; selected = value['selected_count']
    if (n < 1 or selected+value['rejected_count'] != n or value['cell_call_not_assessed_count'] != n
        or value['readiness'] != ('selected_candidates_available' if selected else 'no_selected_cells')):
        fail('SELECTION_SUMMARY_INVALID')
    v2.shape(value['reason_counts'], REASONS)
    for count in value['reason_counts'].values():
        if type(count) is not int or not 0 <= count <= value['rejected_count']: fail('SELECTION_SUMMARY_INVALID')
    for key, name in (('decisions','decisions.tsv.gz'), ('selected','selected.tsv.gz')):
        resource=value[key];v2.shape(resource, ('path','sha256','size_bytes'));v2.sha(resource['sha256'])
        if resource['path'] != name or type(resource['size_bytes']) is not int or not 0 < resource['size_bytes'] <= qc.MAX_SIDECAR:
            fail('SELECTION_RESOURCE_INVALID')
    if value['identity_sha256'] != identity(value): fail('SELECTION_IDENTITY_MISMATCH')
    return value


@dataclass(frozen=True)
class CellSelectionManifest:
    canonical_bytes: bytes

    def __post_init__(self):
        if type(self.canonical_bytes) is not bytes or len(self.canonical_bytes)>qc.MAX_MANIFEST: fail('SELECTION_MANIFEST_LIMIT')
        value=json.loads(self.canonical_bytes, object_pairs_hook=v2._pairs, parse_constant=lambda _:fail('SELECTION_MANIFEST_INVALID'))
        if canonical(validate(value)) != self.canonical_bytes: fail('SELECTION_MANIFEST_INVALID')

    def to_dict(self): return json.loads(self.canonical_bytes)


def load_manifest(path, expected_sha256):
    v2.absolute_path(str(path));v2.sha(expected_sha256)
    with open(path,'rb') as f: raw=f.read(qc.MAX_MANIFEST+1)
    if hashlib.sha256(raw).hexdigest()!=expected_sha256: fail('SELECTION_MANIFEST_MISMATCH')
    return CellSelectionManifest(raw)


def qc_rows(path, value):
    rows=qc.gzip_lines(Path(path).parent/value['table']['path'],qc.MAX_BARCODES+1)
    if next(rows,None)!=qc.HEADER: fail('SELECTION_QC_ROW_INVALID')
    previous=None;order=hashlib.sha256(qc.ORDER_DOMAIN);n=0
    for n,line in enumerate(rows,1):
        fields=line.decode('ascii').rstrip('\n').split('\t')
        if len(fields)!=16: fail('SELECTION_QC_ROW_INVALID')
        ns,bc=fields[:2];counts=tuple(int(x) for x in fields[2:10]);pair=(ns,bc)
        if qc.row_bytes(ns,bc,counts)!=line or previous is not None and pair<=previous:
            fail('SELECTION_QC_ROW_INVALID')
        order.update(qc.identity(ns,bc));previous=pair
        if n%1024==0: cancellation_checkpoint()
        yield ns,bc,counts
    if n!=value['row_count'] or order.hexdigest()!=value['ordered_barcode_sha256']: fail('SELECTION_QC_ORDER_MISMATCH')
