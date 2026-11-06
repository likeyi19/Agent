"""Integrity-only reading of accepted retired raw scientific publications.

This boundary cannot qualify a receipt, reconstruct science, or create new
scientific authority. The accepted RunStore supplies its exact authority.
"""
from pathlib import Path

from agent.schemas.verification_authority import AuthorityError, digest
from agent.schemas.orchestration import _serialize
from .authority_context import current

RETIRED_TOOLS = frozenset(('prepare_scATAC_fragments', 'prepare_scATAC_bam_fragments', 'import_scATAC_fragments',
    'compute_scATAC_qc', 'select_scATAC_cells', 'build_scATAC_cell_by_ccre', 'build_scATAC_cell_by_features',
    'build_cell_by_features', 'import_external_fragments', 'prepare_neutral_bam_fragments'))
RETIRED_KINDS = frozenset(('fastq_fragment_production', 'bam_fragment_production', 'external_fragment_adoption', 'qc', 'selection', 'generic_fragments'))


def normalized(kind, args):
    args = _serialize(args)
    if kind == 'fastq_fragment_production': return {key:str(value) for key,value in args.items()}
    if kind in ('bam_fragment_production', 'external_fragment_adoption'):
        from . import bam_fragment_manifest, external_fragment_manifest
        return (bam_fragment_manifest if kind == 'bam_fragment_production' else external_fragment_manifest).validate_arguments(args)
    if kind == 'selection':
        from ._cell_selection_contract import arguments
        return arguments(args)
    if kind == 'qc':
        from ._barcode_qc_contract import validate_arguments
        return validate_arguments(args)
    return args


def accepted(kind, path, sha, *, context=None):
    context = current() if context is None else context
    key = (str(path), sha)
    if context is None or key not in context.accepted:
        raise AuthorityError('Retired raw methods require existing accepted execution authority; receipts cannot qualify new science')
    record = _serialize(context.accepted[key]['record'])
    if record['schema_version'] != 2 or record['completion'] != 'succeeded' or record['scope'] != 'scientific_correctness.v1':
        raise AuthorityError('No reusable accepted historical scientific authority')
    description, proof = context.describe(kind, Path(path), sha)
    if (record['resources']['manifest_identity'] != description['manifest']
            or kind != 'generic_fragments' and record['resources']['scientific_proof_sha256'] != proof):
        raise AuthorityError('Accepted historical publication changed')
    return description['manifest'], record


def publication_record(context, tool_name, args, result, execution_identity, result_metadata):
    from .scientific_authority import TOOLS
    from agent.schemas.verification_authority import VerifiedArtifactAuthority
    value, record = accepted(TOOLS[tool_name], result['manifest_path'], result['manifest_sha256'], context=context)
    if (record['execution_identity'] != execution_identity or record['arguments_sha256'] != digest(normalized(TOOLS[tool_name], args))
            or record['resources']['result_metadata'] != _serialize(result_metadata)):
        raise AuthorityError('Historical publication differs from its accepted execution')
    # accepted() checked the complete original closure, including the receipt,
    # profile, resources and dependency associations. It issues no new authority.
    return VerifiedArtifactAuthority(record)


def verify_public(kind, args, result):
    value, record = accepted(kind, result['manifest_path'], result['manifest_sha256'])
    if record['arguments_sha256'] != digest(normalized(kind, args)):
        raise AuthorityError('Historical result differs from its accepted arguments')
    return value


from typing import TypedDict


class ScATACFragmentsResult(TypedDict):
    status: str
    manifest_path: str
    manifest_sha256: str
    artifact_type: str
    artifact_schema_version: int
    contract_version: str
    species: str
    assembly: str
    n_libraries: int
    n_fragment_records: int
    total_support: int



class BamFragmentsResult(TypedDict):
    status: str
    manifest_path: str
    manifest_sha256: str
    artifact_type: str
    artifact_schema_version: int
    contract_version: str
    source_profile: str
    eligible_pairs: int
    n_templates: int
    species: str | dict
    assembly: str
    namespace: str
    n_libraries: int
    n_fragment_records: int
    total_support: int
    strand_mode: str



class ExternalFragmentsResult(TypedDict):
    status: str
    manifest_path: str
    manifest_sha256: str
    artifact_type: str
    artifact_schema_version: int
    contract_version: str
    source_profile: str
    source_encoding: str
    species: str | dict
    assembly: str
    namespace: str
    n_libraries: int
    n_fragment_records: int
    total_support: int
    strand_mode: str



class BarcodeQCResult(TypedDict):
    status: str
    manifest_path: str
    manifest_sha256: str
    artifact_type: str
    artifact_schema_version: int
    contract_version: str
    n_observed_barcodes: int
    n_fragment_records: int
    n_qc_fragment_records: int
    tss_defined: int
    tss_undefined: int
    science_profile_sha256: str
    qc_resource_identity_sha256: str
    resource_qualification: str



class CellSelectionResult(TypedDict):
    status: str
    manifest_path: str
    manifest_sha256: str
    artifact_type: str
    artifact_schema_version: int
    contract_version: str
    n_observed_barcodes: int
    n_selected: int
    n_rejected: int
    selection_method: str
    cell_call_method: str
    cell_call_state: str
    readiness: str
    ordered_selected_sha256: str
    qc_identity_sha256: str
    resource_qualification: str



class CellByCCREResult(TypedDict):
    status: str
    manifest_path: str
    manifest_sha256: str
    artifact_type: str
    artifact_schema_version: int
    contract_version: str
    identity_sha256: str
    matrix_path: str
    matrix_sha256: str
    logical_matrix_sha256: str
    species: str
    assembly: str
    n_cells: int
    n_features: int
    nnz: int
    total_count: int
    zero_row_count: int
    matrix_semantics: str
    matrix_profile_id: str
    matrix_profile_sha256: str
    ordered_selected_sha256: str
    ordered_feature_sha256: str
    readiness: str
    diagnostic: dict



def matrix_summary(value,path,sha):
    from . import scatac_matrix_contract as m
    return CellByCCREResult(status='success',manifest_path=str(path),manifest_sha256=sha,
        artifact_type=m.ARTIFACT,artifact_schema_version=1,contract_version=m.CONTRACT,
        identity_sha256=value['identity_sha256'],matrix_path=str(path.parent/'matrix.h5ad'),
        matrix_sha256=value['matrix']['sha256'],logical_matrix_sha256=value['logical_matrix_sha256'],
        species=value['species'],assembly=value['assembly'],n_cells=value['shape'][0],n_features=value['shape'][1],
        **{k:value[k] for k in ('nnz','total_count','zero_row_count','ordered_selected_sha256','ordered_feature_sha256','readiness','diagnostic')},
        matrix_semantics=m.PROFILE.matrix_semantics,matrix_profile_id=m.PROFILE.profile_id,matrix_profile_sha256=m.PROFILE_SHA256)



def barcode_qc_summary(value,path,sha):
    from . import _barcode_qc_contract as m
    from .scatac_qc_profile import PROFILE_SHA256
    return BarcodeQCResult(status='success',manifest_path=str(path),manifest_sha256=sha,
        artifact_type=m.ARTIFACT,artifact_schema_version=1,contract_version=m.CONTRACT,
        n_observed_barcodes=value['row_count'],n_fragment_records=value['summary']['n_fragment_records'],
        n_qc_fragment_records=value['summary']['n_qc_fragment_records'],tss_defined=value['summary']['tss_defined'],
        tss_undefined=value['summary']['tss_undefined'],science_profile_sha256=PROFILE_SHA256,
        qc_resource_identity_sha256=value['qc_resource_identity_sha256'],resource_qualification=value['resource_qualification']['mode'])



def cell_selection_summary(value,path,sha):
    from . import _cell_selection_contract as m
    from .scatac_selection_profile import PROFILE_SHA256, PROFILE
    return CellSelectionResult(status='success',manifest_path=str(path),manifest_sha256=sha,
        artifact_type=m.ARTIFACT,artifact_schema_version=1,contract_version=m.CONTRACT,
        n_observed_barcodes=value['row_count'],n_selected=value['selected_count'],n_rejected=value['rejected_count'],
        selection_method=PROFILE['selection_method'],cell_call_method='none',cell_call_state='not_assessed',
        readiness=value['readiness'],ordered_selected_sha256=value['ordered_selected_sha256'],
        qc_identity_sha256=value['qc_identity_sha256'],resource_qualification=value['qc_lineage']['resource_qualification']['mode'])



def feature_matrix_summary(value, path, sha):
    from .fragment_feature_matrix_contract import contract_for
    contract = contract_for(value)
    return matrix_summary(value, path, sha) | dict(artifact_type=contract.ARTIFACT,
        contract_version=contract.CONTRACT, matrix_profile_id=contract.PROFILE.profile_id,
        matrix_profile_sha256=contract.PROFILE_SHA256)
