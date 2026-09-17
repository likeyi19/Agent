"""Public external matrix adoption using the existing matrix publication lifecycle."""
from copy import deepcopy
from pathlib import Path
from typing import TypedDict
from . import scatac_matrix as shared, scatac_matrix_contract as m, external_matrix_contract as e
from . import _external_matrix_io as scientific
from .cell_by_ccre_verifier import verify_cell_by_ccre
from .scatac_fragments_v2_verifier import take_snapshots, check_snapshots
from agent.tools._cancellation import cancellation_checkpoint

RECOVERY_POLICY = 'adopt-scatac-cell-by-ccre-v1'
ARGUMENTS = ('source_path','source_sha256','reference_manifest_path','reference_manifest_sha256',
             'species','assembly','matrix_semantics','output_dir')


class AdoptedCellByCCREResult(TypedDict):
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
    source_logical_matrix_sha256: str
    source_sha256: str
    reference_identity_sha256: str
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
    ordered_cells_sha256: str
    ordered_feature_sha256: str
    readiness: str


def arguments(value, *, contract=e):
    args=deepcopy(dict(value)); m.shape(args,ARGUMENTS)
    for key in ARGUMENTS:
        if key.endswith('_sha256'): m.sha(args[key])
        elif key.endswith('_path') or key=='output_dir':
            args[key]=str(args[key]) if isinstance(args[key],Path) else args[key]
            m.absolute_path(args[key])
    contract.validate_binding(args['species'],args['assembly'])
    if args['matrix_semantics'] not in e.SEMANTICS: m.fail('MATRIX_SEMANTICS_INVALID')
    return args


def _publication(args,execution_identity, *, contract=e):
    args=arguments(args, contract=contract)
    token=shared.digest(dict(arguments=args,policy=contract.RECOVERY_POLICY,execution_identity=execution_identity,matrix_profile_sha256=contract.PROFILE_SHA256))
    return args,Path(args['output_dir'])/('cell-by-ccre-'+token),token


def _receipt(destination,args,token=None, *, contract=e):
    return shared._receipt(destination,args,token,policy=contract.RECOVERY_POLICY,profile=contract.PROFILE_SHA256)


def _summary(value,path,sha, *, contract=e):
    if value['contract_version']!=contract.CONTRACT: m.fail('MATRIX_CONTRACT_INVALID')
    return dict(status='success',manifest_path=str(path),manifest_sha256=sha,artifact_type=contract.ARTIFACT,
        artifact_schema_version=1,contract_version=contract.CONTRACT,identity_sha256=value['identity_sha256'],
        matrix_path=str(path.parent/'matrix.h5ad'),matrix_sha256=value['matrix']['sha256'],
        source_sha256=value['source']['sha256'],reference_identity_sha256=value['reference']['identity_sha256'],
        n_cells=value['shape'][0],n_features=value['shape'][1],matrix_profile_id=contract.PROFILE,matrix_profile_sha256=contract.PROFILE_SHA256,
        **{k:value[k] for k in ('logical_matrix_sha256','source_logical_matrix_sha256','species','assembly',
             'nnz','total_count','zero_row_count','matrix_semantics','ordered_cells_sha256','ordered_feature_sha256','readiness')})


def verify_public_result(args,result, *, contract=e):
    args=arguments(args, contract=contract); path=Path(result['manifest_path']); destination=path.parent.parent
    if (path!=path.resolve() or path.name!='manifest.json' or path.parent.name!='artifact'
            or destination.parent!=Path(args['output_dir'])): m.fail('MATRIX_RESULT_MISMATCH')
    before=take_snapshots([path,destination/'receipt.json'])
    receipt=_receipt(destination,args, contract=contract)
    if receipt['manifest_sha256']!=result['manifest_sha256']: m.fail('MATRIX_RECOVERY_MISMATCH')
    value=shared._load(path,result['manifest_sha256']); contract.validate(value)
    if (value['source']['path']!=args['source_path'] or value['source']['sha256']!=args['source_sha256']
            or any(value[k]!=args[k] for k in ('species','assembly','matrix_semantics'))
            or any(value['reference']['manifest_'+k]!=args['reference_manifest_'+k] for k in ('path','sha256'))
            or dict(result)!=_summary(value,path,result['manifest_sha256'], contract=contract)): m.fail('MATRIX_RESULT_MISMATCH')
    verify_cell_by_ccre(path,expected_sha256=result['manifest_sha256'])
    check_snapshots(before)
    return value


def recover_matrix(arguments,execution_identity, *, contract=e):
    args,destination,token=_publication(arguments,execution_identity, contract=contract)
    if not destination.is_dir() or destination.is_symlink(): m.fail('MATRIX_RECOVERY_UNAVAILABLE')
    receipt=_receipt(destination,args,token, contract=contract); path=destination/'artifact/manifest.json'
    result=_summary(shared._load(path,receipt['manifest_sha256']),path,receipt['manifest_sha256'], contract=contract)
    verify_public_result(args,result, contract=contract)
    return result


def execute_matrix(arguments,execution_identity=None, *, contract=e):
    cancellation_checkpoint(); args,destination,token=_publication(arguments,execution_identity, contract=contract)
    return shared._execute_publication(args,destination,token,lambda output: scientific.build(args,output,contract=contract),
        lambda value,path,sha: _summary(value,path,sha,contract=contract),
        contract.RECOVERY_POLICY,contract.PROFILE_SHA256)


def adopt_scATAC_cell_by_ccre(source_path,source_sha256,reference_manifest_path,reference_manifest_sha256,
                             species,assembly,matrix_semantics,output_dir)->AdoptedCellByCCREResult:
    """Adopt an exact full canonical int64 CSR H5AD with declared value semantics."""
    return execute_matrix(locals())
