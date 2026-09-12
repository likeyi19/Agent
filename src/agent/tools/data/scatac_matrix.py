"""M11.5c public matrix boundary: durable envelope around unchanged M11.5b science."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import TypedDict

from agent.tools._cancellation import cancellation_checkpoint
from . import scatac_matrix_contract as m, scatac_fragments_v2 as v2
from . import scatac_cell_by_ccre as scientific
from .cell_by_ccre_verifier import verify_cell_by_ccre
from .scatac_qc_reference import _fsync_dir
from .scatac_fragments_v2_verifier import take_snapshots, check_snapshots

RECOVERY_POLICY='build-scatac-cell-by-ccre-v1'
ARGUMENTS=tuple(f'{prefix}_{suffix}' for prefix in ('fragments_manifest','selected_cells_manifest','reference_manifest')
                for suffix in ('path','sha256'))+('output_dir',)


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


def arguments(value):
    args=dict(value);m.shape(args,ARGUMENTS)
    for k in ARGUMENTS:
        if k.endswith('_sha256'):m.sha(args[k])
        else:
            if isinstance(args[k],Path):args[k]=str(args[k])
            m.absolute_path(args[k])
    return args


def executable():
    value=os.environ.get('AGENT_MATRIX_BEDTOOLS')
    if not value:m.fail('MATRIX_BACKEND_UNAVAILABLE')
    return value


def digest(value):return hashlib.sha256(m.canonical(value)).hexdigest()


def _publication(args,execution_identity):
    args=arguments(args)
    token=digest(dict(arguments=args,policy=RECOVERY_POLICY,execution_identity=execution_identity,
                      matrix_profile_sha256=m.PROFILE_SHA256))
    return args,Path(args['output_dir'])/('cell-by-ccre-'+token),token


def _summary(value,path,sha):
    return CellByCCREResult(status='success',manifest_path=str(path),manifest_sha256=sha,
        artifact_type=m.ARTIFACT,artifact_schema_version=1,contract_version=m.CONTRACT,
        identity_sha256=value['identity_sha256'],matrix_path=str(path.parent/'matrix.h5ad'),
        matrix_sha256=value['matrix']['sha256'],logical_matrix_sha256=value['logical_matrix_sha256'],
        species=value['species'],assembly=value['assembly'],n_cells=value['shape'][0],n_features=value['shape'][1],
        **{k:value[k] for k in ('nnz','total_count','zero_row_count','ordered_selected_sha256','ordered_feature_sha256','readiness','diagnostic')},
        matrix_semantics=m.PROFILE.matrix_semantics,matrix_profile_id=m.PROFILE.profile_id,matrix_profile_sha256=m.PROFILE_SHA256)


def _load(path,sha):
    m.sha(sha)
    with path.open('rb') as f:raw=f.read(m.MAX_MANIFEST_BYTES+1)
    if hashlib.sha256(raw).hexdigest()!=sha:m.fail('MATRIX_MANIFEST_MISMATCH')
    return m.load_manifest_bytes(raw)


def _receipt(destination,args,token=None):
    try:
        with (destination/'receipt.json').open('rb') as f:raw=f.read(16385)
        if len(raw)>16384:m.fail('MATRIX_RECOVERY_MISMATCH')
        r=json.loads(raw,object_pairs_hook=v2._pairs,parse_constant=lambda _:m.fail('MATRIX_RECOVERY_MISMATCH'))
        m.shape(r,('artifact_type','schema_version','policy','execution_identity','arguments_sha256','manifest_sha256','matrix_profile_sha256'))
        if (r['artifact_type']!='agent.cell-by-ccre-receipt' or type(r['schema_version']) is not int or r['schema_version']!=1
                or r['policy']!=RECOVERY_POLICY or r['arguments_sha256']!=digest(args)
                or r['matrix_profile_sha256']!=m.PROFILE_SHA256
                or destination.name!='cell-by-ccre-'+r['execution_identity']
                or token is not None and r['execution_identity']!=token):m.fail('MATRIX_RECOVERY_MISMATCH')
        m.sha(r['execution_identity']);m.sha(r['manifest_sha256'])
        return r
    except (OSError,ValueError,TypeError,KeyError):m.fail('MATRIX_RECOVERY_MISMATCH')


def verify_public_result(args,result):
    args=arguments(args);path=Path(result['manifest_path']);destination=path.parent.parent
    if (path!=path.resolve() or path.name!='manifest.json' or path.parent.name!='artifact'
            or destination.parent!=Path(args['output_dir'])):m.fail('MATRIX_RESULT_MISMATCH')
    before=take_snapshots([path,destination/'receipt.json'])
    receipt=_receipt(destination,args)
    if receipt['manifest_sha256']!=result['manifest_sha256']:m.fail('MATRIX_RECOVERY_MISMATCH')
    value=_load(path,result['manifest_sha256'])
    for key,prefix in (('fragments','fragments_manifest'),('selection','selected_cells_manifest'),('reference','reference_manifest')):
        p=value['upstream'][key]
        if p['manifest_path']!=args[prefix+'_path'] or p['manifest_sha256']!=args[prefix+'_sha256']:m.fail('MATRIX_LINEAGE_INVALID')
    if dict(result)!=_summary(value,path,result['manifest_sha256']):m.fail('MATRIX_RESULT_MISMATCH')
    verify_cell_by_ccre(path,expected_sha256=result['manifest_sha256'],bedtools_path=executable())
    check_snapshots(before)
    return value


def recover_matrix(arguments,execution_identity):
    args,destination,token=_publication(arguments,execution_identity)
    if not destination.is_dir() or destination.is_symlink():m.fail('MATRIX_RECOVERY_UNAVAILABLE')
    receipt=_receipt(destination,args,token);path=destination/'artifact/manifest.json'
    result=_summary(_load(path,receipt['manifest_sha256']),path,receipt['manifest_sha256'])
    # Existing lifecycle requires fresh reconstruction here; never rerun production.
    verify_public_result(args,result)
    return result


def execute_matrix(arguments,execution_identity=None):
    cancellation_checkpoint();args,destination,token=_publication(arguments,execution_identity)
    output=Path(args['output_dir'])
    if output!=output.resolve() or destination.exists() or destination.is_symlink():m.fail('MATRIX_OUTPUT_CONFLICT')
    backend=executable()
    # Fresh accepted authority binding occurs inside the unchanged constructor,
    # before projections/intersections/H5AD. PLAN_ONLY never enters this function.
    output.mkdir(parents=True,exist_ok=True)
    with (output/('.'+destination.name+'.lock')).open('a+b') as lease:
        while True:
            try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:cancellation_checkpoint();time.sleep(0.1)
        if destination.exists() or destination.is_symlink():m.fail('MATRIX_OUTPUT_CONFLICT')
        stage=Path(tempfile.mkdtemp(prefix='.matrix-attempt-',dir=output))
        try:
            built=scientific.build_cell_by_ccre(
                fragments_manifest_path=args['fragments_manifest_path'],fragments_manifest_sha256=args['fragments_manifest_sha256'],
                selection_manifest_path=args['selected_cells_manifest_path'],selection_manifest_sha256=args['selected_cells_manifest_sha256'],
                reference_manifest_path=args['reference_manifest_path'],reference_manifest_sha256=args['reference_manifest_sha256'],
                output_dir=stage/'artifact',bedtools_path=backend)
            value=_load(Path(built['manifest_path']),built['manifest_sha256'])
            receipt=dict(artifact_type='agent.cell-by-ccre-receipt',schema_version=1,policy=RECOVERY_POLICY,
                execution_identity=token,arguments_sha256=digest(args),manifest_sha256=built['manifest_sha256'],matrix_profile_sha256=m.PROFILE_SHA256)
            (stage/'receipt.json').write_bytes(m.canonical(receipt))
            with (stage/'receipt.json').open('rb') as f:os.fsync(f.fileno())
            _fsync_dir(stage);cancellation_checkpoint()
            if destination.exists() or destination.is_symlink():m.fail('MATRIX_OUTPUT_CONFLICT')
            os.rename(stage,destination);_fsync_dir(output)
            from .authority_context import publication_moved
            publication_moved(stage, destination)
            # No cancellation checkpoint after the outer immutable publication.
            return _summary(value,destination/'artifact/manifest.json',built['manifest_sha256'])
        finally:
            if stage.exists():shutil.rmtree(stage)


def build_scATAC_cell_by_ccre(fragments_manifest_path,fragments_manifest_sha256,
        selected_cells_manifest_path,selected_cells_manifest_sha256,
        reference_manifest_path,reference_manifest_sha256,output_dir)->CellByCCREResult:
    """Full ordered canonical fragment-record counts for exact QC-selected cells."""
    return execute_matrix(locals())
