"""Registered explicit QC selection, verified atomic publication and recovery."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import TypedDict

from . import _cell_selection_contract as m, _barcode_qc_contract as qc, scatac_fragments_v2 as v2
from .scatac_qc_profile import canonical
from .scatac_selection_profile import PROFILE_SHA256, PROFILE, thresholds, fail
from .scatac_qc_reference import _fsync_dir
from .scatac_fragments_v2_verifier import take_snapshots,check_snapshots
from .barcode_qc_verifier import verify_barcode_qc
from .cell_selection_verifier import verify_cell_selection
from agent.tools._cancellation import cancellation_checkpoint

RECOVERY_POLICY=m.POLICY


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


def _publication(arguments, execution_identity):
    args=m.arguments(arguments)
    token=qc.digest(dict(arguments=args, policy=m.POLICY, execution_identity=execution_identity,
        selection_profile_sha256=PROFILE_SHA256, thresholds=thresholds(args)))
    return args,Path(args['output_dir'])/('cell-selection-'+token),token


def _summary(value,path,sha):
    return CellSelectionResult(status='success',manifest_path=str(path),manifest_sha256=sha,
        artifact_type=m.ARTIFACT,artifact_schema_version=1,contract_version=m.CONTRACT,
        n_observed_barcodes=value['row_count'],n_selected=value['selected_count'],n_rejected=value['rejected_count'],
        selection_method=PROFILE['selection_method'],cell_call_method='none',cell_call_state='not_assessed',
        readiness=value['readiness'],ordered_selected_sha256=value['ordered_selected_sha256'],
        qc_identity_sha256=value['qc_identity_sha256'],resource_qualification=value['qc_lineage']['resource_qualification']['mode'])


def _receipt(destination,args,token=None):
    try:
        with (destination/'receipt.json').open('rb') as f: raw=f.read(16385)
        if len(raw)>16384: fail('SELECTION_RECOVERY_MISMATCH')
        r=json.loads(raw,object_pairs_hook=v2._pairs,parse_constant=lambda _:fail('SELECTION_RECOVERY_MISMATCH'))
        v2.shape(r,('artifact_type','schema_version','policy','execution_identity','arguments_sha256','manifest_sha256'))
        if (r['artifact_type']!='agent.cell-selection-receipt' or type(r['schema_version']) is not int or r['schema_version']!=1
            or r['policy']!=m.POLICY or r['arguments_sha256']!=qc.digest(args)
            or destination.name!='cell-selection-'+r['execution_identity']
            or token is not None and r['execution_identity']!=token): fail('SELECTION_RECOVERY_MISMATCH')
        v2.sha(r['execution_identity']);v2.sha(r['manifest_sha256'])
        return r
    except (OSError,ValueError,TypeError,KeyError): fail('SELECTION_RECOVERY_MISMATCH')


def verify_public_result(arguments,result):
    args=m.arguments(arguments);path=Path(result['manifest_path']);destination=path.parent
    if (path!=path.resolve() or path.name!='manifest.json' or destination.parent!=Path(args['output_dir'])): fail('SELECTION_RESULT_MISMATCH')
    before=take_snapshots([path,destination/'receipt.json'])
    receipt=_receipt(destination,args)
    if receipt['manifest_sha256']!=result['manifest_sha256']: fail('SELECTION_RECOVERY_MISMATCH')
    verified=verify_cell_selection(path,expected_sha256=result['manifest_sha256']);value=verified.to_dict()
    if value['arguments']!=args or dict(result)!=_summary(value,path,result['manifest_sha256']): fail('SELECTION_RESULT_MISMATCH')
    check_snapshots(before)
    return verified


def recover_cell_selection(arguments,execution_identity):
    args,destination,token=_publication(arguments,execution_identity)
    if not destination.is_dir() or destination.is_symlink(): fail('SELECTION_RECOVERY_UNAVAILABLE')
    receipt=_receipt(destination,args,token);path=destination/'manifest.json'
    value=m.load_manifest(path,receipt['manifest_sha256']).to_dict()
    result=_summary(value,path,receipt['manifest_sha256'])
    verify_public_result(args,result)
    return result


def execute_cell_selection(arguments,execution_identity=None):
    from ._cell_selection_production import produce
    cancellation_checkpoint()
    args,destination,token=_publication(arguments,execution_identity)
    output=Path(args['output_dir'])
    if output!=output.resolve() or destination.exists() or destination.is_symlink(): fail('SELECTION_OUTPUT_CONFLICT')
    qp=Path(args['barcode_qc_manifest_path'])
    q=verify_barcode_qc(qp,expected_sha256=args['barcode_qc_manifest_sha256']).to_dict()
    before=take_snapshots([qp,qp.parent/q['table']['path'],qp.parent/q['histogram']['path']])
    output.mkdir(parents=True,exist_ok=True)
    with (output/('.'+destination.name+'.lock')).open('a+b') as lease:
        # Nonblocking acquisition permits cooperative cancellation while waiting.
        import time
        while True:
            try: fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError: cancellation_checkpoint();time.sleep(0.1)
        if destination.exists() or destination.is_symlink(): fail('SELECTION_OUTPUT_CONFLICT')
        stage=Path(tempfile.mkdtemp(prefix='.selection-attempt-',dir=output))
        try:
            outputs=produce(qp,q,thresholds(args),stage)
            value=dict(artifact_type=m.ARTIFACT,schema_version=1,contract_version=m.CONTRACT,
                policy=m.POLICY,arguments=args,qc_identity_sha256=q['identity_sha256'],qc_lineage=m.lineage(q),
                selection_profile=PROFILE,selection_profile_sha256=PROFILE_SHA256,thresholds=thresholds(args),**outputs)
            value['identity_sha256']=m.identity(value)
            payload=canonical(m.validate(value));path=stage/'manifest.json';path.write_bytes(payload)
            sha=qc.resource(path)['sha256']
            verify_cell_selection(path,expected_sha256=sha)
            (stage/'receipt.json').write_bytes(canonical(dict(artifact_type='agent.cell-selection-receipt',schema_version=1,
                policy=m.POLICY,execution_identity=token,arguments_sha256=qc.digest(args),manifest_sha256=sha)))
            for p in stage.iterdir():
                with p.open('rb') as f: os.fsync(f.fileno())
            _fsync_dir(stage);check_snapshots(before);cancellation_checkpoint()
            if destination.exists() or destination.is_symlink(): fail('SELECTION_OUTPUT_CONFLICT')
            os.rename(stage,destination);_fsync_dir(output)
            # No cancellation after publication: executor checkpoints verified
            # success before observing later cancellation, retaining this artifact.
            return _summary(value,destination/'manifest.json',sha)
        finally:
            if stage.exists(): shutil.rmtree(stage)


def select_scATAC_cells(barcode_qc_manifest_path,barcode_qc_manifest_sha256,
        min_qc_fragment_records,min_tss_enrichment,output_dir,*,min_tss_flank_evidence=None,
        max_qc_fragment_records=None,max_nucleosome_signal=None)->CellSelectionResult:
    """Select candidate cells by explicit thresholds; statistical calling is not assessed."""
    return execute_cell_selection(locals())
