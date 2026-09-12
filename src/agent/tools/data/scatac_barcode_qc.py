"""Registered per-observed-barcode QC, verified atomic publication and recovery."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import TypedDict

from . import _barcode_qc_contract as m, _barcode_qc_binding as binding, scatac_fragments_v2 as v2
from .scatac_qc_profile import PROFILE_SHA256, PROFILE, canonical, fail
from .scatac_qc_reference import load_scatac_qc_reference_bundle, _fsync_dir
from .scatac_fragments_v2_verifier import take_snapshots,check_snapshots
from .barcode_qc_verifier import verify_barcode_qc
from agent.tools._cancellation import cancellation_checkpoint

RECOVERY_POLICY=m.POLICY


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


def _publication_token(args,execution_identity,qc_identity,qualification,backend):
    return m.digest(dict(arguments=args,policy=m.POLICY,execution_identity=execution_identity,
        science_profile_sha256=PROFILE_SHA256,qc_identity=qc_identity,
        resource_qualification=qualification,backend=backend))


def _publication(arguments,execution_identity):
    args=m.validate_arguments(arguments)
    _,qc,_=load_scatac_qc_reference_bundle(args['qc_reference_manifest_path'],expected_sha256=args['qc_reference_manifest_sha256'])
    qualification=binding.resource_qualification(qc);_,backend=binding.backend_runtime()
    token=_publication_token(args,execution_identity,qc.resource_identity_sha256,qualification,backend)
    return args,Path(args['output_dir'])/('barcode-qc-'+token),token


def _summary(value,path,sha):
    return BarcodeQCResult(status='success',manifest_path=str(path),manifest_sha256=sha,
        artifact_type=m.ARTIFACT,artifact_schema_version=1,contract_version=m.CONTRACT,
        n_observed_barcodes=value['row_count'],n_fragment_records=value['summary']['n_fragment_records'],
        n_qc_fragment_records=value['summary']['n_qc_fragment_records'],tss_defined=value['summary']['tss_defined'],
        tss_undefined=value['summary']['tss_undefined'],science_profile_sha256=PROFILE_SHA256,
        qc_resource_identity_sha256=value['qc_resource_identity_sha256'],resource_qualification=value['resource_qualification']['mode'])


def _receipt(destination,args,token=None):
    try:
        with (destination/'receipt.json').open('rb') as f: raw=f.read(16385)
        if len(raw)>16384: fail('QC_RECOVERY_MISMATCH')
        r=json.loads(raw,object_pairs_hook=v2._pairs,parse_constant=lambda _:fail('QC_RECOVERY_MISMATCH'))
        v2.shape(r,('artifact_type','schema_version','policy','execution_identity','arguments_sha256','manifest_sha256'))
        if (r['artifact_type']!='agent.barcode-qc-receipt' or type(r['schema_version']) is not int or r['schema_version']!=1
            or r['policy']!=m.POLICY or r['arguments_sha256']!=m.digest(args)
            or destination.name!='barcode-qc-'+r['execution_identity']
            or token is not None and r['execution_identity']!=token): fail('QC_RECOVERY_MISMATCH')
        v2.sha(r['execution_identity']);v2.sha(r['manifest_sha256'])
        return r
    except (OSError,ValueError,TypeError,KeyError): fail('QC_RECOVERY_MISMATCH')


def verify_public_result(arguments,result):
    args=m.validate_arguments(arguments);path=Path(result['manifest_path']);destination=path.parent
    if (path!=path.resolve() or path.name!='manifest.json' or destination.parent!=Path(args['output_dir'])): fail('QC_RESULT_MISMATCH')
    before=take_snapshots([path,destination/'receipt.json'])
    receipt=_receipt(destination,args)
    if receipt['manifest_sha256']!=result['manifest_sha256']: fail('QC_RECOVERY_MISMATCH')
    verified=verify_barcode_qc(path,expected_sha256=result['manifest_sha256']);value=verified.to_dict()
    if value['arguments']!=args or dict(result)!=_summary(value,path,result['manifest_sha256']): fail('QC_RESULT_MISMATCH')
    check_snapshots(before)
    return verified


def recover_barcode_qc(arguments,execution_identity):
    args,destination,token=_publication(arguments,execution_identity)
    if not destination.is_dir() or destination.is_symlink(): fail('QC_RECOVERY_UNAVAILABLE')
    receipt=_receipt(destination,args,token);path=destination/'manifest.json'
    value=m.load_manifest(path,receipt['manifest_sha256']).to_dict()
    result=_summary(value,path,receipt['manifest_sha256'])
    verify_public_result(args,result)
    return result


def execute_barcode_qc(arguments,execution_identity=None):
    from ._barcode_qc_production import produce
    cancellation_checkpoint()
    args,destination,token=_publication(arguments,execution_identity)
    output=Path(args['output_dir'])
    if output!=output.resolve() or destination.exists() or destination.is_symlink(): fail('QC_OUTPUT_CONFLICT')
    bound=binding.bind(args)
    if token!=_publication_token(args,execution_identity,bound.reference.resource_identity_sha256,
            bound.resource_qualification,bound.backend_identity): fail('QC_BINDING_MISMATCH')
    output.mkdir(parents=True,exist_ok=True)
    with (output/('.'+destination.name+'.lock')).open('a+b') as lease:
        # Nonblocking acquisition permits cooperative cancellation while waiting.
        import time
        while True:
            try: fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError: cancellation_checkpoint();time.sleep(0.1)
        if destination.exists() or destination.is_symlink(): fail('QC_OUTPUT_CONFLICT')
        stage=Path(tempfile.mkdtemp(prefix='.qc-attempt-',dir=output))
        try:
            outputs=produce(bound,stage)
            value=dict(artifact_type=m.ARTIFACT,schema_version=1,contract_version=m.CONTRACT,
                policy=m.POLICY,arguments=args,reference_identity_sha256=bound.reference.parent_reference_identity_sha256,
                qc_resource_identity_sha256=bound.reference.resource_identity_sha256,science_profile_sha256=PROFILE_SHA256,
                tss_method=PROFILE.tss_method,producer_authority=bound.producer_authority,
                resource_qualification=bound.resource_qualification,backend_identity=bound.backend_identity,**outputs)
            value['identity_sha256']=m.manifest_identity(value)
            payload=canonical(m.validate_manifest(value));path=stage/'manifest.json';path.write_bytes(payload)
            sha=m.resource(path)['sha256']
            verify_barcode_qc(path,expected_sha256=sha)
            shutil.rmtree(stage/'scratch')
            (stage/'receipt.json').write_bytes(canonical(dict(artifact_type='agent.barcode-qc-receipt',schema_version=1,
                policy=m.POLICY,execution_identity=token,arguments_sha256=m.digest(args),manifest_sha256=sha)))
            for p in stage.iterdir():
                with p.open('rb') as f: os.fsync(f.fileno())
            _fsync_dir(stage);bound.unchanged();cancellation_checkpoint()
            if destination.exists() or destination.is_symlink(): fail('QC_OUTPUT_CONFLICT')
            os.rename(stage,destination);_fsync_dir(output)
            from .authority_context import publication_moved
            publication_moved(stage, destination)
            # No cancellation after publication: executor checkpoints verified
            # success before observing later cancellation, retaining this artifact.
            return _summary(value,destination/'manifest.json',sha)
        finally:
            if stage.exists(): shutil.rmtree(stage)


def compute_scATAC_qc(fragments_manifest_path,fragments_manifest_sha256,
        qc_reference_manifest_path,qc_reference_manifest_sha256,output_dir)->BarcodeQCResult:
    """Compute verified metrics for every observed namespace/barcode; no selection."""
    return execute_barcode_qc(locals())
