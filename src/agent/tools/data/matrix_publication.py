"""M11.5c public matrix boundary: durable envelope around unchanged M11.5b science."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

from agent.tools._cancellation import cancellation_checkpoint
from . import scatac_matrix_contract as m, scatac_fragments_v2 as v2
from .scatac_qc_reference import _fsync_dir

RECOVERY_POLICY='build-scatac-cell-by-ccre-v1'
ARGUMENTS=tuple(f'{prefix}_{suffix}' for prefix in ('fragments_manifest','selected_cells_manifest','reference_manifest')
                for suffix in ('path','sha256'))+('output_dir',)








def digest(value):return hashlib.sha256(m.canonical(value)).hexdigest()






def _load(path,sha):
    m.sha(sha)
    with path.open('rb') as f:raw=f.read(m.MAX_MANIFEST_BYTES+1)
    if hashlib.sha256(raw).hexdigest()!=sha:m.fail('MATRIX_MANIFEST_MISMATCH')
    return m.load_manifest_bytes(raw)


def _receipt(destination,args,token=None, *, policy=RECOVERY_POLICY, profile=m.PROFILE_SHA256):
    try:
        with (destination/'receipt.json').open('rb') as f:raw=f.read(16385)
        if len(raw)>16384:m.fail('MATRIX_RECOVERY_MISMATCH')
        r=json.loads(raw,object_pairs_hook=v2._pairs,parse_constant=lambda _:m.fail('MATRIX_RECOVERY_MISMATCH'))
        m.shape(r,('artifact_type','schema_version','policy','execution_identity','arguments_sha256','manifest_sha256','matrix_profile_sha256'))
        if (r['artifact_type']!='agent.cell-by-ccre-receipt' or type(r['schema_version']) is not int or r['schema_version']!=1
                or r['policy']!=policy or r['arguments_sha256']!=digest(args)
                or r['matrix_profile_sha256']!=profile
                or destination.name!='cell-by-ccre-'+r['execution_identity']
                or token is not None and r['execution_identity']!=token):m.fail('MATRIX_RECOVERY_MISMATCH')
        m.sha(r['execution_identity']);m.sha(r['manifest_sha256'])
        return r
    except (OSError,ValueError,TypeError,KeyError):m.fail('MATRIX_RECOVERY_MISMATCH')








def _execute_publication(args,destination,token,build,summary,policy,profile, *,
                         load_manifest=_load, receipt_factory=None):
    """Shared matrix receipt, lease, cancellation and atomic publication envelope."""
    output=Path(args['output_dir'])
    if output!=output.resolve() or destination.exists() or destination.is_symlink():m.fail('MATRIX_OUTPUT_CONFLICT')
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
            built=build(stage/'artifact')
            value=load_manifest(Path(built['manifest_path']),built['manifest_sha256'])
            receipt=dict(artifact_type='agent.cell-by-ccre-receipt',schema_version=1,policy=policy,
                execution_identity=token,arguments_sha256=digest(args),manifest_sha256=built['manifest_sha256'],matrix_profile_sha256=profile)
            if receipt_factory is not None:
                receipt=receipt_factory(args,token,built['manifest_sha256'])
            (stage/'receipt.json').write_bytes(m.canonical(receipt))
            with (stage/'receipt.json').open('rb') as f:os.fsync(f.fileno())
            _fsync_dir(stage);cancellation_checkpoint()
            if destination.exists() or destination.is_symlink():m.fail('MATRIX_OUTPUT_CONFLICT')
            os.rename(stage,destination);_fsync_dir(output)
            from .authority_context import publication_moved
            publication_moved(stage, destination)
            # No cancellation checkpoint after the outer immutable publication.
            return summary(value,destination/'artifact/manifest.json',built['manifest_sha256'])
        finally:
            if stage.exists():shutil.rmtree(stage)
