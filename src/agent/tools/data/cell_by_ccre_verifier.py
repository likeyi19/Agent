"""External H5AD representation conservation and accepted raw-matrix reading.

No fragment intersection or scientific matrix reconstruction remains.
"""
from pathlib import Path
import tempfile
import time
import h5py
from . import scatac_matrix_contract as m, _matrix_storage as io
from agent.tools._file_identity import file_sha256
from .scatac_fragments_v2_verifier import take_snapshots, check_snapshots
from .authority_context import owned_verification

@owned_verification('matrix')
def verify_cell_by_ccre(manifest_path, *, expected_sha256, bedtools_path=None, limits=io.MatrixLimits(), scratch_parent=None):
    path = Path(manifest_path)
    from .external_matrix_contract import is_external
    from agent.tools._file_identity import file_closure
    if file_closure([path])[0]['sha256'] != expected_sha256: m.fail('MATRIX_MANIFEST_MISMATCH')
    value = m.load_manifest_bytes(path.read_bytes())
    if is_external(value):
        return _verify_external(path, value, expected_sha256, limits, scratch_parent)
    from .raw_history import accepted
    _, record = accepted('matrix', path, expected_sha256)
    return dict(record['resources']['result_metadata'])

def _verify_external(path, value, expected_sha256, limits, scratch_parent):
    """Matrix-owner conservation proof; does not invoke the adoption writer."""
    from . import external_matrix_contract as e, _external_matrix_io as external
    contract = e.contract_for(value)
    started = time.monotonic()
    source = Path(value['source']['path']); payload = path.parent/'matrix.h5ad'
    rp = value['reference']
    args = dict(reference_manifest_path=rp['manifest_path'],reference_manifest_sha256=rp['manifest_sha256'],
                species=value['species'],assembly=value['assembly'],matrix_semantics=value['matrix_semantics'])
    ref, reference_snapshots = external.reference(args, limits, contract=contract)
    before = tuple(sorted(set(reference_snapshots + take_snapshots([path,source,payload]))))
    if (ref.reference_identity_sha256 != rp['identity_sha256']
            or ref.ccre.ordered_feature_sha256 != value['ordered_feature_sha256']): m.fail('MATRIX_REFERENCE_MISMATCH')
    for file,record in ((source,value['source']),(payload,value['matrix'])):
        if file.stat().st_size > limits.max_scratch_bytes: m.fail('MATRIX_RESOURCE_LIMIT')
        if file.stat().st_size != record['size_bytes'] or file_sha256(file) != record['sha256']: m.fail('MATRIX_PAYLOAD_MISMATCH')
    with tempfile.TemporaryDirectory(prefix='.matrix-conservation-',dir=scratch_parent) as directory:
        budget = io.Budget(directory,limits)
        with io.database(Path(directory)/'axes.sqlite',budget) as db, h5py.File(source,'r') as src, h5py.File(payload,'r') as out:
            n,p,cells = external.inspect_axes(src,ref,args,db,budget)
            if (value['shape'] != [n,p] or value['ordered_cells_sha256'] != cells
                    or external.inspect_axes(out,ref,args,db,budget,published=True) != (n,p,cells)):
                m.fail('MATRIX_CONSERVATION_MISMATCH')
            source_summary = e.logical_identity(n,p,io.csr_rows(src,n,p,budget),value['matrix_semantics'])
            # Read source CSR directly, independently of the chunk-copy writer.
            # Canonical source validation above guarantees unique sorted entries.
            def compare():
                for row,(cols,values) in enumerate(io.csr_rows(out,n,p,budget)):
                    left,right = map(int,src['X']['indptr'][row:row+2])
                    if (cols != src['X']['indices'][left:right].astype(object).tolist()
                            or values != src['X']['data'][left:right].astype(object).tolist()):
                        m.fail('MATRIX_CONSERVATION_MISMATCH')
                    yield cols,values
            summary = e.logical_identity(n,p,compare(),value['matrix_semantics'])
            if source_summary != summary or any(value[k] != v for k,v in summary.items()): m.fail('MATRIX_CONSERVATION_MISMATCH')
    check_snapshots(before)
    return dict(manifest_path=str(path),manifest_sha256=expected_sha256,identity_sha256=value['identity_sha256'],
                **summary,verification_seconds=time.monotonic()-started,verification_scratch_peak_bytes=budget.peak_bytes)
