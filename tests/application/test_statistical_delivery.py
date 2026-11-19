"""Simulated native-layout fixtures exercise delivery, not biological validity."""
from io import BytesIO
from types import SimpleNamespace
import hashlib

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from agent.application import statistical_delivery as owner
from agent.application.delivery_policy import DeliveryPolicy
from agent.application.matrix_delivery import MatrixDownloadPreparer, MatrixDeliveryError
from agent.tools.analysis.h5ad_delivery import DESTINATION_OVERHEAD_BYTES


def fixture(tmp_path, tool):
    x = sparse.csr_matrix([[2, 0, 4], [0, 5, 1]], dtype=np.int64) if tool == 'build_replicate_pseudobulk' else None
    data = ad.AnnData(X=x, obs=pd.DataFrame({'condition':['treated','control'],
        'replicate':['mouse-2','mouse-1']},index=['sample-2','sample-1']),
        var=pd.DataFrame({'logFC':[1.,np.nan,-1.], 'da_status':['tested','filtered','tested']},
                         index=['peak-3','peak-1','peak-2']),
        uns={'method':{'source_path':'/private/server/input.h5ad', 'model':{'checkpoint_path':'/private/model'},
            'normalization':'TMM', 'optional':None, 'version':2,
            'credentials':{'value':'private-credential-fixture'}}})
    path = tmp_path/'result.h5ad';data.write_h5ad(path,compression='gzip')
    item = SimpleNamespace(path=path,managed_root=tmp_path,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        size_bytes=path.stat().st_size,contract=owner.CONTRACT,role=owner.ROLE,
        manifest=dict(tool_name=tool,accepted_step_sha256='a'*64),manifest_sha256=None,
        reference=SimpleNamespace(filename=owner.TOOLS[tool][2]))
    return item,data


@pytest.mark.parametrize('tool',list(owner.TOOLS))
def test_statistical_snapshot_preserves_axes_values_and_safe_methods(tmp_path,tool):
    item,source=fixture(tmp_path,tool);before=item.path.read_bytes()
    manager=MatrixDownloadPreparer();snapshot=manager.prepare(item)
    try:
        payload=b''.join(snapshot.iter_chunks())
        assert snapshot.source_sha256==item.sha256
        assert snapshot.sha256==hashlib.sha256(payload).hexdigest()
        result=ad.read_h5ad(BytesIO(payload))
        pd.testing.assert_frame_equal(source.obs,result.obs)
        pd.testing.assert_frame_equal(source.var,result.var)
        assert result.uns['method']['normalization']=='TMM' and result.uns['method']['optional'] is None
        assert 'source_path' not in result.uns['method']
        assert 'credentials' not in result.uns['method']
        assert 'checkpoint_path' not in result.uns['method']['model']
        assert result.uns['agent_scientific_export']['source_sha256']==item.sha256
        if source.X is None:assert result.X is None
        else:assert sparse.isspmatrix_csr(result.X) and (source.X!=result.X).nnz==0
    finally:snapshot.close()
    assert manager.active_count==0 and item.path.read_bytes()==before


@pytest.mark.parametrize('damage',['changed_bytes','private_identity','external_link'])
def test_statistical_export_rejects_invalid_bytes_or_scientific_identity(tmp_path,damage):
    item,_=fixture(tmp_path,'build_replicate_pseudobulk')
    if damage=='changed_bytes':item.path.write_bytes(b'changed')
    else:
        with h5py.File(item.path,'r+') as f:
            if damage=='private_identity':f['obs/replicate'][0]='/private/identity'
            else:f['external']=h5py.ExternalLink('/private/file','/data')
        item.sha256=hashlib.sha256(item.path.read_bytes()).hexdigest();item.size_bytes=item.path.stat().st_size
    manager=MatrixDownloadPreparer()
    with pytest.raises(MatrixDeliveryError):manager.prepare(item)
    assert manager.active_count==0


def test_axis_only_delivery_does_not_admit_missing_count_matrix(tmp_path):
    from agent.tools.analysis.peak_h5ad import inspect
    item,_=fixture(tmp_path,'run_replicate_differential_accessibility')
    with pytest.raises(ValueError,match='requires X'):
        inspect(item.path)
    inspect(item.path,require_matrix=False)


def test_native_peak_export_reencodes_filters_and_enforces_completed_size(tmp_path):
    from anndata.io import write_elem
    from agent.application.native_delivery import definitions

    plugin = pytest.importorskip('hdf5plugin')
    counts = np.random.default_rng(4).integers(1, 65537, size=(64, 4096), dtype=np.uint32)
    data = ad.AnnData(
        sparse.csr_matrix(counts),
        obs=pd.DataFrame(index=[f'cell-{i}' for i in range(64, 0, -1)]),
        var=pd.DataFrame(
            {'chrom': ['chr1'] * 4096},
            index=[f'chr1:{i * 10}-{i * 10 + 9}' for i in range(4096, 0, -1)],
        ),
    )
    path = tmp_path / 'result.h5ad'
    with h5py.File(path, 'w') as handle:
        handle.attrs.update({'encoding-type': 'anndata', 'encoding-version': '0.1.0'})
        write_elem(handle, 'X', data.X, dataset_kwargs=plugin.Blosc(
            cname='zstd', clevel=5, shuffle=plugin.Blosc.BITSHUFFLE))
        write_elem(handle, 'obs', data.obs, dataset_kwargs={'compression': 'gzip'})
        write_elem(handle, 'var', data.var)
        write_elem(handle, 'uns', {'method': {
            'source_path': '/private/input.h5ad', 'counting': 'paired-insertion',
        }})
        assert handle['X/data'].compression == 'unknown'
        for key in ('var/_index', 'var/chrom'):
            assert handle[key].chunks is None
            assert h5py.check_string_dtype(handle[key].dtype).length is None

    before = path.read_bytes()
    spec = dict(definitions())['run_snapatac2']
    item = SimpleNamespace(
        path=path, managed_root=tmp_path, sha256=hashlib.sha256(before).hexdigest(),
        size_bytes=len(before), contract=spec.contract, role='result.h5ad',
        manifest=dict(tool_name='run_snapatac2', function_name='snapatac2.pp.make_peak_matrix',
                      accepted_step_sha256='a' * 64),
        manifest_sha256=None,
    )
    preparer = MatrixDownloadPreparer()
    with preparer.prepare(item) as snapshot:
        payload = b''.join(snapshot.iter_chunks())
        assert len(payload) > len(before) + DESTINATION_OVERHEAD_BYTES
        assert snapshot.source_sha256 == item.sha256
    result = ad.read_h5ad(BytesIO(payload))
    pd.testing.assert_frame_equal(result.obs, data.obs)
    pd.testing.assert_frame_equal(result.var, data.var)
    assert sparse.isspmatrix_csr(result.X) and result.X.dtype == counts.dtype
    assert (result.X != data.X).nnz == 0
    assert result.uns['method'] == {'counting': 'paired-insertion'}
    assert result.uns['agent_scientific_export']['source_sha256'] == item.sha256
    with h5py.File(BytesIO(payload), 'r') as handle:
        assert handle['X/data'].compression == 'gzip'

    capped = MatrixDownloadPreparer(DeliveryPolicy(delivered_bytes=len(before) + DESTINATION_OVERHEAD_BYTES))
    with pytest.raises(MatrixDeliveryError) as error:
        capped.prepare(item)
    assert error.value.code == 'SCIENTIFIC_DOWNLOAD_LIMIT'
    assert capped.active_count == 0
    assert capped._reserved_bytes == 0
    assert preparer.active_count == 0
    assert path.read_bytes() == before
