"""New native result families: source positions, safe records and closed dispatch."""
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from agent.tools.analysis import native_matrix as owner, peak_h5ad
from agent.tools.analysis.native_matrix_worker import returned_data
from agent.tools.analysis.native_tables import MACS3


def test_ai_method_offers_retain_their_scientific_parameters():
    for package in owner.PACKAGES:
        assert set(k for f in package.FUNCTIONS.values() for k in f) == set(k for k,_ in package.NATIVE_ARGUMENTS)


@pytest.mark.parametrize('function,kw', [
    ('tl.leiden', {'partition_type':'exec'}), ('pp.neighbors', {'adata':'other.h5ad'}),
    ('tl.umap', {'init_pos':[[1,2]]}), ('tl.rank_genes_groups', {'callback':'eval'}),
])
def test_native_kwargs_cannot_supply_code_or_unbound_objects(function, kw):
    with pytest.raises(ValueError): owner.SCANPY.native_arguments(dict(function_name=function, kwargs=kw))


def test_values_are_not_repaired_and_omission_stays_omission():
    for name, kw in [('pp.normalize_total', {'target_sum':-1}), ('tl.leiden', {'resolution':-7}),
                     ('pp.highly_variable_genes', {'n_top_genes':-3})]:
        package = owner.SNAPATAC2 if name == 'tl.leiden' else owner.SCANPY
        assert package.native_arguments(dict(function_name=name, kwargs=kw)) == kw
        assert package.native_arguments(dict(function_name=name)) == {}
    assert MACS3.native_arguments(dict(function_name='pileup', kwargs={'extsize':-1})) == {'extsize':-1}
    assert MACS3.parameter_prerequisites(dict(function_name='cmbreps')) == ('control_path',)
    with pytest.raises(ValueError): MACS3.native_arguments(dict(function_name='pileup',kwargs={'ofile':'/tmp/escape'}))


@pytest.mark.parametrize('forgery', [None, 'positions', 'unauthorized'])
def test_subset_lineage_uses_positions_even_for_duplicate_feature_names(tmp_path, monkeypatch, forgery):
    a=ad.AnnData(sparse.csr_matrix([[1.,2.,3.],[4.,5.,6.]]),
        obs=pd.DataFrame(index=['c2','c1']),var=pd.DataFrame(index=['peak','peak','third']))
    source=tmp_path/'source.h5ad';a.write_h5ad(source);original=source.read_bytes()
    def native(self,args,stage,matrix):
        b=matrix[:,[1,2]].copy();b.write_h5ad(stage/'result.h5ad')
        return dict(completion='succeeded',versions={'scanpy':'SIMULATED'},implementation=[],
            native_output_type='NoneType',result_slots=[],source_axis_positions={'cells':[0,1], 'features':[0,2] if forgery=='positions' else [1,2]})
    monkeypatch.setattr(owner.MatrixPackage,'native',native)
    args=dict(input_path=str(source),output_dir=str(tmp_path/'out'),function_name='pp.highly_variable_genes' if forgery!='unauthorized' else 'pp.pca',kwargs={'subset':True} if forgery!='unauthorized' else {})
    if forgery:
        with pytest.raises(ValueError):owner.SCANPY.execute(args)
    else:
        result=owner.SCANPY.execute(args);b=ad.read_h5ad(result['matrix_path'])
        np.testing.assert_array_equal(b.X.toarray(),[[2.,3.],[5.,6.]])
        assert b.var_names.tolist()==['peak','third']
        assert b.uns['agent_native_invocation']['source_axis_positions']['features'].tolist()==[1,2]
    assert source.read_bytes()==original


def test_native_interval_categories_keep_bounds_codes_and_order(tmp_path):
    frame=pd.DataFrame({'mean_bin':pd.cut([1.,3.,2.,np.nan],2),'stat':[2.,4.,3.,np.nan]},index=['p4','p2','p1','p3'])
    encoded=returned_data(frame)
    a=ad.AnnData(np.zeros((2,3)));a.uns['agent_native_return']=encoded
    path=tmp_path/'returned.h5ad';a.write_h5ad(path);peak_h5ad.inspect(path)
    restored=ad.read_h5ad(path).uns['agent_native_return'];col=restored['interval_columns']['mean_bin']
    categories=pd.IntervalIndex.from_arrays(col['left'],col['right'],closed=col['closed'])
    decoded=restored['frame'];decoded['mean_bin']=pd.Categorical.from_codes(col['codes'],categories,ordered=col['ordered'])
    pd.testing.assert_frame_equal(decoded[list(restored['column_order'])],frame)


@pytest.mark.parametrize('private', [False,True])
def test_rank_records_storage_and_download_privacy(tmp_path, private):
    from agent.application.matrix_delivery import _redactions
    a=ad.AnnData(sparse.eye(2,format='csr'));a.uns['rank_genes_groups']={
        'names':np.array([('/private/file' if private else 'chr1:0-10','peak2')],dtype=[('a','O'),('b','O')]),
        'scores':np.array([(float('nan'),float('inf'))],dtype=[('a','f8'),('b','f8')])}
    path=tmp_path/'rank.h5ad';a.write_h5ad(path);peak_h5ad.inspect(path)
    with h5py.File(path) as h:
        assert ('/uns/rank_genes_groups' in _redactions(h)) is private
    if private:
        with pytest.raises(ValueError,match='Private'):peak_h5ad.inspect(path,privacy=True)
    else:peak_h5ad.inspect(path,privacy=True)


def test_compound_reference_objects_are_not_scientific_records(tmp_path):
    a=ad.AnnData(np.ones((2,2)));path=tmp_path/'unsafe.h5ad';a.write_h5ad(path)
    with h5py.File(path,'a') as h:
        h['uns'].create_dataset('unsafe',shape=(1,),dtype=np.dtype([('ref',h5py.ref_dtype)]))
    with pytest.raises(ValueError,match='record field'):peak_h5ad.inspect(path)


@pytest.mark.parametrize('size', [256, 257, 65536, 65537])
def test_record_string_bound_is_shared_by_storage_and_privacy(tmp_path, size):
    from agent.application.matrix_delivery import _private_metadata
    a = ad.AnnData(np.ones((2, 2)))
    a.uns['names'] = np.array([('x' * size,)], dtype=[('group', 'O')])
    path = tmp_path / 'record.h5ad'
    a.write_h5ad(path)
    if size > 65536:
        with pytest.raises(ValueError, match='65536'):
            peak_h5ad.inspect(path)
        with h5py.File(path) as handle, pytest.raises(ValueError, match='65536'):
            _private_metadata(handle['uns/names'])
    else:
        peak_h5ad.inspect(path, privacy=True)
        with h5py.File(path) as handle:
            assert list(peak_h5ad.record_strings(handle['uns/names'], 'group')) == ['x' * size]
            assert not _private_metadata(handle['uns/names'])


@pytest.mark.parametrize('function', [
    'computeVariability', 'differentialVariability', 'differentialDeviations',
    'getSampleCorrelation', 'getSampleDistance',
])
def test_chromvar_assay_meaning_and_group_order_follow_native_function(tmp_path, function):
    from agent.tools.analysis import chromvar_native
    a = ad.AnnData(np.array([[1., 2.], [3., 4.]]),
        obs=pd.DataFrame({'group': ['B', 'A']}, index=['c2', 'c1']),
        var=pd.DataFrame(index=['m2', 'm1']))
    a.layers['z_scores'] = np.array([[5., 6.], [7., 8.]])
    source = tmp_path / 'scores.h5ad'
    a.write_h5ad(source)
    before = source.read_bytes()
    stage = tmp_path / 'stage'
    stage.mkdir()
    args = dict(input_path=str(source), function_name=function)
    if function.startswith('differential'):
        args['group_key'] = 'group'
    effective, n, m = chromvar_native.prepare_input(args, stage)
    assert (n, m) == a.shape
    assert effective['measurement'] == ('supplied_chromVAR_z_assay'
        if function in ('computeVariability', 'differentialVariability')
        else 'supplied_chromVAR_deviations_with_z_assay')
    np.testing.assert_array_equal(np.fromfile(stage / 'deviations.bin', dtype='<f8').reshape(a.shape), a.X)
    np.testing.assert_array_equal(np.fromfile(stage / 'z.bin', dtype='<f8').reshape(a.shape), a.layers['z_scores'])
    if 'group_key' in args:
        cells = pd.read_csv(stage / 'cells.tsv', sep='\t')
        assert cells.cell_id.tolist() == ['c2', 'c1'] and cells.group.tolist() == ['B', 'A']
    assert source.read_bytes() == before


def test_bed_strands_survive_delivery_without_admitting_formulas():
    import io
    from agent.application.native_delivery import prepare, PRIVATE_LABEL
    text=prepare(io.BytesIO(b'chr1\t0\t10\t.\t.\t+\nchr1\t20\t30\t.\t.\t-\nchr1\t40\t50\t=SUM(A1)\t.\t+\n'), 'result.tsv').decode()
    assert text.splitlines()[0].endswith('\t+') and text.splitlines()[1].endswith('\t-')
    assert '=SUM' not in text and PRIVATE_LABEL in text


@pytest.mark.parametrize('has_x', [False,True])
def test_reuploaded_native_export_delivers_again_without_science(tmp_path,has_x):
    import hashlib
    from agent.application.statistical_delivery import create_statistical_export
    from agent.tools.analysis.h5ad_delivery import EXPORT_KEY
    a=ad.AnnData(X=np.ones((3,2)) if has_x else None,obs=pd.DataFrame(index=['a','b','c']),var=pd.DataFrame(index=['p','q']))
    a.obsm['X_pca']=np.arange(6).reshape(3,2)
    paths=[tmp_path/(str(i)+'.h5ad') for i in range(3)];a.write_h5ad(paths[0])
    for source,dest in zip(paths,paths[1:]):
        before=source.read_bytes();sha=hashlib.sha256(before).hexdigest()
        with source.open('rb') as inp,dest.open('w+b') as out:
            create_statistical_export(inp,out,dict(tool_name='run_scanpy',accepted_step_sha256='a'*64),
                source_sha256=sha,source_size_bytes=len(before),manifest_sha256='b'*64,max_bytes=1024**2)
        result=ad.read_h5ad(dest);assert result.uns[EXPORT_KEY]['source_sha256']==sha
        assert source.read_bytes()==before
        np.testing.assert_array_equal(result.obsm['X_pca'],a.obsm['X_pca'])
        assert (result.X is not None)==has_x
    assert result.uns[EXPORT_KEY]['supersedes_supplied_export_marker']
