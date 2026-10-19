from pathlib import Path
import io
import tempfile
import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from agent.tools.analysis import pca_embedding as pca
from agent.tools.analysis import embedding_analysis as analysis
from agent.tools.analysis import neighbors_publication as publication

@pytest.fixture
def source(tmp_path):
    x = np.random.default_rng(41).poisson(.6, (24,71)).astype(np.float64)
    x[:,0] += 30  # Distinguish centered PCA from uncentered SVD.
    path = tmp_path/'cells.h5ad'
    ad.AnnData(sparse.csr_matrix(x), obs=pd.DataFrame(index=[f'c{i}' for i in range(len(x))]),
               var=pd.DataFrame(index=[f'p{i}' for i in range(x.shape[1])])).write_h5ad(path)
    return path,x

@pytest.fixture
def embedding(source,tmp_path):
    return pca.pca_embed_cells(source[0],tmp_path/'pca',n_components=5)


def test_sparse_pca_matches_centered_dense_reference(source,embedding,monkeypatch):
    x=source[1]; data=ad.read_h5ad(embedding['analysis_path'])
    _,s,vt=np.linalg.svd(x-x.mean(axis=0),full_matrices=False)
    fit=data.uns['pca_fit']; actual=data.obsm['X_pca']
    np.testing.assert_allclose(fit['variance'],s[:5]**2/(len(x)-1),rtol=1e-8)
    np.testing.assert_allclose(fit['loadings'].T@fit['loadings'],vt[:5].T@vt[:5],atol=1e-8)
    assert actual.shape==(24,5) and 'X_epizoo' not in data.obsm
    assert pca.verify_result(dict(input_path=source[0],output_dir=Path(embedding['analysis_path']).parent,n_components=5),embedding)['method']=='PCA'
    # The scientific execution must never densify the source sparse matrix.
    def forbidden(*a,**k): raise AssertionError('complete sparse matrix densification')
    monkeypatch.setattr(sparse.csr_matrix,'toarray',forbidden)
    pca.pca_embed_cells(source[0],Path(embedding['analysis_path']).parent/'second',n_components=5)


def test_default_dimensions_and_no_input_mutation(source,tmp_path):
    before=source[0].read_bytes()
    result=pca.pca_embed_cells(source[0],tmp_path/'default')
    assert result['embedding_dim']==23
    assert result['parameters']['normalization']=='none'
    assert source[0].read_bytes()==before

@pytest.mark.parametrize('damage',['scores','cell_order','variance','loadings','parameters','source','lower_components'])
def test_independent_verifier_rejects_corruption(source,embedding,damage):
    data=ad.read_h5ad(embedding['analysis_path'])
    if damage=='scores': data.obsm['X_pca'][0,0]+=1
    elif damage=='cell_order': data.obs_names=list(reversed(data.obs_names))
    elif damage=='variance': data.uns['pca_fit']['variance'][0]*=2
    elif damage=='loadings': data.uns['pca_fit']['loadings'][0,0]+=1
    elif damage=='parameters': data.uns[pca.KEY]['parameters']['zero_center']=False
    elif damage=='lower_components':
        x=source[1]; _,singular,vt=np.linalg.svd(x-x.mean(axis=0),full_matrices=False)
        v=vt[8:13];variance=singular[8:13]**2/(len(x)-1)
        data.obsm['X_pca']=((x-x.mean(axis=0))@v.T).astype('float32')
        data.uns['pca_fit'].update(loadings=v,variance=variance,
            variance_ratio=variance/data.uns['pca_fit']['total_variance'])
    elif damage=='source':
        original=ad.read_h5ad(source[0]); original.X.data[0]+=1; original.write_h5ad(source[0])
    data.write_h5ad(embedding['analysis_path'])
    altered=dict(embedding,analysis_sha256=analysis._analysis_digest(embedding['analysis_path']))
    with pytest.raises(ValueError):pca._verify_result(dict(input_path=source[0],n_components=5),altered)

@pytest.mark.parametrize('invalid',['nan','constant','duplicate','components'])
def test_invalid_scientific_inputs(source,tmp_path,invalid):
    data=ad.read_h5ad(source[0]);kwargs={}
    if invalid=='nan':data.X.data[0]=np.nan
    elif invalid=='constant':data.X=sparse.csr_matrix(np.ones(data.shape))
    elif invalid=='duplicate':data.obs_names=['same']*data.n_obs
    else:kwargs['n_components']=24
    data.write_h5ad(source[0])
    with pytest.raises(ValueError):pca.pca_embed_cells(source[0],tmp_path/'bad',**kwargs)


def test_shared_graph_cluster_umap_and_exports(embedding,tmp_path):
    from agent.application.analysis_delivery import create_analysis_export
    from agent.tools.analysis.neighbors_delivery import create_neighbors_export
    g=analysis.build_cell_neighbors(embedding['embedding_path'],embedding['cell_ids_path'],tmp_path/'g')
    manifest,sha=publication.load_manifest(g['manifest_path'])
    assert manifest['contract_version']==publication.CONTRACT
    assert publication.verify_result(manifest['arguments'],g)==g
    c=analysis.cluster_cells(g['analysis_path'],tmp_path/'c')
    from agent.tools.analysis.clustering_evaluation import _validate_evaluation_analysis
    _validate_evaluation_analysis(ad.read_h5ad(c['analysis_path']))
    u=analysis.compute_cell_umap(c['analysis_path'],tmp_path/'u')
    for result,stage in ((embedding,'embedding'),(c,'clustering'),(u,'umap')):
        path=Path(result['analysis_path']);before=path.read_bytes()
        with path.open('rb') as stream,tempfile.TemporaryFile('w+b') as output:
            create_analysis_export(stream,output,dict(n_cells=24,stage=stage),source_sha256=result['analysis_sha256'],
                source_size_bytes=len(before),manifest_sha256=None,max_bytes=len(before)+65536)
            with h5py.File(output,'r') as h:
                from anndata.io import read_elem
                exported=read_elem(h)
        np.testing.assert_array_equal(exported.obsm['X_pca'],ad.read_h5ad(path).obsm['X_pca'])
        assert exported.uns['agent_scientific_export']['method']=='PCA'
        assert exported.uns['agent_scientific_export']['source_sha256']==result['analysis_sha256']
        assert exported.uns['agent_scientific_export']['input_sha256']==embedding['source_sha256']
        assert str(tmp_path) not in str(exported.uns) and path.read_bytes()==before
    path=Path(g['analysis_path'])
    with path.open('rb') as stream,tempfile.TemporaryFile('w+b') as output:
        create_neighbors_export(stream,output,manifest,source_sha256=manifest['analysis']['sha256'],
            source_size_bytes=path.stat().st_size,manifest_sha256=sha,max_bytes=path.stat().st_size+65536)
        with h5py.File(output,'r') as h:
            assert 'X_pca' in h['obsm'] and 'embedding' not in h['uns/agent_milestone6']


def test_feature_positions_and_hvg_metadata_are_not_implicitly_filtered(source,tmp_path):
    data=ad.read_h5ad(source[0]);data.var_names=['same']*data.n_vars
    data.var['highly_variable']=[True]+[False]*(data.n_vars-1)
    data.write_h5ad(source[0]);before=source[0].read_bytes()
    result=pca.pca_embed_cells(source[0],tmp_path/'all-features',n_components=5)
    fit=ad.read_h5ad(result['analysis_path']).uns['pca_fit']
    assert fit['loadings'].shape==(5,71)
    assert result['n_features']==71 and source[0].read_bytes()==before


def test_embedding_channels_do_not_enable_cross_method_label_transfer():
    from agent.orchestration.registry import build_default_tool_registry
    from agent.orchestration.semantic_compiler import build_m92_semantic_compiler_contract
    registry=build_default_tool_registry()
    channels=build_m92_semantic_compiler_contract(registry).step_output_channels
    consumers={c.consumer_tool_name for c in channels if c.producer_tool_name=='pca_embed_cells'}
    assert consumers=={'build_cell_neighbors'}


def test_method_observations_are_not_pca_execution(source,monkeypatch):
    import scanpy as sc
    def forbidden(*a,**k):raise AssertionError('observation executed PCA')
    monkeypatch.setattr(sc.pp,'pca',forbidden)
    observation=pca.observe_methods(source[0])
    assert observation['pca']=='eligible' and observation['epizoo']==[]
    assert str(source[0]) not in str(observation)
