"""A small simulated analysis checks exact export values and source pinning."""
import hashlib
import tempfile
from pathlib import Path
from types import SimpleNamespace
import anndata as ad
import h5py
import numpy as np
import pytest
from test_embedding_analysis import embedding_artifacts, neighbors_result, clustering_result
from agent.tools.analysis.embedding_analysis import compute_cell_umap
from agent.application import analysis_delivery as delivery
from agent.application.matrix_delivery import MatrixDownloadPreparer, MatrixDeliveryError


@pytest.fixture
def umap_result(clustering_result, tmp_path):
    return compute_cell_umap(clustering_result['analysis_path'], tmp_path/'umap')


def test_export_keeps_all_rows_embeddings_labels_and_coordinates(umap_result):
    path=Path(umap_result['analysis_path']); original=path.read_bytes()
    assert hashlib.sha256(original).hexdigest()==umap_result['analysis_sha256']
    data=ad.read_h5ad(path)
    with path.open('rb') as source, tempfile.TemporaryFile('w+b') as destination:
        delivery.create_analysis_export(source,destination,dict(n_cells=data.n_obs),
            source_sha256=umap_result['analysis_sha256'],source_size_bytes=len(original),
            manifest_sha256=None,max_bytes=len(original)+65536)
        with h5py.File(destination,'r') as h:
            from anndata.io import read_elem
            exported=read_elem(h)
    assert exported.obs_names.equals(data.obs_names)
    assert exported.obs['leiden'].equals(data.obs['leiden'])
    assert set(exported.obsm)=={'X_epizoo','X_umap'}
    for key in exported.obsm:np.testing.assert_array_equal(exported.obsm[key],data.obsm[key])
    assert set(exported.uns)=={'agent_scientific_export'}
    assert str(path.parent) not in str(exported.uns)
    assert exported.X is None and path.read_bytes()==original


def test_shared_snapshot_rejects_substitution(umap_result):
    path=Path(umap_result['analysis_path'])
    artifact=SimpleNamespace(path=path,managed_root=path.parent,sha256='0'*64,
        size_bytes=path.stat().st_size,contract=delivery.CONTRACT,role=delivery.ROLE,
        manifest={'n_cells':umap_result['n_cells']},manifest_sha256=None)
    prep=MatrixDownloadPreparer()
    with pytest.raises(MatrixDeliveryError):prep.prepare(artifact)
    assert prep.active_count==0


def test_shared_snapshot_delivers_valid_export(umap_result):
    path=Path(umap_result['analysis_path'])
    artifact=SimpleNamespace(path=path,managed_root=path.parent,sha256=umap_result['analysis_sha256'],
        size_bytes=path.stat().st_size,contract=delivery.CONTRACT,role=delivery.ROLE,
        manifest={'n_cells':umap_result['n_cells']},manifest_sha256=None)
    prep=MatrixDownloadPreparer()
    with prep.prepare(artifact) as download:
        assert download.filename=='analysis.h5ad'
        assert download.source_sha256==umap_result['analysis_sha256']
    assert prep.active_count==0
