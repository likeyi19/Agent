"""UMAP dependency and optional-label invariants on small simulated embeddings."""
from pathlib import Path

import anndata as ad
import numpy as np
import pytest

from agent.orchestration import PlanStep, build_default_tool_registry, verify_step
from agent.tools.analysis.embedding_analysis import (
    PROVENANCE_KEY, _analysis_digest, _cell_order_digest,
    cluster_cells, compute_cell_umap,
)
from test_embedding_analysis import embedding_artifacts, neighbors_result


def test_labels_do_not_change_native_umap_coordinates(neighbors_result, tmp_path):
    source = Path(neighbors_result['analysis_path'])
    before = source.read_bytes()
    clustered = cluster_cells(source, tmp_path/'cluster')
    plain = compute_cell_umap(source, tmp_path/'plain')
    labeled = compute_cell_umap(clustered['analysis_path'], tmp_path/'labeled')
    a, b = ad.read_h5ad(plain['analysis_path']), ad.read_h5ad(labeled['analysis_path'])
    np.testing.assert_array_equal(a.obsm['X_umap'], b.obsm['X_umap'])
    assert a.obs_names.equals(b.obs_names)
    assert 'leiden' not in a.obs and 'leiden' in b.obs
    assert plain['clustering_present'] is False
    assert 'clustering_present' not in labeled
    assert 'clustering_present' not in b.uns[PROVENANCE_KEY]
    assert source.read_bytes() == before


@pytest.mark.parametrize('mutation', [
    'missing_variant', 'invented_labels', 'invented_clustering', 'embeddings',
    'distances', 'connectivities', 'cell_order', 'result_variant',
])
def test_verification_rejects_inconsistent_unclustered_output(neighbors_result, tmp_path, mutation):
    args = dict(analysis_path=neighbors_result['analysis_path'], output_dir=str(tmp_path/'umap'))
    result = compute_cell_umap(**args)
    path = Path(result['analysis_path'])
    artifact = ad.read_h5ad(path)
    if mutation == 'missing_variant':
        del artifact.uns[PROVENANCE_KEY]['clustering_present']
    elif mutation == 'invented_labels':
        artifact.obs['leiden'] = 'dummy'
    elif mutation == 'invented_clustering':
        artifact.uns[PROVENANCE_KEY]['parameters']['clustering'] = dict(resolution=1.0)
    elif mutation == 'embeddings':
        artifact.obsm['X_epizoo'][0, 0] += 1
    elif mutation in ('distances', 'connectivities'):
        artifact.obsp[mutation].data[0] += 1
    elif mutation == 'cell_order':
        artifact.obs_names = list(reversed(artifact.obs_names))
        artifact.uns[PROVENANCE_KEY]['cell_order_sha256'] = _cell_order_digest(tuple(artifact.obs_names))
    else:
        del result['clustering_present']
    artifact.write_h5ad(path)
    result['analysis_sha256'] = _analysis_digest(path)  # Integrity alone is insufficient.
    checked = verify_step(PlanStep('umap', 'compute_cell_umap', args), args, result,
                          build_default_tool_registry())
    assert not checked.passed


@pytest.mark.parametrize('mutation', ['missing_graph', 'wrong_stage', 'unexpected_labels', 'missing_labels'])
def test_invalid_source_fails_before_native_umap(neighbors_result, tmp_path, monkeypatch, mutation):
    path = neighbors_result['analysis_path']
    if mutation == 'missing_labels':
        path = cluster_cells(path, tmp_path/'cluster')['analysis_path']
    artifact = ad.read_h5ad(path)
    if mutation == 'missing_graph':
        del artifact.obsp['connectivities']
    elif mutation == 'wrong_stage':
        artifact.uns[PROVENANCE_KEY]['stage'] = 'umap'
    elif mutation == 'unexpected_labels':
        artifact.obs['leiden'] = 'dummy'
    else:
        del artifact.obs['leiden']
    changed = tmp_path/'changed.h5ad'
    artifact.write_h5ad(changed)
    import scanpy as sc
    def forbidden(*args, **kwargs):
        pytest.fail('Invalid source reached UMAP')
    monkeypatch.setattr(sc.tl, 'umap', forbidden)
    with pytest.raises(ValueError):
        compute_cell_umap(changed, tmp_path/'umap')
