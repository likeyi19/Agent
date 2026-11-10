"""Accepted historical PCA descriptors remain readable; execution is retired."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from agent.orchestration import build_default_tool_registry
from agent.tools.analysis import embedding_analysis as common, pca_embedding as history, pca_publication


@pytest.mark.parametrize('contract', [history.CONTRACT, history.TFIDF_CONTRACT])
def test_historical_descriptor_preserves_method_and_representation(contract):
    data = ad.AnnData(obs=pd.DataFrame(index=['a','b','c']), obsm={'X_pca': np.ones((3,2),dtype='float32')})
    preprocessing = 'tfidf_l2' if contract == history.TFIDF_CONTRACT else 'none'
    data.uns[history.KEY] = dict(contract=contract,method='PCA',representation='X_pca',embedding_dim=2,
        embedding_dtype='float32',parameters=history._parameters(2,7,preprocessing),publication_path='/old/manifest.json',
        source_path='/old/counts.h5ad',source_sha256='a'*64,cell_order_sha256=common._cell_order_digest(('a','b','c')),
        feature_order_sha256='b'*64,software_versions={'scanpy':'historical'})
    checked = history.descriptor(data)
    assert checked['parameters']['normalization'] == ('tfidf_l2' if preprocessing == 'tfidf_l2' else 'none')
    assert checked['software_versions'] == {'scanpy':'historical'}
    data.uns[history.KEY]['parameters']['zero_center'] = False
    with pytest.raises(ValueError): history.descriptor(data)


def test_retired_pca_has_read_only_registry_metadata_and_no_worker_or_public_writer():
    registry = build_default_tool_registry()
    assert not registry.contains('pca_embed_cells')
    with pytest.raises(LookupError): registry.get('pca_embed_cells')
    old = registry.accepted('pca_embed_cells')
    assert old.result_contract.name == 'PCAEmbeddingResult' and old.planning is None
    assert not hasattr(old, 'function') and not hasattr(history, 'pca_embed_cells')
    assert not hasattr(pca_publication, 'publish')
    from agent.tools._native_worker import dispatch
    with pytest.raises(ValueError, match='Unregistered'): dispatch('pca', {})
    for tool, fn in [('run_scanpy','pp.pca'),('run_episcanpy','tl.pca')]:
        assert fn in dict(registry.get(tool).native_functions.functions)
    from agent.tools.scientific_authority import issue
    with pytest.raises(ValueError, match='Retired PCA'): issue(None,'pca_embed_cells',{}, {}, 'a'*64)


def test_unaccepted_historical_pca_cannot_acquire_authority(tmp_path,monkeypatch):
    # Even an old well-shaped manifest cannot qualify without its original
    # accepted authority. No original source, numerical fit or current library
    # versions are consulted by this read-only boundary.
    monkeypatch.setattr(pca_publication,'load',lambda *a:(dict(result={}), 'a'*64))
    with pytest.raises(ValueError,match='existing accepted execution'):
        pca_publication.verify_publication(tmp_path/'old.json')
