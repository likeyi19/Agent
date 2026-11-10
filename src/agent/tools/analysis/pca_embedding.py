"""Read-only descriptor for accepted historical centered-PCA artifacts."""
import anndata as ad
import numpy as np

from . import embedding_analysis as common

CONTRACT = 'cell-pca-embedding.v1'
TFIDF_CONTRACT = 'cell-pca-embedding.v2'
KEY = 'agent_cell_embedding'
REPRESENTATION = 'X_pca'


def _parameters(n, seed, preprocessing="none"):
    value = dict(n_components=int(n), solver='arpack', zero_center=True, whitening=False,
                normalization='none', scaling='none', feature_selection='all', matrix_slot='X',
                random_seed=seed, computation_dtype='float64', output_dtype='float32')
    if preprocessing == 'tfidf_l2':
        value.update(normalization='tfidf_l2', idf='log((1+n_cells)/(1+document_frequency))+1',
                     tf='counts', norm='l2', smooth_idf=True, sublinear_tf=False)
    elif preprocessing != 'none':
        raise ValueError('Unsupported PCA preprocessing.')
    return value


def descriptor(data):
    value = data.uns.get(KEY)
    if not isinstance(value, dict) or value.get('contract') not in (CONTRACT, TFIDF_CONTRACT):
        raise ValueError('Unsupported embedding producer contract.')
    if set(value) != {'contract','method','representation','embedding_dim','embedding_dtype','parameters',
                       'publication_path','source_path','source_sha256','cell_order_sha256','feature_order_sha256','software_versions'}:
        raise ValueError('Invalid embedding provenance.')
    k = value['embedding_dim']; params = value['parameters']
    preprocessing = 'none' if value['contract'] == CONTRACT else 'tfidf_l2'
    if (value['method'] != 'PCA' or value['representation'] != REPRESENTATION
            or not isinstance(k, (int, np.integer)) or k <= 0 or value['embedding_dtype'] != 'float32'
            or params != _parameters(k, params.get('random_seed'), preprocessing)):
        raise ValueError('Invalid PCA method or parameters.')
    common._validate_seed(int(params['random_seed']))
    values = np.asarray(data.obsm[REPRESENTATION])
    ids = common._cell_ids_from_adata(data)
    if (data.X is not None or data.n_vars != 0 or values.shape != (len(ids), k)
            or values.dtype != np.float32 or not common._all_finite(values)
            or common._cell_order_digest(ids) != value['cell_order_sha256']):
        raise ValueError('Invalid PCA representation or cell identity.')
    for field in ('source_sha256','cell_order_sha256','feature_order_sha256'):
        if not isinstance(value[field], str) or len(value[field]) != 64 or any(c not in '0123456789abcdef' for c in value[field]):
            raise ValueError('Invalid PCA source identity.')
    return dict(value, embedding_dim=int(k), parameters=_parameters(k, int(params['random_seed']), preprocessing))
