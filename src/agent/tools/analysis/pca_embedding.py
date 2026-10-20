"""Centered Scanpy PCA on the supplied X values, without normalization/filtering.

The compact publication carries fit state and exact input identity. Verification
checks the centered projection and eigenpair residual independently of PCA.
"""
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from . import embedding_analysis as common
from agent.tools.data._fragments_common import snapshots, unchanged

CONTRACT = 'cell-pca-embedding.v1'
KEY = 'agent_cell_embedding'
REPRESENTATION = 'X_pca'


def _source(path, *, require_variance=True):
    data = ad.read_h5ad(path, backed='r')
    try:
        ids = tuple(str(x) for x in data.obs_names)
        features = tuple(str(x) for x in data.var_names)
        if (min(data.shape) < 2 or len(set(ids)) != len(ids)
                or any(not x or '\n' in x or '\r' in x for x in ids)):
            raise ValueError('PCA requires at least two cells/features and unique nonempty cell identities.')
        matrix = data.X
        if hasattr(matrix, 'group'):
            group = matrix.group
            estimated = sum(group[k].size * group[k].dtype.itemsize for k in ('data','indices','indptr'))
            estimated += group['data'].size * 8
        else:
            estimated = data.n_obs * data.n_vars * (matrix.dtype.itemsize + 8)
        if estimated > 2 * 1024**3:
            raise ValueError('PCA input storage and numeric copy exceed the 2 GiB execution bound.')
        x = matrix.to_memory() if hasattr(matrix, 'to_memory') else matrix[:]
    finally:
        data.file.close()
    if sparse.issparse(x):
        x.check_format(full_check=True)
    values = x.data if sparse.issparse(x) else x
    if values.dtype.kind not in 'biuf' or not np.isfinite(values).all():
        raise ValueError('PCA requires finite real numeric X measurements.')
    # Only sparse storage or an already dense source is copied; never densify sparse X.
    x = x.astype(np.float64)
    if sparse.issparse(x):
        x.sum_duplicates()  # Canonical storage preserves the logical supplied measurements.
    if sparse.issparse(x):
        from sklearn.utils.sparsefuncs import mean_variance_axis
        means, variances = mean_variance_axis(x, axis=0)
    else:
        means, variances = x.mean(axis=0), x.var(axis=0)
    variance = float(np.sum(variances) * x.shape[0] / (x.shape[0]-1))
    if require_variance and (not np.isfinite(variance) or variance <= 0):
        raise ValueError('PCA requires positive finite total centered variance.')
    return x, ids, features, means, variance


def _parameters(n, seed):
    return dict(n_components=int(n), solver='arpack', zero_center=True, whitening=False,
                normalization='none', scaling='none', feature_selection='all', matrix_slot='X',
                random_seed=seed, computation_dtype='float64', output_dtype='float32')


def descriptor(data):
    value = data.uns.get(KEY)
    if not isinstance(value, dict) or value.get('contract') != CONTRACT:
        raise ValueError('Unsupported embedding producer contract.')
    if set(value) != {'contract','method','representation','embedding_dim','embedding_dtype','parameters',
                       'publication_path','source_path','source_sha256','cell_order_sha256','feature_order_sha256','software_versions'}:
        raise ValueError('Invalid embedding provenance.')
    k = value['embedding_dim']; params = value['parameters']
    if (value['method'] != 'PCA' or value['representation'] != REPRESENTATION
            or not isinstance(k, (int, np.integer)) or k <= 0 or value['embedding_dtype'] != 'float32'
            or params != _parameters(k, params.get('random_seed'))):
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
    return dict(value, embedding_dim=int(k), parameters=_parameters(k, int(params['random_seed'])))


def pca_embed_cells(input_path: str | Path, output_dir: str | Path, *,
                    n_components: int | None = None, random_seed: int = 0,
                    overwrite: bool = False):
    common._validate_overwrite(overwrite); common._validate_seed(random_seed)
    path = common._resolve_existing_file(input_path, 'input_path')
    root = common._resolve_output_dir(output_dir)
    output = root / (path.stem + '.pca.h5ad')
    sidecar = root / (path.stem + '.pca_obs_names.txt')
    manifest = root / (path.stem + '.pca.manifest.json')
    for p in (output, sidecar, manifest): common._ensure_output_available(p, overwrite=overwrite)
    before = snapshots([path]); sha = common._analysis_digest(path)
    x, ids, features, means, total = _source(path)
    if n_components is not None and (type(n_components) is not int or not 0 < n_components < min(x.shape)):
        raise ValueError('PCA n_components must be positive and below min(n_cells, n_features).')
    k = min(50, min(x.shape)-1) if n_components is None else n_components
    # Bound fit/workspace allocations separately from the sparse source storage.
    if 8 * (x.shape[0]+x.shape[1]) * max(2*k+1,20) > 2 * 1024**3:
        raise ValueError('PCA solver workspace estimate exceeds 2 GiB; use fewer components.')
    with common._preserve_loaded_torch_threads():
        import scanpy as sc
        scores, loadings, ratios, variances = sc.pp.pca(x, n_comps=k, zero_center=True,
            svd_solver='arpack', random_state=random_seed, return_info=True, dtype='float32')
    publication = dict(publication_path=str(manifest), contract=CONTRACT, method='PCA', representation=REPRESENTATION,
        embedding_dim=int(scores.shape[1]), embedding_dtype=str(scores.dtype),
        parameters=_parameters(scores.shape[1], random_seed), source_path=str(path), source_sha256=sha,
        cell_order_sha256=common._cell_order_digest(ids), feature_order_sha256=common._cell_order_digest(features),
        software_versions=common._software_versions())
    data = ad.AnnData(obs=pd.DataFrame(index=ids), obsm={REPRESENTATION:scores})
    data.uns[KEY] = publication
    data.uns['pca_fit'] = dict(loadings=loadings, mean=means, variance=variances,
                              variance_ratio=ratios, total_variance=total)
    unchanged(before)
    common._atomic_write_h5ad(data, output, overwrite=overwrite, validator=descriptor)
    # Ordered sidecar uses the same exact UTF-8 convention as legacy embeddings.
    import os, tempfile
    fd, name = tempfile.mkstemp(dir=root, suffix='.ids.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(''.join(v+'\n' for v in ids)); f.flush(); os.fsync(f.fileno())
        common._ensure_output_available(sidecar, overwrite=overwrite)
        os.replace(name, sidecar)
    finally:
        Path(name).unlink(missing_ok=True)
    unchanged(before)
    result = dict(status='success', input_path=str(path), embedding_path=str(output), cell_ids_path=str(sidecar),
        analysis_path=str(output), analysis_sha256=common._analysis_digest(output), n_cells=len(ids),
        n_features=len(features), embedding_dim=int(scores.shape[1]), embedding_dtype=str(scores.dtype),
        finite=True, cell_order_preserved=True, backend='Scanpy', method='PCA', representation=REPRESENTATION,
        parameters=publication['parameters'], software_versions=publication['software_versions'], source_sha256=sha)
    from .pca_publication import publish
    return publish(dict(input_path=path,output_dir=root,n_components=n_components,random_seed=random_seed,overwrite=overwrite),result,manifest)


def _verify_result(arguments, result):
    path = Path(result['analysis_path']); before = snapshots([path, result['input_path'], result['cell_ids_path']])
    if (common._analysis_digest(path) != result['analysis_sha256'] or
            Path(arguments['input_path']).resolve() != Path(result['input_path']) or
            result['embedding_path'] != str(path) or result['method'] != 'PCA'):
        raise ValueError('PCA publication identity differs from execution.')
    data = common._read_analysis(path); meta = descriptor(data)
    if (meta['source_path'] != str(Path(result['input_path']).resolve())
            or meta['publication_path'] != str(path.with_suffix('.manifest.json'))
            or meta['software_versions'] != common._software_versions()):
        raise ValueError('PCA source locator, publication or software provenance differs.')
    if common._analysis_digest(result['input_path']) != meta['source_sha256']:
        raise ValueError('PCA source bytes changed.')
    x, ids, features, mean, total = _source(result['input_path'])
    k = min(50,min(x.shape)-1) if arguments.get('n_components') is None else arguments['n_components']
    if (tuple(data.obs_names) != ids or common._load_cell_ids(Path(result['cell_ids_path'])) != ids
            or common._cell_order_digest(features) != meta['feature_order_sha256']
            or meta['parameters'] != _parameters(k, arguments.get('random_seed',0))):
        raise ValueError('PCA source axes or applied parameters differ.')
    fit = data.uns['pca_fit']; v = np.asarray(fit['loadings']); scores = np.asarray(data.obsm[REPRESENTATION])
    variances = np.asarray(fit['variance'])
    if v.shape != (k,len(features)) or not np.isfinite(v).all() or variances.shape != (k,):
        raise ValueError('Invalid PCA fit state.')
    if (not np.allclose(v@v.T, np.eye(k), atol=1e-7) or not np.allclose(fit['mean'],mean,atol=1e-10)
            or not np.allclose(fit['variance_ratio'],variances/total,rtol=1e-6)
            or not np.allclose(fit['total_variance'],total,rtol=1e-8)
            or not np.isfinite(variances).all() or (variances < -1e-10).any()
            or (np.diff(variances) > 1e-8).any()):
        raise ValueError('PCA variance or orthogonality is invalid.')
    projected = x @ v.T - mean @ v.T
    if not np.allclose(scores, projected, atol=2e-5, rtol=2e-5):
        raise ValueError('PCA scores do not equal the centered source projection.')
    covariance_product = (x.T @ projected - mean[:,None]*projected.sum(axis=0))/(len(ids)-1)
    residual = covariance_product - v.T*variances
    if np.linalg.norm(residual) > 1e-6 * max(1.,np.linalg.norm(variances)):
        raise ValueError('PCA fit fails the independent centered eigenpair residual.')
    if k < min(x.shape)-1:
        # An eigenpair residual alone could accept the wrong (lower) components.
        # Independently bound the largest omitted variance with SciPy's native
        # eigensolver on a sparse centered operator, never a dense covariance.
        from scipy.sparse.linalg import LinearOperator, eigsh
        def omitted_covariance(vector):
            orthogonal = vector - v.T @ (v @ vector)
            projected_vector = x @ orthogonal - mean @ orthogonal
            product = (x.T @ projected_vector - mean * projected_vector.sum())/(len(ids)-1)
            return product - v.T @ (v @ product)
        operator = LinearOperator((len(features),len(features)), matvec=omitted_covariance, dtype=np.float64)
        largest = eigsh(operator, k=1, which='LA', tol=1e-6, return_eigenvectors=False,
                        v0=np.random.default_rng(0).standard_normal(len(features)))[0]
        if largest > variances[-1] + 1e-6 * max(1.,variances[0]):
            raise ValueError('PCA fit omits a higher-variance principal component.')
    expected = dict(n_cells=len(ids), n_features=len(features), embedding_dim=k, embedding_dtype='float32',
                    parameters=meta['parameters'], source_sha256=meta['source_sha256'],
                    software_versions=meta['software_versions'], representation=REPRESENTATION,
                    finite=True, cell_order_preserved=True, backend='Scanpy', status='success')
    if any(result.get(key)!=value for key,value in expected.items()):
        raise ValueError('PCA result differs from its scientific publication.')
    unchanged(before)
    return meta


def observe_methods(path, resources=()):
    """Non-executable suitability observations for exact offered H5AD inputs.

    No inference, decomposition or workflow selection. Execution owners still
    validate the source and resources they actually consume.
    """
    before = snapshots([path])
    try:
        x, ids, features, _, total = _source(path, require_variance=False)
    except (ValueError, TypeError, OSError, KeyError):
        return dict(pca='invalid_or_unsupported', epizoo=[])
    matches = []
    from . import epizoo_embedding as wrapper
    native = wrapper.epizoo_backend
    for resource in resources:
        cfg = native._normalize_species(resource.species)
        if x.shape[1] != cfg.raw_dimension:
            continue
        try:
            data = ad.AnnData(x, obs=pd.DataFrame(index=ids), var=pd.DataFrame(index=features))
            native._validate_adata(data, cfg)
            pins = resource.inputs()['expected_resource_identity']
            wrapper._check_resource_files(wrapper._pinned_resource_paths(resource.checkpoint_path, resource.species), pins)
            reference = native._load_resources(cfg, native.DEFAULT_RESOURCES_DIR)
            native._validate_retained_feature_order(data, reference)
            if (np.asarray(x[:,reference.filter_indices].sum(axis=1)).ravel() <= 0).any():
                continue
            matches.append(dict(resource_id=resource.resource_id, model_species=resource.species))
        except (ValueError, TypeError, OSError):
            continue
    unchanged(before)
    return dict(pca='eligible' if np.isfinite(total) and total > 0 else 'invalid_or_unsupported', pca_measurements='supplied X; centered; no normalization or filtering',
                epizoo=matches, n_cells=len(ids), n_features=len(features))


def verify_result(arguments, result):
    from .pca_publication import verify_result as verify
    return verify(arguments,result)['result']
