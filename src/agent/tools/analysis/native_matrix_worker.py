"""Isolated, repository-controlled Python package dispatch; no user code."""
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import sys
import traceback


def returned_data(result):
    """Only ordinary native data, written with AnnData's data-only HDF5 encodings."""
    import numpy as np
    import pandas as pd
    from scipy import sparse
    if isinstance(result, pd.DataFrame):
        intervals = {k: result[k] for k in result if isinstance(result[k].dtype, pd.CategoricalDtype)
                     and isinstance(result[k].cat.categories, pd.IntervalIndex)}
        if intervals:
            return dict(kind='dataframe_with_interval_categories', column_order=result.columns.to_numpy(),
                frame=result.drop(columns=list(intervals)), interval_columns={k: dict(
                    codes=v.cat.codes.to_numpy(), left=v.cat.categories.left.to_numpy(),
                    right=v.cat.categories.right.to_numpy(), closed=v.cat.categories.closed,
                    ordered=v.cat.ordered) for k,v in intervals.items()})
    if isinstance(result, pd.Index):
        return result.to_numpy()
    if isinstance(result, (pd.DataFrame, np.ndarray)) or sparse.issparse(result):
        return result
    if type(result) is dict:
        return {str(k): returned_data(v) for k, v in result.items()}
    if isinstance(result, (list, tuple)):
        return {str(i): returned_data(v) for i, v in enumerate(result)}
    if result is None or isinstance(result, (str, bool, int, float, np.generic)):
        return result
    # SnapATAC2 diff_test returns a Polars table, not an AnnData matrix.
    if type(result).__module__.startswith('polars.') and type(result).__name__ == 'DataFrame':
        return result.to_pandas()
    raise TypeError('Unrepresented native return type: ' + str(type(result)))


def main(request_path):
    raw = Path(request_path).read_bytes(); req = json.loads(raw)
    stage = Path(req['stage'])
    try:
        import anndata as ad
        import numpy as np
        a = ad.read_h5ad(stage / 'input.h5ad')
        package = req['package']; name = req['function_name']; kwargs = req['kwargs']
        version = importlib.metadata.version(package)
        if version != req['version']:
            raise ValueError('Installed package version differs from qualified dispatch')
        if package == 'episcanpy':
            import episcanpy as epi
            dispatch = {'pp.binarize': epi.pp.binarize, 'pp.tfidf': epi.pp.tfidf,
                'tl.pca': epi.tl.pca, 'pp.neighbors': epi.pp.neighbors, 'tl.dpt': epi.tl.dpt,
                'tl.tsne': epi.tl.tsne, 'tl.umap': epi.tl.umap,
                'pp.log1p': epi.pp.log1p, 'pp.lsi': epi.pp.lsi, 'pp.normalize_total': epi.pp.normalize_total,
                'tl.leiden': epi.tl.leiden, 'tl.louvain': epi.tl.louvain,
                'tl.kmeans': epi.tl.kmeans, 'tl.hc': epi.tl.hc, 'tl.rank_features': epi.tl.rank_features}
        elif package == 'scanpy':
            import scanpy as sc
            dispatch = {'tl.diffmap': sc.tl.diffmap, 'pp.scale': sc.pp.scale,
                'pp.calculate_qc_metrics': sc.pp.calculate_qc_metrics, 'pp.normalize_total': sc.pp.normalize_total,
                'pp.log1p': sc.pp.log1p, 'pp.highly_variable_genes': sc.pp.highly_variable_genes,
                'pp.pca': sc.pp.pca, 'pp.neighbors': sc.pp.neighbors, 'tl.leiden': sc.tl.leiden,
                'tl.umap': sc.tl.umap, 'tl.tsne': sc.tl.tsne, 'tl.paga': sc.tl.paga,
                'tl.rank_genes_groups': sc.tl.rank_genes_groups, 'tl.score_genes': sc.tl.score_genes, 'tl.dpt': sc.tl.dpt}
        elif package == 'snapatac2':
            import snapatac2 as snap
            dispatch = {'pp.select_features': snap.pp.select_features, 'tl.spectral': snap.tl.spectral,
                'tl.umap': snap.tl.umap, 'pp.knn': snap.pp.knn, 'tl.leiden': snap.tl.leiden,
                'tl.dbscan': snap.tl.dbscan,
                'pp.mnc_correct': snap.pp.mnc_correct, 'tl.marker_regions': snap.tl.marker_regions,
                'tl.diff_test': snap.tl.diff_test}
        else:
            raise ValueError('Unsupported package')
        function = dispatch[name]
        # A positional marker follows native subsetting, including duplicate labels.
        marker = None
        if name == 'pp.highly_variable_genes' and kwargs.get('subset'):
            marker = '_agent_source_position'
            while marker in a.var: marker += '_'
            a.var[marker] = np.arange(a.n_vars)
        obs_before, var_before = set(a.obs), set(a.var)
        result = function(a, **kwargs)
        native_type = type(result).__module__ + '.' + type(result).__name__
        independent = result is not None and not isinstance(result, ad.AnnData)
        if isinstance(result, ad.AnnData):
            a = result
        axis_positions = None
        if marker is not None:
            axis_positions = dict(cells=list(range(a.n_obs)), features=a.var.pop(marker).tolist())
            var_before.discard(marker)
        slots = []
        if independent:
            # Do not turn inplace=False into a mutation of the scientific slots.
            a.uns['agent_native_return'] = dict(native_type=native_type, value=returned_data(result))
            if name == 'pp.calculate_qc_metrics':
                a.uns['agent_native_return']['axes'] = {'0': 'ordered source cells', '1': 'ordered source features'}
            elif name == 'pp.normalize_total':
                a.uns['agent_native_return']['axes'] = {'X': 'ordered source cells by features', 'norm_factor': 'ordered source cells'}
            slots = ['uns.agent_native_return']
        else:
            slots = ['obs.' + k for k in a.obs if k not in obs_before]
            slots += ['var.' + k for k in a.var if k not in var_before]
            if name in ('tl.pca', 'pp.pca'): slots += ['obsm.X_pca', 'varm.PCs', 'uns.pca']
            elif name == 'pp.neighbors':
                key = kwargs.get('key_added') or 'neighbors'
                slots += ['uns.' + key, 'obsp.' + a.uns[key]['distances_key'], 'obsp.' + a.uns[key]['connectivities_key']]
            elif name == 'tl.diffmap': slots += ['obsm.X_diffmap', 'uns.diffmap_evals']
            elif name == 'tl.dpt': slots += ['obs.' + k for k in a.obs if k.startswith('dpt_')]
            elif name == 'tl.tsne': slots += ['obsm.X_tsne', 'uns.tsne']
            elif name == 'tl.umap':
                slots += ['obsm.X_' + kwargs.get('key_added', 'umap')] if package == 'snapatac2' else ['obsm.X_umap', 'uns.umap']
            elif name in ('tl.leiden', 'tl.louvain', 'tl.kmeans', 'tl.hc', 'tl.dbscan'):
                slots += ['obs.' + kwargs.get('key_added', name.split('.')[1])]
                if package != 'snapatac2' and name in ('tl.leiden', 'tl.louvain'): slots += ['uns.' + kwargs.get('key_added', name.split('.')[1])]
            elif name == 'tl.paga': slots += ['uns.paga']
            elif name in ('tl.rank_genes_groups', 'tl.rank_features'):
                slots += ['uns.' + (kwargs.get('key_added') or ('rank_features_groups' if name == 'tl.rank_features' else 'rank_genes_groups'))]
            elif name == 'tl.score_genes': slots += ['obs.' + kwargs.get('score_name', 'score')]
            elif name == 'pp.highly_variable_genes':
                slots += ['uns.hvg'] + ['var.' + k for k in a.var if k in ('highly_variable', 'means', 'dispersions', 'dispersions_norm', 'variances', 'variances_norm', 'highly_variable_rank', 'highly_variable_nbatches', 'highly_variable_intersection')]
            elif name == 'pp.tfidf': slots += ['layers.' + kwargs.get('layer_tfidf_key', 'tf-idf')]
            elif name == 'pp.lsi': slots += ['obsm.X_lsi']
            elif name == 'pp.select_features': slots += ['var.selected']
            elif name == 'tl.spectral': slots += ['obsm.X_spectral', 'uns.spectral_eigenvalue']
            elif name == 'pp.knn': slots += ['obsp.distances']
            elif name == 'pp.mnc_correct': slots += ['obsm.' + (kwargs.get('key_added') or kwargs.get('use_rep', 'X_spectral') + '_mnn')]
        layer_semantics = {}; measurement = None
        if not independent and name in ('pp.binarize', 'pp.log1p', 'pp.scale', 'pp.normalize_total'):
            changed = not (package == 'episcanpy' and name == 'pp.normalize_total' and not kwargs.get('inplace', True))
            if changed:
                semantics = {'pp.binarize': 'native_binarized_supplied_measurement', 'pp.log1p': 'log1p_transformed_measurement',
                    'pp.scale': 'scanpy_scaled_measurement', 'pp.normalize_total': package + '_library_size_normalized'}[name]
                target = 'layers.' + kwargs['layer'] if kwargs.get('layer') else 'obsm.' + kwargs['obsm'] if kwargs.get('obsm') else 'X'
                slots.append(target)
                if target == 'X': measurement = semantics
                elif kwargs.get('layer'): layer_semantics[kwargs['layer']] = semantics
                if name == 'pp.normalize_total':
                    layers = kwargs.get('layers') or []
                    layers = list(a.layers) if layers == 'all' else layers
                    slots += ['layers.' + k for k in layers]
                    layer_semantics.update({k: semantics for k in layers})
                    if kwargs.get('key_added'): slots += ['obs.' + kwargs['key_added']]
                if name == 'pp.log1p': slots += ['uns.log1p']
                if name == 'pp.scale': slots += ['var.' + k for k in ('mean', 'std') if k in a.var]
        if name == 'pp.tfidf': layer_semantics = {kwargs.get('layer_tfidf_key', 'tf-idf'): 'episcanpy_tfidf'}
        # Native variants may omit auxiliary slots; list only actual outputs.
        slots = sorted(set(s for s in slots if s == 'X' or s.partition('.')[2] in getattr(a, s.partition('.')[0])))
        a.write_h5ad(stage / 'result.h5ad')
        versions = {p: importlib.metadata.version(p) for p in (package, 'anndata', 'numpy', 'scipy', 'scikit-learn')}
        if package == 'episcanpy': versions['scanpy'] = importlib.metadata.version('scanpy')
        source = Path(inspect.getsourcefile(function))
        receipt = dict(completion='succeeded', package=package, function_name=name,
            request_sha256=hashlib.sha256(raw).hexdigest(), versions=versions,
            native_signature=str(inspect.signature(function)), native_output_type=native_type, result_slots=slots,
            function_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            measurement=measurement, layer_semantics=layer_semantics)
        effective = {k: p.default for k, p in inspect.signature(function).parameters.items()
            if p.default is not inspect.Parameter.empty}
        effective.update(kwargs)
        receipt['effective_parameters'] = effective
        receipt['result_shapes'] = {s: list(getattr(a, s.partition('.')[0])[s.partition('.')[2]].shape)
            for s in slots if s.startswith(('obsm.', 'obsp.'))}
        if axis_positions is not None: receipt['source_axis_positions'] = axis_positions
        (stage / 'native-receipt.json').write_text(json.dumps(receipt, indent=2))
    except Exception as exc:
        (stage / 'native-error.json').write_text(json.dumps(dict(message=str(exc), type=type(exc).__name__)))
        traceback.print_exc()
        return 1
    return 0


def parameter_metadata(package, name):
    """Signature/default introspection only; never loads researcher data."""
    import importlib
    fn = importlib.import_module(package)
    for component in name.split('.'):
        fn = getattr(fn, component)
    metadata = {}
    import typing
    try:
        annotations = typing.get_type_hints(fn)
    except (NameError, TypeError, AttributeError):
        annotations = {}
    for key, parameter in inspect.signature(fn).parameters.items():
        item = {}
        annotation = annotations.get(key, parameter.annotation)
        if isinstance(annotation, str):
            if annotation in ('bool', 'int', 'float', 'str'):
                item['types'] = [annotation]
            # Native forward annotations may be Literal strings. Resolve only
            # the installed package's own type annotations, never user code.
        else:
            if typing.get_origin(annotation) is typing.Literal:
                item['choices'] = list(typing.get_args(annotation))
            if annotation in (bool, int, float, str):
                item['types'] = [annotation.__name__]
        default = parameter.default
        if default is not inspect.Parameter.empty:
            try:
                serialized = json.loads(json.dumps(default, allow_nan=False))
            except (TypeError, ValueError):
                pass
            else:
                item['default_when_omitted'] = serialized
                if 'types' not in item and type(default) in (bool, int, float, str):
                    item['types'] = [type(default).__name__]
        metadata[key] = item
    return metadata


if __name__ == '__main__':
    if sys.argv[1] == '--parameters':
        print(json.dumps(parameter_metadata(sys.argv[2], sys.argv[3])))
    else:
        sys.exit(main(sys.argv[1]))
