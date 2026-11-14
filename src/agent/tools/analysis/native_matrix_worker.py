"""Isolated, repository-controlled Python package dispatch; no user code."""
from collections.abc import Mapping
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import traceback
import time
from datetime import datetime, timezone


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
    if isinstance(result, Mapping):
        if any(not isinstance(k, str) for k in result):
            raise TypeError('Native return mapping requires string keys for faithful HDF5 publication')
        return {k: returned_data(v) for k, v in result.items()}
    if isinstance(result, (list, tuple)):
        return {str(i): returned_data(v) for i, v in enumerate(result)}
    if result is None or isinstance(result, (str, bool, int, float, np.generic)):
        return result
    # SnapATAC2 diff_test returns a Polars table, not an AnnData matrix.
    if type(result).__module__.startswith('polars.') and type(result).__name__ == 'DataFrame':
        return result.to_pandas()
    raise TypeError('Unrepresented native return type: ' + str(type(result)))


def _receipt_default(value):
    # Installed signatures may use immutable mappings (Scanpy metric_kwds).
    # Preserve their exact contents as JSON objects, without stringifying values.
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError('Unsupported native receipt value: ' + type(value).__name__)


def main(request_path):
    started = time.perf_counter()
    raw = Path(request_path).read_bytes(); req = json.loads(raw)
    stage = Path(req['stage'])
    sys.path.insert(0, str(Path(__file__).parent))
    from native_clustering import RequiredClusterCount
    try:
        import anndata as ad
        import numpy as np
        a = ad.read_h5ad(stage / 'input.h5ad')
        loaded = time.perf_counter()
        package = req['package']; name = req['function_name']; kwargs = req['kwargs']
        version = importlib.metadata.version(package)
        if version != req['version']:
            raise ValueError('Installed package version differs from qualified dispatch')
        # Shared installed callable resolution. Registry method offers and this
        # AI publication's specialized scientific slots remain separate owners.
        from native_callable_worker import functions
        from native_clustering import invoke, valid_graphs, graph_record
        from cell_result import grouping_key
        function = functions(package)[name]
        activity = package == 'episcanpy' and name == 'tl.geneactivity'
        call_kwargs = kwargs
        activity_source = None
        if activity:
            from native_gene_activity import source_metadata
            implementation = Path(inspect.getsourcefile(function))
            activity_implementation = dict(path=str(implementation.resolve()),
                sha256=hashlib.sha256(implementation.read_bytes()).hexdigest(), size_bytes=implementation.stat().st_size)
            activity_source = source_metadata(stage / 'input.h5ad', req['activity_inputs'],
                slot='raw.X' if kwargs.get('raw') else req['activity_inputs'].get('count_slot', 'X'))
            if kwargs.get('raw'):
                activity_source['raw_current_features_equal'] = (
                    a.raw is not None and a.raw.var_names.tolist() == a.var_names.tolist())
            call_kwargs = dict(kwargs, gtf_file=req['gene_anno_path'])
        graphs = valid_graphs(a, req.get('graph_inputs', []))
        # A positional marker follows native subsetting, including duplicate labels.
        marker = None
        if name == 'pp.highly_variable_genes' and kwargs.get('subset'):
            marker = '_agent_source_position'
            while marker in a.var: marker += '_'
            a.var[marker] = np.arange(a.n_vars)
        obs_before, var_before = set(a.obs), set(a.var)
        # SnapATAC2 2.9's KD-tree FFI accepts f64, while EpiZoo publishes
        # f32. Widen only the selected local view for the native call. Every
        # f32 value is exact in f64; preserve the published representation.
        restored_rep = None
        adaptation = None
        if package == 'snapatac2' and name == 'pp.knn' and kwargs.get('method', 'kdtree') == 'kdtree':
            key = kwargs.get('use_rep', 'X_spectral')
            values = a.obsm[key]
            if values.dtype == np.dtype('float32'):
                restored_rep = (key, values)
                a.obsm[key] = values.astype(np.float64)
                adaptation = dict(slot='obsm.' + key, source_dtype='float32', execution_dtype='float64',
                    conversion='lossless widening for native KD-tree FFI; original representation restored')
        try:
            invoking = time.perf_counter()
            native_started_at = datetime.now(timezone.utc).isoformat()
            result, selection = invoke(function, a, package, name, call_kwargs,
                                       req.get('parameter_selection'), graphs)
            invoked = time.perf_counter()
            native_finished_at = datetime.now(timezone.utc).isoformat()
        finally:
            if restored_rep is not None:
                a.obsm[restored_rep[0]] = restored_rep[1]
        native_type = type(result).__module__ + '.' + type(result).__name__
        independent = result is not None and not isinstance(result, ad.AnnData)
        if isinstance(result, ad.AnnData):
            a = result
        graphs = valid_graphs(a, graphs)
        graph = graph_record(a, package, name, kwargs)
        if graph is not None:
            graph['source_sha256'] = req['source_sha256']
            graph['visualization'] = graph['representation'] in req.get('visualization_representations', [])
            graphs = [g for g in graphs if g['graph'] != graph['graph']] + [graph]
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
                slots += ['obs.' + grouping_key(package, name, kwargs)]
                if package != 'snapatac2' and name in ('tl.leiden', 'tl.louvain'): slots += ['uns.' + name.split('.')[1]]
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
        if activity:
            slots = ['X'] + ['var.' + key for key in a.var]
            measurement = 'episcanpy_gene_domain_sum'
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
        writing = time.perf_counter()
        a.write_h5ad(stage / 'result.h5ad')
        written = time.perf_counter()
        versions = {p: importlib.metadata.version(p) for p in (package, 'anndata', 'numpy', 'scipy', 'scikit-learn')}
        if package == 'episcanpy': versions['scanpy'] = importlib.metadata.version('scanpy')
        source = Path(inspect.getsourcefile(function))
        receipt = dict(completion='succeeded', package=package, function_name=name,
            native_started_at=native_started_at, native_finished_at=native_finished_at,
            timing_seconds=dict(import_and_input_loading=loaded-started,
                native_preparation=invoking-loaded, native_call_including_asw=invoked-invoking,
                return_preparation=writing-invoked, h5ad_writing=written-writing),
            request_sha256=hashlib.sha256(raw).hexdigest(), versions=versions,
            native_signature=str(inspect.signature(function)), native_output_type=native_type, result_slots=slots,
            function_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            measurement=measurement, layer_semantics=layer_semantics)
        effective = {k: p.default for k, p in inspect.signature(function).parameters.items()
            if p.default is not inspect.Parameter.empty}
        effective.update(kwargs)
        receipt['effective_parameters'] = effective
        if activity:
            from native_gene_activity import axis_identity
            receipt['gene_activity_axes'] = axis_identity(stage / 'result.h5ad')
            receipt['gene_activity_source'] = activity_source
            receipt['inherited_slots'] = activity_source['slots']
            receipt['implementation'] = [activity_implementation]
        if selection is not None:
            receipt['parameter_selection'] = selection
        if graphs:
            receipt['graph_inputs'] = graphs
        if adaptation is not None:
            receipt['input_adaptation'] = adaptation
        receipt['result_shapes'] = {s: list(getattr(a, s.partition('.')[0])[s.partition('.')[2]].shape)
            for s in slots if s.startswith(('obsm.', 'obsp.'))}
        if axis_positions is not None: receipt['source_axis_positions'] = axis_positions
        (stage / 'native-receipt.json').write_text(json.dumps(receipt, indent=2, default=_receipt_default))
    except Exception as exc:
        error = dict(message=str(exc), type=type(exc).__name__)
        if isinstance(exc, RequiredClusterCount):
            error.update(parameter_selection=exc.evidence, request_sha256=hashlib.sha256(raw).hexdigest())
        (stage / 'native-error.json').write_text(json.dumps(error))
        traceback.print_exc()
        return 1
    return 0


def parameter_metadata(package, name):
    """Signature/default introspection only; never loads researcher data."""
    import importlib
    fn = importlib.import_module(package)
    for component in name.split('.'):
        fn = getattr(fn, component)
    sys.path.insert(0, str(Path(__file__).parent))
    from native_callable_worker import signature_metadata
    return signature_metadata(fn)


if __name__ == '__main__':
    if sys.argv[1] == '--parameters':
        print(json.dumps(parameter_metadata(sys.argv[2], sys.argv[3])))
    else:
        sys.exit(main(sys.argv[1]))
