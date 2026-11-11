"""One bounded ASW scan inside the existing native matrix worker.

Native packages own clustering; sklearn owns silhouette. Graph descriptions are
carried only from the invocation's verified input publication, never loose uns.
"""
from copy import deepcopy
import hashlib
import inspect
import json

ELIGIBLE = {
    ('snapatac2', 'tl.leiden'): 'resolution',
    ('scanpy', 'tl.leiden'): 'resolution',
    ('episcanpy', 'tl.leiden'): 'resolution',
    ('episcanpy', 'tl.louvain'): 'resolution',
    ('episcanpy', 'tl.kmeans'): 'num_clusters',
    ('episcanpy', 'tl.hc'): 'num_clusters',
}
SEED = 0
MAX_SAMPLE = 2000


def candidates(parameter, n_cells):
    return ([0.1, 0.25, 0.5, 1.0, 1.5, 2.0] if parameter == 'resolution'
            else [k for k in (2, 3, 4, 6, 8, 10) if k < n_cells])


def array_sha(value):
    """Hash exact numeric storage without densifying a sparse matrix."""
    import numpy as np
    from scipy import sparse
    h = hashlib.sha256()
    h.update(str(value.shape).encode())
    if sparse.issparse(value):
        h.update(value.format.encode())
        parts = (value.indptr, value.indices, value.data)
    else:
        parts = (np.asarray(value),)
    for part in parts:
        h.update(str(part.dtype).encode())
        # nditer avoids a second complete dense representation for strided views.
        for block in np.nditer(part, flags=['external_loop', 'buffered', 'zerosize_ok'],
                               op_flags=['readonly'], order='C', buffersize=65536):
            h.update(block.tobytes())
    return h.hexdigest()


class RequiredClusterCount(ValueError):
    """ASW completed without a recommendation and the native count is required."""
    def __init__(self, evidence):
        self.evidence = evidence
        super().__init__('ASW unavailable and ' + evidence['function'] + ' requires explicit num_clusters')


def cells_sha(a):
    return hashlib.sha256(json.dumps(a.obs_names.tolist(), ensure_ascii=False).encode()).hexdigest()


def representation(a, record):
    key = record['representation']
    values = a.X if key == 'X' else a.obsm[key]
    dims = record['dimensions']
    values = values[:, dims] if dims is not None else values
    if record.get('execution_dtype'):
        values = values.astype(record['execution_dtype'], copy=False)
    return values


def graph_record(a, package, name, kwargs):
    """Describe the representation actually used by the installed graph builder."""
    if package == 'snapatac2' and name == 'pp.knn':
        if kwargs.get('inplace', True) is False:
            return None
        key, dims = kwargs.get('use_rep', 'X_spectral'), kwargs.get('use_dims')
        if isinstance(dims, int):
            dims = list(range(a.obsm[key].shape[1]))[:dims]
        record = dict(graph='distances', representation=key, dimensions=dims,
                      metric='euclidean', metric_kwds={})
        if kwargs.get('method') == 'hora':
            record['execution_dtype'] = 'float32'
    elif package in ('scanpy', 'episcanpy') and name == 'pp.neighbors':
        from scanpy import settings
        key, n_pcs = kwargs.get('use_rep'), kwargs.get('n_pcs')
        # Mirrors the installed selector, after the native call has produced any
        # native implicit PCA. This never computes a representation a second time.
        if key is None:
            key = 'X' if n_pcs == 0 or a.n_vars <= settings.N_PCS else 'X_pca'
        dims = None if key == 'X' or n_pcs is None else list(range(a.obsm[key].shape[1]))[:n_pcs]
        neighbors = kwargs.get('key_added') or 'neighbors'
        record = dict(graph=a.uns[neighbors]['connectivities_key'], neighbors_key=neighbors,
                      representation=key, dimensions=dims, metric=kwargs.get('metric', 'euclidean'),
                      metric_kwds=dict(kwargs.get('metric_kwds', {})))
    else:
        return None
    record.update(function=package + '.' + name, cells_sha256=cells_sha(a),
                  graph_sha256=array_sha(a.obsp[record['graph']]),
                  representation_sha256=array_sha(representation(a, record)))
    return record


def valid_graphs(a, records):
    retained = []
    for record in records:
        try:
            valid = (record['cells_sha256'] == cells_sha(a)
                     and record['graph_sha256'] == array_sha(a.obsp[record['graph']])
                     and record['representation_sha256'] == array_sha(representation(a, record)))
        except KeyError:
            valid = False
        if valid:
            retained.append(record)
    return retained


def evaluation_input(a, package, name, kwargs, graphs):
    """No inferred representation, graph conversion, or visualization substitute."""
    if name in ('tl.kmeans', 'tl.hc'):
        record = dict(representation='X_pca', dimensions=None, metric='euclidean', metric_kwds={},
                      graph=None, cells_sha256=cells_sha(a))
        values = representation(a, record)  # Same fixed input as epiScanpy 0.4.0.
        record['representation_sha256'] = array_sha(values)
    else:
        if kwargs.get('restrict_to') is not None:
            return None, None, 'Restricted clustering is not adapted for ASW selection.'
        if name == 'tl.louvain' and kwargs.get('flavor', 'vtraag') != 'vtraag':
            return None, None, 'ASW resolution selection requires the native resolution-aware vtraag flavor.'
        if package == 'snapatac2':
            key = 'distances'
            matches = [r for r in graphs if r['graph'] == key and r['function'] == 'snapatac2.pp.knn']
        else:
            neighbors = kwargs.get('neighbors_key') or 'neighbors'
            key = kwargs.get('obsp') or a.uns.get(neighbors, {}).get('connectivities_key', 'connectivities')
            matches = [r for r in graphs if r['graph'] == key
                       and r['function'] in ('scanpy.pp.neighbors', 'episcanpy.pp.neighbors')]
        if len(matches) != 1:
            return None, None, 'No exact accepted graph/representation association is available.'
        record = deepcopy(matches[0])
        values = representation(a, record)
    if record['representation'] == 'X_umap' or record.get('visualization'):
        return None, record, 'Visualization coordinates are not an ASW clustering representation.'
    return values, record, None


def invoke(function, a, package, name, kwargs, policy, graphs):
    """Return the native result and evidence; retain winning labels without replay."""
    import anndata as ad
    import numpy as np
    import pandas as pd
    from scipy import sparse
    from sklearn.metrics import silhouette_score
    try:
        from .cell_result import grouping_key
    except ImportError:  # Isolated worker imports repository-controlled siblings.
        from cell_result import grouping_key
    parameter = ELIGIBLE.get((package, name))
    if parameter is None:
        return function(a, **kwargs), None
    default = inspect.signature(function).parameters[parameter].default
    evidence = dict(package=package, function=package + '.' + name, parameter=parameter,
                    origin='explicit' if parameter in kwargs else 'native_default',
                    selected_value=kwargs.get(parameter, None if default is inspect.Parameter.empty else default),
                    candidates=[], sample_count=0, evaluation_seed=SEED,
                    cells_sha256=cells_sha(a), criterion='sklearn.metrics.silhouette_score',
                    tie_rule='first candidate in ascending order', status='not_requested')
    if parameter in kwargs or policy != 'asw':
        result = function(a, **kwargs)
        _, record, reason = evaluation_input(a, package, name, kwargs, graphs)
        evidence['input'] = record
        if reason:
            evidence['input_unavailable_reason'] = reason
        return result, evidence
    values, record, unavailable = evaluation_input(a, package, name, kwargs, graphs)
    evidence['input'] = record
    if unavailable is None and record['metric'] == 'precomputed' and sparse.issparse(values):
        unavailable = 'Sparse precomputed input does not establish complete pairwise distances for ASW.'
    if unavailable is None:
        sample = np.sort(np.random.default_rng(SEED).choice(a.n_obs, min(MAX_SAMPLE, a.n_obs), replace=False))
        evidence.update(sample_count=len(sample), sample_positions_sha256=array_sha(sample))
        sampled = values[sample]
        if record['metric'] == 'precomputed':
            sampled = sampled[:, sample]
        numeric = sampled.data if sparse.issparse(sampled) else sampled
        if not np.isfinite(numeric).all() or sampled.shape[1] == 0:
            unavailable = 'ASW representation is empty or nonfinite.'
    key = grouping_key(package, name, kwargs)
    best = fallback = None
    before_obs = a.obs[key].copy() if key in a.obs else None
    uns_key = name.split('.')[1]
    before_uns = deepcopy(a.uns.get(uns_key))
    def restore():
        if before_obs is None:
            if key in a.obs:
                a.obs.pop(key)
        else:
            a.obs[key] = before_obs.copy()
        if before_uns is None:
            a.uns.pop(uns_key, None)
        else:
            a.uns[uns_key] = deepcopy(before_uns)
    if unavailable is None:
        for value in candidates(parameter, a.n_obs):
            restore()
            # Every native call has the same input population, graph and kwargs;
            # only this one candidate parameter changes. Native errors propagate.
            result = function(a, **dict(kwargs, **{parameter: value}))
            target = result if isinstance(result, ad.AnnData) else a
            independent = result is not None and not isinstance(result, ad.AnnData)
            labels = np.asarray(result if independent else target.obs[key])
            selected = labels[sample]
            count = len(np.unique(selected))
            score, reason = None, None
            if pd.isna(selected).any() or np.any(selected.astype(str) == '-1'):
                reason = 'Missing or unassigned native labels in the fixed evaluation sample.'
            elif not 2 <= count < len(sample):
                reason = 'ASW requires 2 <= number of labels < evaluation sample size.'
            else:
                try:
                    score = float(silhouette_score(sampled, selected, metric=record['metric'], **record['metric_kwds']))
                except ValueError as exc:
                    reason = 'Undefined ASW: ' + str(exc)
                if score is not None and not np.isfinite(score):
                    score, reason = None, 'Native representation produced a nonfinite silhouette.'
            evidence['candidates'].append(dict(value=value, asw=score, n_labels=count, undefined_reason=reason))
            # Keep only native-written grouping data (and native params), avoiding
            # a full count/graph copy for every candidate or replay of the winner.
            snapshot = (value, score, result.copy() if independent else target.obs[key].copy(),
                        deepcopy(target.uns.get(uns_key)), independent)
            if value == default:
                fallback = snapshot
            if score is not None and (best is None or score > best[1]):
                best = snapshot
    restore()
    winner = best or fallback
    if best is not None:
        evidence.update(origin='automatic', status='selected', selected_value=best[0])
    else:
        evidence.update(status='unavailable', reason=unavailable or 'No candidate has a defined ASW.')
        if default is inspect.Parameter.empty:
            if package == 'episcanpy' and name in ('tl.kmeans', 'tl.hc') and parameter == 'num_clusters':
                raise RequiredClusterCount(evidence)
            raise ValueError('ASW unavailable and ' + package + '.' + name + ' requires explicit ' + parameter)
    if winner is None:
        return function(a, **kwargs), evidence
    kwargs[parameter] = winner[0]
    if winner[4]:
        return winner[2], evidence
    target = a.copy() if kwargs.get('copy', False) else a
    target.obs[key] = winner[2]
    if winner[3] is not None:
        target.uns[uns_key] = winner[3]
    return (target if kwargs.get('copy', False) else None), evidence
