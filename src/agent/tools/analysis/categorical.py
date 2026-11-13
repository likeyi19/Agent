"""Read-only categorical observations and descriptive counts, never inference.

Only selected observations and qualified 2D coordinates are decoded; X and graphs
are never read. The application binds accepted identities and source lineage.
"""
from collections import Counter
import json
import math

import h5py
import numpy as np
import pandas as pd
from anndata.io import read_elem

READ_VERSION = 'categorical-observations.v1'


def ordered_ids(index, *, source):
    # Names are labels on a positional axis, not necessarily unique join keys.
    return tuple(str(value) for value in index)


def require_same_cells(left, right, *, left_source=None, right_source=None):
    if left_source is None or left_source != right_source:
        if not left or not right or any(not value for value in (*left, *right)):
            raise ValueError('Cross-artifact association requires nonempty cell identifiers.')
        if len(set(left)) != len(left) or len(set(right)) != len(right):
            raise ValueError('Cross-artifact association requires unique cell identifiers; repeated identifiers require the same pinned artifact for positional association.')
    if len(left) != len(right):
        raise ValueError('Scientific artifacts must contain identical cell counts.')
    if left != right:
        raise ValueError('Scientific entities must have identical cell identities and exact order.')


def coordinates(path, sha, field='X_umap'):
    """Read only a pinned two-dimensional UMAP and its exact ordered obs axis."""
    from agent.tools._file_identity import checked_file
    with h5py.File(checked_file(path, sha), 'r') as handle:
        from .peak_h5ad import inspect
        inspect(handle, require_matrix=False, observation_fields=())
        obs = handle['obs']
        inspect(handle, require_matrix=False, observation_fields=(obs.attrs['_index'],))
        ids = ordered_ids(_series(obs[obs.attrs['_index']]), source='Coordinates')
        if (not isinstance(handle.get('obsm', getlink=True), h5py.HardLink)
                or not isinstance(handle['obsm'], h5py.Group)
                or not isinstance(field, str) or '/' in field or field not in handle['obsm']):
            raise ValueError('Select an exact coordinate slot.')
        # Validate the selected storage before decoding, not unrelated X.
        item = handle['obsm'][field]
        if not isinstance(handle['obsm'].get(field, getlink=True), h5py.HardLink):
            raise ValueError('Indirect coordinate storage is unsupported.')
        if not isinstance(item, h5py.Dataset) or item.shape != (len(ids), 2):
            raise ValueError('Selected coordinates must be a two-dimensional cell layout.')
        if item.is_virtual or item.external or item.dtype.kind not in 'fi':
            raise ValueError('Unsupported coordinate storage.')
        import math
        if item.chunks and math.prod(item.chunks) * item.dtype.itemsize > 16 * 1024**2:
            raise ValueError('Coordinate storage chunk exceeds the read bound.')
        creation = item.id.get_create_plist()
        if any(creation.get_filter(i)[0] not in (1, 2, 3, 32001, 32015) for i in range(creation.get_nfilters())):
            raise ValueError('Unsupported coordinate filter.')
        values = np.asarray(read_elem(item))
        if values.shape != (len(ids), 2) or values.dtype.kind not in 'fi' or not np.isfinite(values).all():
            raise ValueError('Selected coordinates must be a finite two-dimensional cell layout.')
        return ids, values


def label_token(value, *, source, allow_boolean=False):
    """Typed identities keep text, integer and float category labels distinct."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        if not allow_boolean:
            raise ValueError(f'{source} contains unsupported boolean label values.')
        payload, kind = value, 'boolean'
    elif isinstance(value, str):
        if not value.strip():
            raise ValueError(f'{source} contains blank-string label values.')
        payload, kind = value, 'string'
    elif isinstance(value, int):
        payload, kind = str(value), 'integer'
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f'{source} contains non-finite label values.')
        payload, kind = value.hex(), 'float'
    else:
        raise ValueError(f'{source} contains unsupported or unhashable label value type {type(value).__name__}.')
    return json.dumps(dict(type=kind, value=payload), ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)


def _labels(series):
    # Uncategorised numerical columns are measurements, not inferred categories.
    categorical = isinstance(series.dtype, pd.CategoricalDtype)
    if not categorical and not (pd.api.types.is_string_dtype(series.dtype) or series.dtype == object):
        raise ValueError('Select a categorical or text observation field.')
    missing = series.isna().tolist()
    values = series.tolist()
    tokens = [None if absent else label_token(value, source='Observation', allow_boolean=True)
              for value, absent in zip(values, missing)]
    categories = list(series.cat.categories) if categorical else list(dict.fromkeys(
        value for value, absent in zip(values, missing) if not absent))
    labels = {label_token(value, source='Category', allow_boolean=True): value.item() if isinstance(value, np.generic) else value
              for value in categories}
    return tokens, labels, categorical and bool(series.cat.ordered)


def _page(rows, offset, total):
    return dict(records=rows, total_records=total, offset=offset, returned_records=len(rows),
                omitted_records=total-len(rows), complete=offset == 0 and len(rows) == total,
                order='declared category order, otherwise first observed order; not a ranking')


def counts(series, *, offset=0, limit=8):
    tokens, labels, ordered = _labels(series)
    observed = Counter(tokens)
    n = len(tokens); labeled = n - observed[None]
    from itertools import islice
    rows = [dict(category=label, count=observed[token],
                 proportion_all=observed[token]/n if n else None,
                 proportion_labeled=observed[token]/labeled if labeled else None)
            for token, label in islice(labels.items(), offset, offset+limit)]
    return dict(read_version=READ_VERSION, statistic='categorical_counts', n_entities=n,
                n_labeled=labeled, n_missing=observed[None], n_categories=len(labels),
                ordered_categories=ordered, denominator_all=n, denominator_labeled=labeled,
                **_page(rows, offset, len(labels)))


def association(left_ids, left, right_ids, right, *, offset=0, limit=8, left_source=None, right_source=None):
    require_same_cells(left_ids, right_ids, left_source=left_source, right_source=right_source)
    if len(left) != len(left_ids) or len(right) != len(right_ids):
        raise ValueError('Categorical vectors must match the exact cell axis.')
    a, al, _ = _labels(left); b, bl, _ = _labels(right)
    pairs = Counter((x, y) for x, y in zip(a, b) if x is not None and y is not None)
    am, bm = Counter(), Counter()
    for (x, y), count in pairs.items():
        am[x] += count; bm[y] += count
    n = len(a); paired = sum(pairs.values())
    # Sparse contingency records avoid a potentially enormous category product.
    ai, bi = {k:i for i,k in enumerate(al)}, {k:i for i,k in enumerate(bl)}
    selected = sorted(pairs.items(), key=lambda p:(ai[p[0][0]],bi[p[0][1]]))[offset:offset+limit]
    rows = [dict(left=al[x], right=bl[y], count=count, row_total=am[x], column_total=bm[y],
                 proportion_paired=count/paired, proportion_row=count/am[x], proportion_column=count/bm[y])
            for (x,y),count in selected]
    return dict(read_version=READ_VERSION, statistic='categorical_contingency', n_entities=n,
                n_paired=paired, n_left_missing=a.count(None), n_right_missing=b.count(None),
                n_both_missing=sum(x is None and y is None for x,y in zip(a,b)),
                denominator=paired, zero_pairs='Unlisted pairs have zero count only when complete=true.',
                left_categories=counts(left, offset=offset, limit=limit),
                right_categories=counts(right, offset=offset, limit=limit),
                interpretation='Descriptive association only; source labels are not independent biological validation.',
                **_page(rows, offset, len(pairs)))


def fields(path):
    """Inspect obs encodings only; numerical arrays are not offered as categories."""
    with h5py.File(path, 'r') as handle:
        from .peak_h5ad import inspect
        inspect(handle, require_matrix=False, observation_fields=())
        obs = handle['obs']
        if any(not isinstance(obs.get(key, getlink=True), h5py.HardLink) for key in obs.attrs['column-order']):
            raise ValueError('Indirect observation fields are unsupported.')
        return [str(key) for key in obs.attrs['column-order']
                if (obs[key].attrs.get('encoding-type') in ('categorical', 'string-array', 'nullable-string-array')
                    or 'categories' in obs[key].attrs
                    or isinstance(obs[key], h5py.Dataset) and h5py.check_string_dtype(obs[key].dtype) is not None)]


def _series(item):
    # AnnData owns the older dataframe 0.1 categorical encoding too. Use its
    # selected-column reader, not a second implementation or full obs decoding.
    if 'categories' in item.attrs or item.parent.attrs.get('encoding-version') == '0.1.0':
        from anndata._io.specs.methods import read_series
        return read_series(item)
    return read_elem(item)


def selected_observations(handle, fields, *, max_logical_bytes=128 * 1024**2):
    """AnnData selected-column decoding after bounded storage validation.

    Legacy category levels are a separate referenced dataset. Validate that
    dependency before AnnData follows it; never read X or unrelated obs fields.
    """
    from .peak_h5ad import inspect
    inspect(handle, require_matrix=False, observation_fields=())
    obs = handle['obs']; index = obs.attrs['_index']
    requested = set(fields)
    fields = tuple(dict.fromkeys((index, *fields)))
    allowed = {index, *obs.attrs['column-order']}
    if any(field not in allowed or '/' in field for field in fields):
        raise ValueError('Selected observation column is absent or not an exact column name')
    dependencies = list(fields)
    for field in fields:
        if not isinstance(obs.get(field, getlink=True), h5py.HardLink):
            raise ValueError('Indirect observation storage is unsupported')
        item = obs[field]
        if 'categories' in item.attrs:
            try:
                target = obs[item.attrs['categories']]
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError('Invalid legacy categorical reference') from exc
            # Inspect every path component, including a referenced dataset's parents.
            parent = handle
            for part in target.name.strip('/').split('/'):
                if not isinstance(parent.get(part, getlink=True), h5py.HardLink):
                    raise ValueError('Indirect categorical levels are unsupported')
                parent = parent[part]
            if not isinstance(target, h5py.Dataset) or target.ndim != 1:
                raise ValueError('Categorical levels require a vector dataset')
            dependencies.append(target.name)
    inspect(handle, require_matrix=False, observation_fields=tuple(dict.fromkeys(dependencies)), max_logical_bytes=max_logical_bytes)
    values = {field: _series(obs[field]) for field in fields}
    axis = values.pop(index)
    if (getattr(axis, 'ndim', None) != 1 or any(
            getattr(value, 'ndim', None) != 1 or len(value) != len(axis) for value in values.values())):
        raise ValueError('Selected H5AD observations require one value per cell; scalar broadcasting is forbidden')
    # Explicit index-column selection has the same original values, not a new axis.
    if index in requested:
        values[index] = axis
    return pd.DataFrame(values, index=axis)


def observations(path, sha, field):
    from agent.tools._file_identity import checked_file
    path = checked_file(path, sha)
    with h5py.File(path, 'r') as handle:
        frame = selected_observations(handle, (field,))
        ids = ordered_ids(frame.index, source='Accepted observation axis')
        values = frame[field].reset_index(drop=True)
    if len(values) != len(ids):
        raise ValueError('Observation field and entity axis differ.')
    _labels(values)
    return ids, values


def axis(path, sha):
    from agent.tools._file_identity import checked_file
    with h5py.File(checked_file(path, sha), 'r') as handle:
        from .peak_h5ad import inspect
        inspect(handle, require_matrix=False, observation_fields=())
        obs = handle['obs']
        inspect(handle, require_matrix=False, observation_fields=(obs.attrs['_index'],))
        return ordered_ids(_series(obs[obs.attrs['_index']]), source='Accepted observation axis')
