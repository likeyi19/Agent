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
    values = tuple(str(value) for value in index)
    if not values or any(not value for value in values) or len(set(values)) != len(values):
        raise ValueError(f'{source} requires nonempty unique cell identifiers.')
    return values


def require_same_cells(left, right):
    if len(left) != len(right):
        raise ValueError('Scientific artifacts must contain identical cell counts.')
    if left != right:
        raise ValueError('Scientific entities must have identical cell identities and exact order.')


def coordinates(path, sha):
    """Read only a pinned two-dimensional UMAP and its unique ordered obs axis."""
    from agent.tools.analysis.marker_annotation import checked_file
    with h5py.File(checked_file(path, sha), 'r') as handle:
        from .peak_h5ad import inspect
        inspect(handle, observation_fields=())
        obs = handle['obs']
        inspect(handle, observation_fields=(obs.attrs['_index'],))
        ids = ordered_ids(_series(obs[obs.attrs['_index']]), source='Coordinates')
        values = np.asarray(read_elem(handle['obsm']['X_umap']))
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


def association(left_ids, left, right_ids, right, *, offset=0, limit=8):
    require_same_cells(left_ids, right_ids)
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
        inspect(handle, observation_fields=())
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
    if item.parent.attrs.get('encoding-version') == '0.1.0':
        from anndata._io.specs.methods import read_series
        return read_series(item)
    return read_elem(item)


def observations(path, sha, field):
    from .marker_annotation import checked_file
    path = checked_file(path, sha)
    with h5py.File(path, 'r') as handle:
        from .peak_h5ad import inspect
        inspect(handle, observation_fields=())
        obs = handle['obs']
        # Exact obs member lookup, never HDF5 path traversal supplied by a model.
        if field not in obs.attrs['column-order']:
            raise ValueError('Selected categorical field is not an observation column.')
        selected = [obs.attrs['_index'], field]
        # Legacy AnnData uses an HDF5 object reference for the category levels.
        # Check that referenced dataset with the same bounded storage owner.
        if 'categories' in obs[field].attrs:
            selected.append(obs[obs[field].attrs['categories']].name)
        inspect(handle, observation_fields=selected)
        ids = ordered_ids(_series(obs[obs.attrs['_index']]), source='Accepted observation axis')
        values = pd.Series(_series(obs[field]))
    if len(values) != len(ids):
        raise ValueError('Observation field and entity axis differ.')
    _labels(values)
    return ids, values


def axis(path, sha):
    from .marker_annotation import checked_file
    with h5py.File(checked_file(path, sha), 'r') as handle:
        from .peak_h5ad import inspect
        inspect(handle, observation_fields=())
        obs = handle['obs']
        inspect(handle, observation_fields=(obs.attrs['_index'],))
        return ordered_ids(_series(obs[obs.attrs['_index']]), source='Accepted observation axis')
