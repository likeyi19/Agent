"""Matrix authority binding and bounded mechanical storage (no overlap science)."""
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import sqlite3
import time

import h5py
import numpy as np

from agent.tools._cancellation import cancellation_checkpoint
from . import scatac_matrix_contract as m
from ._barcode_qc_contract import MAX_BARCODES

CHUNK = 4096


@dataclass(frozen=True)
class MatrixLimits:
    """Operational ceilings, never scientific truncation or recovery defaults."""
    max_features: int = 2_000_000
    max_selected: int = 10_000_000
    max_scratch_bytes: int = 100 * 1024**3
    max_seconds: int = 24 * 3600

    def __post_init__(self):
        for value in (self.max_features, self.max_selected, self.max_scratch_bytes, self.max_seconds):
            m.integer(value, 1)
        if self.max_features > 2_000_000 or self.max_selected > MAX_BARCODES:
            m.fail('MATRIX_RESOURCE_LIMIT')


class Budget:
    def __init__(self, root, limits):
        self.root, self.limits = Path(root), limits
        self.started = time.monotonic()
        self.peak_bytes = 0

    def check(self):
        cancellation_checkpoint()
        size = sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())
        self.peak_bytes = max(size, self.peak_bytes)
        if size > self.limits.max_scratch_bytes:
            m.fail('MATRIX_SCRATCH_LIMIT')
        if time.monotonic() - self.started > self.limits.max_seconds:
            m.fail('MATRIX_TIME_LIMIT')


@contextmanager
def database(path, budget):
    db = sqlite3.connect(path)
    try:
        db.execute('PRAGMA journal_mode=OFF')  # disposable private scratch only
        db.execute('PRAGMA cache_size=-8192')
        db.execute('PRAGMA temp_store=FILE')
        db.execute('PRAGMA mmap_size=0')
        db.execute(f'PRAGMA max_page_count={max(1, budget.limits.max_scratch_bytes // 8192)}')
        db.execute('CREATE TABLE cells (row INTEGER PRIMARY KEY, ns TEXT, bc TEXT, rendered TEXT, seen INTEGER DEFAULT 0, UNIQUE(ns,bc))')
        db.execute('CREATE TABLE features (col INTEGER PRIMARY KEY, rank INTEGER, chrom TEXT, start INTEGER, end INTEGER, name TEXT UNIQUE)')
        # Indexes are built while inserting, never by a population-scale sort.
        db.execute('CREATE INDEX feature_order ON features(rank,start,end,col)')
        yield db
    except sqlite3.Error as exc:
        raise m.ScATACMatrixError('MATRIX_STORAGE_FAILED') from exc
    finally:
        db.close()










def append(dataset, values):
    start = len(dataset); dataset.resize((start + len(values),)); dataset[start:] = values


def write_axis(file, name, columns, strings, cursor, length, budget):
    group = file.create_group(name)
    group.attrs.update({'encoding-type': 'dataframe', 'encoding-version': '0.2.0', '_index': '_index',
                        'column-order': np.array(columns[1:], dtype=h5py.string_dtype())})
    for key in columns:
        dtype = h5py.string_dtype() if key in strings else np.dtype('int64')
        ds = group.create_dataset(key, shape=(length,), dtype=dtype)
        ds.attrs.update({'encoding-type': 'string-array' if key in strings else 'array', 'encoding-version': '0.2.0'})
    offset = 0
    while rows := cursor.fetchmany(CHUNK):
        for i, key in enumerate(columns): group[key][offset:offset+len(rows)] = [r[i] for r in rows]
        offset += len(rows); budget.check()
    if offset != length: m.fail('MATRIX_AXIS_INVALID')




def csr_rows(f, n, p, budget):
    """Strict physical reader; at most one full feature-width sparse row."""
    x = f['X']
    dims = np.asarray(x.attrs.get('shape', []))
    if (x.attrs.get('encoding-type') != 'csr_matrix' or x.attrs.get('encoding-version') != '0.1.0'
            or dims.dtype.kind not in 'iu' or dims.shape != (2,) or list(dims) != [n,p]
            or set(x) != {'data','indices','indptr'}):
        m.fail('MATRIX_CSR_INVALID')
    data, indices, offsets = (x[k] for k in ('data','indices','indptr'))
    if (any(not isinstance(a,h5py.Dataset) or a.ndim != 1 for a in (data,indices,offsets))
            or data.dtype.kind != 'i' or data.dtype.itemsize != 8
            or indices.dtype.kind not in 'iu' or offsets.dtype.kind not in 'iu'
            or len(offsets) != n+1 or len(data) != len(indices) or int(offsets[0]) != 0
            or int(offsets[-1]) != len(data)):
        m.fail('MATRIX_CSR_INVALID')
    left = 0
    for row in range(n):
        right = int(offsets[row+1])
        if not left <= right <= len(data) or right-left > p: m.fail('MATRIX_CSR_INVALID')
        yield indices[left:right].astype(object).tolist(), data[left:right].astype(object).tolist()
        left = right
        if row % CHUNK == 0: budget.check()
