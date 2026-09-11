"""Matrix authority binding and bounded mechanical storage (no overlap science)."""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from pathlib import Path
import sqlite3
import time

import h5py
import numpy as np

from agent.tools._cancellation import cancellation_checkpoint
from . import scatac_matrix_contract as m
from . import _matrix_bedtools as bed
from ._cell_selection_contract import SELECTED_HEADER, load_manifest as load_selection
from ._barcode_qc_contract import gzip_lines, MAX_BARCODES
from ._barcode_qc_contract import load_manifest as load_qc
from ._barcode_qc_binding import resource_qualification
from .scatac_qc_reference import load_scatac_qc_reference_bundle
from .cell_selection_verifier import verify_cell_selection
from .scatac_fragment_reader import open_verified_fragments
from .scatac_fragments_v2_verifier import FragmentVerificationRuntime, take_snapshots, check_snapshots
from .scatac_reference import load_scatac_reference_bundle, reinspect_scatac_reference_bundle_sources

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


@dataclass
class BoundMatrix:
    fragments: object
    selection: dict
    reference: object
    upstream: dict
    snapshots: tuple
    qc_reference: object
    qualification: dict

    def unchanged(self):
        check_snapshots(self.snapshots)
        self.fragments.verification.check_unchanged()
        if resource_qualification(self.qc_reference) != self.qualification:
            m.fail('MATRIX_LINEAGE_INVALID')


def bind(upstream, limits):
    """Selection verification includes fresh QC and producer-specific authority."""
    m.shape(upstream, ('fragments', 'selection', 'reference'))
    for p in upstream.values():
        m.absolute_path(p['manifest_path']); m.sha(p['manifest_sha256'])
    snapshots = take_snapshots([p['manifest_path'] for p in upstream.values()])
    sp, fp, rp = (upstream[k] for k in ('selection', 'fragments', 'reference'))
    _, reference, _ = load_scatac_reference_bundle(rp['manifest_path'], expected_sha256=rp['manifest_sha256'])
    if reference.ccre.feature_count > limits.max_features: m.fail('MATRIX_RESOURCE_LIMIT')
    resources = [reference.genome.fasta.path, reference.genome.fai.path, reference.ccre.bed.path,
                 str(Path(sp['manifest_path']).parent / 'selected.tsv.gz'),
                 str(Path(sp['manifest_path']).parent / 'decisions.tsv.gz')]
    snapshots = tuple(sorted(snapshots + take_snapshots(resources)))
    # Read hash-bound metadata to snapshot the selection/QC resource closure
    # before scientific verification; metadata alone never confers authority.
    declared = load_selection(sp['manifest_path'],sp['manifest_sha256']).to_dict()
    if declared['selected_count'] > limits.max_selected: m.fail('MATRIX_RESOURCE_LIMIT')
    qp = Path(declared['arguments']['barcode_qc_manifest_path'])
    snapshots += take_snapshots([qp])
    q = load_qc(qp,declared['arguments']['barcode_qc_manifest_sha256']).to_dict()
    qrp = q['arguments']['qc_reference_manifest_path']
    snapshots += take_snapshots([qrp])
    _, qr, _ = load_scatac_qc_reference_bundle(qrp,expected_sha256=q['arguments']['qc_reference_manifest_sha256'])
    resources = [qp.parent/q[k]['path'] for k in ('table','histogram')]
    resources += [r.path for r in (qr.parent_manifest,qr.annotation.resource,qr.tss,qr.lineage)]
    snapshots = tuple(sorted(set(snapshots + take_snapshots(resources))))
    qualification = resource_qualification(qr)
    selection = verify_cell_selection(sp['manifest_path'], expected_sha256=sp['manifest_sha256']).to_dict()
    args = selection['qc_lineage']['arguments']
    if (args['fragments_manifest_path'] != fp['manifest_path']
            or args['fragments_manifest_sha256'] != fp['manifest_sha256']):
        m.fail('MATRIX_LINEAGE_INVALID')
    if reference.ccre.feature_count > limits.max_features or selection['selected_count'] > limits.max_selected:
        m.fail('MATRIX_RESOURCE_LIMIT')
    reinspect_scatac_reference_bundle_sources(reference)
    fragments = open_verified_fragments(fp['manifest_path'], expected_sha256=fp['manifest_sha256'], runtime=FragmentVerificationRuntime())
    fr = fragments.manifest['reference']
    if (fr['manifest_path'] != rp['manifest_path'] or fr['manifest_sha256'] != rp['manifest_sha256']
            or fr['reference_identity_sha256'] != reference.reference_identity_sha256
            or selection['qc_lineage']['reference_identity_sha256'] != reference.reference_identity_sha256):
        m.fail('MATRIX_LINEAGE_INVALID')
    ids = (fragments.manifest['fragments_identity_sha256'], selection['identity_sha256'], reference.reference_identity_sha256)
    contracts = ('scatac-fragments.v2', 'scatac-cell-selection.v1', 'scatac-reference-bundle.v1')
    pointers = {k: dict(manifest_path=upstream[k]['manifest_path'], manifest_sha256=upstream[k]['manifest_sha256'],
                        identity_sha256=identity, contract_version=contract)
                for k, identity, contract in zip(('fragments', 'selection', 'reference'), ids, contracts)}
    result = BoundMatrix(fragments, selection, reference, pointers, snapshots, qr, qualification)
    result.unchanged()
    return result


def load_axes(db, bound, budget):
    sp = Path(bound.upstream['selection']['manifest_path']).parent / 'selected.tsv.gz'
    lines = gzip_lines(sp, MAX_BARCODES + 1)
    if next(lines, None) != SELECTED_HEADER:
        m.fail('MATRIX_ROW_IDENTITY_INVALID')
    digest = hashlib.sha256(b'agent.ordered-selected-cells.v1\0'); count = 0
    from .scatac_selection_profile import decode_cell_id
    for line in lines:
        fields = line.decode('ascii').rstrip('\n').split('\t')
        if len(fields) != 4:
            m.fail('MATRIX_ROW_IDENTITY_INVALID')
        index, ns, bc, rendered = fields
        if index != str(count) or decode_cell_id(rendered) != (ns, bc):
            m.fail('MATRIX_ROW_IDENTITY_INVALID')
        db.execute('INSERT INTO cells(row,ns,bc,rendered) VALUES(?,?,?,?)', (count, ns, bc, rendered))
        digest.update((ns+'\t'+bc+'\n').encode('ascii')); count += 1
        if count % CHUNK == 0: budget.check()
    if count != bound.selection['selected_count'] or digest.hexdigest() != bound.selection['ordered_selected_sha256']:
        m.fail('MATRIX_ROW_IDENTITY_INVALID')
    contigs = dict(bound.fragments.contigs); ranks = {c: i for i, c in enumerate(contigs)}
    # Also validates the qualified profile's contig/length domain.
    bed.genome_order_bytes(bound.fragments.contigs)
    digest = hashlib.sha256(); count = 0
    with open(bound.reference.ccre.bed.path, 'rb') as f:
        for line in iter(lambda: f.readline(65537), b''):
            if len(line) > 65536:
                m.fail('MATRIX_COLUMN_IDENTITY_INVALID')
            fields = line.decode('utf-8').rstrip('\n').split('\t')
            chrom, start, end = fields[:3]; start, end = int(start), int(end)
            m.interval(chrom, start, end)
            if chrom not in contigs or end > contigs[chrom]: m.fail('MATRIX_COLUMN_IDENTITY_INVALID')
            name = f'{chrom}:{start}-{end}'
            db.execute('INSERT INTO features VALUES(?,?,?,?,?,?)', (count, ranks[chrom], chrom, start, end, name))
            digest.update((name+'\n').encode()); count += 1
            if count % CHUNK == 0: budget.check()
    if count != bound.reference.ccre.feature_count or digest.hexdigest() != bound.reference.ccre.ordered_feature_sha256:
        m.fail('MATRIX_COLUMN_IDENTITY_INVALID')
    db.commit(); bound.unchanged(); budget.check()


def annotations(bound, logical):
    return dict(matrix_semantics='fragment_counts', matrix_profile_id=m.PROFILE.profile_id,
                matrix_profile_sha256=m.PROFILE_SHA256, species=bound.reference.species,
                assembly=bound.reference.target_assembly, coordinate_system='zero-based-half-open',
                ordered_selected_sha256=bound.selection['ordered_selected_sha256'],
                ordered_feature_sha256=bound.reference.ccre.ordered_feature_sha256, logical_matrix_sha256=logical)


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


def write_h5(path, db, bound, rows, budget):
    n, p = bound.selection['selected_count'], bound.reference.ccre.feature_count
    with h5py.File(path, 'x') as f:
        f.attrs.update({'encoding-type': 'anndata', 'encoding-version': '0.1.0'})
        for name in ('layers','obsm','obsp','varm','varp','uns'):
            f.create_group(name).attrs.update({'encoding-type': 'dict', 'encoding-version': '0.1.0'})
        write_axis(f, 'obs', ('_index','namespace','barcode_identifier','matrix_row_index'),
                   ('_index','namespace','barcode_identifier'), db.execute('SELECT rendered,ns,bc,row FROM cells ORDER BY row'), n, budget)
        write_axis(f, 'var', ('_index','chrom','start','end','matrix_column_index'), ('_index','chrom'),
                   db.execute('SELECT name,chrom,start,end,col FROM features ORDER BY col'), p, budget)
        x = f.create_group('X'); x.attrs.update({'encoding-type':'csr_matrix','encoding-version':'0.1.0','shape':np.array([n,p],dtype='int64')})
        for name in ('data','indices','indptr'):
            x.create_dataset(name, shape=(0,), maxshape=(None,), chunks=(CHUNK,), dtype='int64', compression='gzip')
        offsets = [0]; endpoint = 0
        def emit():
            nonlocal endpoint
            for cols, values in rows:
                for offset in range(0, len(cols), CHUNK):
                    append(x['indices'], cols[offset:offset+CHUNK]); append(x['data'], values[offset:offset+CHUNK])
                    budget.check()
                endpoint = m.checked_add(endpoint, len(cols)); offsets.append(endpoint)
                if len(offsets) >= CHUNK:
                    append(x['indptr'],offsets); offsets.clear()
                yield cols, values
        summary = m.logical_matrix_identity(n, p, emit())
        append(x['indptr'],offsets)
        for key, value in annotations(bound, summary['logical_matrix_sha256']).items():
            ds = f['uns'].create_dataset(key, data=value, dtype=h5py.string_dtype())
            ds.attrs.update({'encoding-type':'string','encoding-version':'0.2.0'})
        f.flush(); budget.check()
    return summary


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
