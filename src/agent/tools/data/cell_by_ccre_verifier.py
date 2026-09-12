"""Complete independent matrix reconstruction; never imports the producer engine."""
import hashlib
from pathlib import Path
import tempfile
import time

import h5py
import numpy as np

from . import scatac_matrix_contract as m, _matrix_bedtools as bed
from . import _cell_by_ccre_io as io
from .scatac_fragments_v2_verifier import take_snapshots, check_snapshots
from agent.tools._cancellation import cancellation_checkpoint


class IntervalIndex:
    """Balanced implicit interval tree with subtree maximum end (32 bytes/feature).

    Sorted starts alone cannot reject nested or long intervals. Every subtree
    carries its maximum end; a query visits both children whenever necessary.
    """
    def __init__(self, rows, count):
        self.values = np.empty((count,4),dtype=np.int64)
        n = 0
        for n,(start,end,col) in enumerate(rows,1):
            self.values[n-1,:3] = start,end,col
            if n % io.CHUNK == 0: cancellation_checkpoint()
        if n != count: m.fail('MATRIX_COLUMN_IDENTITY_INVALID')
        def augment(left,right):
            if left >= right: return -1
            mid = (left+right)//2
            if mid % io.CHUNK == 0: cancellation_checkpoint()
            maximum = max(int(self.values[mid,1]),augment(left,mid),augment(mid+1,right))
            self.values[mid,3] = maximum
            return maximum
        augment(0,count)

    def query(self,start,end):
        stack = [(0,len(self.values))]
        while stack:
            left,right = stack.pop()
            if left >= right: continue
            mid = (left+right)//2
            if self.values[mid,3] <= start or self.values[left,0] >= end: continue
            a,b,col,_ = self.values[mid]
            if a < end and b > start: yield int(col)
            stack.append((left,mid)); stack.append((mid+1,right))


def reconstruct(db, bound, budget):
    db.execute('CREATE TABLE expected (cell INTEGER, feature INTEGER, amount INTEGER, PRIMARY KEY(cell,feature)) WITHOUT ROWID')
    # Iterate by contig once across all libraries using an independent disk spool.
    db.execute('CREATE TABLE source (rank INTEGER, ns TEXT, ordinal INTEGER, cell INTEGER, start INTEGER, end INTEGER, PRIMARY KEY(rank,ns,ordinal)) WITHOUT ROWID')
    ranks = {c:i for i,(c,_) in enumerate(bound.fragments.contigs)}
    total = 0
    if not bound.selection['selected_count']:
        return m.overlap_diagnostic(0,0)
    if bound.selection['selected_count']:
        for library in bound.fragments.libraries:
            for ordinal, record in enumerate(bound.fragments.iter_fragments(library.namespace)):
                found = db.execute('SELECT row FROM cells WHERE ns=? AND bc=?',record.cell_identity).fetchone()
                if found is not None:
                    db.execute('UPDATE cells SET seen=1 WHERE row=?',found)
                    db.execute('INSERT INTO source VALUES(?,?,?,?,?,?)',
                               (ranks[record.contig],library.namespace,ordinal,found[0],record.start,record.end))
                    total += 1
                    if total > m.INT64_MAX: m.fail('MATRIX_INTEGER_INVALID')
                if ordinal % io.CHUNK == 0: budget.check()
    if db.execute('SELECT row FROM cells WHERE seen=0 LIMIT 1').fetchone(): m.fail('MATRIX_SELECTED_CELL_ABSENT')
    hits = 0; contributions = 0
    for rank in range(len(ranks)):
        count = db.execute('SELECT count(*) FROM features WHERE rank=?',(rank,)).fetchone()[0]
        index = IntervalIndex(db.execute('SELECT start,end,col FROM features WHERE rank=? ORDER BY start,end,col',(rank,)),count)
        budget.check()
        for cell,start,end in db.execute('SELECT cell,start,end FROM source WHERE rank=? ORDER BY ns,ordinal',(rank,)):
            matched = False
            for col in index.query(start,end):
                matched = True
                old = db.execute('SELECT amount FROM expected WHERE cell=? AND feature=?',(cell,col)).fetchone()
                amount = 1 if old is None else old[0]+1
                if amount > m.INT64_MAX: m.fail('MATRIX_INTEGER_INVALID')
                db.execute('INSERT OR REPLACE INTO expected VALUES(?,?,?)',(cell,col,amount))
                contributions += 1
                if contributions > m.INT64_MAX: m.fail('MATRIX_INTEGER_INVALID')
                if contributions % io.CHUNK == 0: budget.check()
            hits += matched
        del index
    db.commit(); budget.check()
    return m.overlap_diagnostic(total,hits)


def verify_axes(f,db,bound,budget):
    if (set(f) != {'X','obs','var','uns','layers','obsm','obsp','varm','varp'}
            or f.attrs.get('encoding-type') != 'anndata' or f.attrs.get('encoding-version') != '0.1.0'):
        m.fail('MATRIX_H5AD_INVALID')
    # Reject external/soft links, virtual datasets, and external raw storage.
    def inspect(group):
        for name in group:
            if not isinstance(group.get(name,getlink=True),h5py.HardLink): m.fail('MATRIX_H5AD_INVALID')
            item = group[name]
            if isinstance(item,h5py.Dataset):
                if item.is_virtual or item.external: m.fail('MATRIX_H5AD_INVALID')
            elif isinstance(item,h5py.Group):
                if item.name.count('/') > 2: m.fail('MATRIX_H5AD_INVALID')
                inspect(item)
    inspect(f)
    for name in ('layers','obsm','obsp','varm','varp'):
        if len(f[name]) or f[name].attrs.get('encoding-type') != 'dict': m.fail('MATRIX_H5AD_INVALID')
    for name,columns,strings,query,length in (
        ('obs',('_index','namespace','barcode_identifier','matrix_row_index'),('_index','namespace','barcode_identifier'),
         'SELECT rendered,ns,bc,row FROM cells ORDER BY row',bound.selection['selected_count']),
        ('var',('_index','chrom','start','end','matrix_column_index'),('_index','chrom'),
         'SELECT name,chrom,start,end,col FROM features ORDER BY col',bound.reference.ccre.feature_count)):
        group = f[name]
        if (set(group) != set(columns) or group.attrs.get('_index') != '_index'
                or group.attrs.get('encoding-type') != 'dataframe' or group.attrs.get('encoding-version') != '0.2.0'
                or list(group.attrs.get('column-order',[])) != list(columns[1:])): m.fail('MATRIX_AXIS_INVALID')
        for key in columns:
            ds = group[key]
            if not isinstance(ds,h5py.Dataset) or ds.shape != (length,): m.fail('MATRIX_AXIS_INVALID')
            if key in strings:
                if h5py.check_string_dtype(ds.dtype) is None: m.fail('MATRIX_AXIS_INVALID')
            elif ds.dtype.kind not in 'iu': m.fail('MATRIX_AXIS_INVALID')
            if ds.attrs.get('encoding-type') != ('string-array' if key in strings else 'array') or ds.attrs.get('encoding-version') != '0.2.0':
                m.fail('MATRIX_AXIS_INVALID')
        cursor = db.execute(query); offset = 0
        while expected := cursor.fetchmany(io.CHUNK):
            actual = [group[k].asstr()[offset:offset+len(expected)].tolist() if k in strings
                      else group[k][offset:offset+len(expected)].tolist() for k in columns]
            if list(zip(*actual)) != expected: m.fail('MATRIX_AXIS_INVALID')
            offset += len(expected); budget.check()


from .authority_context import owned_verification

@owned_verification('matrix')
def verify_cell_by_ccre(manifest_path, *, expected_sha256, bedtools_path, limits=io.MatrixLimits(), scratch_parent=None):
    """Fresh complete scientific verification; returns bounded evidence, no matrix copy."""
    start = time.monotonic(); path = Path(manifest_path)
    m.absolute_path(str(path)); m.sha(expected_sha256)
    before = take_snapshots([path,path.parent/'matrix.h5ad'])
    with path.open('rb') as f: raw = f.read(m.MAX_MANIFEST_BYTES+1)
    if hashlib.sha256(raw).hexdigest() != expected_sha256: m.fail('MATRIX_MANIFEST_MISMATCH')
    value = m.load_manifest_bytes(raw)
    payload = path.parent/'matrix.h5ad'
    if payload.stat().st_size != value['matrix']['size_bytes'] or bed.file_sha(payload) != value['matrix']['sha256']:
        m.fail('MATRIX_PAYLOAD_MISMATCH')
    if bed.qualify_runtime(bedtools_path) != value['backend']: m.fail('MATRIX_BACKEND_INVALID')
    bound = io.bind(value['upstream'],limits)
    if (bound.upstream != value['upstream'] or bound.reference.species != value['species']
            or bound.reference.target_assembly != value['assembly']
            or bound.selection['ordered_selected_sha256'] != value['ordered_selected_sha256']
            or bound.reference.ccre.ordered_feature_sha256 != value['ordered_feature_sha256']
            or value['shape'] != [bound.selection['selected_count'],bound.reference.ccre.feature_count]):
        m.fail('MATRIX_LINEAGE_INVALID')
    with tempfile.TemporaryDirectory(prefix='.matrix-verification-',dir=scratch_parent) as directory:
        budget = io.Budget(directory,limits)
        with io.database(Path(directory)/'verify.sqlite',budget) as db:
            io.load_axes(db,bound,budget)
            diagnostic = reconstruct(db,bound,budget)
            with h5py.File(payload,'r') as f:
                verify_axes(f,db,bound,budget)
                expected_uns = io.annotations(bound,value['logical_matrix_sha256'])
                if set(f['uns']) != set(expected_uns): m.fail('MATRIX_H5AD_INVALID')
                for key,text in expected_uns.items():
                    ds = f['uns'][key]
                    if (not isinstance(ds,h5py.Dataset) or ds.shape != () or ds.size != 1
                            or ds.attrs.get('encoding-type') != 'string'
                            or ds.attrs.get('encoding-version') != '0.2.0'
                            or ds.asstr()[()] != text): m.fail('MATRIX_H5AD_INVALID')
                n,p = value['shape']
                def compare():
                    for row,(columns,amounts) in enumerate(io.csr_rows(f,n,p,budget)):
                        cursor = iter(db.execute('SELECT feature,amount FROM expected WHERE cell=? ORDER BY feature',(row,)))
                        for col,amount in zip(columns,amounts):
                            if next(cursor,None) != (col,amount): m.fail('MATRIX_SCIENCE_MISMATCH')
                        if next(cursor,None) is not None: m.fail('MATRIX_SCIENCE_MISMATCH')
                        yield columns,amounts
                summary = m.logical_matrix_identity(n,p,compare())
            if any(value[key] != result for key,result in summary.items()) or diagnostic != value['diagnostic']:
                m.fail('MATRIX_SCIENCE_MISMATCH')
            budget.check()
    bound.unchanged(); check_snapshots(before)
    return dict(manifest_path=str(path),manifest_sha256=expected_sha256,identity_sha256=value['identity_sha256'],
                **summary,diagnostic=diagnostic,verification_seconds=time.monotonic()-start,
                verification_scratch_peak_bytes=budget.peak_bytes)
