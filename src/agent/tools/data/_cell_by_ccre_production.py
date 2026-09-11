"""Qualified BEDTools production, disk-backed record identity and incidence counts."""
import selectors
import sqlite3
import subprocess

from . import scatac_matrix_contract as m, _matrix_bedtools as bed
from ._cell_by_ccre_io import CHUNK


def project(db, bound, root, budget):
    db.execute('CREATE TABLE records (id INTEGER PRIMARY KEY, ns TEXT, ordinal INTEGER, row INTEGER, rank INTEGER, chrom TEXT, start INTEGER, end INTEGER, UNIQUE(ns,ordinal))')
    db.execute('CREATE INDEX record_order ON records(rank,start,end,id)')
    ranks = {c:i for i,(c,_) in enumerate(bound.fragments.contigs)}
    serial = 0
    for library in bound.fragments.libraries:
        for ordinal, record in enumerate(bound.fragments.iter_fragments(library.namespace)):
            cell = db.execute('SELECT row FROM cells WHERE ns=? AND bc=?', record.cell_identity).fetchone()
            if cell is not None:
                db.execute('UPDATE cells SET seen=1 WHERE row=?', cell)
                db.execute('INSERT INTO records VALUES(?,?,?,?,?,?,?,?)',
                           (serial,library.namespace,ordinal,cell[0],ranks[record.contig],record.contig,record.start,record.end))
                serial = m.checked_add(serial,1)
            if ordinal % CHUNK == 0: budget.check()
    if db.execute('SELECT row FROM cells WHERE seen=0 LIMIT 1').fetchone() is not None:
        m.fail('MATRIX_SELECTED_CELL_ABSENT')
    db.commit()
    for name, query in (
        ('a.bed','SELECT chrom,start,end,id FROM records ORDER BY rank,start,end,id'),
        ('b.bed','SELECT chrom,start,end,col FROM features ORDER BY rank,start,end,col')):
        with (root/name).open('x') as f:
            for i,row in enumerate(db.execute(query)):
                f.write('\t'.join(map(str,row))+'\n')
                if i % CHUNK == 0: budget.check()
    (root/'genome.tsv').write_bytes(bed.genome_order_bytes(bound.fragments.contigs))
    budget.check()
    return serial


def incidence(db, line):
    try:
        fields = line.decode('utf-8').rstrip('\n').split('\t')
        if len(fields) != 8: m.fail('MATRIX_INCIDENCE_INVALID')
        fid, col = int(fields[3]), int(fields[7])
        a = db.execute('SELECT chrom,start,end,row FROM records WHERE id=?',(fid,)).fetchone()
        b = db.execute('SELECT chrom,start,end FROM features WHERE col=?',(col,)).fetchone()
        if (a is None or b is None or fields != [*map(str,a[:3]),str(fid),*map(str,b),str(col)]
                or not m.overlaps(a[:3],b)):
            m.fail('MATRIX_INCIDENCE_INVALID')
        db.execute('INSERT INTO incidences VALUES(?,?)',(fid,col))
    except sqlite3.IntegrityError as exc:
        raise m.ScATACMatrixError('MATRIX_DUPLICATE_INCIDENCE') from exc
    except (UnicodeError,ValueError) as exc:
        if isinstance(exc,m.ScATACMatrixError): raise
        raise m.ScATACMatrixError('MATRIX_INCIDENCE_INVALID') from exc
    old = db.execute('SELECT value FROM counts WHERE row=? AND col=?',(a[3],col)).fetchone()
    value = m.checked_add(old[0] if old else 0,1)
    db.execute('INSERT OR REPLACE INTO counts VALUES(?,?,?)',(a[3],col,value))


def intersect(db, executable, root, budget):
    """Pipe backpressure bounds stdout; both pipes drained with cancellation polling."""
    argv = bed.intersection_argv(executable,root/'a.bed',root/'b.bed',root/'genome.tsv')
    with subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=bed.ENVIRONMENT) as process:
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout,selectors.EVENT_READ,'out')
                selector.register(process.stderr,selectors.EVENT_READ,'err')
                pending = b''; errors = 0; count = 0
                while selector.get_map():
                    budget.check()
                    for key,_ in selector.select(timeout=0.05):
                        block = key.fileobj.read1(65536)
                        if not block:
                            selector.unregister(key.fileobj); continue
                        if key.data == 'err':
                            errors += len(block)
                            if errors > bed.MAX_STDERR_BYTES: m.fail('MATRIX_BACKEND_LIMIT')
                            continue
                        pending += block
                        while b'\n' in pending:
                            line,pending = pending.split(b'\n',1)
                            if len(line) >= 65536: m.fail('MATRIX_INCIDENCE_INVALID')
                            incidence(db,line+b'\n'); count += 1
                            if count % CHUNK == 0: budget.check()
                        if len(pending) > 65536: m.fail('MATRIX_INCIDENCE_INVALID')
                if pending: m.fail('MATRIX_INCIDENCE_INVALID')
                while process.poll() is None:
                    budget.check()
                    try: process.wait(timeout=0.05)
                    except subprocess.TimeoutExpired: pass
                if process.returncode: m.fail('MATRIX_INTERSECTION_FAILED')
        finally:
            if process.poll() is None:
                process.terminate()
                try: process.wait(timeout=2)
                except subprocess.TimeoutExpired: process.kill(); process.wait()
    db.commit(); budget.check()


def construct_counts(db, bound, executable, root, budget):
    db.execute('CREATE TABLE incidences (fid INTEGER, col INTEGER, PRIMARY KEY(fid,col)) WITHOUT ROWID')
    db.execute('CREATE TABLE counts (row INTEGER, col INTEGER, value INTEGER, PRIMARY KEY(row,col)) WITHOUT ROWID')
    if not bound.selection['selected_count']:
        return m.overlap_diagnostic(0,0)
    total = project(db,bound,root,budget)
    intersect(db,executable,root,budget)
    # Sorted incidence primary key permits streaming distinct record counting.
    hits = 0; previous = None
    for i,(fid,) in enumerate(db.execute('SELECT fid FROM incidences ORDER BY fid,col')):
        if fid != previous: hits = m.checked_add(hits,1); previous = fid
        if i % CHUNK == 0: budget.check()
    return m.overlap_diagnostic(total,hits)


def rows(db, n):
    cursor = iter(db.execute('SELECT row,col,value FROM counts ORDER BY row,col'))
    current = next(cursor,None)
    for row in range(n):
        columns = []; values = []
        while current is not None and current[0] == row:
            columns.append(current[1]); values.append(current[2]); current = next(cursor,None)
        yield columns,values
    if current is not None: m.fail('MATRIX_CSR_INVALID')
