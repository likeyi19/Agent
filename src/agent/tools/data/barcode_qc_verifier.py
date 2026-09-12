"""Independent per-barcode reconstruction without production or BEDTools calls."""
from bisect import bisect_left,bisect_right
from contextlib import closing
from fractions import Fraction
import hashlib
from itertools import zip_longest
from pathlib import Path
import sqlite3
import tempfile

from . import _barcode_qc_contract as m
from ._barcode_qc_binding import bind
from .scatac_qc_profile import fail
from .scatac_fragments_v2_verifier import take_snapshots,check_snapshots
from agent.tools._cancellation import cancellation_checkpoint


def verify_barcode_qc(path,*,expected_sha256, fragments_authority=None):
    path=Path(path);before=take_snapshots([path])
    manifest=m.load_manifest(path,expected_sha256);value=manifest.to_dict()
    bound=bind(value['arguments'], fragments_authority=fragments_authority)
    if (value['reference_identity_sha256']!=bound.reference.parent_reference_identity_sha256
        or value['qc_resource_identity_sha256']!=bound.reference.resource_identity_sha256
        or value['producer_authority']!=bound.producer_authority
        or value['resource_qualification']!=bound.resource_qualification
        or value['backend_identity']!=bound.backend_identity): fail('QC_BINDING_MISMATCH')
    before=tuple(sorted(before+take_snapshots([path.parent/value[k]['path'] for k in ('table','histogram')])))
    for k in ('table','histogram'):
        if m.resource(path.parent/value[k]['path'])!=value[k]: fail('QC_SIDECAR_MISMATCH')
    # Sorted TSS bases retain opposite-strand multiplicity. At most 1M positions.
    positions={c.name:[] for c in bound.reference.contigs}
    with open(bound.reference.tss.path) as f:
        for i,line in enumerate(f):
            chrom,start,end,strand=line.rstrip('\n').split('\t');positions[chrom].append(int(start))
            if i%1024==0: cancellation_checkpoint()
    qc={c.name for c in bound.reference.contigs if c.classification=='primary_nuclear_qc'}
    hist=[0]*1001;records=0
    with tempfile.TemporaryDirectory(prefix='qc-reconstruct-') as tmp,closing(sqlite3.connect(Path(tmp)/'verify.sqlite')) as db:
        db.execute('PRAGMA cache_size=-8192');db.execute('PRAGMA temp_store=FILE')
        db.execute('CREATE TABLE observed(namespace TEXT,barcode TEXT,total INTEGER,qc INTEGER,center INTEGER,left_count INTEGER,right_count INTEGER,free INTEGER,mono INTEGER,longer INTEGER,PRIMARY KEY(namespace,barcode)) WITHOUT ROWID')
        fields='total qc center left_count right_count free mono longer'.split()
        sql='INSERT INTO observed VALUES (?,?,?,?,?,?,?,?,?,?) ON CONFLICT(namespace,barcode) DO UPDATE SET '+','.join(f'{k}={k}+excluded.{k}' for k in fields)
        for library in bound.fragments.libraries:
            for r in bound.fragments.iter_fragments(library.namespace):
                records+=1
                if records>m.MAX_RECORDS: fail('QC_RESOURCE_LIMIT')
                vals=[1,0,0,0,0,0,0,0]
                if r.contig in qc:
                    vals[1]=1;length=r.end-r.start
                    vals[5 if length<=146 else 6 if length<=293 else 7]=1
                    hist[(length-1) if length<=1000 else 1000]+=1
                    bases=positions[r.contig]
                    for point in (r.start,r.end-1):
                        # Invert each interval against the endpoint. Separate
                        # implementation of inclusive integer-base membership.
                        vals[2]+=bisect_right(bases,point+50)-bisect_left(bases,point-50)
                        vals[3]+=bisect_right(bases,point+2000)-bisect_left(bases,point+1901)
                        vals[4]+=bisect_right(bases,point-1901)-bisect_left(bases,point-2000)
                db.execute(sql,(r.namespace,r.barcode_identifier,*vals))
                if records%1024==0: cancellation_checkpoint()
        n=db.execute('SELECT count(*) FROM observed').fetchone()[0]
        if n!=value['row_count'] or n>m.MAX_BARCODES: fail('QC_BARCODE_UNIVERSE_MISMATCH')
        actual=m.gzip_lines(path.parent/'barcodes.tsv.gz',m.MAX_BARCODES+1)
        if next(actual,None)!=m.HEADER: fail('QC_ROW_INVALID')
        order=hashlib.sha256(m.ORDER_DOMAIN);totals=[0]*8;minimum=None;maximum=0
        number=[0,0];low=[None,None];high=[None,None]
        def rational(f,reason):
            return ('NA','NA',reason) if f is None else (str(f.numerator),str(f.denominator),'DEFINED')
        for i,(row,line) in enumerate(zip_longest(db.execute('SELECT * FROM observed ORDER BY namespace COLLATE BINARY,barcode COLLATE BINARY'),actual)):
            if row is None or line is None: fail('QC_BARCODE_UNIVERSE_MISMATCH')
            ns,bc,total,nqc,c,l,r,free,mono,longer=row
            tss=None if l+r==0 else Fraction(c,101)/Fraction(l+r,200)
            nuc=None if free==0 else Fraction(mono,free)
            expected=('\t'.join((ns,bc,*map(str,row[2:]),*rational(tss,'ZERO_TSS_BACKGROUND'),*rational(nuc,'ZERO_NUCLEOSOME_FREE')))+'\n').encode('ascii')
            if line!=expected: fail('QC_SCIENTIFIC_MISMATCH')
            order.update((ns+'\t'+bc+'\n').encode('ascii'))
            totals=[a+b for a,b in zip(totals,row[2:])]
            minimum=total if minimum is None else min(minimum,total);maximum=max(maximum,total)
            for j,f in enumerate((tss,nuc)):
                if f is not None:
                    number[j]+=1;low[j]=f if low[j] is None else min(low[j],f);high[j]=f if high[j] is None else max(high[j],f)
            if i%1024==0: cancellation_checkpoint()
        if order.hexdigest()!=value['ordered_barcode_sha256']: fail('QC_ORDER_MISMATCH')
        summary=dict(zip(m.COUNTS,totals),depth_min=minimum,depth_max=maximum,
            depth_mean=m.fraction_json(Fraction(records,n)),histogram_records=sum(hist),histogram_overflow=hist[1000])
        for j,prefix in enumerate(('tss','nucleosome')):
            summary.update({prefix+'_defined':number[j],prefix+'_undefined':n-number[j],
                prefix+'_min':m.fraction_json(low[j]),prefix+'_max':m.fraction_json(high[j])})
        if summary!=value['summary']: fail('QC_SUMMARY_MISMATCH')
        expected=(f'{i+1}\t{count}\n'.encode() for i,count in enumerate(hist))
        if any(a!=b for a,b in zip_longest(expected,m.gzip_lines(path.parent/'lengths.tsv.gz',1001))): fail('QC_HISTOGRAM_MISMATCH')
    bound.unchanged();check_snapshots(before);cancellation_checkpoint()
    return manifest
