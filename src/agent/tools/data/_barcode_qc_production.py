"""Disk-backed QC production; BEDTools provides interval incidence only."""
from contextlib import closing
from fractions import Fraction
import hashlib
from pathlib import Path
import sqlite3
import subprocess

from . import _barcode_qc_contract as m
from .scatac_qc_profile import tss_windows, fragment_length_bin, tss_enrichment, nucleosome_signal, fail
from ._qc_bedtools import qc_intersection_argv, ENVIRONMENT
from agent.tools._cancellation import cancellation_checkpoint

MAX_SCRATCH_BYTES=64*1024**3


def _db(path):
    db=sqlite3.connect(path)
    db.execute('PRAGMA cache_size=-8192');db.execute('PRAGMA temp_store=FILE')
    db.execute('CREATE TABLE metrics(ns TEXT,bc TEXT,'+','.join(f'c{i} INTEGER NOT NULL' for i in range(8))+',PRIMARY KEY(ns,bc)) WITHOUT ROWID')
    db.execute('CREATE TABLE endpoints(id INTEGER PRIMARY KEY,rank INTEGER,pos INTEGER,ns TEXT,bc TEXT)')
    db.execute('CREATE TABLE windows(rank INTEGER,start INTEGER,end INTEGER,label TEXT PRIMARY KEY)')
    return db


def _intersection(bound, scratch, a, b, genome):
    argv=qc_intersection_argv(bound.executable,insertions_path=a,windows_path=b,genome_path=genome)
    output=scratch/'incidences.tsv';errors=scratch/'bedtools.stderr'
    cancellation_checkpoint()
    with output.open('xb') as out,errors.open('xb') as err:
        process=subprocess.Popen(argv,stdout=out,stderr=err,env=ENVIRONMENT)
        try:
            while True:
                try:
                    status=process.wait(timeout=0.2);break
                except subprocess.TimeoutExpired:
                    cancellation_checkpoint()
                    if output.stat().st_size>m.MAX_SIDECAR or errors.stat().st_size>1024*1024: fail('QC_INTERSECTION_LIMIT')
            if status!=0: fail('QC_INTERSECTION_FAILED')
            if output.stat().st_size>m.MAX_SIDECAR: fail('QC_INTERSECTION_LIMIT')
        finally:
            if process.poll() is None:
                process.terminate()
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired: process.kill();process.wait()
    cancellation_checkpoint()
    return output


def produce(bound,directory):
    """Return deterministic sidecars and summaries in a private stage."""
    directory=Path(directory);scratch=directory/'scratch';scratch.mkdir()
    dictionary={c.name:(i,c) for i,c in enumerate(bound.reference.contigs)}
    names=[c.name for c in bound.reference.contigs];hist=[0]*1001;records=0
    with closing(_db(scratch/'production.sqlite')) as db:
        placeholders=','.join('?' for _ in range(10))
        upsert='INSERT INTO metrics VALUES ('+placeholders+') ON CONFLICT(ns,bc) DO UPDATE SET '+','.join(f'c{i}=c{i}+excluded.c{i}' for i in range(8))
        for library in bound.fragments.libraries:
            for r in bound.fragments.iter_fragments(library.namespace):
                records+=1
                if records>m.MAX_RECORDS: fail('QC_RESOURCE_LIMIT')
                rank,contig=dictionary[r.contig];qc=contig.classification=='primary_nuclear_qc'
                counts=[1,int(qc),0,0,0,0,0,0]
                if qc:
                    length=r.end-r.start;kind=fragment_length_bin(length)
                    counts[{'nucleosome_free':5,'mononucleosomal':6,'longer':7}[kind]]=1
                    hist[min(length,1001)-1]+=1
                    db.executemany('INSERT INTO endpoints(rank,pos,ns,bc) VALUES (?,?,?,?)',
                        [(rank,r.start,r.namespace,r.barcode_identifier),(rank,r.end-1,r.namespace,r.barcode_identifier)])
                m.identity(r.namespace,r.barcode_identifier)
                db.execute(upsert,(r.namespace,r.barcode_identifier,*counts))
                if records%1024==0:
                    cancellation_checkpoint()
                    if (scratch/'production.sqlite').stat().st_size>MAX_SCRATCH_BYTES: fail('QC_SCRATCH_LIMIT')
        n=db.execute('SELECT count(*) FROM metrics').fetchone()[0]
        if not 1<=n<=m.MAX_BARCODES: fail('QC_RESOURCE_LIMIT')
        # Resource bytes have already been independently rederived in bind().
        from .scatac_qc_reference import _lines
        for i,line in enumerate(_lines(bound.reference.tss.path)):
            chrom,p,end,strand=line.decode().rstrip('\n').split('\t');rank,c=dictionary[chrom]
            for label,(start,end) in zip(('c','l','r'),tss_windows(int(p),c.length)):
                db.execute('INSERT INTO windows VALUES (?,?,?,?)',(rank,start,end,f'{label}:{i}'))
            if i%1024==0: cancellation_checkpoint()
        a=scratch/'endpoints.bed';b=scratch/'windows.bed';genome=scratch/'genome.tsv'
        genome.write_text(''.join(f'{c.name}\t{c.length}\n' for c in bound.reference.contigs))
        cancellation_checkpoint()
        with a.open('w') as f:
            for i,(rid,rank,pos) in enumerate(db.execute('SELECT id,rank,pos FROM endpoints ORDER BY rank,pos,id')):
                f.write(f'{names[rank]}\t{pos}\t{pos+1}\t{rid}\n')
                if i%1024==0: cancellation_checkpoint()
        with b.open('w') as f:
            for i,(rank,start,end,label) in enumerate(db.execute('SELECT * FROM windows ORDER BY rank,start,end,label')):
                f.write(f'{names[rank]}\t{start}\t{end}\t{label}\n')
                if i%1024==0: cancellation_checkpoint()
        output=_intersection(bound,scratch,a,b,genome)
        with output.open('rb') as f:
            for i,line in enumerate(iter(lambda:f.readline(m.MAX_ROW+1),b'')):
                if len(line)>m.MAX_ROW or not line.endswith(b'\n'): fail('QC_INTERSECTION_INVALID')
                try: fields=line[:-1].decode('ascii').split('\t')
                except UnicodeDecodeError: fail('QC_INTERSECTION_INVALID')
                if len(fields)!=8 or fields[7][:2] not in ('c:','l:','r:'): fail('QC_INTERSECTION_INVALID')
                try: rid=int(fields[3])
                except ValueError: fail('QC_INTERSECTION_INVALID')
                if not 1<=rid<=2*m.MAX_RECORDS: fail('QC_INTERSECTION_INVALID')
                row=db.execute('SELECT ns,bc,rank,pos FROM endpoints WHERE id=?',(rid,)).fetchone()
                window=db.execute('SELECT rank,start,end FROM windows WHERE label=?',(fields[7],)).fetchone()
                # Bind both returned BED records to their exact generated inputs.
                # A correct aggregate cannot excuse malformed incidence output.
                if (row is None or window is None
                    or fields[:4]!=[names[row[2]],str(row[3]),str(row[3]+1),str(rid)]
                    or fields[4:7]!=[names[window[0]],str(window[1]),str(window[2])]
                    or row[2]!=window[0] or not window[1]<=row[3]<window[2]): fail('QC_INTERSECTION_INVALID')
                col={'c':2,'l':3,'r':4}[fields[7][0]]
                db.execute(f'UPDATE metrics SET c{col}=c{col}+1 WHERE ns=? AND bc=?',row[:2])
                if i%1024==0: cancellation_checkpoint()
        order=hashlib.sha256(m.ORDER_DOMAIN);totals=[0]*8;depth_min=None;depth_max=0
        defined=[0,0];extrema=[[None,None],[None,None]]
        def rows():
            nonlocal depth_min,depth_max
            yield m.HEADER
            for i,row in enumerate(db.execute('SELECT * FROM metrics ORDER BY ns COLLATE BINARY,bc COLLATE BINARY')):
                ns,bc,*counts=row;order.update(m.identity(ns,bc))
                for j,count in enumerate(counts): totals[j]+=count
                depth_min=counts[0] if depth_min is None else min(depth_min,counts[0]);depth_max=max(depth_max,counts[0])
                ratios=(tss_enrichment(*counts[2:5])[0],nucleosome_signal(counts[5],counts[6])[0])
                for j,value in enumerate(ratios):
                    if value is not None:
                        defined[j]+=1;lo,hi=extrema[j]
                        extrema[j]=[value if lo is None else min(lo,value),value if hi is None else max(hi,value)]
                if i%1024==0: cancellation_checkpoint()
                yield m.row_bytes(ns,bc,counts)
        m.write_gzip(directory/'barcodes.tsv.gz',rows())
        m.write_gzip(directory/'lengths.tsv.gz',(f'{i+1}\t{x}\n'.encode() for i,x in enumerate(hist)))
        summary=dict(zip(m.COUNTS,totals),depth_min=depth_min,depth_max=depth_max,
            depth_mean=m.fraction_json(Fraction(totals[0],n)),histogram_records=sum(hist),histogram_overflow=hist[-1])
        for j,prefix in enumerate(('tss','nucleosome')):
            summary.update({prefix+'_defined':defined[j],prefix+'_undefined':n-defined[j],
                prefix+'_min':m.fraction_json(extrema[j][0]),prefix+'_max':m.fraction_json(extrema[j][1])})
    bound.unchanged();cancellation_checkpoint()
    return dict(row_count=n,ordered_barcode_sha256=order.hexdigest(),summary=summary,
        table=m.resource(directory/'barcodes.tsv.gz'),histogram=m.resource(directory/'lengths.tsv.gz'))
