"""Opt-in native comparison on supplied real fragments; never a Python peak caller.

Run with the isolated MACS3 path, original gzip fragments, barcode text and an
explicit output directory. Source preparation preserves records/coordinates.
"""
import argparse,gzip,hashlib,json,subprocess,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--macs3',required=True);p.add_argument('--fragments',required=True)
    p.add_argument('--barcodes',required=True);p.add_argument('--genome-size',required=True,type=int);p.add_argument('--output',required=True)
    a=p.parse_args();root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    selected=set(Path(a.barcodes).read_text().splitlines());projection=root/'unit-selected.frag';raw_projection=root/'support-selected.frag';n=0;support=0
    with gzip.open(a.fragments,'rt') as f,projection.open('w') as out,raw_projection.open('w') as raw:
        for line in f:
            if line.startswith('#'):continue
            row=line.rstrip('\n').split('\t')
            if row[3] in selected:
                n+=1;support+=int(row[4]);raw.write('\t'.join(row[:5])+'\n');out.write('\t'.join(row[:4]+['1'])+'\n')
    results=[]
    for name,source,options in [('selected-native',str(raw_projection),['--max-count','1']),
                                ('selected-unit',str(projection),[]),('selected-weighted',str(raw_projection),[])]:
        out=root/name;out.mkdir();command=[a.macs3,'callpeak','-t',source,'-f','FRAG','-g',str(a.genome_size),'-q','0.05','-n','peaks','--threads','2','--process-backend','auto','--outdir',str(out),*options]
        start=time.monotonic()
        with (out/'execution.log').open('wb') as log:subprocess.run(['/usr/bin/time','-v',*command],stdout=log,stderr=log,check=True)
        peak=out/'peaks_peaks.narrowPeak';results.append(dict(name=name,command=command,seconds=time.monotonic()-start,peak_count=sum(1 for _ in peak.open()),sha256=hashlib.sha256(peak.read_bytes()).hexdigest()))
    value=dict(records=n,original_support=support,n_barcodes=len(selected),results=results,
        unit_matches_native_cap=results[0]['sha256']==results[1]['sha256'],weighted_differs=results[0]['sha256']!=results[2]['sha256'])
    (root/'agreement.json').write_text(json.dumps(value,indent=2)+'\n')
    assert value['unit_matches_native_cap'] and value['weighted_differs'],value
    raw_projection.unlink()
    projection.unlink() # deterministic exchange only; commands, source and outputs retained
    print(json.dumps(value,indent=2))

if __name__=='__main__':main()
