"""Fixed native export of one selected fragment-backed H5AD for MACS3."""
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import sys
import time


def main(request_path):
    import snapatac2 as snap
    import hdf5plugin  # Native H5AD codecs; no storage conversion.
    request=json.loads(Path(request_path).read_text());stage=Path(request['stage'])
    data=snap.read(request['input_path'],backed='r')
    started=time.monotonic()
    try:
        cells=list(data.obs_names)
        references=data.uns['reference_sequences']
        contigs=dict(zip(references['reference_seq_name'].to_list(),references['reference_seq_length'].to_list()))
        declarations={k:data.uns[k] for k in ('species','assembly') if k in data.uns}
        exported=snap.ex.export_fragments(data,groupby=['population']*len(cells),out_dir=str(stage),suffix='.frag')
    finally:
        data.close()
    export_seconds=time.monotonic()-started
    described=time.monotonic()
    source=Path(exported['population']);destination=stage/'input.frag'
    excluded=set(request['exclude_contigs']);observed=set();n=0;support=0;used=set()
    # Counting metadata is descriptive. The native exporter owns all coordinates
    # and multiplicities; only explicitly requested contigs are removed.
    with source.open() as stream:
        if excluded:
            with destination.open('w') as out:
                for line in stream:
                    fields=line.rstrip('\n').split('\t')
                    if fields[0] in excluded:continue
                    out.write(line);n+=1;support+=int(fields[4]);observed.add(fields[3]);used.add(fields[0])
        else:
            for line in stream:
                fields=line.rstrip('\n').split('\t');n+=1;support+=int(fields[4]);observed.add(fields[3]);used.add(fields[0])
    if not excluded:source.rename(destination)
    (stage/'cells.tsv').write_text('cell_id\n'+''.join(c+'\n' for c in cells))
    implementations=[Path(inspect.getsourcefile(snap.ex.export_fragments)),Path(snap._snapatac2.__file__)]
    receipt=dict(function='snapatac2.ex.export_fragments',versions={'snapatac2':importlib.metadata.version('snapatac2')},
        groupby='one pooled selected native population',ids='native obs_names',min_frag_length=None,max_frag_length=None,
        contigs=contigs,declarations=declarations,n_cells=len(cells),n_observed_barcodes=len(observed),n_fragment_records=n,
        exported_support=support,analyzed_contigs=[c for c in contigs if c in used],
        export_seconds=export_seconds,description_seconds=time.monotonic()-described,
        implementation=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in implementations])
    (stage/'export-receipt.json').write_text(json.dumps(receipt,sort_keys=True)+'\n')


if __name__=='__main__':main(sys.argv[1])
