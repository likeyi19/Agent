"""Opt-in spot check of a real peak matrix, not a replacement matrix engine."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    import anndata as ad
    import numpy as np
    import pysam
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix',required=True,type=Path)
    parser.add_argument('--fragments',required=True,type=Path)
    parser.add_argument('--output-json',required=True,type=Path)
    args=parser.parse_args()
    matrix=ad.read_h5ad(args.matrix)
    assert matrix.uns['matrix_semantics']=='fragment_counts'
    assert matrix.obs.namespace.nunique()==1
    barcodes={barcode:i for i,barcode in enumerate(matrix.obs.barcode_identifier.astype(str))}
    assert len(barcodes)==matrix.n_obs
    indices=sorted(set([0,matrix.n_vars-1,int(np.asarray(matrix.X.sum(0)).argmax()),
        *np.random.default_rng(20261031).choice(matrix.n_vars,min(5,matrix.n_vars),replace=False)]))
    checks=[]
    with pysam.TabixFile(str(args.fragments)) as source:
        for index in indices:
            feature=matrix.var.iloc[index];counts=np.zeros(matrix.n_obs,dtype=np.int64);support=0
            for line in source.fetch(str(feature['chrom']),int(feature['start']),int(feature['end'])):
                chrom,start,end,barcode,weight,*_=line.split('\t')
                if int(start)<int(feature['end']) and int(end)>int(feature['start']) and barcode in barcodes:
                    counts[barcodes[barcode]]+=1;support+=int(weight)
            actual=matrix.X[:,index].toarray().ravel()
            assert np.array_equal(counts,actual),matrix.var_names[index]
            checks.append(dict(feature_index=int(index),feature=str(matrix.var_names[index]),
                fragment_records=int(counts.sum()),source_support=support,all_cells_equal=True))
    def sha(path):
        with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
    value=dict(matrix=dict(path=str(args.matrix),sha256=sha(args.matrix),shape=list(matrix.shape),
        nnz=int(matrix.X.nnz),total_count=int(matrix.X.sum()),dtype=str(matrix.X.dtype)),
        fragments=dict(path=str(args.fragments),sha256=sha(args.fragments)),
        checks=checks,scope='Eight selected features at most; all selected cells. Not a full matrix reconstruction.')
    with args.output_json.open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')


if __name__=='__main__':main()
