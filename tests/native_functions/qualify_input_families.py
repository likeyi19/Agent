"""Opt-in 1G direct installed-native comparisons; writes only new external evidence."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmwrite, mmread
from agent.tools.analysis import motifmatchr, chromvar_native
from agent.tools.analysis.native_tables import MACS3
from agent.tools.analysis.native_matrix import SIGNAC

root = Path(sys.argv[1]).resolve(); root.mkdir(parents=True, exist_ok=False)
rt = Path('/home/likeyi/program/agent-data/resources/runtimes')
os.environ['AGENT_SCIENTIFIC_RESOURCE_ROOT'] = '/home/likeyi/program/agent-data/resources/library'
records = []
def run(command, label):
    with (root/(label+'.log')).open('w') as f:
        subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, check=True, timeout=600)
def case(label, fn):
    if len(sys.argv)>2 and label not in sys.argv[2:]: return
    try:
        value = fn(); records.append(dict(case=label, status='passed', detail=value))
    except Exception as e:
        traceback.print_exc(); records.append(dict(case=label, status='failed', error=str(e)))
    (root/'results.json').write_text(json.dumps(records,indent=2)); print(label, records[-1]['status'], flush=True)
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def execute(owner, name, source, **kw):
    args = dict(input_path=str(source), output_dir=str(root/name), **kw)
    before = sha(source); result = owner.execute(args, name)
    assert owner.recover(args, name) == result and sha(source) == before
    return result

generator = np.random.default_rng(47)
a = ad.AnnData(sparse.csr_matrix(generator.poisson(generator.uniform(.1, 2, 1200), (48, 1200)).astype(float)),
    obs=pd.DataFrame(index=['left'+str(i) for i in range(48)]),
    var=pd.DataFrame({'bias': generator.uniform(.2, .8, 1200)},index=['peak'+str(i) for i in range(1200)]))
a.write_h5ad(root/'counts.h5ad'); mmwrite(root/'counts.mtx',a.X.T)
a.var[['bias']].to_csv(root/'bias.tsv',sep='\t',index=False)
b = a[:17].copy(); b.obs_names=['right'+str(i) for i in range(17)]; b.X=sparse.csr_matrix(generator.binomial(1,.2,b.shape).astype(float));b.write_h5ad(root/'second.h5ad');mmwrite(root/'second.mtx',b.X.T)
left=a.copy();left.X=(left.X>0).astype(float);left.write_h5ad(root/'binary.h5ad');mmwrite(root/'binary.mtx',left.X.T)
script=root/'direct-statistics.R';script.write_text('''args <- commandArgs(TRUE); setwd(args[1]); suppressPackageStartupMessages(library(Matrix))
if(args[2]=='background') {
 library(chromVAR);library(BiocParallel);set.seed(7);register(SerialParam(RNGseed=7))
 x<-as(readMM('counts.mtx'),'CsparseMatrix');bias<-read.delim('bias.tsv')$bias
 r<-chromVAR::getBackgroundPeaks(x,bias=bias,niterations=5,bs=10)
 write.table(r,'background-direct.tsv',sep='\\t',quote=FALSE,row.names=FALSE,col.names=FALSE)
} else {
 library(Signac);x<-as(readMM(if(args[2]=='Jaccard') 'binary.mtx' else 'counts.mtx'),'CsparseMatrix');y<-as(readMM('second.mtx'),'CsparseMatrix')
 r<-if(args[2]=='Jaccard') Signac::Jaccard(t(x),t(y)) else Signac::corSparse(x,y,cov=args[2]=='cov')
 writeMM(Matrix(r,sparse=TRUE),paste0(args[2],'-direct.mtx'))
}
''')
def background():
    out=execute(chromvar_native,'background',root/'counts.h5ad',function_name='getBackgroundPeaks',kwargs={'niterations':5,'bs':10},seed=7)
    run([str(rt/'chromvar/env/bin/Rscript'),str(script),str(root),'background'],'background-direct')
    actual=pd.read_csv(out['table_path'],sep='\t').iloc[:,2:].to_numpy();expected=np.loadtxt(root/'background-direct.tsv')
    np.testing.assert_array_equal(actual,expected)
    assert actual.min()>=1 and actual.max()<=a.n_vars
    return dict(shape=list(actual.shape),native_index_base=1,manifest_sha256=out['manifest_sha256'])
case('background', background)
for name in ('Jaccard','corSparse','cov'):
    def pair(name=name):
        out=execute(SIGNAC,name,root/('binary.h5ad' if name=='Jaccard' else 'counts.h5ad'),function_name='Jaccard' if name=='Jaccard' else 'corSparse',kwargs={'cov':True} if name=='cov' else {},second_path=str(root/'second.h5ad'))
        run([str(rt/'signac-1/env/bin/Rscript'),str(script),str(root),name],name+'-direct')
        actual=ad.read_h5ad(out['matrix_path']).uns['Signac_pairwise']
        np.testing.assert_allclose(actual['values'],mmread(root/(name+'-direct.mtx')).toarray(),rtol=1e-12,atol=1e-14,equal_nan=True)
        assert list(actual['row_cells'])==list(a.obs_names) and list(actual['column_cells'])==list(b.obs_names)
        return dict(shape=list(actual['values'].shape),source=out['source_identity'])
    case(name,pair)

for name,values in [('t1',[2,40,40,2,5,5]),('t2',[2,5,5,2,40,40]),('c1',[2]*6),('c2',[2]*6)]:
    (root/(name+'.bdg')).write_text(''.join(f'chr1\t{i*300}\t{(i+1)*300}\t{v}\n' for i,v in enumerate(values)))
def differential():
    kw={'cutoff':1,'min-len':100,'max-gap':50,'d1':1,'d2':1}
    out=execute(MACS3,'bdgdiff',root/'t1.bdg',function_name='bdgdiff',kwargs=kw,control_path=str(root/'c1.bdg'),treatment2_path=str(root/'t2.bdg'),control2_path=str(root/'c2.bdg'))
    direct=root/'macs-direct';direct.mkdir()
    run([str(rt/'macs3/env/bin/macs3'),'bdgdiff','--t1',str(root/'t1.bdg'),'--t2',str(root/'t2.bdg'),'--c1',str(root/'c1.bdg'),'--c2',str(root/'c2.bdg'),'--cutoff','1','--min-len','100','--max-gap','50','--d1','1','--d2','1','--outdir',str(direct),'--ofile','result.tsv','condition2.tsv','common.tsv'],'macs-direct')
    for name in ('result.tsv','condition2.tsv','common.tsv'):assert (root/'bdgdiff'/name).read_bytes()==(direct/name).read_bytes()
    return dict(files=out['result_files'],manifest_sha256=out['manifest_sha256'])
case('bdgdiff',differential)

original=Path('/home/likeyi/program/agent-data/datasets/derived/10x-genomics/pbmc-1k-nextgem/cell-atac-2.0.0/ordinary-peak-counts/atac_pbmc_1k_nextgem.h5ad')
source=ad.read_h5ad(original,backed='r'); subset=source[:96,:2000].to_memory();source.file.close();subset.write_h5ad(root/'pbmc-peaks.h5ad')
(root/'pbmc-source.json').write_text(json.dumps(dict(original=str(original),original_sha256=sha(original),selection='first 96 cells and 2000 peaks, exact ordered slicing',subset_sha256=sha(root/'pbmc-peaks.h5ad')),indent=2))
for mode in ('scores','positions'):
    def motif(mode=mode):
        out=execute(motifmatchr,'motif-'+mode,root/'pbmc-peaks.h5ad',function_name='matchMotifs',kwargs={'bg':'even','p.cutoff':.0001,'out':mode},coordinate_system='zero_based_half_open')
        direct=root/('motif-'+mode+'-direct.R');direct.write_text('''suppressPackageStartupMessages({library(motifmatchr);library(TFBSTools);library(GenomicRanges);library(Rsamtools);library(SummarizedExperiment)})
args<-commandArgs(TRUE);setwd(args[1]);p<-read.delim('peaks.tsv',check.names=FALSE,quote='')
gr<-GRanges(p$chrom,IRanges(p$start+1,p$end));names(gr)<-p$feature_id
pfms<-readJASPARMatrix('/home/likeyi/program/agent-data/resources/references/motifs/JASPAR2026/JASPAR2026_CORE_vertebrates_non-redundant_pfms_jaspar.txt',matrixClass='PFM')
r<-motifmatchr::matchMotifs(pfms,gr,genome=FaFile('/home/likeyi/program/agent-data/resources/references/human/hg38/10x-arc-2020a/genome.fa'),p.cutoff=.0001,bg='even',out=args[2])
stopifnot(identical(r,readRDS('native-result.rds')))
if(args[2]=='scores') for(a in c('motifMatches','motifScores','motifCounts')) cat(a,dim(assay(r,a)),'\\n')
if(args[2]=='positions') cat('motifs',length(r),'hits',sum(lengths(r)),'\\n')
''')
        run([str(rt/'chromvar/env/bin/Rscript'),str(direct),str(root/('motif-'+mode)),mode], 'motif-'+mode+'-direct')
        return dict(peaks=out['n_peaks'],motifs=out['n_motifs'],n_matches=out['n_matches'],n_positions=out.get('n_positions'),manifest_sha256=out['manifest_sha256'],comparison='identical complete native R objects')
    case('motif-'+mode,motif)
if any(row['status']!='passed' for row in records):sys.exit(1)
