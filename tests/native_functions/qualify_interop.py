"""Opt-in installed-package STE-NFI-1E parity. Evidence root must be outside Git.

PYTHONPATH=src python tests/native_functions/qualify_interop.py /absolute/evidence
No credentials, provider calls or automatic scientific retries.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import zipfile

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread, mmwrite
from agent.tools.analysis import cicero_cds, chromvar_native
from agent.tools.analysis.native_matrix import EPISCANPY, SCANPY, SIGNAC, SNAPATAC2, NativeInvocationError
from agent.tools.analysis.native_tables import MACS3

root = Path(sys.argv[1]).resolve(); root.mkdir(parents=True, exist_ok=False)
rt = Path(os.environ.get('AGENT_NATIVE_RUNTIME_ROOT', '/home/likeyi/program/agent-data/resources/runtimes'))
for key, rel in [('SCANPY_PYTHON','episcanpy-0.4.0/env/bin/python'), ('EPISCANPY_PYTHON','episcanpy-0.4.0/env/bin/python'), ('SNAPATAC2_PYTHON','native-matrix-1/env/bin/python'), ('SIGNAC_RSCRIPT','signac-1/env/bin/Rscript'), ('CICERO_RSCRIPT','cicero-1/env/bin/Rscript'), ('CHROMVAR_RSCRIPT','chromvar/env/bin/Rscript'), ('MACS3_EXECUTABLE','macs3/env/bin/macs3')]:
    os.environ['AGENT_' + key] = str(rt / rel)
os.environ['OPENBLAS_NUM_THREADS'] = '1'; os.environ['OMP_NUM_THREADS'] = '1'
records = []
rng = np.random.default_rng(71)
a = ad.AnnData(sparse.csr_matrix(rng.poisson(.6, (80,120)).astype(float)),
    obs=pd.DataFrame(index=['cell'+str(i) for i in range(80)]),
    var=pd.DataFrame(index=['chr1_'+str(10000*i+1)+'_'+str(10000*i+500) for i in range(120)]))
a.obsm['provided'] = rng.normal(size=(80,4)); a.obs['depth'] = np.asarray(a.X.sum(1)).ravel(); a.uns['iroot'] = 0
a.write_h5ad(root/'ordinary.h5ad')
signed = a.copy(); signed.X.data[:10] = -.25; signed.write_h5ad(root/'signed.h5ad')
(root/'genome.tsv').write_text('chr1\t2000000\n')


def run(command, label):
    with (root/(label+'.log')).open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy(), check=True, timeout=600)


def equal(a,b):
    if sparse.issparse(a): a = a.toarray()
    if sparse.issparse(b): b = b.toarray()
    np.testing.assert_allclose(a,b,rtol=2e-6,atol=2e-7,equal_nan=True)


def case(label, fn):
    try:
        detail = fn(); records.append(dict(case=label,status='passed',detail=detail))
    except Exception as e:
        traceback.print_exc(); records.append(dict(case=label,status='failed',error=str(e)))
    (root/'results.json').write_text(json.dumps(records,indent=2)); print(label,records[-1]['status'],flush=True)


outputs = {}
def matrix(label, owner, name, source, kwargs):
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    args = dict(input_path=str(source),output_dir=str(root/label),function_name=name,kwargs=kwargs)
    result = owner.execute(args,'parity-'+label); outputs[label] = Path(result['matrix_path'])
    assert owner.recover(args,'parity-'+label) == result
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    pkg = owner.package
    code = 'import anndata as ad, json, sys\nimport '+pkg+' as pkg\na=ad.read_h5ad(sys.argv[1])\nr=pkg.'+name+'(a,**json.loads(sys.argv[3]))\nif isinstance(r,ad.AnnData):a=r\na.write_h5ad(sys.argv[2])\n'
    script = root/(label+'-reference.py'); script.write_text(code)
    reference = root/(label+'-reference.h5ad')
    run([str(rt/owner.runtime/'env/bin/python'),'-I',str(script),str(source),str(reference),json.dumps(kwargs)],label+'-reference')
    actual, ref = ad.read_h5ad(result['matrix_path']),ad.read_h5ad(reference)
    if ref.X is not None: equal(actual.X,ref.X)
    for group in ('obsm','varm','obsp','layers'):
        for k,v in getattr(ref,group).items():equal(getattr(actual,group)[k],v)
    for k in ref.obs:
        if ref.obs[k].dtype.kind in 'biuf': equal(actual.obs[k],ref.obs[k])
        else: assert actual.obs[k].tolist()==ref.obs[k].tolist()
    assert actual.obs_names.tolist()==ref.obs_names.tolist() and actual.var_names.tolist()==ref.var_names.tolist()
    return result

for label,name,source,kw in [
    ('signed-binarize','pp.binarize','signed.h5ad',{}),
    ('signed-tfidf','pp.tfidf','signed.h5ad',{}),
    ('pca','tl.pca','signed.h5ad',{'n_comps':5,'random_state':7}),
    ('neighbors','pp.neighbors','ordinary.h5ad',{'use_rep':'provided','n_neighbors':6}),
    ('tsne','tl.tsne','ordinary.h5ad',{'use_rep':'provided','perplexity':8,'random_state':7,'use_fast_tsne':False}),
]:
    case(label,lambda label=label,name=name,source=source,kw=kw:matrix(label,EPISCANPY,name,root/source,kw))
for label,name,source,kw in [('diffmap','tl.diffmap','neighbors',{'n_comps':5}),('dpt','tl.dpt','diffmap',{'n_dcs':5}),('umap','tl.umap','neighbors',{'random_state':7,'maxiter':30})]:
    case(label,lambda label=label,name=name,source=source,kw=kw:matrix(label,SCANPY if name=='tl.diffmap' else EPISCANPY,name,outputs[source],kw))


def no_x():
    obj = ad.read_h5ad(root/'ordinary.h5ad'); obj.X=None; obj.write_h5ad(root/'no-x.h5ad')
    return matrix('embedding-no-X',EPISCANPY,'pp.neighbors',root/'no-x.h5ad',{'use_rep':'provided','n_neighbors':6})
case('embedding-no-X',no_x)
case('scanpy-scale',lambda:matrix('scanpy-scale',SCANPY,'pp.scale',root/'signed.h5ad',{'zero_center':False}))


def native_error():
    kwargs={'n_comps':-3,'random_state':7}
    try: EPISCANPY.execute(dict(input_path=str(root/'signed.h5ad'),output_dir=str(root/'native-error'),function_name='tl.pca',kwargs=kwargs))
    except NativeInvocationError as error:
        code="import anndata as ad,episcanpy as e\na=ad.read_h5ad("+repr(str(root/'signed.h5ad'))+")\ntry:e.tl.pca(a,n_comps=-3,random_state=7)\nexcept Exception as e:print(str(e))\n"
        script=root/'invalid-reference.py';script.write_text(code);run([str(rt/'episcanpy-0.4.0/env/bin/python'),'-I',str(script)],'invalid-reference')
        assert str(error) in (root/'invalid-reference.log').read_text()
        return dict(native_error=str(error),kwargs=kwargs)
    raise AssertionError('Native invalid parameter unexpectedly succeeded')
case('native-error',native_error)


def chromvar():
    out=chromvar_native.execute(dict(input_path=str(root/'signed.h5ad'),output_dir=str(root/'chromvar-signed'),function_name='getFragmentsPerPeak'))
    mmwrite(root/'signed.mtx',signed.X.T)
    script=root/'chromvar-reference.R';script.write_text("library(chromVAR);library(Matrix);x=as(readMM(commandArgs(TRUE)[1]),'CsparseMatrix');write.table(getFragmentsPerPeak(x),commandArgs(TRUE)[2],row.names=FALSE,col.names=FALSE)")
    ref=root/'chromvar-reference.tsv';run([str(rt/'chromvar/env/bin/Rscript'),'--vanilla',str(script),str(root/'signed.mtx'),str(ref)],'chromvar-reference')
    equal(pd.read_csv(out['table_path'],sep='\t')['value'],np.loadtxt(ref));return out
case('chromvar-signed',chromvar)


def macs():
    src=root/'ordinary.bdg';src.write_text('chr1\t0\t100\t-2\nchr1\t100\t200\t3\n')
    out=MACS3.execute(dict(input_path=str(src),output_dir=str(root/'macs-no-metadata'),function_name='bdgopt',kwargs={'method':'add','extra-param':-2}))
    ref=root/'macs-reference';ref.mkdir();run([str(rt/'macs3/env/bin/macs3'),'bdgopt','-i',str(src),'-m','add','-p','-2','-o','result.tsv','--outdir',str(ref)],'macs-reference')
    assert Path(out['table_path']).read_bytes()==(ref/'result.tsv').read_bytes();return out
case('macs-no-metadata',macs)


def cds_make():
    kwargs=dict(k=8,silent=True,return_agg_info=True,summary_stats=['depth'])
    args=dict(input_path=str(root/'ordinary.h5ad'),output_dir=str(root/'cds-make'),function_name='make_cicero_cds',reduced_coordinates_key='provided',seed=12,kwargs=kwargs)
    before=hashlib.sha256((root/'ordinary.h5ad').read_bytes()).hexdigest()
    out=cicero_cds.execute(args,'cds-make');outputs['cds']=Path(out['cds_path'])
    assert cicero_cds.recover(args,'cds-make')==out
    assert hashlib.sha256((root/'ordinary.h5ad').read_bytes()).hexdigest()==before
    mmwrite(root/'reference-counts.mtx',a.X.T)
    a.obs.to_csv(root/'reference-obs.tsv',sep='\t',index_label='id')
    a.var.to_csv(root/'reference-var.tsv',sep='\t',index_label='id')
    np.savetxt(root/'reference-coordinates.tsv',a.obsm['provided'],delimiter='\t')
    script=root/'cds-reference.R';script.write_text('''library(cicero);library(Matrix)
setwd(commandArgs(TRUE)[1])
x=as(readMM('reference-counts.mtx'),'CsparseMatrix')
pd=read.delim('reference-obs.tsv',row.names=1);fd=read.delim('reference-var.tsv',row.names=1)
dimnames(x)=list(rownames(fd),rownames(pd))
cds=newCellDataSet(x,phenoData=new('AnnotatedDataFrame',data=pd),featureData=new('AnnotatedDataFrame',data=fd),expressionFamily=VGAM::negbinomial.size(),lowerDetectionLimit=0)
coords=as.matrix(read.table('reference-coordinates.tsv'));rownames(coords)=colnames(cds)
set.seed(12);r=make_cicero_cds(cds,coords,k=8,silent=TRUE,return_agg_info=TRUE,summary_stats='depth')
writeMM(as(exprs(r[[1]]),'CsparseMatrix'),'reference-aggregated.mtx')
write.table(r[[2]],'reference-aggregation.tsv',sep='\\t',quote=FALSE,row.names=FALSE)
write.table(pData(r[[1]]),'reference-pdata.tsv',sep='\\t',quote=FALSE)
write.table(fData(r[[1]]),'reference-fdata.tsv',sep='\\t',quote=FALSE)
saveRDS(r[[1]],'reference-only.rds')
''')
    run([str(rt/'cicero-1/env/bin/Rscript'),'--vanilla',str(script),str(root)],'cds-reference')
    with zipfile.ZipFile(out['cds_path']) as z:
        with z.open('exprs.mtx') as stream:equal(mmread(stream),mmread(root/'reference-aggregated.mtx'))
        obj=json.loads(z.read('object.json'))
    assert Path(root/'cds-make/aggregation.tsv').read_bytes()==(root/'reference-aggregation.tsv').read_bytes()
    pd_ref=pd.read_csv(root/'reference-pdata.tsv',sep='\t',index_col=0)
    assert obj['observations']['ids']==pd_ref.index.tolist()
    for key in ('Size_Factor','mean_depth','num_genes_expressed'):
        equal(np.array(obj['observations']['columns'][key]['values'],dtype=float),pd_ref[key])
    return out
case('cds-make',cds_make)


def cds_later(name,kw):
    args=dict(input_path=str(outputs['cds']),output_dir=str(root/name),function_name=name,genomic_coords_path=str(root/'genome.tsv'),seed=7,kwargs=kw)
    out=cicero_cds.execute(args,'parity-'+name)
    script=root/(name+'-reference.R');script.write_text("library(cicero)\nsetwd(commandArgs(TRUE)[1]);cds=readRDS('reference-only.rds');g=read.table('genome.tsv');set.seed(7)\nr="+name+"(cds,genomic_coords=g,"+','.join(k+'='+str(v).upper() if isinstance(v,bool) else k+'='+str(v) for k,v in kw.items())+")\n"+("r=data.frame(distance_parameter=unlist(r))\n" if name=='estimate_distance_parameter' else '')+"write.table(r,'"+name+"-reference.tsv',sep='\\t',quote=FALSE,row.names=FALSE)\n")
    run([str(rt/'cicero-1/env/bin/Rscript'),'--vanilla',str(script),str(root)],name+'-reference')
    actual,ref=pd.read_csv(out['table_path'],sep='\t'),pd.read_csv(root/(name+'-reference.tsv'),sep='\t')
    assert list(actual)==list(ref) and len(actual)==len(ref)
    for key in ref:
        if ref[key].dtype.kind in 'biuf':equal(actual[key],ref[key])
        else:assert actual[key].tolist()==ref[key].tolist()
    return out
case('estimate_distance_parameter',lambda:cds_later('estimate_distance_parameter',dict(window=500000,sample_num=2,distance_constraint=100000,max_sample_windows=20)))
case('run_cicero',lambda:cds_later('run_cicero',dict(window=500000,sample_num=2,silent=True)))
print(json.dumps(dict(passed=sum(r['status']=='passed' for r in records),failed=sum(r['status']=='failed' for r in records))),flush=True)
if any(r['status']=='failed' for r in records):sys.exit(1)
