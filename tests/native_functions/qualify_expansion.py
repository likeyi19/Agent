"""Opt-in STE-NFI-1F installed-native parity; write evidence outside Git.

PYTHONPATH=src python tests/native_functions/qualify_expansion.py /absolute/new/evidence
"""
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
from agent.tools.analysis.native_matrix import SCANPY, EPISCANPY, SNAPATAC2, SIGNAC
from agent.tools.analysis import chromvar_native
from scipy.io import mmwrite

root = Path(sys.argv[1]).resolve(); root.mkdir(parents=True, exist_ok=False)
rt = Path(os.environ.get('AGENT_NATIVE_RUNTIME_ROOT', '/home/likeyi/program/agent-data/resources/runtimes'))
for key, rel in [('SCANPY_PYTHON','episcanpy-0.4.0/env/bin/python'),('EPISCANPY_PYTHON','episcanpy-0.4.0/env/bin/python'),('SNAPATAC2_PYTHON','native-matrix-1/env/bin/python'),('SIGNAC_RSCRIPT','signac-1/env/bin/Rscript'),('CHROMVAR_RSCRIPT','chromvar/env/bin/Rscript'),('MACS3_EXECUTABLE','macs3/env/bin/macs3')]:
    os.environ['AGENT_'+key] = str(rt/rel)
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', NUMBA_NUM_THREADS='1')
records=[]; outputs={}
rng=np.random.default_rng(71)
x=rng.poisson(1.5,(90,600)).astype(float);x[:30,:20]+=5;x[30:60,20:40]+=5
obj=ad.AnnData(sparse.csr_matrix(x),obs=pd.DataFrame({'group':pd.Categorical(np.repeat(['a','b','c'],30)),'batch':pd.Categorical(np.tile(['one','two'],45))},index=['cell_'+str(i) for i in range(90)]),var=pd.DataFrame(index=['chr1:'+str(i*100)+'-'+str(i*100+90) for i in range(600)]))
obj.obsm['provided']=np.concatenate([rng.normal(i*10,.2,(30,5)) for i in range(3)])
obj.obsm['X_pca']=obj.obsm['provided'].copy();obj.obsm['X_spectral']=obj.obsm['provided'].copy()
obj.uns['iroot']=0;obj.uns['omic']='ATAC';obj.uns['matrix_semantics']='simulated_peak_counts';obj.raw=obj.copy()
obj.write_h5ad(root/'input.h5ad')

def equal(a,b):
    if sparse.issparse(a): a=a.toarray()
    if sparse.issparse(b): b=b.toarray()
    if isinstance(a,pd.DataFrame): pd.testing.assert_frame_equal(a,b,check_exact=False,rtol=2e-6,atol=2e-7);return
    if isinstance(a,dict):
        assert set(a)==set(b)
        for k in a: equal(a[k],b[k])
    elif isinstance(a,np.ndarray) and a.dtype.names:
        assert a.dtype.names==b.dtype.names
        for k in a.dtype.names:equal(a[k],b[k])
    elif a is None:assert b is None
    elif np.asarray(a).dtype.kind in 'biufc':np.testing.assert_allclose(a,b,rtol=2e-6,atol=2e-7,equal_nan=True)
    else:np.testing.assert_array_equal(a,b)

def run(cmd,label):
    with (root/(label+'.log')).open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=os.environ.copy(),check=True,timeout=600)

def case(label,fn):
    if len(sys.argv)>2 and label not in sys.argv[2:]: return
    try:detail=fn();records.append(dict(case=label,status='passed',detail=detail))
    except Exception as e:traceback.print_exc();records.append(dict(case=label,status='failed',error=str(e)))
    (root/'results.json').write_text(json.dumps(records,indent=2));print(label,records[-1]['status'],flush=True)

reference_script=root/'direct.py'
reference_script.write_text('''import sys,json,anndata as ad,numpy as np,pandas as pd
import importlib
p=importlib.import_module(sys.argv[1]); module,name=sys.argv[2].split('.')
a=ad.read_h5ad(sys.argv[3]);kw=json.loads(sys.argv[5]);r=getattr(getattr(p,module),name)(a,**kw)
if isinstance(r,ad.AnnData):a=r
elif r is not None:
 if isinstance(r,tuple): r={str(i):v for i,v in enumerate(r)}
 elif type(r).__module__.startswith('polars.'):r=r.to_pandas()
 elif isinstance(r,pd.DataFrame) and 'mean_bin' in r and isinstance(r['mean_bin'].dtype,pd.CategoricalDtype):
  v=r['mean_bin'];r=dict(kind='dataframe_with_interval_categories',column_order=r.columns.to_numpy(),frame=r.drop(columns='mean_bin'),interval_columns={'mean_bin':dict(codes=v.cat.codes.to_numpy(),left=v.cat.categories.left.to_numpy(),right=v.cat.categories.right.to_numpy(),closed=v.cat.categories.closed,ordered=v.cat.ordered)})
 elif isinstance(r,dict):r={k:(v.to_numpy() if isinstance(v,pd.Index) else v) for k,v in r.items()}
 a.uns['expected_return']=r
a.write_h5ad(sys.argv[4])
''')

def matrix(label,owner,name,kw,source=None):
    source=source or root/'input.h5ad';before=hashlib.sha256(source.read_bytes()).hexdigest()
    args=dict(input_path=str(source),output_dir=str(root/label),function_name=name,kwargs=kw)
    result=owner.execute(args,'parity-'+label);outputs[label]=Path(result['matrix_path'])
    assert owner.recover(args,'parity-'+label)==result
    assert hashlib.sha256(source.read_bytes()).hexdigest()==before
    ref=root/(label+'-direct.h5ad')
    run([str(rt/owner.runtime/'env/bin/python'),'-I',str(reference_script),owner.package,name,str(source),str(ref),json.dumps(kw)],label+'-direct')
    actual,expected=ad.read_h5ad(result['matrix_path']),ad.read_h5ad(ref)
    assert actual.obs_names.tolist()==expected.obs_names.tolist() and actual.var_names.tolist()==expected.var_names.tolist()
    equal(actual.X,expected.X)
    pd.testing.assert_frame_equal(actual.obs,expected.obs);pd.testing.assert_frame_equal(actual.var,expected.var)
    for group in ('layers','obsm','varm','obsp','varp'):
        for k,v in getattr(expected,group).items():equal(getattr(actual,group)[k],v)
    for k,v in expected.uns.items():
        if k=='expected_return':equal(actual.uns['agent_native_return']['value'],v)
        elif k not in ('matrix_semantics','agent_native_invocation'):equal(actual.uns[k],v)
    return {'function':result['function_name'],'kwargs':kw,'output_type':result['native_output_type'],'slots':result['result_slots'],'source_sha256':before,'result_sha256':result['matrix_sha256']}

for label,name,kw in [
 ('qc-default','pp.calculate_qc_metrics',{}),('qc-inplace','pp.calculate_qc_metrics',{'inplace':True,'var_type':'peaks'}),
 ('normalize','pp.normalize_total',{'target_sum':1000}),('normalize-return','pp.normalize_total',{'inplace':False}),
 ('log-copy','pp.log1p',{'copy':True}),('pca','pp.pca',{'n_comps':8,'random_state':7}),
 ('neighbors','pp.neighbors',{'use_rep':'provided','n_neighbors':10,'random_state':7}),
 ('tsne','tl.tsne',{'use_rep':'provided','perplexity':10,'random_state':7}),
 ('rank','tl.rank_genes_groups',{'groupby':'group','method':'t-test','use_raw':False,'pts':True}),
 ('score','tl.score_genes',{'gene_list':obj.var_names[:12].tolist(),'use_raw':False,'random_state':7}),
]:case(label,lambda label=label,name=name,kw=kw:matrix(label,SCANPY,name,kw))
for label,kw in [('hvg',{'n_top_genes':40}),('hvg-return',{'n_top_genes':40,'inplace':False}),('hvg-subset',{'n_top_genes':40,'subset':True})]:
    case(label,lambda label=label,kw=kw:matrix(label,SCANPY,'pp.highly_variable_genes',kw,outputs.get('log-copy')))
for label,name,kw,src in [
 ('leiden','tl.leiden',{'random_state':7},'neighbors'),('umap','tl.umap',{'random_state':7,'maxiter':30},'neighbors'),
 ('paga','tl.paga',{'groups':'group'},'neighbors'),('diffmap','tl.diffmap',{'n_comps':8},'neighbors'),('dpt','tl.dpt',{'n_dcs':8},'diffmap')]:
    case(label,lambda label=label,name=name,kw=kw,src=src:matrix(label,SCANPY,name,kw,outputs[src]))
for name,kw,src in [('tl.leiden',{'random_state':7},'neighbors'),('tl.louvain',{'random_state':7},'neighbors'),('tl.kmeans',{'num_clusters':3},'pca'),('tl.hc',{'num_clusters':3},'pca'),('tl.rank_features',{'groupby':'group','use_raw':False,'n_features':12},'log-copy')]:
    label='epi-'+name.split('.')[1];case(label,lambda label=label,name=name,kw=kw,src=src:matrix(label,EPISCANPY,name,kw,outputs[src]))
for name,kw in [('tl.dbscan',{'eps':2}),('pp.mnc_correct',{'batch':'batch','n_clusters':3,'n_neighbors':2}),('tl.marker_regions',{'groupby':'group','pvalue':.5}),('tl.diff_test',{'cell_group1':list(range(30)),'cell_group2':list(range(30,60)),'features':list(range(20))})]:
    label='snap-'+name.split('.')[1]+('-return' if kw.get('inplace') is False else '')
    case(label,lambda label=label,name=name,kw=kw:matrix(label,SNAPATAC2,name,kw))

# Changed non-inplace mapping of older functions needs its own native checks.
for label,name,kw in [
 ('snap-select-return','pp.select_features',{'inplace':False,'n_features':40}),
 ('snap-spectral-return','tl.spectral',{'inplace':False,'features':None,'n_comps':4,'random_state':7}),
 ('snap-knn-return','pp.knn',{'inplace':False,'use_rep':'provided','n_neighbors':8}),
 ('snap-umap-return','tl.umap',{'inplace':False,'use_rep':'provided','random_state':7,'n_epochs':20}),
]:case(label,lambda label=label,name=name,kw=kw:matrix(label,SNAPATAC2,name,kw))

# R statistics are compared with independently constructed native objects.
mmwrite(root/'counts.mtx',obj.X.T)
dev=obj[:,:30].copy();dev.X=rng.normal(size=dev.shape);dev.layers['z_scores']=dev.X*3;dev.raw=None;dev.write_h5ad(root/'deviations.h5ad')
mmwrite(root/'deviations.mtx',sparse.csc_matrix(dev.X.T));mmwrite(root/'z.mtx',sparse.csc_matrix(dev.layers['z_scores'].T))

def r_case(name,kw,group=False):
    isdev=name!='getTotalFragments';source=root/('deviations.h5ad' if isdev else 'input.h5ad')
    label='chromvar-'+name;args=dict(input_path=str(source),output_dir=str(root/label),function_name=name,kwargs=kw)
    if group:args['group_key']='group'
    result=chromvar_native.execute(args,'parity-'+label);assert chromvar_native.recover(args,'parity-'+label)==result
    script=root/(label+'.R');out=root/(label+'-direct.tsv')
    script.write_text("library(chromVAR);library(Matrix);library(SummarizedExperiment)\na=commandArgs(TRUE);x=as(readMM(a[1]),'CsparseMatrix');colnames(x)=paste0('cell_',0:89)\n"+("z=as.matrix(readMM(a[2]));dimnames(z)=dimnames(x);x=new('chromVARDeviations',SummarizedExperiment(list(deviations=as.matrix(x),z=z)))\n" if isdev else '')+"r=do.call(chromVAR::"+name+",c(list(object=x),jsonlite::fromJSON(a[4]),"+("list(groups=rep(c('a','b','c'),each=30))" if group else 'list()')+"));if(inherits(r,'dist'))r=as.matrix(r);write.table(r,a[3],sep='\\t',row.names=FALSE,col.names=FALSE,quote=FALSE)\n")
    run([str(rt/'chromvar/env/bin/Rscript'),'--vanilla',str(script),str(root/('deviations.mtx' if isdev else 'counts.mtx')),str(root/'z.mtx'),str(out),json.dumps(kw)],label+'-direct')
    actual=pd.read_csv(result['table_path'],sep='\t');actual=actual.drop(columns=['source_index','source_id'],errors='ignore').to_numpy()
    expected=np.loadtxt(out,ndmin=2);equal(actual,expected)
    return result
for name,kw,group in [('getTotalFragments',{},False),('getSampleCorrelation',{'threshold':0},False),('getSampleDistance',{'threshold':0,'initial_dims':5},False),('differentialDeviations',{},True),('differentialVariability',{},True)]:case(name,lambda name=name,kw=kw,group=group:r_case(name,kw,group))

def signac():
    result=SIGNAC.execute(dict(input_path=str(root/'input.h5ad'),output_dir=str(root/'corSparse'),function_name='corSparse'))
    script=root/'corSparse.R';script.write_text("library(Signac);library(Matrix);a=commandArgs(TRUE);x=as(readMM(a[1]),'CsparseMatrix');write.table(corSparse(x),a[2],row.names=FALSE,col.names=FALSE,sep='\\t')")
    out=root/'corSparse-direct.tsv';run([str(rt/'signac-1/env/bin/Rscript'),'--vanilla',str(script),str(root/'counts.mtx'),str(out)],'corSparse-direct')
    equal(ad.read_h5ad(result['matrix_path']).obsp['Signac_corSparse'],np.loadtxt(out));return result
case('corSparse',signac)
print(json.dumps({'passed':sum(r['status']=='passed' for r in records),'failed':sum(r['status']=='failed' for r in records)}))
sys.exit(any(r['status']=='failed' for r in records))
