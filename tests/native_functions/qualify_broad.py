"""Opt-in installed-native parity, writes evidence only to an explicit external root.

PYTHONPATH=src python tests/native_functions/qualify_broad.py /external/evidence
Small synthetic count/embedding fixtures qualify numerical adapters, not biology.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmwrite
from agent.tools.analysis.native_matrix import EPISCANPY, SNAPATAC2, SIGNAC
from agent.tools.analysis.native_tables import MACS3, CICERO
from agent.tools.analysis.sequence_prediction import library

root = Path(sys.argv[1]).resolve(); root.mkdir(parents=True, exist_ok=True)
results = []; failures = []
source = root / 'synthetic-counts.h5ad'
rng = np.random.default_rng(17)
a = ad.AnnData(sparse.csr_matrix(rng.poisson(1.2, size=(48, 80)).astype(np.int32)),
    obs=pd.DataFrame(index=['cell' + str(i) for i in range(48)]),
    var=pd.DataFrame(index=['chr1_' + str(100*i+1) + '_' + str(100*i+80) for i in range(80)]))
a.uns['matrix_semantics'] = 'peak_counts'; a.layers['counts'] = a.X.copy()
a.obsm['provided'] = rng.normal(size=(48, 6)); a.write_h5ad(source)
env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
rt = library().root.parent / 'runtimes'


def run(command, label):
    with (root / (label + '.log')).open('w') as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)


def equal(a, b, tol=1e-9):
    if sparse.issparse(a) or sparse.issparse(b):
        a, b = sparse.csr_matrix(a), sparse.csr_matrix(b)
        assert a.shape == b.shape
        delta = a - b
        assert not delta.nnz or abs(delta).max() <= tol * max(1, abs(a).max())
    else:
        np.testing.assert_allclose(a, b, rtol=tol, atol=tol)


def case(label, package, function, src, kwargs, **extra):
    args = dict(input_path=str(src), output_dir=str(root / label), function_name=function, kwargs=kwargs, **extra)
    before = hashlib.sha256(src.read_bytes()).hexdigest()
    try:
        output = package.execute(args, 'qualification-' + label)
        assert package.recover(args, 'qualification-' + label) == output
        assert hashlib.sha256(src.read_bytes()).hexdigest() == before
        if package in (EPISCANPY, SNAPATAC2):
            reference = root / (label + '-reference.h5ad')
            code = '''import anndata as ad,json,sys
import episcanpy as epi
import snapatac2 as snap
'''.replace('import episcanpy as epi\n' if package is SNAPATAC2 else 'import snapatac2 as snap\n', '') + '''a=ad.read_h5ad(sys.argv[1]); kw=json.loads(sys.argv[3])
r=CALL(a,**kw)
if r is not None:
 if NAME=='tl.umap': a.obsm['X_'+kw.get('key_added','umap')]=r
 elif NAME=='pp.knn': a.obsp['distances']=r
 elif NAME=='tl.leiden': a.obs[kw.get('key_added','leiden')]=r
 elif isinstance(r,ad.AnnData): a=r
a.write_h5ad(sys.argv[2])
'''.replace('CALL', ('epi.' if package is EPISCANPY else 'snap.') + function).replace('NAME', repr(function))
            script = root / (label + '-reference.py'); script.write_text(code)
            run([str(rt / package.runtime / 'env/bin/python'), '-I', str(script), str(src), str(reference), json.dumps(kwargs)], label + '-reference')
            actual, ref = ad.read_h5ad(output['matrix_path']), ad.read_h5ad(reference)
            equal(actual.X, ref.X)
            for container in ('layers', 'obsm', 'obsp'):
                for key, value in getattr(ref, container).items():
                    equal(getattr(actual, container)[key], value, 1e-6)
            for key in ref.obs:
                assert actual.obs[key].tolist() == ref.obs[key].tolist()
            assert actual.obs_names.tolist() == ref.obs_names.tolist() and actual.var_names.tolist() == ref.var_names.tolist()
        elif package is SIGNAC:
            inp = ad.read_h5ad(src); mmwrite(root / (label + '.mtx'), inp.X.T)
            code = '''library(Signac);library(Matrix)
x=as(readMM(commandArgs(TRUE)[1]),'CsparseMatrix')
r=RunSVD(x,n=5,scale.embeddings=FALSE,verbose=FALSE)
jsonlite::write_json(list(d=r@misc$d,embeddings=unname(r@cell.embeddings),loadings=unname(r@feature.loadings)),commandArgs(TRUE)[2],digits=NA)
'''
            script = root / (label + '-reference.R'); script.write_text(code); ref = root / (label + '-reference.json')
            run([str(rt / 'signac-1/env/bin/Rscript'), '--vanilla', str(script), str(root / (label + '.mtx')), str(ref)], label + '-reference')
            ref = json.loads(ref.read_text()); actual = ad.read_h5ad(output['matrix_path'])
            equal(actual.uns['Signac_svd']['misc']['d'], ref['d'], 1e-5)
            u, v = actual.obsm['X_lsi'], np.array(ref['embeddings'])
            equal(u @ u.T, v @ v.T, 1e-4)
        elif package is MACS3:
            outdir = root / (label + '-reference'); outdir.mkdir()
            command = [str(rt / 'macs3/env/bin/macs3'), function]
            command += ['-t', str(src), '-c', extra['control_path']] if function == 'bdgcmp' else ['-i', str(src)]
            for k, v in kwargs.items():
                if k == 'no-trackline':
                    if v: command.append('--no-trackline')
                else:
                    command.append('--' + k); command.extend(str(x) for x in (v if isinstance(v, list) else [v]))
            command += ['-o', 'result.tsv', '--outdir', str(outdir)]
            run(command, label + '-reference')
            assert Path(output['table_path']).read_bytes() == (outdir / 'result.tsv').read_bytes()
        elif package is CICERO:
            target = root / (label + '-reference.tsv'); script = root / (label + '-reference.R')
            code = '''library(cicero)
x=read.delim(commandArgs(TRUE)[1],quote='',check.names=FALSE,colClasses='character',na.strings=NULL)
if ('coaccess' %in% names(x)) x$coaccess=as.numeric(x$coaccess)
set.seed(17)
y=do.call(cicero::FUNCTION,c(list(x),jsonlite::fromJSON(commandArgs(TRUE)[3])))
write.table(y,commandArgs(TRUE)[2],sep='\\t',quote=FALSE,row.names=FALSE)
'''.replace('FUNCTION', function)
            script.write_text(code)
            run([str(rt / 'cicero-1/env/bin/Rscript'), '--vanilla', str(script), str(src), str(target), json.dumps(kwargs)], label + '-reference')
            assert Path(output['table_path']).read_bytes() == target.read_bytes()
        results.append(dict(label=label, function=output['function_name'], args=args, result=output, independent_parity=True, source_preserved=True, exact_recovery=True))
        print(label, 'PASS', flush=True)
        return Path(output.get('matrix_path') or output['table_path'])
    except Exception as exc:
        import traceback
        failures.append(dict(label=label, error=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
        print(label, 'FAIL', repr(exc), flush=True)
    finally:
        (root / 'parity.json').write_text(json.dumps(dict(results=results, failures=failures), indent=2))


norm = case('epi-normalize', EPISCANPY, 'pp.normalize_total', source, {'target_sum':1000, 'key_added':'size', 'layers':['counts'], 'layer_norm':'after'})
case('epi-normalize-return', EPISCANPY, 'pp.normalize_total', source, {'inplace':False})
case('signac-svd', SIGNAC, 'RunSVD', source, {'n':5, 'scale.embeddings':False, 'verbose':False})
for inplace in (True, False):
    case('snap-umap-' + str(inplace), SNAPATAC2, 'tl.umap', source, dict(use_rep='provided', n_neighbors=5, n_epochs=30, init='random', random_state=17, inplace=inplace))
    graph = case('snap-knn-' + str(inplace), SNAPATAC2, 'pp.knn', source, dict(use_rep='provided', n_neighbors=5, inplace=inplace))
    if graph:
        case('snap-leiden-' + str(inplace), SNAPATAC2, 'tl.leiden', graph, dict(random_state=17, min_cluster_size=2, inplace=inplace))
if norm:
    case('cross-epi-signac', SIGNAC, 'RunSVD', norm, {'n':5, 'scale.embeddings':False, 'verbose':False})
signal = root / 'signal.bedGraph'; signal.write_text('chr1\t0\t100\t1\nchr1\t100\t500\t10\nchr1\t500\t800\t2\nchr1\t800\t1000\t8\nchr1\t1000\t1200\t0\n')
control = root / 'control.bedGraph'; control.write_text('chr1\t0\t600\t1\nchr1\t600\t1200\t2\n')
opt = case('macs-opt', MACS3, 'bdgopt', signal, {'method':'multiply', 'extra-param':2}, signal_semantics='count_pileup')
case('macs-compare', MACS3, 'bdgcmp', signal, {'method':'FE', 'pseudocount':1}, signal_semantics='count_pileup', control_path=str(control), control_semantics='expected_count', assembly='synthetic_chr1')
case('macs-peaks', MACS3, 'bdgpeakcall', opt or signal, {'cutoff':5, 'min-length':50, 'max-gap':30, 'no-trackline':True}, signal_semantics='scaled_pileup')
case('macs-broad', MACS3, 'bdgbroadcall', signal, {'cutoff-peak':5, 'cutoff-link':1, 'min-length':50, 'lvl1-max-gap':30, 'lvl2-max-gap':500, 'no-trackline':True}, signal_semantics='count_pileup')
connections = root / 'connections.tsv'; connections.write_text('Peak1\tPeak2\tcoaccess\nchr1_1_100\tchr1_80_200\t0.9\nchr1_1_100\tchr1_180_300\t0.8\nchr1_80_200\tchr1_180_300\t0.7\nchr1_101_150\tchr1_140_170\t0.9\nchr1_101_150\tchr1_160_190\t0.8\nchr1_140_170\tchr1_160_190\t0.7\n')
ccans = case('cicero-ccans', CICERO, 'generate_ccans', connections, {'coaccess_cutoff_override':.2}, seed=17)
if ccans:
    case('cicero-overlap', CICERO, 'find_overlapping_ccans', ccans, {'min_overlap':1})
print('PASS',len(results),'FAIL',len(failures),flush=True)
sys.exit(bool(failures))
