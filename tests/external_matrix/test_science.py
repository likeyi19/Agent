import hashlib
from pathlib import Path
import h5py
import numpy as np
import pytest
from agent.tools.data import _external_matrix_io as io, external_matrix_contract as e, scatac_matrix_contract as m
from agent.tools.data.cell_by_ccre_verifier import verify_cell_by_ccre


def build(case):
    return io.build(case,case['output_dir'])


def test_conservation(case):
    result=build(case)
    proof=verify_cell_by_ccre(result['manifest_path'],expected_sha256=result['manifest_sha256'])
    assert proof['nnz']==4 and proof['total_count']==6 and proof['zero_row_count']==1
    value=m.load_manifest_bytes(Path(result['manifest_path']).read_bytes())
    assert value['logical_matrix_sha256']==value['source_logical_matrix_sha256']
    assert 'upstream' not in value and 'diagnostic' not in value and 'ordered_selected_sha256' not in value
    import anndata as ad
    a=ad.read_h5ad(Path(result['manifest_path']).parent/'matrix.h5ad')
    assert list(a.obs_names)==['cell-z','cell-a','cell-empty'] and a.X.dtype==np.int64
    assert list(a.var_names)==['chr1:0-10','chr1:10-20','chr1:30-40']


@pytest.mark.parametrize('mutation',['reorder','missing','extra','duplicate_cell','empty_cell','negative','zero','float','nan','fraction','overflow','duplicate_entry','offset','out_of_bounds','species','assembly','sha','binary'])
def test_reject(case,mutation):
    with h5py.File(case['source_path'],'r+') as f:
        if mutation=='reorder': f['var/_index'][:]=['chr1:10-20','chr1:0-10','chr1:30-40']
        elif mutation in ('missing','extra'): f['X'].attrs['shape']=[3,2 if mutation=='missing' else 4]
        elif mutation=='duplicate_cell': f['obs/_index'][1]='cell-z'
        elif mutation=='empty_cell': f['obs/_index'][1]=''
        elif mutation in ('negative','zero','overflow'): f['X/data'][0]={'negative':-1,'zero':0,'overflow':2**63-1}[mutation]
        elif mutation in ('float','nan','fraction'):
            values=f['X/data'][:].astype(float); values[0]={'float':1.,'nan':float('nan'),'fraction':1.5}[mutation]
            del f['X/data']; f['X'].create_dataset('data',data=values)
        elif mutation=='duplicate_entry': f['X/indices'][1]=0
        elif mutation=='offset': f['X/indptr'][1]=5
        elif mutation=='out_of_bounds': f['X/indices'][1]=3
    case['source_sha256']=hashlib.sha256(Path(case['source_path']).read_bytes()).hexdigest()
    if mutation=='species': case['species']='mouse'
    if mutation=='assembly': case['assembly']='mm10'
    if mutation=='sha': case['source_sha256']='0'*64
    if mutation=='binary': case['matrix_semantics']='binary_accessibility'
    with pytest.raises(ValueError): build(case)


def test_marginal_preserving_forgery(case):
    result=build(case); path=Path(result['manifest_path']); payload=path.parent/'matrix.h5ad'
    with h5py.File(payload,'r+') as f: f['X/data'][:]=[2,1,1,2]
    value=m.load_manifest_bytes(path.read_bytes())
    forged=e.logical_identity(3,3,[([0,1],[2,1]),([0,1],[1,2]),([],[])],'fragment_counts')['logical_matrix_sha256']
    value['logical_matrix_sha256']=value['source_logical_matrix_sha256']=forged
    value['matrix']['sha256']=hashlib.sha256(payload.read_bytes()).hexdigest()
    value['matrix']['size_bytes']=payload.stat().st_size
    value['identity_sha256']=e.identity(value); raw=m.canonical(value); path.write_bytes(raw)
    with pytest.raises(ValueError,match='MATRIX_CONSERVATION_MISMATCH'):
        verify_cell_by_ccre(path,expected_sha256=hashlib.sha256(raw).hexdigest())


@pytest.mark.parametrize('semantics',['fragment_counts','insertion_counts','binary_accessibility'])
@pytest.mark.parametrize('empty',[False,True])
def test_declared_semantics_and_empty(case,semantics,empty):
    import anndata as ad
    a=ad.read_h5ad(case['source_path'])
    if empty: a=a[:0].copy()
    elif semantics=='binary_accessibility': a.X.data[:]=1
    a.write_h5ad(case['source_path'])
    case['source_sha256']=hashlib.sha256(Path(case['source_path']).read_bytes()).hexdigest()
    case['matrix_semantics']=semantics
    result=build(case)
    value=m.load_manifest_bytes(Path(result['manifest_path']).read_bytes())
    assert value['matrix_semantics']==semantics and value['shape']==[0 if empty else 3,3]
    assert value['readiness']==('no_source_cells' if empty else 'matrix_available')


@pytest.mark.parametrize('mutation',['reference_identity','coordinates','dense','csc','bool','external_link','metadata'])
def test_h5_boundary(case,mutation):
    import anndata as ad
    a=ad.read_h5ad(case['source_path'])
    if mutation=='reference_identity': a.uns['reference_identity_sha256']='0'*64
    elif mutation=='coordinates':
        a.var['chrom']=['chr1']*3; a.var['start']=[0,11,30]; a.var['end']=[10,20,40]
    elif mutation=='dense': a.X=a.X.toarray()
    elif mutation=='csc': a.X=a.X.tocsc()
    elif mutation=='bool': a.X=a.X.astype(bool)
    elif mutation=='metadata': a.obs['unreviewed']=['x']*3
    a.write_h5ad(case['source_path'])
    if mutation=='external_link':
        with h5py.File(case['source_path'],'r+') as f:
            del f['X/data']; f['X']['data']=h5py.ExternalLink('/nonexistent.h5','data')
    case['source_sha256']=hashlib.sha256(Path(case['source_path']).read_bytes()).hexdigest()
    with pytest.raises(ValueError): build(case)
