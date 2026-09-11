"""Rehashed scientific forgeries, not merely stale checksum detection."""
import copy
import json
from pathlib import Path
import shutil

import h5py
import numpy as np
import pytest
from scipy import sparse

from agent.tools.data.scatac_cell_by_ccre import build_cell_by_ccre
from agent.tools.data.cell_by_ccre_verifier import verify_cell_by_ccre
from agent.tools.data import scatac_matrix_contract as m, _matrix_bedtools as bed


def replace_x(f, array, index_dtype='int64', compression=None):
    x=sparse.csr_matrix(array,dtype='int64')
    for key,values in (('data',x.data),('indices',x.indices.astype(index_dtype)),('indptr',x.indptr.astype(index_dtype))):
        del f['X'][key];f['X'].create_dataset(key,data=values,compression=compression)
    return x


def rehash(path, value):
    payload=path.parent/'matrix.h5ad'
    value['matrix'].update(sha256=bed.file_sha(payload),size_bytes=payload.stat().st_size)
    value['identity_sha256']=m.manifest_identity(value)
    path.write_bytes(m.canonical(value))
    return bed.file_sha(path)


def test_complete_rehashed_forgery_rejection(fixture_factory,tmp_path):
    result=build_cell_by_ccre(**fixture_factory())
    original=Path(result['manifest_path']); manifest=json.loads(original.read_bytes())
    baseline=np.array([[1,2,3,2,0],[1,1,2,1,0],[0,0,0,0,0]],dtype='int64')
    cases=['missing','false','duplicate','row_sum','column_sum','value','rows','columns',
           'upstream','profile','physical','zero','offset','unsorted','duplicate_column','negative','shape']
    for case in cases:
        target=tmp_path/f'forged-{case}';target.mkdir()
        payload=target/'matrix.h5ad';shutil.copyfile(original.parent/'matrix.h5ad',payload)
        value=copy.deepcopy(manifest); array=baseline.copy()
        with h5py.File(payload,'r+') as f:
            if case=='missing':array[0,1]-=1
            elif case=='false':array[2,4]=1
            elif case=='duplicate':array[0,1]+=1
            elif case=='row_sum':array[0,1]-=1;array[0,2]+=1
            elif case=='column_sum':array[0,1]-=1;array[1,1]+=1
            elif case=='value':array[1,1]+=7
            elif case=='rows':
                array=array[[1,0,2]]
                for key in ('_index','namespace','barcode_identifier'):
                    data=f['obs'][key][:];f['obs'][key][:]=data[[1,0,2]]
            elif case=='columns':
                array=array[:,[1,0,2,3,4]]
                for key in ('_index','chrom','start','end'):
                    data=f['var'][key][:];f['var'][key][:]=data[[1,0,2,3,4]]
            if case in cases[:8]:
                x=replace_x(f,array);value.update(m.csr_identity(x));f['uns/logical_matrix_sha256'][()]=value['logical_matrix_sha256']
            elif case=='upstream':value['upstream']['fragments']['identity_sha256']='a'*64
            elif case=='profile':value['profile']['support']='weighted'
            elif case=='physical':f['X/data'][0]=23
            elif case=='zero':f['X/data'][0]=0
            elif case=='negative':f['X/data'][0]=-1
            elif case=='offset':f['X/indptr'][1]=1000000
            elif case=='unsorted':f['X/indices'][:2]=[1,0]
            elif case=='duplicate_column':f['X/indices'][:2]=[0,0]
            elif case=='shape':f['X'].attrs['shape']=[3,6]
        path=target/'manifest.json'
        digest=rehash(path,value) if case!='physical' else result['manifest_sha256']
        if case=='physical':path.write_bytes(original.read_bytes())
        with pytest.raises(m.ScATACMatrixError):
            verify_cell_by_ccre(path,expected_sha256=digest,bedtools_path='/usr/bin/bedtools')


def test_hdf5_layout_and_index_width_do_not_define_logical_identity(fixture_factory):
    result=build_cell_by_ccre(**fixture_factory());path=Path(result['manifest_path']);value=json.loads(path.read_bytes())
    baseline=np.array([[1,2,3,2,0],[1,1,2,1,0],[0,0,0,0,0]],dtype='int64')
    for dtype,compression in [('int32',None),('int64','gzip')]:
        with h5py.File(path.parent/'matrix.h5ad','r+') as f:replace_x(f,baseline,dtype,compression)
        digest=rehash(path,value)
        verified=verify_cell_by_ccre(path,expected_sha256=digest,bedtools_path='/usr/bin/bedtools')
        assert verified['logical_matrix_sha256']==result['logical_matrix_sha256']


def test_verifier_never_calls_production(fixture_factory,monkeypatch):
    from agent.tools.data import _cell_by_ccre_production as production
    result=build_cell_by_ccre(**fixture_factory())
    for name in ('intersect','incidence','project','construct_counts','rows'):
        monkeypatch.setattr(production,name,lambda *a:pytest.fail('Verifier reused production'))
    verify_cell_by_ccre(result['manifest_path'],expected_sha256=result['manifest_sha256'],bedtools_path='/usr/bin/bedtools')
