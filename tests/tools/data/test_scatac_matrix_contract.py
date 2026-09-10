from dataclasses import asdict, FrozenInstanceError
import hashlib

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from agent.tools.data import scatac_matrix_contract as m
from agent.tools.data.scatac_selection_profile import encode_cell_id
from agent.tools.analysis.replicate_pseudobulk import _matrix_digest_and_nnz


@pytest.mark.parametrize('start,end,expected', [(0,10,False),(9,11,True),(12,15,True),
    (5,25,True),(10,20,True),(19,21,True),(20,25,False)])
def test_half_open(start, end, expected):
    assert m.overlaps(('chr1',start,end),('chr1',10,20)) is expected
    assert not m.overlaps(('chr2',start,end),('chr1',10,20))


def test_profile_frozen():
    with pytest.raises(FrozenInstanceError):
        m.PROFILE.support = 'weighted'
    assert m.PROFILE_SHA256 == hashlib.sha256(m.canonical(asdict(m.PROFILE))).hexdigest()
    assert m.PROFILE.profile_id == 'canonical-fragment-record-overlap-counts.v1'
    assert m.PROFILE_SHA256 == 'fb7d8f48728e6b71a833fd7024de9d05e0782ea4096d3b31f1aa861245b76394'


@pytest.mark.parametrize('index_dtype', [np.int32, np.int64, np.uint64])
def test_exact_m8_digest_and_physical_index_independence(index_dtype):
    matrix = csr_matrix(np.array([[0,2,1],[0,0,0],[4,0,0]],dtype=np.int64))
    expected, nnz = _matrix_digest_and_nnz(matrix,3,3,'fragment_counts')
    matrix.indices = matrix.indices.astype(index_dtype)
    matrix.indptr = matrix.indptr.astype(index_dtype)
    assert m.csr_identity(matrix) == dict(logical_matrix_sha256=expected, nnz=nnz,
                                         total_count=7,zero_row_count=1)


def test_empty_digest_m8_encoding():
    matrix = csr_matrix((0,3),dtype=np.int64)
    digest, _ = _matrix_digest_and_nnz(matrix,0,3,'fragment_counts')
    assert m.csr_identity(matrix)['logical_matrix_sha256'] == digest
    assert m.logical_matrix_identity(1,3,[([],[])])['logical_matrix_sha256'] != digest


@pytest.mark.parametrize('columns,values', [([1,1],[1,2]),([2,1],[1,2]),([3],[1]),([-1],[1]),
    ([0],[0]),([0],[-1]),([0],[True]),([0],[1.0]),([0],[2**63]),([0,1],[m.INT64_MAX,1])])
def test_bad_canonical_rows(columns,values):
    with pytest.raises(m.ScATACMatrixError):m.logical_matrix_identity(1,3,[(columns,values)])


@pytest.mark.parametrize('mode', ['float','bad_indptr','bad_length','float_index','explicit_zero','duplicate'])
def test_csr_rejects_without_repair(mode):
    a=csr_matrix(np.array([[1,2]],dtype=np.int64))
    if mode=='float':a=a.astype(float)
    if mode=='bad_indptr':a.indptr[0]=1
    if mode=='bad_length':a.indptr[-1]=1
    if mode=='float_index':a.indices=a.indices.astype(float)
    if mode=='explicit_zero':a.data[0]=0
    if mode=='duplicate':a.indices[1]=a.indices[0]
    with pytest.raises(m.ScATACMatrixError):m.csr_identity(a)


def test_overflow_and_row_count():
    for rows,n in [([([0],[m.INT64_MAX]),([0],[1])],2),([],1),([([],[])],0)]:
        with pytest.raises(m.ScATACMatrixError):m.logical_matrix_identity(n,3,rows)
    assert m.checked_add(m.INT64_MAX-1,1)==m.INT64_MAX


def test_row_authority_namespace_and_empty():
    rows=[(i,ns,bc,encode_cell_id(ns,bc)) for i,(ns,bc) in enumerate([('z','AA-1'),('a','AA-1')])]
    count,digest=m.selected_axis_identity(rows)
    assert count==2
    assert digest==hashlib.sha256(b'agent.ordered-selected-cells.v1\0z\tAA-1\na\tAA-1\n').hexdigest()
    # No sorting: even given nonlexical authority, preserve exact caller order.
    with pytest.raises(m.ScATACMatrixError):m.selected_axis_identity(rows[::-1])
    with pytest.raises(m.ScATACMatrixError):m.selected_axis_identity([rows[0],(1,*rows[0][1:])])
    assert m.selected_axis_identity([])==(0,hashlib.sha256(b'agent.ordered-selected-cells.v1\0').hexdigest())


def manifest(empty=False):
    n=0 if empty else 2
    stats=m.logical_matrix_identity(n,3,[] if empty else [([1],[2]),([],[])])
    value=dict(artifact_type=m.ARTIFACT,schema_version=1,contract_version=m.CONTRACT,
        profile=asdict(m.PROFILE),profile_sha256=m.PROFILE_SHA256,species='mouse',assembly='mm10',
        upstream={k:dict(manifest_path='/explicit/'+k+'.json',manifest_sha256='a'*64,
            identity_sha256='b'*64,contract_version=c) for k,c in zip(('fragments','selection','reference'),
            ('scatac-fragments.v2','scatac-cell-selection.v1','scatac-reference-bundle.v1'))},
        ordered_selected_sha256=m.selected_axis_identity([])[1] if empty else 'c'*64,
        ordered_feature_sha256='d'*64,shape=[n,3],**stats,
        matrix=dict(path='matrix.h5ad',sha256='e'*64,size_bytes=100,format='csr',dtype='int64'),
        backend=dict(profile_id='bedtools-ccre-record-incidence.v1',runtime_sha256='f'*64),
        diagnostic=m.overlap_diagnostic(0,0) if empty else m.overlap_diagnostic(4,2),
        readiness='no_selected_cells' if empty else 'matrix_available')
    value['identity_sha256']=m.manifest_identity(value)
    return value


@pytest.mark.parametrize('empty',[False,True])
def test_closed_manifest_roundtrip_no_io(empty,monkeypatch):
    import builtins
    value=manifest(empty)
    monkeypatch.setattr(builtins,'open',lambda *a,**k:pytest.fail('contract performed IO'))
    assert m.load_manifest_bytes(m.canonical(value))==value


@pytest.mark.parametrize('key,value',[('schema_version',True),('artifact_type','other'),('profile_sha256','0'*64),
    ('nnz',True),('nnz',7),('total_count',0),('zero_row_count',3),('shape',[2,0]),('readiness','epizoo_ready'),
    ('assembly','hg38'),('ordered_selected_sha256','invalid'),('unexpected',1)])
def test_rehashed_invalid_manifest(key,value):
    v=manifest();v[key]=value;v['identity_sha256']=m.manifest_identity(v)
    with pytest.raises(m.ScATACMatrixError):m.validate_manifest(v)


@pytest.mark.parametrize('change',['weighted','old_fragments','path','dtype','backend','fraction','zero_reason'])
def test_nested_contract_mutations(change):
    v=manifest()
    if change=='weighted':v['profile']['support']='weighted'
    if change=='old_fragments':v['upstream']['fragments']['contract_version']='scatac-fragments.v1'
    if change=='path':v['matrix']['path']='../matrix.h5ad'
    if change=='dtype':v['matrix']['dtype']='float32'
    if change=='backend':v['backend']['profile_id']='bedtools-qc-point-incidence.v1'
    if change=='fraction':v['diagnostic']['fraction_selected_fragment_records_overlapping_any_ccre']={'numerator':2,'denominator':4}
    if change=='zero_reason':v['diagnostic']['undefined_reason']='other'
    v['identity_sha256']=m.manifest_identity(v)
    with pytest.raises(m.ScATACMatrixError):m.validate_manifest(v)


@pytest.mark.parametrize('raw',[b'{}',b'{"x":1,"x":2}',b'{"x":NaN}',b'\xff',b'[]'])
def test_strict_json(raw):
    with pytest.raises(m.ScATACMatrixError):m.load_manifest_bytes(raw)


def test_diagnostic_once_per_record_and_empty():
    assert m.overlap_diagnostic(3,2)['fraction_selected_fragment_records_overlapping_any_ccre']==dict(numerator=2,denominator=3)
    assert m.overlap_diagnostic(0,0)['undefined_reason']=='NO_SELECTED_FRAGMENT_RECORDS'
    with pytest.raises(m.ScATACMatrixError):m.overlap_diagnostic(2,3)


def test_redistribution_changes_identity_even_equal_marginals():
    # Full reconstruction in M11.5b must distinguish these equal row/column sums.
    a=m.logical_matrix_identity(2,2,[([0],[2]),([1],[2])])
    b=m.logical_matrix_identity(2,2,[([1],[2]),([0],[2])])
    assert a['total_count']==b['total_count']
    assert a['logical_matrix_sha256']!=b['logical_matrix_sha256']


def test_hdf5_compression_chunking_independent(tmp_path):
    import anndata as ad
    import h5py
    x=csr_matrix(np.array([[0,2,1],[0,0,0],[4,0,0]],dtype=np.int64))
    expected=m.csr_identity(x)
    hashes=[]
    for i,compression in enumerate([None,'gzip','lzf']):
        path=tmp_path/f'matrix{i}.h5ad'
        ad.AnnData(X=x.copy()).write_h5ad(path,compression=compression)
        # Exercise another physical dataset chunk shape without changing values.
        with h5py.File(path,'r+') as f:
            values=f['X/data'][:];del f['X/data']
            f['X'].create_dataset('data',data=values,chunks=(i+1,),compression=compression)
        assert m.csr_identity(ad.read_h5ad(path).X)==expected
        hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
    assert len(set(hashes))==3


def test_reference_axis_preserves_exact_order_and_full_zero_columns():
    rows=[(0,'z:5-8','z',5,8),(1,'a:0-2','a',0,2),(2,'z:1-3','z',1,3)]
    count,digest=m.reference_axis_identity(rows,{'z':10,'a':10})
    assert count==3 and digest==hashlib.sha256(b'z:5-8\na:0-2\nz:1-3\n').hexdigest()
    with pytest.raises(m.ScATACMatrixError):m.reference_axis_identity(rows[::-1],{'z':10,'a':10})
    with pytest.raises(m.ScATACMatrixError):m.reference_axis_identity(rows,{'z':6,'a':10})
    with pytest.raises(m.ScATACMatrixError):m.reference_axis_identity([],{'z':10})
    with pytest.raises(m.ScATACMatrixError):m.reference_axis_identity([(0,'z:05-8','z',5,8)],{'z':10})
