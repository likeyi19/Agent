"""ICTA feature preparation; no replacement statistical method."""
import numpy as np
import pytest
from scipy import sparse
from agent.tools.analysis.local_subdivision import testable_features as select


@pytest.mark.parametrize('storage',[np.array,sparse.csr_matrix,sparse.csc_matrix])
@pytest.mark.parametrize('features,expected',[(None,[2,3]),(['p3','p1','p2'],[3,2]),([3,0,2],[3,2]),([-1,-4,-2],[-1,-2]),([],[]),([False,False,True,False],[2])])
def test_exact_comparison_constants_and_explicit_intersection(storage,features,expected):
    x=storage([[0,2,0,1],[0,9,1,0],[1,0,1,1]])
    before=x.copy()
    selected,info=select(x,[0,1],['p0','p1','p2','p3'],features)
    assert selected==expected
    assert info['constant_zero_feature_count']==1 and info['constant_one_feature_count']==1
    assert info['testable_feature_count']==len(expected)
    np.testing.assert_array_equal(x.toarray() if sparse.issparse(x) else x,before.toarray() if sparse.issparse(before) else before)


def test_sparse_stored_zero_matches_actual_native_response():
    # Native sets even explicitly stored zero values to one.
    x=sparse.csc_matrix((np.array([0.,3.,0.]),np.array([0,1,0]),np.array([0,2,3])),shape=(2,2))
    selected,info=select(x,[0,1],['constant','mixed'])
    assert selected==[1] and info['constant_one_feature_count']==1
    assert x.data.tolist()==[0.,3.,0.]


def test_no_testable_features_and_invalid_selection():
    assert select(np.ones((2,3)),[0,1],['a','b','c'])[0]==[]
    with pytest.raises(ValueError):select(np.ones((2,3)),[0,1],['a','b','c'],[3])
    with pytest.raises(ValueError):select(np.ones((2,3)),[0,1],['a','b','c'],['absent'])


@pytest.mark.parametrize('storage',[np.array,sparse.csr_matrix,sparse.csc_matrix])
def test_one_parent_preparation_keeps_each_comparison_and_selector_exact(tmp_path,monkeypatch,storage):
    from types import SimpleNamespace
    import pandas as pd
    from agent.tools.analysis import local_subdivision as owner
    from agent.schemas.verification_authority import AuthorityError
    source=tmp_path/'accepted';source.write_text('exact accepted source')
    obs=pd.DataFrame({'leiden':['A','B','A','-1']},index=['a','b','c','d'])
    var=pd.DataFrame(index=['p0','p1','p2','p3'])
    x=storage([[0,1,0,1],[0,1,2,0],[0,1,1,0],[0,1,0,1]])
    from agent.tools._file_identity import file_sha256
    native=dict(matrix_path=str(source),matrix_sha256=file_sha256(source),function_name='snapatac2.tl.leiden')
    binding=SimpleNamespace(to_dict=lambda:dict(run_id='accepted',step_id='leiden'))
    monkeypatch.setattr(owner,'_read',lambda *a,**k:(None,native))
    monkeypatch.setattr(owner,'describe',lambda *a:dict(grouping='leiden'))
    reads=[]
    def metadata(*a):reads.append(1);return obs,var,x
    monkeypatch.setattr(owner,'_metadata',metadata)
    prepare=owner.differential_preparer(binding,store=None,registry=None)
    for child,features in [('A',None),('B',[3,0,2]),('A',['p3','p2']),('B',[False,False,True,True])]:
        kwargs=dict(min_pct=.125)
        if features is not None:kwargs['features']=features
        actual,proof=prepare(child,kwargs)
        comparison=owner._comparison(native,child,obs)
        expected,counts=select(x,comparison['cell_group1']+comparison['cell_group2'],var.index.tolist(),features)
        assert actual==dict(kwargs,**comparison,features=expected)
        assert all(proof[k]==v for k,v in counts.items())
        assert proof['comparison']==comparison
        assert proof['cell_ids']==obs.index[comparison['cell_group1']+comparison['cell_group2']].tolist()
    assert reads==[1]
    source.touch()
    prepare('A',{})  # Same bytes retain their scientific identity.
    assert reads==[1]
    source.write_text('changed accepted source')
    with pytest.raises(AuthorityError,match='changed'):prepare('A',{})
