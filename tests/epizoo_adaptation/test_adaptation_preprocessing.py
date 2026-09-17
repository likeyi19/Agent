from pathlib import Path
import numpy as np
import pytest
from agent.tools.models.epizoo_adaptation import preprocessing as p
from agent.tools.models.epizoo_adaptation.contract import read_json
from epizoo.data import processing as owner


def test_single_uses_public_owner(matrix_factory,monkeypatch,tmp_path):
    joint,cells,manifests,reference=p.corpus([matrix_factory()])
    calls=[];original=owner.compute_document_frequency
    def counted(*args,**kwargs):calls.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(owner,'compute_document_frequency',counted)
    df,sentences=p.preprocess(joint)
    assert len(calls)==1
    np.testing.assert_array_equal(df,[1,2,1])
    expected=owner.generate_cell_sentences(owner.compute_tfidf(joint,verbose=False))
    assert sentences==expected.obs.cell_indices.tolist()
    assert joint.X.dtype==np.int64 and cells[0]==dict(dataset_index=0,cell_id='cell0')
    root=tmp_path/'preprocessing';root.mkdir()
    metadata=p.save_preprocessing(root,joint,cells,manifests,df,sentences)
    assert read_json(root/'preprocessing.json')==metadata
    assert metadata['matrix_identities']==[manifests[0]['identity_sha256']]


def test_joint_corpus(matrix_factory):
    a=matrix_factory([[1,0,0]]);b=matrix_factory([[1,2,0],[0,0,3]])
    joint,cells,_,_=p.corpus([a,b]);df,sentences=p.preprocess(joint)
    np.testing.assert_array_equal(df,owner.compute_document_frequency(joint.X,dtype=np.int64))
    assert cells==[dict(dataset_index=0,cell_id='cell0'),dict(dataset_index=1,cell_id='cell0'),dict(dataset_index=1,cell_id='cell1')]
    expected=owner.generate_cell_sentences(owner.compute_tfidf(joint,verbose=False))
    assert sentences==expected.obs.cell_indices.tolist()
    assert df.tolist()==[2,1,1]


@pytest.mark.parametrize('semantics',['insertion_counts','binary_accessibility'])
def test_unqualified_semantics(matrix_factory,semantics):
    with pytest.raises(ValueError,match='fragment_counts'):
        p.corpus([matrix_factory([[1,0,1]],semantics)])


@pytest.mark.parametrize('values',[[[0,0,0]],[[1,1,1]]])
def test_zero_or_no_negative_cell(matrix_factory,values):
    with pytest.raises(ValueError,match='positive and inaccessible'):
        p.corpus([matrix_factory(values)])


def test_duplicate_pooling(matrix_factory):
    binding=matrix_factory()
    with pytest.raises(ValueError,match='twice'):p.corpus([binding,binding])


def test_matrix_mutation(matrix_factory):
    import h5py
    binding=matrix_factory()
    with h5py.File(Path(binding['manifest_path']).parent/'matrix.h5ad','r+') as f:f['X/data'][0]=7
    with pytest.raises(ValueError):p.corpus([binding])


def test_incompatible_reference(matrix_factory,monkeypatch):
    a=matrix_factory();b=matrix_factory()
    real=p.matrix.load_manifest_bytes
    def changed(raw):
        value=real(raw)
        if value['source']['path'].endswith('input2.h5ad'):value['assembly']='other'
        return value
    monkeypatch.setattr(p.matrix,'load_manifest_bytes',changed)
    with pytest.raises(ValueError):p.corpus([a,b])


def test_zero_feature_preserved(matrix_factory):
    joint,*_=p.corpus([matrix_factory([[1,0,0],[2,0,0]])])
    df,sentences=p.preprocess(joint)
    assert joint.n_vars==3 and df.tolist()==[2,0,0] and sentences==[[4],[4]]
