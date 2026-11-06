"""General comparisons use explicit library/cell joins, never feature matrices."""
import csv
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score, recall_score

from agent.tools.analysis import cell_label_comparison as owner


def inputs(tmp_path):
    a = ad.AnnData(np.ones((5,2)), obs=pd.DataFrame({'prediction':pd.Categorical(['A','B',None,'B','A'])}, index=['a','b','c','d','only-p']))
    a.write_h5ad(tmp_path/'prediction.h5ad')
    # Different features and order, extra reference cell, missing reference label.
    b = ad.AnnData(np.zeros((5,7)), obs=pd.DataFrame({'reference':pd.Categorical(['A','B','B',None,'A'])}, index=['a','c','b','d','only-r']))
    b.write_h5ad(tmp_path/'reference.h5ad')
    return dict(prediction_path=str(tmp_path/'prediction.h5ad'),reference_path=str(tmp_path/'reference.h5ad'),
        prediction_label_key='prediction',reference_label_key='reference',shared_library_id='physical-library-1',output_dir=str(tmp_path/'out'))


def test_native_metrics_coverage_no_feature_or_order_equality(tmp_path):
    args=inputs(tmp_path); result=owner.compare_cell_annotations(**args)
    assert result['matched_cells']==4 and result['prediction_only_cells']==result['reference_only_cells']==1
    assert result['evaluated_cells']==3 and result['evaluation_coverage']==.6
    assert result['reference_label_missing_cells']==result['unassigned_prediction_cells']==1
    y,p=['A','B','B'],['A','B','<missing>']
    assert result['accuracy']==accuracy_score(y,p)
    assert result['macro_f1']==f1_score(y,p,labels=['A','B'],average='macro',zero_division=0)
    assert result['balanced_accuracy']==recall_score(y,p,labels=['A','B'],average='macro',zero_division=0)
    assert result['assigned_accuracy']==1 and result['ari']==result['nmi']==1
    report=json.loads(Path(result['comparison_path']).read_text())
    assert report['confusion']['counts']==[[1,0,0],[0,1,1]]
    rows=list(csv.DictReader(Path(result['aligned_path']).open(),delimiter='\t'))
    assert [r['cell_id'] for r in rows]==['a','b','c','d','only-p','only-r']
    owner.verify_result(args,result)
    Path(result['aligned_path']).write_text('tampered')
    with pytest.raises(ValueError): owner.verify_result(args,result)


@pytest.mark.parametrize('delimiter,extension',[('\t','tsv'),(',','csv')])
def test_reference_table_and_exact_case_no_mapping(tmp_path,delimiter,extension):
    args=inputs(tmp_path); path=tmp_path/('labels.'+extension)
    with path.open('w',newline='') as f:
        w=csv.writer(f,delimiter=delimiter);w.writerows([['cell','type'],['b','a'],['a','B'],['c','B']])
    args.update(reference_path=str(path),reference_cell_key='cell',reference_label_key='type')
    result=owner.compare_cell_annotations(**args)
    assert result['accuracy']==0
    assert result['matched_cells']==3
    assert 'a' in {r['label'] for r in json.loads(Path(result['comparison_path']).read_text())['per_class']}


def test_identity_required_and_unrelated_libraries_do_not_match(tmp_path):
    args=inputs(tmp_path);args.pop('shared_library_id')
    with pytest.raises(ValueError,match='bare barcodes'):owner.compare(args)
    for name,library in [('prediction','unrelated-1'),('reference','unrelated-2')]:
        a=ad.read_h5ad(args[name+'_path']);a.obs['library']=library;a.write_h5ad(args[name+'_path'])
    args.update(prediction_sample_key='library',reference_sample_key='library')
    with pytest.raises(ValueError,match='No cells match'):owner.compare(args)


def test_multi_library_duplicate_barcodes_are_scoped(tmp_path):
    args=inputs(tmp_path);args.pop('shared_library_id')
    for name,key,labels in [('prediction','prediction',['A','B']),('reference','reference',['A','B'])]:
        a=ad.AnnData(obs=pd.DataFrame({key:labels,'library':['one','two'],'barcode':['same','same']},index=['row1','row2']))
        a.write_h5ad(args[name+'_path'])
    args.update(prediction_sample_key='library',reference_sample_key='library',prediction_cell_key='barcode',reference_cell_key='barcode')
    summary,*_=owner.compare(args)
    assert summary['accuracy']==1 and summary['matched_cells']==2
    a=ad.read_h5ad(args['reference_path']);a.obs['library']='one';a.write_h5ad(args['reference_path'])
    with pytest.raises(ValueError,match='Duplicate'):owner.compare(args)


def test_all_reference_missing_is_explicit_no_accuracy(tmp_path):
    args=inputs(tmp_path);a=ad.read_h5ad(args['reference_path']);a.obs['reference']=pd.Categorical([None]*5,categories=['A']);a.write_h5ad(args['reference_path'])
    result=owner.compare_cell_annotations(**args)
    assert result['accuracy'] is result['macro_f1'] is result['balanced_accuracy'] is None
    assert result['evaluated_cells']==0 and result['reference_label_missing_cells']==4


def test_inputs_immutable_and_recovery_only_saved_completion(tmp_path):
    from agent.tools._file_identity import file_closure
    args=inputs(tmp_path);before=file_closure([args['prediction_path'],args['reference_path']])
    result=owner.execute(args,'execution-1')
    assert owner.recover(args,'execution-1')==result
    with pytest.raises(ValueError):owner.recover(args,'execution-2')
    assert file_closure([args['prediction_path'],args['reference_path']])==before
    with pytest.raises(ValueError,match='empty'):owner.execute(args,'execution-2')


@pytest.mark.parametrize("mapped", [False, True])
def test_comparison_reuses_genuine_umap_and_rejects_wrong_ancestry(tmp_path,monkeypatch,mapped):
    from agent.tools.analysis.pca_embedding import pca_embed_cells
    from agent.tools.analysis.embedding_analysis import build_cell_neighbors, compute_cell_umap
    from agent.tools.analysis.marker_annotation import sha256
    import scanpy as sc
    from scipy import sparse
    source=tmp_path/'cells.h5ad'
    data=ad.AnnData(sparse.csr_matrix(np.random.default_rng(6).poisson(2,(16,24))),
        obs=pd.DataFrame({'prediction':pd.Categorical(['A']*8+['B']*8)},index=[f'cell{i}' for i in range(16)]))
    data.write_h5ad(source)
    pca=pca_embed_cells(source,tmp_path/'pca',n_components=3)
    neighbors=build_cell_neighbors(pca['embedding_path'],pca['cell_ids_path'],tmp_path/'neighbors',n_neighbors=4)
    umap=compute_cell_umap(neighbors['analysis_path'],tmp_path/'umap')
    expected=ad.read_h5ad(umap['analysis_path']).obsm['X_umap'].copy()
    reference=tmp_path/'truth.tsv';pd.DataFrame({'cell':data.obs_names[:12][::-1], 'type':['B']*4+['A']*8}).to_csv(reference,sep='\t',index=False)
    args=dict(prediction_path=str(source),reference_path=str(reference),prediction_label_key='prediction',reference_label_key='type',reference_cell_key='cell',
        shared_library_id='explicit-synthetic-library',umap_path=umap['analysis_path'],umap_sha256=sha256(umap['analysis_path']),output_dir=str(tmp_path/'comparison'))
    if mapped:
        args.update(prediction_mapping={'A':'Common'},reference_mapping={'A':'Common'})
    monkeypatch.setattr(sc.tl,'umap',lambda *a,**k:pytest.fail('UMAP refit during comparison'))
    output=owner.compare_cell_annotations(**args)
    overlay=json.loads((Path(output['manifest_path']).parent/'overlay.json').read_text())
    np.testing.assert_array_equal(overlay['coordinates'],expected)
    assert overlay['reference'][12:]==[None]*4 and overlay['matched'][12:]==[False]*4
    assert output['umap_available'] is True
    if mapped:
        assert overlay['label_policy']=='explicit_harmonized_labels'
        assert overlay['predicted'][:8]==['Common']*8 and overlay['reference'][:8]==['Common']*8
        assert output['original_metrics']['accuracy']==output['accuracy']
    # Equal cells and labels on another byte source do not qualify these coordinates.
    wrong=tmp_path/'other.h5ad';data.uns['unrelated_source']='different';data.write_h5ad(wrong)
    args.update(prediction_path=str(wrong),output_dir=str(tmp_path/'other-comparison'))
    with pytest.raises(ValueError,match='exact original-source'):owner.compare_cell_annotations(**args)


@pytest.mark.parametrize('body', ['a,A,unexpected\nb,B,unexpected\n', '0,a,A\n1,b,B\n'])
def test_label_table_rejects_rows_that_do_not_match_header(tmp_path,body):
    path=tmp_path/'malformed.csv'
    path.write_text('cell,type\n'+body)
    with pytest.raises(ValueError,match='row width'):
        owner.observations(path,'type','cell')


def test_h5ad_label_column_must_be_per_cell_not_broadcast_scalar(tmp_path):
    import h5py
    args=inputs(tmp_path)
    with h5py.File(args['prediction_path'],'r+') as handle:
        del handle['obs/prediction']
        scalar=handle['obs'].create_dataset('prediction',data='A',dtype=h5py.string_dtype())
        scalar.attrs['encoding-type']='string'
        scalar.attrs['encoding-version']='0.2.0'
    with pytest.raises(ValueError,match='one value per cell'):
        owner.observations(args['prediction_path'],'prediction')

def test_reference_table_keeps_quoted_labels_and_omitted_trailing_labels(tmp_path):
    path=tmp_path/'labels.csv'
    path.write_text('cell,type\na,"T, memory"\nb\nc,NA\n')
    ids,samples,labels,axis=owner.observations(path,'type','cell')
    assert ids==['a','b','c'] and samples is None
    assert labels==['T, memory',None,'NA']

def test_all_missing_predictions_are_errors_with_no_assigned_denominator(tmp_path):
    args=inputs(tmp_path)
    data=ad.read_h5ad(args['prediction_path'])
    data.obs['prediction']=pd.Categorical([None]*data.n_obs,categories=['A','B'])
    data.write_h5ad(args['prediction_path'])
    result=owner.compare_cell_annotations(**args)
    assert result['evaluated_cells']==result['unassigned_prediction_cells']==3
    assert result['evaluation_coverage']==.6
    assert result['accuracy']==result['macro_f1']==result['balanced_accuracy']==0.0
    assert result['assigned_accuracy'] is None and result['partition_cells']==0
    assert result['ari'] is result['nmi'] is None
    report=json.loads(Path(result['comparison_path']).read_text())
    assert report['confusion']['counts']==[[0,0,1],[0,0,2]]
