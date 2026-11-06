"""Explicit vocabularies change comparison views, never original scientific labels."""
import csv
import json
from pathlib import Path

import anndata as ad
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score, recall_score, adjusted_rand_score, normalized_mutual_info_score

from agent.tools.analysis import cell_label_comparison as owner
from agent.tools.data.fastq_verification_authority import file_closure


@pytest.fixture
def args(tmp_path):
    p = tmp_path/'prediction.h5ad'
    ad.AnnData(obs=pd.DataFrame({'label':pd.Categorical(['CD4 Naive T','CD4 Memory T','B',None,'B','B'])},
        index=['a','b','c','d','e','p-only'])).write_h5ad(p)
    r = tmp_path/'reference.tsv'
    r.write_text('cell\tlabel\ne\t\nd\tB\nc\tB\nb\tCD4 T cell\na\tCD4 T cell\nr-only\tB\n')
    return dict(prediction_path=str(p), reference_path=str(r), prediction_label_key='label',
        reference_label_key='label', reference_cell_key='cell', shared_library_id='explicit-fixture-library',
        output_dir=str(tmp_path/'strict'))


def test_both_sides_many_to_one_metrics_denominators_and_preservation(args, tmp_path):
    strict = owner.execute(args, 'strict')
    before = file_closure([args['prediction_path'],args['reference_path'],*map(str,Path(args['output_dir']).iterdir())])
    mapped_args = dict(args,output_dir=str(tmp_path/'mapped'),
        prediction_mapping={'CD4 Naive T':'CD4 T','CD4 Memory T':'CD4 T'},reference_mapping={'CD4 T cell':'CD4 T'})
    mapped = owner.execute(mapped_args,'mapped')
    assert mapped['label_policy']=='explicit_harmonized_labels' and strict['label_policy']=='exact_labels_no_mapping'
    assert mapped['original_metrics']['accuracy']==strict['accuracy']==.25
    assert mapped['matched_cells']==5 and mapped['evaluated_cells']==4 and mapped['partition_cells']==3
    assert mapped['prediction_only_cells']==mapped['reference_only_cells']==1
    assert mapped['reference_label_missing_cells']==mapped['unassigned_prediction_cells']==1
    y,p=['CD4 T','CD4 T','B','B'],['CD4 T','CD4 T','B','<missing>']
    expected=dict(accuracy=accuracy_score(y,p),macro_f1=f1_score(y,p,labels=['B','CD4 T'],average='macro',zero_division=0),
        balanced_accuracy=recall_score(y,p,labels=['B','CD4 T'],average='macro',zero_division=0),
        ari=adjusted_rand_score(y[:3],p[:3]),nmi=normalized_mutual_info_score(y[:3],p[:3],average_method='arithmetic'))
    for key,value in expected.items(): assert mapped[key]==value
    assert mapped['metric_denominators']['accuracy_cells']==4
    assert mapped['metric_denominators']['macro_f1_classes']==2
    assert mapped['metric_denominators']['original_macro_f1_classes']==4
    assert mapped['metric_denominators']['partition_scope']=='mapped_partitions'
    coverage=mapped['mapping']['prediction']
    assert coverage['rule_covered_cells']==coverage['changed_cells']==2
    assert coverage['rule_coverage_labeled']==.4 and coverage['unmapped_labeled_cells']==3
    assert coverage['unmapped_categories']==['B'] and coverage['mapped_vocabulary']==['B','CD4 T']
    assert coverage['missing_cells']==1 and coverage['rule_source']=='explicit_request_parameter'
    rows=list(csv.DictReader(Path(mapped['aligned_path']).open(),delimiter='\t'))
    assert len(rows)==7 and rows[0]['predicted_label']=='CD4 Naive T' and rows[0]['mapped_predicted_label']=='CD4 T'
    assert rows[0]['agreement']=='False' and rows[0]['mapped_agreement']=='True'
    assert rows[3]['predicted_label']==rows[3]['mapped_predicted_label']==''
    report=json.loads(Path(mapped['comparison_path']).read_text())
    assert report['original_comparison']['metrics']['accuracy']==strict['accuracy']
    assert 'mapped partitions, not original clustering' in mapped['limitations']
    assert file_closure(p['path'] for p in before)==before
    owner.verify_result(mapped_args,mapped)
    assert owner.recover(mapped_args,'mapped')==mapped
    with pytest.raises(ValueError):owner.recover(dict(mapped_args,prediction_mapping={'B':'T'}),'mapped')


@pytest.mark.parametrize('side,rule', [('prediction',{'CD4 Naive T':'CD4 T cell'}),('reference',{'CD4 T cell':'CD4 Naive T'})])
def test_either_side_one_to_one_and_unmentioned_labels_unchanged(args,side,rule):
    result=owner.compare_cell_annotations(**args,**{side+'_mapping':rule})
    assert result['accuracy']==.5
    assert result['mapping'][side]['rules']==[dict(original_label=k,common_label=v) for k,v in rule.items()]
    other='reference' if side=='prediction' else 'prediction'
    assert result['mapping'][other]['changed_cells']==0


@pytest.mark.parametrize('mapping', [{}, {'absent':'B'}, {'CD4 Naive T':['B','T']}, {'CD4 Naive T':None}, {'':'B'}, {'CD4 Naive T':'unassigned'}, {None:'B'}])
def test_invalid_unknown_split_or_missing_mapping_is_rejected(args,mapping):
    with pytest.raises(ValueError):owner.compare_cell_annotations(**args,prediction_mapping=mapping)
    assert not list(Path(args['output_dir']).glob('manifest.json'))


@pytest.mark.parametrize('label',['unassigned','Unknown','unlabeled','unlabelled'])
def test_unassigned_literal_cannot_be_promoted_into_biological_class(args,label):
    data=ad.read_h5ad(args['prediction_path']);data.obs['label']=label;data.write_h5ad(args['prediction_path'])
    with pytest.raises(ValueError,match='Missing/unassigned'):owner.compare_cell_annotations(**args,prediction_mapping={label:'B'})


def test_mapping_is_simultaneous_and_exact_case(args):
    result=owner.compare_cell_annotations(**args,prediction_mapping={'CD4 Naive T':'B','B':'CD4 T cell'})
    assert result['mapping']['prediction']['mapped_vocabulary']==['B','CD4 Memory T','CD4 T cell']
    with pytest.raises(ValueError,match='does not exist'):owner.compare(dict(args,prediction_mapping={'b':'B'}))


def test_mapping_never_repairs_cell_identity(args):
    Path(args['reference_path']).write_text('cell\tlabel\nwrong\tCD4 T cell\n')
    with pytest.raises(ValueError,match='No cells match'):owner.compare(dict(args,prediction_mapping={'CD4 Naive T':'CD4 T cell'}))


def test_default_exact_comparison_new_facts_without_mapping(args):
    result=owner.compare_cell_annotations(**args)
    assert result['accuracy']==.25 and result['mapping']['applied'] is False
    assert result['vocabulary']['prediction_only_names']==['CD4 Memory T','CD4 Naive T']
    assert result['vocabulary']['reference_only_names']==['CD4 T cell']
    assert result['vocabulary']['category_names_differ']
    for key,value in result['original_metrics'].items():assert result[key]==value
    rows=list(csv.DictReader(Path(result['aligned_path']).open(),delimiter='\t'))
    assert 'mapped_predicted_label' not in rows[0]


def test_compatible_vocabulary_disagreement_is_not_vocabulary_difference(args):
    Path(args['reference_path']).write_text('cell\tlabel\na\tB\nb\tCD4 Naive T\nc\tCD4 Memory T\n')
    result=owner.compare_cell_annotations(**args)
    assert result['accuracy']==0 and result['vocabulary']['category_names_differ'] is False


def test_actual_historical_publication_still_verifies_without_new_fields(tmp_path):
    # Construct the pre-extension envelope with independently fixed historical profile.
    # The acceptance audit additionally checks the real preserved SRV-1B publication.
    from agent.tools.analysis.sequence_prediction import write
    args=dict(output_dir=str(tmp_path),prediction_path='/historical/original',reference_path='/historical/reference')
    pins=[]
    for name in owner.DELIVERY_FILES:
        path=tmp_path/name;path.write_text('{}');pins+=file_closure([path])
    result={k:None for k in owner.RESULT_FIELDS};result.update(status='success',contract_version=owner.CONTRACT)
    value=dict(contract=owner.CONTRACT,profile=owner.HISTORICAL_PROFILE,arguments=args,result=result,
        execution_identity='historical',files=pins,sources=[],native_completion='succeeded')
    manifest=tmp_path/'manifest.json';sha=write(manifest,value)
    loaded=owner.verify_publication(manifest,expected_sha256=sha)
    assert loaded['contract_version']==owner.CONTRACT and 'mapping' not in loaded


def test_bounded_preview_keeps_full_vocabularies_and_unchanged_coverage_in_download(args):
    labels=[f'Category{i}' for i in range(64)]
    data=ad.AnnData(obs=pd.DataFrame({'label':labels},index=[f'cell{i}' for i in range(64)]))
    data.write_h5ad(args['prediction_path'])
    pd.DataFrame({'cell':data.obs_names,'label':labels}).to_csv(args['reference_path'],sep='\t',index=False)
    result=owner.compare_cell_annotations(**args,prediction_mapping={'Category0':'Common'},reference_mapping={'Category0':'Common'})
    assert len(result['vocabulary']['prediction'])==50
    assert result['vocabulary']['categories_omitted']['prediction']==14
    coverage=result['mapping']['prediction']
    assert coverage['unmapped_labeled_cells']==63 and coverage['rule_covered_cells']==1
    assert len(coverage['unmapped_categories'])==50 and coverage['categories_omitted']['unmapped_categories']==13
    report=json.loads(Path(result['comparison_path']).read_text())
    assert len(report['vocabulary']['prediction'])==64
    assert len(report['mapping']['prediction']['unmapped_categories'])==63
    assert result['accuracy']==result['original_metrics']['accuracy']==1
