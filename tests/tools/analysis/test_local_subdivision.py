"""Independent decision expectations and controlled evidence-reader contracts.

Mocked bindings below are not genuine native acceptance; native_raw qualifies it.
"""
from copy import deepcopy
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from agent.schemas.prior_output import PriorOutputBinding
from agent.schemas.verification_authority import AuthorityError
from agent.tools.analysis import local_subdivision as owner


def rows(count, q=.009):
    return [{'feature name': f'p{i}', 'log2(fold_change)': 1., 'p-value': q/2, 'adjusted p-value':q} for i in range(count)]


def child(positions, count, **extra):
    return dict(source_positions=positions, evidence_status='accepted', differential=dict(rows=rows(count)), **extra)


@pytest.mark.parametrize('counts,outcome,residual', [
    ([4,0], 'no_supported_subdivision',[5,1,7]),
    ([5,4], 'partial_supported_subdivision',[7]),
    ([5,5], 'supported_subdivision',[])])
def test_partial_partition_and_no_population_size_gate(counts,outcome,residual):
    children=[child([5,1],counts[0]),child([7],counts[1])]
    status,_,left=owner._decide(children,[5,1,7],.01,5)
    assert status==outcome and left==residual
    assert sorted([i for c in children if c['retained'] for i in c['source_positions']]+left)==[1,5,7]


def test_distinct_strict_native_statistics_and_parameters():
    native=rows(5)+rows(5)+rows(1,.01)
    native += [dict(rows(1)[0], **{'feature name':'negative','log2(fold_change)':-1}),
               dict(rows(1)[0], **{'feature name':'nonfinite','adjusted p-value':float('nan')})]
    assert owner._qualifying(native,.01)==['p0','p1','p2','p3','p4']
    assert owner._qualifying(native,.009)==[]
    c=child([0],0); c['differential']['rows']=native
    assert owner._decide([c],[0],.01,5)[0]=='supported_subdivision'
    assert owner._decide([c],[0],.01,6)[0]=='no_supported_subdivision'
    assert owner._decide([c],[0],.009,1)[0]=='no_supported_subdivision'


@pytest.mark.parametrize('annotation', [None,dict(status='assigned',preferred=['T']),
                                      dict(status='ambiguous',significant=['T','B'])])
def test_annotation_does_not_control_acceptance_or_continuation(annotation):
    supported=child([0],5,annotation=annotation)
    unsupported=child([1],4,annotation=annotation)
    outcome,_,residual=owner._decide([supported,unsupported],[0,1],.01,5)
    assert outcome=='partial_supported_subdivision' and residual==[1]
    assert supported['retained'] and not unsupported['retained']
    assert supported['annotation']==annotation


@pytest.mark.parametrize('status,outcome', [('missing_input','missing_input'),('native_execution_failure','native_execution_failure')])
def test_unassessed_is_never_unsupported(status,outcome):
    c=dict(source_positions=[1],evidence_status=status)
    assert owner._decide([child([0],5),c],[0,1],.01,5)[0]==outcome
    assert c['retained'] is None and c['qualifying_peak_count'] is None


@pytest.fixture
def bound(tmp_path,monkeypatch):
    """Controlled accepted-reader data, with an explicit ordered source relationship."""
    bindings=[]; natives=[]; steps={}
    selected=[4,1,3]
    parent_obs=pd.DataFrame({'leiden':pd.Categorical(['out','P','out','P','P'])},index=['c0','c1','c2','c3','c4'])
    local_obs=parent_obs.iloc[selected].copy(); local_obs['leiden']=pd.Categorical(['a','b','a'])
    for i,method in enumerate(('snapatac2.tl.leiden',)+owner.LOCAL_METHODS):
        obs=parent_obs if i==0 else local_obs
        path=tmp_path/f'{i}.h5ad'
        ad.AnnData(np.ones((len(obs),8)),obs=obs,var=pd.DataFrame(index=[f'p{k}' for k in range(8)])).write_h5ad(path)
        native=dict(matrix_path=str(path),matrix_sha256=str(i)*64,manifest_path=str(tmp_path/f'{i}.json'),
                    manifest_sha256=str(i)*64,function_name=method,measurement='peak_counts',
                    contract_version='native-snapatac2-matrix.v1',parameters=dict(kwargs={},count_slot='X'),
                    result_slots=[['obs.leiden','var.selected','obsm.X_spectral','obsp.distances','obs.leiden'][i]])
        args={}
        if i==0:
            native['source_identity']=dict(sha256='d'*64,declarations={'matrix_semantics':'peak_counts'})
        if i:
            prev=natives[-1]; args['input_path']=prev['matrix_path']
            native['source_identity']=dict(sha256=prev['matrix_sha256'],accepted_publication={k:prev[k] for k in ('manifest_path','manifest_sha256')})
        if i==1:
            native['parameters']['cell_subset']=dict(positions=selected,cell_ids=local_obs.index.tolist(),
                source_sha256=natives[0]['matrix_sha256'],groupby='leiden',groups=['P'],
                complete_group_membership=True,position_axis='obs')
        if i>=3:
            native['parameters']['graph_inputs']=[dict(function=owner.LOCAL_METHODS[2],representation='X_spectral',
                graph='distances',source_sha256=natives[2]['matrix_sha256'])]
        binding=PriorOutputBinding('run',str(i),'run_snapatac2','dataset','native_matrix_h5ad.v1','a'*64,str(i)*64,'b'*64)
        bindings.append(binding); natives.append(native)
        steps[binding]=SimpleNamespace(result=native,resolved_arguments=args)
    monkeypatch.setattr(owner,'validate_binding',lambda binding,**kw: steps[binding])
    return bindings,natives,steps


def assess(bound,**options):
    bindings,_,_=bound
    return owner.assess_subdivision(bindings[0],bindings[1:],store=None,registry=None,**options)


def test_one_cluster_no_differential_needed_and_budget_separate(bound):
    _,native,_=bound
    a=ad.read_h5ad(native[-1]['matrix_path']); a.obs['leiden']=pd.Categorical(['a']*3); a.write_h5ad(native[-1]['matrix_path'])
    result=assess(bound,budget_reason={'reason':'maximum analysis depth','maximum_levels':3})
    assert result['outcome']=='no_candidate_subdivision'
    assert result['residual_parent']['positions']==[4,1,3]
    assert result['execution_budget']['supplied_reason']['maximum_levels']==3
    assert not result['execution_budget']['enforced']


def test_missing_gene_only_peak_source_is_unassessed(bound):
    _,native,_=bound; native[0]['measurement']='gene_activity'; native[0]['parameters']['gene_activity']={'interface':'gene-activity.v1'}
    result=assess(bound)
    assert result['outcome']=='missing_input' and 'peak' in result['reason']


def test_derive_positions_and_rest_no_researcher_hashes(bound):
    bindings,_,_=bound
    kwargs=owner.differential_kwargs(bindings[-1],'a',store=None,registry=None)
    assert kwargs==dict(cell_group1=[0,2],cell_group2=[1],direction='positive')
    assert assess(bound)['outcome']=='missing_input'


@pytest.mark.parametrize('mutation', ['source_path','source_sha','publication','repeat','partial','layer','global_graph','inherited_spectral'])
def test_source_and_population_substitution_rejected(bound,mutation):
    bindings,native,steps=bound
    if mutation=='source_path': steps[bindings[2]].resolved_arguments['input_path']='unrelated.h5ad'
    if mutation=='source_sha': native[2]['source_identity']['sha256']='f'*64
    if mutation=='publication': native[2]['source_identity']['accepted_publication']['manifest_path']='unrelated.json'
    if mutation=='repeat': native[1]['parameters']['cell_subset']['positions']=[4,1,3,4]
    if mutation=='partial': native[1]['parameters']['cell_subset']['complete_group_membership']=False
    if mutation=='layer': native[2]['parameters']['count_slot']='layers/gene_activity'
    if mutation=='global_graph': native[3]['parameters']['graph_inputs'][0]['source_sha256']='f'*64
    if mutation=='inherited_spectral': native[2]['result_slots']=[]
    with pytest.raises(AuthorityError): assess(bound)


@pytest.mark.parametrize('options', [dict(adjusted_p_threshold=float('nan')),dict(adjusted_p_threshold=-.1),
                                     dict(minimum_peak_features=0),dict(minimum_peak_features=1.5)])
def test_interpretation_parameter_validation_not_native_admission(options):
    with pytest.raises(ValueError): owner.assess_subdivision(None,store=None,registry=None,**options)


def test_annotation_full_evidence_preserved_and_bound(bound,tmp_path):
    bindings,natives,steps=bound
    local=natives[-1]
    native=deepcopy(local); native['function_name']='scanpy.tl.rank_genes_groups'
    native['matrix_path']=str(tmp_path/'markers.h5ad'); native['manifest_path']=str(tmp_path/'markers.json')
    native['source_identity']=dict(sha256=local['matrix_sha256'],accepted_publication={k:local[k] for k in ('manifest_path','manifest_sha256')})
    annotated=ad.read_h5ad(local['matrix_path']); annotated.obs['native_metric']=1.
    annotated.write_h5ad(native['matrix_path'])
    binding=PriorOutputBinding('marker','marker','run_scanpy','dataset','native_matrix_h5ad.v1','a'*64,'c'*64,'b'*64)
    steps[binding]=SimpleNamespace(result=native,resolved_arguments={'input_path':local['matrix_path']})
    annotation=dict(provenance=dict(binding=binding.to_dict(),native_result=native),
        reference={'resource':'unchanged-fixture'},candidates=[{'evidence':'full'}],
        clusters=[dict(cluster='a',groupby='leiden',status='assigned',preferred=['T']),
                  dict(cluster='b',groupby='leiden',status='ambiguous',significant=['T','B'])])
    result=assess(bound,annotation=annotation,annotation_bindings=[binding])
    assert result['annotation']==annotation
    assert [c['exploratory_unnamed'] for c in result['children']]==[False,True]
    assert result['outcome']=='missing_input'  # Names never supply missing differential evidence.
    annotation['provenance']['binding']['accepted_step_sha256']='f'*64
    with pytest.raises(AuthorityError): assess(bound,annotation=annotation,annotation_bindings=[binding])


def test_native_peak_counting_semantics_use_accepted_ancestry(bound,monkeypatch):
    from agent.orchestration import verification_authority
    bindings,natives,_=bound
    original=dict(function_name='snapatac2.pp.make_peak_matrix',matrix_sha256='e'*64,
                  manifest_path='peak-manifest.json',manifest_sha256='f'*64)
    natives[0]['measurement']='paired_insertion_counts'
    natives[0]['source_identity']=dict(sha256=original['matrix_sha256'],accepted_publication={k:original[k] for k in ('manifest_path','manifest_sha256')})
    context=SimpleNamespace(validate_anchors=lambda:None,proofs={'peak':SimpleNamespace(result_metadata=original)})
    monkeypatch.setattr(verification_authority,'_load_context',lambda *a:context)
    result=assess(bound)
    assert result['peak_source']==dict(origin='accepted_native_peak_matrix',native_result=original)
    original['function_name']='snapatac2.pp.add_tile_matrix'
    assert assess(bound)['peak_source'] is None  # Do not invent peak calling from bins.


@pytest.mark.parametrize('selector', [[2,0],['c2','c0'],[True,False,True]])
def test_native_selector_forms_preserve_membership_without_order_gate(selector):
    assert sorted(owner._positions(selector,['c0','c1','c2']))==[0,2]


def test_branches_stop_independently():
    first=[child([9],4),child([8],0)]
    second=[child([4],5),child([3],6)]
    assert owner._decide(first,[9,8],.01,5)[0]=='no_supported_subdivision'
    assert owner._decide(second,[4,3],.01,5)[0]=='supported_subdivision'


@pytest.mark.parametrize('labels,expected', [(['a','-1','a'],'missing_input'),(['-1']*3,'no_candidate_subdivision')])
def test_native_unassigned_cells_are_residual_not_a_candidate(bound,labels,expected):
    bindings,native,_=bound
    a=ad.read_h5ad(native[-1]['matrix_path']); a.obs['leiden']=pd.Categorical(labels); a.write_h5ad(native[-1]['matrix_path'])
    result=assess(bound)
    assert result['outcome']==expected
    assert all(c['cluster']!='-1' for c in result['children'])
    assert result['residual_parent']['positions']==[4,1,3]
    assert result['unassigned_local_positions']==[i for i,label in enumerate(labels) if label=='-1']
    with pytest.raises(ValueError,match='unassigned'):
        owner.differential_kwargs(bindings[-1],'-1',store=None,registry=None)


@pytest.mark.parametrize('measurement', ['peak_counts','binary_accessibility','fragment_counts','researcher_count_unit'])
@pytest.mark.parametrize('field', ['declarations','supplied_declarations'])
def test_explicit_peak_declaration_is_independent_of_measurement(bound,measurement,field):
    _,native,_=bound
    parent=native[0]; parent['measurement']=measurement
    parent['source_identity']=dict(sha256='d'*64,measurement=measurement,
        **{field:{'matrix_semantics':'peak_accessibility'}})
    result=assess(bound)
    assert result['peak_source']['declaration']==dict(source_sha256='d'*64,field=field,matrix_semantics='peak_accessibility')
    assert result['peak_source']['native_result']['measurement']==measurement
    assert result['parent_membership']['positions']==[4,1,3]


@pytest.mark.parametrize('measurement', ['binary_accessibility','fragment_counts','gene_activity','native_fragment_positions_and_lengths'])
def test_units_alone_do_not_establish_peak_feature_identity(bound,measurement):
    _,native,_=bound
    native[0]['measurement']=measurement
    native[0]['source_identity']=dict(sha256='d'*64,declarations={'matrix_semantics':measurement})
    result=assess(bound)
    assert result['peak_source'] is None and result['outcome']=='missing_input'


@pytest.mark.parametrize('function', ['snapatac2.pp.add_tile_matrix','snapatac2.pp.import_fragments','snapatac2.pp.make_gene_matrix','episcanpy.tl.geneactivity'])
def test_known_native_representation_cannot_be_overridden_by_downstream_declaration(bound,monkeypatch,function):
    from agent.orchestration import verification_authority
    _,native,_=bound
    original=dict(function_name=function,matrix_sha256='e'*64,
        manifest_path='origin.json',manifest_sha256='f'*64)
    parent=native[0]
    parent['source_identity'].update(sha256='e'*64,accepted_publication={k:original[k] for k in ('manifest_path','manifest_sha256')})
    context=SimpleNamespace(validate_anchors=lambda:None,proofs={'origin':SimpleNamespace(result_metadata=original)})
    monkeypatch.setattr(verification_authority,'_load_context',lambda *a:context)
    assert assess(bound)['peak_source'] is None


def test_exact_accepted_peak_declaration_survives_preserving_chain(bound,monkeypatch):
    from agent.orchestration import verification_authority
    _,native,_=bound
    original=deepcopy(native[0]); original.update(matrix_sha256='e'*64,manifest_path='origin.json',manifest_sha256='f'*64)
    original['source_identity']=dict(sha256='d'*64,declarations={'matrix_semantics':'binary_accessibility'},
                                    supplied_declarations={'matrix_semantics':'peak_accessibility'})
    native[0]['measurement']='binary_accessibility'
    native[0]['source_identity']=dict(sha256='e'*64,declarations={'matrix_semantics':'binary_accessibility'},
                                    accepted_publication={k:original[k] for k in ('manifest_path','manifest_sha256')})
    context=SimpleNamespace(validate_anchors=lambda:None,proofs={'origin':SimpleNamespace(result_metadata=original)})
    monkeypatch.setattr(verification_authority,'_load_context',lambda *a:context)
    assert assess(bound)['peak_source']['native_result']==original
    native[0]['source_identity']['sha256']='c'*64
    assert assess(bound)['peak_source'] is None
