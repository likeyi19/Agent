"""Shared native parameter/content contracts; no provider or scientific run."""
import json
from dataclasses import replace
import anndata as ad
import numpy as np
import pandas as pd
import pytest
from agent.schemas import AgentRequest, StepOutputRef
from agent.orchestration import build_default_tool_registry
from agent.orchestration.scientific_inputs import (OperationPreparationContext,
    ScientificInputRequired, ScientificDiagnostic, validate_input_properties)
from agent.tools.analysis.native_matrix import SNAPATAC2
from agent.tools.analysis.snapatac_raw import BARCODE_EXTRACTION, BARCODE_CHOICES


def context(selected=None, tool='run_snapatac2', answers=None, observe=True):
    registry=build_default_tool_registry()
    return OperationPreparationContext(registry.get(tool),'method',registry,selected or {},
        {},answers or {},[],None,None,{},observe_existing=observe)


@pytest.mark.parametrize('kwargs', [{}, {'barcode_tag':None}, {'barcode_regex':None},
                                    {'barcode_tag':None,'barcode_regex':None}])
def test_missing_barcode_is_one_conditional_choice(kwargs):
    args=dict(function_name='pp.make_fragment_file',input_path='/unopened.bam',kwargs=kwargs)
    with pytest.raises(ScientificInputRequired) as exc:SNAPATAC2.prepare_operation(args,context())
    requirement,=exc.value.requirements
    assert requirement.kind=='resource' and requirement.member is None
    assert requirement.applicability==BARCODE_EXTRACTION
    assert requirement.offers==BARCODE_CHOICES


@pytest.mark.parametrize('kwargs', [{'barcode_tag':'RG'}, {'barcode_regex':'^([^:]+)'},
                                    {'barcode_tag':'RG','barcode_regex':None}])
def test_explicit_single_mechanism_keeps_native_value(kwargs):
    args=dict(function_name='pp.make_fragment_file',input_path='/unopened.bam',kwargs=kwargs)
    assert SNAPATAC2.prepare_operation(args,context())==args


@pytest.mark.parametrize('kwargs', [{'barcode_tag':'RG','barcode_regex':'(.*)'},
                                    {'barcode_tag':''}, {'barcode_regex':4}])
def test_conflicting_or_invalid_barcode_is_scientific_diagnostic(kwargs):
    with pytest.raises(ScientificDiagnostic) as exc:
        SNAPATAC2.prepare_operation(dict(function_name='pp.make_fragment_file',input_path='/unopened.bam',kwargs=kwargs),context())
    assert exc.value.code=='NATIVE_BARCODE_EXTRACTION_INVALID'


def selected():
    return {'import':('run_snapatac2',dict(function_name='pp.import_fragments',input_path='/fragments.tsv')),
        'tile':('run_snapatac2',dict(function_name='pp.add_tile_matrix',input_path=StepOutputRef('import','matrix_path'))),
        'peak':('run_snapatac2',dict(function_name='pp.make_peak_matrix',input_path=StepOutputRef('import','matrix_path')))}


@pytest.mark.parametrize('producer,consumer,valid', [('import','inspect_scATAC',False),
    ('tile','inspect_scATAC',True),('peak','inspect_scATAC',True),
    ('import','call_scATAC_peaks',True),('tile','call_scATAC_peaks',True),('peak','call_scATAC_peaks',False)])
def test_future_contents_not_h5ad_suffix_determine_compatibility(producer,consumer,valid):
    ctx=context(selected(),consumer,observe=False)
    args={('path' if consumer=='inspect_scATAC' else 'input_path'):StepOutputRef(producer,'matrix_path')}
    if valid:validate_input_properties(args,ctx)
    else:
        with pytest.raises(ScientificDiagnostic,match='requires'):validate_input_properties(args,ctx)


def test_existing_storage_is_bounded_and_x_null_is_not_matrix(tmp_path,monkeypatch):
    path=tmp_path/'fragments.h5ad'
    a=ad.AnnData(obs=pd.DataFrame(index=['a','b']))
    a.obsm['fragment_paired']=np.ones((2,4));a.write_h5ad(path)
    # No broad AnnData read may materialize the genome-sized obsm during preparation.
    monkeypatch.setattr(ad,'read_h5ad',lambda *a,**k:pytest.fail('Broad H5AD read'))
    with pytest.raises(ScientificDiagnostic,match='requires x'):
        validate_input_properties(dict(path=str(path)),context(tool='inspect_scATAC'))
    validate_input_properties(dict(input_path=str(path)),context(tool='call_scATAC_peaks'))
    from agent.tools.analysis.peak_h5ad import scientific_properties
    assert scientific_properties(path,('x','fragment_storage'))==dict(x=False,fragment_storage=True)


@pytest.mark.parametrize('member,value', [('barcode_tag','RG'),('barcode_regex','^([^:]+)')])
def test_exact_choice_then_value_rechecks_same_candidate_without_model(tmp_path,member,value):
    from test_preparation_owner_contracts import Model,setup
    from agent.application import preparation_state as records
    from agent.schemas import VerificationResult,VerificationCheck
    model=Model('run_snapatac2','pp.make_fragment_file',roles=False)
    app,executor=setup(tmp_path,model)
    path=tmp_path/'source.bam';path.write_bytes(b'Unopened deterministic source identity')
    first=app.submit_turn('session','task','Convert this BAM to fragments.',expected_generation=0,
        execution_inputs=dict(input_path=str(path)))
    assert first.status=='clarification',first.to_dict()
    saved=app._application.sessions.load('session');origin=saved.interactions[0]
    cid=origin.preparation['candidate_id'];calls=len(model.calls)
    # Reload saved Preparation before answering; no new Candidate is acquired.
    from agent.application import InteractiveAgentApplication
    from agent.orchestration import PlanningModelProfile
    from agent.providers import PlanningModelFactoryRegistry
    app=InteractiveAgentApplication(app._application.workspace_root,registry=app._application.registry,
        executor=executor,approved_source_roots=(tmp_path,),
        model_profiles=(PlanningModelProfile('offline','scripted',model.model_id),),default_profile_id='offline',
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted':lambda _:model}))
    chosen=next(o for o in first.clarification_choices if o.get('member')==member)
    # Stop at the existing preflight gate, before downstream output-retention
    # selection. Both exact Answers and full Preparation run without a provider.
    checked=[]
    def preflight(plan):
        checked.append(plan)
        return VerificationResult(False,'plan',plan.plan_id,(VerificationCheck('withheld',False,'No science'),))
    executor.preflight=preflight
    # Provider access after method acquisition is forbidden for either Answer.
    def no_model(**kw):pytest.fail('Exact Answer attempted a model call')
    model.complete=no_model
    second=app.submit_turn('session','mechanism','',expected_generation=0,
        clarification_choice=dict(question_turn_id='task',handle=chosen['handle']))
    assert second.status=='clarification',second.to_dict()
    editor,=second.clarification_choices
    assert editor['action']=='parameter' and editor['member']==member
    assert editor['step_id']=='method' and editor['native_function']=='pp.make_fragment_file'
    done=app.submit_turn('session','value','',expected_generation=0,
        clarification_choice=dict(question_turn_id='mechanism',handle=editor['handle']),
        parameter_overrides=[dict(tool='run_snapatac2',native_function='pp.make_fragment_file',
            step_id='method',argument='kwargs',value={member:value})])
    assert done.error.code=='PLAN_PREFLIGHT_FAILED',done.to_dict()
    assert dict(checked[-1].steps[0].arguments['kwargs'])=={member:value}
    state=app._application.sessions.load('session')
    assert state.interactions[0]==origin
    assert {i.preparation['candidate_id'] for i in state.interactions}=={cid}
    assert len(model.calls)==calls
    answers=records.active_answers(state.interactions,state.interactions[-1].preparation)
    assert len(answers)==2


def candidate(matrix=None):
    from agent.orchestration.semantic_compiler import (SemanticPlanCandidate,SemanticPlanStep,
        SemanticRequestInputSource as Input,SemanticStepOutputSource as Output,
        SemanticRegisteredChoiceSource as Choice)
    steps=[SemanticPlanStep('extract','run_snapatac2',(Input('dataset','input_path'),Choice('function_name','pp.make_fragment_file'))),
        SemanticPlanStep('import','run_snapatac2',(Output('dataset','extract','fragments'),Choice('function_name','pp.import_fragments')))]
    if matrix:
        steps.append(SemanticPlanStep('matrix','run_snapatac2',(Output('dataset','import','dataset'),Choice('function_name',matrix))))
    steps.append(SemanticPlanStep('explore','inspect_scATAC',(Output('dataset','matrix' if matrix else 'import','dataset'),)))
    return SemanticPlanCandidate(tuple(steps))


def test_original_g1_candidate_rejected_purely_before_resource_preparation():
    from agent.orchestration.semantic_compiler import analyze_candidate,SemanticPlanCompileError
    objective='这是一份人类的单细胞 ATAC BAM 数据，帮我整理成 H5AD 文件，再看看数据的情况。'
    request=AgentRequest('original-g1',objective,dict(input_path='/unopened.bam'))
    def no_prepare(*a,**kw):pytest.fail('Known mismatch reached input materialization')
    rejected=analyze_candidate(request,candidate(),build_default_tool_registry(),prepare_inputs=no_prepare)
    assert rejected.error.code=='SCIENTIFIC_INPUT_PROPERTY_REQUIRED'
    assert request.prompt==objective
    assert [s.tool_name for s in candidate().steps]==['run_snapatac2','run_snapatac2','inspect_scATAC']
    # Either genuine matrix producer satisfies the content contract. No recipe
    # is selected or inserted by the Compiler; unbound resources remain needs.
    for function in ('pp.add_tile_matrix','pp.make_peak_matrix'):
        analysis=analyze_candidate(request,candidate(function),build_default_tool_registry())
        assert analysis.error is None,analysis.error
        assert analysis.candidate.steps==candidate(function).steps


def test_actual_transmitted_catalog_has_owner_condition_and_storage_contracts():
    from agent.orchestration.semantic_prompt import build_semantic_planning_catalog,build_semantic_planning_prompt
    registry=build_default_tool_registry();request=AgentRequest('catalog','Analyze the supplied BAM and explore its data.',dict(input_path='/hidden/source.bam'))
    catalog=build_semantic_planning_catalog(request,registry)
    payload=json.loads(build_semantic_planning_prompt(request,registry))
    contracts=payload['catalog']['scientific_contracts']
    assert contracts==json.loads(json.dumps(catalog['scientific_contracts']))
    functions=contracts['run_snapatac2']['functions']
    assert functions['pp.make_fragment_file']['conditional_parameters']==json.loads(json.dumps((BARCODE_EXTRACTION,)))
    assert functions['pp.make_fragment_file']['outputs']==['fragments'] or 'fragments' in functions['pp.make_fragment_file']['outputs']
    assert functions['pp.import_fragments']['produces']['dataset']['x'] is False
    assert functions['pp.add_tile_matrix']['produces']['dataset']['x'] is True
    assert contracts['inspect_scATAC']['requires']['dataset']==['x']
    assert contracts['call_scATAC_peaks']['requires']['dataset']==['paired_fragments']
    assert contracts['epizoo_embed_cells']['produces']['dataset']['x'] is False
    assert '/hidden/source.bam' not in json.dumps(payload)


def test_compact_epizoo_can_feed_representation_consumer_without_x():
    selected={'embedding':('epizoo_embed_cells',{}),'graph':('run_scanpy',dict(function_name='pp.neighbors',
        input_path=StepOutputRef('embedding','analysis_path'),kwargs={'use_rep':'X_epizoo'}))}
    args=dict(function_name='pp.neighbors',input_path=StepOutputRef('embedding','analysis_path'),kwargs={'use_rep':'X_epizoo'})
    ctx=context(selected,'run_scanpy',observe=False)
    validate_input_properties(args,ctx)
    from agent.tools.analysis.native_matrix import SCANPY
    assert SCANPY.prepare_operation(args,ctx)==args
    with pytest.raises(ScientificDiagnostic):
        validate_input_properties(dict(function_name='pp.pca',input_path=args['input_path']),ctx)


@pytest.mark.parametrize('function,kwargs', [('pp.log1p',{'obsm':'X_epizoo'}),('pp.scale',{'obsm':'X_epizoo'}),
    ('pp.log1p',{'layer':'counts'}),('pp.highly_variable_genes',{'layer':'counts'}),('pp.calculate_qc_metrics',{'use_raw':True})])
def test_matrix_selectors_preserve_native_non_x_targets(function,kwargs):
    args=dict(function_name=function,input_path=StepOutputRef('embedding','analysis_path'),kwargs=kwargs)
    validate_input_properties(args,context({'embedding':('epizoo_embed_cells',{})},'run_scanpy',observe=False))


def test_selected_layer_becomes_native_x_for_downstream_inspection(tmp_path):
    path=tmp_path/'layers.h5ad';a=ad.AnnData(obs=pd.DataFrame(index=['a','b']),var=pd.DataFrame(index=['x','y']))
    a.layers['counts']=np.eye(2);a.write_h5ad(path)
    selected={'pca':('run_scanpy',dict(input_path=str(path),function_name='pp.pca',count_slot='counts'))}
    validate_input_properties(dict(path=StepOutputRef('pca','matrix_path')),context(selected,'inspect_scATAC'))
    assert context(selected).planned(StepOutputRef('pca','matrix_path'))['x'] is True


@pytest.mark.parametrize('paired,strategy,measurement', [(True,'fragment','fragment_counts'),
    (True,'paired-insertion','paired_insertion_counts'),(False,'fragment','snapatac2_single_end_insertions')])
def test_planned_measurement_uses_actual_native_storage(paired,strategy,measurement):
    graph=selected();graph['import'][1]['kwargs']={'is_paired':paired}
    graph['peak'][1]['kwargs']={'counting_strategy':strategy}
    props=context(graph,observe=False).planned(StepOutputRef('peak','matrix_path'))
    assert props['measurement']==measurement
    assert props['fragment_storage'] is False


@pytest.mark.parametrize('sparse',[False,True])
def test_existing_x_accepts_anndata_fixed_byte_attributes(tmp_path,sparse):
    import h5py
    from scipy.sparse import csr_matrix
    path=tmp_path/'encoded.h5ad';a=ad.AnnData(csr_matrix(np.eye(3)) if sparse else np.eye(3));a.write_h5ad(path)
    with h5py.File(path,'r+') as f:
        for axis in ('obs','var'):
            value=f[axis].attrs['_index'];del f[axis].attrs['_index'];f[axis].attrs['_index']=np.bytes_(value)
        value=f['X'].attrs['encoding-type'];del f['X'].attrs['encoding-type'];f['X'].attrs['encoding-type']=np.bytes_(value)
    from agent.tools.analysis.peak_h5ad import scientific_properties
    assert scientific_properties(path,('x',))=={'x':True}


def test_barcode_choice_is_bound_to_original_source_bytes(tmp_path):
    from test_preparation_owner_contracts import Model,setup
    model=Model('run_snapatac2','pp.make_fragment_file',roles=False);app,executor=setup(tmp_path,model)
    path=tmp_path/'source.bam';path.write_bytes(b'original exact source')
    first=app.submit_turn('session','task','Convert the BAM to fragments.',expected_generation=0,execution_inputs=dict(input_path=str(path)))
    choice=first.clarification_choices[0]
    before=app._application.sessions.load('session').interactions[0]
    path.write_bytes(b'different exact source')
    model.complete=lambda **kwargs:pytest.fail('Changed source Answer called a model')
    result=app.submit_turn('session','choice','',expected_generation=0,
        clarification_choice=dict(question_turn_id='task',handle=choice['handle']))
    assert result.error.code=='LOCAL_RESOURCE_INTEGRITY_INVALID',result.to_dict()
    state=app._application.sessions.load('session')
    assert state.interactions[0]==before and not state.turns and not state.revisions and not executor.plans


def test_explicit_scalar_counting_strategy_is_the_actual_native_measurement():
    graph=selected();graph['peak'][1]['counting_strategy']='fragment'
    props=context(graph,observe=False).planned(StepOutputRef('peak','matrix_path'))
    assert props['counting_strategy']=='fragment' and props['measurement']=='fragment_counts'
