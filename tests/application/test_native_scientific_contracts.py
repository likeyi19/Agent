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


def test_peak_matrix_requires_a_region_source_and_accepts_the_macs3_port(tmp_path):
    from agent.orchestration.semantic_compiler import (AnalysisDisposition, SemanticPlanCandidate,
        SemanticPlanStep, SemanticRequestInputSource as Input, SemanticStepOutputSource as Output,
        SemanticRegisteredChoiceSource as Choice, analyze_candidate)

    registry = build_default_tool_registry()
    req = AgentRequest('regions', 'Use the selected cell population to build a peak matrix',
        dict(input_path=str(tmp_path / 'fragments.h5ad'), output_dir=str(tmp_path / 'output')))
    sources = (Input('dataset', 'input_path'), Choice('function_name', 'pp.make_peak_matrix'))
    unbound = SemanticPlanCandidate((SemanticPlanStep('matrix', 'run_snapatac2', sources),))
    analysis = analyze_candidate(req, unbound, registry)
    assert analysis.disposition is AnalysisDisposition.NEEDS_INPUT
    assert [(need.step_id, need.port) for need in analysis.requirements] == [('matrix', 'peaks')]
    macs = SemanticPlanStep('call', 'call_scATAC_peaks', (Input('dataset', 'input_path'),))
    matrix = SemanticPlanStep('matrix', 'run_snapatac2', sources + (Output('peaks', 'call', 'peaks'),))
    composed = analyze_candidate(req, SemanticPlanCandidate((matrix, macs)), registry,
        bindings={'call': {'effective_genome_size': 100, 'genome_size_basis': 'technical fixture'}})
    assert composed.disposition is AnalysisDisposition.COMPLETE, composed.error
    assert composed.operations[0].arguments['peaks_path'] == StepOutputRef('call', 'peaks_path')


def test_peak_region_table_and_explicit_list_are_effective_inputs(tmp_path):
    from agent.application.native_raw_handoff import prepare_operation

    path = tmp_path / 'regions.h5ad'
    dataset = ad.AnnData(np.eye(2))
    dataset.uns['peaks'] = pd.DataFrame({'region': ['chr1:1-5']})
    dataset.write_h5ad(path)
    ctx = context()
    args = dict(function_name='pp.make_peak_matrix', input_path=str(path), kwargs={'use_x': True})
    prepared = prepare_operation(args, ctx)
    assert prepared['kwargs']['use_rep'] == 'peaks'
    validate_input_properties(prepared, ctx)
    args['kwargs']['use_rep'] = ['chr1:5-10']
    assert prepare_operation(args, ctx) == args
    validate_input_properties(args, ctx)


@pytest.mark.parametrize('member,value', [('barcode_tag','RG'),('barcode_regex','^([^:]+)')])
def test_exact_choice_then_value_rechecks_same_candidate_without_model(tmp_path,member,value):
    from test_preparation_owner_contracts import Model,setup
    from agent.application import preparation_state as records
    from agent.schemas import VerificationResult,VerificationCheck
    model=Model('run_snapatac2','pp.make_fragment_file',roles=False)
    app,executor=setup(tmp_path,model)
    path=tmp_path/'source.bam';path.write_bytes(b'BAM\x01')
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
    assert functions['pp.make_fragment_file']['parameters']['conditions']==json.loads(json.dumps((BARCODE_EXTRACTION,)))
    assert functions['pp.make_fragment_file']['outputs']==['fragments'] or 'fragments' in functions['pp.make_fragment_file']['outputs']
    assert functions['pp.import_fragments']['outputs']['dataset']['x'] is False
    assert functions['pp.add_tile_matrix']['outputs']['dataset']['x'] is True
    assert contracts['inspect_scATAC']['inputs']['dataset']=={'x':True}
    assert contracts['call_scATAC_peaks']['inputs']['dataset']['paired_fragments'] is True
    assert contracts['epizoo_embed_cells']['outputs']['dataset']['x'] is False
    assert '/hidden/source.bam' not in json.dumps(payload)


def test_all_ai_scientific_selections_project_owner_facts_on_registered_ports():
    registry=build_default_tool_registry()
    covered=0
    for name in registry.names():
        tool=registry.get(name)
        if name=='invoke_native_function':
            assert tool.scientific_contract is None
            continue
        functions=(dict(function_name=function) for function,_ in tool.native_functions.functions) if tool.native_functions else ({},)
        for arguments in functions:
            contract=tool.selected_scientific_contract(arguments)
            assert contract.projection(),(name,arguments)
            assert set(contract.inputs)<={port.name for port in tool.semantic_planning.consumer_ports}
            assert set(contract.outputs)<={port.name for port in tool.semantic_planning.producer_ports}
            covered+=1
    assert covered==106


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


@pytest.mark.parametrize('inplace,kwargs,allowed', [
    (None, {}, False), (False, {}, False), (True, {}, True),
    (None, {'min_tsse': 0, 'max_tsse': None}, True),
    (None, {'min_tsse': None, 'max_tsse': 10}, False),
])
def test_selected_tss_field_contract_for_future_fragment_objects(inplace,kwargs,allowed):
    upstream={'import':('run_snapatac2',dict(function_name='pp.import_fragments'))}
    path=StepOutputRef('import','matrix_path')
    if inplace is not None:
        upstream['qc']=('run_snapatac2',dict(function_name='metrics.tsse',input_path=path,kwargs={'inplace':inplace}))
        path=StepOutputRef('qc','matrix_path')
    args=dict(function_name='pp.filter_cells',input_path=path,kwargs=kwargs)
    for observe in (False,True):
        ctx=context(upstream,observe=observe)
        if allowed:validate_input_properties(args,ctx)
        else:
            with pytest.raises(ScientificDiagnostic,match='obs.tsse') as error:
                validate_input_properties(args,ctx)
            assert error.value.code=='SCIENTIFIC_INPUT_PROPERTY_REQUIRED'


def test_compiler_rejects_wrong_future_type_and_missing_tss():
    from agent.orchestration.semantic_compiler import (analyze_candidate, SemanticPlanCandidate,
        SemanticPlanStep, SemanticStepOutputSource as Output, SemanticRegisteredChoiceSource as Choice)
    request=AgentRequest('filter-c','Filter the supplied dataset.',dict(input_path='/unopened.bam'))
    steps=candidate().steps[:2]
    registry=build_default_tool_registry()
    for step,port,property_name in [('extract','fragments','format'),('import','dataset','obs.tsse')]:
        consumer=SemanticPlanStep('filter','run_snapatac2',(Output('dataset',step,port),Choice('function_name','pp.filter_cells')))
        used=steps[:1] if step=='extract' else steps
        analysis=analyze_candidate(request,SemanticPlanCandidate((*used,consumer)),registry)
        assert analysis.error.code=='SCIENTIFIC_INPUT_PROPERTY_REQUIRED'
        assert property_name in str(analysis.error)


def test_named_field_lookup_is_not_limited_by_offer_slots(tmp_path):
    data=ad.AnnData(np.eye(4));data.obs['n_fragment']=[1000]*4;data.obs['tsse']=[8.0]*4
    for index in range(70):data.layers[f'other_{index:02}']=data.X.copy()
    data.layers['z_scores']=data.X.copy()
    path=tmp_path/'selected.h5ad';data.write_h5ad(path)
    ctx=context(tool='run_chromvar')
    args=dict(input_path=str(path),function_name='computeVariability')
    assert 'z_scores' not in ctx.existing(str(path)).slots('layers')
    validate_input_properties(args,ctx)
    validate_input_properties(dict(input_path=str(path),function_name='pp.filter_cells'),context())
    del data.layers['z_scores'];data.write_h5ad(path)
    with pytest.raises(ScientificDiagnostic,match='z_scores'):
        validate_input_properties(args,ctx)


def test_matrix_effect_and_selected_reduction_reach_consumer():
    graph=selected();counts=StepOutputRef('peak','matrix_path')
    graph['log']=('run_scanpy',dict(input_path=counts,function_name='pp.log1p'))
    graph['lsi']=('run_signac',dict(input_path=counts,function_name='RunSVD'))
    ctx=context(graph,'run_scanpy',observe=False)
    assert ctx.planned(counts)['measurement']=='paired_insertion_counts'
    assert ctx.planned(StepOutputRef('log','matrix_path'))['measurement']=='log1p_transformed_measurement'
    args=dict(input_path=StepOutputRef('lsi','matrix_path'),function_name='pp.neighbors',kwargs={'use_rep':'X_lsi'})
    validate_input_properties(args,ctx)
    validate_input_properties(args,context(graph,'run_scanpy'))


def test_bedgraph_subtype_is_not_ccan_membership():
    upstream={'ccan':('run_cicero',dict(function_name='generate_ccans'))}
    ctx=context(upstream,'run_macs3',observe=False)
    with pytest.raises(ScientificDiagnostic,match='table_format'):
        validate_input_properties(dict(input_path=StepOutputRef('ccan','table_path'),function_name='bdgopt'),ctx)


def test_numeric_tss_and_boolean_feature_mask_use_exact_dtypes(tmp_path):
    data=ad.AnnData(np.eye(4));data.obs['n_fragment']=[1000]*4;data.obs['tsse']=['not-a-number']*4
    data.var['selected']=['not-a-mask']*4
    path=tmp_path/'selected.h5ad';data.write_h5ad(path)
    for function,name in [('pp.filter_cells','obs.tsse'),('tl.spectral','var.selected')]:
        with pytest.raises(ScientificDiagnostic,match=name):
            validate_input_properties(dict(input_path=str(path),function_name=function),context())


def test_scanpy_default_graph_uses_native_fixed_slots_not_aliases(tmp_path):
    data=ad.AnnData(np.eye(4))
    data.uns['neighbors']={'connectivities_key':'different_connectivities','distances_key':'different_distances'}
    data.obsp['connectivities']=np.eye(4);data.obsp['distances']=np.eye(4)
    path=tmp_path/'selected.h5ad';data.write_h5ad(path)
    args=dict(input_path=str(path),function_name='tl.leiden')
    validate_input_properties(args,context(tool='run_scanpy'))
    with pytest.raises(ScientificDiagnostic,match='graph.snapatac2_distances'):
        validate_input_properties(args,context())
    with pytest.raises(ScientificDiagnostic,match='not both'):
        validate_input_properties(dict(args,kwargs={'obsp':'connectivities','neighbors_key':'neighbors'}),context(tool='run_scanpy'))


def test_bedgraph_observer_does_not_treat_file_extension_as_format(tmp_path):
    path=tmp_path/'table.bedgraph';path.write_text('peak\tCCAN\nchr1_0_10\t1\n')
    with pytest.raises(ScientificDiagnostic,match='table_format'):
        validate_input_properties(dict(input_path=str(path),function_name='bdgopt'),context(tool='run_macs3'))
    path.write_text('chr1\t0\t10\t2\nchr1\t10\t20\tinvalid\n')
    with pytest.raises(ScientificDiagnostic,match='table_format'):
        validate_input_properties(dict(input_path=str(path),function_name='bdgopt'),context(tool='run_macs3'))


def test_measurement_follows_exact_layer_selection_and_retained_other_slots(tmp_path):
    data=ad.AnnData(np.eye(4));data.layers['selected']=np.eye(4);data.layers['other']=np.eye(4)
    data.uns['matrix_semantics']='insertion_counts'
    data.uns['layer_semantics']={'selected':'fragment_counts','other':'insertion_counts'}
    path=tmp_path/'source';data.write_h5ad(path)
    graph={'log':('run_scanpy',dict(input_path=str(path),function_name='pp.log1p',kwargs={'layer':'selected'})),
           'pca':('run_scanpy',dict(input_path=StepOutputRef('log','matrix_path'),function_name='pp.pca',count_slot='selected'))}
    ctx=context(graph,'compute_motif_accessibility')
    for observe in (False,True):
        from dataclasses import replace
        scoped=replace(ctx,observe_existing=observe)
        with pytest.raises(ScientificDiagnostic,match='count_values.layers.selected'):
            validate_input_properties(dict(input_path=StepOutputRef('log','matrix_path'),count_slot='selected'),scoped)
        with pytest.raises(ScientificDiagnostic,match='count_values.X'):
            validate_input_properties(dict(input_path=StepOutputRef('pca','matrix_path')),scoped)
        validate_input_properties(dict(input_path=StepOutputRef('log','matrix_path'),count_slot='other'),scoped)
    data.uns['layer_semantics']['selected']='log1p_transformed_measurement'
    data.write_h5ad(path)
    with pytest.raises(ScientificDiagnostic,match='count_values.layers.selected'):
        validate_input_properties(dict(input_path=str(path),count_slot='selected'),ctx)


def test_selected_physical_format_checks_magic_and_structure_not_suffix(tmp_path):
    import gzip
    from agent.tools.analysis.peak_h5ad import scientific_properties
    bam=tmp_path/'bam.tsv';bam.write_bytes(gzip.compress(b'BAM\x01'))
    fragments=tmp_path/'fragments.bam';fragments.write_bytes(gzip.compress(b'chr1\t1\t2\tbc\t1\n'))
    opaque=tmp_path/'opaque.bam';opaque.write_bytes(b'unknown')
    assert scientific_properties(bam,('format',))['format']=='bam'
    assert scientific_properties(fragments,('format',))['format']=='fragments'
    assert scientific_properties(opaque,('format',))['format']=='non_hdf5'
    for path,function in ((bam,'pp.import_fragments'),(fragments,'pp.make_fragment_file'),(opaque,'pp.make_fragment_file')):
        with pytest.raises(ScientificDiagnostic,match='format'):
            validate_input_properties(dict(input_path=str(path),function_name=function),context())


def test_mixed_graph_requirements_are_per_selected_operand():
    scan={'graph':('run_scanpy',dict(function_name='pp.neighbors'))}
    mixed={**scan,'knn':('run_snapatac2',dict(input_path=StepOutputRef('graph','matrix_path'),function_name='pp.knn'))}
    source=StepOutputRef('knn','matrix_path')
    validate_input_properties(dict(input_path=source,function_name='tl.leiden'),context(mixed))
    validate_input_properties(dict(input_path=source,function_name='tl.leiden'),context(mixed,'run_scanpy'))
    with pytest.raises(ScientificDiagnostic,match='scanpy_distances'):
        validate_input_properties(dict(input_path=source,function_name='tl.paga'),context(mixed,'run_scanpy'))
    validate_input_properties(dict(input_path=source,function_name='tl.paga',kwargs={'model':'v1.0'}),context(mixed,'run_scanpy'))


def test_named_distance_selector_follows_exact_physical_slot(tmp_path):
    from scipy import sparse
    data=ad.AnnData(np.eye(4));data.obsp['distances']=sparse.eye(4,format='csr')
    data.obsp['alternate']=sparse.eye(4,format='csr')
    data.uns['n']={'distances_key':'distances','connectivities_key':'alternate'}
    path=tmp_path/'named.h5ad';data.write_h5ad(path)
    upstream={'knn':('run_snapatac2',dict(input_path=str(path),function_name='pp.knn'))}
    args=dict(input_path=StepOutputRef('knn','matrix_path'),function_name='tl.paga',kwargs={'neighbors_key':'n'})
    with pytest.raises(ScientificDiagnostic,match='scanpy_distances'):
        validate_input_properties(args,context(upstream,'run_scanpy'))
    validate_input_properties(dict(args,kwargs={'neighbors_key':'n','model':'v1.0'}),context(upstream,'run_scanpy'))


def test_exact_graph_keeps_writer_fact_after_representation_changes(tmp_path):
    from agent.tools.analysis.native_clustering import graph_record, valid_graphs
    from agent.tools.analysis.peak_h5ad import scientific_properties
    from scipy import sparse
    data=ad.AnnData(np.eye(4))
    data.obsm['X_spectral']=np.eye(4)
    data.obsp['distances']=sparse.eye(4,format='csr')
    data.obsp['connectivities']=sparse.eye(4,format='csr')
    data.uns['neighbors']={'connectivities_key':'connectivities','distances_key':'distances'}
    record=graph_record(data,'snapatac2','pp.knn',{})
    data.obsm['X_spectral']=np.ones((4,4))
    assert valid_graphs(data,[record])==[]
    assert valid_graphs(data,[record],require_representation=False)==[record]
    path=tmp_path/'mixed.h5ad';data.write_h5ad(path)
    names=('graph.snapatac2_distances','graph.scanpy_connectivities.neighbors','graph.scanpy_distances.neighbors')
    assert scientific_properties(path,names,graph_inputs=[record])==dict(zip(names,(True,True,False)))
    assert scientific_properties(path,names)['graph.snapatac2_distances'] is False


def test_named_peak_frame_checks_first_native_column(tmp_path):
    from agent.tools.analysis.peak_h5ad import scientific_properties
    data=ad.AnnData(np.eye(2));path=tmp_path/'regions.h5ad'
    data.uns['regions']=pd.DataFrame({'score':[1.0],'region':['chr1:1-2']})
    data.write_h5ad(path)
    assert scientific_properties(path,('regions.regions',))=={'regions.regions':False}
    data.uns['regions']=data.uns['regions'][['region','score']]
    data.write_h5ad(path)
    assert scientific_properties(path,('regions.regions',))=={'regions.regions':True}


def test_snap_knn_obsm_x_does_not_fall_back_to_matrix_x(tmp_path):
    from agent.tools.analysis.native_clustering import graph_record, array_sha
    data=ad.AnnData(np.eye(4));data.obsm['X']=np.full((4,2),2.0);data.obsp['distances']=np.eye(4)
    path=tmp_path/'source';data.write_h5ad(path)
    args=dict(input_path=str(path),function_name='pp.knn',kwargs={'use_rep':'X'})
    validate_input_properties(args,context())
    result=graph_record(data,'snapatac2','pp.knn',args['kwargs'])
    assert result['representation_slot']=='obsm.X'
    assert result['representation_sha256']==array_sha(data.obsm['X'])


def test_layer_promotion_preserves_observed_measurement(tmp_path):
    from agent.tools.analysis.native_matrix import prepare_input
    data=ad.AnnData(np.eye(4));data.layers['log']=np.eye(4);data.uns['layer_semantics']={'log':'log1p_transformed_measurement'}
    path=tmp_path/'source';data.write_h5ad(path)
    matrix,metadata=prepare_input(dict(input_path=str(path),count_slot='log'),tmp_path)
    assert matrix.uns['matrix_semantics']==metadata['matrix_semantics']=='log1p_transformed_measurement'


def test_scanpy_neighborhood_defaults_and_named_x_are_distinct(tmp_path):
    from agent.tools.analysis.native_clustering import array_sha, graph_record
    path=tmp_path/'compact'
    data=ad.AnnData(shape=(8,55))
    data.obsm['X_pca']=np.eye(8);data.obsm['X']=np.ones((8,3));data.write_h5ad(path)
    args=dict(input_path=str(path),function_name='pp.neighbors')
    ctx=context(tool='run_scanpy')
    validate_input_properties(args,ctx)
    validate_input_properties(dict(args,kwargs={'use_rep':'X'}),ctx)
    data.obsp['connectivities']=np.eye(8);data.obsp['distances']=np.eye(8)
    data.uns['neighbors']={'connectivities_key':'connectivities','distances_key':'distances'}
    record=graph_record(data,'scanpy','pp.neighbors',{'use_rep':'X'})
    assert record['representation_slot']=='obsm.X'
    assert record['representation_sha256']==array_sha(data.obsm['X'])
    compact=context({'embed':('epizoo_embed_cells',{})},'run_scanpy',observe=False)
    with pytest.raises(ScientificDiagnostic,match='scanpy_auto_rep'):
        validate_input_properties(dict(args,input_path=StepOutputRef('embed','analysis_path')),compact)


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
    path=tmp_path/'source.bam';path.write_bytes(b'BAM\x01')
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
