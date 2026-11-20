"""Production planning and exact resource binding; no native execution in this witness."""
import json
from types import SimpleNamespace
import pytest
from agent.application import InteractiveAgentApplication
from agent.orchestration import PlanningModelProfile, build_default_tool_registry
from agent.providers import PlanningModelFactoryRegistry
from agent.schemas import StepOutputRef
from test_human_epizoo_mapping import WithholdScience


def gene_catalog():
    from agent.application.input_sets import ScientificInputSet
    return [ScientificInputSet('genes', 'Gene annotation', dict(
        gene_anno_path='/references/genes.gtf', gene_anno_sha256='b'*64)).definition()]


@pytest.mark.parametrize('species,expected',[('human','human'),('Human','Human'),('HUMAN','HUMAN'),
                                            ('mouse','mouse'),('Mouse','Mouse'),('MOUSE','MOUSE')])
def test_declared_species_binds_the_exact_cellmarker_value(monkeypatch,species,expected):
    from agent.application.annotation_preparation import preparer
    from agent.application.raw_handoff import AnnotationDeclaration,declaration_dict
    from agent.tools.analysis import sequence_prediction
    from agent.tools.data import cellmarker
    monkeypatch.setattr(sequence_prediction,'library',lambda:SimpleNamespace(entries=lambda:[dict(kind='cellmarker',id='installed')]))
    monkeypatch.setattr(cellmarker,'CellMarkerEvidence',lambda *a:SimpleNamespace(identity={},metadata={'species':['Human','Mouse']}))
    captured=dict(annotation=declaration_dict(AnnotationDeclaration(species=species)))
    args=dict(group_column='original_groups',activity_path=StepOutputRef('activity','matrix_path'))
    prepare=preparer(None,None,captured)
    tool=build_default_tool_registry().get('annotate_scATAC_cell_types')
    bound=prepare(tool,args,{})
    assert bound['species']==expected
    assert prepare(tool,bound,{})==bound
    with pytest.raises(ValueError,match='differs from the captured declaration'):
        prepare(tool,dict(args,species='unrelated species'),{})
    assert captured['annotation']['species']==species


@pytest.mark.parametrize('package,function',[('snapatac2','tl.leiden'),('scanpy','tl.leiden'),
    ('episcanpy','tl.louvain'),('episcanpy','tl.kmeans'),('episcanpy','tl.hc')])
def test_annotation_grouping_uses_the_actual_cluster_before_a_retaining_umap(monkeypatch,package,function):
    from agent.application.annotation_preparation import preparer
    from agent.application.raw_handoff import AnnotationDeclaration,declaration_dict
    from agent.tools.analysis import sequence_prediction
    from agent.tools.data import cellmarker
    monkeypatch.setattr(sequence_prediction,'library',lambda:SimpleNamespace(entries=lambda:[dict(kind='cellmarker',id='installed')]))
    monkeypatch.setattr(cellmarker,'CellMarkerEvidence',lambda *a:SimpleNamespace(identity={},metadata={'species':['Human','Mouse']}))
    captured=dict(annotation=declaration_dict(AnnotationDeclaration(species='Human')))
    cluster=StepOutputRef('cluster','matrix_path');umap=StepOutputRef('umap','matrix_path')
    args=dict(input_path=umap,grouping_path=umap,activity_path=StepOutputRef('activity','matrix_path'))
    selected=dict(cluster=('run_'+package,dict(function_name=function,input_path='/original.h5ad')),
        umap=('run_'+package,dict(function_name='tl.umap',input_path=cluster)))
    prepare=preparer(None,None,captured);tool=build_default_tool_registry().get('annotate_scATAC_cell_types')
    bound=prepare(tool,args,selected)
    assert bound['grouping_path']==cluster and bound['input_path']==umap
    assert args['grouping_path']==umap and prepare(tool,bound,selected)==bound


@pytest.mark.parametrize('function,kwargs',[('pp.log1p',{}),('tl.leiden',{'inplace':False})])
def test_annotation_grouping_does_not_invent_a_cluster_from_unqualified_slots(monkeypatch,function,kwargs):
    from agent.application.annotation_preparation import preparer
    from agent.orchestration.semantic_compiler import ScientificInputRequired
    from agent.application.raw_handoff import AnnotationDeclaration,declaration_dict
    from agent.tools.analysis import sequence_prediction
    from agent.tools.data import cellmarker
    monkeypatch.setattr(sequence_prediction,'library',lambda:SimpleNamespace(entries=lambda:[dict(kind='cellmarker',id='installed')]))
    monkeypatch.setattr(cellmarker,'CellMarkerEvidence',lambda *a:SimpleNamespace(identity={},metadata={'species':['Human','Mouse']}))
    captured=dict(annotation=declaration_dict(AnnotationDeclaration(species='Human')))
    ref=StepOutputRef('selected','matrix_path')
    args=dict(input_path=ref,grouping_path=ref,activity_path=StepOutputRef('activity','matrix_path'))
    selected=dict(selected=('run_scanpy',dict(function_name=function,kwargs=kwargs,input_path='/original.h5ad')))
    with pytest.raises(ScientificInputRequired,match='clustering'):
        preparer(None,None,captured)(build_default_tool_registry().get('annotate_scATAC_cell_types'),args,selected)


def test_compiled_annotation_records_the_prepared_clustering_dependency(monkeypatch,tmp_path):
    from agent.application.annotation_preparation import preparer
    from agent.application.raw_handoff import AnnotationDeclaration,declaration_dict
    from agent.tools.analysis import sequence_prediction
    from agent.tools.data import cellmarker
    from agent.orchestration.active_context import input_preparation
    from agent.orchestration.semantic_compiler import (SemanticPlanCandidate,SemanticPlanStep as Step,
        SemanticRequestInputSource as Input,SemanticStepOutputSource as Output,
        SemanticRegisteredChoiceSource as Choice,compile_semantic_plan,build_semantic_compiler_contract)
    from agent.schemas import AgentRequest
    monkeypatch.setattr(sequence_prediction,'library',lambda:SimpleNamespace(entries=lambda:[dict(kind='cellmarker',id='installed')]))
    monkeypatch.setattr(cellmarker,'CellMarkerEvidence',lambda *a:SimpleNamespace(identity={},metadata={'species':['Human','Mouse']}))
    registry=build_default_tool_registry()
    candidate=SemanticPlanCandidate((
        Step('cluster','run_snapatac2',(Input('dataset','input_path'),Choice('function_name','tl.leiden'))),
        Step('umap','run_snapatac2',(Output('dataset','cluster','dataset'),Choice('function_name','tl.umap'))),
        Step('annotation','annotate_scATAC_cell_types',(Output('source','umap','dataset'),Output('groups','umap','dataset'),
            Choice('activity_method','episcanpy.tl.geneactivity'),Input('gene_anno','gene_anno_path')))))
    request=AgentRequest('grouping-binding','Technical compilation only',dict(input_path='/original.h5ad',
        output_dir=str(tmp_path/'outputs'),gene_anno_path='/genes.gtf',gene_anno_sha256='b'*64))
    with input_preparation(preparer(None,None,dict(annotation=declaration_dict(AnnotationDeclaration(species='Human'))))):
        compiled=compile_semantic_plan(request,candidate,registry,build_semantic_compiler_contract(registry))
    annotation=compiled.steps[-1]
    assert annotation.arguments['grouping_path']==StepOutputRef('cluster','matrix_path')
    assert annotation.arguments['input_path']==StepOutputRef('umap','matrix_path')
    assert set(annotation.depends_on)=={'cluster','umap'}


@pytest.mark.parametrize('species,available', [('Homo sapiens',['Human']), ('unavailable',['Human','Mouse']), ('human',['Human','HUMAN'])])
def test_species_requires_unique_casefold_equality(species,available):
    from agent.tools.data.cellmarker import resolve_species
    with pytest.raises(ValueError,match='no unique'):
        resolve_species(species,available)


@pytest.mark.parametrize('declaration,resolved',[('human','Human'),('Human','Human'),('HUMAN','Human'),('mouse','Mouse'),('MOUSE','Mouse')])
def test_species_uses_only_actual_reference_values(declaration,resolved):
    from agent.tools.data.cellmarker import resolve_species
    assert resolve_species(declaration,['Human','Mouse'])==resolved
    assert resolve_species(None,['Human','Mouse']) is None


@pytest.mark.parametrize('choice,invalid_port', [('generate',None), ('skip',None), ('available',None),
                                               ('generate','activity_source'), ('generate','groups'),
                                               ('generate','ambiguous_native_dataset'), ('generate','omit_umap'),
                                               ('generate','omit_annotation'), ('generate','wrong_output'),
                                               ('generate','wrong_identity')])
def test_optional_umap_click_preserves_supplied_groups_and_ordinary_planning(tmp_path, monkeypatch, choice, invalid_port):
    import anndata as ad
    import numpy as np
    import pandas as pd
    from scipy import sparse
    from agent.tools.analysis import sequence_prediction, annotation_input
    from agent.tools.data import cellmarker
    from agent.application.raw_handoff import AnnotationDeclaration, declaration_dict
    from agent.application.input_sets import ScientificInputSet
    monkeypatch.setattr(sequence_prediction, 'library', lambda: SimpleNamespace(
        entries=lambda: [dict(kind='cellmarker', id='installed')]))
    monkeypatch.setattr(cellmarker, 'CellMarkerEvidence', lambda *a: SimpleNamespace(identity={},metadata={'species':['Human','Mouse']}))
    monkeypatch.setattr(annotation_input, 'observe', annotation_input._observe)
    calls = []
    class Model:
        model_id = 'optional-umap'
        def complete(self, *, prompt, response_schema):
            value = json.loads(prompt); calls.append(value)
            if 'turn_schema_version' in value:
                assert sum('turn_schema_version' in row for row in calls) == 1
                return json.dumps(dict(turn_schema_version=1, decision=dict(kind='request_roles',
                    target='annotate_scATAC_cell_types', required_roles=[], roles=[],
                    annotation=declaration_dict(AnnotationDeclaration(source='source.0')),
                    arguments=[dict(tool='annotate_scATAC_cell_types', argument='group_column',
                        source='source.0', mode='offered', value='groups', literal='groups', evidence='groups')])))
            if 'output_selection_schema_version' in value:
                assert value['question'] == 'Annotate using groups.'
                assert value['current_utterance'] == ('' if choice != 'available' else 'Annotate using groups.')
                outputs=[dict(name='annotation',step_id='annotation',output_key='annotated_h5ad_path')]
                if choice=='generate': outputs.append(dict(name='umap',step_id='umap',output_key='matrix_path'))
                if value.get('requested_output_uses'):
                    assert set(value['requested_output_uses']) == {'visualization_labels', 'visualization_coordinates'}
                    for output in outputs:
                        output['use'] = ['visualization_labels' if output['step_id']=='annotation' else 'visualization_coordinates']
                if invalid_port == 'omit_umap': outputs = [o for o in outputs if o['step_id'] != 'umap']
                if invalid_port == 'omit_annotation': outputs = [o for o in outputs if o['step_id'] != 'annotation']
                if invalid_port == 'wrong_output': outputs[1]['output_key'] = 'n_cells'
                if invalid_port == 'wrong_identity': outputs[1]['step_id'] = 'spectral'
                return json.dumps(dict(outputs=outputs))
            task=value['research_context']['task']
            if choice != 'available': assert task['annotation_umap']['generate'] == (choice=='generate')
            source=lambda target, **kw: dict(target=target,source=kw)
            steps=[]
            if choice=='generate':
                steps.append(dict(step_id='inspect',tool='inspect_scATAC',control_dependencies=[],
                    sources=[source('dataset',kind='input',input='input_path')]))
                for sid,function,parent in [('features','pp.select_features','inspect'),('spectral','tl.spectral','features'),('umap','tl.umap','spectral')]:
                    steps.append(dict(step_id=sid,tool='run_snapatac2',control_dependencies=[],sources=[
                        source('dataset',**(dict(kind='input',input='input_path') if parent is None else dict(kind='step_port',step=parent,source_port='dataset'))),
                        source('function_name',kind='registered_choice',choice=function)]))
            steps.append(dict(step_id='annotation',tool='annotate_scATAC_cell_types',control_dependencies=[],sources=[
                source('source',kind='input',input='input_path'),source('group_column',kind='input',input='group_column'),
                source('activity_method',kind='registered_choice',choice='episcanpy.tl.geneactivity')]))
            if invalid_port == 'ambiguous_native_dataset' and 'repair' not in value:
                steps[2]['sources'][0]=source('dataset',kind='step',step='features')
            elif invalid_port in {'activity_source', 'groups'} and 'repair' not in value:
                # Reproduce both actual browser failures. The existing compiler
                # must reject inspection as a supplemental native annotation input.
                steps[-1]['sources'].append(source(invalid_port,kind='step_port',step='inspect',source_port='dataset'))
            if 'repair' in value:
                expected='AMBIGUOUS_SOURCE_PORT' if invalid_port=='ambiguous_native_dataset' else 'WRONG_SOURCE_PORT'
                assert expected in json.dumps(value['repair'])
            return json.dumps(dict(schema_version=4,decision=dict(kind='plan',steps=steps)))
    registry=build_default_tool_registry(); executor=WithholdScience(registry)
    profile=PlanningModelProfile('planning','scripted','optional-umap')
    app=InteractiveAgentApplication(tmp_path/'workspace',registry=registry,executor=executor,
        model_profiles=(profile,),default_profile_id='planning',approved_source_roots=(tmp_path,),
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted':lambda _:Model()}))
    app.create_session('session')
    a=ad.AnnData(sparse.eye(6,format='csr'),obs=pd.DataFrame(dict(groups=pd.Categorical(['A']*3+['B']*3)),
        index=[f'cell{i}' for i in range(6)]),uns=dict(species='human',assembly='hg38'))
    if choice=='available': a.obsm['X_umap']=np.arange(12,dtype=float).reshape(6,2)
    path=tmp_path/'peaks.h5ad';a.write_h5ad(path)
    record=app.resources.register('source',path,label='Peak counts',attribution='Synthetic interaction fixture')
    binding=app.resources.compose_group([record.resource_id],attachment_candidates=True)
    inputs=tuple(ScientificInputSet(**row) for row in gene_catalog())
    outcome=app.submit_turn('session','request','Annotate using groups.',expected_generation=0,registered_input=binding,input_sets=inputs)
    if choice!='available':
        assert outcome.status=='clarification',outcome.to_dict()
        assert outcome.response.text.startswith('Generate UMAP?')
        assert {row['label'] for row in outcome.clarification_choices} == {'Generate UMAP','Skip UMAP'}
        assert not executor.plans
        outcome=app.submit_turn('session','click','',expected_generation=0,input_sets=inputs,
            clarification_choice=dict(question_turn_id='request',handle=next(c['handle'] for c in outcome.clarification_choices if c['choice_identity']=='annotation-umap:'+choice)))
    if invalid_port in {'omit_umap', 'omit_annotation', 'wrong_output', 'wrong_identity'}:
        assert outcome.error.code == 'PLANNER_OUTPUT_INVALID', outcome.to_dict()
        assert not executor.plans
        return
    assert outcome.error.code=='TEST_SCIENCE_WITHHELD',outcome.to_dict()
    plan=executor.plans[0]
    assert [s.arguments.get('function_name') for s in plan.steps[:-1]] == (
        [None,'pp.select_features','tl.spectral','tl.umap'] if choice=='generate' else [])
    assert plan.steps[-1].arguments['group_column']=='groups'
    assert 'grouping_path' not in plan.steps[-1].arguments
    assert 'activity_source_path' not in plan.steps[-1].arguments
    assert sum('repair' in row for row in calls)==int(invalid_port is not None)
    assert sum('turn_schema_version' in row for row in calls)==1


@pytest.mark.parametrize('selected', [None, 'resource.other'])
def test_annotation_reference_component_choices_and_exact_selection(monkeypatch, selected):
    from agent.application.annotation_preparation import preparer
    from agent.orchestration.semantic_compiler import ScientificInputRequired
    from agent.application.raw_handoff import AnnotationDeclaration, declaration_dict
    from agent.tools.analysis import sequence_prediction
    from agent.tools.data import cellmarker
    from agent.application.input_sets import ScientificInputSet
    monkeypatch.setattr(sequence_prediction, 'library', lambda: SimpleNamespace(
        entries=lambda: [dict(kind='cellmarker', id='installed')]))
    monkeypatch.setattr(cellmarker, 'CellMarkerEvidence', lambda *a: SimpleNamespace(identity={},metadata={'species':['Human','Mouse']}))
    rows = gene_catalog() + [ScientificInputSet('other', 'Other gene reference', dict(
        gene_anno_path='/references/other.gtf', gene_anno_sha256='c'*64)).definition()]
    prepare = preparer(None, None, dict(resource_catalog=rows,
        annotation=declaration_dict(AnnotationDeclaration(resource=selected))))
    args = dict(group_column='supplied', activity_method='episcanpy.tl.geneactivity')
    tool = build_default_tool_registry().get('annotate_scATAC_cell_types')
    if selected is None:
        with pytest.raises(ScientificInputRequired) as failure:
            prepare(tool, args, {})
        assert {row['handle'] for row in failure.value.options} == {'resource.genes', 'resource.other'}
    else:
        bound = prepare(tool, args, {})
        assert bound['gene_anno_path'] == '/references/other.gtf'
        assert bound['gene_anno_sha256'] == 'c'*64


@pytest.mark.parametrize('evidence_available,failure_then_answer', [(True, False), (False, False), (False, True)])
def test_ordinary_annotation_request_plans_registered_prerequisites(tmp_path, monkeypatch, evidence_available, failure_then_answer):
    from agent.tools.analysis import sequence_prediction
    from agent.tools.data import cellmarker
    # Resource binding is scripted, with exact captured identity, not biological evidence.
    lib=SimpleNamespace(entries=lambda:[dict(kind='cellmarker',id='installed')] if evidence_available else [])
    monkeypatch.setattr(sequence_prediction,'library',lambda:lib)
    monkeypatch.setattr(cellmarker,'CellMarkerEvidence',lambda *a:SimpleNamespace(identity={'sha256':'a'*64},metadata={'species':['Human','Mouse']}))
    calls=[]
    continuing=[]
    reject=[]
    additions=[]
    class Model:
        model_id='planning-witness'
        def complete(self,*,prompt,response_schema):
            value=json.loads(prompt);calls.append(value)
            if 'turn_schema_version' in value:
                if continuing:
                    from agent.application.raw_handoff import AnnotationDeclaration,declaration_dict
                    return json.dumps(dict(turn_schema_version=1,decision=dict(kind='answer_roles',pending='@roles',roles=[],
                        annotation=declaration_dict(additions.pop(0) if additions else AnnotationDeclaration()))))
                return json.dumps(dict(turn_schema_version=1,decision=dict(kind='execute_plan',target='annotate_scATAC_cell_types')))
            if 'output_selection_schema_version' in value:
                return json.dumps(dict(outputs=[dict(name='annotation',step_id='annotation',output_key='annotated_h5ad_path')]))
            if reject:
                return json.dumps(dict(schema_version=4,decision=dict(kind='unsupported',reason='Source declarations remain unknown.')))
            source=lambda target, **kw:dict(target=target,source=kw)
            steps=[dict(step_id='activity',tool='run_episcanpy',sources=[
                source('dataset',kind='input',input='input_path'),
                source('function_name',kind='registered_choice',choice='tl.geneactivity')],control_dependencies=[]),dict(step_id='groups',tool='run_snapatac2',sources=[
                source('dataset',kind='input',input='input_path'),
                source('function_name',kind='registered_choice',choice='tl.leiden')],control_dependencies=[]),
                dict(step_id='annotation',tool='annotate_scATAC_cell_types',sources=[
                source('source',kind='input',input='input_path'),
                source('groups',kind='step_port',step='groups',source_port='dataset'),
                source('activity',kind='step_port',step='activity',source_port='dataset')],control_dependencies=[])]
            return json.dumps(dict(schema_version=4,decision=dict(kind='plan',steps=steps)))
    profile=PlanningModelProfile('planning','scripted','planning-witness')
    registry=build_default_tool_registry();executor=WithholdScience(registry)
    app=InteractiveAgentApplication(tmp_path/'workspace',registry=registry,executor=executor,
        model_profiles=(profile,),default_profile_id='planning',
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted':lambda _:Model()}))
    app.create_session('session')
    import anndata as ad
    import numpy as np
    from scipy import sparse
    path=tmp_path/'cells.h5ad';ad.AnnData(sparse.eye(4,format='csr')).write_h5ad(path)
    outcome=app.submit_turn('session','annotation','Please annotate these scATAC cells.',expected_generation=0,
        execution_inputs=dict(input_path=str(path)))
    if not evidence_available:
        state=app._application.sessions.load('session')
        pending=state.interactions[-1]
        assert pending.status=='clarification',outcome.to_dict()
        assert pending.prerequisite is None
        assert pending.preparation['task']['context']['intake_owner']=='annotation'
        assert 'CellMarker' in ' '.join(m for r in pending.admitted['preparation_result']['requirements'] for m in r.get('messages', ()))
        assert not state.turns and not executor.plans
        lib.entries=lambda:[dict(kind='cellmarker',id='installed')]
        continuing.append(True)
        if failure_then_answer:
            reject.append(True)
            from agent.application.raw_handoff import AnnotationDeclaration
            additions.append(AnnotationDeclaration(species='human',assembly='hg38',resource='resource.genes'))
        outcome=app.submit_turn('session','continue','Use the now-installed CellMarker resource and continue.',expected_generation=0)
        continued=app._application.sessions.load('session').interactions[-1]
        assert continued.admitted['inputs']['input_path']==str(path)
        assert continued.preparation['task_origin']=='annotation'
        assert 'continuation' not in continued.admitted
        if failure_then_answer:
            # Changing a deterministic declaration must reuse the candidate.
            # The provider's queued rejection is deliberately never requested.
            assert sum('semantic_prompt_version' in value for value in calls)==1
            assert continued.admitted['annotation']['species']=='human'
            assert continued.admitted['annotation']['assembly']=='hg38'
            assert continued.admitted['annotation']['resource']=='resource.genes'
    assert outcome.error and outcome.error.code=='TEST_SCIENCE_WITHHELD',outcome.to_dict()
    activity,groups,annotation=executor.plans[0].steps
    assert annotation.tool_name=='annotate_scATAC_cell_types'
    assert annotation.arguments['grouping_path']==StepOutputRef('groups','matrix_path')
    assert set(annotation.depends_on)=={'groups','activity'}
    assert annotation.arguments['cellmarker_id']=='installed'
    assert dict(annotation.arguments['cellmarker_identity'])=={'sha256':'a'*64}
    assert len([c for c in calls if 'schema_version' in c])<=2


def test_unaccepted_activity_and_reuse_only_use_existing_scoped_question(monkeypatch):
    from agent.application.annotation_preparation import preparer
    from agent.orchestration.semantic_compiler import ScientificInputRequired
    from agent.application.raw_handoff import AnnotationDeclaration,declaration_dict
    from agent.tools.analysis import sequence_prediction
    from agent.tools.data import cellmarker
    lib=SimpleNamespace(entries=lambda:[dict(kind='cellmarker',id='installed')])
    monkeypatch.setattr(sequence_prediction,'library',lambda:lib)
    monkeypatch.setattr(cellmarker,'CellMarkerEvidence',lambda *a:SimpleNamespace(identity={'sha256':'a'*64},metadata={'species':['Human','Mouse']}))
    tool=build_default_tool_registry().get('annotate_scATAC_cell_types')
    prepare=preparer(None,None,{})
    with pytest.raises(ScientificInputRequired,match='file locator alone'):
        prepare(tool,dict(grouping_path=StepOutputRef('groups','matrix_path'),activity_path='/loose/genes.h5ad'),None)
    prepare=preparer(None,None,dict(annotation=declaration_dict(AnnotationDeclaration(grouping_strategy='reuse_only'))))
    with pytest.raises(ScientificInputRequired,match='reuse existing groups'):
        prepare(tool,dict(grouping_path=StepOutputRef('groups','matrix_path')),None)


def test_selected_gene_reference_binding_preserves_native_activity_source_semantics(monkeypatch):
    from agent.application.annotation_preparation import preparer
    from agent.orchestration.semantic_compiler import ScientificInputRequired
    from agent.orchestration.active_context import input_preparation,prepare_selected_inputs
    from agent.tools.analysis import sequence_prediction
    from agent.tools.data import cellmarker
    lib=SimpleNamespace(entries=lambda:[dict(kind='cellmarker',id='installed')])
    monkeypatch.setattr(sequence_prediction,'library',lambda:lib)
    monkeypatch.setattr(cellmarker,'CellMarkerEvidence',lambda *a:SimpleNamespace(identity={'sha256':'a'*64},metadata={'species':['Human','Mouse']}))
    tool=build_default_tool_registry().get('annotate_scATAC_cell_types')
    args=dict(grouping_path=StepOutputRef('groups','matrix_path'),activity_method='snapatac2.pp.make_gene_matrix')
    with input_preparation(preparer(None,None,dict(resource_catalog=gene_catalog()))):
        with pytest.raises(ScientificInputRequired,match='fragment/insertion source'):
            prepare_selected_inputs(tool,args,{})
        explicit=prepare_selected_inputs(tool,dict(args,activity_kwargs={'use_x':True}),{})
        assert explicit['gene_anno_path']=='/references/genes.gtf'
        assert explicit['gene_anno_sha256']=='b'*64
        assert explicit['activity_kwargs']=={'use_x':True}
        source=StepOutputRef('fragments','matrix_path')
        default=prepare_selected_inputs(tool,dict(args,activity_source_path=source),{})
        assert default['activity_source_path']==source and 'activity_kwargs' not in default


@pytest.mark.parametrize('selection', ['explicit', 'click', 'species'])
def test_uploaded_original_column_binding_and_deterministic_choice(tmp_path, monkeypatch, selection):
    import anndata as ad
    import pandas as pd
    import numpy as np
    from scipy import sparse
    from agent.tools.analysis import sequence_prediction, annotation_input
    from agent.tools.data import cellmarker
    from agent.application.raw_handoff import AnnotationDeclaration, declaration_dict
    lib=SimpleNamespace(entries=lambda:[dict(kind='cellmarker',id='installed')])
    monkeypatch.setattr(sequence_prediction,'library',lambda:lib)
    monkeypatch.setattr(cellmarker,'CellMarkerEvidence',lambda *a:SimpleNamespace(identity={'sha256':'a'*64},metadata={'species':['Human','Mouse']}))
    # Exercise actual observation reader without a process budget in this planning-only witness.
    monkeypatch.setattr(annotation_input,'observe',annotation_input._observe)
    calls=[]
    class Model:
        model_id='supplied-groups'
        def complete(self,*,prompt,response_schema):
            value=json.loads(prompt);calls.append(value)
            if 'turn_schema_version' in value:
                assert sum('turn_schema_version' in c for c in calls)==1, 'Click must not call interpreter'
                args=[]
                if selection in ('explicit', 'species'):
                    args=[dict(tool='annotate_scATAC_cell_types',argument='group_column',source='source.0',
                        mode='offered',value='my_groups',literal='supplied groups',evidence='supplied groups')]
                return json.dumps(dict(turn_schema_version=1,decision=dict(kind='request_roles',target='annotate_scATAC_cell_types',
                    required_roles=[],roles=[],arguments=args,
                    annotation=declaration_dict(AnnotationDeclaration(source='source.0',grouping_strategy='reuse_only',generate_umap=False,
                        species='Homo sapiens' if selection=='species' else None)))))
            if 'output_selection_schema_version' in value:
                return json.dumps(dict(outputs=[dict(name='annotation',step_id='annotation',output_key='annotated_h5ad_path')]))
            source=lambda target, **kw:dict(target=target,source=kw)
            sources=[source('source',kind='input',input='input_path'),
                     source('activity_method',kind='registered_choice',choice='episcanpy.tl.geneactivity')]
            # Compiler binds the exact user input; no model-supplied column text.
            if selection in ('explicit', 'species') or len([c for c in calls if 'schema_version' in c])>1:
                sources.append(source('group_column',kind='input',input='group_column'))
            return json.dumps(dict(schema_version=4,decision=dict(kind='plan',steps=[dict(step_id='annotation',
                tool='annotate_scATAC_cell_types',sources=sources,control_dependencies=[])])))
    profile=PlanningModelProfile('planning','scripted','supplied-groups')
    registry=build_default_tool_registry();executor=WithholdScience(registry)
    app=InteractiveAgentApplication(tmp_path/'workspace',registry=registry,executor=executor,
        model_profiles=(profile,),default_profile_id='planning',
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted':lambda _:Model()}),
        approved_source_roots=(tmp_path,))
    app.create_session('session')
    path=tmp_path/'peaks.h5ad'
    ad.AnnData(sparse.csr_matrix(np.eye(6)),obs=pd.DataFrame(dict(
        my_groups=pd.Categorical(['alpha']*3+['beta']*3),
        alternate=pd.Categorical(['one','two']*3)),index=[f'cell{i}' for i in range(6)]),
        uns=dict(species='human',assembly='hg38')).write_h5ad(path)
    record=app.resources.register('groups',path,label='Peak counts',attribution='Supplied grouping fixture')
    group=app.resources.compose_group([record.resource_id],attachment_candidates=True)
    from agent.application.input_sets import ScientificInputSet
    inputs=tuple(ScientificInputSet(**row) for row in gene_catalog())
    first=app.submit_turn('session','task','Annotate using supplied groups.',expected_generation=0,registered_input=group,input_sets=inputs)
    if selection=='species':
        assert first.status=='clarification',first.to_dict()
        assert {c['label'] for c in first.clarification_choices}=={'Human','Mouse'}
        assert not executor.plans
        assert app._application.sessions.load('session').interactions[0].admitted['annotation']['species']=='Homo sapiens'
        choice=next(c for c in first.clarification_choices if c['label']=='Human')
        planning_calls = sum('semantic_prompt_version' in c for c in calls)
        first=app.submit_turn('session','species-choice','',expected_generation=0,
            clarification_choice=dict(question_turn_id='task',handle=choice['handle']),input_sets=inputs)
        state=app._application.sessions.load('session')
        from agent.application.preparation_state import resource_bindings
        latest=state.interactions[-1]
        assert resource_bindings(state.interactions,latest.preparation)['annotation']['species']['value']=='Human'
        assert latest.admitted['annotation']['species']=='Homo sapiens'
        assert state.interactions[0].admitted['annotation']['species']=='Homo sapiens'
        # This catalog intentionally has no species-qualified gene reference.
        # The next real prerequisite must remain distinct from species binding.
        assert 'gene annotation resource' in ' '.join(m for r in state.interactions[-1].admitted['preparation_result']['requirements'] for m in r.get('messages', ()))
        assert sum('turn_schema_version' in c for c in calls)==1
        assert sum('semantic_prompt_version' in c for c in calls)==planning_calls
        assert not executor.plans
        return
    if selection=='click':
        assert first.status=='clarification',first.to_dict()
        assert {c['label'] for c in first.clarification_choices}=={'my_groups','alternate'}
        assert not executor.plans
        choice=next(c for c in first.clarification_choices if c['label']=='my_groups')
        first=app.submit_turn('session','choice','',expected_generation=0,
            clarification_choice=dict(question_turn_id='task',handle=choice['handle']),input_sets=inputs)
    assert first.error and first.error.code=='TEST_SCIENCE_WITHHELD',first.to_dict()
    step,=executor.plans[0].steps
    assert step.arguments['group_column']=='my_groups'
    assert step.arguments['input_path']==record.source_path
    assert 'grouping_path' not in step.arguments
    assert sum('turn_schema_version' in c for c in calls)==1


@pytest.fixture
def accepted_native_groupings(monkeypatch):
    """Exercise real lineage traversal with a simulated accepted publication store."""
    from agent.application import annotation_groups, result_reading
    from agent.orchestration import verification_authority
    from agent.orchestration.prior_outputs import accepted_step_digest
    from agent.schemas import PlanStep, PriorOutputRef, StepExecutionResult, StepStatus
    from agent.schemas.prior_output import PriorOutputBinding
    from agent.schemas.verification_authority import AuthorityError
    from agent.tools.analysis import sequence_prediction
    from agent.tools.data import cellmarker

    monkeypatch.setattr(sequence_prediction, 'library', lambda: SimpleNamespace(
        entries=lambda: [dict(kind='cellmarker', id='installed')]))
    monkeypatch.setattr(cellmarker, 'CellMarkerEvidence', lambda *a: SimpleNamespace(
        identity={'sha256': 'a'*64}, metadata={'species': ['Human', 'Mouse']}))
    registry = build_default_tool_registry()
    tool_name = 'run_scanpy'
    run_id = 'accepted-run'
    state = SimpleNamespace(lifecycle_status=SimpleNamespace(value='SUCCEEDED'))
    rows = {}
    references = {}
    for index, (step_id, function, slot, parent, kwargs) in enumerate((
            ('cluster', 'tl.leiden', 'obs.leiden', None, {'resolution': 1.5}),
            ('alternative', 'tl.leiden', 'obs.leiden', None, {'resolution': 0.5}),
            ('umap', 'tl.umap', 'obsm.X_umap', 'cluster', {})), start=1):
        source_path = '/original.h5ad' if parent is None else rows[parent][0].result['matrix_path']
        source_sha = 'a'*64 if parent is None else rows[parent][0].result['matrix_sha256']
        step = StepExecutionResult(step_id, tool_name, StepStatus.SUCCEEDED,
            result=dict(contract_version='native-scanpy-matrix.v1', function_name='scanpy.' + function,
                        matrix_path=f'/{step_id}.h5ad', matrix_sha256=str(index)*64, n_cells=4,
                        n_clusters=2, source_identity={'sha256': source_sha},
                        parameters={'kwargs': kwargs}, result_slots=[slot]),
            resolved_arguments={'input_path': source_path, 'input_sha256': source_sha})
        plan = PlanStep(step_id, tool_name, {'input_path': source_path if parent is None
                        else StepOutputRef(parent, 'matrix_path')}, () if parent is None else (parent,))
        rows[step_id] = (step, plan, state)
        binding = PriorOutputBinding(run_id, step_id, tool_name, 'dataset', 'native_matrix_h5ad.v1',
            accepted_step_digest(step), 'b'*64, 'c'*64)
        references[step_id] = PriorOutputRef(binding, 'matrix_path')

    def accepted_step(store, bound_run, step):
        assert bound_run == run_id
        return rows[step]

    def validate(binding, **_):
        if binding != references[binding.step_id].binding:
            raise AuthorityError('Historical grouping anchor changed')

    monkeypatch.setattr(result_reading, '_accepted_step', accepted_step)
    monkeypatch.setattr(annotation_groups, '_accepted_step', accepted_step)
    monkeypatch.setattr(verification_authority, '_accepted', lambda store, rid, step: (
        None, SimpleNamespace(record={'resources': {'result_metadata': accepted_step(store, rid, step)[0].result}}),
        None, None))
    monkeypatch.setattr(annotation_groups, 'validate_binding', validate)
    monkeypatch.setattr(annotation_groups, 'bind_output', lambda store, reg, rid, step, port: references[step].binding)
    umap = references['umap'].binding
    revision = SimpleNamespace(revision_id='revision', outputs=[umap])
    sessions = SimpleNamespace(_application=SimpleNamespace(run_store=object(), registry=registry),
        _interaction_session_id='session', load=lambda _: SimpleNamespace(revisions=[revision]))
    interaction = SimpleNamespace(base_revision_id=revision.revision_id)
    return sessions, interaction, references


@pytest.mark.parametrize('planned', ['umap', 'cluster', None])
def test_accepted_grouping_offer_binds_the_canonical_producer(accepted_native_groupings, planned):
    from agent.application import annotation_groups
    from agent.application.annotation_preparation import preparer
    from agent.application.raw_handoff import AnnotationDeclaration, declaration_dict

    sessions, interaction, refs = accepted_native_groupings
    offer, = annotation_groups.choices(sessions, interaction)
    admitted = dict(annotation=declaration_dict(AnnotationDeclaration(groups_result=offer['handle'])))
    tool = sessions._application.registry.get('annotate_scATAC_cell_types')
    args = dict(input_path=refs['umap'], activity_path=StepOutputRef('activity', 'matrix_path'))
    if planned is not None:
        args['grouping_path'] = refs[planned]
    prepare = preparer(sessions, interaction, admitted)
    bound = prepare(tool, args, {})
    assert bound['grouping_path'] == refs['cluster']
    assert bound['input_path'] == args['input_path']
    assert bound['activity_path'] == args['activity_path']
    assert prepare(tool, bound, {}) == bound


def test_accepted_grouping_rejects_a_different_clustering_on_the_same_source(accepted_native_groupings):
    from agent.application import annotation_groups
    from agent.application.annotation_preparation import preparer
    from agent.application.raw_handoff import AnnotationDeclaration, declaration_dict
    from agent.application.result_reading import handoff
    from agent.orchestration.scientific_inputs import ScientificDiagnostic

    sessions, interaction, refs = accepted_native_groupings
    app = sessions._application
    offered, = annotation_groups.choices(sessions, interaction)
    upstream = handoff(app.run_store, offered['run_id'], offered['step_id'])
    alternative = handoff(app.run_store, offered['run_id'], 'alternative')
    assert upstream['source'] == alternative['source']
    assert upstream['groups']['field'] == alternative['groups']['field'] == 'leiden'
    admitted = dict(annotation=declaration_dict(AnnotationDeclaration(groups_result=offered['handle'])))
    tool = app.registry.get('annotate_scATAC_cell_types')
    args = dict(input_path=refs['umap'], grouping_path=refs['alternative'],
                activity_path=StepOutputRef('activity', 'matrix_path'))
    with pytest.raises(ScientificDiagnostic, match='clustering conflicts') as caught:
        preparer(sessions, interaction, admitted)(tool, args, {})
    assert caught.value.code == 'ANNOTATION_GROUP_SOURCE_CONFLICT'


def test_accepted_grouping_requires_the_planned_anchor(accepted_native_groupings):
    from dataclasses import replace
    from agent.application import annotation_groups
    from agent.application.annotation_preparation import preparer
    from agent.application.raw_handoff import AnnotationDeclaration, declaration_dict
    from agent.schemas.verification_authority import AuthorityError

    sessions, interaction, refs = accepted_native_groupings
    offer, = annotation_groups.choices(sessions, interaction)
    admitted = dict(annotation=declaration_dict(AnnotationDeclaration(groups_result=offer['handle'])))
    tool = sessions._application.registry.get('annotate_scATAC_cell_types')
    stale = replace(refs['umap'], binding=replace(refs['umap'].binding, accepted_step_sha256='0'*64))
    with pytest.raises(AuthorityError, match='anchor changed'):
        preparer(sessions, interaction, admitted)(tool, dict(input_path=refs['umap'], grouping_path=stale), {})


def test_reuse_only_follows_selected_dependency_semantics(accepted_native_groupings):
    from agent.application.annotation_preparation import preparer
    from agent.orchestration.semantic_compiler import ScientificInputRequired
    from agent.application.raw_handoff import AnnotationDeclaration, declaration_dict

    sessions, interaction, refs = accepted_native_groupings
    prepare = preparer(sessions, interaction, dict(
        annotation=declaration_dict(AnnotationDeclaration(grouping_strategy='reuse_only'))))
    tool = sessions._application.registry.get('annotate_scATAC_cell_types')
    args = dict(grouping_path=StepOutputRef('identity', 'matrix_path'), activity_path=refs['umap'])
    selected = {'identity': ('run_scanpy', dict(function_name='tl.umap', input_path=refs['umap']))}
    assert prepare(tool, args, selected)['grouping_path'] == refs['cluster']
    selected['identity'][1]['input_path'] = StepOutputRef('new_groups', 'matrix_path')
    selected['new_groups'] = ('run_snapatac2', dict(function_name='tl.leiden', input_path=refs['umap']))
    with pytest.raises(ScientificInputRequired, match='reuse existing groups'):
        prepare(tool, args, selected)
