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
                                               ('generate','ambiguous_native_dataset')])
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
                outputs=[dict(name='annotation',step_id='annotation',output_key='annotated_h5ad_path')]
                if choice=='generate': outputs.append(dict(name='umap',step_id='umap',output_key='matrix_path'))
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
            elif invalid_port is not None and 'repair' not in value:
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
            clarification_choice=dict(question_turn_id='request',handle='annotation-umap:'+choice))
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
    from agent.application.annotation_preparation import preparer, AnnotationPrerequisite
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
        with pytest.raises(AnnotationPrerequisite) as failure:
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
        assert pending.prerequisite['field']=='annotation'
        assert 'CellMarker' in pending.admitted['choices'][0]
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
        assert continued.admitted['continuation']['origin_turn_id']=='annotation'
        if failure_then_answer:
            assert outcome.error.code=='UNSUPPORTED_REQUEST'
            assert continued.status=='failed'
            assert continued.prerequisite['origin_turn_id']=='annotation'
            assert not app._application.sessions.load('session').turns and not executor.plans
            from dataclasses import replace
            from agent.schemas.orchestration import _serialize
            from agent.application.session_state import SessionError
            changed = _serialize(continued.prerequisite)
            changed['binding_turn_id'] = 'continue'
            with pytest.raises(SessionError, match='captured scientific prerequisite'):
                replace(continued, prerequisite=changed)
            with pytest.raises(SessionError, match='requires a clarification'):
                replace(continued, admitted=dict(_serialize(continued.admitted), native_plan={}))
            reject.clear()
            outcome=app.submit_turn('session','answer','Human, hg38. Continue the original task.',expected_generation=0)
            continued=app._application.sessions.load('session').interactions[-1]
            assert continued.admitted['inputs']['input_path']==str(path)
            assert continued.admitted['continuation']['origin_turn_id']=='annotation'
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
    from agent.application.annotation_preparation import preparer,AnnotationPrerequisite
    from agent.application.raw_handoff import AnnotationDeclaration,declaration_dict
    from agent.tools.analysis import sequence_prediction
    from agent.tools.data import cellmarker
    lib=SimpleNamespace(entries=lambda:[dict(kind='cellmarker',id='installed')])
    monkeypatch.setattr(sequence_prediction,'library',lambda:lib)
    monkeypatch.setattr(cellmarker,'CellMarkerEvidence',lambda *a:SimpleNamespace(identity={'sha256':'a'*64},metadata={'species':['Human','Mouse']}))
    tool=build_default_tool_registry().get('annotate_scATAC_cell_types')
    prepare=preparer(None,None,{})
    with pytest.raises(AnnotationPrerequisite,match='file locator alone'):
        prepare(tool,dict(grouping_path=StepOutputRef('groups','matrix_path'),activity_path='/loose/genes.h5ad'),None)
    prepare=preparer(None,None,dict(annotation=declaration_dict(AnnotationDeclaration(grouping_strategy='reuse_only'))))
    with pytest.raises(AnnotationPrerequisite,match='reuse existing groups'):
        prepare(tool,dict(grouping_path=StepOutputRef('groups','matrix_path')),None)


def test_selected_gene_reference_binding_preserves_native_activity_source_semantics(monkeypatch):
    from agent.application.annotation_preparation import preparer,AnnotationPrerequisite
    from agent.orchestration.active_context import input_preparation,prepare_selected_inputs
    from agent.tools.analysis import sequence_prediction
    from agent.tools.data import cellmarker
    lib=SimpleNamespace(entries=lambda:[dict(kind='cellmarker',id='installed')])
    monkeypatch.setattr(sequence_prediction,'library',lambda:lib)
    monkeypatch.setattr(cellmarker,'CellMarkerEvidence',lambda *a:SimpleNamespace(identity={'sha256':'a'*64},metadata={'species':['Human','Mouse']}))
    tool=build_default_tool_registry().get('annotate_scATAC_cell_types')
    args=dict(grouping_path=StepOutputRef('groups','matrix_path'),activity_method='snapatac2.pp.make_gene_matrix')
    with input_preparation(preparer(None,None,dict(resource_catalog=gene_catalog()))):
        with pytest.raises(AnnotationPrerequisite,match='fragment/insertion source'):
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
        first=app.submit_turn('session','species-choice','',expected_generation=0,
            clarification_choice=dict(question_turn_id='task',handle=choice['handle']),input_sets=inputs)
        state=app._application.sessions.load('session')
        assert state.interactions[-1].admitted['annotation']['species']=='Human'
        assert state.interactions[0].admitted['annotation']['species']=='Homo sapiens'
        # This catalog intentionally has no species-qualified gene reference.
        # The next real prerequisite must remain distinct from species binding.
        assert 'gene annotation resource' in state.interactions[-1].admitted['choices'][0]
        assert sum('turn_schema_version' in c for c in calls)==1
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


def test_reuse_only_follows_selected_dependency_semantics(monkeypatch):
    from agent.application.annotation_preparation import preparer, AnnotationPrerequisite
    from agent.application.raw_handoff import AnnotationDeclaration, declaration_dict
    from agent.schemas import PriorOutputRef
    from agent.schemas.prior_output import PriorOutputBinding
    from agent.tools.analysis import sequence_prediction
    from agent.tools.data import cellmarker
    monkeypatch.setattr(sequence_prediction,'library',lambda:SimpleNamespace(entries=lambda:[dict(kind='cellmarker',id='installed')]))
    monkeypatch.setattr(cellmarker,'CellMarkerEvidence',lambda *a:SimpleNamespace(identity={'sha256':'a'*64},metadata={'species':['Human','Mouse']}))
    binding=PriorOutputBinding('run','groups','run_snapatac2','dataset','native_matrix_h5ad.v1','a'*64,'b'*64,'c'*64)
    ref=PriorOutputRef(binding,'matrix_path')
    prepare=preparer(None,None,dict(annotation=declaration_dict(AnnotationDeclaration(grouping_strategy='reuse_only'))))
    tool=build_default_tool_registry().get('annotate_scATAC_cell_types')
    args=dict(grouping_path=StepOutputRef('identity','matrix_path'),activity_path=ref)
    selected={'identity':('run_scanpy',dict(function_name='pp.log1p',input_path=ref))}
    assert prepare(tool,args,selected)['grouping_path']==args['grouping_path']
    selected['identity'][1]['input_path']=StepOutputRef('new_groups','matrix_path')
    selected['new_groups']=('run_snapatac2',dict(function_name='tl.leiden',input_path=ref))
    with pytest.raises(AnnotationPrerequisite,match='reuse existing groups'):
        prepare(tool,args,selected)
