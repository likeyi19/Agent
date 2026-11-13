"""Production planning and exact resource binding; no native execution in this witness."""
import json
from types import SimpleNamespace
import pytest
from agent.application import InteractiveAgentApplication
from agent.orchestration import PlanningModelProfile, build_default_tool_registry
from agent.providers import PlanningModelFactoryRegistry
from agent.schemas import StepOutputRef
from test_human_epizoo_mapping import WithholdScience


@pytest.mark.parametrize('evidence_available', [True, False])
def test_ordinary_annotation_request_plans_registered_prerequisites(tmp_path, monkeypatch, evidence_available):
    from agent.tools.analysis import sequence_prediction
    from agent.tools.data import cellmarker
    # Resource binding is scripted, with exact captured identity, not biological evidence.
    lib=SimpleNamespace(entries=lambda:[dict(kind='cellmarker',id='installed')] if evidence_available else [])
    monkeypatch.setattr(sequence_prediction,'library',lambda:lib)
    monkeypatch.setattr(cellmarker,'CellMarkerEvidence',lambda *a:SimpleNamespace(identity={'sha256':'a'*64}))
    calls=[]
    continuing=[]
    class Model:
        model_id='planning-witness'
        def complete(self,*,prompt,response_schema):
            value=json.loads(prompt);calls.append(value)
            if 'turn_schema_version' in value:
                if continuing:
                    from agent.application.raw_handoff import AnnotationDeclaration,declaration_dict
                    return json.dumps(dict(turn_schema_version=1,decision=dict(kind='answer_roles',pending='@roles',roles=[],
                        annotation=declaration_dict(AnnotationDeclaration()))))
                return json.dumps(dict(turn_schema_version=1,decision=dict(kind='execute_plan',target='annotate_scATAC_cell_types')))
            if 'output_selection_schema_version' in value:
                return json.dumps(dict(outputs=[dict(name='annotation',step_id='annotation',output_key='annotated_h5ad_path')]))
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
        outcome=app.submit_turn('session','continue','Use the now-installed CellMarker resource and continue.',expected_generation=0)
        continued=app._application.sessions.load('session').interactions[-1]
        assert continued.admitted['inputs']['input_path']==str(path)
        assert continued.admitted['continuation']['origin_turn_id']=='annotation'
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
    monkeypatch.setattr(cellmarker,'CellMarkerEvidence',lambda *a:SimpleNamespace(identity={'sha256':'a'*64}))
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
    lib=SimpleNamespace(entries=lambda:[dict(kind='cellmarker',id='installed'),
        dict(kind='gene_annotation',id='genes',resource={})],
        check=lambda _:dict(file=dict(path='/references/genes.gtf',sha256='b'*64)))
    monkeypatch.setattr(sequence_prediction,'library',lambda:lib)
    monkeypatch.setattr(cellmarker,'CellMarkerEvidence',lambda *a:SimpleNamespace(identity={'sha256':'a'*64}))
    tool=build_default_tool_registry().get('annotate_scATAC_cell_types')
    args=dict(grouping_path=StepOutputRef('groups','matrix_path'),activity_method='snapatac2.pp.make_gene_matrix')
    with input_preparation(preparer(None,None,{})):
        with pytest.raises(AnnotationPrerequisite,match='fragment/insertion source'):
            prepare_selected_inputs(tool,args,{})
        explicit=prepare_selected_inputs(tool,dict(args,activity_kwargs={'use_x':True}),{})
        assert explicit['gene_anno_path']=='/references/genes.gtf'
        assert explicit['gene_anno_sha256']=='b'*64
        assert explicit['activity_kwargs']=={'use_x':True}
        source=StepOutputRef('fragments','matrix_path')
        default=prepare_selected_inputs(tool,dict(args,activity_source_path=source),{})
        assert default['activity_source_path']==source and 'activity_kwargs' not in default


@pytest.mark.parametrize('selection', ['explicit', 'click'])
def test_uploaded_original_column_binding_and_deterministic_choice(tmp_path, monkeypatch, selection):
    import anndata as ad
    import pandas as pd
    import numpy as np
    from scipy import sparse
    from agent.tools.analysis import sequence_prediction, annotation_input
    from agent.tools.data import cellmarker
    from agent.application.raw_handoff import AnnotationDeclaration, declaration_dict
    lib=SimpleNamespace(entries=lambda:[dict(kind='cellmarker',id='installed'),dict(kind='gene_annotation',id='genes',resource={})],
        check=lambda _:dict(file=dict(path='/reference/genes.gtf',sha256='b'*64)))
    monkeypatch.setattr(sequence_prediction,'library',lambda:lib)
    monkeypatch.setattr(cellmarker,'CellMarkerEvidence',lambda *a:SimpleNamespace(identity={'sha256':'a'*64}))
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
                if selection=='explicit':
                    args=[dict(tool='annotate_scATAC_cell_types',argument='group_column',source='source.0',
                        mode='offered',value='my_groups',literal='supplied groups',evidence='supplied groups')]
                return json.dumps(dict(turn_schema_version=1,decision=dict(kind='request_roles',target='annotate_scATAC_cell_types',
                    required_roles=[],roles=[],arguments=args,
                    annotation=declaration_dict(AnnotationDeclaration(source='source.0',grouping_strategy='reuse_only')))))
            if 'output_selection_schema_version' in value:
                return json.dumps(dict(outputs=[dict(name='annotation',step_id='annotation',output_key='annotated_h5ad_path')]))
            source=lambda target, **kw:dict(target=target,source=kw)
            sources=[source('source',kind='input',input='input_path'),
                     source('activity_method',kind='registered_choice',choice='episcanpy.tl.geneactivity')]
            # Compiler binds the exact user input; no model-supplied column text.
            if selection=='explicit' or len([c for c in calls if 'schema_version' in c])>1:
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
        alternate=pd.Categorical(['one','two']*3)),index=[f'cell{i}' for i in range(6)])).write_h5ad(path)
    record=app.resources.register('groups',path,label='Peak counts',attribution='Supplied grouping fixture')
    group=app.resources.compose_group([record.resource_id],attachment_candidates=True)
    first=app.submit_turn('session','task','Annotate using supplied groups.',expected_generation=0,registered_input=group)
    if selection=='click':
        assert first.status=='clarification',first.to_dict()
        assert {c['label'] for c in first.clarification_choices}=={'my_groups','alternate'}
        assert not executor.plans
        choice=next(c for c in first.clarification_choices if c['label']=='my_groups')
        first=app.submit_turn('session','choice','',expected_generation=0,
            clarification_choice=dict(question_turn_id='task',handle=choice['handle']))
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
    monkeypatch.setattr(cellmarker,'CellMarkerEvidence',lambda *a:SimpleNamespace(identity={'sha256':'a'*64}))
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
