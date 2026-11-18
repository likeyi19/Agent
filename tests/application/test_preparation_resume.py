"""Explicit resume uses fresh Session authority and never executes the task."""
from pathlib import Path
import json
import pytest

from agent.application import ResearchAgentApplication, OutputSelection
from agent.application.local_resources import LocalResourceAdmission
from agent.schemas import AgentPlan, PlanStep, AgentRequest
from test_structured_interaction import setup


@pytest.mark.parametrize('advance,change_source', [(False,False),(True,False),(True,True)])
def test_explicit_resume_reissues_question_without_replanning_or_reservation(tmp_path, advance, change_source, direct=False):
    app, executor, model, inputs = setup(tmp_path, missing=(dict(
        tool='run_scanpy', native_function='pp.scale', argument='kwargs', member='max_value'),))
    owner=LocalResourceAdmission(app._application._workspace,approved_source_roots=(tmp_path,))
    source=owner.register('counts',inputs['input_path'],input_type='h5ad',session_id='session',label='Counts',attribution='Fixture')
    binding=owner.compose_h5ad(source.resource_id)
    first=app.submit_turn('session','task','缩放此数据',expected_generation=0,
        **(dict(execution_inputs=inputs) if direct else dict(registered_input=binding)))
    assert first.status=='clarification', first.to_dict()
    before=app._application.sessions.load('session')
    task=before.interactions[-1].preparation['task_id']
    candidate=before.interactions[-1].preparation['candidate_id']
    if advance:
        class Inspection:
            def plan(self, request, registry):
                return AgentPlan(request.request_id+':plan',request.request_id,'inspection-fixture',
                    (PlanStep('inspect','inspect_scATAC',dict(path=inputs['input_path'])),))
        other=ResearchAgentApplication(app._application.workspace_root,planner=Inspection())
        state=other.sessions.run('session','unrelated',AgentRequest('unrelated','Inspect the fixture.',{}),
            (OutputSelection('source','inspect','input_path'),),expected_generation=0)
        assert state.generation==1 and state.active_revision_id is not None
    if not advance:
        other=app.submit_turn('session','another-task','Scale another dataset.',expected_generation=0)
        assert other.status=='clarification',other.to_dict()
        waiting=app._application.sessions.load('session').interactions[-1]
        assert waiting.preparation['task_id'] != task and waiting.preparation['candidate_id'] is not None
        assert any(r['port']=='dataset' and r['argument']=='input_path' for r in waiting.admitted['preparation_result']['requirements'])
    state=app._application.sessions.load('session')
    accepted=state.revisions,state.turns,state.navigation
    runs={p:p.read_bytes() for p in (Path(app._application.workspace_root)/'run_state').glob('*.json')}
    if change_source:
        Path(inputs['input_path']).write_bytes(b'changed source bytes')
    model.interpret=False
    calls=len(model.calls)
    result=app.submit_turn('session','resume','',expected_generation=state.generation,resume_task_id=task)
    saved=app._application.sessions.load('session')
    assert 'preparation' not in saved.interactions[-1].snapshot
    assert 'prerequisite' not in saved.interactions[-1].snapshot
    assert saved.interactions[-1].preparation['task_id']==task
    assert saved.interactions[-1].preparation['candidate_id']==candidate
    assert len(model.calls)==calls and not executor.plans
    assert (saved.revisions,saved.turns,saved.navigation)==accepted
    assert all(p.read_bytes()==b for p,b in runs.items())
    assert saved.interactions[:len(before.interactions)]==before.interactions
    if change_source:
        assert result.status=='failed' and result.error.code=='PREPARATION_SOURCE_CHANGED',result.to_dict()
        assert saved.interactions[-1].admitted['preparation_result']['status']=='INVALID'
    else:
        assert result.status=='clarification', result.to_dict()
        assert result.clarification_choices[0]['handle']!=first.clarification_choices[0]['handle']
        assert result.response.text == ''
        assert saved.interactions[-1].admitted['preparation_result']['status']=='NEEDS_INPUT'
        rejected=app.submit_turn('session','stale','',expected_generation=state.generation,
            clarification_choice=dict(question_turn_id='task',handle=first.clarification_choices[0]['handle']),
            parameter_overrides=[dict(tool='run_scanpy',native_function='pp.scale',argument='kwargs',
                step_id='scale',value={'max_value':5})])
        assert rejected.status=='failed' and rejected.error.code=='INTERACTIVE_TURN_CONFLICT'
        assert app._application.sessions.load('session').turns==saved.turns and not executor.plans
    assert ResearchAgentApplication(app._application.workspace_root).sessions.load('session')==app._application.sessions.load('session')
    assert app.submit_turn('session','resume','',expected_generation=state.generation,resume_task_id=task)==result
    assert len(model.calls)==calls and not executor.plans


@pytest.mark.parametrize('change_source', [False, True])
def test_direct_source_task_pin_revalidates_exact_bytes(tmp_path, change_source):
    test_explicit_resume_reissues_question_without_replanning_or_reservation(
        tmp_path, advance=True, change_source=change_source, direct=True)


@pytest.mark.parametrize('changed', ['selected', 'unused'])
def test_candidate_less_resume_checks_declared_file_but_not_unused_offers(tmp_path, monkeypatch, changed):
    from agent.tools.analysis import annotation_input
    monkeypatch.setattr(annotation_input, "observe", annotation_input._observe)
    import anndata as ad
    import numpy as np
    from agent.application import InteractiveAgentApplication
    from agent.application.raw_handoff import AnnotationDeclaration, declaration_dict
    from agent.orchestration import PlanningModelProfile
    from agent.providers import PlanningModelFactoryRegistry
    calls=[]
    class Model:
        model_id='declared-source-resume'
        def complete(self, *, prompt, response_schema):
            value=json.loads(prompt);calls.append(value)
            if value.get('focus')=='outcome' or 'grounding_review_version' in value:
                from test_response_contracts import OutcomeModel
                return OutcomeModel('Please supply the scientific declarations.').complete(prompt=prompt,response_schema=response_schema)
            assert 'turn_schema_version' in value
            return json.dumps(dict(turn_schema_version=1,decision=dict(kind='request_roles',
                target='annotate_scATAC_cell_types',required_roles=[],roles=[],
                annotation=declaration_dict(AnnotationDeclaration(source='source.0')))))
    app=InteractiveAgentApplication(tmp_path/'workspace',approved_source_roots=(tmp_path,),
        model_profiles=(PlanningModelProfile('offline','scripted',Model.model_id),),default_profile_id='offline',
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted':lambda _:Model()}))
    app.create_session('session')
    records={}
    for name in ('selected','unused'):
        path=tmp_path/(name+'.h5ad');ad.AnnData(np.eye(3)).write_h5ad(path)
        records[name]=app.resources.register(name,path,input_type='h5ad',session_id='session',label=name,attribution='Fixture')
    group=app.resources.compose_group([r.resource_id for r in records.values()],attachment_candidates=True)
    first=app.submit_turn('session','task','Annotate the selected source.',expected_generation=0,registered_input=group)
    assert first.status=='clarification',first.to_dict()
    before=app._application.sessions.load('session')
    assert before.interactions[-1].preparation['candidate_origin'] is None
    original_bytes=Path(records[changed].source_path).read_bytes()
    Path(records[changed].source_path).write_bytes(b'changed bytes')
    resumed=app.submit_turn('session','resume','',expected_generation=0,resume_task_id=first.preparation_task_id)
    if changed=='selected':
        assert resumed.status=='failed' and resumed.error.code=='PREPARATION_SOURCE_CHANGED',resumed.to_dict()
        Path(records[changed].source_path).write_bytes(original_bytes)
        restored=app.submit_turn('session','restored','',expected_generation=0,resume_task_id=first.preparation_task_id)
        assert restored.status=='clarification',restored.to_dict()
    else:
        assert resumed.status=='clarification',resumed.to_dict()
    state=app._application.sessions.load('session')
    assert state.interactions[:len(before.interactions)]==before.interactions
    assert not state.turns and not state.revisions and sum('turn_schema_version' in c for c in calls)==1 and not any('semantic_prompt_version' in c for c in calls)
    assert ResearchAgentApplication(app._application.workspace_root).sessions.load('session')==state
