"""Bounded production research preparation using fake providers/withheld science."""
import json
from types import SimpleNamespace

import pytest

from agent.application import InteractiveAgentApplication
from agent.orchestration import PlanningModelProfile, build_default_tool_registry
from agent.providers import PlanningModelFactoryRegistry
from test_human_epizoo_mapping import WithholdScience


@pytest.mark.parametrize('generation_race', [False, True])
def test_downstream_required_parameters_resume_same_candidate(tmp_path, generation_race):
    calls=[]
    class Model:
        model_id='spp-test'
        def complete(self, *, prompt, response_schema):
            value=json.loads(prompt); calls.append(value)
            if 'turn_schema_version' in value:
                return json.dumps(dict(turn_schema_version=1,decision=dict(kind='execute_plan',target='call_scATAC_peaks')))
            if 'semantic_prompt_version' in value:
                return json.dumps(dict(schema_version=4,decision=dict(kind='plan',steps=[
                    dict(step_id='inspect',tool='inspect_scATAC',sources=[dict(target='dataset',source=dict(kind='input',input='input_path'))],control_dependencies=[]),
                    dict(step_id='call_peaks',tool='call_scATAC_peaks',sources=[dict(target='dataset',source=dict(kind='input',input='input_path'))],control_dependencies=[])])))
            if 'output_selection_schema_version' in value:
                return json.dumps(dict(outputs=[dict(name='peaks',step_id='call_peaks',output_key='peaks_path')]))
            if value.get('focus')=='outcome' or 'grounding_review_version' in value:
                from test_response_contracts import OutcomeModel
                return OutcomeModel('科学执行尚未开始。请提供所需参数。').complete(prompt=prompt,response_schema=response_schema)
            raise AssertionError('Unexpected model phase: '+str(list(value)))
        def close(self): pass
    registry=build_default_tool_registry(); executor=WithholdScience(registry)
    app=InteractiveAgentApplication(tmp_path/'workspace',registry=registry,executor=executor,
        model_profiles=(PlanningModelProfile('offline','scripted',Model.model_id),),default_profile_id='offline',
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted':lambda _:Model()}))
    app.create_session('session')
    observed_race = []
    if generation_race:
        # Inject a changed generation at the atomic reservation mutation itself.
        # No invalid Session is written; require the specific canonical CAS guard.
        from agent.application.session_state import SessionConflictError
        store = app._application.sessions._store
        update = store._update
        def inject(session_id, change):
            if change.__name__ != 'link':
                return update(session_id, change)
            def concurrent(state):
                stale = SimpleNamespace(**{**state.__dict__, 'generation': state.generation + 1})
                with pytest.raises(SessionConflictError, match='Scientific preparation changed before reservation'):
                    change(stale)
                observed_race.append(True)
                raise SessionConflictError('Scientific preparation changed before reservation.')
            return update(session_id, concurrent)
        store._update = inject
    # Canonical source capture now pins actual bytes before later continuation.
    # Keep this compiler/CAS witness independent of native scientific execution.
    import anndata as ad
    import numpy as np
    path=tmp_path/'native.h5ad'
    data=ad.AnnData(np.ones((3, 4)))
    # A storage-schema fixture for parameter/CAS continuation, never exported or
    # treated as a qualified peak-calling population by native science.
    from scipy.sparse import csr_matrix
    data.obsm['fragment_paired']=csr_matrix(np.asarray([[2],[3],[4]],dtype=np.uint32))
    data.write_h5ad(path)
    first=app.submit_turn('session','ask','请进行峰调用',expected_generation=0,
                          execution_inputs={'input_path':str(path)})
    assert first.status=='clarification',first.to_dict()
    assert len(first.clarification_choices)==2
    assert '科学执行尚未开始' in first.response.text
    state=app._application.sessions.load('session')
    candidate_id=state.interactions[-1].preparation['candidate_id']
    assert not state.turns and not executor.plans
    for index,(argument,value) in enumerate([('effective_genome_size',100),('genome_size_basis','explicit fixture basis')]):
        choice=next(c for c in first.clarification_choices if c['argument']==argument)
        first=app.submit_turn('session','answer'+str(index),'',expected_generation=0,
            clarification_choice=dict(question_turn_id=first.turn_id,handle=choice['handle']),
            parameter_overrides=[dict(tool='call_scATAC_peaks',step_id='call_peaks',argument=argument,value=value)])
        if index==0:
            assert first.status=='clarification', first.to_dict()
            assert len(first.clarification_choices)==1
            assert not executor.plans
    if generation_race:
        assert first.status == 'failed', first.to_dict()
        state = app._application.sessions.load('session')
        assert not executor.plans and not state.turns
        assert observed_race == [True] and state.generation == 0
        return
    assert len(executor.plans)==1,first.to_dict()
    plan=executor.plans[0]
    assert plan.steps[1].arguments['effective_genome_size']==100
    assert plan.steps[1].arguments['genome_size_basis']=='explicit fixture basis'
    state=app._application.sessions.load('session')
    assert {i.preparation['candidate_id'] for i in state.interactions}=={candidate_id}
    assert sum('candidate' in i.preparation for i in state.interactions)==1
    assert sum('turn_schema_version' in c for c in calls)==1
    assert sum('semantic_prompt_version' in c for c in calls)==1


def test_http_exact_parameter_uses_canonical_question_after_reload(tmp_path):
    from fastapi.testclient import TestClient
    from agent.web.app import create_app
    from helpers import wait_turn
    from test_structured_interaction import setup, setting
    app, executor, model, inputs = setup(tmp_path, missing=(dict(
        tool='run_scanpy', native_function='pp.scale', argument='kwargs', member='max_value'),))
    first = app.submit_turn('session','task','Scale these counts.',expected_generation=0,execution_inputs=inputs)
    assert first.status == 'clarification'
    before = app._application.sessions.load('session')
    candidate_id = before.interactions[0].preparation['candidate_id']
    model.interpret = False
    with TestClient(create_app(app)) as client:
        saved = client.get('/api/v1/sessions/session/turns/task').json()
        choice, = saved['clarification_choices']
        assert choice['handle'].startswith('spp.') and choice['candidate_id'] == candidate_id
        body = dict(turn_id='answer',utterance='',expected_generation=0,
            clarification_choice=dict(question_turn_id='task',handle=choice['handle']),
            parameter_overrides=[dict(setting('kwargs',{'max_value':4}),step_id=choice['step_id'])])
        response = client.post('/api/v1/sessions/session/turns',json=body)
        assert response.status_code == 202, response.text
        final = wait_turn(client,'answer')
        assert final['error']['code'] == 'TEST_SCIENCE_WITHHELD', final
        again = client.get('/api/v1/sessions/session/turns/answer').json()
        assert again == final
    assert len(executor.plans) == 1
    assert executor.plans[0].steps[0].arguments['kwargs']['max_value'] == 4
    assert executor.plans[0].steps[0].arguments['input_path'] == inputs['input_path']
    state = app._application.sessions.load('session')
    assert state.interactions[0] == before.interactions[0]
    assert {i.preparation['candidate_id'] for i in state.interactions} == {candidate_id}
    assert sum('semantic_prompt_version' in call for call in model.calls) == 1
    assert sum('turn_schema_version' in call for call in model.calls) == 1
