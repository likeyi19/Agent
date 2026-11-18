"""Invalid model output selections remain classified, durable HTTP failures.

Only model decisions use the existing scripted helper. The real application,
Planner, compiler, output validator and HTTP worker admit no scientific work.
"""
import json

import pytest
from fastapi.testclient import TestClient

from agent.web.app import create_app
from helpers import ScriptedModel, harness, submission, wait_turn


@pytest.mark.parametrize('invalid', ['duplicate_name', 'unoffered_key'])
@pytest.mark.parametrize('chinese', [False, True])
def test_invalid_output_selection_has_safe_code_and_no_replay(tmp_path, monkeypatch, invalid, chinese):
    original = ScriptedModel.complete

    def complete(self, *, prompt, response_schema):
        value = json.loads(prompt)
        if chinese and 'turn_schema_version' in value:
            self.calls.append(value)
            return json.dumps(dict(turn_schema_version=1,decision=dict(kind='execute_plan',target='inspect_scATAC')))
        if 'output_selection_schema_version' not in value:
            return original(self, prompt=prompt, response_schema=response_schema)
        self.calls.append(value)
        outputs = [dict(name='inspection', step_id='inspect', output_key='n_cells')]
        if invalid == 'duplicate_name':
            outputs.append(dict(name='inspection', step_id='inspect', output_key='n_features'))
        else:
            outputs[0]['output_key'] = 'unoffered_private_output'
        return json.dumps(dict(outputs=outputs))

    monkeypatch.setattr(ScriptedModel, 'complete', complete)
    backend = harness(tmp_path)
    payload = submission()
    if chinese:
        payload['utterance']='检查这个数据集。'
    with TestClient(create_app(backend.service, input_sets=backend.input_sets)) as client:
        assert client.post('/api/v1/sessions', json={'session_id': 'session'}).status_code == 201
        assert client.post('/api/v1/sessions/session/turns', json=payload).status_code == 202
        view = wait_turn(client)
        assert view['status'] == 'failed' and view['revision_id'] is None
        assert view['error']['code'] == 'PLANNER_OUTPUT_INVALID'
        assert view['response']['error']['code'] == 'PLANNER_OUTPUT_INVALID'
        assert view['response']['text']=='The operation failed validation or execution.'
        prompts=[c for m in backend.models for c in m.calls if c.get('focus')=='outcome']
        assert len(prompts)==1 and prompts[0]['question']==payload['utterance']
        assert prompts[0]['evidence']['outcome']['preparation']['status']=='READY'
        assert prompts[0]['evidence']['outcome']['science_started'] is False
        assert prompts[0]['evidence']['outcome']['error']['code']=='PLANNER_OUTPUT_INVALID'
        assert 'unoffered_private_output' not in json.dumps(view)
        assert not backend.science_calls
        state = backend.service._application.sessions.load('session')
        assert not state.turns and not state.revisions and state.generation == 0
        run = backend.service._application.run_store.load(view['run_id'])
        assert run.plan is None and not run.steps
        assert run.errors[-1].code == 'PLANNER_OUTPUT_INVALID'
        assert run.errors[-1].exception_type == 'PlannerError'
        assert len(backend.models) == 1 and len(backend.models[0].calls) == 4
        before = state.to_dict()
        calls = [(model, len(model.calls)) for model in backend.models]
        assert client.get('/api/v1/sessions/session/turns/first').json() == view
        assert client.get('/api/v1/sessions/session/turns/first/status').json() == view
        reopened = client.get('/api/v1/sessions/session').json()
        assert reopened['turns'][0]['response'] == view['response']
        assert client.post('/api/v1/sessions/session/turns', json=payload).status_code == 202
        assert wait_turn(client) == view
        assert client.get('/api/v1/sessions/session').json() == reopened
        assert calls == [(model, len(model.calls)) for model in backend.models]
        assert backend.service._application.sessions.load('session').to_dict() == before
        assert not backend.science_calls
