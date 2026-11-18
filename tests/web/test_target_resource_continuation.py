"""Existing InputSet continuation binds a target model without replacing H5AD."""
import json

from fastapi.testclient import TestClient
import pytest

import helpers
from helpers import submission, wait_turn
from test_h5ad_composition_api import web_composition
from test_upload_api import attach
from agent.web.config import ScientificInputSet
from agent.application.resource_continuation import catalog, offers


def target_choice():
    return ScientificInputSet('target', 'Reviewed target model', dict(
        species=dict(scientific_name='Danio rerio', taxonomy_id=7955), assembly='danRer11',
        matrix_semantics='binary_accessibility', target_model_manifest_path='/operator/target/manifest.json',
        target_model_manifest_sha256='a' * 64), h5ad_companion=True)


@pytest.mark.parametrize('legacy_request',[False,True])
@pytest.mark.parametrize('structured_context',[False,True])
def test_inspection_keeps_explicit_context_without_preplan_resource_intake(tmp_path,monkeypatch,legacy_request,structured_context):
    original=helpers.ScriptedModel
    class Model(original):
        def complete(self,*,prompt,response_schema):
            value=json.loads(prompt)
            if 'turn_schema_version' in value:
                self.calls.append(value)
                return json.dumps(dict(turn_schema_version=1,decision=dict(kind='request_resources' if legacy_request else 'execute_plan',
                    target='inspect_scATAC',arguments=[])))
            return super().complete(prompt=prompt,response_schema=response_schema)
    monkeypatch.setattr(helpers,'ScriptedModel',Model)
    choice=target_choice();backend,uploads,app=web_composition(tmp_path,companions=(choice,))
    with TestClient(app) as client:
        source=attach(client,backend.source.read_bytes()).json()
        client.post('/api/v1/sessions',json=dict(session_id='session'))
        body=submission(resource_id=source['resource_id'],input_set_id='target' if structured_context else None,
            utterance='Inspect the supplied file.')
        assert client.post('/api/v1/sessions/session/turns',json=body).status_code==202
        result=wait_turn(client)
        state=backend.service._application.sessions.load('session')
        if legacy_request:
            assert result['error']['code']=='INTERACTIVE_INTENT_BINDING_INVALID',result
            assert not state.turns and not state.revisions and not backend.science_calls
        else:
            assert result['status']=='succeeded',result
            current=state.interactions[0]
            assert current.preparation['candidate_id'] is not None
            assert current.admitted['inputs']['input_path']==uploads.resources.load(source['resource_id']).source_path
            if structured_context:
                assert current.submission['registered_input']['resource_choice']==choice.resource_choice()
                assert current.submission['execution_inputs']['target_model_manifest_path']==choice.inputs()['target_model_manifest_path']
            assert backend.science_calls==['inspect_scATAC']
            assert client.post('/api/v1/sessions/session/turns',json=body).status_code==202
            assert backend.science_calls==['inspect_scATAC']
        for model in backend.models:
            assert all('/operator/target' not in json.dumps(prompt) for prompt in model.calls)


def test_target_catalog_does_not_change_legacy_resource_selection():
    target = target_choice()
    legacy = ScientificInputSet('human', 'Human defaults', {'species': 'human'}, h5ad_companion=True)
    assert [row['input_set_id'] for row in catalog((legacy, target))] == ['target']
    assert offers(dict(registered_input={'composition': 'h5ad-science.v1'}), (legacy, target)) == [
        dict(handle='resource.target', label=target.label)]
