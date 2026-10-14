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


@pytest.mark.parametrize('replace_source', [False, True])
def test_target_resource_reply_preserves_registered_source_and_objective(tmp_path, monkeypatch, replace_source):
    original = helpers.ScriptedModel
    class Model(original):
        def complete(self, *, prompt, response_schema):
            value = json.loads(prompt)
            if 'turn_schema_version' in value:
                self.calls.append(value)
                decision = (dict(kind='request_resources', target='inspect_scATAC', arguments=[])
                    if value['utterance'].startswith('Inspect') else
                    dict(kind='answer_resources', pending='@resources', choice='resource.target', arguments=[]))
                return json.dumps(dict(turn_schema_version=1, decision=decision))
            return super().complete(prompt=prompt, response_schema=response_schema)
    monkeypatch.setattr(helpers, 'ScriptedModel', Model)
    choice = target_choice()
    backend, uploads, app = web_composition(tmp_path, companions=(choice,))
    with TestClient(app) as client:
        source = attach(client, backend.source.read_bytes()).json()
        client.post('/api/v1/sessions', json={'session_id': 'session'})
        objective = 'Inspect this file with the reviewed target species context.'
        first = submission(resource_id=source['resource_id'], input_set_id=None, utterance=objective)
        first.pop('input_set_id')
        assert client.post('/api/v1/sessions/session/turns', json=first).status_code == 202
        assert wait_turn(client)['status'] == 'clarification'
        assert not backend.science_calls
        if replace_source:
            import anndata as ad
            changed = ad.read_h5ad(backend.source)
            changed.obs_names = ['other-a', 'other-b']
            altered = tmp_path / 'other.h5ad'
            changed.write_h5ad(altered)
            source = attach(client, altered.read_bytes(), upload_id='other').json()
        answer = submission(turn='answer', utterance='Use the reviewed target model.',
                            resource_id=source['resource_id'], input_set_id='target')
        assert client.post('/api/v1/sessions/session/turns', json=answer).status_code == 202
        result = wait_turn(client, 'answer')
        if replace_source:
            assert result['status'] == 'clarification'
            assert not backend.science_calls
        else:
            assert result['status'] == 'succeeded', result
            run = backend.service._application.run_store.load(result['run_id'])
            assert run.request.prompt == objective
            state = backend.service._application.sessions.load('session')
            admitted = state.interactions[-1].admitted
            assert admitted['inputs']['input_path'] == uploads.resources.load(source['resource_id']).source_path
            assert admitted['inputs']['target_model_manifest_path'] == choice.inputs()['target_model_manifest_path']
            assert admitted['registered_input']['resource_choice'] == choice.resource_choice()
            assert len(state.revisions) == 1
        for model in backend.models:
            for prompt in model.calls:
                assert '/operator/target' not in json.dumps(prompt)


def test_target_catalog_does_not_change_legacy_resource_selection():
    target = target_choice()
    legacy = ScientificInputSet('human', 'Human defaults', {'species': 'human'}, h5ad_companion=True)
    assert [row['input_set_id'] for row in catalog((legacy, target))] == ['target']
    assert offers(dict(registered_input={'composition': 'h5ad-science.v1'}), (legacy, target)) == [
        dict(handle='resource.target', label=target.label)]
