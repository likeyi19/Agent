"""Uploads compose one typed binding with exact operator-reviewed resources."""
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from agent.application.local_resources import LocalResourceAdmission, RegisteredInput
from agent.application.uploads import H5ADUploadAdmission
from agent.web.app import create_app
from agent.web.config import QualifiedEpiZooResource, ScientificInputSet

from helpers import harness, submission, wait_turn
from test_config import qualified_resource
from test_upload_api import attach


def web_composition(tmp_path, *, companions=None, resources=()):
    backend = harness(tmp_path)
    root = backend.service._application._workspace.root / 'uploads'
    root.mkdir()
    backend.service.resources = LocalResourceAdmission(backend.service._application._workspace,
        approved_source_roots=(root,), registry=backend.service._application.registry)
    uploads = H5ADUploadAdmission(backend.service.resources, root)
    companions = ((ScientificInputSet('mouse', 'Mouse declarations', {'species': 'mouse'}, h5ad_companion=True),)
                  if companions is None else companions)
    app = create_app(backend.service, uploads=uploads, input_sets=companions, epizoo_resources=resources)
    return backend, uploads, app


def resource(**changes):
    return QualifiedEpiZooResource(**qualified_resource(**changes))


@pytest.mark.parametrize('explicit,expected,second_default', [
    (None, 'mouse-default', False), ('mouse-explicit', 'mouse-explicit', False),
    ('mouse-explicit', 'mouse-explicit', True),
])
def test_exact_resource_selection_uses_explicit_choice_before_unique_default(tmp_path, monkeypatch, explicit, expected, second_default):
    resources = (resource(resource_id='mouse-default'), resource(resource_id='mouse-explicit', default=second_default,
        checkpoint_path='/operator/other.pt', checkpoint_sha256='d' * 64))
    backend, uploads, app = web_composition(tmp_path, resources=resources)
    calls = []
    for name in ('validate_submission', 'submit_turn'):
        original = getattr(backend.service, name)
        def capture(*args, _name=name, _original=original, **kwargs):
            calls.append((_name, kwargs))
            return _original(*args, **kwargs)
        monkeypatch.setattr(backend.service, name, capture)
    with TestClient(app) as client:
        catalog = client.get('/api/v1/epizoo-resources')
        assert catalog.json() == {'choices': [r.choice() for r in resources]}
        assert '/operator' not in catalog.text and 'sha256' not in catalog.text and 'qualification' not in catalog.text
        uploaded = attach(client, backend.source.read_bytes()).json()
        assert not backend.models and not backend.science_calls
        client.post('/api/v1/sessions', json={'session_id': 'session'})
        body = submission(resource_id=uploaded['resource_id'], input_set_id='mouse')
        if explicit is not None:
            body['epizoo_resource_id'] = explicit
        response = client.post('/api/v1/sessions/session/turns', json=body)
        assert response.status_code == 202, response.text
        original = wait_turn(client)
        assert original['status'] == 'succeeded'
        assert [name for name, _ in calls] == ['validate_submission', 'submit_turn']
        binding = calls[0][1]['registered_input']
        assert isinstance(binding, RegisteredInput) and binding == calls[1][1]['registered_input']
        assert all('execution_inputs' not in values for _, values in calls)
        assert binding.resource_id == uploaded['resource_id'] and binding.composition == 'h5ad-science.v1'
        assert binding.execution_inputs['species'] == 'mouse'
        assert binding.execution_inputs['input_path'] == uploads.resources.load(uploaded['resource_id']).source_path
        assert binding.execution_inputs['expected_resource_identity']['resource_id'] == expected
        # Pins are captured even for an ordinary inspection; no embedding is selected by Web.
        assert backend.science_calls == ['inspect_scATAC']
        stored = backend.service._application.sessions.load('session').interactions[0].submission
        assert stored['execution_inputs']['expected_resource_identity']['resource_id'] == expected
        Path(binding.execution_inputs['input_path']).unlink()
        counts = (len(backend.models), len(backend.science_calls))
        assert client.post('/api/v1/sessions/session/turns', json=body).status_code == 202
        assert wait_turn(client) == original and counts == (len(backend.models), len(backend.science_calls))


@pytest.mark.parametrize('inputs,resources,error', [
    ({}, (), 'EPIZOO_RESOURCE_REQUIRED'),
    ({'species': 'mouse'}, (), 'EPIZOO_RESOURCE_REQUIRED'),
    ({'species': 'mouse'}, (resource(resource_id='first'), resource(resource_id='second')), 'EPIZOO_RESOURCE_AMBIGUOUS'),
])
def test_missing_or_ambiguous_resources_do_not_block_inspection(tmp_path, inputs, resources, error):
    companion = ScientificInputSet('mouse', 'Explicit declarations', inputs, h5ad_companion=True)
    backend, _, app = web_composition(tmp_path, companions=(companion,), resources=resources)
    with TestClient(app) as client:
        uploaded = attach(client, backend.source.read_bytes()).json()
        client.post('/api/v1/sessions', json={'session_id': 'session'})
        response = client.post('/api/v1/sessions/session/turns',
            json=submission(resource_id=uploaded['resource_id'], input_set_id='mouse'))
        assert response.status_code == 202, response.text
        assert wait_turn(client)['status'] == 'succeeded'
        state = backend.service._application.sessions.load('session')
        assert state.interactions[0].submission['registered_input']['resource_selection_error'] == error
        assert backend.science_calls == ['inspect_scATAC']


@pytest.mark.parametrize('inputs,selection,expected', [
    ({'species': 'mouse'}, 'absent', 'EPIZOO_RESOURCE_SELECTION_INVALID'),
    ({'species': 'human'}, 'mouse-reviewed', 'EPIZOO_RESOURCE_SELECTION_INVALID'),
    ({}, 'mouse-reviewed', 'H5AD_SPECIES_REQUIRED'),
    ({'input_path': '/private/other.h5ad', 'species': 'mouse'}, None, 'LOCAL_RESOURCE_BINDING_INVALID'),
    ({'extra': 'unsupported'}, None, 'LOCAL_RESOURCE_BINDING_INVALID'),
])
def test_invalid_explicit_selection_and_companion_source_injection_fail_before_providers(tmp_path, inputs, selection, expected):
    companions = (ScientificInputSet('companion', 'Explicit declarations', inputs, h5ad_companion=True),)
    backend, _, app = web_composition(tmp_path, companions=companions, resources=(resource(),))
    with TestClient(app) as client:
        uploaded = attach(client, backend.source.read_bytes()).json()
        client.post('/api/v1/sessions', json={'session_id': 'session'})
        body = submission(resource_id=uploaded['resource_id'], input_set_id='companion')
        if selection is not None:
            body['epizoo_resource_id'] = selection
        response = client.post('/api/v1/sessions/session/turns', json=body)
        assert response.status_code == 400, response.text
        assert response.json()['error']['code'] == expected and '/private' not in response.text
        assert not backend.models and not backend.science_calls


def test_companion_and_epizoo_selection_require_selected_upload(tmp_path):
    backend, _, app = web_composition(tmp_path, resources=(resource(),))
    with TestClient(app) as client:
        client.post('/api/v1/sessions', json={'session_id': 'session'})
        for body in (submission(input_set_id='mouse'),
                     submission(input_set_id=None, epizoo_resource_id='mouse-reviewed')):
            response = client.post('/api/v1/sessions/session/turns', json=body)
            assert response.status_code == 400 and response.json()['error']['code'] == 'INTERACTIVE_INPUT_INVALID'
        assert not backend.models and not backend.science_calls


def test_restarted_web_retries_do_not_replay_and_changed_pins_conflict(tmp_path):
    configured = resource()
    companion = ScientificInputSet('mouse', 'Mouse declarations', {'species': 'mouse'}, h5ad_companion=True)
    backend, uploads, app = web_composition(tmp_path, companions=(companion,), resources=(configured,))
    with TestClient(app) as client:
        uploaded = attach(client, backend.source.read_bytes()).json()
        client.post('/api/v1/sessions', json={'session_id': 'session'})
        body = submission(resource_id=uploaded['resource_id'], input_set_id='mouse')
        assert client.post('/api/v1/sessions/session/turns', json=body).status_code == 202
        original = wait_turn(client)
    counts = (len(backend.models), len(backend.science_calls))
    Path(uploads.resources.load(uploaded['resource_id']).source_path).unlink()
    # Rebuild the HTTP worker without receipts. Immutable application capture is
    # the duplicate owner; historical sources are not reopened for this read.
    restarted = create_app(backend.service, uploads=uploads, input_sets=(companion,), epizoo_resources=(configured,))
    with TestClient(restarted) as client:
        assert client.get('/api/v1/sessions/session').status_code == 200
        assert client.post('/api/v1/sessions/session/turns', json=body).status_code == 202
        assert wait_turn(client) == original
    changed = create_app(backend.service, uploads=uploads, input_sets=(companion,),
        epizoo_resources=(resource(checkpoint_sha256='d' * 64),))
    with TestClient(changed) as client:
        response = client.post('/api/v1/sessions/session/turns', json=body)
        assert response.status_code == 409 and response.json()['error']['code'] == 'INTERACTIVE_TURN_CONFLICT'
    assert counts == (len(backend.models), len(backend.science_calls))
