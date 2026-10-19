"""Ordered independent sources use the existing interactive Web boundary."""
import pytest

from agent.application.local_resources import RegisteredInputGroup
from agent.schemas.orchestration import _serialize
from agent.web.app import SubmitTurn
from agent.web.config import ScientificInputSet
from helpers import submission, wait_turn
from test_upload_api import attach, uploaded_web


FIRST = 'local-' + '1' * 64
SECOND = 'local-' + '2' * 64


@pytest.mark.parametrize('changes,status', [
    ({'resource_ids': []}, 422),
    ({'resource_ids': [FIRST]}, 404),
    ({'resource_ids': ['local-' + str(value) * 64 for value in range(9)]}, 422),
    ({'resource_ids': [FIRST, FIRST]}, 422),
    ({'resource_ids': [FIRST, '/private/path.h5ad']}, 422),
    ({'resource_ids': [FIRST, 'local-' + 'A' * 64]}, 422),
    ({'resource_ids': [FIRST, 2]}, 422),
    ({'resource_ids': [FIRST, SECOND], 'resource_id': FIRST}, 422),
])
def test_selection_transport_rejects_invalid_or_unregistered_inputs(uploaded_web, changes, status):
    client, backend, _, _ = uploaded_web
    client.post('/api/v1/sessions', json={'session_id': 'session'})
    response = client.post('/api/v1/sessions/session/turns',
        json=submission(input_set_id=None, **changes))
    assert response.status_code == status, response.text
    if status == 404:
        # One-item selections are valid transport; identity lookup still fails closed.
        assert response.json()['error']['code'] == 'LOCAL_RESOURCE_UNAVAILABLE'
    assert not backend.models and not backend.science_calls
    assert not backend.service._application.sessions.load('session').interactions


def test_ordered_source_selection_uses_existing_facade_and_persists_exact_registration(uploaded_web, monkeypatch):
    client, backend, uploads, _ = uploaded_web
    first = attach(client, backend.source.read_bytes(), upload_id='first').json()
    second = attach(client, backend.source.read_bytes(), upload_id='second').json()
    assert not backend.models and not backend.science_calls
    calls = []
    for method in ('validate_submission', 'submit_turn'):
        original = getattr(backend.service, method)

        def observed(*args, _method=method, _original=original, **kwargs):
            calls.append((_method, kwargs))
            return _original(*args, **kwargs)

        monkeypatch.setattr(backend.service, method, observed)
    client.post('/api/v1/sessions', json={'session_id': 'session'})
    selected = [second['resource_id'], first['resource_id']]
    payload = submission(input_set_id=None, resource_ids=selected,
        utterance='Please clarify the roles of these selected inputs.')
    response = client.post('/api/v1/sessions/session/turns', json=payload)
    assert response.status_code == 202, response.text
    turn = wait_turn(client)
    assert turn['status'] == 'clarification' and turn['revision_id'] is None
    assert [method for method, _ in calls] == ['validate_submission', 'submit_turn']
    assert all(type(kwargs['registered_input']) is RegisteredInputGroup
        and 'execution_inputs' not in kwargs for _, kwargs in calls)
    assert calls[0][1]['registered_input'] == calls[1][1]['registered_input']
    interaction = backend.service._application.sessions.load('session').interactions[0]
    actual = _serialize(interaction.submission['registered_input'])
    assert actual['composition'] == 'scientific-role-selection.v1'
    assert [(member['resource_id'], member['record_sha256']) for member in actual['members']] == [
        (value, uploads.resources.load(value).record_sha256) for value in selected]
    assert [member['source_path'] for member in actual['members']] == [
        uploads.resources.load(value).source_path for value in selected]
    assert not backend.science_calls


@pytest.mark.parametrize('changes', [
    {'epizoo_resource_id': 'resource'},
    {'input_set_id': 'h5ad-context'},
    {'input_set_id': 'bam-context'},
    {'input_set_id': 'fragments-context'},
    {'input_set_id': 'fastq-context'},
])
def test_independent_group_cannot_reuse_one_source_companion_context(uploaded_web, changes):
    client, backend, uploads, _ = uploaded_web
    from agent.web.app import create_app
    from fastapi.testclient import TestClient

    first = attach(client, backend.source.read_bytes(), upload_id='first').json()
    second = attach(client, backend.source.read_bytes(), upload_id='second').json()
    contexts = tuple(ScientificInputSet(name + '-context', name + ' context', {},
        **{name + '_companion': True}) for name in ('h5ad', 'bam', 'fragments', 'fastq'))
    with TestClient(create_app(backend.service, uploads=uploads, input_sets=contexts)) as grouped:
        grouped.post('/api/v1/sessions', json={'session_id': 'session'})
        response = grouped.post('/api/v1/sessions/session/turns', json=submission(**(
            dict(input_set_id=None, resource_ids=[first['resource_id'], second['resource_id']]) | changes)))
        assert response.status_code == 400, response.text
        assert response.json()['error']['code'] == 'INTERACTIVE_INPUT_INVALID'
    assert not backend.models and not backend.science_calls


def test_transport_retains_submitted_order_without_assigning_scientific_roles():
    value = SubmitTurn(turn_id='turn', expected_generation=0, utterance='Compare selected inputs.',
        resource_ids=[SECOND, FIRST])
    assert value.resource_ids == [SECOND, FIRST]
    assert SubmitTurn(turn_id='single', expected_generation=0, utterance='Inspect selected input.',
                      resource_ids=[FIRST]).resource_ids == [FIRST]
    assert 'roles' not in SubmitTurn.model_fields


def test_unregistered_member_fails_before_the_interpreter(uploaded_web):
    client, backend, _, _ = uploaded_web
    first = attach(client, backend.source.read_bytes()).json()
    client.post('/api/v1/sessions', json={'session_id': 'session'})
    response = client.post('/api/v1/sessions/session/turns', json=submission(input_set_id=None,
        resource_ids=[first['resource_id'], FIRST]))
    assert response.status_code == 404, response.text
    assert not backend.models and not backend.science_calls
