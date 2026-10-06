"""M18.3 HTTP contracts use the actual durable interactive application."""
import json

import pytest
from fastapi.testclient import TestClient

from agent.application import InteractiveAgentApplication, InteractiveBoundaryError
from agent.orchestration import PlanningModelProfile
from agent.providers import PlanningModelFactoryRegistry
from agent.web.app import create_app
from helpers import harness, submission, wait_turn


@pytest.fixture
def web(tmp_path):
    backend = harness(tmp_path)
    with TestClient(create_app(backend.service, input_sets=backend.input_sets)) as client:
        yield client, backend
    backend.release.set()


def create_session(client, session='session'):
    response = client.post('/api/v1/sessions', json={'session_id': session})
    assert response.status_code in (200, 201), response.text
    return response.json()


def submit(client, body=None, *, session='session'):
    return client.post(f'/api/v1/sessions/{session}/turns', json=body or submission())


def assert_error(response, code, status):
    assert response.status_code == status, response.text
    value = response.json()
    assert set(value) == {'error'} and set(value['error']) == {'code', 'message'}
    assert value['error']['code'] == code
    assert value['error']['message']
    return value


def test_health_choices_and_empty_create_reopen(web, tmp_path):
    client, backend = web
    assert client.get('/api/v1/health').status_code == 200
    choices = client.get('/api/v1/models').json()['choices']
    assert [c['profile_id'] for c in choices] == ['alpha', 'beta']
    assert choices[0]['is_default'] and not choices[1]['is_default']
    assert choices[0]['display_label'] == 'Default test model'
    assert client.get('/api/v1/input-sets').json() == {'choices': [
        {'input_set_id': 'tiny', 'display_label': 'Tiny sparse matrix'}]}
    view = create_session(client)
    assert view['session_id'] == 'session' and view['generation'] == 0
    assert view['active_revision_id'] is None and view['turns'] == []
    assert client.get('/api/v1/sessions/session').json() == view
    assert str(tmp_path) not in json.dumps(choices)
    assert not backend.models and not backend.science_calls


def test_generated_session_identity(web):
    client, backend = web
    response = client.post('/api/v1/sessions', json={})
    assert response.status_code in (200, 201)
    view = response.json()
    assert view['session_id'] and view['generation'] == 0
    assert client.get('/api/v1/sessions/' + view['session_id']).json() == view
    assert not backend.models


def test_page_and_static_assets_resolve_without_domain_operations(web):
    client, backend = web
    page = client.get('/')
    assert page.status_code == 200 and '<h1>Agent</h1>' in page.text
    assert page.headers['content-type'].startswith('text/html')
    for path in ('/static/app.js', '/static/styles.css'):
        response = client.get(path)
        assert response.status_code == 200 and response.text
        assert response.headers['x-content-type-options'] == 'nosniff'
    assert client.get('/static/..%2F..%2Finteractive.py').status_code == 404
    assert not backend.models and not backend.science_calls


def test_first_execution_followup_and_exact_reopen_without_provider(web, tmp_path):
    client, backend = web
    create_session(client)
    accepted = submit(client)
    assert accepted.status_code == 202, accepted.text
    assert accepted.json() == {'session_id': 'session', 'turn_id': 'first'}
    executed = wait_turn(client)
    assert executed['status'] == 'succeeded'
    assert executed['response']['kind'] == 'execute'
    assert executed['response']['status'] == 'activated' and executed['revision_id']
    assert executed['steps'][0]['status'] == 'SUCCEEDED'
    assert backend.science_calls == ['inspect_scATAC']
    assert len(backend.models) == 1
    assert client.get('/api/v1/sessions/session/turns/first/status').json() == executed
    follow = submission('follow', 1, 'What does this result show?', profile_id='beta')
    assert submit(client, follow).status_code == 202
    answer = wait_turn(client, 'follow')
    assert answer['response']['kind'] == 'answer' and answer['response']['status'] == 'answered'
    assert answer['profile_id'] == 'beta' and answer['response']['scientific']['support'] == 'supported'
    assert backend.science_calls == ['inspect_scATAC']
    reopened = client.get('/api/v1/sessions/session').json()
    assert reopened['generation'] == 1 and reopened['active_revision_id'] == executed['revision_id']
    assert [t['response'] for t in reopened['turns']] == [executed['response'], answer['response']]
    before = [(m, len(m.calls)) for m in backend.models]
    assert submit(client).status_code == 202
    assert wait_turn(client) == executed
    assert client.get('/api/v1/sessions/session').json() == reopened
    assert [(m, len(m.calls)) for m in backend.models] == before
    assert_error(submit(client, submission('stale', 0)), 'INTERACTIVE_GENERATION_CONFLICT', 409)
    assert [(m, len(m.calls)) for m in backend.models] == before
    assert str(tmp_path) not in json.dumps(reopened)
    assert 'authority_payload' not in json.dumps(reopened)


def test_clarification_is_persisted_without_science(web):
    client, backend = web
    create_session(client)
    body = submission(utterance='Please clarify the stricter threshold.')
    assert submit(client, body).status_code == 202
    view = wait_turn(client)
    assert view['response']['kind'] == 'clarify'
    assert view['response']['clarification']['reason'] == 'missing_parameter_value'
    assert view['response']['clarification']['value_required']
    assert not backend.science_calls
    assert client.get('/api/v1/sessions/session').json()['turns'][0]['response'] == view['response']


@pytest.mark.parametrize('body,code,status', [
    (submission(profile_id='unknown'), 'INTERACTIVE_MODEL_UNAVAILABLE', 400),
    (submission(generation=1), 'INTERACTIVE_GENERATION_CONFLICT', 409),
    (submission(predecessor_turn_id='absent'), 'INTERACTIVE_REFERENCE_INVALID', 404),
    (submission(input_set_id='absent'), 'INTERACTIVE_INPUT_INVALID', 400),
])
def test_rejected_admission_before_provider(web, body, code, status):
    client, backend = web
    create_session(client)
    assert_error(submit(client, body), code, status)
    assert not backend.models and not backend.science_calls
    assert client.get('/api/v1/sessions/session').json()['turns'] == []


@pytest.mark.parametrize('extra', [
    {'execution_inputs': {'input_path': '/server/private/input.h5ad'}},
    {'output_dir': '/server/private'}, {'plan': {}}, {'authority_payload': {}},
    {'profile': {'provider_id': 'openai', 'api_key': 'secret-key-sentinel'}},
    {'provider_endpoint': 'https://untrusted.example'},
])
def test_browser_cannot_submit_internal_or_provider_configuration(web, extra):
    client, backend = web
    create_session(client)
    response = submit(client, dict(submission(), **extra))
    assert response.status_code == 422
    assert set(response.json()) == {'error'}
    assert 'secret-key-sentinel' not in response.text and '/server/private' not in response.text
    assert not backend.models and not backend.science_calls


@pytest.mark.parametrize('extra', [
    {'expected_generation': True}, {'expected_generation': -1}, {'utterance': ''},
    {'utterance': 'x' * 4097}, {'turn_id': ''},
])
def test_invalid_submission_uses_clean_error(web, extra):
    client, backend = web
    create_session(client)
    response = submit(client, dict(submission(), **extra))
    assert response.status_code == 422
    assert set(response.json()) == {'error'}
    assert not backend.models


@pytest.mark.parametrize('identifier', ['group/one', 'group\\one', '.', '..'])
def test_unaddressable_session_ids_rejected_before_creation(web, identifier):
    client, backend = web
    response = client.post('/api/v1/sessions', json={'session_id': identifier})
    assert_error(response, 'WEB_REQUEST_INVALID', 422)
    with pytest.raises(InteractiveBoundaryError) as caught:
        backend.service.reopen_session(identifier)
    assert caught.value.error.code == 'INTERACTIVE_SESSION_INVALID'
    assert not backend.models


@pytest.mark.parametrize('identifier', ['turn/one', 'turn\\one', '.', '..'])
def test_unaddressable_turn_ids_rejected_before_admission(web, identifier):
    client, backend = web
    create_session(client)
    assert_error(submit(client, submission(turn=identifier)), 'WEB_REQUEST_INVALID', 422)
    assert client.get('/api/v1/sessions/session').json()['turns'] == []
    assert not backend.models and not backend.science_calls


def test_unknown_session_turn_and_malformed_body_are_safe(web):
    client, backend = web
    assert_error(client.get('/api/v1/sessions/absent'), 'INTERACTIVE_SESSION_INVALID', 404)
    create_session(client)
    assert_error(client.get('/api/v1/sessions/session/turns/absent'), 'INTERACTIVE_REFERENCE_INVALID', 404)
    bad = client.post('/api/v1/sessions/session/turns', content='{', headers={'Content-Type': 'application/json'})
    assert bad.status_code == 422 and set(bad.json()) == {'error'}
    assert 'traceback' not in bad.text.lower() and not backend.models


def test_nonblocking_active_retry_conflict_and_cooperative_cancel(tmp_path):
    backend = harness(tmp_path, blocked=True)
    with TestClient(create_app(backend.service, input_sets=backend.input_sets, max_workers=1)) as client:
        try:
            create_session(client)
            assert submit(client).status_code == 202
            assert backend.started.wait(20)
            pending = client.get('/api/v1/sessions/session/turns/first/status').json()
            assert pending['status'] == 'running' and pending['response'] is None
            assert pending['steps'][0]['status'] == 'RUNNING' and pending['steps'][0]['attempts'] == 1
            assert not {'percentage', 'eta', 'worker_alive'} & set(pending)
            assert submit(client).status_code == 202
            conflicting = dict(submission(), utterance='Inspect a different matrix.')
            assert_error(submit(client, conflicting), 'INTERACTIVE_TURN_CONFLICT', 409)
            assert len(backend.models) == 1 and backend.science_calls == ['inspect_scATAC']
            busy = submit(client, submission('another'))
            assert_error(busy, 'WEB_WORKERS_BUSY', 503)
            assert busy.headers['Retry-After'] == '2'
            assert len(backend.models) == 1
            cancelled = client.post('/api/v1/sessions/session/turns/first/cancel')
            assert cancelled.status_code == 200 and cancelled.json()['turn_id'] == 'first'
            backend.release.set()
            final = wait_turn(client)
            assert final['status'] == 'cancelled' and final['revision_id'] is None
            assert backend.science_calls == ['inspect_scATAC']
        finally:
            backend.release.set()


def test_new_server_reopens_exact_history_and_retries_without_provider(tmp_path):
    backend = harness(tmp_path)
    with TestClient(create_app(backend.service, input_sets=backend.input_sets)) as client:
        create_session(client)
        assert submit(client).status_code == 202
        executed = wait_turn(client)
        history = client.get('/api/v1/sessions/session').json()

    def forbidden(profile):
        raise AssertionError('Reopen or retry constructed a provider.')

    reopened = InteractiveAgentApplication(tmp_path / 'workspace',
        model_profiles=tuple(PlanningModelProfile(key, 'scripted', 'model/' + key) for key in ('alpha', 'beta')),
        default_profile_id='alpha',
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': forbidden}))
    with TestClient(create_app(reopened, input_sets=backend.input_sets)) as client:
        assert client.get('/api/v1/sessions/session').json() == history
        assert submit(client).status_code == 202
        assert wait_turn(client) == executed
        assert backend.science_calls == ['inspect_scATAC']


def test_unexpected_scientific_failure_does_not_expose_exception(tmp_path):
    backend = harness(tmp_path / 'failure', fail_science=True)
    with TestClient(create_app(backend.service, input_sets=backend.input_sets)) as client:
        create_session(client)
        assert submit(client).status_code == 202
        failed = wait_turn(client)
        assert failed['status'] == 'failed'
        assert failed['error']['code'] and failed['response']['text']
        encoded = json.dumps(failed)
        assert 'secret-key-sentinel' not in encoded and '/server/private' not in encoded
        assert 'traceback' not in encoded.lower()


def test_pre_capture_provider_failure_is_safe_transient_status(tmp_path):
    backend = harness(tmp_path, fail_factory=True)
    with TestClient(create_app(backend.service, input_sets=backend.input_sets)) as client:
        create_session(client)
        assert submit(client).status_code == 202
        failed = wait_turn(client, terminal_status='failed')
        assert failed['status'] == 'failed' and failed['response'] is None
        assert failed['error']['code'] == 'INTERACTIVE_MODEL_UNAVAILABLE'
        assert 'secret-key-sentinel' not in json.dumps(failed)
        assert '/server/private' not in json.dumps(failed)
        assert client.get('/api/v1/sessions/session').json()['turns'] == []
        assert submit(client).status_code == 202
        assert wait_turn(client, terminal_status='failed') == failed
        assert len(backend.models) == 1 and not backend.science_calls


def test_validate_submission_is_read_only_and_normalizes_default_profile(tmp_path):
    backend = harness(tmp_path)
    backend.service.create_session('session')
    before = backend.service.reopen_session('session')
    arguments = dict(expected_generation=0, execution_inputs={'input_path': str(backend.source)})
    default = backend.service.validate_submission('session', 'first', 'Inspect the supplied matrix.', **arguments)
    explicit = backend.service.validate_submission('session', 'first', 'Inspect the supplied matrix.',
        profile_id='alpha', **arguments)
    assert len(default) == 64 and default == explicit
    assert backend.service.reopen_session('session') == before
    assert not backend.models and not backend.science_calls
