"""Streamed attachments enter the existing registered-input turn boundary."""
import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient

from agent.application.local_resources import LocalResourceAdmission, RegisteredInput
from agent.application.uploads import H5ADUploadAdmission
from agent.web.app import create_app

from helpers import harness, submission, wait_turn


@pytest.fixture
def uploaded_web(tmp_path):
    backend = harness(tmp_path)
    root = backend.service._application._workspace.root / 'uploads'
    root.mkdir()
    backend.service.resources = LocalResourceAdmission(backend.service._application._workspace,
        approved_source_roots=(root,), registry=backend.service._application.registry)
    uploads = H5ADUploadAdmission(backend.service.resources, root)
    app = create_app(backend.service, input_sets=backend.input_sets, uploads=uploads)
    with TestClient(app) as client:
        yield client, backend, uploads, app


def attach(client, content, *, upload_id='first-upload', filename='tiny.h5ad', **kwargs):
    return client.put('/api/v1/uploads/' + upload_id, params={'filename': filename},
        content=content, headers={'Content-Type': 'application/octet-stream'}, **kwargs)


def test_upload_registration_discovery_and_retry_use_only_safe_metadata(uploaded_web, tmp_path):
    client, backend, uploads, _ = uploaded_web
    data = backend.source.read_bytes()
    response = attach(client, data)
    assert response.status_code == 201, response.text
    resource = response.json()
    assert set(resource) == {'resource_id', 'input_type', 'label', 'status'}
    assert resource['input_type'] == 'h5ad' and resource['status'] == 'registered'
    assert str(tmp_path) not in response.text
    assert client.get('/api/v1/resources').json() == {'enabled': True, 'choices': [resource]}
    assert uploads.resolve(resource['resource_id']).resource_id == resource['resource_id']
    retried = attach(client, data)
    assert retried.status_code == 201 and retried.json() == resource
    conflicting = attach(client, data + b'changed')
    assert conflicting.status_code == 409
    assert client.get('/api/v1/resources').json()['choices'] == [resource]
    assert not backend.models and not backend.science_calls


@pytest.mark.parametrize('upload_id', ['turns', 'activate'])
def test_upload_identity_does_not_enter_json_submission_middleware(uploaded_web, upload_id):
    client, backend, _, _ = uploaded_web
    response = attach(client, backend.source.read_bytes(), upload_id=upload_id)
    assert response.status_code == 201, response.text
    assert not backend.models and not backend.science_calls


def test_registered_selection_uses_both_existing_facade_methods(uploaded_web, monkeypatch):
    client, backend, _, _ = uploaded_web
    resource = attach(client, backend.source.read_bytes()).json()
    calls = []
    for method in ('validate_submission', 'submit_turn'):
        original = getattr(backend.service, method)
        def traced(*args, _method=method, _original=original, **kwargs):
            calls.append((_method, kwargs))
            return _original(*args, **kwargs)
        monkeypatch.setattr(backend.service, method, traced)
    assert client.post('/api/v1/sessions', json={'session_id': 'session'}).status_code == 201
    body = submission(resource_id=resource['resource_id'], input_set_id=None, profile_id='beta')
    response = client.post('/api/v1/sessions/session/turns', json=body)
    assert response.status_code == 202, response.text
    turn = wait_turn(client)
    assert turn['status'] == 'succeeded' and turn['revision_id']
    assert backend.science_calls == ['inspect_scATAC']
    assert [method for method, _ in calls] == ['validate_submission', 'submit_turn']
    assert all(isinstance(kwargs['registered_input'], RegisteredInput) and 'execution_inputs' not in kwargs
               for _, kwargs in calls)
    assert calls[0][1]['registered_input'] == calls[1][1]['registered_input']
    assert backend.models[0].model_id == 'model/beta'
    state = backend.service._application.sessions.load('session')
    attribution = state.interactions[0].submission['registered_input']
    assert attribution['resource_id'] == resource['resource_id']
    assert attribution['tool_name'] == 'inspect_scATAC'


@pytest.mark.parametrize('changes,status', [
    ({'resource_id': 'not-a-resource'}, 422),
    ({'resource_id': 'local-' + '0' * 64}, 404),
    ({'source_path': '/private/source.h5ad'}, 422),
    ({'source_sha256': '0' * 64}, 422),
    ({'registered_input': {}}, 422),
])
def test_forged_resource_and_source_injection_are_rejected(uploaded_web, changes, status):
    client, backend, _, _ = uploaded_web
    client.post('/api/v1/sessions', json={'session_id': 'session'})
    body = submission(input_set_id=None, **changes)
    response = client.post('/api/v1/sessions/session/turns', json=body)
    assert response.status_code == status, response.text
    assert set(response.json()) == {'error'}
    assert '/private/source' not in response.text
    assert not backend.models and not backend.science_calls


def test_conflicting_input_selection_and_stale_generation_fail_before_models(uploaded_web):
    client, backend, _, _ = uploaded_web
    resource = attach(client, backend.source.read_bytes()).json()
    client.post('/api/v1/sessions', json={'session_id': 'session'})
    response = client.post('/api/v1/sessions/session/turns',
        json=submission(resource_id=resource['resource_id']))
    assert response.status_code == 400 and response.json()['error']['code'] == 'INTERACTIVE_INPUT_INVALID'
    response = client.post('/api/v1/sessions/session/turns',
        json=submission(resource_id=resource['resource_id'], input_set_id=None, generation=1))
    assert response.status_code == 409 and response.json()['error']['code'] == 'INTERACTIVE_GENERATION_CONFLICT'
    assert not backend.models and not backend.science_calls


def test_completed_retry_preserves_source_lifetime_contract(uploaded_web):
    client, backend, uploads, _ = uploaded_web
    resource = attach(client, backend.source.read_bytes()).json()
    client.post('/api/v1/sessions', json={'session_id': 'session'})
    body = submission(resource_id=resource['resource_id'], input_set_id=None)
    assert client.post('/api/v1/sessions/session/turns', json=body).status_code == 202
    original = wait_turn(client)
    calls = (len(backend.models), len(backend.science_calls))
    record = uploads.resources.load(resource['resource_id'])
    from pathlib import Path
    Path(record.source_path).unlink()
    assert client.post('/api/v1/sessions/session/turns', json=body).status_code == 202
    assert wait_turn(client) == original
    assert (len(backend.models), len(backend.science_calls)) == calls
    response = client.post('/api/v1/sessions/session/turns', json=submission(turn='new', generation=1,
        resource_id=resource['resource_id'], input_set_id=None))
    assert response.status_code == 400 and response.json()['error']['code'] == 'LOCAL_RESOURCE_INTEGRITY_INVALID'
    assert (len(backend.models), len(backend.science_calls)) == calls


def test_corrupt_h5ad_registers_bytes_then_fails_in_existing_scientific_owner(uploaded_web):
    client, backend, _, _ = uploaded_web
    response = attach(client, b'corrupt scientific content')
    assert response.status_code == 201 and not backend.models and not backend.science_calls
    client.post('/api/v1/sessions', json={'session_id': 'session'})
    response = client.post('/api/v1/sessions/session/turns',
        json=submission(resource_id=response.json()['resource_id'], input_set_id=None))
    assert response.status_code == 202
    turn = wait_turn(client)
    assert turn['status'] == 'failed' and turn['revision_id'] is None
    assert backend.science_calls == ['inspect_scATAC']
    run = backend.service._application.run_store.load(turn['run_id'])
    assert turn['error']['code'] == run.errors[0].code


def test_upload_is_opt_in_and_keeps_ordinary_json_bounds(tmp_path):
    backend = harness(tmp_path)
    with TestClient(create_app(backend.service, input_sets=backend.input_sets)) as client:
        assert client.get('/api/v1/resources').json() == {'enabled': False, 'choices': []}
        assert attach(client, b'data').status_code == 404
        client.post('/api/v1/sessions', json={'session_id': 'session'})
        oversized = dict(submission(), padding='x' * 32768)
        response = client.post('/api/v1/sessions/session/turns', json=oversized)
        assert response.status_code == 413
        response = client.post('/api/v1/sessions/session/turns',
            json=submission(resource_id='local-' + '0' * 64, input_set_id=None))
        assert response.status_code == 400
        assert not backend.models and not backend.science_calls


@pytest.mark.parametrize('filename', ['../tiny.h5ad', '/tiny.h5ad', 'folder\\tiny.h5ad', '.', '..'])
def test_unsafe_filenames_never_register(uploaded_web, filename):
    client, backend, _, _ = uploaded_web
    response = attach(client, b'data', filename=filename)
    assert response.status_code == 400
    assert client.get('/api/v1/resources').json()['choices'] == []
    assert not backend.models and not backend.science_calls


def test_upload_content_type_field_injection_origin_and_length_fail_closed(uploaded_web):
    client, backend, _, _ = uploaded_web
    url = '/api/v1/uploads/first-upload'
    assert client.put(url, params={'filename': 'tiny.h5ad'}, content=b'data').status_code == 415
    assert client.put(url, params={'filename': 'tiny.h5ad', 'source_path': '/private/source'},
        content=b'data', headers={'Content-Type': 'application/octet-stream'}).status_code == 422
    assert client.put(url, params={'filename': 'tiny.h5ad'}, content=b'data',
        headers={'Content-Type': 'application/octet-stream', 'Origin': 'https://other.example'}).status_code == 403
    for length, status in (('-1', 422), ('invalid', 422), ('999999999999999999999999', 413)):
        assert client.put(url, params={'filename': 'tiny.h5ad'}, content=b'data',
            headers={'Content-Type': 'application/octet-stream', 'Content-Length': length}).status_code == status
    response = client.get('/api/v1/resources')
    assert response.json()['choices'] == []
    assert response.headers['Cache-Control'] == 'no-store'
    assert response.headers['X-Content-Type-Options'] == 'nosniff'
    assert not backend.models and not backend.science_calls


def test_actual_stream_size_limit_without_content_length(uploaded_web):
    _, backend, _, _ = uploaded_web
    root = backend.service.resources._approved[0]
    uploads = H5ADUploadAdmission(backend.service.resources, root, max_bytes=8)
    with TestClient(create_app(backend.service, uploads=uploads)) as client:
        assert attach(client, b'123456789').status_code == 413
    app = create_app(backend.service, uploads=uploads)
    async def exercise():
        async def chunks():
            yield b'1234'
            yield b'5678'
            yield b'9'
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://testserver') as client:
            response = await client.put('/api/v1/uploads/chunked', params={'filename': 'tiny.h5ad'},
                content=chunks(), headers={'Content-Type': 'application/octet-stream'})
            assert response.status_code == 413, response.text
            assert (await client.get('/api/v1/resources')).json()['choices'] == []
    asyncio.run(exercise())
    assert not backend.models and not backend.science_calls


def test_content_length_mismatch_does_not_publish(uploaded_web):
    client, _, _, _ = uploaded_web
    response = client.put('/api/v1/uploads/first-upload', params={'filename': 'tiny.h5ad'}, content=b'1234',
        headers={'Content-Type': 'application/octet-stream', 'Content-Length': '8'})
    assert response.status_code == 400
    assert client.get('/api/v1/resources').json()['choices'] == []


def test_valid_content_length_accepts_leading_zeroes_without_integer_overflow(uploaded_web):
    client, backend, _, _ = uploaded_web
    response = client.put('/api/v1/uploads/first-upload', params={'filename': 'tiny.h5ad'}, content=b'1234',
        headers={'Content-Type': 'application/octet-stream', 'Content-Length': '0' * 100 + '4'})
    assert response.status_code == 201, response.text
    assert not backend.models and not backend.science_calls
