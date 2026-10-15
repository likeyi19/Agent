"""USI-1: real streamed bytes, scoped references and durable neutral turn binding."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agent.application import InteractiveAgentApplication, InteractiveBoundaryError
from agent.application.local_resources import LocalResourceAdmission
from agent.application.uploads import H5ADUploadAdmission
from agent.providers import PlanningModelFactoryRegistry
from agent.web.app import create_app
from helpers import harness, submission, wait_turn


@pytest.fixture
def chat(tmp_path):
    backend = harness(tmp_path)
    root = backend.service._application.workspace_root / 'uploads'
    root.mkdir()
    backend.service.resources = LocalResourceAdmission(backend.service._application._workspace,
        approved_source_roots=(root,), registry=backend.service._application.registry)
    uploads = H5ADUploadAdmission(backend.service.resources, root)
    with TestClient(create_app(backend.service, uploads=uploads)) as client:
        for session in ('session', 'other'):
            assert client.post('/api/v1/sessions', json={'session_id': session}).status_code == 201
        yield client, backend, uploads


def upload(client, name, data, *, token='file', kind='attachment', session='session'):
    return client.put(f'/api/v1/sessions/{session}/uploads/{token}',
        params={'filename': name, 'input_type': kind}, content=data,
        headers={'Content-Type': 'application/octet-stream'})


def test_h5ad_and_neutral_attachment_bind_execute_and_survive_restart(chat):
    client, backend, uploads = chat
    h5ad = upload(client, 'matrix.h5ad', backend.source.read_bytes(), kind='h5ad').json()
    staged = upload(client, 'notes.tsv.gz', b'opaque scientific source', token='notes').json()
    ids = [h5ad['resource_id'], staged['resource_id']]
    assert not backend.models and not backend.science_calls
    body = submission(input_set_id=None, resource_id=ids[0], attachment_ids=ids, profile_id='beta')
    assert client.post('/api/v1/sessions/session/turns', json=body).status_code == 202
    turn = wait_turn(client)
    assert turn['status'] == 'succeeded'
    assert backend.science_calls == ['inspect_scATAC']
    assert backend.models[0].model_id == 'model/beta'
    assert [a['resource_id'] for a in turn['attachments']] == ids
    assert [a['use'] for a in turn['attachments']] == ['consumed', 'staged']
    assert all(set(a) == {'resource_id', 'label', 'input_type', 'use'} for a in turn['attachments'])
    state = backend.service._application.sessions.load('session')
    assert [a['resource_id'] for a in state.interactions[0].submission['attachments']] == ids
    assert state.interactions[0].submission['attachments'][0]['use'] == 'offered_for_analysis'
    assert 'attachments' not in state.revisions[0].__dataclass_fields__
    before = {p: p.read_bytes() for p in backend.service._application.workspace_root.rglob('*') if p.is_file()}
    def forbidden(profile):
        raise AssertionError('Reading/retrying persisted history must not construct a model.')
    restored = InteractiveAgentApplication(backend.service._application.workspace_root,
        model_profiles=tuple(backend.service._profiles.values()), default_profile_id='alpha',
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': forbidden}),
        approved_source_roots=(uploads.root,))
    restored_uploads = H5ADUploadAdmission(restored.resources, uploads.root)
    with TestClient(create_app(restored, uploads=restored_uploads)) as reopened:
        assert reopened.get('/api/v1/sessions/session/turns/first').json() == turn
        assert {r['resource_id'] for r in reopened.get('/api/v1/sessions/session/attachments').json()['choices']} == set(ids)
        # Completed retries do not require historical source bytes, nor replay science.
        for rid in ids:
            Path(uploads.resources.load(rid).source_path).unlink()
        assert reopened.post('/api/v1/sessions/session/turns', json=body).status_code == 202
        assert reopened.get('/api/v1/sessions/session/turns/first').json() == turn
    assert all(p.read_bytes() == data for p, data in before.items() if p.exists())


@pytest.mark.parametrize('name', ['reads.fastq.gz', 'reads.bam', 'fragments.tsv.gz', 'genome.fa',
                                  'genome.fa.fai', 'peaks.bed', 'groups.tsv', 'notes.txt', 'table.csv'])
def test_neutral_bytes_do_not_assign_roles(chat, name):
    client, backend, uploads = chat
    response = upload(client, name, b'bytes do not prove a format or role')
    assert response.status_code == 201, response.text
    record = uploads.resources.load(response.json()['resource_id'])
    assert record.input_type == 'attachment' and record.session_id == 'session'
    assert record.source_sha256 and record.size_bytes == len(b'bytes do not prove a format or role')
    assert not backend.models and not backend.science_calls
    assert client.get('/api/v1/resources').json()['choices'] == []
    assert client.get('/api/v1/sessions/other/attachments').json()['choices'] == []


def test_scope_cannot_be_bypassed_by_attachment_or_scientific_selection(chat):
    client, backend, uploads = chat
    rid = upload(client, 'tiny.h5ad', backend.source.read_bytes(), kind='h5ad').json()['resource_id']
    for changes in ({'attachment_ids': [rid]}, {'resource_id': rid}, {'resource_ids': [rid]}):
        response = client.post('/api/v1/sessions/other/turns', json=submission(input_set_id=None, **changes))
        assert response.status_code == 404, response.text
    with pytest.raises(InteractiveBoundaryError):
        backend.service.validate_submission('other', 'attempt', 'Inspect.', expected_generation=0,
            registered_input=uploads.resolve(rid), attachment_ids=())
    assert not backend.models and not backend.science_calls


def test_staged_binding_is_not_qualification_and_is_part_of_retry_identity(chat):
    client, backend, uploads = chat
    rid = upload(client, 'groups.tsv', b'cell_id\tgroup\nc1\tT').json()['resource_id']
    body = submission(input_set_id=None, utterance='Please clarify the required scientific inputs.', attachment_ids=[rid])
    assert client.post('/api/v1/sessions/session/turns', json=body).status_code == 202
    turn = wait_turn(client)
    assert turn['status'] == 'clarification'
    assert turn['attachments'][0]['use'] == 'staged'
    assert backend.science_calls == []
    assert client.post('/api/v1/sessions/session/turns', json=body).status_code == 202
    assert client.post('/api/v1/sessions/session/turns', json={**body, 'attachment_ids': []}).status_code == 409
    assert uploads.resources.load(rid).input_type == 'attachment'


@pytest.mark.parametrize('ids', [['/tmp/file.tsv'], ['local-' + '0' * 64], ['x'] * 17])
def test_no_client_path_or_unknown_reference_substitution(chat, ids):
    client, backend, _ = chat
    response = client.post('/api/v1/sessions/session/turns',
        json=submission(input_set_id=None, attachment_ids=ids))
    assert response.status_code in (404, 422), response.text
    assert not backend.models and not backend.science_calls


def test_scoped_upload_limits_conflicts_path_and_corrupt_h5ad(chat):
    client, backend, uploads = chat
    assert upload(client, '../escape', b'abc').status_code == 400
    uploads.max_bytes = 4
    assert upload(client, 'huge.bam', b'12345').status_code == 413
    assert upload(client, 'good.tsv', b'1234').status_code == 201
    assert upload(client, 'good.tsv', b'1234').status_code == 201
    assert upload(client, 'good.tsv', b'4321').status_code == 409
    assert list(uploads.staging.iterdir()) == []
    bad = upload(client, 'bad.h5ad', b'bad', token='bad', kind='h5ad').json()['resource_id']
    assert client.post('/api/v1/sessions/session/turns', json=submission(
        input_set_id=None, resource_id=bad, attachment_ids=[bad])).status_code == 202
    turn = wait_turn(client)
    assert turn['status'] == 'failed' and turn['revision_id'] is None
    assert turn['attachments'][0]['use'] == 'offered_for_analysis'
    assert backend.science_calls == ['inspect_scATAC']


def test_source_identity_survives_task_scoped_use(chat):
    client, backend, uploads = chat
    resource = upload(client, 'matrix.h5ad', backend.source.read_bytes(), kind='h5ad').json()['resource_id']
    original = uploads.resources.load(resource).to_dict()
    pending = submission(input_set_id=None, utterance='Please clarify the task.', attachment_ids=[resource])
    assert client.post('/api/v1/sessions/session/turns', json=pending).status_code == 202
    assert wait_turn(client)['attachments'][0]['use'] == 'staged'
    execute = submission(turn='inspect', input_set_id=None, attachment_ids=[resource], resource_id=resource)
    assert client.post('/api/v1/sessions/session/turns', json=execute).status_code == 202
    turn = wait_turn(client, 'inspect')
    assert turn['status'] == 'succeeded' and turn['attachments'][0]['use'] == 'consumed'
    assert uploads.resources.load(resource).to_dict() == original
    assert backend.science_calls == ['inspect_scATAC']


def test_equal_client_upload_tokens_do_not_collide_across_sessions(chat):
    client, _, uploads = chat
    first = upload(client, 'notes.tsv', b'first').json()['resource_id']
    second = upload(client, 'notes.tsv', b'second', session='other').json()['resource_id']
    assert first != second
    assert uploads.resources.load(first).session_id == 'session'
    assert uploads.resources.load(second).session_id == 'other'


def test_catalog_integrity_and_duplicate_references_fail_closed(chat):
    import json
    client, backend, uploads = chat
    rid = upload(client, 'notes.tsv', b'notes').json()['resource_id']
    assert client.post('/api/v1/sessions/session/turns', json=submission(
        input_set_id=None, attachment_ids=[rid, rid])).status_code == 422
    path = uploads.resources._path(rid)
    record = json.loads(path.read_text())
    record['record']['session_id'] = 'other'
    path.write_text(json.dumps(record))
    assert client.post('/api/v1/sessions/other/turns', json=submission(
        input_set_id=None, attachment_ids=[rid])).status_code == 400
    assert not backend.models and not backend.science_calls
