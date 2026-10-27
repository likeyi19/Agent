"""Previous uploads remain candidates; language selection alone assigns roles."""
import json
from pathlib import Path

import pytest

from helpers import ScriptedModel, submission, wait_turn
from test_chat_attachments import chat, upload


def send(client, turn, text, generation, **extra):
    body = submission(turn, generation, text, input_set_id=None, **extra)
    response = client.post('/api/v1/sessions/session/turns', json=body)
    assert response.status_code == 202, response.text
    return wait_turn(client, turn), body


@pytest.fixture
def reuse_model(monkeypatch):
    original = ScriptedModel.complete
    seen = []

    def complete(self, *, prompt, response_schema):
        p = json.loads(prompt)
        if 'turn_schema_version' in p:
            seen.append(p)
            if p['utterance'].startswith('Reuse'):
                sources = p['dialogue']['selected_sources']
                assert p['dialogue']['session_source_candidates'] is True
                assert all(set(s) == {'source', 'position', 'label', 'input_type'} for s in sources)
                selected = sources[-1 if 'last' in p['utterance'] else 1 if 'second' in p['utterance'] else 0]['source']
                return json.dumps(dict(turn_schema_version=1, decision=dict(kind='execute_plan',
                    target='inspect_scATAC', roles=[dict(tool='inspect_scATAC', port='dataset', source=selected)])))
        return original(self, prompt=prompt, response_schema=response_schema)

    monkeypatch.setattr(ScriptedModel, 'complete', complete)
    return seen


def test_inspection_then_explicit_reuse_without_reupload_and_exact_retry(chat, reuse_model):
    client, backend, uploads = chat
    rid = upload(client, 'original.h5ad', backend.source.read_bytes(), kind='h5ad').json()['resource_id']
    first, _ = send(client, 'first', 'Inspect this file.', 0, resource_id=rid, attachment_ids=[rid])
    assert first['status'] == 'succeeded'
    second, body = send(client, 'second', 'Reuse the file and inspect it again.', 1)
    assert second['status'] == 'succeeded', second
    state = backend.service._application.sessions.load('session')
    current = state.interactions[-1]
    assert current.submission['execution_inputs'] == {}
    assert 'registered_input' not in current.submission and 'attachments' not in current.submission
    assert current.admitted['registered_input']['members'][0]['resource_id'] == rid
    assert current.admitted['roles'][0]['source'] == 'source.0'
    run = backend.service._application.run_store.load(second['run_id'])
    assert run.steps[0].resolved_arguments['path'] == uploads.resources.load(rid).source_path
    assert backend.science_calls == ['inspect_scATAC', 'inspect_scATAC']
    calls = len(backend.models)
    # Later upload availability and missing bytes cannot change completed retries.
    upload(client, 'later.txt', b'not a scientific input', token='later')
    Path(uploads.resources.load(rid).source_path).unlink()
    assert client.post('/api/v1/sessions/session/turns', json=body).status_code == 202
    assert client.get('/api/v1/sessions/session/turns/second').json() == second
    assert len(backend.models) == calls and len(backend.science_calls) == 2
    assert backend.service.reopen_session('session').active_revision_id == second['revision_id']


def test_candidates_do_not_bind_answers_or_unselected_execution(chat, reuse_model):
    client, backend, _ = chat
    rid = upload(client, 'original.h5ad', backend.source.read_bytes(), kind='h5ad').json()['resource_id']
    send(client, 'first', 'Inspect this file.', 0, resource_id=rid, attachment_ids=[rid])
    answer, _ = send(client, 'answer', 'How many cells?', 1)
    assert answer['status'] == 'answered'
    assert backend.science_calls == ['inspect_scATAC']
    result, _ = send(client, 'no-role', 'Inspect again.', 1)
    assert result['status'] != 'succeeded'
    state = backend.service._application.sessions.load('session')
    assert state.interactions[-1].admitted['inputs'] == {}
    assert 'registered_input' not in state.interactions[-1].admitted
    assert backend.science_calls == ['inspect_scATAC']


def test_multiple_sources_require_exact_model_selection_and_explicit_inputs_win(chat, reuse_model):
    client, backend, uploads = chat
    ids = [upload(client, name, backend.source.read_bytes(), token=str(i), kind='h5ad').json()['resource_id']
           for i, name in enumerate(('first.h5ad', 'second.h5ad'))]
    for i, rid in enumerate(ids):
        result, _ = send(client, f'inspect-{i}', 'Inspect this file.', i,
                         resource_id=rid, attachment_ids=[rid])
        assert result['status'] == 'succeeded'
    # Current input remains primary while earlier uploads are offered as candidates.
    assert reuse_model[-1]['dialogue']['selected_sources'][0]['label'] == 'second.h5ad'
    result, _ = send(client, 'reuse', 'Reuse the second file and inspect it.', 2)
    assert result['status'] == 'succeeded', result
    state = backend.service._application.sessions.load('session')
    assert [m['resource_id'] for m in state.interactions[-1].submission['available_input']['members']] == ids
    assert state.interactions[-1].admitted['roles'][0]['source'] == 'source.1'
    run = backend.service._application.run_store.load(result['run_id'])
    assert run.steps[0].resolved_arguments['path'] == uploads.resources.load(ids[1]).source_path
    # Another Session receives none of these private candidates.
    profile, inputs, offered, state, duplicate = backend.service._checked_submission(
        'other', 'check', 'Inspect my file.', expected_generation=0)
    assert 'available_input' not in offered


def test_selected_prior_source_mutation_fails_before_science(chat, reuse_model):
    client, backend, uploads = chat
    rid = upload(client, 'original.h5ad', backend.source.read_bytes(), kind='h5ad').json()['resource_id']
    send(client, 'first', 'Inspect this file.', 0, resource_id=rid, attachment_ids=[rid])
    Path(uploads.resources.load(rid).source_path).write_bytes(b'changed')
    result, _ = send(client, 'reuse', 'Reuse the file.', 1)
    assert result['status'] == 'failed', result
    assert backend.science_calls == ['inspect_scATAC']


def test_candidate_bound_never_silently_truncates_history(chat, reuse_model):
    client, backend, _ = chat
    rid = upload(client, 'matrix.h5ad', backend.source.read_bytes(), kind='h5ad').json()['resource_id']
    notes = [upload(client, f'note-{i}.txt', b'note', token=f'note-{i}').json()['resource_id']
             for i in range(16)]
    send(client, 'first', 'Inspect this file.', 0, resource_id=rid, attachment_ids=[rid, *notes[:15]])
    send(client, 'second', 'Inspect this file.', 1, resource_id=rid, attachment_ids=[notes[15]])
    _, _, offered, _, _ = backend.service._checked_submission(
        'session', 'next', 'Which file?', expected_generation=2)
    assert offered['available_input_overflow'] is True and 'available_input' not in offered
    assert len(backend.science_calls) == 2


def test_missing_unselected_source_does_not_block_evidence_answer(chat, reuse_model):
    client, backend, uploads = chat
    rid = upload(client, 'matrix.h5ad', backend.source.read_bytes(), kind='h5ad').json()['resource_id']
    send(client, 'first', 'Inspect this file.', 0, resource_id=rid, attachment_ids=[rid])
    Path(uploads.resources.load(rid).source_path).unlink()
    result, _ = send(client, 'answer', 'How many cells?', 1)
    assert result['status'] == 'answered'
    assert backend.science_calls == ['inspect_scATAC']


def test_all_sixteen_offered_candidate_positions_are_selectable(chat, reuse_model):
    client, backend, uploads = chat
    notes = [upload(client, f'note-{i}.txt', b'note', token=f'note-{i}').json()['resource_id']
             for i in range(15)]
    rid = upload(client, 'matrix.h5ad', backend.source.read_bytes(), kind='h5ad').json()['resource_id']
    send(client, 'first', 'Inspect this file.', 0, resource_id=rid, attachment_ids=[*notes, rid])
    result, _ = send(client, 'reuse', 'Reuse the last file and inspect it.', 1)
    assert result['status'] == 'succeeded', result
    current = backend.service._application.sessions.load('session').interactions[-1]
    assert current.admitted['roles'][0]['source'] == 'source.15'
    run = backend.service._application.run_store.load(result['run_id'])
    assert run.steps[0].resolved_arguments['path'] == uploads.resources.load(rid).source_path


def test_new_attachment_keeps_historical_source_and_retry_offer(chat, reuse_model):
    client, backend, uploads = chat
    rid = upload(client, 'matrix.h5ad', backend.source.read_bytes(), kind='h5ad').json()['resource_id']
    send(client, 'first', 'Inspect this file.', 0, resource_id=rid, attachment_ids=[rid])
    note = upload(client, 'reference.tsv', b'opaque companion', token='reference').json()['resource_id']
    result, body = send(client, 'reuse', 'Reuse the second file and inspect it.', 1, attachment_ids=[note])
    assert result['status'] == 'succeeded', result
    prompt = reuse_model[-1]['dialogue']
    assert prompt['current_turn_sources'] == ['source.0']
    assert [s['label'] for s in prompt['selected_sources']] == ['reference.tsv', 'matrix.h5ad']
    state = backend.service._application.sessions.load('session')
    assert [m['resource_id'] for m in state.interactions[-1].submission['registered_input']['members']] == [note]
    before = len(backend.models)
    other = upload(client, 'later.h5ad', backend.source.read_bytes(), token='later', kind='h5ad').json()['resource_id']
    send(client, 'later', 'Inspect this file.', 2, resource_id=other, attachment_ids=[other])
    Path(uploads.resources.load(rid).source_path).unlink()
    assert client.post('/api/v1/sessions/session/turns', json=body).status_code == 202
    assert client.get('/api/v1/sessions/session/turns/reuse').json() == result
    assert len(backend.models) == before + 1


def test_overflow_preserves_current_files_without_selecting_historical_subset(chat):
    client, backend, _ = chat
    rid = upload(client, 'matrix.h5ad', backend.source.read_bytes(), kind='h5ad').json()['resource_id']
    notes = [upload(client, f'note-{i}.txt', b'note', token=f'note-{i}').json()['resource_id']
             for i in range(16)]
    send(client, 'first', 'Inspect.', 0, resource_id=rid, attachment_ids=[rid, *notes[:15]])
    send(client, 'second', 'Inspect.', 1, resource_id=rid, attachment_ids=[notes[15]])
    _, _, offered, _, _ = backend.service._checked_submission(
        'session', 'next', 'Use this attachment.', expected_generation=2, attachment_ids=[notes[0]])
    assert offered['available_input_overflow'] is True and 'available_input' not in offered
    assert [m['resource_id'] for m in offered['registered_input']['members']] == [notes[0]]


def test_mixed_candidate_clarification_keeps_captured_handles(chat, monkeypatch):
    client, backend, uploads = chat
    original = ScriptedModel.complete
    prompts = []
    def complete(self, *, prompt, response_schema):
        value = json.loads(prompt)
        if 'turn_schema_version' in value and value['utterance'] in ('Clarify my source.', 'Use the earlier matrix.'):
            prompts.append(value)
            decision = (dict(kind='request_roles', target='inspect_scATAC',
                required_roles=[dict(tool='inspect_scATAC', port='dataset')], roles=[])
                if value['utterance'] == 'Clarify my source.' else dict(kind='answer_roles',
                pending='@roles', roles=[dict(tool='inspect_scATAC', port='dataset', source='source.1')]))
            return json.dumps(dict(turn_schema_version=1, decision=decision))
        return original(self, prompt=prompt, response_schema=response_schema)
    monkeypatch.setattr(ScriptedModel, 'complete', complete)
    rid = upload(client, 'matrix.h5ad', backend.source.read_bytes(), kind='h5ad').json()['resource_id']
    send(client, 'first', 'Inspect.', 0, resource_id=rid, attachment_ids=[rid])
    note = upload(client, 'reference.tsv', b'companion', token='reference').json()['resource_id']
    pending, _ = send(client, 'pending', 'Clarify my source.', 1, attachment_ids=[note])
    assert pending['status'] == 'clarification'
    later = upload(client, 'later.txt', b'later companion', token='later').json()['resource_id']
    result, body = send(client, 'answer', 'Use the earlier matrix.', 1, attachment_ids=[later])
    assert result['status'] == 'succeeded', result
    assert [s['label'] for s in prompts[-1]['dialogue']['selected_sources']] == ['reference.tsv', 'matrix.h5ad', 'later.txt']
    step = backend.service._application.run_store.load(result['run_id']).steps[0]
    assert step.resolved_arguments['path'] == uploads.resources.load(rid).source_path
    assert client.post('/api/v1/sessions/session/turns', json=body).status_code == 202
    assert backend.science_calls == ['inspect_scATAC', 'inspect_scATAC']
