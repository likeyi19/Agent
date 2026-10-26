"""Real HTTP uploads retain the selected primary in canonical raw candidates.

Only semantic model responses are scripted. Registration, typed admission,
source observations, prerequisite persistence and HTTP workers are real. Tiny
synthetic H5ADs establish software scope, never biological qualification.
"""
from dataclasses import asdict
import json

import anndata as ad
from fastapi.testclient import TestClient
import pytest

from agent.application.epizoo_handoff import EpiZooDeclaration
from agent.application.raw_handoff import AnnotationDeclaration
from agent.web.app import create_app
from test_uploaded_multiturn import body, submit, uploaded_backend


def model_for(field, decisions):
    class Model:
        def __init__(self, profile):
            self.model_id = profile.model_id
            self.calls = []

        def complete(self, *, prompt, response_schema):
            value = json.loads(prompt)
            self.calls.append(value)
            assert 'turn_schema_version' in value, 'Raw prerequisites must not reach planning or science.'
            source = decisions[value['utterance']]
            if source == 'generic_pending':
                return json.dumps(dict(turn_schema_version=1, decision=dict(kind='request_roles',
                    target='pca_embed_cells', required_roles=[dict(tool='pca_embed_cells', port='dataset')],
                    roles=[], arguments=[])))
            if source == 'answer_same':
                declared = EpiZooDeclaration() if field == 'epizoo' else AnnotationDeclaration()
                return json.dumps(dict(turn_schema_version=1, decision=dict(kind='answer_roles',
                    pending='@roles', roles=[], arguments=[], **{field: asdict(declared)})))
            declared = (EpiZooDeclaration(operation='inference', source=source) if field == 'epizoo'
                        else AnnotationDeclaration(source=source))
            target = 'epizoo_embed_cells' if field == 'epizoo' else 'annotate_scATAC_cell_types'
            return json.dumps(dict(turn_schema_version=1, decision=dict(kind='request_roles',
                target=target, required_roles=[], roles=[], arguments=[], **{field: asdict(declared)})))

    return Model


def source_observations(monkeypatch, field):
    if field == 'epizoo':
        from agent.tools.models import epizoo_inputs as owner
        name = 'observe_source'
    else:
        from agent.tools.analysis import annotation_input as owner
        name = 'observe'
    original, calls = getattr(owner, name), []

    def observed(path, sha):
        calls.append((str(path), sha))
        return original(path, sha)

    monkeypatch.setattr(owner, name, observed)
    return calls


def sources(client, backend, tmp_path):
    def upload(payload, upload_id, filename):
        response = client.put('/api/v1/sessions/session/uploads/' + upload_id,
            params={'filename': filename}, content=payload,
            headers={'Content-Type': 'application/octet-stream'})
        assert response.status_code == 201, response.text
        return response.json()

    first = upload(backend.source.read_bytes(), 'source-a', 'Selected A.h5ad')
    second_path = tmp_path / 'distinct-b.h5ad'
    second = ad.read_h5ad(backend.source)
    second.obs_names = ['other-cell-a', 'other-cell-b']
    second.write_h5ad(second_path)
    other = upload(second_path.read_bytes(), 'source-b', 'Companion B.h5ad')
    return [backend.service.resources.load(r['resource_id']) for r in (first, other)]


def no_science(backend):
    state = backend.service._application.sessions.load('session')
    assert not backend.science_calls and not state.turns and not state.revisions
    assert state.generation == 0 and state.active_revision_id is None
    assert not list(backend.service._application.run_store.root.glob('*.json'))
    assert all(len(model.calls) == 1 for model in backend.models)
    return state


@pytest.mark.parametrize('field', ['annotation', 'epizoo'])
@pytest.mark.parametrize('pending_kind', ['none', 'generic', 'raw'])
def test_ordinary_single_upload_cannot_redirect_raw_request_to_companion(
        tmp_path, monkeypatch, field, pending_kind):
    # Captured candidate order B,A never makes companion B the selected A.
    # A fresh RequestRoles remains distinct from continuing the old declaration.
    wrong_source = 'source.1' if pending_kind == 'none' else 'source.0'
    backend, uploads, _ = uploaded_backend(tmp_path,
        model_type=model_for(field, {'Start with B.': ('generic_pending' if pending_kind == 'generic' else 'source.0'),
                                    'Use the selected A.': wrong_source}))
    observations = source_observations(monkeypatch, field)
    with TestClient(create_app(backend.service, uploads=uploads)) as client:
        assert client.post('/api/v1/sessions', json={'session_id': 'session'}).status_code == 201
        selected, companion = sources(client, backend, tmp_path)
        previous = None
        if pending_kind != 'none':
            initial = submit(client, body(companion.resource_id, utterance='Start with B.'))
            assert initial['status'] == 'clarification', initial
            previous = no_science(backend).interactions[-1]
            assert previous.prerequisite['field'] == ('roles' if pending_kind == 'generic' else field)
        before = list(observations)
        result = submit(client, body(selected.resource_id, turn='wrong-primary',
            utterance='Use the selected A.', attachment_ids=[companion.resource_id]))
        assert result['status'] == 'failed', result
        assert result['response']['error']['code'] == 'INTERACTIVE_INTENT_BINDING_INVALID'
        assert result['response']['clarification'] is None
        assert observations == before  # The companion must fail before source observation.
        state = no_science(backend)
        interaction = state.interactions[-1]
        assert interaction.admitted is None and interaction.prerequisite is None
        assert interaction.submission['registered_input']['resource_id'] == selected.resource_id
        assert interaction.submission['registered_input']['record_sha256'] == selected.record_sha256
        members = interaction.snapshot['dialogue']['source_view']['registered_input']['members']
        candidate = members[int(wrong_source.split('.')[1])]
        assert candidate['resource_id'] == companion.resource_id
        assert candidate['record_sha256'] == companion.record_sha256
        assert {m['resource_id'] for m in members} == {selected.resource_id, companion.resource_id}
        if previous is not None:
            assert state.interactions[0] == previous


@pytest.mark.parametrize('field', ['annotation', 'epizoo'])
@pytest.mark.parametrize('pending_kind', ['generic', 'raw'])
def test_new_single_upload_raw_request_uses_exact_current_primary_after_pending_source(
        tmp_path, monkeypatch, field, pending_kind):
    backend, uploads, _ = uploaded_backend(tmp_path,
        model_type=model_for(field, {'Begin with A.': ('generic_pending' if pending_kind == 'generic' else 'source.0'),
                                    'Start a new task on B.': 'source.1'}))
    observations = source_observations(monkeypatch, field)
    with TestClient(create_app(backend.service, uploads=uploads)) as client:
        assert client.post('/api/v1/sessions', json={'session_id': 'session'}).status_code == 201
        first, current = sources(client, backend, tmp_path)
        initial = submit(client, body(first.resource_id, utterance='Begin with A.'))
        assert initial['status'] == 'clarification', initial
        original = no_science(backend).interactions[-1]
        assert original.prerequisite['field'] == ('roles' if pending_kind == 'generic' else field)
        request = body(current.resource_id, turn='new-b', utterance='Start a new task on B.')
        outcome = submit(client, request)
        assert outcome['status'] == 'clarification', outcome
        assert outcome['response']['clarification']['reason'] == 'missing_' + field + '_declarations'
        state = no_science(backend)
        assert state.interactions[0] == original
        interaction = state.interactions[-1]
        assert interaction.submission['registered_input']['resource_id'] == current.resource_id
        assert interaction.submission['registered_input']['record_sha256'] == current.record_sha256
        members = interaction.snapshot['dialogue']['source_view']['registered_input']['members']
        assert [m['resource_id'] for m in members] == [first.resource_id, current.resource_id]
        assert interaction.admitted[field]['source'] == 'source.1'
        assert interaction.admitted[field + '_source_evidence']['source_sha256'] == current.source_sha256
        assert interaction.prerequisite['origin_turn_id'] == 'new-b'
        assert interaction.prerequisite['binding_turn_id'] == 'new-b'
        assert interaction.prerequisite['field'] == field
        assert 'continuation' not in interaction.admitted
        expected = [(first.source_path, first.source_sha256)] if pending_kind == 'raw' else []
        assert observations == expected + [(current.source_path, current.source_sha256)]
        counts = len(backend.models), len(observations)
        assert submit(client, request) == outcome
        assert counts == (len(backend.models), len(observations))
        assert no_science(backend) == state


@pytest.mark.parametrize('field', ['annotation', 'epizoo'])
def test_same_single_primary_raw_reply_retains_exact_pending_source(tmp_path, monkeypatch, field):
    backend, uploads, _ = uploaded_backend(tmp_path,
        model_type=model_for(field, {'Begin with A.': 'source.0', 'Continue with the same selected file.': 'answer_same'}))
    observations = source_observations(monkeypatch, field)
    with TestClient(create_app(backend.service, uploads=uploads)) as client:
        assert client.post('/api/v1/sessions', json={'session_id': 'session'}).status_code == 201
        selected, _ = sources(client, backend, tmp_path)
        initial = submit(client, body(selected.resource_id, utterance='Begin with A.'))
        assert initial['status'] == 'clarification', initial
        original = no_science(backend).interactions[-1]
        request = body(selected.resource_id, turn='reply', utterance='Continue with the same selected file.')
        result = submit(client, request)
        assert result['status'] == 'clarification', result
        state = no_science(backend)
        assert state.interactions[0] == original
        continued = state.interactions[-1]
        assert continued.snapshot['prerequisite']['origin_turn_id'] == 'first'
        assert continued.prerequisite['origin_turn_id'] == 'first'
        assert continued.prerequisite['field'] == field
        assert continued.admitted[field] == original.admitted[field]
        assert continued.admitted['continuation']['origin_turn_id'] == 'first'
        assert continued.admitted['continuation']['binding_turn_id'] == 'first'
        assert continued.admitted['registered_input'] == original.admitted['registered_input']
        assert continued.submission['registered_input']['resource_id'] == selected.resource_id
        assert continued.submission['registered_input']['record_sha256'] == selected.record_sha256
        members = continued.snapshot['dialogue']['source_view']['registered_input']['members']
        assert len(members) == 1 and members[0]['resource_id'] == selected.resource_id
        assert continued.admitted[field + '_source_evidence']['source_sha256'] == selected.source_sha256
        assert observations == [(selected.source_path, selected.source_sha256)] * 2
        counts = len(backend.models), len(observations)
        assert submit(client, request) == result
        assert counts == (len(backend.models), len(observations))
        assert no_science(backend) == state
