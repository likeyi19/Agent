"""USI-2 candidate attachments reuse role continuation and genuine inspection."""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agent.application import InteractiveAgentApplication
from agent.providers import PlanningModelFactoryRegistry
from agent.application.uploads import H5ADUploadAdmission
from agent.web.app import create_app
from helpers import ScriptedModel, harness, submission, wait_turn
from test_chat_attachments import upload


OBJECTIVE = 'Inspect the dataset among these attachments.'
ROLE = dict(tool='inspect_scATAC', port='dataset', source='source.0')


class RoleModel(ScriptedModel):
    def complete(self, *, prompt, response_schema):
        value = json.loads(prompt)
        if 'turn_schema_version' in value:
            self.calls.append(value)
            if value['utterance'] == OBJECTIVE:
                decision = dict(kind='request_roles', target='inspect_scATAC',
                    required_roles=[dict(tool='inspect_scATAC', port='dataset')], roles=[])
            else:
                decision = dict(kind='answer_roles', pending='@roles',
                    roles=[ROLE | dict(source='source.1' if value['utterance'] == 'Use the second file.' else 'source.0')])
            return json.dumps(dict(turn_schema_version=1, decision=decision))
        return super().complete(prompt=prompt, response_schema=response_schema)


def role_web(tmp_path, *, model_type=RoleModel):
    backend = harness(tmp_path)
    profiles = tuple(backend.service._profiles.values())
    def factory(profile):
        model = model_type(profile)
        backend.models.append(model)
        return model
    backend.service = InteractiveAgentApplication(tmp_path / 'workspace',
        model_profiles=profiles, default_profile_id='alpha',
        approved_source_roots=(tmp_path,), registry=backend.service._application.registry,
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': factory}))
    root = backend.service._application._workspace._ensure_directory(tmp_path / 'workspace' / 'uploads')
    uploads = H5ADUploadAdmission(backend.service.resources, root)
    return backend, uploads


def send(client, *, turn='first', text=OBJECTIVE, ids=(), generation=0):
    body = submission(turn=turn, utterance=text, input_set_id=None,
                      attachment_ids=list(ids), generation=generation)
    response = client.post('/api/v1/sessions/session/turns', json=body)
    assert response.status_code == 202, response.text
    return wait_turn(client, turn), body


@pytest.mark.parametrize('initial_kind', ['request_roles', 'execute_plan'])
def test_missing_file_preserves_objective_without_run_then_upload_continues(tmp_path, initial_kind):
    class MissingFileModel(RoleModel):
        def complete(self, *, prompt, response_schema):
            value = json.loads(prompt)
            if (initial_kind == 'execute_plan' and 'turn_schema_version' in value
                    and value['utterance'] == OBJECTIVE):
                self.calls.append(value)
                return json.dumps(dict(turn_schema_version=1, decision=dict(
                    kind='execute_plan', target='inspect_scATAC', argument=None)))
            return super().complete(prompt=prompt, response_schema=response_schema)
    backend, uploads = role_web(tmp_path, model_type=MissingFileModel)
    with TestClient(create_app(backend.service, uploads=uploads)) as client:
        client.post('/api/v1/sessions', json=dict(session_id='session'))
        pending, original = send(client)
        assert pending['status'] == 'clarification', pending
        assert 'dataset to inspect' in pending['response']['text']
        state = backend.service._application.sessions.load('session')
        assert not state.turns and not state.revisions and not backend.science_calls
        assert not list((tmp_path / 'workspace' / 'runs').glob('**/*.json'))
        assert not list((tmp_path / 'workspace' / 'run_state').glob('**/*.json'))
        assert state.interactions[0].prerequisite['field'] == 'roles'
        prompt = backend.models[0].calls[0]
        assert prompt['dialogue']['selected_sources'] == []
        assert len(backend.models[0].calls) == 1
        matrix = upload(client, 'matrix.h5ad', backend.source.read_bytes(), kind='h5ad').json()['resource_id']
        accepted, body = send(client, turn='answer', text='Here is the file.', ids=(matrix,))
        assert accepted['status'] == 'succeeded', accepted
        assert backend.science_calls == ['inspect_scATAC']
        run = backend.service._application.run_store.load(accepted['run_id'])
        assert run.request.prompt == OBJECTIVE
        assert run.request.inputs['input_path'] == uploads.resources.load(matrix).source_path
        assert backend.models[-1].calls[0]['dialogue']['pending_prerequisite']['objective'] == OBJECTIVE
        calls = sum(len(m.calls) for m in backend.models)
        assert client.post('/api/v1/sessions/session/turns', json=body).status_code == 202
        assert client.post('/api/v1/sessions/session/turns', json=original).status_code == 202
        assert client.get('/api/v1/sessions/session').status_code == 200
        assert backend.science_calls == ['inspect_scATAC']
        assert sum(len(m.calls) for m in backend.models) == calls


@pytest.mark.parametrize('field,target', [
    ('fastq', 'prepare_scATAC_fragments'), ('bam', 'prepare_scATAC_bam_fragments'),
    ('fragments', 'import_scATAC_fragments'), ('annotation', 'annotate_scATAC_cell_types'),
    ('epizoo', 'epizoo_embed_cells'),
])
def test_empty_source_set_uses_existing_typed_resource_prerequisites(tmp_path, field, target):
    from dataclasses import asdict, replace
    from agent.application.raw_handoff import empty
    def valid_enums(value):
        if isinstance(value, dict):
            if 'enum' in value:
                assert value['enum'], 'Empty enums cannot be sent to a structured provider'
            for child in value.values():
                valid_enums(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                valid_enums(child)
    class EmptySourceModel(RoleModel):
        def complete(self, *, prompt, response_schema):
            value = json.loads(prompt)
            assert 'turn_schema_version' in value, 'Missing files entered planning'
            valid_enums(response_schema)
            declared = empty(field)
            if field == 'epizoo':
                declared = replace(declared, operation='inference')
            self.calls.append(value)
            return json.dumps(dict(turn_schema_version=1, decision=dict(
                kind='request_roles', target=target, required_roles=[], roles=[],
                **{field: asdict(declared)}, arguments=[])))
    backend, uploads = role_web(tmp_path, model_type=EmptySourceModel)
    with TestClient(create_app(backend.service, uploads=uploads)) as client:
        client.post('/api/v1/sessions', json=dict(session_id='session'))
        pending, _ = send(client)
        assert pending['status'] == 'clarification', pending
        state = backend.service._application.sessions.load('session')
        assert state.interactions[0].prerequisite['field'] == field
        assert not state.turns and not state.revisions and not backend.science_calls
        assert len(backend.models[0].calls) == 1
        assert not list((tmp_path / 'workspace' / 'runs').glob('**/*.json'))
        assert not list((tmp_path / 'workspace' / 'run_state').glob('**/*.json'))


@pytest.mark.parametrize('field,target,declarations', [
    ('fastq', 'prepare_scATAC_fragments', (
        ('select_scATAC_cells', 'min_tss_enrichment', '4', 4),
        ('select_scATAC_cells', 'min_qc_fragment_records', '100', 100))),
    ('bam', 'prepare_scATAC_bam_fragments', (
        ('select_scATAC_cells', 'min_tss_enrichment', '4', 4),
        ('select_scATAC_cells', 'min_qc_fragment_records', '100', 100))),
    ('fragments', 'import_scATAC_fragments', (
        ('select_scATAC_cells', 'min_tss_enrichment', '4', 4),
        ('select_scATAC_cells', 'min_qc_fragment_records', '100', 100))),
    ('annotation', 'annotate_scATAC_cell_types', (
        ('annotate_scATAC_cell_types', 'count_slot', 'X', 'X'),
        ('annotate_scATAC_cell_types', 'group_column', 'clusters', 'clusters'))),
    ('epizoo', 'epizoo_embed_cells', (
        ('cluster_cells', 'resolution', '0.5', .5),
        ('compute_cell_umap', 'min_dist', '0.1', .1))),
])
def test_raw_declaration_arguments_continue_without_science(tmp_path, field, target, declarations):
    """Admission retains typed intent and candidates; none are qualified inputs."""
    from dataclasses import asdict, replace
    from agent.application.raw_handoff import empty
    from agent.schemas.orchestration import _serialize

    class DeclarationModel(RoleModel):
        def complete(self, *, prompt, response_schema):
            value = json.loads(prompt)
            assert 'turn_schema_version' in value, 'Incomplete declarations entered planning'
            self.calls.append(value)
            continuation = bool(value['dialogue'].get('pending_prerequisite'))
            tool, name, literal, _ = declarations[int(continuation)]
            start = value['utterance'].index(literal)
            declared = empty(field)
            if field == 'epizoo' and not continuation:
                declared = replace(declared, operation='inference')
            decision = (dict(kind='answer_roles', pending='@roles', roles=[]) if continuation
                else dict(kind='request_roles', target=target, required_roles=[], roles=[]))
            decision.update({field: asdict(declared)}, arguments=[dict(tool=tool,
                argument=name, literal=literal, start=start, end=start + len(literal))])
            return json.dumps(dict(turn_schema_version=1, decision=decision))

    backend, uploads = role_web(tmp_path, model_type=DeclarationModel)
    with TestClient(create_app(backend.service, uploads=uploads)) as client:
        client.post('/api/v1/sessions', json=dict(session_id='session'))
        attachment = upload(client, 'unqualified.attachment', b'simulated candidate bytes').json()['resource_id']
        record = uploads.resources.load(attachment)
        pending, _ = send(client, text='Use ' + declarations[0][2] + '.', ids=(attachment,))
        assert pending['status'] == 'clarification', pending
        original = backend.service._application.sessions.load('session').interactions[0]
        assert original.admitted[field + '_arguments'] == {declarations[0][1]: declarations[0][3]}
        continued, _ = send(client, turn='answer', text='Also use ' + declarations[1][2] + '.')
        assert continued['status'] == 'clarification', continued
        state = backend.service._application.sessions.load('session')
        assert state.interactions[0] == original
        assert state.interactions[-1].prerequisite['field'] == field
        admitted = state.interactions[-1].admitted
        assert admitted[field + '_arguments'] == {name: expected for _, name, _, expected in declarations}
        assert admitted['arguments'][0]['argument'] == declarations[1][1]
        assert _serialize(admitted[field + '_sources']) == _serialize(original.admitted[field + '_sources'])
        previous = state.interactions[-1]
        repeated, _ = send(client, turn='reconfirm', text='Keep ' + declarations[1][2] + '.')
        assert repeated['status'] == 'clarification', repeated
        state = backend.service._application.sessions.load('session')
        assert state.interactions[:2] == (original, previous)
        assert state.interactions[-1].admitted[field + '_arguments'] == admitted[field + '_arguments']
        assert state.interactions[-1].admitted['continuation']['binding_turn_id'] == 'answer'
        assert uploads.resources.load(attachment) == record
        assert not state.turns and not state.revisions and not backend.science_calls
        assert sum(len(model.calls) for model in backend.models) == 3
        assert not list((tmp_path / 'workspace' / 'runs').glob('**/*.json'))
        assert not list((tmp_path / 'workspace' / 'run_state').glob('**/*.json'))


def test_candidate_roles_continue_original_objective_and_only_accept_bound_source(tmp_path):
    backend, uploads = role_web(tmp_path)
    with TestClient(create_app(backend.service, uploads=uploads)) as client:
        client.post('/api/v1/sessions', json=dict(session_id='session'))
        matrix = upload(client, 'matrix.h5ad', backend.source.read_bytes(), kind='h5ad').json()['resource_id']
        notes = upload(client, 'notes.tsv', b'not scientific proof', token='notes').json()['resource_id']
        before = {rid: uploads.resources.load(rid).to_dict() for rid in (matrix, notes)}
        pending, original_body = send(client, ids=(matrix, notes))
        assert pending['status'] == 'clarification' and pending['revision_id'] is None, pending
        assert 'dataset to inspect' in pending['response']['text']
        assert not backend.science_calls
        prompt = backend.models[0].calls[0]
        assert prompt['dialogue']['attachment_candidates'] is True
        assert [m['source'] for m in prompt['dialogue']['selected_sources']] == ['source.0', 'source.1']
        encoded = json.dumps(prompt)
        for record in before.values():
            for key in ('resource_id', 'source_path', 'source_sha256'):
                assert record[key] not in encoded
        accepted, body = send(client, turn='answer', text='Use the first file.')
        assert accepted['status'] == 'succeeded', accepted
        assert [a['use'] for a in accepted['attachments']] == ['consumed', 'staged']
        assert backend.science_calls == ['inspect_scATAC']
        run = backend.service._application.run_store.load(accepted['run_id'])
        assert run.request.prompt == OBJECTIVE
        assert run.request.inputs['input_path'] == before[matrix]['source_path']
        assert notes not in json.dumps(run.request.to_dict())
        assert before == {rid: uploads.resources.load(rid).to_dict() for rid in before}
        assert client.post('/api/v1/sessions/session/turns', json=body).status_code == 202
        assert backend.science_calls == ['inspect_scATAC']
        assert client.post('/api/v1/sessions/session/turns', json={**original_body, 'attachment_ids': [notes, matrix]}).status_code == 409
        revision = client.get(f"/api/v1/sessions/session/revisions/{accepted['revision_id']}").json()
        assert revision
    # Reopen/retry without provider construction or access to historical source bytes.
    def forbidden(_):
        raise AssertionError('Historical read constructed a provider')
    restored = InteractiveAgentApplication(tmp_path / 'workspace',
        model_profiles=tuple(backend.service._profiles.values()), default_profile_id='alpha',
        approved_source_roots=(tmp_path,),
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': forbidden}))
    restored_uploads = H5ADUploadAdmission(restored.resources, uploads.root)
    with TestClient(create_app(restored, uploads=restored_uploads)) as client:
        assert client.get('/api/v1/sessions/session/turns/answer').json() == accepted
        for record in before.values():
            Path(record['source_path']).unlink()
        assert client.post('/api/v1/sessions/session/turns', json=body).status_code == 202
        assert client.get('/api/v1/sessions/session/turns/answer').json() == accepted


def test_followup_upload_appends_without_replacing_captured_attachment(tmp_path):
    backend, uploads = role_web(tmp_path)
    with TestClient(create_app(backend.service, uploads=uploads)) as client:
        client.post('/api/v1/sessions', json=dict(session_id='session'))
        notes = upload(client, 'notes.txt', b'neutral', token='notes').json()['resource_id']
        pending, _ = send(client, ids=(notes,))
        assert pending['status'] == 'clarification'
        matrix = upload(client, 'matrix.h5ad', backend.source.read_bytes(), kind='h5ad').json()['resource_id']
        accepted, _ = send(client, turn='answer', text='Use the second file.', ids=(matrix,))
        assert accepted['status'] == 'succeeded', accepted
        assert [a['resource_id'] for a in accepted['attachments']] == [notes, matrix]
        assert [a['use'] for a in accepted['attachments']] == ['staged', 'consumed']
        prompt = backend.models[-1].calls[0]
        assert prompt['dialogue']['pending_prerequisite']['objective'] == OBJECTIVE
        assert [m['source'] for m in prompt['dialogue']['selected_sources']] == ['source.0', 'source.1']


def test_llm_role_does_not_qualify_neutral_bytes(tmp_path):
    backend, uploads = role_web(tmp_path)
    with TestClient(create_app(backend.service, uploads=uploads)) as client:
        client.post('/api/v1/sessions', json=dict(session_id='session'))
        neutral = upload(client, 'misleading.h5ad', b'invalid matrix').json()['resource_id']
        send(client, ids=(neutral,))
        result, _ = send(client, turn='answer', text='Use the first file.')
        assert result['status'] == 'failed' and result['revision_id'] is None, result
        assert result['attachments'][0]['use'] != 'consumed'
        assert uploads.resources.load(neutral).input_type == 'attachment'
        assert not backend.service._application.sessions.load('session').revisions


@pytest.mark.parametrize('variant', ['foreign', 'unknown', 'stale', 'changed_bytes', 'reordered'])
def test_pending_attachment_references_fail_closed(tmp_path, variant):
    backend, uploads = role_web(tmp_path)
    with TestClient(create_app(backend.service, uploads=uploads)) as client:
        for session in ('session', 'other'):
            client.post('/api/v1/sessions', json=dict(session_id=session))
        first = upload(client, 'first.h5ad', backend.source.read_bytes(), kind='h5ad').json()['resource_id']
        second = upload(client, 'second.h5ad', backend.source.read_bytes(), kind='h5ad', token='second').json()['resource_id']
        send(client, ids=(first, second))
        ids = []
        if variant == 'foreign':
            ids = [upload(client, 'foreign.h5ad', backend.source.read_bytes(), kind='h5ad', session='other').json()['resource_id']]
        elif variant == 'unknown':
            ids = ['local-' + 'f' * 64]
        elif variant == 'changed_bytes':
            Path(uploads.resources.load(first).source_path).write_bytes(b'changed')
        elif variant == 'reordered':
            ids = [second, first]
        body = submission(turn='answer', utterance='Use the first file.', input_set_id=None,
            attachment_ids=ids, generation=1 if variant == 'stale' else 0)
        response = client.post('/api/v1/sessions/session/turns', json=body)
        if variant in ('foreign', 'unknown', 'stale'):
            assert response.status_code in (404, 409), response.text
            assert len(backend.models) == 1
        else:
            assert response.status_code == 202, response.text
            result = wait_turn(client, 'answer')
            assert result['revision_id'] is None and result['status'] != 'succeeded', result
        assert not backend.science_calls
        state = backend.service._application.sessions.load('session')
        assert not state.revisions and state.interactions[0].prerequisite['field'] == 'roles'


def test_staged_unused_bytes_do_not_gate_independent_scientific_consumption(tmp_path):
    backend, uploads = role_web(tmp_path)
    with TestClient(create_app(backend.service, uploads=uploads)) as client:
        client.post('/api/v1/sessions', json=dict(session_id='session'))
        matrix = upload(client, 'matrix.h5ad', backend.source.read_bytes(), kind='h5ad').json()['resource_id']
        notes = upload(client, 'notes.txt', b'notes', token='notes').json()['resource_id']
        send(client, ids=(matrix, notes))
        Path(uploads.resources.load(notes).source_path).unlink()
        accepted, _ = send(client, turn='answer', text='Use the first file.')
        assert accepted['status'] == 'succeeded', accepted
        assert [item['use'] for item in accepted['attachments']] == ['consumed', 'staged']


def test_explicit_candidate_group_cannot_bypass_session_scope(tmp_path):
    from agent.application import InteractiveBoundaryError
    backend, uploads = role_web(tmp_path)
    with TestClient(create_app(backend.service, uploads=uploads)) as client:
        for session in ('session', 'other'):
            client.post('/api/v1/sessions', json=dict(session_id=session))
        rid = upload(client, 'notes.txt', b'notes').json()['resource_id']
        group = uploads.resources.compose_group([rid], attachment_candidates=True)
        with pytest.raises(InteractiveBoundaryError):
            backend.service.validate_submission('other', 'bad', OBJECTIVE,
                expected_generation=0, registered_input=group)
        assert not backend.models and not backend.science_calls


def test_pending_role_survives_new_application_before_answer(tmp_path):
    backend, uploads = role_web(tmp_path)
    with TestClient(create_app(backend.service, uploads=uploads)) as client:
        client.post('/api/v1/sessions', json=dict(session_id='session'))
        rid = upload(client, 'matrix.h5ad', backend.source.read_bytes(), kind='h5ad').json()['resource_id']
        pending, _ = send(client, ids=(rid,))
        assert pending['status'] == 'clarification'
    service = InteractiveAgentApplication(tmp_path / 'workspace',
        model_profiles=tuple(backend.service._profiles.values()), default_profile_id='alpha',
        registry=backend.service._application.registry, approved_source_roots=(tmp_path,),
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': RoleModel}))
    restored_uploads = H5ADUploadAdmission(service.resources, uploads.root)
    with TestClient(create_app(service, uploads=restored_uploads)) as client:
        assert client.get('/api/v1/sessions/session/turns/first').json() == pending
        accepted, _ = send(client, turn='answer', text='Use the first file.')
        assert accepted['status'] == 'succeeded', accepted
        assert service._application.run_store.load(accepted['run_id']).request.prompt == OBJECTIVE
        assert backend.science_calls == ['inspect_scATAC']
