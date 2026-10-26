"""Shared discovery into existing native adoption, including resource continuation."""
from dataclasses import asdict
import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from agent.application.raw_handoff import FragmentsDeclaration
from agent.application.shared_resources import ScientificResourceLibrary
from test_fragments_chat_handoff import ImportOnlyModel
from test_external_fragments_execution import downstream_inputs, web_fragments
from test_chat_attachments import upload
from test_bam_chat_handoff import send


@pytest.mark.parametrize('later', [False, True])
def test_shared_qc_parent_reaches_native_adoption_without_operator_inputset(tmp_path, monkeypatch, later):
    class Model(ImportOnlyModel):
        def complete(self, *, prompt, response_schema):
            value = json.loads(prompt)
            if 'turn_schema_version' in value and value['utterance'] != self.objective:
                self.calls.append(value)
                return json.dumps(dict(turn_schema_version=1, decision=dict(kind='answer_roles',
                    pending='@roles', roles=[], fragments=asdict(FragmentsDeclaration()), arguments=[])))
            return super().complete(prompt=prompt, response_schema=response_schema)
    monkeypatch.setenv('AGENT_QC_ALLOW_SYNTHETIC', '1')
    monkeypatch.delenv('AGENT_QC_RESOURCE_CATALOG', raising=False)
    args = downstream_inputs(tmp_path)
    backend = web_fragments(tmp_path, args, companions=(), model_type=Model)
    library = ScientificResourceLibrary(tmp_path/'library')
    backend.service.resources.library = library
    def install():
        return library.install('qc', {k:v for k,v in args.items() if k.startswith('qc_reference_')},
            label='Synthetic human QC', version='test', provenance='Simulated fixture', usage='Test only')
    if not later: entry = install()
    with TestClient(backend.app) as client:
        client.post('/api/v1/sessions', json=dict(session_id='session'))
        resource = upload(client, 'fragments.txt', Path(args['source_path']).read_bytes()).json()['resource_id']
        done, body = send(client, [resource], utterance=Model.objective)
        if later:
            assert done['status'] == 'clarification', (done, backend.errors)
            initial = backend.service._application.sessions.load('session')
            assert not initial.turns and not initial.revisions
            entry = install()
            done, body = send(client, [], turn='reply', utterance='Use the now available compatible reference.')
        assert done['status'] == 'succeeded', (done, backend.errors)
        run = backend.service._application.run_store.load(done['run_id'])
        step, = run.steps
        assert step.tool_name == 'import_scATAC_fragments'
        assert step.resolved_arguments['reference_bundle_sha256'] == args['reference_bundle_sha256']
        assert step.resolved_arguments['source_sha256'] == backend.service.resources.load(resource).source_sha256
        assert step.result['n_fragment_records'] == 3 and step.result['total_support'] == 6
        assert run.request.prompt == Model.objective
        assert len(library.entries()) == 1


def test_saved_prerequisite_prose_is_not_regenerated_admission_authority(tmp_path, monkeypatch):
    from agent.application import raw_handoff
    from agent.application.session_state import SessionError
    from agent.application.session_store import FileSessionStore
    args = downstream_inputs(tmp_path)
    backend = web_fragments(tmp_path, args, companions=(), model_type=ImportOnlyModel)
    with TestClient(backend.app) as client:
        client.post('/api/v1/sessions', json=dict(session_id='session'))
        resource = upload(client, 'fragments.txt', Path(args['source_path']).read_bytes()).json()['resource_id']
        pending, _ = send(client, [resource], utterance=ImportOnlyModel.objective)
        assert pending['status'] == 'clarification'
        sessions = backend.service._application.sessions
        before = sessions.load('session')
        calls = sum(len(m.calls) for m in backend.models)
        monkeypatch.setattr(raw_handoff, '_resolve_resource', lambda *a, **k: (['Updated prerequisite wording.'], None))
        assert FileSessionStore(sessions._store.root).load('session') == before
        restored = client.get('/api/v1/sessions/session')
        assert restored.status_code == 200
        assert 'Updated prerequisite wording.' not in restored.text
        assert sum(len(m.calls) for m in backend.models) == calls
        # Wording cannot authorize execution: a saved clarification with no
        # remaining prerequisites and no completed exact binding is invalid.
        monkeypatch.setattr(raw_handoff, '_resolve_resource', lambda *a, **k: ([], None))
        with pytest.raises(SessionError): FileSessionStore(sessions._store.root).load('session')
