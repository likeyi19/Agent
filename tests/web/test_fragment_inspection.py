"""Real HTTP/owner/evidence path; scripted language, simulated input only."""
import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from test_attachment_role_handoff import role_web, send, RoleModel
from test_chat_attachments import upload
from agent.web.app import create_app


class InspectionModel(RoleModel):
    indexed = False
    missing = False
    def complete(self, *, prompt, response_schema):
        value = json.loads(prompt); self.calls.append(value)
        if 'turn_schema_version' in value:
            roles = [dict(tool='inspect_scATAC_fragments', port='source', source='source.0')]
            if self.indexed: roles.append(dict(tool='inspect_scATAC_fragments', port='source_index', source='source.1'))
            if self.missing and not value['dialogue'].get('pending_prerequisite'):
                decision = dict(kind='request_roles', target='inspect_scATAC_fragments', roles=[],
                    required_roles=[dict(tool='inspect_scATAC_fragments', port='source')])
            elif self.missing:
                decision = dict(kind='answer_roles', pending='@roles', roles=roles)
            else:
                decision = dict(kind='execute_plan', target='inspect_scATAC_fragments', roles=roles)
            return json.dumps(dict(turn_schema_version=1, decision=decision))
        if 'output_selection_schema_version' in value:
            return json.dumps(dict(outputs=[dict(name='inspection', step_id='inspect', output_key='n_records_observed')]))
        return json.dumps(dict(schema_version=4, decision=dict(kind='plan', steps=[
            dict(step_id='inspect', tool='inspect_scATAC_fragments', sources=[], control_dependencies=[])])))


@pytest.mark.parametrize('indexed,missing', [(False, False), (True, False), (False, True)])
def test_ordinary_attachment_inspection_has_no_preprocessing_prerequisites(tmp_path, indexed, missing):
    class Model(InspectionModel): pass
    Model.indexed, Model.missing = indexed, missing
    backend, uploads = role_web(tmp_path, model_type=Model)
    with TestClient(create_app(backend.service, uploads=uploads)) as client:
        client.post('/api/v1/sessions', json=dict(session_id='session'))
        if missing:
            pending, _ = send(client, text='Inspect only the basic fragment structure.')
            assert pending['status'] == 'clarification'
            state = backend.service._application.sessions.load('session')
            assert not state.turns and not state.revisions
        data = b'chrUnknown\t1\t5\tA\t2\n'
        if indexed:
            import pysam
            source = tmp_path/'source.gz'
            with pysam.BGZFile(str(source), 'wb') as f: f.write(data)
            pysam.tabix_index(str(source), preset='bed')
            data = source.read_bytes()
        resource = upload(client, 'opaque.txt', data).json()['resource_id']
        ids = [resource]
        if indexed:
            ids.append(upload(client, 'index', Path(str(source)+'.tbi').read_bytes(), token='index').json()['resource_id'])
        done, body = send(client, turn='reply' if missing else 'first',
            text='Here is the file.' if missing else 'Inspect only the basic fragment structure.', ids=ids)
        assert done['status'] == 'succeeded', done
        run = backend.service._application.run_store.load(done['run_id'])
        step, = run.steps
        assert step.tool_name == 'inspect_scATAC_fragments'
        assert step.result['n_records_observed'] == 1
        assert step.resolved_arguments['source_sha256'] == uploads.resources.load(resource).source_sha256
        assert 'reference_bundle_path' not in step.resolved_arguments
        if indexed:
            assert step.resolved_arguments['source_index_sha256'] == uploads.resources.load(ids[1]).source_sha256
            assert step.result['index_status'] == 'tbi_header_only_pair_unverified'
        assert run.request.prompt == 'Inspect only the basic fragment structure.'
        state = backend.service._application.sessions.load('session')
        assert len(state.revisions) == 1 and len(state.turns) == 1
        calls = sum(len(m.calls) for m in backend.models)
        assert client.get('/api/v1/sessions/session').status_code == 200
        assert client.post('/api/v1/sessions/session/turns', json=body).status_code == 202
        assert sum(len(m.calls) for m in backend.models) == calls
