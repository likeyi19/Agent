"""Uploaded bytes use ordinary science, durable dialogue, and accepted history."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from fastapi.testclient import TestClient
import pytest

from agent.application import InteractiveAgentApplication
from agent.application.uploads import H5ADUploadAdmission
from agent.orchestration import PlanningModelProfile, ToolRegistry, build_default_tool_registry
from agent.providers import PlanningModelFactoryRegistry
from agent.schemas.orchestration import _serialize
from agent.web.app import create_app
from helpers import ScriptedModel, harness, wait_turn


PROFILES = tuple(PlanningModelProfile(key, 'scripted', 'model/' + key)
                for key in ('alpha', 'beta'))
LABELS = {'alpha': 'Default test model', 'beta': 'Other test model'}


def uploaded_backend(tmp_path, *, model_type=ScriptedModel):
    """Reuse the existing tiny source and count the unchanged inspection owner."""
    backend = harness(tmp_path)
    workspace = backend.service._application._workspace
    root = workspace._ensure_directory(workspace.root / 'uploads')

    def factory(profile):
        model = model_type(profile)
        backend.models.append(model)
        return model

    backend.service = InteractiveAgentApplication(tmp_path / 'workspace',
        model_profiles=PROFILES, default_profile_id='alpha', display_labels=LABELS,
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': factory}),
        registry=backend.service._application.registry, approved_source_roots=(root,))
    uploads = H5ADUploadAdmission(backend.service.resources, root)
    return backend, uploads, root


def upload(client, payload, *, upload_id='first-upload', filename='Supplied matrix.h5ad'):
    response = client.put('/api/v1/uploads/' + upload_id,
        params={'filename': filename}, content=payload,
        headers={'Content-Type': 'application/octet-stream'})
    assert response.status_code == 201, response.text
    record = response.json()
    assert set(record) == {'resource_id', 'input_type', 'label', 'status'}
    assert record['input_type'] == 'h5ad' and record['status'] == 'registered'
    return record


def body(resource_id=None, *, turn='first', generation=0,
         utterance='Inspect the supplied H5AD.', **changes):
    value = dict(turn_id=turn, expected_generation=generation, utterance=utterance)
    if resource_id is not None:
        value['resource_id'] = resource_id
    return value | changes


def submit(client, payload):
    response = client.post('/api/v1/sessions/session/turns', json=payload)
    assert response.status_code == 202, response.text
    return wait_turn(client, payload['turn_id'])


def revision_url(revision_id):
    return '/api/v1/sessions/session/revisions/' + revision_id


def accepted_presentations(client, revision_id):
    path = revision_url(revision_id)
    evidence_response = client.get(path + '/evidence', params={'output_name': 'inspection'})
    assert evidence_response.status_code == 200, evidence_response.text
    evidence = evidence_response.json()
    assert evidence['status'] == 'available'
    facts = {fact['field']: fact for fact in evidence['facts']}
    assert facts['n_cells']['status'] == 'available' and facts['n_cells']['value'] == 2
    assert facts['n_features']['status'] == 'available' and facts['n_features']['value'] == 3
    artifacts = client.get(path + '/artifacts').json()['artifacts']
    assert artifacts and {a['artifact_type'] for a in artifacts} == {'analysis_report'}
    report = artifacts[0]
    report_url = path + '/artifacts/' + report['handle']
    content = client.get(report_url)
    assert content.status_code == 200 and content.content
    assert content.headers['X-Artifact-Presentation'] == 'client_safe_report_projection'
    assert content.headers['X-Artifact-Source-SHA256'] == report['sha256']
    assert content.headers['X-Artifact-Content-SHA256'] == hashlib.sha256(content.content).hexdigest()
    downloaded = client.get(report_url, params={'download': 'true'})
    assert downloaded.content == content.content
    assert downloaded.headers['Content-Disposition'].startswith('attachment;')
    return evidence, artifacts, report_url, content.content


def snapshot(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob('*') if path.is_file()}


def assert_private(value, private_root):
    encoded = json.dumps(value)
    assert str(private_root) not in encoded
    assert all(token not in encoded for token in ('source_path', 'source_sha256',
        'execution_inputs', 'authority_payload', 'api_key', 'provider_prompt'))


def reopened_service(workspace, root):
    def forbidden(*args, **kwargs):
        raise AssertionError('Completed history must not invoke a provider or science.')
    registry = build_default_tool_registry()
    forbidden_registry = ToolRegistry(tuple(replace(registry.get(name), function=forbidden)
                                           for name in registry.names()))
    return InteractiveAgentApplication(workspace, model_profiles=PROFILES,
        default_profile_id='alpha', display_labels=LABELS,
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': forbidden}),
        registry=forbidden_registry, approved_source_roots=(root,))


def test_uploaded_inspection_dialogue_reselection_branch_and_restart(tmp_path):
    backend, uploads, root = uploaded_backend(tmp_path)
    with TestClient(create_app(backend.service, uploads=uploads)) as client:
        assert client.post('/api/v1/sessions', json={'session_id': 'session'}).status_code == 201
        resource = upload(client, backend.source.read_bytes())
        registered = backend.service.resources.load(resource['resource_id'])
        assert Path(registered.source_path).read_bytes() == backend.source.read_bytes()
        assert not backend.models and not backend.science_calls
        assert backend.service._application.sessions.load('session').generation == 0
        choices = client.get('/api/v1/resources').json()
        assert choices == {'enabled': True, 'choices': [resource]}

        first_body = body(resource['resource_id'])
        first = submit(client, first_body)
        assert first['status'] == 'succeeded' and first['revision_id']
        assert first['response']['status'] == 'activated'
        assert backend.science_calls == ['inspect_scATAC'] and len(backend.models) == 1
        run = backend.service._application.run_store.load(first['run_id'])
        step, = run.steps
        assert step.tool_name == 'inspect_scATAC' and step.verification.passed
        assert step.verification.artifact_authority is None
        assert step.resolved_arguments['path'] == registered.source_path
        state = backend.service._application.sessions.load('session')
        assert _serialize(state.interactions[0].submission['registered_input']) == dict(
            resource_id=registered.resource_id, record_sha256=registered.record_sha256,
            tool_name='inspect_scATAC', composition='h5ad-science.v1')
        assert state.revisions[0].outputs[0].accepted_step_sha256
        evidence, artifacts, first_report_url, first_report = accepted_presentations(client, first['revision_id'])
        before_follow = len(backend.science_calls)

        follow = submit(client, body(turn='follow', generation=1,
            utterance='What does this accepted result show?', profile_id='beta'))
        assert follow['response']['kind'] == 'answer'
        assert follow['response']['scientific']['support'] == 'supported'
        assert follow['profile_id'] == 'beta' and len(backend.science_calls) == before_follow
        state = backend.service._application.sessions.load('session')
        assert state.active_revision_id == first['revision_id'] and state.generation == 1
        assert state.interactions[1].admitted['target']['revision_id'] == first['revision_id']
        assert state.interactions[1].admitted['target']['output_name'] == 'inspection'
        assert state.interactions[1].admitted['target']['subject'] is None

        third_body = body(resource['resource_id'], turn='again', generation=1,
                          utterance='Inspect the same registered H5AD again.')
        third = submit(client, third_body)
        assert third['status'] == 'succeeded' and third['revision_id'] != first['revision_id']
        assert len(backend.science_calls) == 2 and client.get('/api/v1/resources').json() == choices
        active = client.get('/api/v1/sessions/session').json()
        assert active['generation'] == 2 and active['active_revision_id'] == third['revision_id']
        count_before_navigation = len(backend.models)
        navigation = client.post('/api/v1/sessions/session/activate', json=dict(
            turn_id='back', expected_generation=2, revision_id=first['revision_id']))
        assert navigation.status_code == 200, navigation.text
        assert navigation.json()['generation'] == 3
        assert len(backend.models) == count_before_navigation and len(backend.science_calls) == 2
        branch = submit(client, body(resource['resource_id'], turn='branch', generation=3,
            utterance='Inspect this registered H5AD on this branch.', profile_id='beta'))
        assert branch['status'] == 'succeeded' and len(backend.science_calls) == 3
        history = client.get('/api/v1/sessions/session').json()
        revisions = {r['revision_id']: r for r in history['revisions']}
        assert history['generation'] == 4 and len(revisions) == 3
        assert revisions[third['revision_id']]['parent_revision_id'] == first['revision_id']
        assert revisions[branch['revision_id']]['parent_revision_id'] == first['revision_id']
        assert revisions[branch['revision_id']]['is_active']
        assert not revisions[third['revision_id']]['is_active']
        assert [t['turn_id'] for t in history['turns'] if t['utterance']] == [
            'first', 'follow', 'again', 'branch']
        assert next(t for t in history['turns'] if t['turn_id'] == 'back')['status'] == 'navigated'
        assert_private([resource, history, evidence, artifacts], tmp_path)
        assert str(tmp_path).encode() not in first_report
        historical_evidence = client.get(revision_url(first['revision_id']) + '/evidence',
            params={'output_name': 'inspection'}).json()
        assert historical_evidence == evidence | {'generation': 4, 'is_active': False}

    # Registered-source lifetime is independent of completed accepted history.
    Path(registered.source_path).unlink()
    read_only = reopened_service(tmp_path / 'workspace', root)
    resumed_uploads = H5ADUploadAdmission(read_only.resources, root)
    before_restart = snapshot(tmp_path)
    with TestClient(create_app(read_only, uploads=resumed_uploads)) as client:
        assert client.get('/api/v1/sessions/session').json() == history
        assert client.get('/api/v1/resources').json() == choices
        assert submit(client, first_body) == first
        assert client.get(first_report_url).content == first_report
        again_evidence, again_artifacts, _, _ = accepted_presentations(client, first['revision_id'])
        assert again_evidence == historical_evidence and again_artifacts == artifacts
        missing = client.post('/api/v1/sessions/session/turns', json=body(
            resource['resource_id'], turn='missing-new', generation=4))
        assert missing.status_code == 400
        assert missing.json()['error']['code'] == 'LOCAL_RESOURCE_INTEGRITY_INVALID'
        assert client.get('/api/v1/sessions/session').json() == history
    assert snapshot(tmp_path) == before_restart

    # A distinct process restores durable bytes with provider/science factories forbidden.
    script = r'''
import json, sys
from pathlib import Path
from fastapi.testclient import TestClient
from agent.application.uploads import H5ADUploadAdmission
from test_uploaded_multiturn import reopened_service
from agent.web.app import create_app
expected = json.loads(sys.stdin.read())
workspace, root = map(Path, sys.argv[1:])
service = reopened_service(workspace, root)
with TestClient(create_app(service, uploads=H5ADUploadAdmission(service.resources, root))) as client:
    assert client.get('/api/v1/sessions/session').json() == expected['history']
    assert client.get('/api/v1/resources').json() == expected['choices']
    assert client.post('/api/v1/sessions/session/turns', json=expected['body']).status_code == 202
    assert client.get('/api/v1/sessions/session/turns/first').json() == expected['first']
    assert client.get(expected['report_url']).text == expected['report']
print('accepted-history-preserved')
'''
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
        PYTHONPATH=os.pathsep.join((str(Path(__file__).parents[2] / 'src'), str(Path(__file__).parent))))
    result = subprocess.run([sys.executable, '-c', script, str(tmp_path / 'workspace'), str(root)],
        input=json.dumps(dict(history=history, choices=choices, body=first_body, first=first,
                             report_url=first_report_url, report=first_report.decode())),
        text=True, capture_output=True, timeout=45, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == 'accepted-history-preserved'
    assert snapshot(tmp_path) == before_restart and len(backend.science_calls) == 3


@pytest.mark.parametrize('mutation', ['changed', 'missing'])
def test_uploaded_source_new_consumption_checks_bytes_but_completed_retry_survives(tmp_path, mutation):
    backend, uploads, _ = uploaded_backend(tmp_path)
    with TestClient(create_app(backend.service, uploads=uploads)) as client:
        client.post('/api/v1/sessions', json={'session_id': 'session'})
        resource = upload(client, backend.source.read_bytes())
        original_body = body(resource['resource_id'])
        first = submit(client, original_body)
        assert first['status'] == 'succeeded'
        source = Path(backend.service.resources.load(resource['resource_id']).source_path)
        if mutation == 'changed':
            source.write_bytes(b'Changed after accepted inspection.')
        else:
            source.unlink()
        before_models = [(model, len(model.calls)) for model in backend.models]
        rejected = client.post('/api/v1/sessions/session/turns', json=body(
            resource['resource_id'], turn='new-consumption', generation=1))
        assert rejected.status_code == 400, rejected.text
        assert rejected.json()['error']['code'] == 'LOCAL_RESOURCE_INTEGRITY_INVALID'
        assert [(model, len(model.calls)) for model in backend.models] == before_models
        assert backend.science_calls == ['inspect_scATAC']
        assert submit(client, original_body) == first
        assert [(model, len(model.calls)) for model in backend.models] == before_models
        state = backend.service._application.sessions.load('session')
        assert state.generation == 1 and len(state.interactions) == 1
        assert client.get('/api/v1/resources').json()['choices'] == [resource]


def test_corrupt_uploaded_h5ad_registers_as_bytes_then_fails_scientific_owner(tmp_path):
    backend, uploads, _ = uploaded_backend(tmp_path)
    with TestClient(create_app(backend.service, uploads=uploads)) as client:
        client.post('/api/v1/sessions', json={'session_id': 'session'})
        resource = upload(client, b'This stable completed file is not HDF5.')
        assert not backend.models and not backend.science_calls
        failed = submit(client, body(resource['resource_id']))
        assert failed['status'] == 'failed' and failed['revision_id'] is None
        assert failed['error']['code'] == 'TOOL_EXCEPTION'
        assert backend.science_calls == ['inspect_scATAC'] and len(backend.models) == 1
        run = backend.service._application.run_store.load(failed['run_id'])
        step, = run.steps
        assert step.error.code == failed['error']['code']
        assert step.verification is None or step.verification.artifact_authority is None
        state = backend.service._application.sessions.load('session')
        assert not state.revisions and state.generation == 0
        assert client.get('/api/v1/resources').json()['choices'] == [resource]
        assert_private(failed, tmp_path)


class EmbeddingIntentModel(ScriptedModel):
    """Provider-authored semantic decision; uploaded H5AD is only a data source."""
    def complete(self, *, prompt, response_schema):
        value = json.loads(prompt)
        if 'turn_schema_version' in value:
            self.calls.append(value)
            return json.dumps(dict(turn_schema_version=1,
                decision=dict(kind='execute_plan', target='epizoo_embed_cells')))
        if 'selection_schema_version' in response_schema.get('properties', {}):
            self.calls.append(value)
            return json.dumps(dict(selection_schema_version=1,
                decision=dict(kind='select', capability_ids=['embedding_analysis'])))
        self.calls.append(value)
        return json.dumps(dict(schema_version=4, decision=dict(kind='plan', steps=[
            dict(step_id='embed', tool='epizoo_embed_cells', sources=[], control_dependencies=[])])))


def test_uploaded_h5ad_does_not_substitute_inspection_for_other_scientific_intent(tmp_path):
    backend, uploads, _ = uploaded_backend(tmp_path, model_type=EmbeddingIntentModel)
    with TestClient(create_app(backend.service, uploads=uploads)) as client:
        client.post('/api/v1/sessions', json={'session_id': 'session'})
        resource = upload(client, backend.source.read_bytes())
        result = submit(client, body(resource['resource_id'], utterance='Compute EpiZoo embeddings.'))
        assert result['status'] == 'clarification' and result['revision_id'] is None, result
        assert result['error'] is None
        assert result['response']['clarification']['reason'] == 'missing_species'
        assert not backend.science_calls and len(backend.models) == 1
        assert all(step['tool_name'] != 'inspect_scATAC' for step in result['steps'])
        state = backend.service._application.sessions.load('session')
        assert not state.revisions and state.generation == 0
        assert state.interactions[0].admitted['tool'] == 'epizoo_embed_cells'
        assert_private(result, tmp_path)
