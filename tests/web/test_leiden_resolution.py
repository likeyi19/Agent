"""One utterance-grounded Leiden argument reaches ordinary verified science.

Language decisions and EpiZoo inference use the established scripted interfaces.
The tiny sparse input, downstream owners, compiler, Session, reports and HTTP
worker are production implementations; no provider or GPU is needed.
"""
from copy import deepcopy
import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from agent.application import InteractiveAgentApplication, ResearchAgentApplication
from agent.application.session_state import SessionConflictError, SessionError, digest
from agent.application.uploads import H5ADUploadAdmission
from agent.schemas.orchestration import _serialize
from agent.web.config import ScientificInputSet

from helpers import wait_turn
import test_h5ad_analysis as analysis
from test_species_handoff import reply, web
from test_uploaded_multiturn import assert_private, body, snapshot, upload


EXPLICIT = 'Run EpiZoo, perform Leiden clustering at resolution 0.7, and generate UMAP.'
OMITTED = 'Run EpiZoo, perform Leiden clustering, and generate UMAP.'
QUESTION = 'What resolution did you use for clustering?'


def argument(utterance, token='0.7', **changes):
    start = utterance.index(token)
    return dict(tool='cluster_cells', argument='resolution', literal=token,
                start=start, end=start + len(token)) | changes


def execute(utterance, token='0.7', **changes):
    return dict(kind='execute_plan', target='compute_cell_umap',
                argument=argument(utterance, token, **changes))


def resolution_harness(tmp_path, monkeypatch, *, decisions=None, missing_species=False,
                       parameters=None, plan_variant=None):
    decisions = {EXPLICIT: execute(EXPLICIT), OMITTED:
        dict(kind='execute_plan', target='compute_cell_umap'),
        'Mouse.': dict(kind='answer_prerequisite', pending='@species', species='mouse'),
        QUESTION: dict(kind='answer_scientific',
            target=dict(output='r0', subject=None), comparison=None, focus='question')} | (decisions or {})
    original_model = analysis.ScientificPlanModel

    class ResolutionModel(original_model):
        def complete(self, *, prompt, response_schema):
            value = json.loads(prompt)
            if 'turn_schema_version' in value and value['utterance'] in decisions:
                self.calls.append(value)
                decision = decisions[value['utterance']]
                if decision['kind'] in ('execute_plan', 'answer_prerequisite'):
                    self.mode = 'chain'
                return json.dumps(dict(turn_schema_version=1, decision=decision))
            if 'dialogue_schema_version' in value and value['question'] == QUESTION:
                self.calls.append(value)
                fields = ('clustering_parameters.resolution', 'clustering_resolution_origin')
                offered = {c['field']: c['claim_id'] for c in value['evidence']['claims']}
                return json.dumps(dict(support='supported', paragraphs=[dict(parts=[
                    dict(kind='text', text='The accepted clustering records:'),
                    *(dict(kind='claim', id=offered[field]) for field in fields)])]))
            result = super().complete(prompt=prompt, response_schema=response_schema)
            parsed = json.loads(result)
            if plan_variant and parsed.get('schema_version') == 4 and self.mode == 'chain':
                steps = parsed['decision']['steps']
                if plan_variant == 'no-cluster':
                    steps[:] = [s for s in steps if s['tool'] != 'cluster_cells']
                    steps[-1]['sources'][0]['step'] = 'neighbors'
                elif plan_variant == 'repeated-cluster':
                    repeated = deepcopy(steps[-2])
                    repeated['step_id'] = 'another-cluster'
                    repeated['sources'][0]['step'] = 'cluster'
                    steps.insert(-1, repeated)
                    steps[-1]['sources'][0]['step'] = 'another-cluster'
                else:
                    raise AssertionError(plan_variant)
                return json.dumps(parsed)
            return result

    monkeypatch.setattr(analysis, 'ScientificPlanModel', ResolutionModel)
    harness = analysis.analysis_harness(tmp_path, monkeypatch)
    values = dict(device='cpu') | (parameters or {})
    if not missing_species:
        values['species'] = 'mouse'
    return harness, (ScientificInputSet('declarations', 'Explicit scientific declarations',
        values, h5ad_companion=True),)


def start(client, harness, *, utterance=EXPLICIT, turn='first', generation=0,
          resource_id=None):
    if resource_id is None:
        resource_id = upload(client, harness.source.read_bytes())['resource_id']
    result, payload = analysis.scientific_submit(client, resource_id,
        utterance=utterance, turn=turn, generation=generation, input_set_id='declarations')
    return result, payload, resource_id


def steps(harness, result):
    return harness.service._application.run_store.load(result['run_id']).steps


def assert_no_science(harness, result):
    assert result['revision_id'] is None and result['status'] != 'succeeded', result
    assert not harness.science_calls and not harness.inference_calls
    assert not harness.service._application.sessions.load('session').revisions


def assert_chain(harness, result, resolution, *, explicit):
    assert result['status'] == 'succeeded' and result['response']['status'] == 'activated', result
    run_steps = steps(harness, result)
    assert [s.tool_name for s in run_steps] == ['inspect_scATAC', 'epizoo_embed_cells',
        'build_cell_neighbors', 'cluster_cells', 'compute_cell_umap']
    assert all(s.verification.passed for s in run_steps)
    inspected, embedded, neighbors, clustered, umap = run_steps
    assert clustered.result['resolution'] == resolution
    assert ('resolution' in clustered.resolved_arguments) is explicit
    if explicit:
        assert clustered.resolved_arguments['resolution'] == resolution
    assert all('resolution' not in s.resolved_arguments for s in run_steps if s is not clustered)
    assert neighbors.result['n_neighbors'] == 15 and neighbors.result['metric'] == 'euclidean'
    assert umap.result['min_dist'] == .5 and umap.result['spread'] == 1.0
    assert len(harness.inference_calls) == 1 and harness.inference_calls[0]['device'] == 'cpu'
    assert embedded.result['resource_provenance']['actual_resource_identity']['resource_id'] == 'mouse-reviewed'
    return clustered


@pytest.mark.parametrize('explicit', [False, True])
def test_normal_chat_execution_uses_explicit_leiden_argument_or_existing_owner_default(
        tmp_path, monkeypatch, explicit):
    harness, inputs = resolution_harness(tmp_path, monkeypatch)
    utterance = EXPLICIT if explicit else OMITTED
    with TestClient(web(harness, inputs)) as client:
        client.post('/api/v1/sessions', json={'session_id': 'session'})
        result, payload, resource_id = start(client, harness, utterance=utterance)
        clustered = assert_chain(harness, result, .7 if explicit else 1.0, explicit=explicit)
        interaction = harness.service._application.sessions.load('session').interactions[0]
        # The received selection stays immutable; only admitted execution gains
        # the exact validated text declaration.
        assert 'resolution' not in interaction.submission['execution_inputs']
        assert interaction.submission['registered_input']['resource_id'] == resource_id
        if explicit:
            assert _serialize(interaction.admitted['argument']) == argument(utterance)
            assert interaction.admitted['inputs']['resolution'] == .7
        else:
            assert 'resolution' not in interaction.admitted['inputs']
        assert clustered.result['random_seed'] == 0
        evidence, artifacts, report = analysis.accepted_result(client, result)
        facts = {f['field']: f for f in evidence['facts']}
        assert facts['clustering_parameters']['value'] == dict(
            resolution=.7 if explicit else 1.0, random_seed=0)
        assert facts['clustering_resolution_origin']['value'] == (
            'explicit_execution_argument' if explicit else 'existing_owner_default')
        handle = next(a['handle'] for a in artifacts if a['artifact_type'] == 'analysis_report')
        delivered = client.get(f"/api/v1/sessions/session/revisions/{result['revision_id']}/artifacts/{handle}",
            params={'download': 'true'})
        assert delivered.content == report and delivered.headers['Content-Disposition'].startswith('attachment;')
        assert_private([result, evidence, artifacts], tmp_path)
        assert str(tmp_path).encode() not in report
        calls = (len(harness.models), len(harness.science_calls), len(harness.inference_calls))
        assert client.post('/api/v1/sessions/session/turns', json=payload).status_code == 202
        assert wait_turn(client) == result
        assert calls == (len(harness.models), len(harness.science_calls), len(harness.inference_calls))


@pytest.mark.parametrize('typed,accepted', [(.7, True), (.9, False)])
def test_text_and_typed_resolution_agree_or_require_explicit_conflict_resolution(
        tmp_path, monkeypatch, typed, accepted):
    harness, inputs = resolution_harness(tmp_path, monkeypatch, parameters={'resolution': typed})
    with TestClient(web(harness, inputs)) as client:
        client.post('/api/v1/sessions', json={'session_id': 'session'})
        result, _, _ = start(client, harness)
        if accepted:
            assert_chain(harness, result, .7, explicit=True)
        else:
            assert_no_science(harness, result)
            assert result['response']['kind'] == 'clarify'
            assert result['response']['clarification']['reason'] == 'conflicting_scientific_parameter'
        original = harness.service._application.sessions.load('session').interactions[0]
        assert original.submission['execution_inputs']['resolution'] == typed


@pytest.mark.parametrize('literal,changes', [
    ('0', {}), ('-0.2', {}), ('NaN', {}), ('inf', {}), ('1e309', {}),
    ('0.7', {'literal': True}), ('0.7', {'tool': 'compute_cell_umap'}),
    ('0.7', {'argument': 'min_dist'}), ('0.7', {'start': 0, 'end': 3}),
])
def test_invalid_explicit_declaration_never_reaches_scientific_production(
        tmp_path, monkeypatch, literal, changes):
    utterance = f'Run EpiZoo, cluster at Leiden resolution {literal}, and generate UMAP.'
    harness, inputs = resolution_harness(tmp_path, monkeypatch,
        decisions={utterance: execute(utterance, literal, **changes)})
    with TestClient(web(harness, inputs)) as client:
        client.post('/api/v1/sessions', json={'session_id': 'session'})
        result, _, _ = start(client, harness, utterance=utterance)
        assert_no_science(harness, result)
        assert result['response']['kind'] == 'clarify', result


@pytest.mark.parametrize('utterance,decision', [
    ('What does Leiden resolution 0.7 mean?', dict(kind='clarify', reason='unavailable_context')),
    ('Use resolution 0.7 for an unsupported analysis.', dict(kind='clarify', reason='unsupported_intent')),
    ('Use 0.7 somewhere in this analysis.', dict(kind='clarify', reason='ambiguous_parameter')),
])
def test_explanation_unsupported_and_ambiguous_language_does_not_authorize_science(
        tmp_path, monkeypatch, utterance, decision):
    harness, inputs = resolution_harness(tmp_path, monkeypatch, decisions={utterance: decision})
    with TestClient(web(harness, inputs)) as client:
        client.post('/api/v1/sessions', json={'session_id': 'session'})
        result, _, _ = start(client, harness, utterance=utterance)
        assert_no_science(harness, result)
        assert result['response']['clarification']['reason'] == decision['reason']


@pytest.mark.parametrize('plan_variant', ['no-cluster', 'repeated-cluster'])
def test_actual_plan_must_offer_one_exact_cluster_consumer_before_any_execution(
        tmp_path, monkeypatch, plan_variant):
    harness, inputs = resolution_harness(tmp_path, monkeypatch, plan_variant=plan_variant)
    with TestClient(web(harness, inputs)) as client:
        client.post('/api/v1/sessions', json={'session_id': 'session'})
        result, _, _ = start(client, harness)
        assert_no_science(harness, result)


def test_pending_resolution_survives_refresh_restart_species_answer_and_retry(tmp_path, monkeypatch):
    harness, inputs = resolution_harness(tmp_path, monkeypatch, missing_species=True)
    with TestClient(web(harness, inputs)) as client:
        client.post('/api/v1/sessions', json={'session_id': 'session'})
        pending, _, resource_id = start(client, harness)
        assert pending['response']['clarification']['reason'] == 'missing_species'
        assert_no_science(harness, pending)
        original = harness.service._application.sessions.load('session').interactions[0]
        assert original.admitted['inputs']['resolution'] == .7
        assert 'resolution' not in original.submission['execution_inputs']
        history = client.get('/api/v1/sessions/session').json()
        calls = len(harness.models)
        assert client.get('/api/v1/sessions/session').json() == history
        assert client.get('/api/v1/sessions/session/turns/first').json() == pending
        assert len(harness.models) == calls
    previous = harness.service
    harness.service = InteractiveAgentApplication(tmp_path / 'workspace',
        model_profiles=tuple(previous._profiles.values()), default_profile_id=previous._default,
        planning_model_factory_registry=previous._factory,
        registry=previous._application.registry, approved_source_roots=(harness.root,))
    harness.uploads = H5ADUploadAdmission(harness.service.resources, harness.root)
    before = snapshot(tmp_path)
    with TestClient(web(harness, inputs)) as client:
        assert client.get('/api/v1/sessions/session').json() == history
        assert snapshot(tmp_path) == before and len(harness.models) == calls
        result, payload = reply(client)
        assert_chain(harness, result, .7, explicit=True)
        state = harness.service._application.sessions.load('session')
        assert state.generation == 1 and len(state.revisions) == 1
        resumed = state.interactions[-1]
        assert resumed.admitted['inputs']['input_path'] == harness.service.resources.load(resource_id).source_path
        assert resumed.admitted['inputs']['resolution'] == .7
        assert resumed.submission['execution_inputs'] == {}
        selection = next(c for c in harness.models[-1].calls if 'output_selection_schema_version' in c)
        assert selection['question'] == EXPLICIT
        analysis.accepted_result(client, result)
        calls = (len(harness.models), len(harness.science_calls), len(harness.inference_calls))
        assert client.post('/api/v1/sessions/session/turns', json=payload).status_code == 202
        assert wait_turn(client, 'answer') == result
        assert calls == (len(harness.models), len(harness.science_calls), len(harness.inference_calls))


@pytest.mark.parametrize('original_explicit,reply_literal,accepted', [
    (False, '0.7', True), (True, '0.9', False),
])
def test_combined_species_parameter_reply_preserves_original_objective_and_conflicts(
        tmp_path, monkeypatch, original_explicit, reply_literal, accepted):
    utterance = f'Mouse; use Leiden resolution {reply_literal}.'
    decision = dict(kind='answer_prerequisite', pending='@species', species='mouse',
                    argument=argument(utterance, reply_literal))
    harness, inputs = resolution_harness(tmp_path, monkeypatch, missing_species=True,
        decisions={utterance: decision})
    with TestClient(web(harness, inputs)) as client:
        client.post('/api/v1/sessions', json={'session_id': 'session'})
        pending, _, _ = start(client, harness, utterance=EXPLICIT if original_explicit else OMITTED)
        assert pending['response']['clarification']['reason'] == 'missing_species'
        result, _ = reply(client, utterance=utterance)
        if accepted:
            assert_chain(harness, result, .7, explicit=True)
            selection = next(c for c in harness.models[-1].calls if 'output_selection_schema_version' in c)
            assert selection['question'] == OMITTED
            sessions = harness.service._application.sessions
            persisted = sessions._store._path('session', '.json')
            original = persisted.read_bytes()
            corrupt = json.loads(original)
            del corrupt['record']['interactions'][-1]['admitted']['argument']
            corrupt['sha256'] = digest(corrupt['record'])
            calls = (len(harness.models), len(harness.science_calls), len(harness.inference_calls))
            try:
                persisted.write_text(json.dumps(corrupt))
                with pytest.raises(SessionError):
                    sessions.load('session')
            finally:
                persisted.write_bytes(original)
            assert sessions.load('session').interactions[-1].admitted['inputs']['resolution'] == .7
            assert calls == (len(harness.models), len(harness.science_calls), len(harness.inference_calls))
        else:
            assert_no_science(harness, result)
            assert result['response']['clarification']['reason'] == 'conflicting_scientific_parameter'
            recovered, _ = reply(client, turn='valid-after-conflict')
            assert_chain(harness, recovered, .7, explicit=True)


def test_current_typed_resolution_continues_omitted_origin_and_cannot_be_rewritten(tmp_path, monkeypatch):
    harness, inputs = resolution_harness(tmp_path, monkeypatch, missing_species=True)
    inputs += (ScientificInputSet('typed-resolution', 'Explicit Leiden resolution',
        dict(device='cpu', resolution=.7), h5ad_companion=True),)
    with TestClient(web(harness, inputs)) as client:
        client.post('/api/v1/sessions', json={'session_id': 'session'})
        pending, _, resource_id = start(client, harness, utterance=OMITTED)
        assert pending['response']['clarification']['reason'] == 'missing_species'
        result, _ = reply(client, resource_id=resource_id, input_set_id='typed-resolution')
        assert_chain(harness, result, .7, explicit=True)
        sessions = harness.service._application.sessions
        continuation = sessions.load('session').interactions[-1]
        assert 'argument' not in continuation.admitted
        assert continuation.submission['execution_inputs']['resolution'] == .7
        assert continuation.admitted['inputs']['resolution'] == .7
        persisted = sessions._store._path('session', '.json')
        accepted = persisted.read_bytes()
        calls = (len(harness.models), len(harness.science_calls), len(harness.inference_calls))
        for changed in (.9, True):
            corrupt = json.loads(accepted)
            corrupt['record']['interactions'][-1]['admitted']['inputs']['resolution'] = changed
            corrupt['sha256'] = digest(corrupt['record'])
            try:
                persisted.write_text(json.dumps(corrupt))
                with pytest.raises(SessionError):
                    sessions.load('session')
            finally:
                persisted.write_bytes(accepted)
        assert sessions.load('session').interactions[-1].admitted['inputs']['resolution'] == .7
        assert calls == (len(harness.models), len(harness.science_calls), len(harness.inference_calls))


def test_stale_generation_does_not_construct_provider_or_redirect_pending_argument(tmp_path, monkeypatch):
    harness, inputs = resolution_harness(tmp_path, monkeypatch, missing_species=True)
    with TestClient(web(harness, inputs)) as client:
        client.post('/api/v1/sessions', json={'session_id': 'session'})
        pending, _, _ = start(client, harness)
        assert pending['response']['clarification']['reason'] == 'missing_species'
        calls = len(harness.models)
        response = client.post('/api/v1/sessions/session/turns',
            json=body(turn='stale', generation=1, utterance='Mouse.'))
        assert response.status_code == 409
        assert response.json()['error']['code'] == 'INTERACTIVE_GENERATION_CONFLICT'
        assert len(harness.models) == calls and not harness.science_calls
        assert harness.service._application.sessions.load('session').interactions[0].admitted['inputs']['resolution'] == .7


def test_changed_dataset_cannot_receive_the_pending_parameter(tmp_path, monkeypatch):
    harness, inputs = resolution_harness(tmp_path, monkeypatch, missing_species=True)
    with TestClient(web(harness, inputs)) as client:
        client.post('/api/v1/sessions', json={'session_id': 'session'})
        pending, _, first = start(client, harness)
        assert pending['response']['clarification']['reason'] == 'missing_species'
        second = upload(client, harness.source.read_bytes(), upload_id='second-upload')['resource_id']
        assert second != first
        result, _ = reply(client, resource_id=second, input_set_id='declarations')
        assert_no_science(harness, result)
        assert result['response']['clarification']['reason'] == 'invalid_prerequisite'


def test_new_objective_uses_omitted_default_instead_of_old_pending_parameter(tmp_path, monkeypatch):
    harness, inputs = resolution_harness(tmp_path, monkeypatch, missing_species=True)
    with TestClient(web(harness, inputs)) as client:
        client.post('/api/v1/sessions', json={'session_id': 'session'})
        pending, _, resource_id = start(client, harness)
        assert pending['response']['clarification']['reason'] == 'missing_species'
    # An ordinary new request supplies the genuine prerequisite through its
    # existing typed control, while its omitted optional resolution stays omitted.
    inputs = (ScientificInputSet('declarations', 'Explicit mouse declaration',
        dict(species='mouse', device='cpu'), h5ad_companion=True),)
    with TestClient(web(harness, inputs)) as client:
        result, _, _ = start(client, harness, utterance=OMITTED, turn='new', resource_id=resource_id)
        assert_chain(harness, result, 1.0, explicit=False)
        resumed = harness.service._application.sessions.load('session').interactions[-1]
        assert 'continuation' not in resumed.admitted and 'resolution' not in resumed.admitted['inputs']


def test_changed_registered_source_blocks_parameter_continuation_before_science(tmp_path, monkeypatch):
    harness, inputs = resolution_harness(tmp_path, monkeypatch, missing_species=True)
    with TestClient(web(harness, inputs)) as client:
        client.post('/api/v1/sessions', json={'session_id': 'session'})
        pending, _, resource_id = start(client, harness)
        assert pending['response']['clarification']['reason'] == 'missing_species'
        Path(harness.service.resources.load(resource_id).source_path).write_bytes(b'Changed registered source.')
        result, _ = reply(client)
        assert_no_science(harness, result)
        assert result['error']['code'] == 'LOCAL_RESOURCE_INTEGRITY_INVALID'


@pytest.mark.parametrize('mutation', ['changed-value', 'boolean-equal-one', 'deleted-argument'])
def test_pending_parameter_cannot_be_rewritten_under_a_valid_outer_session_checksum(
        tmp_path, monkeypatch, mutation):
    literal = '1.0' if mutation == 'boolean-equal-one' else '0.7'
    utterance = EXPLICIT.replace('0.7', literal)
    harness, inputs = resolution_harness(tmp_path, monkeypatch, missing_species=True,
        decisions={utterance: execute(utterance, literal)})
    with TestClient(web(harness, inputs)) as client:
        client.post('/api/v1/sessions', json={'session_id': 'session'})
        pending, _, _ = start(client, harness, utterance=utterance)
        assert pending['response']['clarification']['reason'] == 'missing_species'
        sessions = harness.service._application.sessions
        persisted = sessions._store._path('session', '.json')
        original = persisted.read_bytes()
        corrupt = json.loads(original)
        admitted = corrupt['record']['interactions'][0]['admitted']
        if mutation == 'deleted-argument':
            del admitted['argument']
        else:
            admitted['inputs']['resolution'] = True if mutation == 'boolean-equal-one' else .9
        corrupt['sha256'] = digest(corrupt['record'])
        try:
            persisted.write_text(json.dumps(corrupt))
            with pytest.raises(SessionError):
                sessions.load('session')
        finally:
            persisted.write_bytes(original)
        assert sessions.load('session').interactions[0].admitted['inputs']['resolution'] == float(literal)
        assert not harness.science_calls and not harness.inference_calls


def test_direct_session_retry_compares_received_inputs_before_text_argument_binding(tmp_path, monkeypatch):
    harness, _ = resolution_harness(tmp_path, monkeypatch)
    app = ResearchAgentApplication(tmp_path / 'direct-workspace',
        registry=harness.service._application.registry,
        primary_planning_profile=harness.service._profiles['alpha'],
        planning_model_factory_registry=harness.service._factory)
    model = app.runtime.planner.model
    sessions = app.sessions
    sessions.create('direct-session')
    received = dict(input_path=str(harness.source), species='mouse', device='cpu',
                    checkpoint_path=str(harness.checkpoint))
    result = sessions.respond('direct-session', 'first', EXPLICIT, interpreter=model,
        expected_generation=0, execution_inputs=received)
    assert result.status == 'activated', result
    original = sessions.load('direct-session').interactions[0]
    assert original.submission is None
    assert original.admitted['inputs']['resolution'] == .7
    assert original.admitted['argument_received_inputs_sha256'] == digest(received)
    calls = (len(model.calls), len(harness.science_calls), len(harness.inference_calls))
    assert sessions.respond('direct-session', 'first', EXPLICIT, interpreter=model,
        expected_generation=0, execution_inputs=received).status == 'activated'
    assert calls == (len(model.calls), len(harness.science_calls), len(harness.inference_calls))
    for changed in ({}, received | {'resolution': .7}):
        with pytest.raises(SessionConflictError, match='Execution inputs changed on retry'):
            sessions.respond('direct-session', 'first', EXPLICIT, interpreter=model,
                expected_generation=0, execution_inputs=changed)
    persisted = sessions._store._path('direct-session', '.json')
    accepted = persisted.read_bytes()
    for deleted in ('argument', 'argument_received_inputs_sha256'):
        corrupt = json.loads(accepted)
        del corrupt['record']['interactions'][0]['admitted'][deleted]
        corrupt['sha256'] = digest(corrupt['record'])
        try:
            persisted.write_text(json.dumps(corrupt))
            with pytest.raises(SessionError):
                sessions.load('direct-session')
        finally:
            persisted.write_bytes(accepted)
    assert calls == (len(model.calls), len(harness.science_calls), len(harness.inference_calls))


@pytest.mark.parametrize('explicit', [False, True])
def test_followup_reports_exact_accepted_clustering_value_and_origin_without_science(
        tmp_path, monkeypatch, explicit):
    harness, inputs = resolution_harness(tmp_path, monkeypatch)
    with TestClient(web(harness, inputs)) as client:
        client.post('/api/v1/sessions', json={'session_id': 'session'})
        result, _, _ = start(client, harness, utterance=EXPLICIT if explicit else OMITTED)
        assert_chain(harness, result, .7 if explicit else 1.0, explicit=explicit)
        before = (len(harness.science_calls), len(harness.inference_calls))
        accepted_run = harness.service._application.run_store.load(result['run_id']).to_dict()
        answer, _ = reply(client, utterance=QUESTION, turn='question', generation=1)
        assert answer['response']['kind'] == 'answer', answer
        scientific = answer['response']['scientific']
        assert scientific['support'] == 'supported'
        prompt = next(c for c in harness.models[-1].calls if 'dialogue_schema_version' in c)
        claims = {c['field']: c for c in prompt['evidence']['claims']}
        assert claims['clustering_parameters.resolution']['value'] == (.7 if explicit else 1.0)
        assert claims['clustering_resolution_origin']['value'] == (
            'explicit_execution_argument' if explicit else 'existing_owner_default')
        assert f"Leiden resolution = {0.7 if explicit else 1.0}" in answer['response']['text']
        assert claims['clustering_resolution_origin']['value'] in answer['response']['text']
        assert scientific['targets'] == [dict(revision_id=result['revision_id'],
            output_name='result', subject=None)]
        assert before == (len(harness.science_calls), len(harness.inference_calls))
        assert harness.service._application.run_store.load(result['run_id']).to_dict() == accepted_run
        assert len(harness.service._application.sessions.load('session').revisions) == 1
