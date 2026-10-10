"""A compiler-grounded species question resumes its exact registered H5AD request.

Language decisions and EpiZoo inference are scripted through the established
interfaces. Scientific tools, resource admission, Session persistence, evidence,
reports and HTTP submission use the existing production implementations.
"""
from dataclasses import replace
import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from agent.application import InteractiveAgentApplication
from agent.application.session_state import SessionError, digest
from agent.application.uploads import H5ADUploadAdmission
from agent.schemas.orchestration import _serialize
from agent.web.app import create_app
from agent.web.config import ScientificInputSet

from helpers import wait_turn
import test_h5ad_analysis as analysis
from test_uploaded_multiturn import assert_private, body, snapshot, upload


def species_decision(species='mouse', pending='@species'):
    return dict(kind='answer_prerequisite', pending=pending, species=species)


def handoff_harness(tmp_path, monkeypatch, *, replies=None, companions=None):
    responses = {'Mouse.': species_decision(), "It's human.": species_decision('human')}
    responses.update(replies or {})
    original_model = analysis.ScientificPlanModel

    class HandoffModel(original_model):
        def complete(self, *, prompt, response_schema):
            value = json.loads(prompt)
            if 'turn_schema_version' in value and value['utterance'] in responses:
                self.calls.append(value)
                decision = responses[value['utterance']]
                pending = value.get('dialogue', {}).get('pending_prerequisite')
                if decision['kind'] == 'answer_prerequisite' and pending is not None:
                    # This is a provider-authored fixture plan for the original
                    # objective; the answer is not itself a new workflow recipe.
                    self.mode = 'chain' if pending['objective'].startswith('Analyze') else 'embed'
                return json.dumps(dict(turn_schema_version=1, decision=decision))
            return super().complete(prompt=prompt, response_schema=response_schema)

    monkeypatch.setattr(analysis, 'ScientificPlanModel', HandoffModel)
    harness = analysis.analysis_harness(tmp_path, monkeypatch)
    declarations = ScientificInputSet('declarations', 'Existing explicit scientific settings',
        dict(device='cpu') | (companions or {}), h5ad_companion=True)
    return harness, (declarations,)


def web(harness, inputs, *, resources=None):
    return create_app(harness.service, uploads=harness.uploads, input_sets=inputs,
        epizoo_resources=(harness.resource(),) if resources is None else resources)


def request_species(client, harness, *, objective='Embed the selected H5AD.', generation=0,
                    turn='first', resource_id=None):
    if resource_id is None:
        resource_id = upload(client, harness.source.read_bytes())['resource_id']
    result, payload = analysis.scientific_submit(client, resource_id,
        utterance=objective, turn=turn, generation=generation, input_set_id='declarations')
    assert result['response']['kind'] == 'clarify', result
    assert result['response']['clarification'] == dict(reason='missing_species',
        choices=['human', 'mouse'], value_required=True)
    assert result['revision_id'] is None and not result['steps']
    assert not harness.inference_calls
    interaction = harness.service._application.sessions.load('session').interactions[-1]
    assert _serialize(interaction.prerequisite) == dict(origin_turn_id=turn,
        field='species', resource_context_sha256=interaction.prerequisite['resource_context_sha256'])
    assert len(interaction.prerequisite['resource_context_sha256']) == 64
    assert 'species' not in interaction.submission['execution_inputs']
    return result, payload, resource_id


def reply(client, *, utterance='Mouse.', turn='answer', generation=0, **changes):
    payload = body(turn=turn, generation=generation, utterance=utterance, **changes)
    response = client.post('/api/v1/sessions/session/turns', json=payload)
    assert response.status_code == 202, response.text
    return wait_turn(client, turn, timeout=90), payload


def assert_closed(harness, result):
    assert result['status'] != 'succeeded' and result['revision_id'] is None, result
    assert not harness.inference_calls
    assert not harness.service._application.sessions.load('session').revisions


@pytest.mark.parametrize('objective,tools', [
    ('Embed the selected H5AD.', ['epizoo_embed_cells']),
    ('Analyze the selected H5AD.', ['inspect_scATAC', 'epizoo_embed_cells',
        'build_cell_neighbors', 'cluster_cells', 'compute_cell_umap']),
])
def test_mouse_answer_continues_original_requested_analysis_and_accepted_presentation(
        tmp_path, monkeypatch, objective, tools):
    harness, inputs = handoff_harness(tmp_path, monkeypatch)
    with TestClient(web(harness, inputs)) as client:
        client.post('/api/v1/sessions', json={'session_id': 'session'})
        pending, _, resource_id = request_species(client, harness, objective=objective)
        assert not harness.science_calls
        result, _ = reply(client)
        assert result['status'] == 'succeeded' and result['response']['status'] == 'activated', result
        assert harness.science_calls == tools and len(harness.inference_calls) == 1
        state = harness.service._application.sessions.load('session')
        assert state.generation == 1 and len(state.revisions) == 1
        original, resumed = state.interactions
        assert original.utterance == objective and resumed.utterance == 'Mouse.'
        assert resumed.admitted['inputs']['species'] == 'mouse'
        assert resumed.admitted['inputs']['input_path'] == original.submission['execution_inputs']['input_path']
        assert original.submission['registered_input']['resource_id'] == resource_id
        assert resumed.admitted['inputs']['expected_resource_identity']['resource_id'] == 'mouse-reviewed'
        run = harness.service._application.run_store.load(result['run_id'])
        embedded = next(s for s in run.steps if s.tool_name == 'epizoo_embed_cells')
        assert embedded.resolved_arguments['input_path'] == harness.service.resources.load(resource_id).source_path
        assert embedded.resolved_arguments['species'] == 'mouse'
        assert embedded.result['resource_provenance']['actual_resource_identity']['resource_id'] == 'mouse-reviewed'
        assert all(s.verification.passed for s in run.steps)
        selection_prompt = next(c for c in harness.models[-1].calls if 'output_selection_schema_version' in c)
        assert selection_prompt['question'] == objective
        evidence, artifacts, report = analysis.accepted_result(client, result)
        report_handle = next(a['handle'] for a in artifacts if a['artifact_type'] == 'analysis_report')
        downloaded = client.get(f"/api/v1/sessions/session/revisions/{result['revision_id']}/artifacts/{report_handle}",
            params={'download': 'true'})
        assert downloaded.content == report
        assert downloaded.headers['Content-Disposition'].startswith('attachment;')
        assert_private([pending, result, evidence, artifacts], tmp_path)
        assert str(tmp_path).encode() not in report
        history = client.get('/api/v1/sessions/session').json()
        assert [t['utterance'] for t in history['turns']] == [objective, 'Mouse.']
        assert client.get('/api/v1/resources').json()['choices'][0]['resource_id'] == resource_id


def test_continuation_preserves_previously_valid_typed_companions(tmp_path, monkeypatch):
    settings = dict(n_neighbors=7, metric='cosine',
        resolution=.7, min_dist=.2, spread=1.2)
    harness, inputs = handoff_harness(tmp_path, monkeypatch, companions=settings)
    with TestClient(web(harness, inputs)) as client:
        client.post('/api/v1/sessions', json={'session_id': 'session'})
        request_species(client, harness, objective='Analyze the selected H5AD.')
        result, _ = reply(client)
        assert result['status'] == 'succeeded', result
        resumed = harness.service._application.sessions.load('session').interactions[-1]
        assert {k: resumed.admitted['inputs'][k] for k in settings} == settings
        assert harness.inference_calls[0]['device'] == 'cpu'
        run = harness.service._application.run_store.load(result['run_id'])
        assert next(s for s in run.steps if s.tool_name == 'cluster_cells').result['resolution'] == .7


@pytest.mark.parametrize('inspection', [False, True])
def test_complete_inputs_and_inspection_keep_their_normal_path(tmp_path, monkeypatch, inspection):
    harness, inputs = handoff_harness(tmp_path, monkeypatch)
    if not inspection:
        inputs = (analysis.companion(),)
    with TestClient(web(harness, inputs)) as client:
        client.post('/api/v1/sessions', json={'session_id': 'session'})
        resource = upload(client, harness.source.read_bytes())
        result, _ = analysis.scientific_submit(client, resource['resource_id'],
            utterance='Inspect the selected H5AD.' if inspection else 'Embed the selected H5AD.',
            input_set_id='declarations' if inspection else 'mouse')
        assert result['status'] == 'succeeded', result
        assert harness.service._application.sessions.load('session').interactions[0].prerequisite is None
        assert harness.science_calls == (['inspect_scATAC'] if inspection else ['epizoo_embed_cells'])


def test_pending_species_survives_refresh_and_fresh_application_restart(tmp_path, monkeypatch):
    harness, inputs = handoff_harness(tmp_path, monkeypatch)
    with TestClient(web(harness, inputs)) as client:
        client.post('/api/v1/sessions', json={'session_id': 'session'})
        pending, _, _ = request_species(client, harness)
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
        assert client.get('/api/v1/sessions/session/turns/first').json() == pending
        assert snapshot(tmp_path) == before and len(harness.models) == calls
        result, _ = reply(client)
        assert result['status'] == 'succeeded' and len(harness.inference_calls) == 1, result


def test_same_answer_delivery_is_idempotent_and_completed_work_is_not_replayed(tmp_path, monkeypatch):
    harness, inputs = handoff_harness(tmp_path, monkeypatch)
    with TestClient(web(harness, inputs)) as client:
        client.post('/api/v1/sessions', json={'session_id': 'session'})
        request_species(client, harness)
        result, payload = reply(client)
        assert result['status'] == 'succeeded', result
        calls = (len(harness.models), len(harness.science_calls), len(harness.inference_calls))
        assert client.post('/api/v1/sessions/session/turns', json=payload).status_code == 202
        assert wait_turn(client, 'answer') == result
        assert calls == (len(harness.models), len(harness.science_calls), len(harness.inference_calls))
        # Direct Session retries compare the received empty inputs with the
        # received submission, rather than with newly completed scientific pins.
        sessions = harness.service._application.sessions
        answered = sessions.load('session').interactions[-1]
        provider_calls = len(harness.models[-1].calls)
        direct = sessions.respond('session', 'answer', 'Mouse.', interpreter=harness.models[-1],
            expected_generation=0, execution_inputs={}, submission=_serialize(answered.submission))
        assert direct.status == 'activated'
        assert len(harness.models[-1].calls) == provider_calls
        assert calls == (len(harness.models), len(harness.science_calls), len(harness.inference_calls))
        repeated, _ = reply(client, turn='repeated', generation=1)
        assert repeated['response']['kind'] == 'clarify' and repeated['revision_id'] is None
        assert len(harness.science_calls) == 1 and len(harness.inference_calls) == 1
        assert len(harness.service._application.sessions.load('session').revisions) == 1
    # A new HTTP worker has no transient receipt. The persisted answer owns
    # idempotency even when the accepted input/model bytes are no longer present.
    registered = harness.service._application.sessions.load('session').interactions[0].submission['registered_input']
    Path(harness.service.resources.load(registered['resource_id']).source_path).unlink()
    harness.checkpoint.unlink()
    with TestClient(web(harness, inputs)) as client:
        assert client.post('/api/v1/sessions/session/turns', json=payload).status_code == 202
        assert wait_turn(client, 'answer') == result
        assert (len(harness.models), len(harness.science_calls), len(harness.inference_calls)) == (
            calls[0] + 1, calls[1], calls[2])


def test_ambiguous_answer_keeps_exact_pending_request_for_valid_followup(tmp_path, monkeypatch):
    harness, inputs = handoff_harness(tmp_path, monkeypatch,
        replies={'I am not sure.': dict(kind='clarify', reason='ambiguous_species')})
    with TestClient(web(harness, inputs)) as client:
        client.post('/api/v1/sessions', json={'session_id': 'session'})
        _, _, resource_id = request_species(client, harness)
        uncertain, _ = reply(client, utterance='I am not sure.', turn='uncertain')
        assert uncertain['response']['clarification']['reason'] == 'ambiguous_species'
        assert_closed(harness, uncertain)
        result, _ = reply(client)
        assert result['status'] == 'succeeded', result
        assert harness.service._application.sessions.load('session').interactions[-1].admitted['inputs']['input_path'] == harness.service.resources.load(resource_id).source_path


@pytest.mark.parametrize('utterance', ['Cancel that request.', 'What time is it?'])
def test_cancellation_and_unrelated_turns_do_not_resume_old_science(tmp_path, monkeypatch, utterance):
    harness, inputs = handoff_harness(tmp_path, monkeypatch,
        replies={utterance: dict(kind='clarify', reason='unsupported_intent')})
    with TestClient(web(harness, inputs)) as client:
        client.post('/api/v1/sessions', json={'session_id': 'session'})
        request_species(client, harness)
        stopped, _ = reply(client, utterance=utterance, turn='other')
        assert_closed(harness, stopped)
        attempted, _ = reply(client)
        assert_closed(harness, attempted)
        assert not harness.science_calls


def test_new_objective_is_planned_as_new_request_and_does_not_resume_embedding(tmp_path, monkeypatch):
    harness, inputs = handoff_harness(tmp_path, monkeypatch)
    with TestClient(web(harness, inputs)) as client:
        client.post('/api/v1/sessions', json={'session_id': 'session'})
        _, _, resource_id = request_species(client, harness)
        inspected, _ = reply(client, utterance='Inspect the selected H5AD instead.', turn='changed',
            resource_id=resource_id, input_set_id='declarations')
        assert inspected['status'] == 'succeeded', inspected
        assert harness.science_calls == ['inspect_scATAC'] and not harness.inference_calls
        attempted, _ = reply(client, generation=1)
        assert attempted['revision_id'] is None and not harness.inference_calls
        assert len(harness.service._application.sessions.load('session').revisions) == 1


def test_changing_selected_dataset_cannot_apply_old_species_answer(tmp_path, monkeypatch):
    harness, inputs = handoff_harness(tmp_path, monkeypatch)
    with TestClient(web(harness, inputs)) as client:
        client.post('/api/v1/sessions', json={'session_id': 'session'})
        request_species(client, harness)
        other = upload(client, harness.source.read_bytes(), upload_id='other-upload', filename='Other dataset.h5ad')
        attempted, _ = reply(client, resource_id=other['resource_id'], input_set_id='declarations')
        assert_closed(harness, attempted)
        assert not harness.science_calls


def test_answer_conflicting_with_current_typed_species_fails_closed(tmp_path, monkeypatch):
    harness, inputs = handoff_harness(tmp_path, monkeypatch)
    inputs += (ScientificInputSet('human', 'Explicit human declaration',
        dict(species='human', device='cpu'), h5ad_companion=True),)
    with TestClient(web(harness, inputs)) as client:
        client.post('/api/v1/sessions', json={'session_id': 'session'})
        _, _, resource_id = request_species(client, harness)
        result, _ = reply(client, resource_id=resource_id, input_set_id='human')
        assert_closed(harness, result)
        assert result['response']['clarification']['reason'] == 'conflicting_species', result
        assert not harness.science_calls


def test_matching_typed_mouse_and_explicit_model_selection_can_complete_pending_request(tmp_path, monkeypatch):
    harness, inputs = handoff_harness(tmp_path, monkeypatch)
    inputs += (analysis.companion(),)
    resources = (harness.resource(), harness.resource('mouse-explicit', default=False))
    with TestClient(web(harness, inputs, resources=resources)) as client:
        client.post('/api/v1/sessions', json={'session_id': 'session'})
        _, _, resource_id = request_species(client, harness)
        result, _ = reply(client, resource_id=resource_id, input_set_id='mouse',
            epizoo_resource_id='mouse-explicit')
        assert result['status'] == 'succeeded', result
        continued = harness.service._application.sessions.load('session').interactions[-1]
        assert continued.admitted['inputs']['species'] == 'mouse'
        assert continued.admitted['inputs']['expected_resource_identity']['resource_id'] == 'mouse-explicit'
        assert harness.science_calls == ['epizoo_embed_cells'] and len(harness.inference_calls) == 1


@pytest.mark.parametrize('human', [False, True])
@pytest.mark.parametrize('transport', ['direct', 'web'])
def test_explicit_resource_pins_captured_before_species_are_preserved_or_rejected(tmp_path, monkeypatch, human, transport):
    harness, inputs = handoff_harness(tmp_path, monkeypatch)
    pinned = harness.resource(default=False)
    resources = (pinned, harness.resource('different-default'))
    with TestClient(web(harness, inputs, resources=resources)) as client:
        client.post('/api/v1/sessions', json={'session_id': 'session'})
        resource = upload(client, harness.source.read_bytes())
        if transport == 'web':
            pending, _ = analysis.scientific_submit(client, resource['resource_id'],
                input_set_id='declarations', epizoo_resource_id=pinned.resource_id)
        else:
            # Direct registered composition and Web resource selection both
            # preserve exact pins without declaring or inferring a species.
            binding = harness.service.resources.compose_h5ad(resource['resource_id'],
                dict(device='cpu') | pinned.inputs())
            pending = harness.service.submit_turn('session', 'first', 'Embed the selected H5AD.',
                expected_generation=0, registered_input=binding, epizoo_resources=resources).to_dict()
        assert pending['response']['clarification']['reason'] == 'missing_species'
        original = harness.service._application.sessions.load('session').interactions[0]
        assert 'species' not in original.submission['execution_inputs']
        result, _ = reply(client, utterance="It's human." if human else 'Mouse.')
        if human:
            assert_closed(harness, result)
            assert result['error']['code'] == 'EPIZOO_RESOURCE_SELECTION_INVALID', result
            assert not harness.science_calls
        else:
            assert result['status'] == 'succeeded', result
            continued = harness.service._application.sessions.load('session').interactions[-1]
            for key, value in original.submission['execution_inputs'].items():
                assert continued.admitted['inputs'][key] == value
            assert continued.admitted['inputs']['expected_resource_identity']['resource_id'] == 'mouse-reviewed'
            assert len(harness.inference_calls) == 1


def test_corrupt_persisted_continuation_cannot_change_tool_source_companions_or_registration(tmp_path, monkeypatch):
    harness, inputs = handoff_harness(tmp_path, monkeypatch)
    with TestClient(web(harness, inputs)) as client:
        client.post('/api/v1/sessions', json={'session_id': 'session'})
        request_species(client, harness)
        result, _ = reply(client)
        assert result['status'] == 'succeeded', result
        sessions = harness.service._application.sessions
        persisted = sessions._store._path('session', '.json')
        accepted_bytes = persisted.read_bytes()
        calls = (len(harness.models), len(harness.science_calls), len(harness.inference_calls))
        for field in ('tool', 'source', 'companion', 'registration'):
            corrupt = json.loads(accepted_bytes)
            continued = corrupt['record']['interactions'][-1]['admitted']
            if field == 'tool':
                continued['tool'] = 'inspect_scATAC'
            elif field == 'source':
                continued['inputs']['input_path'] = str(tmp_path / 'another.h5ad')
            elif field == 'companion':
                continued['inputs']['device'] = 'cuda'
            else:
                continued['registered_input']['record_sha256'] = 'f' * 64
            # A valid outer checksum cannot conceal changed scientific bindings.
            corrupt['sha256'] = digest(corrupt['record'])
            try:
                persisted.write_text(json.dumps(corrupt))
                with pytest.raises(SessionError, match='Continuation changed its original scientific request'):
                    sessions.load('session')
            finally:
                persisted.write_bytes(accepted_bytes)
        assert persisted.read_bytes() == accepted_bytes
        assert calls == (len(harness.models), len(harness.science_calls), len(harness.inference_calls))


@pytest.mark.parametrize('decision,reason', [
    (species_decision(pending='invented'), 'invalid_prerequisite'),
    (species_decision('macaque'), 'invalid_decision'),
    (dict(kind='clarify', reason='unsupported_species'), 'unsupported_species'),
])
def test_invalid_reference_or_unsupported_species_never_reaches_science(tmp_path, monkeypatch, decision, reason):
    harness, inputs = handoff_harness(tmp_path, monkeypatch, replies={'Given answer.': decision})
    with TestClient(web(harness, inputs)) as client:
        client.post('/api/v1/sessions', json={'session_id': 'session'})
        request_species(client, harness)
        result, _ = reply(client, utterance='Given answer.')
        assert result['response']['clarification']['reason'] == reason, result
        assert_closed(harness, result)
        assert not harness.science_calls


def test_human_answer_reports_missing_qualified_resource_without_mouse_fallback(tmp_path, monkeypatch):
    harness, inputs = handoff_harness(tmp_path, monkeypatch)
    with TestClient(web(harness, inputs)) as client:
        client.post('/api/v1/sessions', json={'session_id': 'session'})
        request_species(client, harness)
        result, _ = reply(client, utterance="It's human.")
        assert_closed(harness, result)
        assert result['error']['code'] == 'EPIZOO_RESOURCE_REQUIRED', result
        assert not harness.science_calls
        assert result['response']['kind'] != 'clarify'


def test_changed_qualified_catalog_cannot_redirect_pending_request(tmp_path, monkeypatch):
    harness, inputs = handoff_harness(tmp_path, monkeypatch)
    with TestClient(web(harness, inputs)) as client:
        client.post('/api/v1/sessions', json={'session_id': 'session'})
        request_species(client, harness)
    changed = replace(harness.resource(), checkpoint_sha256='d' * 64)
    with TestClient(web(harness, inputs, resources=(changed,))) as client:
        result, _ = reply(client)
        assert_closed(harness, result)
        assert not harness.science_calls


@pytest.mark.parametrize('mutation', ['changed', 'missing'])
def test_original_registered_bytes_are_rechecked_on_conversational_continuation(tmp_path, monkeypatch, mutation):
    harness, inputs = handoff_harness(tmp_path, monkeypatch)
    with TestClient(web(harness, inputs)) as client:
        client.post('/api/v1/sessions', json={'session_id': 'session'})
        _, _, resource_id = request_species(client, harness)
        source = Path(harness.service.resources.load(resource_id).source_path)
        model_type = type(harness.models[-1])
        original_complete = model_type.complete
        triggered = []

        def mutate_after_provider(self, *, prompt, response_schema):
            value = json.loads(prompt)
            completed = original_complete(self, prompt=prompt, response_schema=response_schema)
            if (mutation == 'changed' and value.get('utterance') == 'Mouse.'
                    or mutation == 'missing' and 'output_selection_schema_version' in value):
                triggered.append(mutation)
                if mutation == 'changed':
                    source.write_bytes(b'Changed after the interpreter reply.')
                else:
                    source.unlink()
            return completed

        # Recheck after semantic capture and after the final provider call;
        # source identity cannot become stale while awaiting language decisions.
        monkeypatch.setattr(model_type, 'complete', mutate_after_provider)
        result, _ = reply(client)
        assert triggered == [mutation]
        assert_closed(harness, result)
        assert result['error']['code'] == 'LOCAL_RESOURCE_INTEGRITY_INVALID', result
        assert not harness.science_calls


def test_generation_conflict_and_navigation_never_redirect_pending_source(tmp_path, monkeypatch):
    harness, inputs = handoff_harness(tmp_path, monkeypatch)
    with TestClient(web(harness, inputs)) as client:
        client.post('/api/v1/sessions', json={'session_id': 'session'})
        resource = upload(client, harness.source.read_bytes())
        first, _ = analysis.scientific_submit(client, resource['resource_id'],
            utterance='Inspect the selected H5AD.', input_set_id='declarations')
        assert first['status'] == 'succeeded'
        request_species(client, harness, resource_id=resource['resource_id'], turn='pending', generation=1)
        calls = len(harness.models)
        stale = client.post('/api/v1/sessions/session/turns', json=body(turn='stale', utterance='Mouse.'))
        assert stale.status_code == 409 and stale.json()['error']['code'] == 'INTERACTIVE_GENERATION_CONFLICT'
        assert len(harness.models) == calls
        activated = client.post('/api/v1/sessions/session/activate', json=dict(
            turn_id='switch', expected_generation=1, revision_id=first['revision_id']))
        assert activated.status_code == 200, activated.text
        attempted, _ = reply(client, generation=2)
        assert attempted['revision_id'] is None and not harness.inference_calls
        assert harness.science_calls == ['inspect_scATAC']
        assert len(harness.service._application.sessions.load('session').revisions) == 1
