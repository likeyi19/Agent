"""Rich HTTP interaction reuses accepted science, exact evidence and M17.2."""
import hashlib
import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from agent.application import InteractiveAgentApplication
from agent.providers import PlanningModelFactoryRegistry
from agent.schemas import PriorOutputRef
from agent.web.app import create_app
from helpers import wait_turn
from rich_helpers import PROFILES, rich_harness


@pytest.fixture
def rich(tmp_path):
    harness = rich_harness(tmp_path)
    try:
        with TestClient(create_app(harness.service, input_sets=harness.input_sets)) as client:
            yield client, harness
    finally:
        harness.close()


def revision_path(revision, *, session='analysis'):
    return f'/api/v1/sessions/{session}/revisions/{revision}'


def submit(client, turn, generation, utterance, **kwargs):
    response = client.post('/api/v1/sessions/analysis/turns', json=dict(
        turn_id=turn, expected_generation=generation, utterance=utterance, **kwargs))
    assert response.status_code == 202, response.text
    return wait_turn(client, turn, session='analysis')


def activate(client, turn, generation, revision):
    return client.post('/api/v1/sessions/analysis/activate', json=dict(
        turn_id=turn, expected_generation=generation, revision_id=revision))


def error(response, code, status):
    assert response.status_code == status, response.text
    value = response.json()
    assert set(value) == {'error'} and set(value['error']) == {'code', 'message'}
    assert value['error']['code'] == code


def assert_client_payload(value, private_root):
    assert str(private_root) not in json.dumps(value)
    forbidden = {'authority_payload', '$prior_output', '$ref', 'resolved_arguments',
                 'execution_inputs', 'workspace_root', 'provider_prompt', 'credentials', 'api_key'}
    def inspect(item):
        if isinstance(item, dict):
            assert not forbidden.intersection(item)
            for child in item.values():
                inspect(child)
        elif isinstance(item, list):
            for child in item:
                inspect(child)
    inspect(value)


def test_revision_evidence_histogram_and_artifact_reads_do_no_science(rich, tmp_path):
    client, harness = rich
    before = harness.work.snapshot()
    path = revision_path(harness.initial_revision)
    revision = client.get(path)
    assert revision.status_code == 200, revision.text
    view = revision.json()
    assert view['revision_id'] == harness.initial_revision and view['is_active']
    assert view['parent_revision_id'] is None and view['run_id'] == 'seed-request:run'
    assert set(view['outputs']) == {'fragments', 'qc', 'selection', 'matrix'}
    assert view['session_generation'] == 1
    evidence = client.get(path + '/evidence', params={'output_name': 'qc'})
    assert evidence.status_code == 200, evidence.text
    source = evidence.json()
    assert source['status'] == 'available' and source['limitations']
    assert source['source']['verification_checks']
    assert source['source']['authority_sha256']
    assert 'length_histogram' in source['supported_details']
    detailed = client.get(path + '/evidence', params={'output_name': 'qc', 'detail_section': 'length_histogram', 'limit': 3})
    assert detailed.status_code == 200, detailed.text
    detail = detailed.json()
    assert detail['status'] == 'available'
    assert [f['field'] for f in detail['detail']] == [
        'length_histogram.1', 'length_histogram.2', 'length_histogram.3', 'length_histogram.records_omitted']
    unsupported = client.get(path + '/evidence', params={'output_name': 'qc', 'detail_section': 'unsupported-science'})
    assert unsupported.status_code == 200
    assert unsupported.json()['status'] == 'unsupported'
    artifacts = client.get(path + '/artifacts')
    assert artifacts.status_code == 200 and artifacts.json()['artifacts']
    assert harness.work.snapshot() == before and not harness.models
    assert_client_payload([view, source, detail, artifacts.json()], tmp_path)
    for fact in source['facts'] + detail['detail']:
        if fact['field'] == 'path' or fact['field'].endswith('_path'):
            assert fact['status'] != 'available' and fact['value'] is None


def test_report_delivery_is_pinned_safe_and_downloadable(rich, tmp_path):
    client, harness = rich
    before = harness.work.snapshot()
    path = revision_path(harness.initial_revision)
    artifacts = client.get(path + '/artifacts').json()['artifacts']
    report = next(a for a in artifacts if a['artifact_type'] == 'analysis_report')
    url = path + '/artifacts/' + report['handle']
    delivered = client.get(url)
    assert delivered.status_code == 200, delivered.text
    assert delivered.headers['Content-Type'].startswith('text/plain')
    assert delivered.headers['X-Artifact-Source-SHA256'] == report['sha256']
    assert delivered.headers['X-Artifact-Content-SHA256'] == hashlib.sha256(delivered.content).hexdigest()
    assert delivered.content and str(tmp_path).encode() not in delivered.content
    assert 'authority_payload' not in delivered.text
    assert delivered.headers['X-Artifact-Presentation'] == 'client_safe_report_projection'
    downloaded = client.get(url, params={'download': 'true'})
    assert downloaded.content == delivered.content
    assert downloaded.headers['Content-Disposition'].startswith('attachment;')
    assert str(tmp_path) not in downloaded.headers['Content-Disposition']
    error(client.get(path + '/artifacts/' + '0' * 64), 'INTERACTIVE_ARTIFACT_UNAVAILABLE', 404)
    error(client.get(revision_path('unknown') + '/artifacts/' + report['handle']), 'INTERACTIVE_REFERENCE_INVALID', 404)
    assert harness.work.snapshot() == before and not harness.models


def test_guidance_uses_normal_planner_and_navigation_creates_sibling_branch(rich):
    client, harness = rich
    initial = harness.initial_revision
    before = harness.work.snapshot()
    guide = submit(client, 'guide', 1, 'What could I analyze next using the QC result?')
    assert guide['response']['kind'] == 'answer'
    candidate = guide['response']['guidance']['candidates'][0]
    assert candidate['candidate_id'] and candidate['option'] == 1
    assert candidate['capability'] == 'select_scATAC_cells'
    assert harness.work.snapshot() == before
    original_guide = guide['response']
    selected = submit(client, 'selected', 1, 'Run option 1.', predecessor_turn_id='guide', input_set_id='thresholds')
    assert selected['status'] == 'succeeded' and selected['revision_id'] != initial
    first_child = selected['revision_id']
    selected_run = harness.application.run_store.load(selected['run_id'])
    assert [s.tool_name for s in selected_run.plan.steps] == ['select_scATAC_cells']
    refs = [v for v in selected_run.plan.steps[0].arguments.values() if isinstance(v, PriorOutputRef)]
    assert len(refs) == 2 and refs[0].binding == refs[1].binding
    assert refs[0].binding.run_id == 'seed-request:run' and refs[0].binding.step_id == 'qc'
    assert set(client.get(revision_path(first_child)).json()['outputs']) == {'selection'}
    after = harness.work.snapshot()
    assert {key: after.get(key, 0) - before.get(key, 0) for key in after if after.get(key, 0) != before.get(key, 0)} == {
        'production.selection': 1, 'owner.selection': 1}
    selected_model = harness.models[-1]
    assert any('selection_schema_version' in c for c in selected_model.calls)
    assert any('active_outputs' in c for c in selected_model.calls)
    assert any('output_selection_schema_version' in c for c in selected_model.calls)
    stale = submit(client, 'stale', 2, 'Run option 1.', predecessor_turn_id='guide', input_set_id='thresholds')
    assert stale['response']['kind'] == 'clarify'
    assert harness.work.snapshot() == after
    models_before = len(harness.models)
    switched = activate(client, 'back', 2, initial)
    assert switched.status_code == 200, switched.text
    state = switched.json()
    assert state['generation'] == 3 and state['active_revision_id'] == initial
    assert len(harness.models) == models_before and harness.work.snapshot() == after
    branch = submit(client, 'branch', 3, 'Run option 1.', predecessor_turn_id='guide', input_set_id='thresholds', profile_id='beta')
    assert branch['status'] == 'succeeded' and branch['revision_id'] not in {initial, first_child}
    final = client.get('/api/v1/sessions/analysis').json()
    assert final['generation'] == 4
    revisions = {r['revision_id']: r for r in final['revisions']}
    assert revisions[first_child]['parent_revision_id'] == initial
    assert revisions[branch['revision_id']]['parent_revision_id'] == initial
    assert not revisions[first_child]['is_active'] and revisions[branch['revision_id']]['is_active']
    assert client.get('/api/v1/sessions/analysis/turns/guide').json()['response'] == original_guide
    total = harness.work.snapshot()
    assert total['production.selection'] == after['production.selection'] + 1
    assert total['owner.selection'] == after['owner.selection'] + 1


def test_first_threshold_execution_retains_exact_qc_for_followup_guidance(rich):
    client, harness = rich
    before = harness.work.snapshot()
    selected = submit(client, 'first', 1, 'Set minimum depth to 1.')
    assert selected['status'] == 'succeeded' and selected['revision_id'] != harness.initial_revision
    view = client.get(revision_path(selected['revision_id'])).json()
    assert view['parent_revision_id'] == harness.initial_revision
    assert set(view['outputs']) == {'selection', 'fragments', 'qc'}
    after = harness.work.snapshot()
    assert {key: after.get(key, 0) - before.get(key, 0) for key in after if after.get(key, 0) != before.get(key, 0)} == {
        'production.selection': 1, 'owner.selection': 1}
    discussed = submit(client, 'question', 2, 'What does this selection show?')
    assert discussed['response']['scientific']['support'] == 'supported'
    guide = submit(client, 'guide', 2, 'What could I analyze next using the QC result?')
    assert guide['response']['guidance']['candidates'][0]['capability'] == 'select_scATAC_cells'
    assert harness.work.snapshot() == after


def test_navigation_rejects_stale_conflict_unknown_and_internal_fields(rich):
    client, harness = rich
    before = harness.work.snapshot()
    error(activate(client, 'stale', 0, harness.initial_revision), 'INTERACTIVE_GENERATION_CONFLICT', 409)
    error(activate(client, 'missing', 1, 'unknown'), 'INTERACTIVE_REFERENCE_INVALID', 404)
    valid = activate(client, 'back', 1, harness.initial_revision)
    assert valid.status_code == 200 and valid.json()['generation'] == 2
    identical = activate(client, 'back', 1, harness.initial_revision)
    assert identical.status_code == 200 and identical.json()['generation'] == 2
    error(activate(client, 'back', 2, harness.initial_revision), 'INTERACTIVE_TURN_CONFLICT', 409)
    invalid = client.post('/api/v1/sessions/analysis/activate', json=dict(
        turn_id='invalid', expected_generation=2, revision_id=harness.initial_revision, authority_payload={}))
    assert invalid.status_code == 422
    assert harness.work.snapshot() == before and not harness.models


def test_missing_and_corrupt_evidence_have_explicit_unavailable_view(rich):
    client, harness = rich
    before = harness.work.snapshot()
    path = revision_path(harness.initial_revision)
    missing = client.get(path + '/evidence', params={'output_name': 'absent'})
    assert missing.status_code == 200 and missing.json()['status'] == 'unavailable'
    state = harness.application.sessions.load('analysis')
    source = next(Path(f.path) for f in state.turn('seed').completion_files if Path(f.path).name == 'analysis_evidence.json')
    source.write_bytes(source.read_bytes() + b' ')
    corrupted = client.get(path + '/evidence', params={'output_name': 'qc'})
    assert corrupted.status_code == 200 and corrupted.json()['status'] == 'unavailable'
    assert not corrupted.json()['facts']
    assert harness.work.snapshot() == before and not harness.models


def test_tampered_report_and_wrong_session_handle_fail_closed(rich):
    client, harness = rich
    before = harness.work.snapshot()
    path = revision_path(harness.initial_revision)
    report = next(a for a in client.get(path + '/artifacts').json()['artifacts'] if a['artifact_type'] == 'analysis_report')
    created = client.post('/api/v1/sessions', json={'session_id': 'other'})
    assert created.status_code == 201
    error(client.get(revision_path(harness.initial_revision, session='other') + '/artifacts/' + report['handle']),
        'INTERACTIVE_REFERENCE_INVALID', 404)
    state = harness.application.sessions.load('analysis')
    source = next(Path(f.path) for f in state.turn('seed').completion_files if Path(f.path).suffix == '.md')
    source.write_bytes(source.read_bytes() + b'unsafe changed report')
    error(client.get(path + '/artifacts/' + report['handle']), 'INTERACTIVE_ARTIFACT_UNAVAILABLE', 404)
    assert harness.work.snapshot() == before and not harness.models


def test_new_facade_reconstructs_guidance_revision_evidence_artifacts_without_provider(rich):
    client, harness = rich
    guide = submit(client, 'guide', 1, 'What could I analyze next using the QC result?')
    history = client.get('/api/v1/sessions/analysis').json()
    path = revision_path(harness.initial_revision)
    revision = client.get(path).json()
    evidence = client.get(path + '/evidence', params={'output_name': 'qc'}).json()
    artifacts = client.get(path + '/artifacts').json()
    before = harness.work.snapshot()
    def forbidden(profile):
        raise AssertionError('Rich historical reads constructed a provider.')
    reopened = InteractiveAgentApplication(harness.application.workspace_root, model_profiles=PROFILES,
        default_profile_id='alpha', planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': forbidden}))
    with TestClient(create_app(reopened, input_sets=harness.input_sets)) as fresh:
        assert fresh.get('/api/v1/sessions/analysis').json() == history
        assert fresh.get(path).json() == revision
        assert fresh.get(path + '/evidence', params={'output_name': 'qc'}).json() == evidence
        assert fresh.get(path + '/artifacts').json() == artifacts
        assert fresh.get('/api/v1/sessions/analysis/turns/guide').json()['response'] == guide['response']
    assert harness.work.snapshot() == before
