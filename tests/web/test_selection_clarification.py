"""The existing HTTP conversation carries contract-grounded selection answers.

Only model decisions are scripted; tiny qualified QC, selection production,
verification, persistence, evidence and the HTTP worker use production code.
"""
import json
from pathlib import Path
import sys

from fastapi.testclient import TestClient

from agent.web.app import create_app
from agent.web.config import ScientificInputSet

sys.path.insert(0, str(Path(__file__).parents[1] / 'application'))
from test_scientific_parameter_continuation import (
    runtime, fixture_factory, qc_case, selection_inputs, selection_harness,
    REQUIRED, OBJECTIVE, BOTH, DEPTH, TSS, BAD, assert_pending, accepted_step,
)
from helpers import wait_turn


def web(harness, inputs):
    return create_app(harness.service, input_sets=(
        ScientificInputSet('verified-qc', 'Verified tiny barcode QC', inputs),))


def payload(turn='first', utterance=OBJECTIVE, *, generation=0, selected=False):
    body = dict(turn_id=turn, expected_generation=generation, utterance=utterance)
    if selected:
        body['input_set_id'] = 'verified-qc'
    return body


def post(client, body):
    response = client.post('/api/v1/sessions/session/turns', json=body)
    assert response.status_code == 202, response.text
    return wait_turn(client, body['turn_id'], timeout=90)


def test_first_selection_chat_requests_only_required_thresholds_and_runs_verified_owner(
        tmp_path, selection_inputs):
    harness = selection_harness(tmp_path / 'workspace')
    with TestClient(web(harness, selection_inputs)) as client:
        assert client.post('/api/v1/sessions', json={'session_id': 'session'}).status_code in (200, 201)
        assert client.get('/api/v1/input-sets').json() == {'choices': [
            dict(input_set_id='verified-qc', display_label='Verified tiny barcode QC')]}
        assert not harness.calls and not harness.constructions
        pending = post(client, payload(selected=True))
        assert_pending(pending, *REQUIRED)
        answer_body = payload('answer', BOTH)
        completed = post(client, answer_body)
        accepted_step(harness, completed)
        assert len(harness.service._application.sessions.load('session').revisions) == 1
        evidence = client.get(f"/api/v1/sessions/session/revisions/{completed['revision_id']}/evidence",
            params={'output_name': 'selection'})
        assert evidence.status_code == 200, evidence.text
        facts = {fact['field']: fact for fact in evidence.json()['facts']}
        assert facts['effective_thresholds']['value']['min_qc_fragment_records'] == 1
        assert facts['effective_thresholds']['value']['min_tss_flank_evidence'] is None
        assert facts['effective_thresholds']['value']['min_tss_enrichment'] == dict(numerator=1, denominator=2)
        assert facts['selection_threshold_origins']['value']['min_tss_flank_evidence'] == 'existing_owner_default'
        assert facts['explicit_user_parameters']['value']['parameters'] == {
            name: 'explicit_user_instruction' for name in REQUIRED}
        assert facts['explicit_user_parameters']['value']['tool'] == 'select_scATAC_cells'
        assert facts['explicit_user_parameters']['value']['step_id'] == 'selection'
        assert str(tmp_path) not in json.dumps([pending, completed, evidence.json()])
        counts = len(harness.calls), len(harness.constructions)
        assert post(client, answer_body) == completed
        assert counts == (len(harness.calls), len(harness.constructions))


def test_partial_invalid_threshold_answer_refresh_and_fresh_server_resume_original_task(
        tmp_path, selection_inputs):
    workspace = tmp_path / 'workspace'
    harness = selection_harness(workspace)
    with TestClient(web(harness, selection_inputs)) as client:
        client.post('/api/v1/sessions', json={'session_id': 'session'})
        pending = post(client, payload(selected=True))
        partial = post(client, payload('partial', DEPTH))
        assert_pending(partial, REQUIRED[1])
        invalid = post(client, payload('invalid', BAD))
        assert invalid['response']['clarification']['reason'] == 'invalid_parameter_value', invalid
        assert client.get('/api/v1/sessions/session/turns/first').json() == pending
        assert client.get('/api/v1/sessions/session/turns/partial').json() == partial
        assert not harness.service._application.sessions.load('session').revisions
    restarted = selection_harness(workspace)
    with TestClient(web(restarted, selection_inputs)) as client:
        assert client.get('/api/v1/sessions/session/turns/first').json() == pending
        assert client.get('/api/v1/sessions/session/turns/partial').json() == partial
        assert client.get('/api/v1/sessions/session/turns/invalid').json() == invalid
        assert not restarted.calls and not restarted.constructions
        completed = post(client, payload('remaining', TSS))
        accepted_step(restarted, completed)
        run = restarted.service._application.run_store.load(completed['run_id'])
        assert run.request.prompt == OBJECTIVE
        assert run.steps[0].resolved_arguments['barcode_qc_manifest_sha256'] == selection_inputs['barcode_qc_manifest_sha256']
        assert client.get('/api/v1/sessions/session').json()['generation'] == 1


def test_stale_web_response_is_rejected_before_model_construction(tmp_path, selection_inputs):
    harness = selection_harness(tmp_path / 'workspace')
    with TestClient(web(harness, selection_inputs)) as client:
        client.post('/api/v1/sessions', json={'session_id': 'session'})
        pending = post(client, payload(selected=True))
        calls = len(harness.calls), len(harness.constructions)
        stale = client.post('/api/v1/sessions/session/turns',
            json=payload('stale', BOTH, generation=1))
        assert stale.status_code == 409, stale.text
        assert stale.json()['error']['code'] == 'INTERACTIVE_GENERATION_CONFLICT'
        assert calls == (len(harness.calls), len(harness.constructions))
        assert client.get('/api/v1/sessions/session/turns/first').json() == pending
        assert not harness.service._application.sessions.load('session').revisions
