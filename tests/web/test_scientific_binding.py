"""Generic declarations resume species clarification through ordinary Web science."""
from fastapi.testclient import TestClient
import pytest

import test_h5ad_analysis as analysis
from test_leiden_resolution import (
    EXPLICIT, OMITTED, argument, resolution_harness, start, steps,
)
from test_species_handoff import reply, web


ANSWER = 'This dataset is mouse; use resolution 0.7 and UMAP minimum distance 0.2.'


def species_answer(utterance=ANSWER):
    return dict(kind='answer_prerequisite', pending='@species', species='mouse', arguments=[
        argument(utterance, '0.7'),
        argument(utterance, '0.2', tool='compute_cell_umap', argument='min_dist'),
    ])


@pytest.mark.parametrize('original_explicit', [False, True])
def test_generic_species_reply_preserves_or_corrects_parameters_then_runs_existing_owners(
        tmp_path, monkeypatch, original_explicit):
    initial = EXPLICIT.replace('0.7', '0.6') if original_explicit else OMITTED
    decisions = {ANSWER: species_answer()}
    if original_explicit:
        decisions[initial] = dict(kind='execute_plan', target='compute_cell_umap', argument=argument(initial, '0.6'))
    harness, inputs = resolution_harness(tmp_path, monkeypatch, missing_species=True,
        decisions=decisions)
    with TestClient(web(harness, inputs)) as client:
        client.post('/api/v1/sessions', json={'session_id': 'session'})
        pending, _, resource_id = start(client, harness, utterance=initial)
        assert pending['response']['clarification']['reason'] == 'missing_species'
        assert not harness.science_calls and not harness.inference_calls
        result, payload = reply(client, utterance=ANSWER)
        assert result['status'] == 'succeeded' and result['response']['status'] == 'activated', result
        scientific = steps(harness, result)
        assert [step.tool_name for step in scientific] == ['inspect_scATAC', 'epizoo_embed_cells',
            'build_cell_neighbors', 'cluster_cells', 'compute_cell_umap']
        assert all(step.verification.passed for step in scientific)
        clustered, umap = scientific[-2:]
        assert clustered.resolved_arguments['resolution'] == clustered.result['resolution'] == .7
        assert umap.resolved_arguments['min_dist'] == umap.result['min_dist'] == .2
        assert umap.result['spread'] == 1.0 and 'spread' not in umap.resolved_arguments
        state = harness.service._application.sessions.load('session')
        first, resumed = state.interactions
        if original_explicit:
            assert first.admitted['inputs']['resolution'] == .6
        assert resumed.admitted['inputs']['resolution'] == .7
        assert resumed.admitted['inputs']['min_dist'] == .2
        assert resumed.admitted['registered_input']['resource_id'] == first.submission['registered_input']['resource_id'] == resource_id
        assert resumed.admitted['inputs']['input_path'] == first.submission['execution_inputs']['input_path']
        run = harness.service._application.run_store.load(result['run_id'])
        assert run.request.prompt == initial
        evidence, _, _ = analysis.accepted_result(client, result)
        facts = {item['field']: item['value'] for item in evidence['facts'] if item['status'] == 'available'}
        assert facts['clustering_parameters']['resolution'] == .7
        assert facts['clustering_resolution_origin'] == 'explicit_execution_argument'
        counts = len(harness.models), len(harness.science_calls), len(harness.inference_calls)
        assert client.post('/api/v1/sessions/session/turns', json=payload).status_code == 202
        from helpers import wait_turn
        assert wait_turn(client, 'answer') == result
        assert counts == (len(harness.models), len(harness.science_calls), len(harness.inference_calls))


def test_species_reply_cannot_add_an_operation_to_bind_an_unrelated_parameter(tmp_path, monkeypatch):
    utterance = 'This dataset is mouse; set minimum TSS enrichment to 0.5.'
    decision = dict(kind='answer_prerequisite', pending='@species', species='mouse', arguments=[
        argument(utterance, '0.5', tool='select_scATAC_cells', argument='min_tss_enrichment')])
    harness, inputs = resolution_harness(tmp_path, monkeypatch, missing_species=True,
        decisions={utterance: decision})
    with TestClient(web(harness, inputs)) as client:
        client.post('/api/v1/sessions', json={'session_id': 'session'})
        pending, _, _ = start(client, harness, utterance=OMITTED)
        assert pending['response']['clarification']['reason'] == 'missing_species'
        result, _ = reply(client, utterance=utterance)
        assert result['revision_id'] is None and result['status'] != 'succeeded', result
        assert not harness.science_calls and not harness.inference_calls
        assert not harness.service._application.sessions.load('session').revisions
