"""Accepted-plan edits use real execution owners; provider and EpiZoo are scripted."""
from copy import deepcopy
from fastapi.testclient import TestClient
from agent.schemas import PriorOutputRef
from agent.application.structured_parameters import result_parameters
from helpers import wait_turn
from test_leiden_resolution import resolution_harness, start
from test_species_handoff import web


def test_edit_composed_result_reuses_upstream_and_reopens(tmp_path, monkeypatch):
    harness, inputs = resolution_harness(tmp_path, monkeypatch)
    with TestClient(web(harness, inputs)) as client:
        client.post('/api/v1/sessions', json={'session_id': 'session'})
        first, _, _ = start(client, harness)
        assert first['status'] == 'succeeded', first
        sessions = harness.service._application.sessions
        original = sessions.load('session')
        revision = original.revisions[-1]
        before_run = harness.service._application.run_store.load(revision.run_id).to_dict()
        calls = sum(len(m.calls) for m in harness.models)
        metadata = client.get(f'/api/v1/sessions/session/revisions/{revision.revision_id}/parameters')
        assert metadata.status_code == 200, metadata.text
        cluster = next(s for s in metadata.json()['steps'] if s['tool'] == 'cluster_cells')
        assert next(p for p in cluster['parameters'] if p['argument'] == 'resolution')['value'] == .7
        assert next(p for p in cluster['parameters'] if p['argument'] == 'random_seed')['value'] == 0
        assert sum(len(m.calls) for m in harness.models) == calls
        overrides = [dict(tool=cluster['tool'], step_id=cluster['step_id'], argument='resolution', value=.9)]
        body = dict(turn_id='edited', utterance='', expected_generation=original.generation,
            rerun_revision_id=revision.revision_id, parameter_overrides=overrides)
        response = client.post('/api/v1/sessions/session/turns', json=body)
        assert response.status_code == 202, response.text
        result = wait_turn(client, 'edited')
        assert result['status'] == 'succeeded', result
        assert result['utterance'] == '' and result['rerun_revision_id'] == revision.revision_id
        assert sum(len(m.calls) for m in harness.models) == calls
        state = sessions.load('session')
        assert len(state.revisions) == 2 and state.revisions[0] == revision
        assert harness.service._application.run_store.load(revision.run_id).to_dict() == before_run
        rerun = harness.service._application.run_store.load(state.revisions[-1].run_id)
        assert {s.tool_name for s in rerun.steps} == {'cluster_cells', 'compute_cell_umap'}
        edited = next(s for s in rerun.plan.steps if s.tool_name == 'cluster_cells')
        assert edited.arguments['resolution'] == .9
        assert any(isinstance(v, PriorOutputRef) for v in edited.arguments.values())
        assert state.revisions[-1].parent_revision_id == revision.revision_id
        reopened = client.get('/api/v1/sessions/session').json()
        assert reopened['active_revision_id'] == state.revisions[-1].revision_id
        assert client.post('/api/v1/sessions/session/activate', json=dict(turn_id='back',
            expected_generation=state.generation, revision_id=revision.revision_id)).status_code == 200
        assert sessions.load('session').active_revision_id == revision.revision_id
        assert sum(len(m.calls) for m in harness.models) == calls
        from test_leiden_resolution import QUESTION
        active = sessions.load('session')
        assert client.post('/api/v1/sessions/session/turns', json=dict(turn_id='question',
            utterance=QUESTION, expected_generation=active.generation)).status_code == 202
        answer = wait_turn(client, 'question')
        assert answer['response']['scientific']['availability'] == 'generated', answer
        assert '0.7' in answer['response']['text']
        after_question = sessions.load('session')
        assert after_question.active_revision_id == revision.revision_id
        assert after_question.interactions[-1].base_revision_id == revision.revision_id
        assert len(after_question.revisions) == 2


def test_repeated_operations_edit_only_the_selected_step(tmp_path, monkeypatch):
    from test_leiden_resolution import OMITTED
    harness, inputs = resolution_harness(tmp_path, monkeypatch, plan_variant='repeated-cluster')
    # Both clustering operations consume the accepted neighbor graph. A cluster
    # result is not itself a neighbor-graph producer.
    import test_h5ad_analysis as analysis
    import json
    original_complete = analysis.ScientificPlanModel.complete
    def independent_clusters(self, **kwargs):
        raw = original_complete(self, **kwargs)
        value = json.loads(raw)
        if value.get('schema_version') == 4:
            steps = value['decision']['steps']
            first = next(s for s in steps if s['step_id'] == 'cluster')
            next(s for s in steps if s['step_id'] == 'another-cluster')['sources'] = deepcopy(first['sources'])
            return json.dumps(value)
        return raw
    monkeypatch.setattr(analysis.ScientificPlanModel, 'complete', independent_clusters)
    with TestClient(web(harness, inputs)) as client:
        client.post('/api/v1/sessions', json={'session_id': 'session'})
        first, _, _ = start(client, harness, utterance=OMITTED)
        assert first['status'] == 'succeeded', first
        sessions = harness.service._application.sessions
        before = sessions.load('session')
        revision = before.revisions[-1]
        meta = result_parameters(sessions, 'session', revision.revision_id)
        assert {s['step_id'] for s in meta['steps'] if s['tool'] == 'cluster_cells'} == {'cluster', 'another-cluster'}
        calls = sum(len(m.calls) for m in harness.models)
        response = client.post('/api/v1/sessions/session/turns', json=dict(turn_id='second-edit',
            utterance='', expected_generation=before.generation, rerun_revision_id=revision.revision_id,
            parameter_overrides=[dict(tool='cluster_cells',step_id='another-cluster',argument='resolution',value=.4),
                                 dict(tool='cluster_cells',step_id='another-cluster',argument='random_seed',value=17)]))
        assert response.status_code == 202, response.text
        result = wait_turn(client, 'second-edit')
        assert result['status'] == 'succeeded', result
        assert sum(len(m.calls) for m in harness.models) == calls
        state = sessions.load('session')
        run = harness.service._application.run_store.load(state.revisions[-1].run_id)
        assert [s.step_id for s in run.steps] == ['another-cluster', 'umap']
        edited = run.steps[0]
        assert edited.resolved_arguments['resolution'] == .4
        assert edited.resolved_arguments['random_seed'] == 17
        assert state.revisions[0] == revision
