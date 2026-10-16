"""Ordinary attachment handoff; simulated embeddings, real downstream owners."""
import json
from dataclasses import replace
from pathlib import Path

import anndata as ad
from fastapi.testclient import TestClient
import numpy as np
import pytest

from agent.web.app import create_app
from test_h5ad_analysis import analysis_harness, ScientificPlanModel, CHECKPOINT_HASH
from helpers import wait_turn


def model_reply(self, *, prompt, response_schema):
    p = json.loads(prompt)
    if 'turn_schema_version' in p:
        return json.dumps(dict(turn_schema_version=1, decision=dict(kind='execute_plan',
            target='compute_cell_umap', roles=[dict(tool='epizoo_embed_cells', port='dataset', source='source.0')],
            arguments=[dict(tool='epizoo_embed_cells', argument='species', literal='mouse', value='mouse', evidence='mouse'),
                       dict(tool='compute_cell_umap', argument='min_dist', literal='0', evidence='min_dist 0')])))
    if 'selection_schema_version' in response_schema.get('properties', {}):
        return json.dumps(dict(selection_schema_version=1, decision=dict(kind='select', capability_ids=['embedding_analysis'])))
    if 'output_selection_schema_version' in p:
        return json.dumps(dict(outputs=[dict(name=n, step_id=s, output_key=k) for n, s, k in (
            ('embeddings', 'embed', 'embedding_path'), ('neighbors', 'neighbors', 'analysis_path'),
            ('clusters', 'cluster', 'analysis_path'), ('umap', 'umap', 'analysis_path'))]))
    return json.dumps(dict(schema_version=4, decision=dict(kind='plan', steps=[
        dict(step_id=n, tool=t, sources=[] if upstream is None else [dict(kind='step', target=port, step=upstream)],
             control_dependencies=[]) for n, t, port, upstream in (
            ('embed', 'epizoo_embed_cells', None, None), ('neighbors', 'build_cell_neighbors', 'embedding', 'embed'),
            ('cluster', 'cluster_cells', 'analysis', 'neighbors'), ('umap', 'compute_cell_umap', 'analysis', 'neighbors'))])))


def test_attachment_catalog_pins_independent_downloads_and_restart(tmp_path, monkeypatch):
    harness = analysis_harness(tmp_path, monkeypatch)
    monkeypatch.setattr(ScientificPlanModel, 'complete', model_reply)
    with TestClient(create_app(harness.service, uploads=harness.uploads,
                              epizoo_resources=(harness.resource(),))) as client:
        assert client.post('/api/v1/sessions', json={'session_id': 'session'}).status_code == 201
        response = client.put('/api/v1/sessions/session/uploads/source', params=dict(filename='cells.h5ad', input_type='h5ad'),
            content=harness.source.read_bytes(), headers={'content-type': 'application/octet-stream'})
        assert response.status_code == 201, response.text
        response = client.post('/api/v1/sessions/session/turns', json=dict(turn_id='first', expected_generation=0,
            utterance='Analyze my mouse cells with embeddings, neighbors, Leiden and independent UMAP with min_dist 0.',
            attachment_ids=[response.json()['resource_id']]))
        assert response.status_code == 202, response.text
        result = wait_turn(client, 'first', timeout=90)
        assert result['status'] == 'succeeded', result
        run = harness.service._application.run_store.load(result['run_id'])
        embed, neighbors, clusters, umap = run.steps
        assert embed.resolved_arguments['expected_resource_identity']['checkpoint_sha256'] == CHECKPOINT_HASH
        assert embed.result['resource_provenance']['actual_resource_identity']['checkpoint_sha256'] == CHECKPOINT_HASH
        assert umap.result['min_dist'] == 0 and umap.result['clustering_present'] is False
        assert run.plan.steps[-1].depends_on == ('neighbors',)
        from agent.orchestration.prior_outputs import optional_binding_for_locator
        state = harness.service._application.sessions.load('session')
        for locator in state.revisions[-1].outputs:
            binding = optional_binding_for_locator(locator, harness.service._application.run_store,
                                                  harness.service._application.registry)
            assert (binding is not None) == (locator.name == 'neighbors')
        from agent.application.local_resources import LocalResourceAdmission, ResourceAdmissionError
        from agent.application.dialogue_execution import execution_submission
        submission = execution_submission(state.interactions[-1])
        owner = harness.service.resources
        binding = LocalResourceAdmission._submission_binding(submission)
        wrong = replace(run.plan.steps[0], arguments=dict(run.plan.steps[0].arguments, checkpoint_path='/wrong/model.pth'))
        with pytest.raises(ResourceAdmissionError, match='consum'):
            owner._validate_group_consumption(binding, (wrong, *run.plan.steps[1:]))
        base = '/api/v1/sessions/session/revisions/' + result['revision_id']
        artifacts = client.get(base + '/scientific-artifacts').json()['artifacts']
        assert {a['filename'] for a in artifacts} == {'embeddings.h5ad', 'neighbors.h5ad', 'clusters.h5ad', 'analysis.h5ad'}
        original = ad.read_h5ad(clusters.result['analysis_path'])
        for item in artifacts:
            delivered = client.get(base + '/scientific-artifacts/' + item['handle'])
            assert delivered.status_code == 200, delivered.text
            assert delivered.headers['Content-Disposition'] == f'attachment; filename="{item["filename"]}"'
            path = tmp_path / item['filename']; path.write_bytes(delivered.content)
            data = ad.read_h5ad(path)
            assert tuple(data.obs_names) == tuple(original.obs_names)
            np.testing.assert_array_equal(data.obsm['X_epizoo'], original.obsm['X_epizoo'])
            if item['filename'] == 'clusters.h5ad':
                assert data.obs['leiden'].equals(original.obs['leiden']) and 'X_umap' not in data.obsm
            elif item['filename'] == 'embeddings.h5ad':
                assert not len(data.obs.columns) and set(data.obsm) == {'X_epizoo'}
            assert str(tmp_path) not in json.dumps(dict(data.uns))
        calls = list(harness.science_calls)
        # Reload immutable declaration/resource receipt and accepted evidence.
        assert harness.service.reopen_session('session').active_revision_id == result['revision_id']
        assert harness.science_calls == calls
        victim = next(a for a in artifacts if a['filename'] == 'clusters.h5ad')
        path = Path(clusters.result['analysis_path']); path.write_bytes(path.read_bytes() + b'tampered')
        assert client.get(base + '/scientific-artifacts/' + victim['handle']).status_code != 200
        assert harness.science_calls == calls
