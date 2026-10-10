"""Registered H5AD composition reaches ordinary scientific execution and history.

Only EpiZoo's model loading/inference is scripted in these software checks.
The public embedding owner, downstream scientific tools, verifier, evidence,
report, interactive facade and HTTP worker are the production implementations.
"""
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from agent.application import InteractiveAgentApplication
from agent.application.uploads import H5ADUploadAdmission
from agent.orchestration import PlanningModelProfile, ToolRegistry, build_default_tool_registry
from agent.orchestration.prior_outputs import binding_for_locator
from agent.providers import PlanningModelFactoryRegistry
from agent.schemas.orchestration import _serialize
from agent.schemas.verification_authority import AuthorityError
from agent.tools.analysis import epizoo_embedding
from agent.tools.analysis.embedding_analysis import PROVENANCE_KEY
from agent.web.app import create_app
from agent.web.config import QualifiedEpiZooResource, ScientificInputSet
from helpers import wait_turn
from test_uploaded_multiturn import assert_private, body, reopened_service, snapshot, upload


CHECKPOINT_HASH = hashlib.sha256(b'Qualified software fixture checkpoint.').hexdigest()
FREQUENCIES_HASH = hashlib.sha256(b'Qualified software fixture frequencies.').hexdigest()
FILTER_HASH = hashlib.sha256(b'Qualified software fixture filter.').hexdigest()


class ScientificPlanModel:
    """Exact provider-authored semantic plans; no production recipe is added."""

    def __init__(self, profile):
        self.model_id = profile.model_id
        self.calls = []
        self.mode = None

    def complete(self, *, prompt, response_schema):
        value = json.loads(prompt)
        self.calls.append(value)
        if 'turn_schema_version' in value:
            utterance = value['utterance']
            if utterance == 'What does this accepted result show?':
                decision = dict(kind='answer_scientific',
                    target=dict(output='r0', subject=None), comparison=None, focus='question')
            else:
                self.mode = ('inspect' if utterance.startswith('Inspect') else
                             'chain' if utterance.startswith('Analyze') else 'embed')
                decision = dict(kind='execute_plan', target={
                    'inspect': 'inspect_scATAC', 'chain': 'compute_cell_umap',
                    'embed': 'epizoo_embed_cells'}[self.mode])
            return json.dumps(dict(turn_schema_version=1, decision=decision))
        if 'selection_schema_version' in response_schema.get('properties', {}):
            capabilities = (['processed_inspection'] if self.mode == 'inspect' else
                            ['processed_inspection', 'embedding_analysis'] if self.mode == 'chain' else
                            ['embedding_analysis'])
            return json.dumps(dict(selection_schema_version=1,
                decision=dict(kind='select', capability_ids=capabilities)))
        if 'output_selection_schema_version' in value:
            step, key = {'inspect': ('inspect', 'n_cells'), 'embed': ('embed', 'embedding_path'),
                         'chain': ('umap', 'analysis_path')}[self.mode]
            return json.dumps(dict(outputs=[dict(name='result', step_id=step, output_key=key)]))
        if 'dialogue_schema_version' in value:
            claim = next(c['claim_id'] for c in value['evidence']['claims'] if c['field'] == 'n_cells')
            return json.dumps(dict(support='supported', paragraphs=[dict(parts=[
                dict(kind='text', text='The accepted result records:'), dict(kind='claim', id=claim)])]))

        def step(name, tool, target=None, upstream=None):
            sources = [] if target is None else [dict(kind='step', target=target, step=upstream)]
            return dict(step_id=name, tool=tool, sources=sources, control_dependencies=[])

        if self.mode == 'inspect':
            steps = [step('inspect', 'inspect_scATAC')]
        elif self.mode == 'embed':
            steps = [step('embed', 'epizoo_embed_cells')]
        else:
            steps = [step('inspect', 'inspect_scATAC'),
                     step('embed', 'epizoo_embed_cells', 'dataset', 'inspect'),
                     step('neighbors', 'build_cell_neighbors', 'embedding', 'embed'),
                     step('cluster', 'cluster_cells', 'analysis', 'neighbors'),
                     step('umap', 'compute_cell_umap', 'analysis', 'cluster')]
        return json.dumps(dict(schema_version=4, decision=dict(kind='plan', steps=steps)))


@dataclass
class AnalysisHarness:
    service: InteractiveAgentApplication
    uploads: H5ADUploadAdmission
    root: Path
    source: Path
    checkpoint: Path
    models: list
    science_calls: list
    inference_calls: list

    def resource(self, resource_id='mouse-reviewed', *, default=True, species='mouse'):
        return QualifiedEpiZooResource(resource_id=resource_id, label='Reviewed software resource',
            species=species, checkpoint_path=str(self.checkpoint), checkpoint_sha256=CHECKPOINT_HASH,
            frequencies_sha256=FREQUENCIES_HASH, filter_indices_sha256=FILTER_HASH,
            qualification='Exact deterministic software fixture; no biological qualification.',
            default=default)


def analysis_harness(tmp_path, monkeypatch):
    source = tmp_path / 'source.h5ad'
    ad.AnnData(sparse.csr_matrix(np.tile(np.array([[1, 0, 2, 0]], dtype=np.float32), (32, 1))),
        obs=pd.DataFrame(index=[f'cell-{i:02d}' for i in range(32)]),
        var=pd.DataFrame(index=['peak-a', 'peak-b', 'peak-c', 'peak-d'])).write_h5ad(source)
    checkpoint = tmp_path / 'model.pth'
    checkpoint.write_bytes(b'Qualified software fixture checkpoint.')
    auxiliaries = tmp_path / 'auxiliary'
    auxiliaries.mkdir()
    frequencies = auxiliaries / 'cCRE_frequencies_mouse.npy'
    frequencies.write_bytes(b'Qualified software fixture frequencies.')
    filter_indices = auxiliaries / 'cCRE_filter_idx_mouse.csv'
    filter_indices.write_bytes(b'Qualified software fixture filter.')
    monkeypatch.setattr(epizoo_embedding.epizoo_backend, 'DEFAULT_RESOURCES_DIR', auxiliaries)
    monkeypatch.setattr(epizoo_embedding, 'get_cached_epizoo_model',
        lambda **kwargs: SimpleNamespace(_agent_checkpoint_sha256=CHECKPOINT_HASH,
            _agent_checkpoint_source_snapshot=epizoo_embedding.epizoo_backend._resource_file_snapshot(checkpoint)))
    inference_calls = []

    def inference(model, adata, **arguments):
        inference_calls.append(arguments)
        assert sparse.issparse(adata.X)
        rng = np.random.default_rng(17)
        embeddings = np.concatenate([rng.normal(-1, .25, (16, 512)),
                                     rng.normal(1, .25, (16, 512))]).astype(np.float32)
        metadata = dict(species=dict(name=arguments['species']),
            checkpoint=dict(path=str(checkpoint), sha256=CHECKPOINT_HASH),
            resources=dict(frequencies=dict(path=str(frequencies), sha256=FREQUENCIES_HASH),
                           filter_indices=dict(path=str(filter_indices), sha256=FILTER_HASH)),
            device=arguments['device'], dtype='float32', batch_size=arguments['batch_size'],
            max_length=arguments['max_length'], random_sample=arguments['random_sample'],
            random_seed=arguments['random_seed'], num_workers=arguments['num_workers'],
            amp=dict(requested=arguments['use_amp'], enabled=arguments['device'].startswith('cuda')))
        return SimpleNamespace(embeddings=embeddings, obs_names=tuple(adata.obs_names), metadata=metadata)

    monkeypatch.setattr(epizoo_embedding.epizoo_backend, 'embed_cells', inference)
    root = tmp_path / 'workspace' / 'uploads'
    root.mkdir(parents=True)
    models, science_calls = [], []
    default_registry = build_default_tool_registry()

    def counted(spec):
        def execute(**arguments):
            science_calls.append(spec.name)
            return spec.function(**arguments)
        return replace(spec, function=execute)

    registry = ToolRegistry(tuple(counted(default_registry.get(name)) for name in default_registry.names()))

    def factory(profile):
        model = ScientificPlanModel(profile)
        models.append(model)
        return model

    profiles = tuple(PlanningModelProfile(key, 'scripted', 'model/' + key) for key in ('alpha', 'beta'))
    service = InteractiveAgentApplication(tmp_path / 'workspace', model_profiles=profiles,
        default_profile_id='alpha', planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': factory}),
        registry=registry, approved_source_roots=(root,))
    return AnalysisHarness(service, H5ADUploadAdmission(service.resources, root), root, source,
                           checkpoint, models, science_calls, inference_calls)


def companion(**parameters):
    return ScientificInputSet('mouse', 'Explicit mouse declarations',
        dict(species='mouse', device='cpu') | parameters, h5ad_companion=True)


def scientific_submit(client, resource_id, *, utterance='Embed the selected H5AD.',
                      turn='first', generation=0, input_set_id='mouse', **changes):
    payload = body(resource_id, turn=turn, generation=generation, utterance=utterance, **changes)
    if input_set_id is not None:
        payload['input_set_id'] = input_set_id
    response = client.post('/api/v1/sessions/session/turns', json=payload)
    assert response.status_code == 202, response.text
    return wait_turn(client, turn, timeout=90), payload


def accepted_result(client, result):
    path = f"/api/v1/sessions/session/revisions/{result['revision_id']}"
    evidence = client.get(path + '/evidence', params={'output_name': 'result'})
    assert evidence.status_code == 200, evidence.text
    assert evidence.json()['status'] == 'available'
    artifacts = client.get(path + '/artifacts').json()['artifacts']
    report = next(a for a in artifacts if a['artifact_type'] == 'analysis_report')
    delivered = client.get(path + '/artifacts/' + report['handle'])
    assert delivered.status_code == 200 and delivered.content
    return evidence.json(), artifacts, delivered.content


@pytest.mark.parametrize('explicit', [False, True])
def test_registered_h5ad_same_plan_chain_uses_owner_defaults_or_explicit_parameters(tmp_path, monkeypatch, explicit):
    harness = analysis_harness(tmp_path, monkeypatch)
    parameters = dict(n_neighbors=7, metric='cosine', resolution=.7, min_dist=.2, spread=1.2,
                      neighbors_random_seed=3, cluster_random_seed=4, umap_random_seed=5) if explicit else {}
    with TestClient(create_app(harness.service, uploads=harness.uploads,
            input_sets=(companion(**parameters),), epizoo_resources=(harness.resource(),))) as client:
        client.post('/api/v1/sessions', json={'session_id': 'session'})
        resource = upload(client, harness.source.read_bytes())
        assert not harness.models and not harness.science_calls
        result, _ = scientific_submit(client, resource['resource_id'], utterance='Analyze the selected H5AD.')
        assert result['status'] == 'succeeded', result
        assert result['revision_id'] and result['response']['status'] == 'activated'
        assert harness.science_calls == ['inspect_scATAC', 'epizoo_embed_cells', 'build_cell_neighbors',
                                         'cluster_cells', 'compute_cell_umap']
        run = harness.service._application.run_store.load(result['run_id'])
        assert all(step.verification.passed for step in run.steps)
        inspected, embedded, neighbors, clustered, umap = run.steps
        registered = harness.service.resources.load(resource['resource_id'])
        assert inspected.resolved_arguments['path'] == registered.source_path
        assert embedded.resolved_arguments['input_path'] == registered.source_path
        assert embedded.result['resource_provenance']['actual_resource_identity'] == dict(
            resource_id='mouse-reviewed', checkpoint_sha256=CHECKPOINT_HASH,
            frequencies_sha256=FREQUENCIES_HASH, filter_indices_sha256=FILTER_HASH)
        assert neighbors.result['n_neighbors'] == (7 if explicit else 15)
        assert neighbors.result['metric'] == ('cosine' if explicit else 'euclidean')
        assert clustered.result['resolution'] == (.7 if explicit else 1.0)
        assert umap.result['min_dist'] == (.2 if explicit else .5)
        assert umap.result['spread'] == (1.2 if explicit else 1.0)
        if not explicit:
            assert 'n_neighbors' not in neighbors.resolved_arguments
            assert 'resolution' not in clustered.resolved_arguments
            assert 'min_dist' not in umap.resolved_arguments
        final = ad.read_h5ad(umap.result['analysis_path'])
        assert tuple(final.obs_names) == tuple(f'cell-{i:02d}' for i in range(32))
        assert final.obsm['X_epizoo'].shape == (32, 512)
        assert final.obsm['X_umap'].shape == (32, 2)
        assert sparse.issparse(final.obsp['connectivities'])
        assert final.uns[PROVENANCE_KEY]['parameters']['neighbors']['n_neighbors'] == neighbors.result['n_neighbors']
        evidence, artifacts, report = accepted_result(client, result)
        assert_private([result, evidence, artifacts], tmp_path)
        assert str(tmp_path).encode() not in report


def test_registered_direct_embedding_provenance_answer_retry_and_restart(tmp_path, monkeypatch):
    harness = analysis_harness(tmp_path, monkeypatch)
    inputs, resources = (companion(),), (harness.resource(),)
    with TestClient(create_app(harness.service, uploads=harness.uploads,
            input_sets=inputs, epizoo_resources=resources)) as client:
        client.post('/api/v1/sessions', json={'session_id': 'session'})
        resource = upload(client, harness.source.read_bytes())
        result, payload = scientific_submit(client, resource['resource_id'])
        assert result['status'] == 'succeeded', result
        assert harness.science_calls == ['epizoo_embed_cells']
        run = harness.service._application.run_store.load(result['run_id'])
        step, = run.steps
        registered = harness.service.resources.load(resource['resource_id'])
        assert step.resolved_arguments['input_path'] == registered.source_path
        pins = _serialize(step.resolved_arguments['expected_resource_identity'])
        provenance = step.result['resource_provenance']
        assert provenance['expected_resource_identity'] == provenance['actual_resource_identity'] == pins
        assert provenance['inference_settings']['overwrite'] is False
        evidence, artifacts, report = accepted_result(client, result)
        # New invocation pins do not manufacture historical schema-2 authority.
        locator = harness.service._application.sessions.load('session').revisions[0].outputs[0]
        assert step.verification.artifact_authority is None
        before_inference = list(harness.inference_calls)

        def forbidden_reconstruction(*args, **kwargs):
            raise AssertionError('Unsupported historical embedding reuse must reconstruct no science.')

        with monkeypatch.context() as blocked:
            blocked.setattr('agent.orchestration.verification_authority._load_context', forbidden_reconstruction)
            with pytest.raises(AuthorityError, match='No recorded scientific authority'):
                binding_for_locator(locator, harness.service._application.run_store,
                                    harness.service._application.registry)
        assert harness.inference_calls == before_inference
        assert harness.science_calls == ['epizoo_embed_cells']
        follow, _ = scientific_submit(client, resource['resource_id'], turn='question', generation=1,
            utterance='What does this accepted result show?')
        assert follow['response']['kind'] == 'answer'
        assert follow['response']['scientific']['support'] == 'supported'
        assert harness.science_calls == ['epizoo_embed_cells']
        history = client.get('/api/v1/sessions/session').json()
        before_models = len(harness.models)
        assert client.post('/api/v1/sessions/session/turns', json=payload).status_code == 202
        assert wait_turn(client) == result
        assert len(harness.models) == before_models
        assert harness.science_calls == ['epizoo_embed_cells']
    # The completed retry and accepted presentation do not consume source/model bytes.
    Path(registered.source_path).unlink()
    harness.checkpoint.unlink()
    read_only = reopened_service(tmp_path / 'workspace', harness.root)
    before_restart = snapshot(tmp_path)
    with TestClient(create_app(read_only, uploads=H5ADUploadAdmission(read_only.resources, harness.root),
            input_sets=inputs, epizoo_resources=resources)) as client:
        assert client.get('/api/v1/sessions/session').json() == history
        assert client.post('/api/v1/sessions/session/turns', json=payload).status_code == 202
        assert wait_turn(client) == result
        assert accepted_result(client, result) == (evidence, artifacts, report)
    assert snapshot(tmp_path) == before_restart


@pytest.mark.parametrize('condition,expected', [
    ('missing-species', 'H5AD_SPECIES_REQUIRED'),
    ('missing-resource', 'EPIZOO_RESOURCE_REQUIRED'),
    ('ambiguous-resource', 'EPIZOO_RESOURCE_AMBIGUOUS'),
    ('changed-resource', 'EPIZOO_RESOURCE_IDENTITY_MISMATCH'),
])
def test_embedding_prerequisites_fail_with_safe_codes_and_no_revision(tmp_path, monkeypatch, condition, expected):
    harness = analysis_harness(tmp_path, monkeypatch)
    inputs = (ScientificInputSet('mouse', 'Unspecified species', {'device': 'cpu'}, h5ad_companion=True),) if condition == 'missing-species' else (companion(),)
    resources = () if condition == 'missing-resource' else (harness.resource(),)
    if condition == 'ambiguous-resource':
        resources += (harness.resource('other-reviewed'),)
    if condition == 'changed-resource':
        harness.checkpoint.write_bytes(b'Changed resource bytes.')
    with TestClient(create_app(harness.service, uploads=harness.uploads,
            input_sets=inputs, epizoo_resources=resources)) as client:
        client.post('/api/v1/sessions', json={'session_id': 'session'})
        resource = upload(client, harness.source.read_bytes())
        result, _ = scientific_submit(client, resource['resource_id'])
        assert result['revision_id'] is None, result
        if condition == 'missing-species':
            assert result['status'] == 'clarification' and result['error'] is None, result
            assert result['response']['clarification']['reason'] == 'missing_species'
        else:
            assert result['status'] == 'failed' and result['error']['code'] == expected, result
        assert not harness.inference_calls
        if condition != 'changed-resource':
            assert not harness.science_calls
        assert not harness.service._application.sessions.load('session').revisions
        assert_private(result, tmp_path)


def test_inspection_companions_do_not_require_or_execute_epizoo(tmp_path, monkeypatch):
    harness = analysis_harness(tmp_path, monkeypatch)
    omitted = ScientificInputSet('mouse', 'No embedding declarations', {}, h5ad_companion=True)
    with TestClient(create_app(harness.service, uploads=harness.uploads,
            input_sets=(omitted,), epizoo_resources=())) as client:
        client.post('/api/v1/sessions', json={'session_id': 'session'})
        resource = upload(client, harness.source.read_bytes())
        result, _ = scientific_submit(client, resource['resource_id'], utterance='Inspect the selected H5AD.')
        assert result['status'] == 'succeeded', result
        assert harness.science_calls == ['inspect_scATAC'] and not harness.inference_calls
        accepted_result(client, result)


@pytest.mark.parametrize('parameters,expected,science', [
    ({'random_seed': 11}, 'UNAUTHORIZED_REQUEST_INPUT', []),
    ({'n_neighbors': 32}, 'INVALID_ARGUMENT',
        ['inspect_scATAC', 'epizoo_embed_cells', 'build_cell_neighbors']),
])
def test_explicit_invalid_or_ambiguous_parameters_fail_without_fallback(tmp_path, monkeypatch, parameters, expected, science):
    harness = analysis_harness(tmp_path, monkeypatch)
    with TestClient(create_app(harness.service, uploads=harness.uploads,
            input_sets=(companion(**parameters),), epizoo_resources=(harness.resource(),))) as client:
        client.post('/api/v1/sessions', json={'session_id': 'session'})
        resource = upload(client, harness.source.read_bytes())
        result, _ = scientific_submit(client, resource['resource_id'], utterance='Analyze the selected H5AD.')
        assert result['status'] == 'failed' and result['revision_id'] is None, result
        assert result['error']['code'] == expected, result
        assert harness.science_calls == science
        assert not harness.service._application.sessions.load('session').revisions
        if science:
            run = harness.service._application.run_store.load(result['run_id'])
            rejected = next(step for step in run.steps if step.tool_name == 'build_cell_neighbors')
            assert rejected.resolved_arguments['n_neighbors'] == 32
            assert rejected.result is None
        assert_private(result, tmp_path)
