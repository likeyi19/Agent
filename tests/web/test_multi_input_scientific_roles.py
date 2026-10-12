"""Ordinary multi-upload roles execute genuine bounded scientific owners.

Only semantic model responses are scripted. Registration, source identity,
Interpreter admission, Planner, compiler, Leiden, evaluation, verification,
RunStore, Session publication, evidence and HTTP worker use production code.
The synthetic embeddings are explicit inputs, not foundation-model inference.
"""
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import inspect
import json
from pathlib import Path
import sys
import threading

import anndata as ad
from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from agent.application import InteractiveAgentApplication
from agent.application.uploads import H5ADUploadAdmission
from agent.orchestration import PlanningModelProfile, ToolRegistry, build_default_tool_registry
from agent.providers import PlanningModelFactoryRegistry
from agent.schemas import RunLifecycleStatus, StepOutputRef
from agent.schemas.orchestration import _serialize
from agent.tools.analysis.embedding_analysis import PROVENANCE_KEY, build_cell_neighbors
from agent.web.app import create_app
from helpers import wait_turn


LABEL = 'Cell 类型 Δ'
OBJECTIVE = ('Cluster the first selected dataset, then evaluate its fixed clustering '
             'against the second selected dataset as ground truth.')
REQUEST = OBJECTIVE + f' Use ground-truth column "{LABEL}".'
ANSWER = f'Use ground-truth column "{LABEL}".'
PROFILE = PlanningModelProfile('multi-source', 'scripted', 'scripted/multi-source')
ROLES = [dict(tool='cluster_cells', port='analysis', source='source.0'),
         dict(tool='evaluate_cell_clustering', port='ground_truth_dataset', source='source.1')]
_active_observation = None


def declaration(utterance, label=LABEL, *, tool='evaluate_cell_clustering', argument='label_key'):
    start = utterance.index(label)
    return dict(tool=tool, argument=argument, literal=label, start=start, end=start + len(label))


class MultiInputModel:
    """Script the existing Interpreter, scope, semantic plan and output selection."""
    def __init__(self, *, variant=None, roles=None, before_interpret=None, before_plan=None):
        self.model_id = PROFILE.model_id
        self.variant, self.roles = variant, deepcopy(ROLES if roles is None else roles)
        self.before_interpret, self.before_plan = before_interpret, before_plan
        self.calls = []

    def complete(self, *, prompt, response_schema):
        value = json.loads(prompt)
        self.calls.append(dict(value=value, prompt=prompt, schema=response_schema))
        if 'turn_schema_version' in value:
            if self.before_interpret:
                self.before_interpret()
            utterance = value['utterance']
            if utterance == ANSWER:
                decision = dict(kind='answer_parameters', pending='@parameters',
                                arguments=[declaration(utterance)])
            elif utterance == 'Inspect an unrelated dataset.':
                decision = dict(kind='execute_plan', target='inspect_scATAC', context='none')
            elif utterance == 'Evaluate the selected datasets.':
                decision = dict(kind='clarify', reason='ambiguous_subject')
            else:
                decision = dict(kind='execute_plan', target='evaluate_cell_clustering',
                                roles=deepcopy(self.roles))
                if self.variant != 'missing_label':
                    literal = 'absent_column' if 'absent_column' in utterance else LABEL
                    decision['arguments'] = [declaration(utterance, literal)]
            return json.dumps(dict(turn_schema_version=1, decision=decision), ensure_ascii=False)
        if 'selection_schema_version' in response_schema.get('properties', {}):
            return json.dumps(dict(selection_schema_version=1,
                decision=dict(kind='select', capability_ids=['embedding_analysis'])))
        if 'output_selection_schema_version' in value:
            return json.dumps(dict(outputs=[dict(name='evaluation', step_id='evaluation',
                                                output_key='report_path')]))
        if self.before_plan:
            self.before_plan()
        sources = [dict(target='analysis', source=dict(kind='step', step='clustering')),
                   dict(target='ground_truth_dataset', source=dict(kind='input', input='reference_h5ad_path'))]
        clustering = dict(step_id='clustering', tool='cluster_cells', sources=[
            dict(target='analysis', source=dict(kind='input', input='analysis_path'))], control_dependencies=[])
        evaluation = dict(step_id='evaluation', tool='evaluate_cell_clustering', sources=sources,
                          control_dependencies=[])
        if self.variant == 'swap_plan':
            clustering['sources'][0]['source']['input'] = 'reference_h5ad_path'
            sources[-1]['source']['input'] = 'analysis_path'
        elif self.variant == 'wrong_ground_truth':
            sources[-1]['source']['input'] = 'analysis_path'
        elif self.variant == 'wrong_consumer':
            clustering['tool'] = 'compute_cell_umap'
        steps = [clustering, evaluation]
        if self.variant == 'repeat_consumer':
            steps.insert(1, deepcopy(clustering) | dict(step_id='other_clustering'))
        return json.dumps(dict(schema_version=4, decision=dict(kind='plan', steps=steps)))


@contextmanager
def multi_input_work():
    """Observe real bodies in both the caller and existing HTTP worker threads."""
    global _active_observation
    import scanpy as sc
    from agent.application import turn_decisions
    from agent.orchestration import semantic_compiler, verifier
    from agent.orchestration.executor import PlanExecutor
    from agent.orchestration.llm_planner import LLMPlanner
    from agent.tools.analysis import clustering_evaluation, embedding_analysis, epizoo_embedding
    functions = {
        'production.neighbors': inspect.unwrap(sc.pp.neighbors),
        'production.leiden': inspect.unwrap(sc.tl.leiden),
        'production.evaluation': clustering_evaluation.evaluate_cell_clustering,
        'owner.clustering': embedding_analysis.cluster_cells,
        'source.evaluation': clustering_evaluation._evaluate_sources,
        'production.epizoo': epizoo_embedding.epizoo_backend.embed_cells,
        'verification.clustering': verifier._verify_clustering,
        'verification.evaluation': verifier._verify_clustering_evaluation,
        'compiler': semantic_compiler.compile_semantic_plan,
        'preflight': PlanExecutor.preflight,
        'interpreter': turn_decisions.interpret,
        'planner': LLMPlanner._plan_detailed,
    }
    codes = {function.__code__: name for name, function in functions.items()}
    counts, lock = Counter(), threading.Lock()
    previous_main, previous_thread = sys.getprofile(), threading.getprofile()
    previous_observation = _active_observation
    _active_observation = (counts, lock, codes)
    def observe(frame, event, arg):
        # Existing pool threads retain their installed hook across contexts.
        # Route each actual call to the current observation, including a resumed
        # turn on that same thread, rather than its earlier captured Counter.
        active = _active_observation
        if event == 'call' and active is not None and frame.f_code in active[2]:
            with active[1]:
                active[0][active[2][frame.f_code]] += 1
    sys.setprofile(observe)
    threading.setprofile(observe)
    try:
        yield counts
    finally:
        _active_observation = previous_observation
        sys.setprofile(previous_main)
        threading.setprofile(previous_thread)


@dataclass
class MultiInputHarness:
    service: InteractiveAgentApplication
    uploads: H5ADUploadAdmission
    source: Path
    ground_truth: Path
    ids: tuple
    values: np.ndarray
    models: list


def multi_input_harness(tmp_path, *, variant=None, roles=None, content=None,
                        before_interpret=None, before_plan=None, registry=None):
    """Reusable native-browser fixture; no checkpoint, inference or biological data."""
    root = Path(tmp_path)
    root.mkdir(parents=True, exist_ok=True)
    seed = root / 'fixture-science'
    seed.mkdir()
    ids = tuple(f'cell-{index:02d}-Δ' for index in range(32))
    rng = np.random.default_rng(17)
    values = np.concatenate([rng.normal(-1, .25, (16, 512)),
                             rng.normal(1, .25, (16, 512))]).astype(np.float32)
    embedding, ordered_ids = seed / 'embedding.npy', seed / 'cell-ids.txt'
    np.save(embedding, values, allow_pickle=False)
    ordered_ids.write_text(''.join(value + '\n' for value in ids), encoding='utf-8')
    graph = build_cell_neighbors(embedding, ordered_ids, seed / 'graph')
    source = Path(graph['analysis_path'])
    if content == 'missing_provenance':
        original = ad.read_h5ad(source)
        del original.uns[PROVENANCE_KEY]
        original.write_h5ad(source)
    truth_ids = tuple(reversed(ids)) if content == 'wrong_cell_order' else ids
    ground_truth = seed / 'ground-truth.h5ad'
    obs = pd.DataFrame({LABEL: ['class α'] * 16 + ['class β'] * 16}, index=pd.Index(truth_ids))
    ad.AnnData(X=sparse.csr_matrix((32, 4), dtype=np.float32), obs=obs,
               var=pd.DataFrame(index=[f'feature-{index}' for index in range(4)])).write_h5ad(ground_truth)
    models = []
    def factory(profile):
        model = MultiInputModel(variant=variant, roles=roles, before_interpret=before_interpret,
                                before_plan=before_plan)
        models.append(model)
        return model
    upload_root = root / 'workspace' / 'uploads'
    upload_root.mkdir(parents=True)
    service = InteractiveAgentApplication(root / 'workspace', model_profiles=(PROFILE,),
        default_profile_id=PROFILE.profile_id,
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': factory}),
        approved_source_roots=(upload_root,), registry=registry)
    uploads = H5ADUploadAdmission(service.resources, upload_root)
    return MultiInputHarness(service, uploads, source, ground_truth, ids, values, models)


def upload_sources(client, harness):
    records = []
    for index, (source, label) in enumerate([(harness.source, 'Selected analysis.h5ad'),
                                            (harness.ground_truth, 'Independent truth.h5ad')]):
        response = client.put(f'/api/v1/uploads/source-{index}', params={'filename': label},
            content=source.read_bytes(), headers={'Content-Type': 'application/octet-stream'})
        assert response.status_code == 201, response.text
        records.append(response.json())
    assert client.post('/api/v1/sessions', json={'session_id': 'session'}).status_code == 201
    return records


def multi_body(resource_ids=None, *, turn='first', utterance=REQUEST, generation=0):
    value = dict(turn_id=turn, expected_generation=generation, utterance=utterance)
    if resource_ids is not None:
        value['resource_ids'] = list(resource_ids)
    return value


def execute(client, payload):
    response = client.post('/api/v1/sessions/session/turns', json=payload)
    assert response.status_code == 202, response.text
    return wait_turn(client, payload['turn_id'], timeout=60)


def assert_no_owner(counts):
    assert not {name: count for name, count in counts.items()
                if count and name.startswith(('production.', 'owner.', 'source.', 'verification.'))}


def snapshots(root):
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in Path(root).rglob('*') if path.is_file()}


def test_multiple_uploads_roles_compile_real_leiden_evaluation_and_restore_exact_history(tmp_path):
    harness = multi_input_harness(tmp_path)
    original = snapshots(harness.source.parent.parent)
    with TestClient(create_app(harness.service, uploads=harness.uploads)) as client:
        with multi_input_work() as upload_counts:
            records = upload_sources(client, harness)
        assert not upload_counts and not harness.models
        assert not harness.service._application.sessions.load('session').revisions
        selected = [record['resource_id'] for record in records]
        payload = multi_body(selected)
        pinned = [harness.service.resources.load(resource_id) for resource_id in selected]
        with multi_input_work() as counts:
            result = execute(client, payload)
        assert result['status'] == 'succeeded' and result['revision_id'], result
        assert counts['production.leiden'] == counts['production.evaluation'] == 1
        assert counts['production.epizoo'] == counts['production.neighbors'] == 0
        assert counts['interpreter'] == counts['planner'] == counts['compiler'] == 1
        assert counts['preflight'] >= 1
        assert counts['verification.clustering'] > 0 and counts['verification.evaluation'] > 0
        run = harness.service._application.run_store.load(result['run_id'])
        assert run.lifecycle_status is RunLifecycleStatus.SUCCEEDED
        assert [step.tool_name for step in run.steps] == ['cluster_cells', 'evaluate_cell_clustering']
        cluster, evaluation = run.steps
        assert all(step.verification.passed for step in run.steps)
        assert cluster.resolved_arguments['analysis_path'] == pinned[0].source_path
        assert 'reference_h5ad_path' not in cluster.resolved_arguments
        assert evaluation.resolved_arguments['reference_h5ad_path'] == pinned[1].source_path
        assert evaluation.resolved_arguments['label_key'] == LABEL
        assert run.plan.steps[1].arguments['analysis_path'] == StepOutputRef('clustering', 'analysis_path')
        assert run.plan.steps[1].depends_on == ('clustering',)
        assert evaluation.result['n_cells'] == 32 and evaluation.result['n_reference_classes'] == 2
        assert evaluation.result['n_predicted_clusters'] == 2
        assert all(evaluation.result[key] == pytest.approx(1.) for key in ('nmi', 'ari', 'ami', 'homogeneity'))
        clustered = ad.read_h5ad(cluster.result['analysis_path'])
        assert tuple(clustered.obs_names) == harness.ids
        np.testing.assert_array_equal(clustered.obsm['X_epizoo'], harness.values)
        original_graph = ad.read_h5ad(harness.source)
        for key in ('distances', 'connectivities'):
            assert (clustered.obsp[key] != original_graph.obsp[key]).nnz == 0
        state = harness.service._application.sessions.load('session')
        assert len(state.revisions) == 1 and state.generation == 1
        submission = _serialize(state.interactions[0].submission)
        attribution = submission['registered_input']
        assert [row['resource_id'] for row in attribution['members']] == selected
        assert [row['record_sha256'] for row in attribution['members']] == [row.record_sha256 for row in pinned]
        assert _serialize(state.interactions[0].admitted['roles']) == ROLES
        assert run.request.prompt == REQUEST
        for model in harness.models:
            for call in model.calls:
                assert str(tmp_path) not in call['prompt']
                assert not any(row.record_sha256 in call['prompt'] or row.resource_id in call['prompt'] for row in pinned)
        assert len(harness.models) == 1 and len(harness.models[0].calls) == 4
        calls_before = len(harness.models[0].calls)
        before_reads = snapshots(harness.service._application.workspace_root)
        path = f"/api/v1/sessions/session/revisions/{result['revision_id']}"
        with multi_input_work() as reads:
            revision = client.get(path)
            evidence = client.get(path + '/evidence', params={'output_name': 'evaluation'})
            assert revision.status_code == evidence.status_code == 200
            facts = {row['field']: row for row in evidence.json()['facts']}
            assert evidence.json()['status'] == 'available'
            assert facts['nmi']['value'] == pytest.approx(1.)
            assert facts['label_key']['value'] == LABEL
            assert facts['explicit_user_parameters']['value']['parameters']['label_key'] == 'explicit_user_instruction'
            assert str(tmp_path) not in evidence.text + revision.text
            artifacts = client.get(path + '/artifacts').json()['artifacts']
            report = next(row for row in artifacts if row['artifact_type'] == 'analysis_report')
            content = client.get(path + '/artifacts/' + report['handle'])
            assert content.status_code == 200 and str(tmp_path) not in content.text
            assert execute(client, payload) == result
            history = client.get('/api/v1/sessions/session').json()
        assert not reads and len(harness.models[0].calls) == calls_before
        assert snapshots(harness.service._application.workspace_root) == before_reads
        changed_retry = client.post('/api/v1/sessions/session/turns',
            json=multi_body(list(reversed(selected))))
        assert changed_retry.status_code == 409
        assert changed_retry.json()['error']['code'] == 'INTERACTIVE_TURN_CONFLICT'
    def forbidden(*args, **kwargs):
        raise AssertionError('Restored completed history entered a model or scientific owner.')
    registry = build_default_tool_registry()
    forbidden_registry = ToolRegistry(tuple(replace(registry.get(name), function=forbidden)
                                            for name in registry.names()))
    reopened = InteractiveAgentApplication(harness.service._application.workspace_root,
        model_profiles=(PROFILE,), default_profile_id=PROFILE.profile_id, registry=forbidden_registry,
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': forbidden}),
        approved_source_roots=(harness.uploads.root,))
    with multi_input_work() as restored:
        with TestClient(create_app(reopened, uploads=H5ADUploadAdmission(reopened.resources, harness.uploads.root))) as client:
            assert client.get('/api/v1/sessions/session').json() == history
            assert execute(client, payload) == result
            assert client.get(path + '/evidence', params={'output_name': 'evaluation'}).json() == evidence.json()
    assert not restored and snapshots(harness.source.parent.parent) == original
    (tmp_path / 'multi-input-scientific-witness.json').write_text(json.dumps(dict(
        baseline_scope='Synthetic software acceptance; no biological or foundation-model qualification.',
        selected=[dict(resource_id=row.resource_id, record_sha256=row.record_sha256,
                       source_sha256=row.source_sha256, source_path=row.source_path) for row in pinned],
        roles=ROLES, counts=dict(counts), upload_counts=dict(upload_counts), read_counts=dict(reads),
        restored_counts=dict(restored), run_id=run.run_id, revision_id=result['revision_id'],
        label_key=LABEL, metrics={key: evaluation.result[key] for key in ('nmi', 'ari', 'ami', 'homogeneity')},
        resolved_arguments=[_serialize(step.resolved_arguments) for step in run.steps],
        prompt_sizes=[dict(prompt_bytes=len(call['prompt'].encode('utf-8')),
            schema_bytes=len(json.dumps(call['schema'], sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')))
                      for call in harness.models[0].calls]), indent=2, ensure_ascii=False), encoding='utf-8')


@pytest.mark.parametrize('variant', ['swap_plan', 'wrong_ground_truth', 'wrong_consumer', 'repeat_consumer'])
def test_compiled_consumers_cannot_substitute_or_repeat_selected_roles(tmp_path, variant):
    harness = multi_input_harness(tmp_path, variant=variant)
    with TestClient(create_app(harness.service, uploads=harness.uploads)) as client:
        records = upload_sources(client, harness)
        with multi_input_work() as counts:
            result = execute(client, multi_body([row['resource_id'] for row in records]))
        assert result['revision_id'] is None, result
        assert result['response']['status'] == 'failed', result
        assert_no_owner(counts)
        assert not harness.service._application.sessions.load('session').revisions


@pytest.mark.parametrize('roles', [
    ROLES + [ROLES[0]],
    [ROLES[0], ROLES[1] | dict(source='source.0')],
    [ROLES[0], ROLES[1] | dict(source='source.2')],
    [ROLES[0], ROLES[1] | dict(port='query_dataset')],
    [ROLES[0], ROLES[1] | dict(tool='cluster_cells')],
])
def test_duplicate_conflicting_absent_and_unsupported_role_associations_fail_closed(tmp_path, roles):
    harness = multi_input_harness(tmp_path, roles=roles)
    with TestClient(create_app(harness.service, uploads=harness.uploads)) as client:
        records = upload_sources(client, harness)
        with multi_input_work() as counts:
            result = execute(client, multi_body([row['resource_id'] for row in records]))
        assert result['revision_id'] is None, result
        assert_no_owner(counts)
        assert not harness.service._application.sessions.load('session').revisions


@pytest.mark.parametrize('content', ['missing_column', 'wrong_cell_order', 'missing_provenance'])
def test_scientific_content_rejection_stays_with_existing_owner(tmp_path, content):
    harness = multi_input_harness(tmp_path, content=content)
    utterance = REQUEST.replace(LABEL, 'absent_column') if content == 'missing_column' else REQUEST
    with TestClient(create_app(harness.service, uploads=harness.uploads)) as client:
        records = upload_sources(client, harness)
        with multi_input_work() as counts:
            result = execute(client, multi_body([row['resource_id'] for row in records], utterance=utterance))
        assert result['revision_id'] is None and result['response']['status'] == 'failed', result
        run = harness.service._application.run_store.load(result['run_id'])
        assert run.lifecycle_status is RunLifecycleStatus.FAILED
        assert any(error.code == 'INVALID_ARGUMENT' for error in run.errors)
        assert not harness.service._application.sessions.load('session').revisions
        if content == 'missing_provenance':
            assert counts['owner.clustering'] == 1
            assert counts['production.leiden'] == counts['production.evaluation'] == 0
            assert run.errors[0].tool_name == 'cluster_cells'
        else:
            assert counts['production.leiden'] == counts['production.evaluation'] == 1
            assert counts['source.evaluation'] == 1 and counts['verification.evaluation'] == 0
            assert any(error.code == 'INVALID_ARGUMENT' and error.tool_name == 'evaluate_cell_clustering'
                       for error in run.errors)


def test_selected_source_replacement_during_planning_fails_before_any_owner(tmp_path):
    replace_source = []
    def mutate():
        if replace_source:
            Path(replace_source.pop()).write_bytes(b'Changed after captured selection.')
    harness = multi_input_harness(tmp_path, before_plan=mutate)
    with TestClient(create_app(harness.service, uploads=harness.uploads)) as client:
        records = upload_sources(client, harness)
        selected = [row['resource_id'] for row in records]
        replace_source.append(harness.service.resources.load(selected[1]).source_path)
        with multi_input_work() as counts:
            result = execute(client, multi_body(selected))
        assert result['revision_id'] is None and result['response']['status'] == 'failed', result
        assert_no_owner(counts)
        assert not harness.service._application.sessions.load('session').revisions


def test_pending_consumer_rejects_source_replacement_after_upstream_science(tmp_path):
    """A plan guard must not authorize later consumption of different bytes."""
    replacement = []
    default = build_default_tool_registry()
    original = default.get('cluster_cells').function
    def cluster_then_replace(**arguments):
        result = original(**arguments)
        path = Path(replacement[0])
        truth = ad.read_h5ad(path)
        truth.obs[LABEL] = ['changed α' if index % 2 else 'changed β' for index in range(32)]
        truth.write_h5ad(path)
        return result
    registry = ToolRegistry(tuple(replace(default.get(name), function=cluster_then_replace)
        if name == 'cluster_cells' else default.get(name) for name in default.names()))
    harness = multi_input_harness(tmp_path, registry=registry)
    with TestClient(create_app(harness.service, uploads=harness.uploads)) as client:
        records = upload_sources(client, harness)
        selected = [row['resource_id'] for row in records]
        record = harness.service.resources.load(selected[1])
        replacement.append(record.source_path)
        with multi_input_work() as counts:
            result = execute(client, multi_body(selected))
        assert counts['owner.clustering'] == counts['production.leiden'] == 1
        assert counts['production.evaluation'] == counts['source.evaluation'] == 0, dict(counts)
        assert result['revision_id'] is None and result['response']['status'] == 'failed', result
        assert hashlib.sha256(Path(record.source_path).read_bytes()).hexdigest() != record.source_sha256
        assert not harness.service._application.sessions.load('session').revisions


def test_ambiguous_role_language_and_upload_alone_do_not_execute(tmp_path):
    harness = multi_input_harness(tmp_path)
    with TestClient(create_app(harness.service, uploads=harness.uploads)) as client:
        with multi_input_work() as uploads:
            records = upload_sources(client, harness)
        assert not uploads and not harness.models
        with multi_input_work() as counts:
            result = execute(client, multi_body([row['resource_id'] for row in records],
                                               utterance='Evaluate the selected datasets.'))
        assert result['response']['kind'] == 'clarify' and result['revision_id'] is None, result
        assert_no_owner(counts)
        assert not harness.service._application.sessions.load('session').revisions


@pytest.mark.parametrize('resubmit_selection', [False, True])
def test_missing_label_continuation_preserves_original_selected_roles(tmp_path, resubmit_selection):
    harness = multi_input_harness(tmp_path, variant='missing_label')
    with TestClient(create_app(harness.service, uploads=harness.uploads)) as client:
        records = upload_sources(client, harness)
        selected = [row['resource_id'] for row in records]
        with multi_input_work() as first:
            pending = execute(client, multi_body(selected, utterance=OBJECTIVE))
        assert_no_owner(first)
        assert pending['response']['kind'] == 'clarify', pending
        assert pending['response']['clarification']['choices'] == ['label_key']
        original = harness.service._application.sessions.load('session').interactions[0]
        with multi_input_work() as resumed:
            result = execute(client, multi_body(selected if resubmit_selection else None,
                                               turn='answer', utterance=ANSWER))
        assert result['status'] == 'succeeded' and result['revision_id'], result
        assert resumed['production.leiden'] == resumed['production.evaluation'] == 1
        state = harness.service._application.sessions.load('session')
        assert state.interactions[0] == original
        run = harness.service._application.run_store.load(result['run_id'])
        assert run.request.prompt == OBJECTIVE
        assert run.steps[0].resolved_arguments['analysis_path'] == harness.service.resources.load(selected[0]).source_path
        assert run.steps[1].resolved_arguments['reference_h5ad_path'] == harness.service.resources.load(selected[1]).source_path
        assert run.steps[1].resolved_arguments['label_key'] == LABEL
        assert _serialize(state.interactions[-1].admitted['registered_input'])['members'] == _serialize(original.submission['registered_input'])['members']


def test_first_session_acceptance_rejects_late_byte_identity_change(tmp_path, monkeypatch):
    """Unchanged scientific values cannot authorize a changed registration pin."""
    from agent.orchestration import executor
    harness = multi_input_harness(tmp_path)
    normal = executor.verify_step
    target, mutations = [], []
    def verify_after_replacement(step, *args, **kwargs):
        if step.tool_name == 'evaluate_cell_clustering' and not mutations:
            path = Path(target[0].source_path)
            with path.open('ab') as stream:
                stream.write(b'\0')
            mutations.append(hashlib.sha256(path.read_bytes()).hexdigest())
        return normal(step, *args, **kwargs)
    monkeypatch.setattr(executor, 'verify_step', verify_after_replacement)
    with TestClient(create_app(harness.service, uploads=harness.uploads)) as client:
        records = upload_sources(client, harness)
        selected = [row['resource_id'] for row in records]
        target.append(harness.service.resources.load(selected[1]))
        payload = multi_body(selected)
        with multi_input_work() as counts:
            result = execute(client, payload)
        assert counts['production.leiden'] == counts['production.evaluation'] == 1
        assert counts['verification.evaluation'] > 0 and mutations
        assert mutations[0] != target[0].source_sha256
        assert result['revision_id'] is None and result['response']['status'] == 'failed', result
        assert result['response']['error']['code'] == 'LOCAL_RESOURCE_INTEGRITY_INVALID'
        run = harness.service._application.run_store.load(result['run_id'])
        assert run.lifecycle_status is RunLifecycleStatus.SUCCEEDED
        assert all(step.verification.passed for step in run.steps)
        state = harness.service._application.sessions.load('session')
        assert not state.revisions and state.generation == 0
        calls_before = sum(len(model.calls) for model in harness.models)
        with multi_input_work() as retries:
            assert execute(client, payload) == result
            assert client.get('/api/v1/sessions/session').json()['revisions'] == []
            assert harness.service._application.sessions.recover('session', 'first') == state
        assert not retries and sum(len(model.calls) for model in harness.models) == calls_before


def test_accepted_completed_history_outlives_archived_selected_sources(tmp_path):
    harness = multi_input_harness(tmp_path)
    with TestClient(create_app(harness.service, uploads=harness.uploads)) as client:
        records = upload_sources(client, harness)
        selected = [row['resource_id'] for row in records]
        payload = multi_body(selected)
        result = execute(client, payload)
        assert result['revision_id'] and result['response']['status'] == 'activated', result
        path = f"/api/v1/sessions/session/revisions/{result['revision_id']}"
        evidence = client.get(path + '/evidence', params={'output_name': 'evaluation'}).json()
        history = client.get('/api/v1/sessions/session').json()
        for resource_id in selected:
            source = Path(harness.service.resources.load(resource_id).source_path)
            source.rename(source.with_suffix('.archived'))
        calls_before = sum(len(model.calls) for model in harness.models)
        with multi_input_work() as historical:
            assert execute(client, payload) == result
            assert client.get('/api/v1/sessions/session').json() == history
            assert client.get(path + '/evidence', params={'output_name': 'evaluation'}).json() == evidence
        assert not historical and sum(len(model.calls) for model in harness.models) == calls_before
    def forbidden(*args, **kwargs):
        raise AssertionError('Archived completed input history invoked a provider or science.')
    default = build_default_tool_registry()
    registry = ToolRegistry(tuple(replace(default.get(name), function=forbidden) for name in default.names()))
    reopened = InteractiveAgentApplication(harness.service._application.workspace_root,
        model_profiles=(PROFILE,), default_profile_id=PROFILE.profile_id, registry=registry,
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': forbidden}),
        approved_source_roots=(harness.uploads.root,))
    with multi_input_work() as restored:
        with TestClient(create_app(reopened, uploads=H5ADUploadAdmission(reopened.resources, harness.uploads.root))) as client:
            assert execute(client, payload) == result
            assert client.get('/api/v1/sessions/session').json() == history
            assert client.get(path + '/evidence', params={'output_name': 'evaluation'}).json() == evidence
    assert not restored


def test_pending_answer_cannot_switch_selected_datasets(tmp_path):
    harness = multi_input_harness(tmp_path, variant='missing_label')
    with TestClient(create_app(harness.service, uploads=harness.uploads)) as client:
        records = upload_sources(client, harness)
        selected = [row['resource_id'] for row in records]
        pending = execute(client, multi_body(selected, utterance=OBJECTIVE))
        assert pending['response']['kind'] == 'clarify', pending
        with multi_input_work() as counts:
            response = client.post('/api/v1/sessions/session/turns',
                json=multi_body(list(reversed(selected)), turn='answer', utterance=ANSWER))
            if response.status_code == 202:
                result = wait_turn(client, 'answer', timeout=60)
                assert result['revision_id'] is None, result
            else:
                assert response.status_code in (400, 409), response.text
        assert_no_owner(counts)
        assert not harness.service._application.sessions.load('session').revisions


def test_unrelated_new_task_cannot_inherit_captured_source_roles(tmp_path):
    harness = multi_input_harness(tmp_path, variant='missing_label')
    with TestClient(create_app(harness.service, uploads=harness.uploads)) as client:
        records = upload_sources(client, harness)
        pending = execute(client, multi_body([row['resource_id'] for row in records], utterance=OBJECTIVE))
        assert pending['response']['kind'] == 'clarify', pending
        with multi_input_work() as counts:
            result = execute(client, multi_body(turn='unrelated', utterance='Inspect an unrelated dataset.'))
        assert result['revision_id'] is None, result
        assert_no_owner(counts)
        state = harness.service._application.sessions.load('session')
        assert not state.revisions
        admitted = state.interactions[-1].admitted
        assert admitted['inputs'] == {}
        assert 'roles' not in admitted and 'registered_input' not in admitted
