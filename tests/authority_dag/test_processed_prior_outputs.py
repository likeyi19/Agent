"""UA3.6.2a: genuine processed science through shared historical binding."""
from collections import Counter
from contextlib import contextmanager
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pytest

from agent.application import OutputSelection, ResearchAgentApplication
from agent.application.session_state import SessionError
from agent.orchestration import (
    AgentPlan, AgentRequest, LLMPlanner, PlanExecutor, PlanStep, ToolRegistry,
    build_default_tool_registry,
)
from agent.orchestration.prior_outputs import (
    bind_output, channel_for, prior_output_store, validate_binding, validate_step_references,
)
from agent.orchestration.verification_authority import load_scientific_authority
from agent.schemas import RunLifecycleStatus, StepStatus
from agent.schemas.prior_output import PriorOutputRef
from agent.schemas.run_state import PersistedRunState, fingerprint_plan
from agent.schemas.verification_authority import AuthorityError
from agent.tools.analysis.embedding_analysis import PROVENANCE_KEY


class ScientificPlanModel:
    """Script only typed planning; every registered scientific body is real."""
    model_id = 'scripted-processed-authority'

    def __init__(self, consumer=False, handle=None, before_plan=None):
        self.consumer, self.handle, self.before_plan = consumer, handle, before_plan
        self.calls = []

    def complete(self, *, prompt, response_schema):
        value = json.loads(prompt)
        self.calls.append((value, response_schema, prompt))
        if 'selection_schema_version' in value:
            return json.dumps({'selection_schema_version': 1, 'decision': {
                'kind': 'select', 'capability_ids': ['embedding_analysis']}})
        if self.before_plan:
            self.before_plan()
        if self.consumer:
            items = value['active_outputs']
            assert len(items) == 1
            assert items[0]['producer_tool'] == 'build_cell_neighbors'
            assert items[0]['source_port'] == 'neighbors'
            source = {'target': 'analysis', 'source': {
                'kind': 'context', 'handle': self.handle or items[0]['handle']}}
            step = {'step_id': 'clustering', 'tool': 'cluster_cells',
                    'sources': [source], 'control_dependencies': []}
        else:
            step = {'step_id': 'neighbors', 'tool': 'build_cell_neighbors',
                    'sources': [{'target': 'embedding', 'source': {
                        'kind': 'input', 'input': 'embedding_path'}}], 'control_dependencies': []}
        return json.dumps({'schema_version': 4, 'decision': {'kind': 'plan', 'steps': [step]}})


class SamePlanModel(ScientificPlanModel):
    def complete(self, *, prompt, response_schema):
        value = json.loads(super().complete(prompt=prompt, response_schema=response_schema))
        if value.get('schema_version') == 4:
            value['decision']['steps'].append({'step_id': 'clustering', 'tool': 'cluster_cells',
                'sources': [{'target': 'analysis', 'source': {'kind': 'step', 'step': 'neighbors'}}],
                'control_dependencies': []})
        return json.dumps(value)


@contextmanager
def processed_work():
    """Observe actual algorithm, verifier, integrity and source-check bodies."""
    import scanpy as sc
    from agent.orchestration import semantic_compiler, verifier
    from agent.tools.analysis import embedding_analysis, neighbors_publication
    from agent.tools.data import fastq_verification_authority
    functions = {
        'production.neighbors': sc.pp.neighbors,
        'production.leiden': sc.tl.leiden,
        'verification.neighbors': verifier._verify_neighbors,
        'verification.clustering': verifier._verify_clustering,
        'owner.neighbors': neighbors_publication._verify,
        'verification.publication': neighbors_publication.verify_neighbors_publication.__wrapped__,
        'source.capture': neighbors_publication.capture_inputs,
        'source.check': neighbors_publication.check_sources,
        'source.embeddings': embedding_analysis._load_embeddings,
        'source.cell_ids': embedding_analysis._load_cell_ids,
        'validation.graph_structure': embedding_analysis._validate_neighbors_artifact,
        'artifact.analysis_reads': embedding_analysis._read_analysis,
        'integrity.files': fastq_verification_authority.file_closure,
        'compiler': semantic_compiler.compile_semantic_plan,
        'preflight': PlanExecutor.preflight,
    }
    codes = {function.__code__: name for name, function in functions.items()}
    counts = Counter()
    previous = sys.getprofile()
    def observe(frame, event, arg):
        if event == 'call' and frame.f_code in codes:
            counts[codes[frame.f_code]] += 1
    sys.setprofile(observe)
    try:
        yield counts
    finally:
        sys.setprofile(previous)


@pytest.fixture
def processed_source(tmp_path):
    # The established bounded analysis fixture: two clusters, 512 float32 axes.
    # Include non-ASCII IDs to witness exact persisted ordering without relabeling.
    rng = np.random.default_rng(17)
    values = np.vstack((rng.normal(-1, .25, (16, 512)),
                        rng.normal(1, .25, (16, 512)))).astype(np.float32)
    ids = tuple(f'细胞-{index:02d}' for index in range(32))
    embedding = tmp_path / 'cells.epizoo_embeddings.npy'
    cell_ids = tmp_path / 'cells.epizoo_obs_names.txt'
    np.save(embedding, values, allow_pickle=False)
    cell_ids.write_text(''.join(f'{cell}\n' for cell in ids), encoding='utf-8')
    model = ScientificPlanModel()
    app = ResearchAgentApplication(tmp_path / 'application', planner=LLMPlanner(model))
    app.sessions.create('analysis')
    with processed_work() as counts:
        session = app.sessions.run('analysis', 'original', AgentRequest('neighbors-original',
            'Build a neighbor graph from the explicitly supplied aligned embedding and cell IDs.',
            {'embedding_path': str(embedding), 'cell_ids_path': str(cell_ids)}),
            (OutputSelection('neighbors', 'neighbors', 'analysis_path'),), expected_generation=0)
    run = app.run_store.load('neighbors-original:run')
    assert session.turn('original').status == 'activated', run.to_dict()
    assert run.lifecycle_status is RunLifecycleStatus.SUCCEEDED
    assert counts['production.neighbors'] == 1 and counts['production.leiden'] == 0
    assert run.steps[0].verification.passed
    binding = bind_output(app.run_store, app.registry, run.run_id, 'neighbors', 'neighbors')
    return SimpleNamespace(app=app, model=model, session=session, run=run, binding=binding,
                           counts=counts, values=values, ids=ids, embedding=embedding, cell_ids=cell_ids)


def consumer(source, *, request_id='later-clustering', handle=None, before_plan=None):
    model = ScientificPlanModel(consumer=True, handle=handle, before_plan=before_plan)
    app = ResearchAgentApplication(source.app.workspace_root, planner=LLMPlanner(model))
    request = AgentRequest(request_id, 'Cluster the exact offered neighbor graph with Leiden.', {})
    outputs = (OutputSelection('clustering', 'clustering', 'analysis_path'),)
    return app, model, request, outputs


def preserved_run_bytes(source):
    paths = source.app._workspace.run_paths(source.run.run_id)
    files = {path: path.read_bytes() for root in (paths.scientific, paths.evidence,
                                                paths.visualizations, paths.report)
             for path in root.rglob('*') if path.is_file()}
    state_path = source.app.run_store.state_path(source.run.run_id)
    files[state_path] = state_path.read_bytes()
    files[source.embedding] = source.embedding.read_bytes()
    files[source.cell_ids] = source.cell_ids.read_bytes()
    return files


def test_real_separate_runs_compile_preflight_execute_and_publish_revision(processed_source):
    source = processed_source
    before = preserved_run_bytes(source)
    app, model, request, outputs = consumer(source)
    with processed_work() as counts:
        session = app.sessions.run('analysis', 'later', request, outputs, expected_generation=1,
                                   use_active_context=True)
    run = app.run_store.load(request.request_id + ':run')
    assert session.turn('later').status == 'activated', run.to_dict()
    assert run.lifecycle_status is RunLifecycleStatus.SUCCEEDED
    assert counts['production.neighbors'] == 0 and counts['production.leiden'] == 1
    assert counts['owner.neighbors'] == 0
    assert counts['source.capture'] == counts['source.check'] == counts['source.embeddings'] == counts['source.cell_ids'] == 0
    assert counts['verification.clustering'] > 0
    assert counts['integrity.files'] > 0 and counts['compiler'] > 0 and counts['preflight'] > 0
    assert len(run.plan.steps) == 1 and run.plan.steps[0].tool_name == 'cluster_cells'
    reference = run.plan.steps[0].arguments['analysis_path']
    assert isinstance(reference, PriorOutputRef) and reference.binding == source.binding
    assert reference.output_key == 'analysis_path'
    assert run.steps[0].resolved_arguments['analysis_path'] == source.run.steps[0].result['analysis_path']
    assert run.steps[0].verification.passed and run.steps[0].result['n_clusters'] == 2
    assert run.steps[0].verification.artifact_authority is None  # This owner is not qualified here.
    assert PersistedRunState.from_dict(run.to_dict()) == run
    assert len(session.revisions) == 2 and session.generation == 2
    assert session.revisions[-1].parent_revision_id == session.revisions[0].revision_id
    assert session.revisions[0].outputs == source.session.revisions[0].outputs
    assert session.revisions[-1].outputs[0].run_id == run.run_id
    original = ad.read_h5ad(source.run.steps[0].result['analysis_path'])
    clustered = ad.read_h5ad(run.steps[0].result['analysis_path'])
    assert tuple(original.obs_names) == tuple(clustered.obs_names) == source.ids
    np.testing.assert_array_equal(original.obsm['X_epizoo'], source.values)
    np.testing.assert_array_equal(clustered.obsm['X_epizoo'], source.values)
    for key in ('connectivities', 'distances'):
        assert (original.obsp[key] != clustered.obsp[key]).nnz == 0
    assert clustered.uns[PROVENANCE_KEY]['parameters']['clustering']['resolution'] == 1.0
    evidence = json.loads((app._workspace.run_paths(run.run_id).evidence / 'analysis_evidence.json').read_bytes())
    assert len(evidence['provenance']['prior_outputs']) == 1
    assert all(path.read_bytes() == value for path, value in before.items())
    for _, _, prompt in model.calls:
        assert str(app.workspace_root) not in prompt
        assert source.run.run_id not in prompt
        assert source.binding.authority_sha256 not in prompt
    # The existing fingerprint includes exact historical execution identities.
    changed = replace(reference.binding, accepted_step_sha256='0' * 64)
    other = replace(run.plan, steps=(replace(run.plan.steps[0], arguments={
        **run.plan.steps[0].arguments, 'analysis_path': PriorOutputRef(changed, 'analysis_path')}),))
    assert fingerprint_plan(other) != fingerprint_plan(run.plan)
    # Record actual totals, without claiming wrapper/metadata checks are science.
    print('processed cross-run counts:', json.dumps({'producer': dict(source.counts),
          'consumer': dict(counts)}, sort_keys=True))
    authority = source.run.steps[0].verification.to_dict()['artifact_authority']
    publication = json.loads(Path(source.run.steps[0].result['manifest_path']).read_bytes())
    witness = dict(producer=dict(run_id=source.run.run_id, step_id='neighbors',
        tool='build_cell_neighbors', source_port=source.binding.source_port,
        semantic_type=source.binding.semantic_type, accepted_step_sha256=source.binding.accepted_step_sha256,
        manifest_path=source.run.steps[0].result['manifest_path'], manifest_sha256=source.binding.manifest_sha256,
        analysis_path=source.run.steps[0].result['analysis_path'], analysis_sha256=publication['analysis']['sha256'],
        execution_identity=authority['execution_identity'], authority_sha256=source.binding.authority_sha256,
        science_profile=authority['science_profile'], verifier=authority['verifier']),
        consumer=dict(run_id=run.run_id, step_id='clustering', tool='cluster_cells',
                      analysis_path=run.steps[0].result['analysis_path'], n_clusters=run.steps[0].result['n_clusters']),
        revisions=[revision.revision_id for revision in session.revisions],
        counts=dict(producer=dict(source.counts), consumer=dict(counts)))
    witness_path = source.embedding.parent / 'cross-run-witness.json'
    witness_path.write_text(json.dumps(witness, indent=2, sort_keys=True), encoding='utf-8')
    print('processed cross-run identities:', json.dumps(witness, sort_keys=True))


def test_navigation_and_retry_never_create_authority_or_rerun_neighbors(processed_source):
    source = processed_source
    app, model, request, outputs = consumer(source)
    state = app.sessions.run('analysis', 'later', request, outputs, expected_generation=1,
                             use_active_context=True)
    producer_bytes = preserved_run_bytes(source)
    calls = len(model.calls)
    with processed_work() as counts:
        app.sessions.switch('analysis', 'back', state.revisions[0].revision_id, expected_generation=2)
        app.sessions.switch('analysis', 'forward', state.revisions[1].revision_id, expected_generation=3)
        duplicate = app.sessions.run('analysis', 'later', request, outputs, expected_generation=1,
                                     use_active_context=True)
    assert not counts and len(model.calls) == calls
    assert duplicate.turn('later').status == 'activated'
    assert all(path.read_bytes() == value for path, value in producer_bytes.items())
    with pytest.raises(AuthorityError, match='No recorded scientific authority'):
        bind_output(app.run_store, app.registry, 'later-clustering:run', 'clustering', 'clustered_analysis')


@pytest.mark.parametrize('field,value', [
    ('run_id', 'missing:run'), ('step_id', 'missing'), ('tool_name', 'cluster_cells'),
    ('source_port', 'clustered_analysis'), ('semantic_type', 'clustered_analysis.v1'),
    ('accepted_step_sha256', '0' * 64), ('manifest_sha256', '0' * 64),
    ('authority_sha256', '0' * 64),
])
def test_forged_historical_identity_fails_without_science(processed_source, field, value):
    source = processed_source
    with processed_work() as counts, pytest.raises((ValueError, RuntimeError)) as rejected:
        validate_binding(replace(source.binding, **{field: value}), store=source.app.run_store,
                         registry=source.app.registry, integrity=True)
    assert not {name: count for name, count in counts.items() if name.startswith(('production.', 'owner.'))}
    assert source.app.sessions.load('analysis').generation == 1
    print('rejected historical identity:', field, type(rejected.value).__name__, str(rejected.value))


@pytest.mark.parametrize('case', ['missing_authority', 'legacy_authority', 'failed_run', 'unaccepted_step',
                                  'run_identity', 'resolved_arguments', 'unsupported_verifier',
                                  'incomplete_authority', 'invalid_lineage', 'invalid_proof_metadata',
                                  'missing_member'])
def test_unqualified_source_fails_without_implicit_upgrade(processed_source, case):
    source = processed_source
    state, step = source.run, source.run.steps[0]
    if case == 'failed_run':
        changed = SimpleNamespace(**(vars(state) | {'lifecycle_status': RunLifecycleStatus.FAILED}))
    elif case == 'run_identity':
        changed = SimpleNamespace(**(vars(state) | {'run_id': 'different:run'}))
    else:
        attributes = {}
        if case == 'missing_authority':
            attributes['verification'] = replace(step.verification, artifact_authority=None)
        elif case == 'unaccepted_step':
            attributes['status'] = StepStatus.FAILED
        elif case == 'resolved_arguments':
            attributes['resolved_arguments'] = dict(step.resolved_arguments) | {'n_neighbors': 3}
        elif case == 'missing_member':
            attributes['result'] = {key: value for key, value in step.result.items() if key != 'analysis_path'}
        else:
            authority = json.loads(json.dumps(step.verification.to_dict()['artifact_authority']))
            if case == 'unsupported_verifier':
                authority['verifier']['compatibility_version'] = 'unsupported'
            elif case == 'incomplete_authority':
                authority['resources'] = {}
                with pytest.raises(AuthorityError, match='upstream/resource authority'):
                    replace(step.verification, artifact_authority=authority)
                return
            elif case == 'legacy_authority':
                authority['schema_version'] = 1
                authority['integrity_scope'] = 'artifact_integrity_lineage.v1'
                authority.pop('source_policy')
                authority.pop('historical_sources')
            elif case == 'invalid_proof_metadata':
                authority['resources']['result_metadata']['n_cells'] = 10000
            else:
                authority['upstream'] = {'forged': {'manifest_sha256': '0' * 64}}
            attributes['verification'] = replace(step.verification, artifact_authority=authority)
        changed = SimpleNamespace(**(vars(state) | {'steps': (replace(step, **attributes),)}))
    # A loader view exercises rejected persisted state without modifying accepted history.
    store = SimpleNamespace(load=lambda _: changed)
    before = source.app.run_store.state_path(state.run_id).read_bytes()
    with processed_work() as counts, pytest.raises((AuthorityError, ValueError)) as rejected:
        binding = bind_output(store, source.app.registry, state.run_id, 'neighbors', 'neighbors')
        validate_binding(binding, store=store, registry=source.app.registry, integrity=True)
    assert not {name: count for name, count in counts.items() if name.startswith(('production.', 'owner.'))}
    assert source.app.run_store.state_path(state.run_id).read_bytes() == before
    if case == 'invalid_proof_metadata':
        assert isinstance(rejected.value, AuthorityError)
        assert str(rejected.value) == 'Neighbor proof metadata differs from its owner-qualified result.'
    print('rejected historical source:', case, type(rejected.value).__name__, str(rejected.value))


@pytest.mark.parametrize('member', ['manifest_path', 'n_cells', 'input_analysis_path'])
def test_unauthorized_semantic_member_fails_preflight(processed_source, member):
    source = processed_source
    step = PlanStep('clustering', 'cluster_cells', {
        'analysis_path': PriorOutputRef(source.binding, member), 'output_dir': '/tmp/unused-processed-reuse'})
    with prior_output_store(source.app.run_store):
        with pytest.raises(AuthorityError, match='member mapping'):
            validate_step_references(step, source.app.registry)
        check = PlanExecutor(source.app.registry).preflight(AgentPlan('invalid', 'invalid', 'Explicit.', (step,)))
    assert not check.passed and check.error.code == 'PRIOR_OUTPUT_INVALID'


def test_incompatible_consumer_channel_is_registry_driven(processed_source):
    source = processed_source
    legitimate = channel_for(source.binding, 'cluster_cells', 'analysis', source.app.registry)
    assert legitimate.producer_tool_name == 'build_cell_neighbors'
    assert [(member.argument_name, member.output_key) for member in legitimate.members] == [
        ('analysis_path', 'analysis_path')]
    umap = channel_for(source.binding, 'compute_cell_umap', 'analysis', source.app.registry)
    assert umap.producer_tool_name == 'build_cell_neighbors'
    with pytest.raises(AuthorityError, match='incompatible'):
        channel_for(source.binding, 'evaluate_cell_clustering', 'analysis', source.app.registry)


@pytest.mark.parametrize('target', ['analysis', 'publication'])
def test_publication_bytes_changed_fail_before_consumer_science(processed_source, target):
    source = processed_source
    field = 'analysis_path' if target == 'analysis' else 'manifest_path'
    path = Path(source.run.steps[0].result[field])
    path.write_bytes(path.read_bytes() + b' ')
    app, _, request, outputs = consumer(source)
    with processed_work() as counts, pytest.raises((ValueError, OSError, RuntimeError)):
        app.sessions.run('analysis', 'invalid', request, outputs, expected_generation=1,
                         use_active_context=True)
    assert not {name: count for name, count in counts.items() if name.startswith(('production.', 'owner.'))}
    state = app.sessions.load('analysis')
    assert state.generation == 1 and len(state.revisions) == 1
    assert state.turn('invalid').status == 'failed'


def test_historical_sources_and_explicit_current_freshness_remain_distinct(processed_source):
    source = processed_source
    original = source.cell_ids.read_bytes()
    source.cell_ids.write_text(''.join(f'{cell}\n' for cell in reversed(source.ids)), encoding='utf-8')
    # Historical proof establishes the originally verified exact ordered IDs;
    # it never silently promotes changed current sources to new authority.
    validate_binding(source.binding, store=source.app.run_store, integrity=True)
    with processed_work() as counts, pytest.raises(AuthorityError, match='Current source'):
        load_scientific_authority(source.app.run_store, source.run.run_id, 'neighbors',
                                 source_policy='current_source_freshness.v1')
    assert not {name: count for name, count in counts.items() if name.startswith(('production.', 'owner.'))}
    source.cell_ids.write_bytes(original)


def test_unknown_context_and_foreign_revision_are_not_substituted(processed_source):
    source = processed_source
    app, model, request, outputs = consumer(source, handle='ctx.1')
    with processed_work() as counts:
        state = app.sessions.run('analysis', 'invalid', request, outputs, expected_generation=1,
                                 use_active_context=True)
    assert state.turn('invalid').status == 'failed'
    assert state.generation == 1 and len(state.revisions) == 1
    assert not {name: count for name, count in counts.items() if name.startswith(('production.', 'owner.'))}
    # A separate Session has no selected Revision: it cannot capture or retain
    # the neighbor output just because the RunStore holds a successful run.
    app.sessions.create('foreign')
    with pytest.raises(SessionError, match='captured active revision'):
        app.sessions.start_turn('foreign', 'bad-retain', replace(request, request_id='foreign'), outputs,
                                expected_generation=0, retain=('neighbors',))
    app.sessions.start_turn('foreign', 'empty', replace(request, request_id='empty'), outputs,
                            expected_generation=0)
    with pytest.raises(SessionError):
        app.sessions.active_context('foreign', 'empty')


def test_publication_changes_after_capture_fail_normal_preflight(processed_source):
    source = processed_source
    path = Path(source.run.steps[0].result['analysis_path'])
    changed = []
    def mutate():
        if not changed:
            path.write_bytes(path.read_bytes() + b' ')
            changed.append(True)
    app, _, request, outputs = consumer(source, before_plan=mutate)
    with processed_work() as counts:
        state = app.sessions.run('analysis', 'invalid', request, outputs, expected_generation=1,
                                 use_active_context=True)
    assert state.turn('invalid').status == 'failed'
    assert state.generation == 1 and len(state.revisions) == 1
    run = app.run_store.load(request.request_id + ':run')
    assert run.lifecycle_status is RunLifecycleStatus.FAILED
    assert not {name: count for name, count in counts.items() if name.startswith(('production.', 'owner.'))}


@pytest.mark.parametrize('mutation', ['missing_provenance', 'invalid_provenance', 'cell_order'])
def test_changed_scientific_identity_is_rejected_before_history_consumption(processed_source, mutation):
    source = processed_source
    path = Path(source.run.steps[0].result['analysis_path'])
    artifact = ad.read_h5ad(path)
    if mutation == 'missing_provenance':
        del artifact.uns[PROVENANCE_KEY]
    elif mutation == 'invalid_provenance':
        artifact.uns[PROVENANCE_KEY]['parameters']['neighbors']['use_rep'] = 'unqualified'
    else:
        artifact.obs_names = list(reversed(source.ids))
    artifact.write_h5ad(path)
    with processed_work() as counts, pytest.raises((AuthorityError, ValueError)):
        validate_binding(source.binding, store=source.app.run_store, integrity=True)
    assert not {name: count for name, count in counts.items() if name.startswith(('production.', 'owner.'))}
    assert source.app.sessions.load('analysis').generation == 1


def test_fresh_process_terminal_recovery_never_reopens_historical_sources(processed_source, tmp_path):
    source = processed_source
    app, _, request, outputs = consumer(source)
    state = app.sessions.run('analysis', 'later', request, outputs, expected_generation=1,
                             use_active_context=True)
    producer = preserved_run_bytes(source)
    consumer_state = app.run_store.state_path(request.request_id + ':run').read_bytes()
    source.embedding.rename(source.embedding.with_suffix('.archived'))
    source.cell_ids.rename(source.cell_ids.with_suffix('.archived'))
    result_path = tmp_path / 'terminal-recovery-counts.json'
    script = '''
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'tests/authority_dag'))
from test_processed_prior_outputs import processed_work
from agent.application import ResearchAgentApplication
app = ResearchAgentApplication(sys.argv[1])
with processed_work() as calls:
    result = app.resume('later-clustering:run')
assert result.status.value == 'SUCCEEDED', result.to_dict()
assert app.sessions.recover('analysis', 'later').turn('later').status == 'activated'
Path(sys.argv[2]).write_text(json.dumps(dict(calls)), encoding='utf-8')
'''
    child = subprocess.run([sys.executable, '-B', '-c', script, str(app.workspace_root), str(result_path)],
                           capture_output=True, text=True, env=os.environ.copy(), timeout=60)
    assert child.returncode == 0, child.stdout + child.stderr
    counts = json.loads(result_path.read_text(encoding='utf-8'))
    assert not {name: count for name, count in counts.items() if name.startswith(('production.', 'owner.', 'source.'))}
    assert app.run_store.state_path(request.request_id + ':run').read_bytes() == consumer_state
    assert app.sessions.load('analysis') == state
    for path, value in producer.items():
        if path not in (source.embedding, source.cell_ids):
            assert path.read_bytes() == value
    print('fresh-process terminal counts:', json.dumps(counts, sort_keys=True))


def test_accepted_same_plan_neighbor_recovery_reuses_proof_without_production(processed_source, tmp_path, monkeypatch):
    source = processed_source
    app = ResearchAgentApplication(tmp_path / 'partial-application', planner=LLMPlanner(SamePlanModel()))
    app.sessions.create('partial')
    request = AgentRequest('partial-neighbors', 'Build neighbors and cluster the resulting graph.',
        {'embedding_path': str(source.embedding), 'cell_ids_path': str(source.cell_ids)})
    outputs = (OutputSelection('neighbors', 'neighbors', 'analysis_path'),
               OutputSelection('clustering', 'clustering', 'analysis_path'))
    class ProcessExit(BaseException):
        pass
    update = app.run_store.update
    def interrupt(state, *, expected_revision):
        saved = update(state, expected_revision=expected_revision)
        if any(step.step_id == 'neighbors' and step.status is StepStatus.SUCCEEDED for step in saved.steps):
            raise ProcessExit()
        return saved
    monkeypatch.setattr(app.run_store, 'update', interrupt)
    with processed_work() as initial, pytest.raises(ProcessExit):
        app.sessions.run('partial', 'initial', request, outputs, expected_generation=0)
    pending = app.run_store.load(request.request_id + ':run')
    neighbor = pending.steps[0]
    assert neighbor.verification.artifact_authority is not None
    assert initial['production.neighbors'] == 1 and initial['production.leiden'] == 0
    scientific = app._workspace.run_paths(pending.run_id).scientific
    before = {path: path.read_bytes() for path in scientific.rglob('*') if path.is_file()}
    # Accepted source policy applies when a completed same-plan owner is restored.
    source.embedding.rename(source.embedding.with_suffix('.archived'))
    source.cell_ids.rename(source.cell_ids.with_suffix('.archived'))
    fresh = ResearchAgentApplication(app.workspace_root)
    with processed_work() as resumed:
        result = fresh.resume(pending.run_id)
    assert result.status.value == 'SUCCEEDED', result.to_dict()
    assert resumed['production.neighbors'] == resumed['owner.neighbors'] == 0
    assert resumed['production.leiden'] == 1
    assert not {name: count for name, count in resumed.items() if name.startswith('source.')}
    after = fresh.run_store.load(pending.run_id)
    assert after.plan == pending.plan and after.plan_fingerprint == pending.plan_fingerprint
    assert after.steps[0].result == neighbor.result
    assert after.steps[0].verification.artifact_authority == neighbor.verification.artifact_authority
    assert all(path.read_bytes() == raw for path, raw in before.items())
    state = fresh.sessions.complete_presentation('partial', 'initial')
    assert state.turn('initial').status == 'activated'
    print('same-plan accepted recovery counts:', json.dumps({'initial': dict(initial),
          'resume': dict(resumed)}, sort_keys=True))


@pytest.mark.parametrize('interrupted', [False, True])
def test_legacy_terminal_and_unfinished_read_never_mint_authority(processed_source, tmp_path, monkeypatch, interrupted):
    source = processed_source
    registry = build_default_tool_registry()
    original = registry.get('build_cell_neighbors')
    def legacy_result(**arguments):
        # Run the genuine numerical owner, expose its original result contract,
        # and leave publication qualification absent from this legacy Registry.
        result = original.function(**arguments)
        return {name: value for name, value in result.items() if name not in {
            'artifact_type', 'contract_version', 'manifest_path', 'manifest_sha256'}}
    legacy = ToolRegistry(tuple(replace(registry.get(name), function=legacy_result,
        scientific_authority=None) if name == original.name else registry.get(name)
        for name in registry.names()))
    app = ResearchAgentApplication(tmp_path / 'legacy-application',
        planner=LLMPlanner(SamePlanModel() if interrupted else ScientificPlanModel()), registry=legacy)
    request = AgentRequest('legacy-neighbors', 'Build a neighbor graph from explicit aligned embeddings.',
        {'embedding_path': str(source.embedding), 'cell_ids_path': str(source.cell_ids)})
    class ProcessExit(BaseException):
        pass
    if interrupted:
        update = app.run_store.update
        def interrupt(state, *, expected_revision):
            saved = update(state, expected_revision=expected_revision)
            if any(step.step_id == 'neighbors' and step.status is StepStatus.SUCCEEDED for step in saved.steps):
                raise ProcessExit()
            return saved
        monkeypatch.setattr(app.run_store, 'update', interrupt)
        with pytest.raises(ProcessExit):
            app.run(request)
    else:
        result = app.run(request)
        assert result.status.value == 'SUCCEEDED', result.to_dict()
    before = app.run_store.load(request.request_id + ':run')
    assert before.steps[0].verification.artifact_authority is None
    assert 'manifest_path' not in before.steps[0].result
    current = ResearchAgentApplication(app.workspace_root)
    with processed_work() as counts:
        result = current.resume(before.run_id)
    assert result.status.value == 'SUCCEEDED', result.to_dict()
    after = current.run_store.load(before.run_id)
    assert after.steps[0].verification.artifact_authority is None
    assert after.steps[0].result == before.steps[0].result
    assert counts['production.neighbors'] == counts['owner.neighbors'] == 0
    assert counts['production.leiden'] == int(interrupted)
    with pytest.raises(AuthorityError, match='No recorded scientific authority'):
        bind_output(current.run_store, current.registry, after.run_id, 'neighbors', 'neighbors')
    print('legacy read counts:', interrupted, json.dumps(dict(counts), sort_keys=True))
