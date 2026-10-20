"""UA3.6.2b: ordinary interactive turns over genuine accepted processed science."""
from contextlib import contextmanager
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from agent.application import InteractiveAgentApplication, InteractiveBoundaryError
from agent.orchestration import PlanningModelProfile
from agent.providers import PlanningModelFactoryRegistry
from agent.schemas import RunLifecycleStatus, StepOutputRef
from agent.schemas.prior_output import PriorOutputRef
from agent.tools.analysis.embedding_analysis import PROVENANCE_KEY
from test_processed_prior_outputs import processed_source, processed_work, preserved_run_bytes


MODIFICATION = 'Change the clustering resolution to 0.5 and update the UMAP.'
PROFILE = PlanningModelProfile('conversation', 'scripted', 'scripted/conversation')


@contextmanager
def conversational_work():
    """Extend the existing actual-body accounting to the real UMAP owner."""
    import scanpy as sc
    from agent.orchestration import verifier
    codes = {id(inspect.unwrap(sc.tl.umap).__code__): 'production.umap',
             id(verifier._verify_umap.__code__): 'verification.umap'}
    with processed_work() as counts:
        inherited = sys.gettrace()
        def observe(frame, event, arg):
            inherited(frame, event, arg)
            if event == 'call' and id(frame.f_code) in codes:
                counts[codes[id(frame.f_code)]] += 1
        sys.settrace(observe)
        try:
            yield counts
        finally:
            sys.settrace(inherited)


def tiny_other_input(path):
    value = ad.AnnData(sparse.csr_matrix(np.arange(24).reshape(6, 4), dtype=np.int64),
        obs=pd.DataFrame(index=[f'other-{index}' for index in range(6)]),
        var=pd.DataFrame(index=[f'feature-{index}' for index in range(4)]))
    value.write_h5ad(path)
    return path


class ConversationalModel:
    """Script provider semantics only; scientific execution stays registered."""
    model_id = PROFILE.model_id

    def __init__(self, action='modify', *, retain=False, handle='ctx.0', before_interpret=None,
                 retention_error=None, literal='0.5'):
        self.action, self.retain, self.handle = action, retain, handle
        self.before_interpret, self.calls = before_interpret, []
        self.retention_error, self.literal = retention_error, literal

    def complete(self, *, prompt, response_schema):
        value = json.loads(prompt)
        if 'grounding_review_version' in value:
            return json.dumps(dict(verdicts={row['id']: 'entailed' for row in value['paragraphs']}))
        self.calls.append((value, response_schema, prompt))
        if 'turn_schema_version' in value:
            if self.before_interpret:
                self.before_interpret()
            if self.action == 'question':
                decision = dict(kind='answer_scientific', target=dict(output='r0', subject=None),
                                comparison=None, focus='question')
            elif self.action == 'navigate':
                decision = dict(kind='navigate', relation='parent')
            elif self.action == 'ambiguous':
                decision = dict(kind='clarify', reason='ambiguous_subject')
            elif self.action == 'new':
                decision = dict(kind='execute_plan', target='inspect_scATAC', context='none')
            elif self.action == 'incompatible':
                decision = dict(kind='execute_plan', target='cluster_cells', context='none')
            else:
                literal = self.literal
                start = value['utterance'].index(literal)
                decision = dict(kind='execute_plan', target='compute_cell_umap', context='current', arguments=[dict(
                    tool='cluster_cells', argument='resolution', literal=literal, start=start, end=start+3)])
            return json.dumps(dict(turn_schema_version=1, decision=decision))
        if 'selection_schema_version' in value:
            capabilities = ['processed_inspection'] if self.action == 'new' else ['embedding_analysis']
            return json.dumps(dict(selection_schema_version=1,
                                   decision=dict(kind='select', capability_ids=capabilities)))
        if 'output_selection_schema_version' in value:
            if self.action == 'new':
                outputs = [dict(name='inspection', step_id='inspect', output_key='n_cells')]
            elif self.action == 'incompatible':
                outputs = [dict(name='clusters', step_id='cluster', output_key='analysis_path')]
            else:
                outputs = [dict(name='clusters', step_id='cluster', output_key='analysis_path'),
                           dict(name='umap', step_id='umap', output_key='analysis_path')]
            response = dict(outputs=outputs)
            if self.action not in ('new', 'incompatible'):
                assert value['output_selection_schema_version'] == 2
                offered = value['retained_outputs']
                assert len(offered) == 1 and offered[0]['handle'] == 'retain.0'
                assert offered[0]['context_handle'] == 'ctx.0'
                response['retain'] = ['retain.0'] if self.retain else []
                if self.retention_error == 'unknown':
                    response['retain'] = ['retain.unoffered']
                elif self.retention_error == 'duplicate':
                    response['retain'] = ['retain.0', 'retain.0']
                elif self.retention_error == 'name_collision':
                    response['outputs'][0]['name'] = 'neighbors'
                    response['retain'] = ['retain.0']
                elif self.retention_error == 'total_bound':
                    response['outputs'] = [dict(name=f'selected-{index}', step_id='cluster',
                                               output_key='analysis_path') for index in range(32)]
                    response['retain'] = ['retain.0']
            return json.dumps(response)
        if 'dialogue_schema_version' in value:
            claims = [claim['claim_id'] for claim in value['evidence']['claims'] if claim['field'] == 'n_cells']
            assert len(claims) == 1
            return json.dumps(dict(support='supported', paragraphs=[dict(text=f"The accepted cell count is {next(c['value'] for c in value['evidence']['claims'] if c['claim_id'] == claims[0])}.", kind='finding', citations=[claims[0]], comparisons=[])]))
        assert value['semantic_prompt_version']
        if self.action == 'new':
            assert not value.get('active_outputs')
            steps = [dict(step_id='inspect', tool='inspect_scATAC', sources=[], control_dependencies=[])]
        elif self.action == 'incompatible':
            assert not value.get('active_outputs')
            steps = [dict(step_id='cluster', tool='cluster_cells', sources=[dict(target='analysis',
                source=dict(kind='input', input='analysis_path'))], control_dependencies=[])]
        else:
            items = value['active_outputs']
            assert len(items) == 1
            assert items[0]['producer_tool'] == 'build_cell_neighbors'
            assert items[0]['source_port'] == 'neighbors'
            steps = [dict(step_id='cluster', tool='cluster_cells', sources=[dict(target='analysis',
                source=dict(kind='context', handle=self.handle))], control_dependencies=[]),
                dict(step_id='umap', tool='compute_cell_umap', sources=[dict(target='analysis',
                source=dict(kind='step', step='cluster'))], control_dependencies=[])]
        return json.dumps(dict(schema_version=4, decision=dict(kind='plan', steps=steps)))

    def close(self):
        pass


def interactive(source, *, action='modify', retain=False, handle='ctx.0', before_interpret=None,
                retention_error=None, literal='0.5'):
    models = []
    def factory(profile):
        assert profile == PROFILE
        model = ConversationalModel(action, retain=retain, handle=handle, before_interpret=before_interpret,
                                    retention_error=retention_error, literal=literal)
        models.append(model)
        return model
    facade = InteractiveAgentApplication(source.app.workspace_root, model_profiles=(PROFILE,),
        default_profile_id=PROFILE.profile_id,
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': factory}))
    return facade, models


def accepted_run(facade, turn_id):
    state = facade._application.sessions.load('analysis')
    return facade._application.run_store.load(state.turn(turn_id).run_id)


def assert_preserved(before):
    assert all(path.read_bytes() == original for path, original in before.items())


def persist_model_captures(directory, calls):
    captures = []
    for value, schema, prompt in calls:
        if 'turn_schema_version' in value:
            role = 'interpreter'
        elif 'selection_schema_version' in value:
            role = 'scope_selector'
        elif 'output_selection_schema_version' in value:
            role = 'output_selector'
        else:
            role = 'planner'
        canonical_schema = json.dumps(schema, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
        prompt_bytes, schema_bytes = len(prompt.encode('utf-8')), len(canonical_schema.encode('utf-8'))
        captures.append(dict(role=role, prompt=prompt, schema=schema,
            canonical_schema_json=canonical_schema, prompt_bytes=prompt_bytes,
            schema_bytes=schema_bytes, size_bytes=prompt_bytes+schema_bytes))
    path = directory / 'conversational-model-captures.json'
    path.write_text(json.dumps(captures, indent=2, sort_keys=True, ensure_ascii=False), encoding='utf-8')
    return path, [{key: capture[key] for key in ('role', 'prompt_bytes', 'schema_bytes', 'size_bytes')}
                  for capture in captures]


@pytest.mark.parametrize('retain', [False, True])
def test_ordinary_resolution_change_reuses_neighbors_and_runs_real_leiden_umap(processed_source, retain):
    source = processed_source
    before = preserved_run_bytes(source)
    facade, models = interactive(source, retain=retain)
    with conversational_work() as counts:
        view = facade.submit_turn('analysis', 'modify', MODIFICATION, expected_generation=1)
    assert view.status == 'succeeded', view.to_dict()
    assert view.response.kind == 'execute' and view.revision_id
    run = accepted_run(facade, 'modify')
    assert run.lifecycle_status is RunLifecycleStatus.SUCCEEDED
    assert tuple(step.tool_name for step in run.plan.steps) == ('cluster_cells', 'compute_cell_umap')
    cluster, umap = run.plan.steps
    reference = cluster.arguments['analysis_path']
    assert isinstance(reference, PriorOutputRef) and reference.binding == source.binding
    assert reference.output_key == 'analysis_path'
    assert cluster.arguments['resolution'] == .5
    assert umap.arguments['analysis_path'] == StepOutputRef('cluster', 'analysis_path')
    assert umap.depends_on == ('cluster',)
    assert counts['production.neighbors'] == counts['owner.neighbors'] == 0
    assert counts['production.leiden'] == counts['production.umap'] == 1
    assert not {name: count for name, count in counts.items() if name.startswith('source.')}
    assert counts['verification.clustering'] > 0 and counts['verification.umap'] > 0
    assert counts['integrity.files'] > 0 and counts['compiler'] == 1 and counts['preflight'] > 0
    assert all(step.verification.passed and step.verification.artifact_authority is None for step in run.steps)
    state = facade._application.sessions.load('analysis')
    assert state.generation == 2 and len(state.revisions) == 2
    assert state.revisions[0] == source.session.revisions[0]
    revision = state.revisions[-1]
    assert revision.parent_revision_id == source.session.revisions[0].revision_id
    assert tuple(output.name for output in revision.outputs) == (
        ('clusters', 'umap', 'neighbors') if retain else ('clusters', 'umap'))
    assert state.turn('modify').retained_outputs == (source.session.revisions[0].outputs if retain else ())
    if retain:
        assert revision.outputs[-1] == source.session.revisions[0].outputs[0]
    interaction = next(item for item in state.interactions if item.turn_id == 'modify')
    assert interaction.base_revision_id == source.session.active_revision_id
    assert interaction.base_generation == 1
    declaration = interaction.admitted['arguments'][0]
    assert declaration == dict(tool='cluster_cells', argument='resolution', literal='0.5',
                              start=MODIFICATION.index('0.5'), end=MODIFICATION.index('0.5')+3)
    clustered = ad.read_h5ad(run.steps[0].result['analysis_path'])
    visualized = ad.read_h5ad(run.steps[1].result['analysis_path'])
    original = ad.read_h5ad(source.run.steps[0].result['analysis_path'])
    assert tuple(clustered.obs_names) == tuple(visualized.obs_names) == source.ids
    for artifact in (clustered, visualized):
        np.testing.assert_array_equal(artifact.obsm['X_epizoo'], source.values)
        for key in ('distances', 'connectivities'):
            assert sparse.issparse(artifact.obsp[key]) and np.isfinite(artifact.obsp[key].data).all()
        assert artifact.uns[PROVENANCE_KEY]['parameters']['clustering']['resolution'] == .5
    for key in ('distances', 'connectivities'):
        assert (clustered.obsp[key] != original.obsp[key]).nnz == 0
    # UMAP's existing backend can prune tiny weights in its new output copy;
    # the accepted input publications remain byte-pinned and unchanged.
    assert visualized.obs['leiden'].tolist() == clustered.obs['leiden'].tolist()
    assert visualized.obsm['X_umap'].shape == (32, 2) and np.isfinite(visualized.obsm['X_umap']).all()
    assert_preserved(before)
    assert len(models) == 1 and len(models[0].calls) == 3
    for value, schema, prompt in models[0].calls:
        assert str(source.app.workspace_root) not in prompt
        assert source.run.run_id not in prompt
        assert source.binding.authority_sha256 not in prompt
    with conversational_work() as read_counts:
        rich = facade.revision('analysis', view.revision_id)
        evidence = facade.evidence('analysis', view.revision_id, 'umap', fields=('n_cells', 'n_components'))
    assert not read_counts
    assert rich.result.kind == 'execute' and evidence.status == 'available'
    assert {fact.field: fact.value for fact in evidence.facts} == {'n_cells': 32, 'n_components': 2}
    captures_path, request_sizes = persist_model_captures(source.embedding.parent, models[0].calls)
    witness = dict(producer_run=source.run.run_id, consumer_run=run.run_id,
        producer_authority_sha256=source.binding.authority_sha256,
        base_revision=source.session.active_revision_id, new_revision=revision.revision_id,
        retained=retain, resolution=run.steps[0].result['resolution'], n_cells=32,
        n_clusters=run.steps[0].result['n_clusters'], counts=dict(counts),
        model_captures=str(captures_path), model_request_sizes=request_sizes)
    path = source.embedding.parent / 'conversational-continuation-witness.json'
    path.write_text(json.dumps(witness, indent=2, sort_keys=True), encoding='utf-8')
    print('ordinary processed continuation:', json.dumps(witness, sort_keys=True))


def test_read_only_question_does_not_offer_or_execute_a_scientific_plan(processed_source):
    source = processed_source
    facade, models = interactive(source, action='question')
    before = preserved_run_bytes(source)
    with conversational_work() as counts:
        answer = facade.submit_turn('analysis', 'question', 'How many cells are in this result?', expected_generation=1)
    assert answer.status == 'answered', answer.to_dict()
    assert answer.response.kind == 'answer' and '32' in answer.response.text
    assert not counts
    assert len(models) == 1 and len(models[0].calls) == 2
    state = facade._application.sessions.load('analysis')
    assert state.generation == 1 and state.revisions == source.session.revisions
    assert not any(turn.turn_id == 'question' and turn.run_id is not None for turn in state.turns)
    assert_preserved(before)


def test_unrelated_new_input_never_inherits_old_scientific_artifact(processed_source, tmp_path):
    source = processed_source
    other = tiny_other_input(tmp_path / 'unrelated.h5ad')
    facade, models = interactive(source, action='new')
    before = preserved_run_bytes(source)
    with conversational_work() as counts:
        view = facade.submit_turn('analysis', 'unrelated', 'Inspect this unrelated supplied matrix.',
            expected_generation=1, execution_inputs={'input_path': str(other)})
    assert view.status == 'succeeded', view.to_dict()
    run = accepted_run(facade, 'unrelated')
    assert len(run.plan.steps) == 1 and run.plan.steps[0].tool_name == 'inspect_scATAC'
    assert run.steps[0].resolved_arguments['path'] == str(other)
    assert run.steps[0].result['n_cells'] == 6
    assert not any(isinstance(argument, PriorOutputRef) for step in run.plan.steps for argument in step.arguments.values())
    assert not {name: count for name, count in counts.items() if name.startswith(('production.', 'owner.', 'source.'))}
    state = facade._application.sessions.load('analysis')
    assert state.generation == 2 and state.revisions[0] == source.session.revisions[0]
    assert tuple(output.name for output in state.revisions[-1].outputs) == ('inspection',)
    assert_preserved(before)


def test_explicit_incompatible_new_source_fails_without_historical_substitution(processed_source, tmp_path):
    source = processed_source
    other = tiny_other_input(tmp_path / 'not-a-neighbor-graph.h5ad')
    facade, _ = interactive(source, action='incompatible')
    before = preserved_run_bytes(source)
    with conversational_work() as counts:
        view = facade.submit_turn('analysis', 'incompatible', 'Cluster this supplied analysis.', expected_generation=1,
            execution_inputs={'analysis_path': str(other)})
    assert view.status == 'failed', view.to_dict()
    run = accepted_run(facade, 'incompatible')
    assert run.lifecycle_status is RunLifecycleStatus.FAILED
    assert run.plan.steps[0].arguments['analysis_path'] == str(other)
    assert not {name: count for name, count in counts.items() if name.startswith(('production.', 'owner.', 'source.'))}
    state = facade._application.sessions.load('analysis')
    assert state.generation == 1 and state.revisions == source.session.revisions
    assert_preserved(before)


def test_duplicate_refresh_fresh_process_and_navigation_preserve_execution(processed_source, tmp_path):
    source = processed_source
    facade, models = interactive(source, retain=True)
    view = facade.submit_turn('analysis', 'modify', MODIFICATION, expected_generation=1)
    assert view.status == 'succeeded', view.to_dict()
    state = facade._application.sessions.load('analysis')
    run = accepted_run(facade, 'modify')
    state_bytes = facade._application.run_store.state_path(run.run_id).read_bytes()
    original_bytes = preserved_run_bytes(source)
    calls = len(models[0].calls)
    with conversational_work() as repeated:
        duplicate = facade.submit_turn('analysis', 'modify', MODIFICATION, expected_generation=1)
        refreshed = facade.reopen_session('analysis')
    assert not repeated and duplicate == view and len(models) == 1 and len(models[0].calls) == calls
    assert refreshed.active_revision_id == view.revision_id
    assert_preserved(original_bytes)
    source.embedding.rename(source.embedding.with_suffix('.archived'))
    source.cell_ids.rename(source.cell_ids.with_suffix('.archived'))
    result_path = tmp_path / 'fresh-process-conversation.json'
    script = '''
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'tests/authority_dag'))
from test_conversational_processed_continuation import PROFILE, MODIFICATION, conversational_work
from agent.application import InteractiveAgentApplication
from agent.providers import PlanningModelFactoryRegistry
def forbidden(profile):
    raise AssertionError('Provider constructed during ordinary stored retry/reopen')
facade = InteractiveAgentApplication(sys.argv[1], model_profiles=(PROFILE,), default_profile_id=PROFILE.profile_id,
    planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': forbidden}))
with conversational_work() as counts:
    history = facade.reopen_session('analysis')
    retry = facade.submit_turn('analysis', 'modify', MODIFICATION, expected_generation=1)
assert history.active_revision_id == retry.revision_id
assert retry.status == 'succeeded'
assert not counts
Path(sys.argv[2]).write_text(json.dumps(dict(view=retry.to_dict(), counts=dict(counts))), encoding='utf-8')
'''
    child = subprocess.run([sys.executable, '-B', '-c', script, str(source.app.workspace_root), str(result_path)],
        capture_output=True, text=True, env=os.environ.copy(), timeout=60)
    assert child.returncode == 0, child.stdout + child.stderr
    assert json.loads(result_path.read_text(encoding='utf-8')) == dict(view=view.to_dict(), counts={})
    assert facade._application.run_store.state_path(run.run_id).read_bytes() == state_bytes
    assert facade._application.sessions.load('analysis') == state
    navigation, nav_models = interactive(source, action='navigate')
    with conversational_work() as nav_counts:
        navigated = navigation.submit_turn('analysis', 'back', 'Switch to the parent version.', expected_generation=2)
    assert navigated.status == 'activated', navigated.to_dict()
    assert navigated.response.kind == 'navigate'
    assert not nav_counts and len(nav_models) == 1 and len(nav_models[0].calls) == 1
    latest = facade._application.sessions.load('analysis')
    assert latest.generation == 3 and latest.active_revision_id == source.session.active_revision_id
    assert latest.revisions == state.revisions
    for path, raw in original_bytes.items():
        if path not in (source.embedding, source.cell_ids):
            assert path.read_bytes() == raw


@pytest.mark.parametrize('handle', ['ctx.1', 'other-session-graph'])
def test_unoffered_historical_source_cannot_create_a_scientific_revision(processed_source, handle):
    source = processed_source
    before = preserved_run_bytes(source)
    facade, _ = interactive(source, handle=handle)
    with conversational_work() as counts:
        view = facade.submit_turn('analysis', 'bad-reference', MODIFICATION, expected_generation=1)
    assert view.status == 'failed', view.to_dict()
    assert not {name: count for name, count in counts.items() if name.startswith(('production.', 'owner.', 'source.'))}
    state = facade._application.sessions.load('analysis')
    assert state.generation == 1 and state.revisions == source.session.revisions
    assert_preserved(before)


@pytest.mark.parametrize('artifact', ['analysis_path', 'manifest_path'])
def test_corrupt_eligible_artifact_is_not_silently_omitted_as_nonreusable(processed_source, artifact):
    source = processed_source
    path = Path(source.run.steps[0].result[artifact])
    path.write_bytes(path.read_bytes() + b' ')
    facade, models = interactive(source)
    with conversational_work() as counts:
        view = facade.submit_turn('analysis', 'corrupt', MODIFICATION, expected_generation=1)
    assert view.status != 'succeeded', view.to_dict()
    assert not {name: count for name, count in counts.items() if name.startswith(('production.', 'owner.', 'source.'))}
    assert not any('semantic_prompt_version' in value for model in models for value, _, _ in model.calls)
    state = facade._application.sessions.load('analysis')
    assert state.generation == 1 and state.revisions == source.session.revisions


def test_ambiguous_reference_remains_interpreter_clarification(processed_source):
    source = processed_source
    before = preserved_run_bytes(source)
    facade, models = interactive(source, action='ambiguous')
    with conversational_work() as counts:
        result = facade.submit_turn('analysis', 'ambiguous', 'Use the other earlier result.', expected_generation=1)
    assert result.status == 'clarification', result.to_dict()
    assert result.response.clarification['reason'] == 'ambiguous_subject'
    assert not counts and len(models) == 1 and len(models[0].calls) == 1
    state = facade._application.sessions.load('analysis')
    assert state.generation == 1 and state.revisions == source.session.revisions
    assert_preserved(before)


def test_revision_change_after_interaction_capture_cannot_redirect_or_execute(processed_source):
    source = processed_source
    facade, _ = interactive(source, retain=True)
    initial = facade.submit_turn('analysis', 'modify', MODIFICATION, expected_generation=1)
    assert initial.status == 'succeeded', initial.to_dict()
    facade.activate_revision('analysis', 'capture-base', source.session.active_revision_id, expected_generation=2)
    revisions = facade._application.sessions.load('analysis').revisions
    changes = []
    def change_revision():
        assert not changes
        source.app.sessions.switch('analysis', 'race-navigation', initial.revision_id, expected_generation=3)
        changes.append(True)
    racing, models = interactive(source, retain=True, before_interpret=change_revision)
    with conversational_work() as counts:
        view = racing.submit_turn('analysis', 'racing-modification', MODIFICATION, expected_generation=3)
    assert view.status != 'succeeded', view.to_dict()
    assert changes == [True]
    assert not {name: count for name, count in counts.items() if name.startswith(('production.', 'owner.', 'source.'))}
    state = racing._application.sessions.load('analysis')
    assert state.generation == 4 and state.active_revision_id == initial.revision_id
    assert state.revisions == revisions
    captured = next(item for item in state.interactions if item.turn_id == 'racing-modification')
    assert captured.base_revision_id == source.session.active_revision_id and captured.base_generation == 3
    assert not any(turn.turn_id == 'racing-modification' and turn.status in ('activated', 'stale') for turn in state.turns)


@pytest.mark.parametrize('error', ['unknown', 'duplicate', 'name_collision', 'total_bound'])
def test_invalid_explicit_retention_is_rejected_before_science(processed_source, error):
    source = processed_source
    before = preserved_run_bytes(source)
    facade, models = interactive(source, retain=True, retention_error=error)
    with conversational_work() as counts:
        result = facade.submit_turn('analysis', 'invalid-retention', MODIFICATION, expected_generation=1)
    assert result.status == 'failed', result.to_dict()
    assert result.error.code == 'PLANNER_OUTPUT_INVALID', result.to_dict()
    assert not {name: count for name, count in counts.items() if name.startswith(('production.', 'owner.', 'source.'))}
    state = facade._application.sessions.load('analysis')
    assert state.generation == 1 and state.revisions == source.session.revisions
    assert not any(turn.turn_id == 'invalid-retention' and turn.status in ('activated', 'stale') for turn in state.turns)
    assert any('output_selection_schema_version' in value for model in models for value, _, _ in model.calls)
    assert_preserved(before)


def test_mixed_revision_keeps_only_qualified_graph_in_later_context(processed_source):
    source = processed_source
    facade, _ = interactive(source, retain=True)
    first = facade.submit_turn('analysis', 'modify', MODIFICATION, expected_generation=1)
    assert first.status == 'succeeded', first.to_dict()
    first_run = accepted_run(facade, 'modify')
    first_revision = facade._application.sessions.load('analysis').revisions[-1]
    paths = facade._application._workspace.run_paths(first_run.run_id)
    preserved = {path: path.read_bytes() for path in paths.root.rglob('*') if path.is_file()}
    later, models = interactive(source, retain=True, literal='0.7')
    utterance = MODIFICATION.replace('0.5', '0.7')
    with conversational_work() as counts:
        second = later.submit_turn('analysis', 'modify-again', utterance, expected_generation=2)
    assert second.status == 'succeeded', second.to_dict()
    second_run = accepted_run(later, 'modify-again')
    assert second_run.plan.steps[0].arguments['analysis_path'].binding == source.binding
    assert second_run.steps[0].result['resolution'] == .7
    assert first_run.steps[0].result['resolution'] == .5
    assert counts['production.leiden'] == counts['production.umap'] == 1
    assert counts['production.neighbors'] == counts['owner.neighbors'] == 0
    assert not {name: count for name, count in counts.items() if name.startswith('source.')}
    state = later._application.sessions.load('analysis')
    assert state.generation == 3 and len(state.revisions) == 3
    assert state.revisions[1] == first_revision
    assert state.revisions[-1].parent_revision_id == first_revision.revision_id
    assert_preserved(preserved)
    interpreter = next(value for value, _, _ in models[0].calls if 'turn_schema_version' in value)
    context = interpreter['dialogue']['planning_context']
    assert len(context) == 1 and context[0]['producer_tool'] == 'build_cell_neighbors'
    assert context[0]['source_port'] == 'neighbors'
