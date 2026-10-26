"""Captured handles bind the same files across pending and independent tasks.

Scripted semantic decisions reach ordinary compilation/preflight; the executor
withholds all native science. These fixtures qualify binding, not model intent.
"""
from dataclasses import asdict
import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from agent.application import InteractiveAgentApplication
from agent.application.raw_handoff import AnnotationDeclaration
from agent.orchestration import build_default_tool_registry
from agent.providers import PlanningModelFactoryRegistry
from agent.schemas.orchestration import _serialize
from test_human_epizoo_mapping import WithholdScience
from test_multi_input_role_admission import PROFILE


OLD = 'Keep the first cells pending; use 2 components.'
NEW = 'Analyze the newly attached cells using 3 components.'
COMPARE = NEW + ' Compare the groups with the supplied annotations.'
ANNOTATE = 'Annotate the newly attached cells; I do not know the processing history.'
LABEL = 'Provided cell identity α'
PORT = dict(tool='pca_embed_cells', port='dataset')


def number(text, value):
    start = text.index(value)
    return dict(tool='pca_embed_cells', argument='n_components', literal=value,
                start=start, end=start + len(value))


class Model:
    model_id = PROFILE.model_id

    def __init__(self, calls, *, comparison=False, raw_pending=False, raw_new=False, request_new=False):
        self.calls, self.comparison = calls, comparison
        self.raw_pending, self.raw_new = raw_pending, raw_new
        self.request_new = request_new

    def complete(self, *, prompt, response_schema):
        value = json.loads(prompt)
        self.calls.append(value)
        if 'turn_schema_version' in value:
            text = value['utterance']
            if text == OLD:
                if self.raw_pending:
                    decision = dict(kind='request_roles', target='annotate_scATAC_cell_types',
                        required_roles=[], roles=[], arguments=[],
                        annotation=asdict(AnnotationDeclaration(source='source.0')))
                else:
                    decision = dict(kind='request_roles', target='pca_embed_cells',
                        required_roles=[PORT], roles=[], arguments=[number(text, '2')])
            elif self.raw_new:
                decision = dict(kind='request_roles', target='annotate_scATAC_cell_types',
                    required_roles=[], roles=[], arguments=[],
                    annotation=asdict(AnnotationDeclaration(source='source.1')))
            elif self.request_new:
                decision = dict(kind='request_roles', target='pca_embed_cells',
                    required_roles=[PORT], roles=[], arguments=[number(text, '3')])
            else:
                roles = [dict(PORT, source='source.1')]
                arguments = [number(text, '3')]
                if self.comparison:
                    roles.append(dict(tool='evaluate_cell_clustering',
                                      port='ground_truth_dataset', source='source.1'))
                    phrase = 'supplied annotations'; start = text.index(phrase)
                    arguments.append(dict(tool='evaluate_cell_clustering', argument='label_key',
                        literal=phrase, start=start, end=start + len(phrase), value=LABEL,
                        source='source.1', mode='offered'))
                decision = dict(kind='execute_plan', target='pca_embed_cells',
                                roles=roles, arguments=arguments)
            return json.dumps(dict(turn_schema_version=1, decision=decision), ensure_ascii=False)
        if 'output_selection_schema_version' in value:
            return json.dumps(dict(outputs=[dict(name='selected result',
                step_id='score' if self.comparison else 'reduce',
                output_key='report_path' if self.comparison else 'analysis_path')]))
        assert value['user_request'] == (COMPARE if self.comparison else NEW)
        steps = [dict(step_id='reduce', tool='pca_embed_cells', sources=[], control_dependencies=[])]
        if self.comparison:
            steps.extend([
                dict(step_id='links', tool='build_cell_neighbors',
                     sources=[dict(kind='step', target='embedding', step='reduce')], control_dependencies=[]),
                dict(step_id='partition', tool='cluster_cells',
                     sources=[dict(kind='step', target='analysis', step='links')], control_dependencies=[]),
                dict(step_id='score', tool='evaluate_cell_clustering', sources=[
                    dict(kind='step', target='analysis', step='partition'),
                    dict(kind='input', target='ground_truth_dataset', input='reference_h5ad_path')],
                    control_dependencies=[])])
        return json.dumps(dict(schema_version=4, decision=dict(kind='plan', steps=steps)))


def setup_case(tmp_path, **options):
    root = tmp_path/'inputs'; root.mkdir()
    registry = build_default_tool_registry(); executor = WithholdScience(registry); calls = []
    app = InteractiveAgentApplication(tmp_path/'workspace', model_profiles=(PROFILE,),
        default_profile_id=PROFILE.profile_id, registry=registry, executor=executor,
        approved_source_roots=(root,), planning_model_factory_registry=PlanningModelFactoryRegistry(
            {'scripted': lambda _: Model(calls, **options)}))
    app.create_session('session')
    records = []
    for index, label in enumerate(('Old selected cells', 'New selected cells')):
        path = root/f'cells-{index}.h5ad'
        obs = pd.DataFrame({LABEL: pd.Categorical(['a', 'a', 'b', 'b', 'c', 'c'])},
                           index=[f'cell-{index}-{i}' for i in range(6)])
        data = ad.AnnData(sparse.csr_matrix(np.arange(24).reshape(6, 4) % 5), obs=obs)
        data.var_names = ['chr1:1-2', 'chr1:3-4', 'chr1:5-6', 'chr1:7-8']
        data.write_h5ad(path)
        records.append(app.resources.register(f'cells-{index}', path, label=label,
            attribution='Simulated binding fixture only; no scientific qualification.', session_id='session'))
    return app, records, executor, calls


def submit(app, record, turn, text):
    group = app.resources.compose_group([record.resource_id], attachment_candidates=True)
    return app.submit_turn('session', turn, text, expected_generation=0, registered_input=group)


@pytest.mark.parametrize('comparison', [False, True])
@pytest.mark.parametrize('raw_pending', [False, True])
@pytest.mark.parametrize('available_reversed', [False, True])
def test_independent_task_binds_offered_new_file_with_and_without_observed_choice(
        tmp_path, monkeypatch, comparison, raw_pending, available_reversed):
    app, records, executor, calls = setup_case(tmp_path, comparison=comparison, raw_pending=raw_pending)
    first = submit(app, records[0], 'old', OLD)
    assert first.status == 'clarification' and not executor.plans, first.to_dict()
    if available_reversed:
        # Exercise the existing current-first Session candidate envelope even
        # when its ordinal order differs from a pending task's captured prefix.
        available = app.resources.compose_group([r.resource_id for r in reversed(records)],
                                                 attachment_candidates=True)
        original = app._submission
        def with_available(*args, **kwargs):
            submission = original(*args, **kwargs)
            submission['available_input'] = available.attribution()
            return submission
        monkeypatch.setattr(app, '_submission', with_available)
    text = COMPARE if comparison else NEW
    result = submit(app, records[1], 'new', text)
    assert result.error.code == 'TEST_SCIENCE_WITHHELD', result.to_dict()
    plan, = executor.plans
    by_id = {step.step_id: step for step in plan.steps}
    assert by_id['reduce'].arguments['input_path'] == records[1].source_path
    assert by_id['reduce'].arguments['n_components'] == 3
    if comparison:
        assert by_id['score'].arguments['reference_h5ad_path'] == records[1].source_path
        assert by_id['score'].arguments['label_key'] == LABEL
    state = app._application.sessions.load('session')
    interaction = state.interactions[-1]
    assert 'continuation' not in interaction.admitted
    assert interaction.admitted['inputs']['n_components'] == 3
    assert not interaction.snapshot['dialogue']['source_view']['execution_inputs']
    assert not interaction.snapshot['dialogue']['source_view']['registered_input']['roles']
    assert 'resource_choice' not in interaction.snapshot['dialogue']['source_view']['registered_input']
    assert [m['resource_id'] for m in interaction.admitted['registered_input']['members']] == [r.resource_id for r in records]
    assert app._application.run_store.load(result.run_id).request.prompt == text
    public = next(row['dialogue'] for row in calls if row.get('utterance') == text)
    assert [row['label'] for row in public['selected_sources']] == [r.label for r in records]
    assert all(set(row) == {'source', 'fields', 'complete'} for row in public['source_fields'])
    if available_reversed:
        assert public['current_turn_sources'] == ['source.1']
    assert not state.revisions


@pytest.mark.parametrize('raw_pending', [False, True])
def test_new_raw_request_after_pending_keeps_new_source_and_objective(tmp_path, raw_pending):
    app, records, executor, calls = setup_case(tmp_path, raw_new=True, raw_pending=raw_pending)
    assert submit(app, records[0], 'old', OLD).status == 'clarification'
    result = submit(app, records[1], 'annotation', ANNOTATE)
    assert result.status == 'clarification' and not executor.plans, result.to_dict()
    interaction = app._application.sessions.load('session').interactions[-1]
    assert interaction.admitted['annotation']['source'] == 'source.1'
    assert interaction.prerequisite['origin_turn_id'] == 'annotation'
    assert interaction.admitted['annotation_source_evidence']['source_sha256'] == records[1].source_sha256
    assert interaction.admitted['annotation_arguments'] == {}
    assert 'continuation' not in interaction.admitted
    assert not app._application.sessions.load('session').turns


def test_new_generic_role_request_after_raw_pending_does_not_continue_annotation(tmp_path):
    app, records, executor, calls = setup_case(tmp_path, raw_pending=True, request_new=True)
    assert submit(app, records[0], 'old', OLD).status == 'clarification'
    result = submit(app, records[1], 'new-request', NEW)
    assert result.status == 'clarification' and not executor.plans, result.to_dict()
    interaction = app._application.sessions.load('session').interactions[-1]
    assert interaction.prerequisite['field'] == 'roles'
    assert interaction.prerequisite['origin_turn_id'] == 'new-request'
    assert interaction.admitted['tool'] == 'pca_embed_cells'
    assert interaction.admitted['inputs'] == {'n_components': 3}
    assert not interaction.admitted['roles'] and 'annotation' not in interaction.admitted
    assert 'continuation' not in interaction.admitted
    assert not app._application.sessions.load('session').turns
