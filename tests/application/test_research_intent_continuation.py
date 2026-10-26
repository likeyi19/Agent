"""Real capture/compilation, simulated accepted metadata, science withheld."""
import asyncio
import json
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from agent.application import InteractiveAgentApplication
from agent.application.dialogue_execution import planning_task
from agent.application.session_state import AnalysisSession, OutputSelection, SessionError, digest, selection_uses
from agent.application.uploads import H5ADUploadAdmission
from agent.providers import PlanningModelFactoryRegistry
from agent.schemas.orchestration import _serialize
from test_human_epizoo_mapping import WithholdScience
from test_dialogue_evidence import accepted
from test_scientific_completion import overlay_service


OBJECTIVE = ('I will upload a count matrix. Compute PCA, neighbors and Leiden; compare the new '
    'Leiden distribution with cohort on the selected previous UMAP and color those previous '
    'coordinates by the new Leiden assignments. Omit the independently optional marker analysis.')


def setup(tmp_path, monkeypatch, *, invalid=None, retain_components=False):
    objective = OBJECTIVE + (' Keep the recorded component count from the accepted PCA result.'
                             if retain_components else '')
    seed, coords, _, _, scientific_calls, _ = overlay_service(tmp_path, monkeypatch)
    if retain_components:
        accepted(seed._application, 'pca_embed_cells', name='previous-pca', arguments={'n_components': 3})
    registry = seed._application.registry
    executor, calls, selected = WithholdScience(registry), [], {}

    class Model:
        def __init__(self, profile):
            self.model_id = profile.model_id

        def complete(self, *, prompt, response_schema):
            value = json.loads(prompt)
            calls.append(value)
            if 'turn_schema_version' in value:
                if value['dialogue'].get('pending_prerequisite'):
                    decision = dict(kind='answer_roles', pending='@roles',
                        roles=[dict(tool='pca_embed_cells', port='dataset', source='source.0')])
                else:
                    group, = [g for g in value['dialogue']['result_groups'] if g['tool'] == 'compute_cell_umap']
                    selected['comparison'] = dict(left='new', right=group['handle'], detail=dict(
                        section='category_counts', field='result:leiden', other_field='result:cohort', offset=0, limit=8))
                    selected['visualization'] = dict(coordinates=group['handle'], labels='new', field='result:leiden')
                    if invalid == 'result':
                        selected['comparison']['right'] = 'g999'
                    if invalid == 'source':
                        selected['visualization'].update(labels='source.15', field='source:cohort')
                    decision = dict(kind='request_roles', target='pca_embed_cells',
                        required_roles=[dict(tool='pca_embed_cells', port='dataset')], roles=[],
                        comparison=selected['comparison'], visualization=selected['visualization'],
                        unfulfilled=['optional marker analysis'])
                    if retain_components:
                        pca, = [g for g in value['dialogue']['result_groups'] if g['tool'] == 'pca_embed_cells']
                        decision['kind'] = 'execute_plan'
                        del decision['required_roles']
                        decision['retain_parameters'] = [dict(result=pca['handle'], argument='n_components')]
                return json.dumps(dict(turn_schema_version=1, decision=decision))
            if 'semantic_prompt_version' in value:
                assert value['user_request'] == objective
                task = value['research_context']['task']
                assert task['comparison'] == selected['comparison']
                assert task['visualization'] == selected['visualization']
                assert task['unfulfilled'] == ['optional marker analysis']
                return json.dumps(dict(schema_version=4, decision=dict(kind='plan', steps=[
                    dict(step_id='pca', tool='pca_embed_cells', sources=[], control_dependencies=[]),
                    dict(step_id='graph', tool='build_cell_neighbors', sources=[dict(
                        kind='step', target='embedding', step='pca')], control_dependencies=[]),
                    dict(step_id='labels', tool='cluster_cells', sources=[dict(
                        kind='step', target='analysis', step='graph')], control_dependencies=[])])))
            if 'output_selection_schema_version' in value:
                assert set(value['requested_output_uses']) == {'comparison_result', 'visualization_labels'}
                return json.dumps(dict(outputs=[dict(name='new_labels', step_id='labels',
                    output_key='analysis_path', use=['comparison_result', 'visualization_labels'])]))
            raise AssertionError('Unexpected provider or scientific phase')

    profiles = tuple(seed._profiles.values())
    facade = InteractiveAgentApplication(seed._application.workspace_root,
        model_profiles=profiles, default_profile_id='alpha', registry=registry, executor=executor,
        approved_source_roots=(tmp_path,),
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': Model}))
    upload_root = facade._application.workspace_root / 'uploads'
    upload_root.mkdir()
    uploads = H5ADUploadAdmission(facade.resources, upload_root)
    return SimpleNamespace(facade=facade, uploads=uploads, coords=coords,
        executor=executor, calls=calls, selected=selected, scientific_calls=scientific_calls,
        generation=facade._application.sessions.load('session').generation, objective=objective)


def initial(case):
    return case.facade.submit_turn('session', 'request', case.objective, expected_generation=case.generation)


def upload_matrix(case):
    previous = ad.read_h5ad(case.coords)
    source = case.coords.with_name('simulated-counts.h5ad')
    ad.AnnData(sparse.csr_matrix(np.arange(25).reshape(5, 5) % 4),
        obs=previous.obs.copy(), var=pd.DataFrame(index=[f'feature-{i}' for i in range(5)])).write_h5ad(source)
    content = source.read_bytes()
    async def chunks():
        yield content
    return asyncio.run(case.uploads.receive('new-matrix', source.name, chunks(),
        expected_size=len(content), session_id='session'))


def test_missing_file_upload_retains_exact_intent_and_selects_one_result_for_both_purposes(tmp_path, monkeypatch):
    case = setup(tmp_path, monkeypatch)
    pending = initial(case)
    assert pending.status == 'clarification', pending.to_dict()
    state = case.facade._application.sessions.load('session')
    original = state.interactions[-1]
    scientific_turns, revisions = len(state.turns), state.revisions
    assert original.prerequisite['field'] == 'roles'
    assert _serialize(original.admitted['comparison']) == case.selected['comparison']
    assert _serialize(original.admitted['visualization']) == case.selected['visualization']
    assert original.admitted['references'] == (case.selected['comparison']['right'],)
    assert not case.executor.plans and not case.scientific_calls
    record = upload_matrix(case)
    accepted_bytes = case.coords.read_bytes()
    outcome = case.facade.submit_turn('session', 'upload', 'Here is the matrix; continue the original request.',
        expected_generation=1, attachment_ids=(record.resource_id,))
    assert outcome.status == 'failed' and outcome.error.code == 'TEST_SCIENCE_WITHHELD', outcome.to_dict()
    plan, = case.executor.plans
    assert [step.tool_name for step in plan.steps] == ['pca_embed_cells', 'build_cell_neighbors', 'cluster_cells']
    assert plan.steps[0].arguments['input_path'] == record.source_path
    state = case.facade._application.sessions.load('session')
    assert state.interactions[0] == original
    assert state.revisions == revisions and state.generation == 1
    assert len(state.turns) == scientific_turns + 1
    task = planning_task(state.interactions[-1], state)
    assert task['comparison'] == case.selected['comparison']
    assert task['visualization'] == case.selected['visualization']
    assert task['unfulfilled'] == ['optional marker analysis']
    selection, = state.turn('upload').selections
    assert selection_uses(selection) == ('comparison_result', 'visualization_labels')
    run = case.facade._application.run_store.load(outcome.run_id)
    assert run.request.prompt == OBJECTIVE and not run.steps
    assert case.coords.read_bytes() == accepted_bytes and not case.scientific_calls


@pytest.mark.parametrize('invalid', ['result', 'source'])
def test_unoffered_captured_result_or_source_cannot_enter_pending_intent(tmp_path, monkeypatch, invalid):
    case = setup(tmp_path, monkeypatch, invalid=invalid)
    outcome = initial(case)
    assert outcome.revision_id is None and outcome.status == 'failed'
    assert outcome.response.error.code == 'INTERACTIVE_INTENT_BINDING_INVALID'
    assert outcome.response.clarification is None
    state = case.facade._application.sessions.load('session')
    assert state.interactions[-1].admitted is None
    assert state.interactions[-1].prerequisite is None
    assert len(state.revisions) == len(state.turns) == 1
    assert not case.executor.plans and not case.scientific_calls
    calls = len(case.calls)
    assert initial(case).to_dict() == outcome.to_dict()
    assert len(case.calls) == calls
    assert case.facade._application.sessions.load('session') == state


@pytest.mark.parametrize('mutation', ['result', 'source'])
def test_rechecksummed_invalid_persisted_intent_is_rejected_without_execution(tmp_path, monkeypatch, mutation):
    case = setup(tmp_path, monkeypatch)
    assert initial(case).status == 'clarification'
    sessions = case.facade._application.sessions
    path = sessions._store._path('session', '.json')
    original = path.read_bytes()
    envelope = json.loads(original)
    admitted = envelope['record']['interactions'][0]['admitted']
    if mutation == 'result':
        admitted['comparison']['right'] = 'g999'
    else:
        admitted['visualization'].update(labels='source.15', field='source:cohort')
    envelope['sha256'] = digest(envelope['record'])
    before = len(case.calls)
    try:
        path.write_text(json.dumps(envelope))
        with pytest.raises(SessionError):
            sessions.load('session')
    finally:
        path.write_bytes(original)
    assert len(case.calls) == before and not case.executor.plans and not case.scientific_calls


def test_selection_purposes_decode_legacy_and_current_session_records(tmp_path, monkeypatch):
    case = setup(tmp_path, monkeypatch)
    state = case.facade._application.sessions.load('session')
    record = _serialize(state.to_dict())
    selection = record['turns'][0]['selections'][0]
    selection['use'] = 'visualization_labels'
    legacy = AnalysisSession.from_dict(record)
    assert selection_uses(legacy.turns[0].selections[0]) == ('visualization_labels',)
    selection['use'] = ['comparison_result', 'visualization_labels']
    current = AnalysisSession.from_dict(record)
    assert selection_uses(current.turns[0].selections[0]) == ('comparison_result', 'visualization_labels')
    assert _serialize(current.to_dict())['turns'][0]['selections'][0]['use'] == selection['use']
    selection.pop('use')
    assert selection_uses(AnalysisSession.from_dict(record).turns[0].selections[0]) == ()
    with pytest.raises(SessionError):
        OutputSelection('invalid', 'step0', 'analysis_path', ['comparison_result', 'comparison_result'])


def test_missing_file_retains_accepted_pca_parameter_through_upload_and_compilation(tmp_path, monkeypatch):
    case = setup(tmp_path, monkeypatch, retain_components=True)
    assert initial(case).status == 'clarification'
    state = case.facade._application.sessions.load('session')
    original = state.interactions[-1]
    assert original.admitted['inputs'] == {'n_components': 3}
    retained, = original.admitted['retained_parameters']
    assert retained['tool'] == 'pca_embed_cells' and retained['argument'] == 'n_components'
    assert retained['value'] == 3 and retained['origin'] == 'accepted_argument'
    assert not original.admitted.get('arguments') and not case.executor.plans
    record = upload_matrix(case)
    outcome = case.facade.submit_turn('session', 'upload', 'Use this matrix and retain the prior settings.',
        expected_generation=case.generation, attachment_ids=(record.resource_id,))
    assert outcome.status == 'failed' and outcome.error.code == 'TEST_SCIENCE_WITHHELD', outcome.to_dict()
    plan, = case.executor.plans
    assert [step.tool_name for step in plan.steps] == ['pca_embed_cells', 'build_cell_neighbors', 'cluster_cells']
    assert plan.steps[0].arguments['n_components'] == 3
    assert plan.steps[0].arguments['input_path'] == record.source_path
    state = case.facade._application.sessions.load('session')
    assert state.interactions[0] == original
    assert state.interactions[-1].admitted['inputs']['n_components'] == 3
    assert planning_task(state.interactions[-1], state)['retained_parameters'][0] == _serialize(retained)
    run = case.facade._application.run_store.load(outcome.run_id)
    assert run.request.prompt == case.objective and run.request.inputs['n_components'] == 3
    assert state.generation == case.generation and not case.scientific_calls
