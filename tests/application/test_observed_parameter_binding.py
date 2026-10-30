"""Real capture/role/compiler integration with observed columns and withheld science.

Scripted semantics test binding and lifecycle, not provider accuracy or biology.
No scientific algorithm is fitted by these fixtures.
"""
import asyncio
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from agent.application import InteractiveAgentApplication
from agent.application.session_state import SessionError, canonical
from agent.application.session_store import FileSessionStore
from agent.application.uploads import H5ADUploadAdmission
from agent.providers import PlanningModelFactoryRegistry
from agent.schemas import StepOutputRef
from agent.schemas.orchestration import _serialize
from test_human_epizoo_mapping import WithholdScience
from test_multi_input_role_admission import (
    LABEL, PROFILE, ROLES, TransferBindingModel, transfer_binding_harness,
)


PHRASE = 'supplied reference annotations'
TRANSFER = f'Transfer labels from the first dataset to the second using the {PHRASE}.'


def choice(text, phrase, value, *, tool='transfer_cell_labels', argument='reference_label_key', source=None, mode='offered'):
    start = text.index(phrase)
    result = dict(tool=tool, argument=argument, literal=phrase, start=start,
                  end=start + len(phrase), value=value)
    return result if source is None else result | dict(source=source, mode=mode)


class ObservedTransferModel(TransferBindingModel):
    def __init__(self, calls, *, selected=LABEL, roles=ROLES, bad_literal=False):
        super().__init__(calls, roles=roles)
        self.selected, self.bad_literal = selected, bad_literal

    def complete(self, *, prompt, response_schema):
        value = json.loads(prompt)
        if 'turn_schema_version' in value:
            self.calls.append(dict(value=value, prompt=prompt, schema=response_schema))
            reference = next(role['source'] for role in self.roles if role['port'] == 'reference_dataset')
            argument = choice(value['utterance'], PHRASE, self.selected, source=reference)
            if self.bad_literal:
                argument['literal'] = 'fabricated annotations'
            return json.dumps(dict(turn_schema_version=1, decision=dict(kind='execute_plan',
                target='transfer_cell_labels', roles=self.roles, arguments=[argument])))
        return super().complete(prompt=prompt, response_schema=response_schema)


def transfer_case(tmp_path, *, selected=LABEL, reverse=False, swapped_role=False,
                  bad_literal=False, query_labels=False):
    seed, records, group, executor, calls, companions = transfer_binding_harness(tmp_path)
    reference = ad.read_h5ad(records[0].source_path)
    reference.obs.insert(0, 'batch', ['one'] * reference.n_obs)
    reference_path = Path(records[0].source_path).with_name('reference-rich.h5ad')
    reference.write_h5ad(reference_path)
    records[0] = seed.resources.register('reference-rich', reference_path, label='Reference metadata',
        attribution='Simulated multiple-field discovery fixture; no biological qualification.')
    if query_labels:
        data = ad.read_h5ad(records[1].source_path)
        data.obs[LABEL] = ['query A', 'query B']
        data.obs['query_only'] = ['one', 'two']
        path = Path(records[1].source_path).with_name('query-rich.h5ad')
        data.write_h5ad(path)
        records[1] = seed.resources.register('query-rich', path, label='Query with separate metadata',
            attribution='Simulated source for exact field-role association only.')
    selected_records = list(reversed(records)) if reverse else records
    group = seed.resources.compose_group([r.resource_id for r in selected_records], companions)
    roles = [dict(r, source='source.' + str(1 - int(r['source'].split('.')[1])))
             if reverse else dict(r) for r in ROLES]
    if swapped_role:
        roles = [dict(r, source='source.' + str(1 - int(r['source'].split('.')[1]))) for r in roles]
    app = InteractiveAgentApplication(seed._application.workspace_root,
        model_profiles=(PROFILE,), default_profile_id=PROFILE.profile_id,
        registry=seed._application.registry, executor=executor,
        approved_source_roots=(tmp_path/'approved',),
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted':lambda _:
            ObservedTransferModel(calls, selected=selected, roles=roles, bad_literal=bad_literal)}))
    return app, records, group, executor, calls


@pytest.mark.parametrize('reverse', [False, True])
def test_uploaded_source_column_is_semantically_selected_without_literal_name(tmp_path, reverse):
    app, records, group, executor, calls = transfer_case(tmp_path, reverse=reverse, query_labels=True)
    assert LABEL not in TRANSFER
    result = app.submit_turn('session', 'observed', TRANSFER, expected_generation=0, registered_input=group)
    assert result.error.code == 'TEST_SCIENCE_WITHHELD', result.to_dict()
    step, = executor.plans[0].steps
    assert step.arguments['reference_label_key'] == LABEL
    assert step.arguments['reference_h5ad_path'] == records[0].source_path
    assert step.arguments['query_h5ad_path'] == records[1].source_path
    state = app._application.sessions.load('session')
    assert not state.revisions and len(state.turns) == 1
    interaction, = state.interactions
    assert interaction.admitted['arguments'][0]['literal'] == PHRASE
    assert interaction.admitted['arguments'][0]['value'] == LABEL
    offers = interaction.snapshot['dialogue']['source_fields']
    assert offers[int(reverse)]['resource_id'] == records[0].resource_id
    first = calls[0]['value']['dialogue']['source_fields']
    assert first[int(reverse)]['fields'] == ['batch', LABEL]
    assert all(set(row) == {'source', 'fields', 'complete'} for row in first)
    for call in calls:
        assert str(tmp_path) not in call['prompt']
        assert all(record.resource_id not in call['prompt'] for record in records)
    # The same visible name on query is not a source association or truth claim.
    assert all(LABEL in row['fields'] for row in first)


@pytest.mark.parametrize('failure', ['other_source_only', 'unoffered', 'role_swap', 'fabricated_literal'])
def test_normalization_cannot_substitute_another_file_field_or_unstated_phrase(tmp_path, failure):
    app, _, group, executor, _ = transfer_case(tmp_path,
        selected=('query_only' if failure == 'other_source_only' else
                  'unoffered annotation' if failure == 'unoffered' else LABEL),
        swapped_role=failure == 'role_swap', bad_literal=failure == 'fabricated_literal',
        query_labels=failure == 'other_source_only')
    result = app.submit_turn('session', 'invalid', TRANSFER, expected_generation=0, registered_input=group)
    assert not executor.plans, result.to_dict()
    if failure == 'fabricated_literal':
        assert result.response.kind == 'clarify'
    else:
        assert result.error is not None
    state = app._application.sessions.load('session')
    assert not state.turns and not state.revisions


def test_observed_argument_history_decodes_without_source_access_and_rejects_forged_capture(tmp_path, monkeypatch):
    app, records, group, executor, _ = transfer_case(tmp_path)
    result = app.submit_turn('session', 'observed', TRANSFER, expected_generation=0, registered_input=group)
    assert result.error.code == 'TEST_SCIENCE_WITHHELD', result.to_dict()
    state = app._application.sessions.load('session')
    for record in records:
        Path(record.source_path).unlink()
    monkeypatch.setattr('agent.tools.analysis.categorical.fields',
                        lambda *a, **k: pytest.fail('Historical decode reopened observation source.'))
    store = FileSessionStore(app._application.sessions._store.root)
    assert store.load('session') == state
    # Session filenames use SHA-256 of the raw ID, unlike canonical record digests.
    import hashlib
    path = store.root / (hashlib.sha256(b'session').hexdigest() + '.json')
    envelope = json.loads(path.read_text())
    envelope['record']['interactions'][0]['snapshot']['dialogue']['source_fields'][0]['source_sha256'] = 'f' * 64
    envelope['sha256'] = hashlib.sha256(canonical(envelope['record'])).hexdigest()
    path.write_text(json.dumps(envelope))
    with pytest.raises(SessionError):
        FileSessionStore(store.root).load('session')
    assert len(executor.plans) == 1


OBJECTIVE = ('I will attach an accessibility matrix. Use 2 principal components to group cells, '
             'compare the groups with the supplied annotations, and make a UMAP. '
             'The comparison is required; it should not determine the UMAP coordinates.')
REPLY = 'Use the attached matrix and its supplied annotations; continue the original analysis.'
COMPOUND_ROLES = [dict(tool='pca_embed_cells', port='dataset', source='source.0'),
                  dict(tool='evaluate_cell_clustering', port='ground_truth_dataset', source='source.0')]


class ObservedContinuationModel:
    model_id = PROFILE.model_id

    def __init__(self, calls, *, invalid_dependency=False, reordered=False):
        self.calls, self.invalid_dependency, self.reordered = calls, invalid_dependency, reordered

    def complete(self, *, prompt, response_schema):
        value = json.loads(prompt)
        self.calls.append(value)
        if 'turn_schema_version' in value:
            if value['utterance'] == OBJECTIVE:
                start = OBJECTIVE.index('2')
                decision = dict(kind='request_roles', target='pca_embed_cells', roles=[],
                    required_roles=[{k:r[k] for k in ('tool', 'port')} for r in COMPOUND_ROLES],
                    arguments=[dict(tool='pca_embed_cells', argument='n_components', literal='2',
                                    start=start, end=start + 1)])
            else:
                assert value['dialogue']['pending_prerequisite']['objective'] == OBJECTIVE
                fields = value['dialogue']['source_fields'][0]['fields']
                assert fields == ['cohort', LABEL]
                decision = dict(kind='answer_roles', pending='@roles', roles=COMPOUND_ROLES,
                    arguments=[choice(REPLY, 'supplied annotations', LABEL,
                                     tool='evaluate_cell_clustering', argument='label_key', source='source.0')])
            return json.dumps(dict(turn_schema_version=1, decision=decision))
        if 'output_selection_schema_version' in value:
            return json.dumps(dict(outputs=[dict(name='comparison', step_id='score', output_key='report_path'),
                                           dict(name='map', step_id='view', output_key='analysis_path')]))
        assert value['user_request'] == OBJECTIVE
        steps = [dict(step_id='reduce', tool='pca_embed_cells', sources=[], control_dependencies=[]),
                 dict(step_id='links', tool='build_cell_neighbors', sources=[dict(kind='step', target='embedding', step='reduce')], control_dependencies=[]),
                 dict(step_id='partition', tool='cluster_cells', sources=[dict(kind='step', target='analysis', step='links')], control_dependencies=[]),
                 dict(step_id='score', tool='evaluate_cell_clustering', sources=[dict(kind='step', target='analysis',
                      step='reduce' if self.invalid_dependency else 'partition'),
                      dict(kind='input', target='ground_truth_dataset', input='reference_h5ad_path')], control_dependencies=[]),
                 dict(step_id='view', tool='compute_cell_umap', sources=[dict(kind='step', target='analysis', step='links')], control_dependencies=[])]
        if self.reordered:
            steps = [steps[i] for i in (4, 3, 1, 0, 2)]
        return json.dumps(dict(schema_version=4, decision=dict(kind='plan', steps=steps)))


def continuation_case(tmp_path, *, invalid_dependency=False, reordered=False):
    from agent.orchestration import build_default_tool_registry
    registry = build_default_tool_registry(); executor = WithholdScience(registry); calls = []
    app = InteractiveAgentApplication(tmp_path/'workspace', model_profiles=(PROFILE,),
        default_profile_id=PROFILE.profile_id, registry=registry, executor=executor,
        approved_source_roots=(tmp_path,),
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted':lambda _:
            ObservedContinuationModel(calls, invalid_dependency=invalid_dependency, reordered=reordered)}))
    app.create_session('session')
    upload_root = tmp_path/'workspace'/'uploads'; upload_root.mkdir()
    uploads = H5ADUploadAdmission(app.resources, upload_root)
    return app, executor, calls, uploads


def upload(app, uploads, tmp_path):
    obs = pd.DataFrame({'cohort':['a'] * 3 + ['b'] * 3,
                        LABEL:pd.Categorical(['A','A','B','B','C','C'])}, index=[f'cell{i}' for i in range(6)])
    source = tmp_path/'counts.h5ad'
    ad.AnnData(sparse.csr_matrix(np.arange(24).reshape(6, 4) % 5), obs=obs).write_h5ad(source)
    content = source.read_bytes()
    async def chunks():
        yield content
    return asyncio.run(uploads.receive('observed-matrix', source.name, chunks(),
        expected_size=len(content), session_id='session'))


@pytest.mark.parametrize('reordered', [False, True])
def test_missing_file_continuation_preserves_task_and_explicit_independent_dag_branches(tmp_path, reordered):
    app, executor, calls, uploads = continuation_case(tmp_path, reordered=reordered)
    initial = app.submit_turn('session', 'objective', OBJECTIVE, expected_generation=0)
    assert initial.status == 'clarification' and not executor.plans, initial.to_dict()
    origin = app._application.sessions.load('session').interactions[0]
    assert origin.prerequisite['field'] == 'roles'
    record = upload(app, uploads, tmp_path)
    result = app.submit_turn('session', 'upload', REPLY, expected_generation=0,
                             attachment_ids=(record.resource_id,))
    assert result.error.code == 'TEST_SCIENCE_WITHHELD', result.to_dict()
    plan, = executor.plans
    by_id = {s.step_id:s for s in plan.steps}
    assert by_id['reduce'].arguments['input_path'] == record.source_path
    assert by_id['reduce'].arguments['n_components'] == 2
    assert by_id['score'].arguments['reference_h5ad_path'] == record.source_path
    assert by_id['score'].arguments['label_key'] == LABEL
    assert by_id['score'].arguments['analysis_path'] == StepOutputRef('partition', 'analysis_path')
    assert by_id['score'].depends_on == ('partition',)
    assert by_id['view'].arguments['analysis_path'] == StepOutputRef('links', 'analysis_path')
    assert by_id['view'].depends_on == ('links',)
    assert 'score' not in by_id['view'].depends_on
    state = app._application.sessions.load('session')
    assert state.interactions[0] == origin and not state.revisions
    run = app._application.run_store.load(result.run_id)
    assert run.request.prompt == OBJECTIVE
    assert [v['user_request'] for v in calls if 'semantic_prompt_version' in v] == [OBJECTIVE]


def test_missing_required_comparison_dependency_does_not_silently_run_the_independent_map(tmp_path):
    app, executor, _, uploads = continuation_case(tmp_path, invalid_dependency=True)
    initial = app.submit_turn('session', 'objective', OBJECTIVE, expected_generation=0)
    assert initial.status == 'clarification', initial.to_dict()
    record = upload(app, uploads, tmp_path)
    result = app.submit_turn('session', 'upload', REPLY, expected_generation=0,
                             attachment_ids=(record.resource_id,))
    assert result.error is not None and not executor.plans, result.to_dict()
    assert result.error.code == 'ZERO_VALID_CHANNELS'
    run = app._application.run_store.load(result.run_id)
    assert not run.steps
    failure = run.errors[-1].details
    assert failure['stage'] == 'dependency_reference'
    assert failure['producer_step_id'] == 'reduce'
    assert failure['tool_name'] == 'evaluate_cell_clustering' and failure['target_port'] == 'analysis'
    state = app._application.sessions.load('session')
    assert not state.turns and not state.revisions


def test_rejected_restatement_retains_inherited_upload_for_parameter_repair(tmp_path):
    """Three real captures retain source authority through a new rejected request."""
    from agent.orchestration import build_default_tool_registry
    registry = build_default_tool_registry()
    executor = WithholdScience(registry)
    texts = ('Use 2 principal components for these cells.',
             'Instead use three principal components for these cells.',
             'Use 2 principal components.')
    decisions = {}
    for index, (text, literal) in enumerate(zip(texts, ('2', 'three', '2'))):
        start = text.index(literal)
        declaration = dict(tool='pca_embed_cells', argument='n_components',
                           literal=literal, start=start, end=start + len(literal))
        if index == 1:
            declaration['literal'] = '999'  # Model supplied an ungrounded value.
        decisions[text] = (dict(kind='answer_parameters', pending='@parameters', arguments=[declaration])
                           if index == 2 else dict(kind='execute_plan', target='pca_embed_cells', arguments=[declaration],
                               roles=[dict(tool='pca_embed_cells', port='dataset', source='source.0')]))

    class Model:
        model_id = PROFILE.model_id
        def complete(self, *, prompt, response_schema):
            value = json.loads(prompt)
            if 'turn_schema_version' in value:
                return json.dumps(dict(turn_schema_version=1, decision=decisions[value['utterance']]))
            if 'output_selection_schema_version' in value:
                return json.dumps(dict(outputs=[dict(name='pca', step_id='pca', output_key='embedding_path')]))
            return json.dumps(dict(schema_version=4, decision=dict(kind='plan', steps=[dict(
                step_id='pca', tool='pca_embed_cells', sources=[], control_dependencies=[])])))

    app = InteractiveAgentApplication(tmp_path/'workspace', model_profiles=(PROFILE,),
        default_profile_id=PROFILE.profile_id, registry=registry, executor=executor,
        approved_source_roots=(tmp_path,),
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted':lambda _: Model()}))
    app.create_session('session')
    upload_root = tmp_path/'workspace'/'uploads'; upload_root.mkdir()
    record = upload(app, H5ADUploadAdmission(app.resources, upload_root), tmp_path)
    first = app.submit_turn('session', 'initial', texts[0], expected_generation=0,
                            attachment_ids=(record.resource_id,))
    assert first.error.code == 'TEST_SCIENCE_WITHHELD', first.to_dict()
    executor.plans.clear()
    second = app.submit_turn('session', 'restatement', texts[1], expected_generation=0)
    assert second.status == 'clarification', second.to_dict()
    state = app._application.sessions.load('session')
    origin = state.interactions[-1]
    assert 'registered_input' not in (origin.submission or {})
    assert origin.admitted['registered_input']
    assert not executor.plans
    assert [turn.turn_id for turn in state.turns] == ['initial']
    result = app.submit_turn('session', 'correction', texts[2], expected_generation=0)
    assert result.error.code == 'TEST_SCIENCE_WITHHELD', result.to_dict()
    step, = executor.plans[0].steps
    assert step.arguments['input_path'] == record.source_path
    assert step.arguments['n_components'] == 2
    saved = app._application.sessions.load('session')
    assert saved.interactions[-2] == origin
    assert not saved.revisions
