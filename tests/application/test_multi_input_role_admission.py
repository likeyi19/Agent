"""Independent transfer roles reuse ordinary grouped admission and compilation.

The recording executor stops before any science. Synthetic embedding/ID/model
companions are explicitly supplied typed inputs and do not establish upstream
foundation-model authority or qualify a second scientific workflow.
"""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from agent.application import InteractiveAgentApplication
from agent.application.local_resources import ResourceAdmissionError
from agent.orchestration import PlanningModelProfile, build_default_tool_registry
from agent.providers import PlanningModelFactoryRegistry
from agent.schemas.orchestration import _serialize
from test_human_epizoo_mapping import WithholdScience


PROFILE = PlanningModelProfile('role-transfer', 'scripted', 'scripted/role-transfer')
LABEL = 'Cell Type α'
UTTERANCE = (f'Use the first selected dataset as reference and the second as query. '
             f'Transfer cell labels using reference column "{LABEL}".')
ROLES = [dict(tool='transfer_cell_labels', port='reference_dataset', source='source.0'),
         dict(tool='transfer_cell_labels', port='query_dataset', source='source.1')]


class TransferBindingModel:
    def __init__(self, calls, *, variant=None, roles=None):
        self.model_id = PROFILE.model_id
        self.calls, self.variant = calls, variant
        self.roles = deepcopy(ROLES if roles is None else roles)

    def complete(self, *, prompt, response_schema):
        value = json.loads(prompt)
        self.calls.append(dict(value=value, prompt=prompt, schema=response_schema))
        if 'turn_schema_version' in value:
            start = value['utterance'].index(LABEL)
            return json.dumps(dict(turn_schema_version=1, decision=dict(kind='execute_plan',
                target='transfer_cell_labels', roles=self.roles,
                arguments=[dict(tool='transfer_cell_labels', argument='reference_label_key',
                                literal=LABEL, start=start, end=start + len(LABEL))])))
        if 'selection_schema_version' in response_schema.get('properties', {}):
            return json.dumps(dict(selection_schema_version=1,
                decision=dict(kind='select', capability_ids=['reference_annotation'])))
        if 'output_selection_schema_version' in value:
            return json.dumps(dict(outputs=[dict(name='annotation', step_id='transfer',
                                                output_key='annotation_path')]))
        sources = [dict(target=port, source=dict(kind='input', input=key)) for port, key in (
            ('reference_dataset', 'reference_input_path'), ('query_dataset', 'query_input_path'),
            ('reference_embedding', 'reference_embedding_path'), ('query_embedding', 'query_embedding_path'))]
        if self.variant == 'swap_datasets':
            sources[0]['source']['input'], sources[1]['source']['input'] = (
                sources[1]['source']['input'], sources[0]['source']['input'])
        elif self.variant == 'swap_embeddings':
            sources[2]['source']['input'], sources[3]['source']['input'] = (
                sources[3]['source']['input'], sources[2]['source']['input'])
        elif self.variant == 'wrong_port':
            sources[0]['target'] = 'ground_truth_dataset'
        steps = [dict(step_id='transfer', tool='transfer_cell_labels', sources=sources,
                      control_dependencies=[])]
        if self.variant == 'repeat_consumer':
            steps.append(deepcopy(steps[0]) | dict(step_id='transfer-again'))
        return json.dumps(dict(schema_version=4, decision=dict(kind='plan', steps=steps)))


def transfer_binding_harness(tmp_path, *, variant=None, roles=None):
    approved = tmp_path / 'approved'
    approved.mkdir()
    paths, companions = [], {}
    for role, count in [('reference', 4), ('query', 2)]:
        ids = [f'{role}-{index}' for index in range(count)]
        obs = pd.DataFrame(index=pd.Index(ids))
        if role == 'reference':
            obs[LABEL] = ['α', 'α', 'β', 'β']
        path = approved / (role + '.h5ad')
        ad.AnnData(X=sparse.csr_matrix((count, 3), dtype=np.float32), obs=obs).write_h5ad(path)
        paths.append(path)
        embedding = approved / (role + '.npy')
        np.save(embedding, np.zeros((count, 512), dtype=np.float32), allow_pickle=False)
        ordered_ids = approved / (role + '.txt')
        ordered_ids.write_text(''.join(value + '\n' for value in ids), encoding='utf-8')
        companions.update({role + '_embedding_path': str(embedding),
            role + '_cell_ids_path': str(ordered_ids), role + '_species': 'mouse',
            role + '_checkpoint_path': str(approved / 'synthetic-model.pth')})
    (approved / 'synthetic-model.pth').write_bytes(b'Synthetic provenance fixture; never loaded.')
    registry = build_default_tool_registry()
    executor = WithholdScience(registry)
    calls = []
    app = InteractiveAgentApplication(tmp_path / 'workspace', model_profiles=(PROFILE,),
        default_profile_id=PROFILE.profile_id, registry=registry, executor=executor,
        approved_source_roots=(approved,),
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted':
            lambda _: TransferBindingModel(calls, variant=variant, roles=roles)}))
    records = [app.resources.register(role, path, label=f'Selected {role}',
        attribution='Explicit synthetic source registration, with no biological authority.')
        for role, path in zip(('reference', 'query'), paths)]
    group = app.resources.compose_group([record.resource_id for record in records], companions)
    app.create_session('session')
    return app, records, group, executor, calls, companions


def test_transfer_contract_shares_exact_group_role_binding_and_compiler(tmp_path):
    app, records, group, executor, calls, companions = transfer_binding_harness(tmp_path)
    result = app.submit_turn('session', 'transfer', UTTERANCE, expected_generation=0, registered_input=group)
    assert result.status == 'failed' and result.error.code == 'TEST_SCIENCE_WITHHELD', result.to_dict()
    assert result.revision_id is None and not result.steps
    plan, = executor.plans
    step, = plan.steps
    assert step.tool_name == 'transfer_cell_labels'
    assert step.arguments['reference_h5ad_path'] == records[0].source_path
    assert step.arguments['query_h5ad_path'] == records[1].source_path
    assert step.arguments['reference_label_key'] == LABEL
    for name, value in companions.items():
        assert step.arguments[name] == value
    state = app._application.sessions.load('session')
    assert not state.revisions and state.generation == 0
    interaction, = state.interactions
    assert _serialize(interaction.admitted['roles']) == ROLES
    members = _serialize(interaction.submission['registered_input'])['members']
    assert [(row['resource_id'], row['record_sha256']) for row in members] == [
        (record.resource_id, record.record_sha256) for record in records]
    assert interaction.admitted['inputs']['reference_input_path'] == records[0].source_path
    assert interaction.admitted['inputs']['query_input_path'] == records[1].source_path
    assert len(calls) == 3
    for call in calls:
        assert str(tmp_path) not in call['prompt']
        assert not any(record.record_sha256 in call['prompt'] or record.resource_id in call['prompt'] for record in records)
    (tmp_path / 'secondary-transfer-binding-witness.json').write_text(json.dumps(dict(
        scope='Shared grouped admission and compiler only; the executor deliberately withholds all science.',
        members=members, roles=ROLES, compiled_arguments=_serialize(step.arguments),
        verification='Existing executor preflight passed; no scientific verification claimed.'),
        indent=2, ensure_ascii=False), encoding='utf-8')


def test_explicit_roles_keep_identity_when_user_selection_order_is_reversed(tmp_path):
    roles = [ROLES[0] | dict(source='source.1'), ROLES[1] | dict(source='source.0')]
    app, records, original, executor, _, companions = transfer_binding_harness(tmp_path, roles=roles)
    group = app.resources.compose_group([record.resource_id for record in reversed(records)], companions)
    utterance = (f'Use the second selected dataset as reference and the first as query. '
                 f'Transfer cell labels using reference column "{LABEL}".')
    result = app.submit_turn('session', 'transfer', utterance, expected_generation=0, registered_input=group)
    assert result.error.code == 'TEST_SCIENCE_WITHHELD', result.to_dict()
    step, = executor.plans[0].steps
    assert step.arguments['reference_h5ad_path'] == records[0].source_path
    assert step.arguments['query_h5ad_path'] == records[1].source_path
    assert step.arguments['reference_label_key'] == LABEL
    assert [row['resource_id'] for row in group.members] == [record.resource_id for record in reversed(records)]
    assert group.members != original.members
    assert not app._application.sessions.load('session').revisions


@pytest.mark.parametrize('variant', ['swap_datasets', 'swap_embeddings', 'wrong_port', 'repeat_consumer'])
def test_transfer_lineages_and_consumers_reject_substitution(tmp_path, variant):
    app, _, group, executor, _, _ = transfer_binding_harness(tmp_path, variant=variant)
    result = app.submit_turn('session', 'transfer', UTTERANCE, expected_generation=0, registered_input=group)
    assert result.revision_id is None and not executor.plans, result.to_dict()
    assert result.response.status in ('failed', 'clarification')
    assert not app._application.sessions.load('session').revisions


@pytest.mark.parametrize('roles', [
    ROLES + [ROLES[0]],
    [ROLES[0], ROLES[1] | dict(source='source.0')],
    [ROLES[0], ROLES[1] | dict(source='source.2')],
    [ROLES[0], ROLES[1] | dict(port='ground_truth_dataset')],
])
def test_transfer_duplicate_conflicting_unselected_and_wrong_roles_are_rejected(tmp_path, roles):
    app, _, group, executor, _, _ = transfer_binding_harness(tmp_path, roles=roles)
    result = app.submit_turn('session', 'transfer', UTTERANCE, expected_generation=0, registered_input=group)
    assert result.revision_id is None and not executor.plans, result.to_dict()
    assert not app._application.sessions.load('session').revisions


def test_new_consumption_detects_replaced_registered_source_before_interpretation(tmp_path):
    app, records, group, executor, calls, _ = transfer_binding_harness(tmp_path)
    Path(records[1].source_path).write_bytes(b'replaced after exact source selection')
    with pytest.raises(ResourceAdmissionError) as rejected:
        app.submit_turn('session', 'transfer', UTTERANCE, expected_generation=0, registered_input=group)
    assert rejected.value.code == 'LOCAL_RESOURCE_INTEGRITY_INVALID'
    assert not calls and not executor.plans
    assert not app._application.sessions.load('session').interactions


def test_selection_metadata_cannot_be_changed_after_capture(tmp_path):
    app, _, group, executor, calls, _ = transfer_binding_harness(tmp_path)
    members = _serialize(group.members)
    members[1]['record_sha256'] = 'f' * 64
    forged = replace(group, members=members)
    with pytest.raises(ResourceAdmissionError):
        app.submit_turn('session', 'transfer', UTTERANCE, expected_generation=0, registered_input=forged)
    assert not calls and not executor.plans
    assert not app._application.sessions.load('session').interactions
