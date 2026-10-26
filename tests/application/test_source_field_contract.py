"""Source-qualified observation declarations share ordinary role admission.

Scripted semantics exercise source identity, field intent and compiler/preflight;
the executor deliberately withholds native scientific production.
"""
import json
from pathlib import Path

import anndata as ad
import pandas as pd
import pytest

from agent.application import InteractiveAgentApplication
from agent.application.dialogue_execution import scientific_parameters
from agent.application.turn_decisions import (
    IntentError, ScalarChoiceArgument, ScopedArgument, SourceFieldArgument,
    admit_scientific_argument, decision_schema, parse_decision,
)
from agent.providers import PlanningModelFactoryRegistry
from agent.schemas.orchestration import _serialize
from test_multi_input_role_admission import LABEL, PROFILE, TransferBindingModel, transfer_binding_harness


PHRASE = 'the supplied reference annotations'
REQUEST = f'Transfer labels to the query using {PHRASE}.'


class Model(TransferBindingModel):
    def __init__(self, calls, *, reference, query, mode='offered', field=LABEL,
                 equal_role=False, conflict=False):
        super().__init__(calls)
        self.reference, self.query, self.mode, self.field = reference, query, mode, field
        self.equal_role, self.conflict = equal_role, conflict

    def complete(self, *, prompt, response_schema):
        value = json.loads(prompt)
        if 'turn_schema_version' in value:
            self.calls.append(dict(value=value, prompt=prompt, schema=response_schema))
            literal = self.field if self.mode == 'literal' else PHRASE
            start = value['utterance'].index(literal)
            roles = [dict(tool='transfer_cell_labels', port='query_dataset', source=self.query)]
            if self.equal_role or self.conflict:
                roles.append(dict(tool='transfer_cell_labels', port='reference_dataset',
                                  source=self.query if self.conflict else self.reference))
            argument = dict(tool='transfer_cell_labels', argument='reference_label_key',
                source=self.reference, mode=self.mode, value=self.field,
                literal=literal, start=start, end=start + len(literal))
            return json.dumps(dict(turn_schema_version=1, decision=dict(kind='execute_plan',
                target='transfer_cell_labels', roles=roles, arguments=[argument])), ensure_ascii=False)
        return super().complete(prompt=prompt, response_schema=response_schema)


def case(tmp_path, *, reverse=False, outside_page=False, **options):
    seed, records, group, executor, calls, companions = transfer_binding_harness(tmp_path)
    if outside_page:
        original = ad.read_h5ad(records[0].source_path)
        labels = list(original.obs[LABEL])
        original.obs = pd.DataFrame({f'field_{index}':pd.Categorical(['one'] * original.n_obs)
            for index in range(65)}, index=original.obs_names)
        original.obs[LABEL] = pd.Categorical(labels)
        path = Path(records[0].source_path).with_name('many-fields.h5ad')
        original.write_h5ad(path)
        records[0] = seed.resources.register('many-fields', path, label='Reference with a bounded field page',
            attribution='Simulated metadata paging fixture only.')
    selected = list(reversed(records)) if reverse else records
    group = seed.resources.compose_group([record.resource_id for record in selected], companions)
    reference, query = ('source.1', 'source.0') if reverse else ('source.0', 'source.1')
    options = dict(reference=reference, query=query) | options
    app = InteractiveAgentApplication(seed._application.workspace_root, model_profiles=(PROFILE,),
        default_profile_id=PROFILE.profile_id, registry=seed._application.registry, executor=executor,
        approved_source_roots=(tmp_path/'approved',),
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted':lambda _: Model(calls, **options)}))
    return app, records, group, executor, calls


def nodes(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from nodes(child)
    elif isinstance(value, list):
        for child in value:
            yield from nodes(child)


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('equal_role', [False, True])
def test_field_source_assertion_supplies_only_its_registry_dataset_port(tmp_path, reverse, equal_role):
    app, records, group, executor, calls = case(tmp_path, reverse=reverse, equal_role=equal_role)
    result = app.submit_turn('session', 'field', REQUEST, expected_generation=0, registered_input=group)
    assert result.error.code == 'TEST_SCIENCE_WITHHELD', result.to_dict()
    step, = executor.plans[0].steps
    assert step.arguments['reference_h5ad_path'] == records[0].source_path
    assert step.arguments['query_h5ad_path'] == records[1].source_path
    assert step.arguments['reference_label_key'] == LABEL and LABEL not in REQUEST
    interaction, = app._application.sessions.load('session').interactions
    argument, = interaction.admitted['arguments']
    assert argument['source'] == ('source.1' if reverse else 'source.0')
    assert argument['mode'] == 'offered'
    roles = {(row['tool'],row['port']):row['source'] for row in interaction.admitted['roles']}
    assert len(roles) == 2 and roles[('transfer_cell_labels','reference_dataset')] == argument['source']
    schema = calls[0]['schema']
    variants = [row['properties'] for row in nodes(schema) if set(row.get('properties', {})) ==
                {'tool','argument','source','mode','value','literal','evidence'}
                and row['properties']['tool'].get('enum') == ['transfer_cell_labels']]
    assert variants
    offered = {row['source']:row['fields'] for row in calls[0]['value']['dialogue']['source_fields']}
    for row in variants:
        if row['mode']['enum'] == ['offered']:
            assert row['value']['enum'] == offered[row['source']['enum'][0]]


@pytest.mark.parametrize('change', [dict(reference='source.1'), dict(reference='source.9'),
    dict(conflict=True), dict(field='never offered')])
def test_wrong_source_unknown_handle_conflicting_role_and_unoffered_name_fail_closed(tmp_path, change):
    app, _, group, executor, _ = case(tmp_path, **change)
    result = app.submit_turn('session', 'invalid', REQUEST, expected_generation=0, registered_input=group)
    assert result.error is not None and not executor.plans, result.to_dict()
    state = app._application.sessions.load('session')
    assert not state.turns and not state.revisions


def test_explicit_literal_outside_bounded_offer_is_a_separate_declaration(tmp_path):
    app, records, group, executor, calls = case(tmp_path, outside_page=True, mode='literal')
    text = f'Transfer labels using reference column "{LABEL}".'
    result = app.submit_turn('session', 'exact', text, expected_generation=0, registered_input=group)
    assert result.error.code == 'TEST_SCIENCE_WITHHELD', result.to_dict()
    row = calls[0]['value']['dialogue']['source_fields'][0]
    assert not row['complete'] and LABEL not in row['fields']
    step, = executor.plans[0].steps
    assert step.arguments['reference_h5ad_path'] == records[0].source_path
    assert step.arguments['reference_label_key'] == LABEL


def test_offered_mode_does_not_fall_back_when_its_value_equals_an_unoffered_literal(tmp_path):
    app, _, group, executor, _ = case(tmp_path, field=PHRASE)
    result = app.submit_turn('session', 'no-fallback', REQUEST, expected_generation=0, registered_input=group)
    assert result.error is not None and not executor.plans
    assert not app._application.sessions.load('session').turns


def test_source_qualified_wire_requires_value_and_mode_and_retains_old_readers():
    def wire(argument):
        return json.dumps(dict(turn_schema_version=1, decision=dict(kind='execute_plan',
            target='transfer_cell_labels', arguments=[argument])))
    text = 'Use supplied annotations.'
    base = dict(tool='transfer_cell_labels', argument='reference_label_key',
                literal='supplied annotations', evidence='supplied annotations')
    complete = base | dict(source='source.0', value='cell identity', mode='offered')
    decision = parse_decision(wire(complete), utterance=text)
    assert isinstance(decision.arguments[0], SourceFieldArgument)
    for key in ('value', 'mode'):
        with pytest.raises(IntentError):
            parse_decision(wire({name:value for name,value in complete.items() if name != key}), utterance=text)
    assert type(parse_decision(wire(base), utterance=text).arguments[0]) is ScopedArgument
    assert type(parse_decision(wire(base | dict(value='cell identity')), utterance=text).arguments[0]) is ScalarChoiceArgument


def test_metadata_scope_absence_does_not_advertise_a_source_free_bypass(tmp_path):
    app, _, group, _, calls = case(tmp_path)
    app.submit_turn('session', 'field', REQUEST, expected_generation=0, registered_input=group)
    public = calls[0]['value']
    public['dialogue'].pop('source_fields')
    properties = [row['properties'] for row in nodes(decision_schema(public))
                  if row.get('properties', {}).get('argument', {}).get('enum') == ['reference_label_key']]
    assert not properties
    public['dialogue']['selected_sources'] = []
    properties = [row['properties'] for row in nodes(decision_schema(public))
                  if row.get('properties', {}).get('argument', {}).get('enum') == ['reference_label_key']]
    assert properties and all(set(row) == {'tool','argument','literal','evidence'} for row in properties)


def test_registry_metadata_does_not_advertise_an_unregistered_group_port(tmp_path):
    app, _, _, _, _ = case(tmp_path)
    parameters = scientific_parameters(app._application.registry)
    assert all('choices' not in row for row in parameters if row.get('observation_source'))
    annotation, = [row for row in parameters if (row['tool'],row['argument']) ==
                   ('evaluate_cell_annotation','ground_truth_label_key')]
    assert 'observation_source' not in annotation and annotation['exact_string']


@pytest.mark.parametrize('mode', ['offered', 'literal'])
def test_modes_cannot_silently_reinterpret_the_same_natural_reference(tmp_path, mode):
    app, _, _, _, _ = case(tmp_path)
    argument = app._application.registry.get('transfer_cell_labels').required_arguments['reference_label_key']
    start = REQUEST.index(PHRASE)
    selected = SourceFieldArgument('transfer_cell_labels', 'reference_label_key', PHRASE,
        start, start + len(PHRASE), LABEL, 'source.0', mode)
    if mode == 'offered':
        assert admit_scientific_argument(selected, REQUEST, argument, observation_fields=(LABEL,)) == LABEL
    else:
        with pytest.raises(IntentError) as error:
            admit_scientific_argument(selected, REQUEST, argument, observation_fields=(LABEL,))
        assert error.value.diagnostic_reason == 'source_field_literal_mismatch'


def test_source_field_shape_cannot_be_reused_as_a_static_scalar_choice(tmp_path):
    app, _, _, _, _ = case(tmp_path)
    spec = app._application.registry.get('epizoo_embed_cells').required_arguments['species']
    text = 'Use human cells.'
    selected = SourceFieldArgument('epizoo_embed_cells', 'species', 'human', 4, 9,
                                  'human', 'source.0', 'offered')
    with pytest.raises(IntentError):
        admit_scientific_argument(selected, text, spec)
