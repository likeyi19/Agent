"""One typed Interpreter proposal selects exact offered input consumers."""
from dataclasses import FrozenInstanceError, asdict, replace
import json

import pytest

from agent.application.turn_decisions import (
    Execute, InputRole, IntentError, decision_schema, interpret, parse_decision,
)


def public(*, selected=True):
    value = dict(bases={}, relations=[], dialogue=dict(outputs=[],
        tools=['evaluate_cell_clustering', 'transfer_cell_labels']))
    if selected:
        value['dialogue'].update(selected_sources=[
            dict(source='source.0', position=1, label='Fixed analysis', input_type='h5ad'),
            dict(source='source.1', position=2, label='Independent labels', input_type='h5ad'),
        ], input_roles=[
            dict(tool='evaluate_cell_clustering', port='analysis', role='branch_preserving'),
            dict(tool='evaluate_cell_clustering', port='ground_truth_dataset', role='ground_truth'),
            dict(tool='transfer_cell_labels', port='reference_dataset', role='reference'),
            dict(tool='transfer_cell_labels', port='query_dataset', role='query'),
        ])
    return value


def wire(submitted_roles=None, **changes):
    value = dict(kind='execute_plan', target='evaluate_cell_clustering')
    if submitted_roles is not None:
        value['roles'] = submitted_roles
    value.update(changes)
    return json.dumps(dict(turn_schema_version=1, decision=value))


def roles():
    return [dict(tool='evaluate_cell_clustering', port='analysis', source='source.0'),
            dict(tool='evaluate_cell_clustering', port='ground_truth_dataset', source='source.1')]


def execution(schema):
    return next(item['properties'] for item in schema['properties']['decision']['anyOf']
                if item['properties']['kind']['enum'] == ['execute_plan'])


def test_exact_role_proposals_are_immutable_and_independent_of_upload_order():
    value = parse_decision(wire(roles()))
    assert value.roles == tuple(InputRole(**item) for item in roles())
    assert value.context == 'none'
    assert value.argument is None and value.arguments == ()
    with pytest.raises(FrozenInstanceError):
        value.roles[0].source = 'source.1'
    reversed_roles = list(reversed(roles()))
    assert parse_decision(wire(reversed_roles)).roles == tuple(InputRole(**item) for item in reversed_roles)


@pytest.mark.parametrize('submitted', [
    'source.0', None, [dict(tool='evaluate_cell_clustering', source='source.0')],
    [dict(tool='evaluate_cell_clustering', port='analysis', source='source.16')],
    [dict(tool='evaluate_cell_clustering', port='analysis', source='/private/analysis.h5ad')],
    [dict(tool='evaluate_cell_clustering', port='analysis', source='source.0', role='query')],
    [dict(tool='evaluate_cell_clustering', port='analysis', source='source.0')]*2,
    [dict(tool='evaluate_cell_clustering', port='analysis', source='source.0')]*9,
])
def test_invalid_or_conflicting_role_wire_fails_closed(submitted):
    with pytest.raises(IntentError, match='invalid_decision'):
        # Keep a literal null in this case; omitted roles retain the old wire.
        parse_decision(wire([], roles=submitted))


def test_existing_execute_wire_and_schema_do_not_require_new_selection():
    assert parse_decision(wire()) == Execute('current', 'plan', 'evaluate_cell_clustering', None)
    assert 'roles' not in execution(decision_schema(public(selected=False)))


def test_schema_scopes_each_role_to_its_registry_consumer_and_captured_handles():
    items = execution(decision_schema(public()))['roles']
    assert items['maxItems'] == 8
    assert [(item['properties']['tool']['enum'], item['properties']['port']['enum'])
            for item in items['items']['anyOf']] == [
        ([item['tool']], [item['port']]) for item in public()['dialogue']['input_roles']]
    for item in items['items']['anyOf']:
        assert item['additionalProperties'] is False
        assert item['properties']['source']['enum'] == ['source.0', 'source.1']
        assert item['required'] == ['tool', 'port', 'source']


def test_interpreter_receives_only_safe_captured_choices_and_owns_role_meaning():
    class Model:
        def complete(self, *, prompt, response_schema):
            self.prompt, self.schema = json.loads(prompt), response_schema
            return wire(roles())
    model = Model()
    request = 'Evaluate the first selected analysis against the second selected ground truth.'
    result = interpret(model, request, public())
    assert [asdict(item) for item in result.roles] == roles()
    assert model.prompt['utterance'] == request
    assert model.prompt['dialogue']['selected_sources'] == public()['dialogue']['selected_sources']
    instructions = ' '.join(model.prompt['instructions'])
    assert 'order, filenames and metadata never imply biological roles' in instructions
    assert 'Pending parameter answers retain the exact original selection and roles' in instructions
    assert 'source.0' in json.dumps(model.schema)
    assert '/private/' not in json.dumps(model.prompt)


@pytest.mark.parametrize('selected', [True, False])
def test_grouped_interpreter_utf8_preserves_json_values_and_existing_wire_encoding(selected):
    value = public(selected=selected)
    utterance = 'Use the selected cells 类型 🧬.'
    if selected:
        value['dialogue']['selected_sources'][1]['label'] = 'Ground truth 类型 🧬'
    class Model:
        def complete(self, *, prompt, response_schema):
            self.raw = prompt
            return wire(roles() if selected else None)
    model = Model()
    interpret(model, utterance, value)
    decoded = json.loads(model.raw)
    assert decoded['utterance'] == utterance
    assert decoded['dialogue'] == value['dialogue']
    if selected:
        assert 'Ground truth 类型 🧬' in model.raw
        assert r'\ud83e\uddec' not in model.raw
        assert '/unopened/' not in model.raw
    else:
        assert '类型' not in model.raw
        assert r'\ud83e\uddec' in model.raw


def captured_interaction():
    from agent.application.local_resources import RegisteredInputGroup, group_role_inputs
    from agent.application.session_state import Interaction
    from agent.orchestration.registry import build_default_tool_registry
    from agent.schemas.orchestration import _serialize
    registry = build_default_tool_registry()
    members = [dict(resource_id='local-' + str(index)*64, record_sha256=str(index+2)*64,
                    source_path='/unopened/' + name + '.h5ad', input_type='h5ad', label=name)
               for index, name in enumerate(('analysis', 'truth'))]
    original = RegisteredInputGroup(members, {})
    values = group_role_inputs(original, roles(), registry)
    bound = RegisteredInputGroup(members, values, roles=roles())
    submission = dict(registered_input=original.attribution(), execution_inputs=_serialize(original.execution_inputs))
    admitted = dict(kind='execute', operation='plan', tool='evaluate_cell_clustering',
                    inputs=values, roles=roles(), registered_input=bound.attribution())
    return Interaction('turn', 'Evaluate the selected fixed analysis against selected ground truth.',
                       None, 0, {}, admitted=admitted, submission=submission)


def test_captured_role_replay_uses_private_members_without_opening_scientific_sources(monkeypatch):
    from agent.application.dialogue_execution import _validate_argument_record, _validate_input_role_record
    import builtins
    interaction = captured_interaction()
    def forbidden(*args, **kwargs):
        raise AssertionError('Historical input-role replay opened a source.')
    monkeypatch.setattr(builtins, 'open', forbidden)
    _validate_input_role_record(interaction)
    _validate_argument_record(interaction)


@pytest.mark.parametrize('variant', ['missing_roles', 'reassigned_source', 'changed_alias',
                                     'changed_members', 'changed_role_attribution'])
def test_persisted_role_replay_rejects_scope_or_identity_changes(variant):
    from agent.application.dialogue_execution import _validate_input_role_record
    from agent.application.session_state import SessionError
    from agent.schemas.orchestration import _serialize
    interaction = captured_interaction()
    admitted = _serialize(interaction.admitted)
    if variant == 'missing_roles':
        admitted.pop('roles')
    elif variant == 'reassigned_source':
        admitted['roles'][0]['source'] = 'source.1'
    elif variant == 'changed_alias':
        admitted['inputs']['reference_h5ad_path'] = admitted['inputs']['analysis_path']
    elif variant == 'changed_members':
        admitted['registered_input']['members'][1]['record_sha256'] = 'f'*64
    else:
        admitted['registered_input']['roles'][0]['source'] = 'source.1'
    with pytest.raises(SessionError, match='scientific input roles'):
        _validate_input_role_record(replace(interaction, admitted=admitted))


def parameter_continuation():
    from agent.application.session_state import Interaction
    from agent.schemas.orchestration import _serialize
    original = captured_interaction()
    pending = dict(origin_turn_id='turn', binding_turn_id='turn', field='parameters',
                   resource_context_sha256='f'*64, tool='evaluate_cell_clustering', missing=['label_key'])
    original = replace(original, status='clarification', prerequisite=pending)
    utterance = 'Use the ground-truth column "Cell 类型 Δ".'
    literal = 'Cell 类型 Δ'
    start = utterance.index(literal)
    declaration = dict(tool='evaluate_cell_clustering', argument='label_key',
                       literal=literal, start=start, end=start+len(literal))
    admitted = _serialize(original.admitted)
    admitted.update(inputs=admitted['inputs'] | dict(label_key=literal), arguments=[declaration],
                    continuation=dict(origin_turn_id='turn', binding_turn_id='turn', field='parameters'),
                    revision_id=None)
    current = Interaction('answer', utterance, None, 0,
        dict(prerequisite=pending | dict(predecessor_turn_id='turn')), admitted=admitted,
        submission=dict(execution_inputs={}))
    return original, current


def test_parameter_continuation_preserves_frozen_group_and_exact_roles_without_reselection():
    from agent.application.dialogue_execution import _validate_parameter_prerequisite
    original, current = parameter_continuation()
    _validate_parameter_prerequisite(current, (original,))
    assert current.admitted['registered_input']['members'] == original.admitted['registered_input']['members']
    assert current.admitted['roles'] == original.admitted['roles']


@pytest.mark.parametrize('variant', ['switched_registration', 'changed_roles', 'lost_roles', 'wrong_label_role'])
def test_pending_answer_cannot_replace_registered_members_or_label_scope(variant):
    from agent.application.dialogue_execution import _validate_parameter_prerequisite
    from agent.application.session_state import SessionError
    from agent.schemas.orchestration import _serialize
    original, current = parameter_continuation()
    admitted = _serialize(current.admitted)
    if variant == 'switched_registration':
        admitted['registered_input']['members'][1]['record_sha256'] = 'a'*64
    elif variant == 'changed_roles':
        admitted['roles'][0]['source'] = 'source.1'
    elif variant == 'lost_roles':
        admitted.pop('roles')
    else:
        admitted['arguments'][0]['tool'] = 'transfer_cell_labels'
        admitted['arguments'][0]['argument'] = 'reference_label_key'
    with pytest.raises(SessionError, match='pending scientific parameter binding'):
        _validate_parameter_prerequisite(replace(current, admitted=admitted), (original,))


def test_changed_current_selection_is_not_offered_as_an_answer_to_previous_pending_request():
    from agent.application.dialogue_execution import capture_prerequisite
    from agent.schemas.orchestration import _serialize
    original, _ = parameter_continuation()
    changed = _serialize(original.submission)
    changed['registered_input']['members'][1]['record_sha256'] = 'a'*64
    assert capture_prerequisite((original,), None, 0, changed) is None
