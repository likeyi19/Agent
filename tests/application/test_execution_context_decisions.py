"""The generic execution scope is typed and conditional on captured offers."""
import json

import pytest

from agent.application.turn_decisions import Execute, IntentError, decision_schema, interpret, parse_decision


def public_context(*, offered=False):
    public = dict(bases={}, relations=['current'], dialogue=dict(outputs=[], tools=['cluster_cells']))
    if offered:
        public['dialogue']['planning_context'] = [dict(handle='ctx.0', semantic_type='cell_neighbors.v1',
            producer_tool='build_cell_neighbors', source_port='neighbors')]
    return public


def wire(value):
    return json.dumps(dict(turn_schema_version=1, decision=value))


@pytest.mark.parametrize('context', ['none', 'current'])
def test_execution_scope_is_a_closed_generic_choice(context):
    result = parse_decision(wire(dict(kind='execute_plan', target='cluster_cells', context=context)))
    assert result == Execute('current', 'plan', 'cluster_cells', None, context=context)


@pytest.mark.parametrize('context', [None, True, 1, {}, [], '', 'parent', 'ctx.0'])
def test_context_is_never_a_locator_or_free_form_memory(context):
    with pytest.raises(IntentError, match='invalid_decision'):
        parse_decision(wire(dict(kind='execute_plan', target='cluster_cells', context=context)))


@pytest.mark.parametrize('decision', [
    dict(kind='execute_plan', target='cluster_cells'),
    dict(kind='execute_plan', base='current', operation='plan', target='cluster_cells', delta=None),
    dict(kind='execute', base='current', operation='plan', target='cluster_cells', delta=None),
])
def test_previous_execution_wires_do_not_implicitly_inherit_context(decision):
    assert parse_decision(wire(decision)).context == 'none'


def test_specialized_execution_cannot_bypass_its_existing_context_contract():
    with pytest.raises(ValueError, match='execution context'):
        Execute('current', 'selection', 'selection', None, context='current')


@pytest.mark.parametrize('offered', [False, True])
def test_context_schema_is_offered_only_with_recorded_qualified_items(offered):
    variants = decision_schema(public_context(offered=offered))['properties']['decision']['anyOf']
    execution = next(item for item in variants if item['properties']['kind']['enum'] == ['execute_plan'])
    assert execution['additionalProperties'] is False
    if offered:
        assert execution['properties']['context'] == dict(type='string', enum=['none', 'current'])
        assert 'context' in execution['required']
    else:
        assert 'context' not in execution['properties'] and 'context' not in execution['required']
    assert all('context' not in item['properties'] for item in variants if item is not execution)


@pytest.mark.parametrize('offered', [False, True])
def test_interpreter_receives_bounded_scope_instructions_without_session_identities(offered):
    class Model:
        def complete(self, *, prompt, response_schema):
            value = json.loads(prompt)
            context_rules = [text for text in value['instructions'] if 'context=current' in text]
            assert len(context_rules) == int(offered)
            assert value.get('dialogue', {}).get('planning_context') == (
                public_context(offered=True)['dialogue']['planning_context'] if offered else None)
            return wire(dict(kind='execute_plan', target='cluster_cells', context='current' if offered else 'none'))
    assert interpret(Model(), 'Continue the captured analysis.', public_context(offered=offered)).context == (
        'current' if offered else 'none')
