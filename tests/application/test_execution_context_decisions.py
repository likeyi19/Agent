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
    assert result == Execute('cluster_cells', context=context)


@pytest.mark.parametrize('context', [None, True, 1, {}, [], '', 'parent', 'ctx.0'])
def test_context_is_never_a_locator_or_free_form_memory(context):
    with pytest.raises(IntentError, match='invalid_decision'):
        parse_decision(wire(dict(kind='execute_plan', target='cluster_cells', context=context)))


@pytest.mark.parametrize('decision', [
    dict(kind='execute_plan', target='cluster_cells'),
])
def test_previous_execution_wires_do_not_implicitly_inherit_context(decision):
    assert parse_decision(wire(decision)).context == 'none'


@pytest.mark.parametrize('offered', [False, True])
def test_interpreter_receives_bounded_scope_instructions_without_session_identities(offered):
    class Model:
        def complete(self, *, prompt, response_schema):
            value = json.loads(prompt)
            context_rules = [text for text in value['instructions'] if 'choose context=current only' in text]
            assert len(context_rules) == int(offered)
            assert value.get('dialogue', {}).get('planning_context') == (
                public_context(offered=True)['dialogue']['planning_context'] if offered else None)
            return wire(dict(kind='execute_plan', target='cluster_cells', context='current' if offered else 'none'))
    assert interpret(Model(), 'Continue the captured analysis.', public_context(offered=offered)).context == (
        'current' if offered else 'none')




@pytest.mark.parametrize('field,value', [
    ('delivery', 'explanation'), ('unfulfilled', 'Infer cell lineages'), ('references', 'g0'),
])
def test_research_intention_uniqueness_remains_owned_by_typed_validation(field, value):
    with pytest.raises(ValueError, match='Invalid bounded research intention'):
        Execute('cluster_cells', **{field: (value, value)})
    with pytest.raises(IntentError, match='invalid_decision'):
        parse_decision(wire(dict(kind='execute_plan', target='cluster_cells', **{field: [value, value]})))


@pytest.mark.parametrize('tool,name,value,new_name,new_value,new_input', [
    ('build_cell_neighbors', 'n_neighbors', 7, 'random_seed', 2, 'neighbors_random_seed'),
    ('compute_cell_umap', 'min_dist', .3, 'spread', 1.2, 'spread'),
])
def test_retained_settings_bind_separately_from_current_values_and_defaults(tool, name, value, new_name, new_value, new_input):
    from dataclasses import replace
    from types import SimpleNamespace
    from agent.application.dialogue_execution import (admit, _validate_argument_record,
        _validate_research_intent, _check_parameter_plan)
    from agent.application.session_state import Interaction
    from agent.application.turn_decisions import RetainedParameter, ScopedArgument
    from agent.orchestration import build_default_tool_registry
    from agent.schemas import AgentPlan, PlanStep
    registry = build_default_tool_registry()
    utterance = f'Keep the settings; change {new_name} to {new_value}.'
    literal = str(new_value)
    start = utterance.index(literal)
    row = dict(handle='g0', tool=tool, parameters=[dict(argument=name, value=value, origin='owner_effective_value')])
    interaction = Interaction('change', utterance, 'captured', 0,
                              dict(dialogue=dict(results=[row])))
    sessions = SimpleNamespace(_application=SimpleNamespace(registry=registry), _interaction_session_id='session')
    decision = Execute(tool,
        arguments=(ScopedArgument(tool, new_name, literal, start, start+len(literal)),),
        references=('g0',), delivery=('explanation',), retain_parameters=(RetainedParameter('g0', name),))
    admitted = admit(sessions, interaction, decision, {})
    interaction = replace(interaction, admitted=admitted)
    assert admitted['inputs'] == {name:value, new_input:new_value}
    assert admitted['arguments'][0]['argument'] == new_name
    assert admitted['retained_parameters'] == [dict(result='g0', tool=tool, argument=name,
                                                    value=value, origin='owner_effective_value')]
    _validate_argument_record(interaction)
    _validate_research_intent(interaction)
    plan = AgentPlan('plan', admitted['request_id'], 'Explicit continuation.',
        (PlanStep('operation', tool, {name:value, new_name:new_value}),))
    _check_parameter_plan(registry, plan, admitted, ())
    repeated = AgentPlan('plan', admitted['request_id'], 'Ambiguous consumer.', plan.steps + (
        PlanStep('another', tool, {name:value, new_name:new_value}),))
    with pytest.raises(IntentError, match='ambiguous_parameter'):
        _check_parameter_plan(registry, repeated, admitted, ())
    interaction = replace(interaction, admitted=dict(admitted,
        retained_parameters=[dict(admitted['retained_parameters'][0], value=999)]))
    with pytest.raises(ValueError, match='intention changed'):
        _validate_research_intent(interaction)


@pytest.mark.parametrize('change', [dict(references=['g9']), dict(retain_parameters=[dict(result='g9', argument='random_seed')])])
def test_unoffered_read_references_cannot_bind_a_task(change):
    from types import SimpleNamespace
    from agent.application.dialogue_execution import _research_intent
    decision = parse_decision(wire(dict(kind='execute_plan', target='compute_cell_umap', **change)))
    with pytest.raises(IntentError):
        _research_intent(SimpleNamespace(snapshot=dict(dialogue=dict(results=[]))), decision)


def test_same_input_continuation_retains_exact_scientific_declarations_without_repetition():
    from dataclasses import replace
    from types import SimpleNamespace
    from agent.application.dialogue_execution import admit, _validate_argument_record, _validate_research_intent
    from agent.application.session_state import Interaction
    from agent.application.turn_decisions import RetainedParameter, ScopedArgument
    from agent.orchestration import build_default_tool_registry

    registry = build_default_tool_registry()
    species = dict(scientific_name='Danio rerio', taxonomy_id=7955)
    known = dict(species=species, assembly='danRer11', matrix_semantics='binary_accessibility')
    row = dict(handle='g0', tool='epizoo_embed_cells', parameters=[
        dict(argument=name, value=value, origin='accepted_argument') for name, value in known.items()])
    utterance = 'Use the trained model on the same input with 17 neighbors.'
    interaction = Interaction('continue', utterance, 'captured', 0, dict(dialogue=dict(results=[row])))
    sessions = SimpleNamespace(_application=SimpleNamespace(registry=registry), _interaction_session_id='session')
    start = utterance.index('17')
    decision = Execute('epizoo_embed_cells',
        arguments=(ScopedArgument('build_cell_neighbors', 'n_neighbors', '17', start, start+2),),
        references=('g0',), retain_parameters=tuple(RetainedParameter('g0', name) for name in known))
    admitted = admit(sessions, interaction, decision, {})
    assert admitted['inputs'] == dict(known, n_neighbors=17)
    captured = replace(interaction, admitted=admitted)
    _validate_argument_record(captured)
    _validate_research_intent(captured)
    changed = [dict(row) for row in admitted['retained_parameters']]
    changed[0]['value'] = dict(scientific_name='Mus musculus', taxonomy_id=10090)
    with pytest.raises(ValueError, match='intention changed'):
        _validate_research_intent(replace(captured, admitted=dict(admitted, retained_parameters=changed)))
    with pytest.raises(IntentError, match='unavailable_context'):
        admit(sessions, interaction, replace(decision,
            retain_parameters=(RetainedParameter('g0', 'unknown_measurement'),)), {})


def test_prerequisite_task_rebinds_group_handles_by_exact_captured_source():
    from types import SimpleNamespace
    from agent.application.dialogue_execution import planning_task
    from agent.application.session_state import OutputLocator
    a = OutputLocator('old', 'run-a', 'step-a', 'analysis_path', 'a'*64)
    b = OutputLocator('other', 'run-b', 'step-b', 'analysis_path', 'b'*64)
    revisions = (SimpleNamespace(revision_id='old-revision', outputs=(a,)),
                 SimpleNamespace(revision_id='other-revision', outputs=(b,)))
    old = dict(handle='r0', revision_id='old-revision', output_name='old',
               accepted_step_sha256='a'*64, tool='compute_cell_umap')
    other = dict(handle='r0', revision_id='other-revision', output_name='other',
                 accepted_step_sha256='b'*64, tool='build_cell_neighbors')
    origin = SimpleNamespace(turn_id='original', admitted=dict(delivery=['comparison'], references=['g0']),
        snapshot=dict(dialogue=dict(outputs=[old])))
    child = SimpleNamespace(turn_id='answer', admitted=dict(parameter_continuation=dict(origin_turn_id='original')),
        snapshot=dict(dialogue=dict(outputs=[other, dict(old, handle='r1')])))
    state = SimpleNamespace(revisions=revisions, interactions=(origin, child))
    assert planning_task(child, state) == dict(delivery=['comparison'], references=['g1'])


def test_separable_unfulfilled_requirement_is_bound_planned_and_disclosed_without_science():
    from dataclasses import replace
    from types import SimpleNamespace
    from agent.application.dialogue_execution import admit, planning_task, completion_outcome
    from agent.orchestration import AgentRuntime, LLMPlanner, ToolRegistry, build_default_tool_registry
    from agent.orchestration.active_context import ActivePlanningContext, planning_context
    from agent.schemas import AgentRequest, RunMode, RunStatus
    omitted = 'reconstruct ancestral cell lineages'
    utterance = f'Inspect the matrix and explain it. Optionally {omitted}; skip that independent part if unavailable.'
    def forbidden(**kwargs):
        pytest.fail('Planning executed native science')
    registry = ToolRegistry((replace(build_default_tool_registry().get('inspect_scATAC'), function=forbidden),))
    decision = parse_decision(wire(dict(kind='execute_plan', target='inspect_scATAC',
        delivery=['explanation'], unfulfilled=[omitted])))
    interaction = SimpleNamespace(turn_id='original', utterance=utterance, snapshot={}, submission=None,
                                  base_revision_id=None)
    sessions = SimpleNamespace(_application=SimpleNamespace(registry=registry), _interaction_session_id='session')
    interaction.admitted = admit(sessions, interaction, decision, dict(input_path='/private/input.h5ad'))
    task = planning_task(interaction, SimpleNamespace(interactions=(interaction,)))
    class Model:
        model_id = 'scripted-compound-scope'
        calls = 0
        def complete(self, *, prompt, response_schema):
            self.calls += 1
            payload = json.loads(prompt)
            assert payload['research_context']['task']['unfulfilled'] == [omitted]
            assert payload['research_context']['task']['delivery_status'] == 'deferred'
            assert payload['user_request'] == utterance
            return json.dumps(dict(schema_version=4, decision=dict(kind='plan', steps=[
                dict(step_id='inspect', tool='inspect_scATAC', sources=[], control_dependencies=[])])))
    model = Model()
    with planning_context(ActivePlanningContext('session', None, 0, (), task=task)):
        result = AgentRuntime(planner=LLMPlanner(model), registry=registry).run(
            AgentRequest('request', utterance, interaction.admitted['inputs'], RunMode.PLAN_ONLY))
    assert result.status is RunStatus.PLANNED and model.calls == 1
    state = SimpleNamespace(interactions=(interaction,), active_revision_id='accepted',
        turn=lambda _:SimpleNamespace(status='activated', revision_id='accepted'))
    text = completion_outcome(state, 'original').text
    assert omitted in text and 'not fulfilled' in text
    with pytest.raises(IntentError, match='ungrounded_operand'):
        admit(sessions, interaction, replace(decision, unfulfilled=('invented requirement',)), {})


def test_planning_result_projection_includes_recorded_dependencies_without_unrelated_history():
    from agent.orchestration.active_context import ActivePlanningContext
    results = (dict(handle='g0', relations=['current'], derived_from=['g1']),
               dict(handle='g1', relations=['parent'], derived_from=[]),
               dict(handle='g2', relations=['previous_active'], derived_from=[]))
    context = ActivePlanningContext('session', 'captured', 2, (), results)
    payload = context.public_research_context()
    assert [r['handle'] for r in payload['results']] == ['g0', 'g1']
    assert payload['results'][0]['reusable_inputs'] == []
    with pytest.raises(ValueError, match='unoffered'):
        context.binding('g1')
