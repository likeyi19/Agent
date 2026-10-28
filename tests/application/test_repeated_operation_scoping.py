"""Explicit dataset identities scope repeated operations without step-name policy."""
from dataclasses import asdict, replace
import json
from types import SimpleNamespace

import pytest
from agent.application.dialogue_execution import _bind_arguments, _check_parameter_plan, _source_field_roles
from agent.application.local_resources import (RegisteredInputGroup, group_role_inputs,
    LocalResourceAdmission, ResourceAdmissionError)
from agent.application.turn_decisions import (InputRole, IntentError, ScalarChoiceArgument,
    SourceScopedChoiceArgument, _scientific_arguments, _validate_arguments, parse_decision)
from agent.orchestration import build_default_tool_registry
from agent.schemas import AgentPlan, PlanStep


def example():
    registry = build_default_tool_registry()
    declarations = tuple(SourceScopedChoiceArgument('epizoo_embed_cells', 'species', 'mouse',
        0, 5, 'mouse', 'dataset', f'source.{i}') for i in range(2))
    roles = _source_field_roles(registry, (
        InputRole('transfer_cell_labels', 'reference_dataset', 'source.0'),
        InputRole('transfer_cell_labels', 'query_dataset', 'source.1')), declarations)
    members = [dict(resource_id='local-'+str(i+1)*64, record_sha256='a'*64,
        source_path=f'/registered/{i}.h5ad', source_sha256=str(i+1)*64,
        input_type='h5ad', label=f'File {i}') for i in range(2)]
    binding = RegisteredInputGroup(members, {}, attachment_candidates=True)
    values = _bind_arguments(registry, 'mouse', group_role_inputs(binding,
        [asdict(r) for r in roles], registry), declarations)
    binding = RegisteredInputGroup(members, values, roles=[asdict(r) for r in roles], attachment_candidates=True)
    plan = AgentPlan('plan', 'request', 'Explicit reference and query.', (
        PlanStep('unrelated_b', 'epizoo_embed_cells', dict(input_path=members[0]['source_path'], species='mouse')),
        PlanStep('unrelated_a', 'epizoo_embed_cells', dict(input_path=members[1]['source_path'], species='mouse')),
        PlanStep('result', 'transfer_cell_labels', dict(reference_h5ad_path=members[0]['source_path'],
            query_h5ad_path=members[1]['source_path']))))
    admitted = dict(inputs=values, arguments=[asdict(d) for d in declarations],
        roles=[asdict(r) for r in roles], registered_input=binding.attribution())
    owner = object.__new__(LocalResourceAdmission)
    owner._registry = registry
    return registry, declarations, binding, plan, admitted, owner


def test_explicit_repeated_sources_replay_and_match_without_step_order():
    registry, declarations, binding, plan, admitted, owner = example()
    _validate_arguments(declarations)
    assert _scientific_arguments([asdict(d) for d in declarations]) == declarations
    assert group_role_inputs(binding, binding.roles, registry) == dict(binding.execution_inputs)
    for steps in (plan.steps, tuple(reversed(plan.steps))):
        _check_parameter_plan(registry, replace(plan, steps=steps), admitted, ())
        owner._validate_group_consumption(binding, steps)
    wire = dict(turn_schema_version=1, decision=dict(kind='execute_plan', target='transfer_cell_labels',
        arguments=[dict(tool=d.tool, argument=d.argument, literal='mouse', evidence='mouse',
            value='mouse', port=d.port, source=d.source) for d in declarations]))
    assert parse_decision(json.dumps(wire), utterance='mouse').arguments == declarations


@pytest.mark.parametrize('mutation', ['unscoped', 'missing_scope', 'duplicate_source', 'redirect', 'extra', 'wrong_value', 'missing_role'])
def test_repeated_parameter_scope_rejects_ambiguity_and_substitution(mutation):
    registry, declarations, binding, plan, admitted, owner = example()
    if mutation == 'unscoped':
        admitted['arguments'] = [asdict(ScalarChoiceArgument('epizoo_embed_cells','species','mouse',0,5,'mouse'))]
    elif mutation == 'missing_scope':
        admitted['arguments'] = admitted['arguments'][:1]
    elif mutation == 'missing_role':
        admitted['roles'] = admitted['roles'][:2]
    else:
        steps = list(plan.steps)
        if mutation == 'duplicate_source':
            steps[1] = replace(steps[1], arguments=dict(steps[0].arguments))
        elif mutation == 'redirect':
            steps[1] = replace(steps[1], arguments=dict(input_path='/unregistered.h5ad', species='mouse'))
        elif mutation == 'extra':
            steps.append(replace(steps[0], step_id='extra', arguments=dict(input_path='/other.h5ad', species='mouse')))
        else:
            steps[1] = replace(steps[1], arguments=dict(steps[1].arguments, species='human'))
        plan = replace(plan, steps=tuple(steps))
    with pytest.raises(IntentError):
        _check_parameter_plan(registry, plan, admitted, ())


def test_conflicting_shared_parameter_values_do_not_silently_override_sources():
    registry, declarations, *_ = example()
    declarations = (declarations[0], replace(declarations[1], literal='human', start=6, end=11, value='human'))
    with pytest.raises(IntentError, match='conflicting_scientific_parameter'):
        _bind_arguments(registry, 'mouse human', {}, declarations, replace_existing=True)


@pytest.mark.parametrize('mutation', ['swap', 'substitute', 'unused', 'duplicate'])
def test_repeated_group_roles_keep_reference_query_and_exact_source_consumption(mutation):
    _, _, binding, plan, _, owner = example()
    steps = list(plan.steps)
    if mutation == 'swap':
        args = dict(steps[-1].arguments)
        args['reference_h5ad_path'], args['query_h5ad_path'] = args['query_h5ad_path'], args['reference_h5ad_path']
        steps[-1] = replace(steps[-1], arguments=args)
    elif mutation == 'substitute':
        steps[0] = replace(steps[0], arguments=dict(input_path='/not-selected.h5ad', species='mouse'))
    elif mutation == 'unused':
        steps.pop(0)
    else:
        steps.append(replace(steps[0], step_id='duplicate'))
    with pytest.raises(ResourceAdmissionError):
        owner._validate_group_consumption(binding, tuple(steps))


def test_scoped_records_reject_invalid_historical_shapes_and_duplicate_identity():
    _, declarations, *_ = example()
    with pytest.raises(ValueError):
        _validate_arguments((declarations[0], declarations[0]))
    invalid = asdict(declarations[0]) | {'source': 'source.99'}
    with pytest.raises(ValueError):
        _scientific_arguments([invalid])
    with pytest.raises(IntentError):
        _scientific_arguments([asdict(declarations[0]) | {'step_id': 'first'}])


def test_current_initialization_omits_training_choices_but_historical_declarations_remain_readable():
    from agent.application.dialogue_execution import scientific_parameters, _parameter_input
    from agent.application.epizoo_handoff import schema, EpiZooDeclaration
    registry = build_default_tool_registry()
    parameters = scientific_parameters(registry)
    for name in ('max_steps', 'learning_rate', 'training_batch_size'):
        assert not any(p['tool'] == 'adapt_epizoo_species' and p['argument'] == name for p in parameters)
        assert any(p['tool'] == 'posttrain_epizoo_model' and p['argument'] == name for p in parameters)
        assert _parameter_input(registry, 'adapt_epizoo_species', name)[0] == name
    assert 'posttraining' not in schema([], [])['properties']['operation']['anyOf'][0]['enum']
    assert EpiZooDeclaration(operation='posttraining').operation == 'posttraining'


def test_accepted_parameter_evidence_uses_the_same_source_scope_owner():
    from agent.application.dialogue_evidence import _explicit_user_parameter_facts, _Unavailable
    registry, declarations, binding, plan, admitted, _ = example()
    admitted.update(kind='execute', operation='plan', request_id='request')
    interaction = SimpleNamespace(turn_id='turn', admitted=admitted, utterance='mouse',
        snapshot={}, base_generation=0, base_revision_id=None)
    state = SimpleNamespace(turns=[SimpleNamespace(run_id='run', turn_id='turn')], interactions=[interaction])
    run = SimpleNamespace(run_id='run', request_id='request', plan=plan)
    sessions = SimpleNamespace(_application=SimpleNamespace(registry=registry))
    for planned in plan.steps[:2]:
        step = SimpleNamespace(tool_name=planned.tool_name, step_id=planned.step_id,
            resolved_arguments=planned.arguments)
        facts = _explicit_user_parameter_facts(sessions, state, run, step, planned)
        assert facts[0].value['parameters'] == {'species': 'explicit_user_instruction'}
        bad = replace(planned, arguments=dict(planned.arguments, input_path='/other.h5ad'))
        run.plan = replace(plan, steps=tuple(bad if p.step_id == planned.step_id else p for p in plan.steps))
        with pytest.raises(_Unavailable):
            _explicit_user_parameter_facts(sessions, state, run, step, bad)
        run.plan = plan
