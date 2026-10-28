"""Existing statistical tools admit explicit research intent without a manual spec."""
from dataclasses import asdict, replace
import json

import pytest

from agent.application.dialogue_execution import (
    _bind_arguments, _check_parameter_plan, scientific_parameters,
)
from agent.application.turn_decisions import (
    IntentError, ScopedArgument, ScalarChoiceArgument, _compact_decision_schema,
    _interpretation_public, decision_schema, parse_decision,
)
from agent.orchestration import build_default_tool_registry
from agent.schemas import AgentPlan, PlanStep


def declared_workflow():
    feature = 'validate_scATAC_feature_space'
    bulk = 'build_replicate_pseudobulk'
    da = 'run_replicate_differential_accessibility'
    values = {
        feature: dict(matrix_source='X', matrix_semantics='fragment_counts',
                      species='mouse', genome_assembly='mm10', coordinate_source='none'),
        bulk: dict(replicate_key='donor', group_key='celltype', condition_key='condition', group_source='raw_obs'),
        da: dict(group_value='Microglia', condition_key='condition', numerator_condition='injured',
                 denominator_condition='control', design_type='independent'),
    }
    registry = build_default_tool_registry()
    utterance = ' '.join(f'{tool}.{key}="{value}".' for tool, args in values.items() for key, value in args.items())
    declarations = []
    offset = 0
    for tool, args in values.items():
        for key, value in args.items():
            start = utterance.index('"' + value + '"', offset) + 1
            offset = start + len(value)
            spec = registry.get(tool).required_arguments[key]
            cls = ScalarChoiceArgument if spec.planning.conversational_choice else ScopedArgument
            declarations.append(cls(tool, key, value, start, offset,
                                    **({'value': value} if cls is ScalarChoiceArgument else {})))
    return registry, utterance, tuple(declarations), values


def test_composed_statistical_intent_binds_and_checks_each_explicit_consumer():
    registry, utterance, declarations, expected = declared_workflow()
    assert len(declarations) > 8
    offered = {(p['tool'], p['argument']) for p in scientific_parameters(registry)}
    assert all((d.tool, d.argument) in offered for d in declarations)
    wire = dict(turn_schema_version=1, decision=dict(kind='execute_plan',
        target='run_replicate_differential_accessibility', arguments=[asdict(d) for d in declarations]))
    parsed = parse_decision(json.dumps(wire))
    inputs = _bind_arguments(registry, utterance, {}, parsed.arguments)
    admitted = dict(inputs=inputs, arguments=[asdict(d) for d in declarations])
    plan = AgentPlan('plan', 'request', 'Explicit scientific declarations.', tuple(
        PlanStep(f's{i}', tool, args) for i, (tool, args) in enumerate(expected.items())))
    _check_parameter_plan(registry, plan, admitted, ())
    changed = dict(expected['run_replicate_differential_accessibility'], condition_key='other')
    with pytest.raises(IntentError):
        _check_parameter_plan(registry, replace(plan, steps=plan.steps[:-1] + (
            replace(plan.steps[-1], arguments=changed),)), admitted, ())


def test_shared_input_never_authorizes_an_undeclared_consumer():
    registry, utterance, declarations, expected = declared_workflow()
    declarations = tuple(d for d in declarations if not (
        d.tool == 'run_replicate_differential_accessibility' and d.argument == 'condition_key'))
    inputs = _bind_arguments(registry, utterance, {}, declarations)
    plan = AgentPlan('plan', 'request', 'Do not fan out.', tuple(
        PlanStep(f's{i}', tool, args) for i, (tool, args) in enumerate(expected.items())))
    with pytest.raises(IntentError, match='ambiguous_parameter'):
        _check_parameter_plan(registry, plan, dict(inputs=inputs,
            arguments=[asdict(d) for d in declarations]), ())


def test_conflicting_shared_declarations_fail_even_when_replacing_pending_values():
    registry = build_default_tool_registry()
    utterance = 'Use "condition" and "other".'
    declarations = tuple(ScopedArgument(tool, 'condition_key', value,
        utterance.index(value), utterance.index(value) + len(value))
        for tool, value in (('build_replicate_pseudobulk', 'condition'),
                            ('run_replicate_differential_accessibility', 'other')))
    with pytest.raises(IntentError, match='conflicting_scientific_parameter'):
        _bind_arguments(registry, utterance, {}, declarations, replace_existing=True)


def test_nullable_registered_names_keep_the_exact_user_string():
    registry = build_default_tool_registry()
    text = 'Use layer "raw_counts".'
    declaration = ScopedArgument('validate_scATAC_feature_space', 'layer_key', 'raw_counts', 11, 21)
    assert _bind_arguments(registry, text, {}, (declaration,))['layer_key'] == 'raw_counts'


def expand(schema):
    def visit(node):
        if isinstance(node, dict):
            if '$ref' in node:
                assert set(node) == {'$ref'}
                return visit(schema['$defs'][node['$ref'].removeprefix('#/$defs/')])
            return {k: visit(v) for k, v in node.items() if k != '$defs'}
        return [visit(v) for v in node] if isinstance(node, list) else node
    return visit(schema)


def test_provider_schema_compaction_is_lossless_and_keeps_source_associations():
    registry = build_default_tool_registry()
    public = dict(bases={}, relations=['current'], dialogue=dict(outputs=[], tools=list(registry.names()),
        scientific_parameters=scientific_parameters(registry), resource_source=True,
        selected_sources=[dict(source='source.0', input_type='h5ad'), dict(source='source.1', input_type='h5ad')],
        source_fields=[dict(source='source.0', fields=['cell_type']), dict(source='source.1', fields=['truth'])],
        input_roles=[dict(tool='evaluate_cell_clustering', port='ground_truth_dataset')]))
    original = decision_schema(public)
    compact = _compact_decision_schema(original)
    assert expand(compact) == original
    assert len(json.dumps(compact)) < len(json.dumps(original)) * .7
    assert compact == _compact_decision_schema(original)


def test_cooutput_compaction_preserves_all_handles_and_differing_evidence():
    fields = dict(available_fields=['n_cells'], fields_omitted=False, subjects=dict(candidates=[], complete=False))
    rows = [dict(handle='r0', **fields), dict(handle='r1', **fields),
            dict(handle='r2', **(fields | {'available_fields': ['different']})), dict(handle='r3', **fields)]
    public = dict(dialogue=dict(outputs=rows, result_groups=[dict(canonical_output='r0', outputs=['r0', 'r1', 'r2'])]))
    frozen = json.dumps(public)
    projected = _interpretation_public(public)['dialogue']['outputs']
    assert [r['handle'] for r in projected] == ['r0', 'r1', 'r2', 'r3']
    assert projected[1] == dict(handle='r1', evidence_from='r0')
    assert projected[2] == rows[2] and projected[3] == rows[3]
    assert json.dumps(public) == frozen


def test_composed_declarations_survive_persisted_record_validation():
    from types import SimpleNamespace
    from agent.application.dialogue_execution import _validate_argument_record
    from agent.application.session_state import digest
    registry, utterance, declarations, _ = declared_workflow()
    values = _bind_arguments(registry, utterance, {}, declarations)
    record = SimpleNamespace(utterance=utterance, submission=None, admitted=dict(
        kind='execute', operation='plan', inputs=values,
        arguments=[asdict(d) for d in declarations],
        arguments_received_inputs_sha256=digest({})))
    _validate_argument_record(record)
    record.admitted['inputs']['numerator_condition'] = 'different'
    with pytest.raises(ValueError):
        _validate_argument_record(record)


def test_statistical_source_and_annotation_ground_truth_are_offered_roles():
    from agent.application.local_resources import _group_port
    registry = build_default_tool_registry()
    for tool, name in (('validate_scATAC_feature_space', 'input_path'),
                       ('evaluate_cell_annotation', 'ground_truth_dataset')):
        port, source, argument = _group_port(registry, tool, name)
        assert port.name == name and len(source.members) == 1
        assert argument.planning.accepted_artifact_kinds


def test_group_binding_retains_composed_scalar_intent_without_new_file_authority():
    from agent.application.local_resources import RegisteredInputGroup, group_role_inputs, ResourceAdmissionError
    registry, utterance, declarations, _ = declared_workflow()
    values = _bind_arguments(registry, utterance, {}, declarations)
    member = dict(resource_id='local-' + 'a'*64, record_sha256='b'*64,
        source_path='/registered/counts.h5ad', source_sha256='c'*64, input_type='h5ad', label='Counts')
    roles = [dict(tool='validate_scATAC_feature_space', port='input_path', source='source.0')]
    binding = RegisteredInputGroup([member], values, attachment_candidates=True)
    assert group_role_inputs(binding, roles, registry) == values | {'input_path': member['source_path']}
    independent_species = values | {'species': 'Danio rerio'}
    assert group_role_inputs(RegisteredInputGroup([member], independent_species,
        attachment_candidates=True), roles, registry)['species'] == 'Danio rerio'
    with pytest.raises(ResourceAdmissionError):
        group_role_inputs(RegisteredInputGroup([member], values | {'hidden_path':'/other.h5ad'},
            attachment_candidates=True), roles, registry)
