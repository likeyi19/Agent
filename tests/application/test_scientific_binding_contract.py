"""Shared typed numeric binding; semantics and ranges remain Registry-owned."""
from dataclasses import asdict, replace
import json

import pytest

from agent.application.turn_decisions import (
    Execute, IntentError, LeidenResolution, ParameterAnswer, ScopedArgument,
    SpeciesAnswer, admit_leiden_resolution, admit_scientific_argument,
    decision_schema, interpret, parse_decision,
)
from agent.orchestration import build_default_tool_registry


@pytest.fixture(scope='module')
def registry():
    return build_default_tool_registry()


def declaration(utterance, literal, tool='select_scATAC_cells', argument='min_tss_enrichment'):
    start = utterance.index(literal)
    return ScopedArgument(tool, argument, literal, start, start + len(literal))


def wire(value):
    return json.dumps(dict(turn_schema_version=1, decision=value))


def public(*, pending=False):
    value = dict(bases={}, relations=[], dialogue=dict(outputs=[],
        tools=['cluster_cells', 'select_scATAC_cells', 'compute_cell_umap'],
        scientific_parameters=[
            dict(tool='cluster_cells', argument='resolution', description='Registered clustering resolution.', required=False),
            dict(tool='select_scATAC_cells', argument='min_qc_fragment_records', description='Registered depth threshold.', required=True),
            dict(tool='select_scATAC_cells', argument='min_tss_enrichment', description='Registered exact TSS threshold.', required=True),
            dict(tool='compute_cell_umap', argument='min_dist', description='Registered UMAP parameter.', required=False),
        ]))
    if pending:
        value['dialogue']['pending_prerequisite'] = dict(handle='@parameters', field='parameters',
            objective='Select candidate cells from the accepted QC result.',
            missing=['min_qc_fragment_records', 'min_tss_enrichment'])
    return value


def test_initial_request_and_partial_or_complete_answer_share_typed_declarations():
    utterance = 'Use a minimum depth of 1200 and minimum TSS enrichment 4.5.'
    declarations = (declaration(utterance, '1200', argument='min_qc_fragment_records'),
                    declaration(utterance, '4.5'))
    assert parse_decision(wire(dict(kind='execute_plan', target='select_scATAC_cells',
        arguments=[asdict(item) for item in declarations]))) == Execute(
            'current', 'plan', 'select_scATAC_cells', None, arguments=declarations)
    for arguments in (declarations[:1], declarations):
        assert parse_decision(wire(dict(kind='answer_parameters', pending='@parameters',
            arguments=[asdict(item) for item in arguments]))) == ParameterAnswer('@parameters', arguments)


def test_species_identity_stays_distinct_from_registered_parameter_bindings():
    utterance = 'This is human; use resolution 0.7.'
    argument = declaration(utterance, '0.7', 'cluster_cells', 'resolution')
    assert parse_decision(wire(dict(kind='answer_prerequisite', pending='@species',
        species='human', arguments=[asdict(argument)]))) == SpeciesAnswer(
            '@species', 'human', arguments=(argument,))


@pytest.mark.parametrize('literal,argument,expected', [
    ('1200', 'min_qc_fragment_records', 1200),
    ('0', 'min_qc_fragment_records', 0),
    ('4', 'min_tss_enrichment', 4),
    ('4.000', 'min_tss_enrichment', '4.000'),
    ('4/3', 'min_tss_enrichment', '4/3'),
    ('0.123456789012345678', 'min_tss_enrichment', '0.123456789012345678'),
])
def test_registered_type_drives_binding_and_exact_ratio_owner_validates(registry, literal, argument, expected):
    utterance = 'Use threshold ' + literal + '.'
    spec = registry.get('select_scATAC_cells').required_arguments[argument]
    result = admit_scientific_argument(declaration(utterance, literal, argument=argument), utterance, spec)
    assert result == expected and type(result) is type(expected)


@pytest.mark.parametrize('literal,argument', [
    ('-1', 'min_qc_fragment_records'), ('1.5', 'min_qc_fragment_records'),
    ('1e3', 'min_qc_fragment_records'), ('1000000000000000001', 'min_qc_fragment_records'),
    ('-1', 'min_tss_enrichment'), ('-0.1', 'min_tss_enrichment'),
    ('1e3', 'min_tss_enrichment'), ('1/0', 'min_tss_enrichment'),
    ('0.1234567890123456789', 'min_tss_enrichment'), ('NaN', 'min_tss_enrichment'),
])
def test_invalid_explicit_values_never_become_default_or_float_thresholds(registry, literal, argument):
    utterance = 'Use threshold ' + literal + '.'
    spec = registry.get('select_scATAC_cells').required_arguments[argument]
    with pytest.raises(IntentError, match='invalid_parameter_value'):
        admit_scientific_argument(declaration(utterance, literal, argument=argument), utterance, spec)


def test_additional_existing_registered_parameter_needs_no_tool_specific_parser(registry):
    utterance = 'Use UMAP minimum distance .2.'
    argument = declaration(utterance, '.2', 'compute_cell_umap', 'min_dist')
    result = admit_scientific_argument(argument, utterance,
        registry.get('compute_cell_umap').optional_arguments['min_dist'])
    assert result == .2 and type(result) is float


def test_historical_resolution_class_and_admission_stay_strict_and_share_validation(registry):
    utterance = 'Use resolution 1.'
    generic = declaration(utterance, '1', 'cluster_cells', 'resolution')
    historical = LeidenResolution(**asdict(generic))
    spec = registry.get('cluster_cells').optional_arguments['resolution']
    assert admit_scientific_argument(generic, utterance, spec) == admit_leiden_resolution(historical, utterance, spec) == 1.0
    with pytest.raises(IntentError, match='invalid_decision'):
        admit_leiden_resolution(generic, utterance, spec)
    with pytest.raises(ValueError):
        LeidenResolution(**(asdict(generic) | {'tool': 'compute_cell_umap'}))


@pytest.mark.parametrize('utterance,literal', [
    ('Use threshold 14.5.', '4.5'), ('Use threshold -4.5.', '4.5'),
    ('Use threshold 4.5/2.', '4.5'), ('Use threshold 4.50.', '4.5'),
    ('Use threshold 4.5%.', '4.5'), ('Use threshold 4.5e2.', '4.5'),
])
def test_generic_numeric_spans_reject_truncation_and_units(registry, utterance, literal):
    with pytest.raises(IntentError, match='ungrounded_operand'):
        admit_scientific_argument(declaration(utterance, literal), utterance,
            registry.get('select_scATAC_cells').required_arguments['min_tss_enrichment'])


def test_exact_literal_and_utterance_identity_are_checked(registry):
    utterance = 'Use threshold 4.5.'
    candidate = declaration(utterance, '4.5')
    spec = registry.get('select_scATAC_cells').required_arguments['min_tss_enrichment']
    for invalid in (replace(candidate, literal='4.6'), replace(candidate, end=100)):
        with pytest.raises(IntentError, match='ungrounded_operand'):
            admit_scientific_argument(invalid, utterance, spec)


@pytest.mark.parametrize('changes', [
    {'tool': ''}, {'tool': '../cluster_cells'}, {'argument': 'resolution.x'},
    {'literal': ''}, {'literal': '1' * 81}, {'start': True}, {'end': False},
    {'start': -1}, {'start': 5, 'end': 3}, {'end': 4097}, {'unexpected': 1},
])
def test_scoped_argument_shape_is_closed(changes):
    candidate = asdict(declaration('Use depth 1200.', '1200', argument='min_qc_fragment_records')) | changes
    with pytest.raises(IntentError, match='invalid_decision'):
        parse_decision(wire(dict(kind='execute_plan', target='select_scATAC_cells', arguments=[candidate])))


@pytest.mark.parametrize('kind', ['execute_plan', 'answer_parameters', 'answer_prerequisite'])
def test_duplicate_argument_identities_cannot_choose_first_or_last(kind):
    argument = asdict(declaration('Use threshold 4.', '4'))
    value = dict(kind=kind, arguments=[argument, argument])
    if kind == 'execute_plan':
        value['target'] = 'select_scATAC_cells'
    else:
        value['pending'] = '@species' if kind == 'answer_prerequisite' else '@parameters'
        if kind == 'answer_prerequisite':
            value['species'] = 'human'
    with pytest.raises(IntentError, match='invalid_decision'):
        parse_decision(wire(value))


def test_historical_and_generic_identity_cannot_conflict():
    argument = asdict(declaration('Use resolution 0.7.', '0.7', 'cluster_cells', 'resolution'))
    with pytest.raises(IntentError, match='invalid_decision'):
        parse_decision(wire(dict(kind='execute_plan', target='cluster_cells',
            argument=argument, arguments=[argument])))


def test_combined_historical_and_generic_declarations_share_the_eight_argument_bound(registry):
    from agent.application.dialogue_execution import scientific_parameters
    candidates = [item for item in scientific_parameters(registry)
        if (item['tool'], item['argument']) != ('cluster_cells', 'resolution')]
    assert len(candidates) >= 8
    arguments = [dict(tool=item['tool'], argument=item['argument'], literal='1', start=0, end=1)
        for item in candidates[:8]]
    legacy = dict(tool='cluster_cells', argument='resolution', literal='1', start=0, end=1)
    with pytest.raises(IntentError, match='invalid_decision'):
        parse_decision(wire(dict(kind='execute_plan', target='cluster_cells', argument=legacy, arguments=arguments)))
    accepted = parse_decision(wire(dict(kind='execute_plan', target='cluster_cells',
        argument=legacy, arguments=arguments[:7])))
    assert len(accepted.arguments) == 7 and accepted.argument.tool == 'cluster_cells'


@pytest.mark.parametrize('arguments', [None, True, {}, '4', [], [{}] * 9])
def test_parameter_answer_requires_one_to_eight_typed_bindings(arguments):
    with pytest.raises(IntentError, match='invalid_decision'):
        parse_decision(wire(dict(kind='answer_parameters', pending='@parameters', arguments=arguments)))


def test_argument_schema_offers_exact_registered_pairs_without_cross_product():
    variants = decision_schema(public(pending=True))['properties']['decision']['anyOf']
    execute = next(value for value in variants if value['properties']['kind']['enum'] == ['execute_plan'])
    answer = next(value for value in variants if value['properties']['kind']['enum'] == ['answer_parameters'])
    assert answer['properties']['pending']['enum'] == ['@parameters']
    assert answer['properties']['arguments']['minItems'] == 1
    offered = execute['properties']['arguments']
    assert offered['maxItems'] == 8
    pairs = {(value['properties']['tool']['enum'][0], value['properties']['argument']['enum'][0])
        for value in offered['items']['anyOf']}
    assert pairs == {(value['tool'], value['argument']) for value in public()['dialogue']['scientific_parameters']}
    assert ('cluster_cells', 'min_tss_enrichment') not in pairs
    assert all(value['additionalProperties'] is False for value in offered['items']['anyOf'])
    assert all(value['properties']['kind']['enum'] != ['answer_prerequisite'] for value in variants)


def test_parameter_answer_schema_requires_pending_and_offered_registered_contracts():
    assert all(value['properties']['kind']['enum'] != ['answer_parameters']
        for value in decision_schema(public())['properties']['decision']['anyOf'])
    value = public(pending=True)
    value['dialogue']['scientific_parameters'] = []
    assert all(item['properties']['kind']['enum'] != ['answer_parameters']
        for item in decision_schema(value)['properties']['decision']['anyOf'])


def test_semantic_association_and_partial_answer_meaning_belong_to_existing_interpreter():
    utterance = 'Minimum depth should be 1200.'
    candidate = declaration(utterance, '1200', argument='min_qc_fragment_records')

    class Model:
        def complete(self, *, prompt, response_schema):
            captured = json.loads(prompt)
            assert captured['utterance'] == utterance
            assert any('Partial answers are allowed' in item for item in captured['instructions'])
            assert any('original explicit execution request' in item for item in captured['instructions'])
            return wire(dict(kind='answer_parameters', pending='@parameters', arguments=[asdict(candidate)]))

    assert interpret(Model(), utterance, public(pending=True)) == ParameterAnswer('@parameters', (candidate,))
