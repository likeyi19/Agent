"""Shared typed declarations; semantics and ranges remain Registry-owned."""
from dataclasses import asdict, replace
import json

import pytest

from agent.application.turn_decisions import (
    Execute, IntentError, LeidenResolution, ParameterAnswer, ScopedArgument,
    SpeciesAnswer, admit_leiden_resolution, admit_scientific_argument,
    decision_schema, interpret, parse_decision,
)
from agent.orchestration import build_default_tool_registry
from agent.orchestration.registry import (
    ArtifactSemanticKind, PlanningProvenanceRole, PlanningSourceEligibility,
)


STRING_DECLARATIONS = (
    ('evaluate_cell_clustering', 'label_key', PlanningProvenanceRole.GROUND_TRUTH),
    ('transfer_cell_labels', 'reference_label_key', PlanningProvenanceRole.REFERENCE),
    ('evaluate_cell_annotation', 'ground_truth_label_key', PlanningProvenanceRole.GROUND_TRUTH),
)


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
    spec = registry.get('select_scATAC_cells').optional_arguments[argument]
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
    spec = registry.get('select_scATAC_cells').optional_arguments[argument]
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
            registry.get('select_scATAC_cells').optional_arguments['min_tss_enrichment'])


def test_exact_literal_and_utterance_identity_are_checked(registry):
    utterance = 'Use threshold 4.5.'
    candidate = declaration(utterance, '4.5')
    spec = registry.get('select_scATAC_cells').optional_arguments['min_tss_enrichment']
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
    assert len(accepted.arguments) == 8 and accepted.argument is None
    assert accepted.arguments[0].tool == 'cluster_cells'


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


def test_registry_metadata_offers_only_reviewed_free_text_declarations(registry):
    from agent.application.dialogue_execution import scientific_parameters
    from agent.application.turn_decisions import is_exact_string_argument

    offered = {(item['tool'], item['argument']) for item in scientific_parameters(registry)}
    reviewed = {(tool, name) for tool, name, _ in STRING_DECLARATIONS}
    reviewed.add(('import_scATAC_fragments', 'namespace'))
    reviewed.update((tool_name, name) for tool_name in registry.names()
        for name, spec in {**registry.get(tool_name).required_arguments,
                           **registry.get(tool_name).optional_arguments}.items()
        if spec.planning is not None and spec.planning.conversational_exact_string)
    eligible = set()
    unreviewed = set()
    for tool_name in registry.names():
        tool = registry.get(tool_name)
        for name, spec in {**tool.required_arguments, **tool.optional_arguments}.items():
            if is_exact_string_argument(spec):
                eligible.add((tool_name, name))
            elif str in spec.accepted_types and not spec.choices and not any(
                    value in spec.accepted_types for value in (int, float)):
                unreviewed.add((tool_name, name))
    assert eligible == reviewed
    assert reviewed <= offered
    assert unreviewed and not unreviewed.intersection(offered)


@pytest.mark.parametrize('tool,name,role', STRING_DECLARATIONS)
def test_reviewed_strings_keep_direct_source_and_reference_or_ground_truth_roles(registry, tool, name, role):
    from agent.application.dialogue_execution import _parameter_input, scientific_parameters

    spec = registry.get(tool).required_arguments[name]
    assert spec.accepted_types == (str,) and not spec.choices
    assert spec.planning.scientific_parameter
    assert spec.planning.source_eligibility is PlanningSourceEligibility.REQUEST_INPUT_ONLY
    assert spec.planning.provenance_role is role
    assert not spec.planning.accepted_artifact_kinds and not spec.planning.conversational_choice
    assert _parameter_input(registry, tool, name) == (name, spec)
    item, = [value for value in scientific_parameters(registry)
        if (value['tool'], value['argument']) == (tool, name)]
    assert item['required'] is True and 'choices' not in item


@pytest.mark.parametrize('changes', [
    {'accepted_types': (str, type(None))},
    {'accepted_types': (str, int)},
    {'choices': ('cell_type',)},
    {'planning': None},
])
def test_free_text_eligibility_requires_exact_string_type_without_choices(registry, changes):
    from agent.application.turn_decisions import is_exact_string_argument

    spec = registry.get('evaluate_cell_clustering').required_arguments['label_key']
    assert not is_exact_string_argument(replace(spec, **changes))


@pytest.mark.parametrize('changes', [
    {'scientific_parameter': False},
    {'source_eligibility': PlanningSourceEligibility.REQUEST_INPUT_OR_UPSTREAM_RESULT},
    {'accepted_artifact_kinds': (ArtifactSemanticKind.RAW_SCATAC,)},
    {'provenance_role': None},
    {'provenance_role': PlanningProvenanceRole.QUERY},
    {'provenance_role': PlanningProvenanceRole.BRANCH_PRESERVING},
])
def test_arbitrary_scientific_strings_and_query_operands_do_not_gain_declaration_eligibility(registry, changes):
    from agent.application.turn_decisions import is_exact_string_argument

    spec = registry.get('evaluate_cell_clustering').required_arguments['label_key']
    assert not is_exact_string_argument(replace(spec,
        planning=replace(spec.planning, observation_source=None, **changes)))


@pytest.mark.parametrize('tool,name', [
    ('epizoo_embed_cells', 'checkpoint_path'),
    ('epizoo_embed_cells', 'device'),
    ('evaluate_cell_clustering', 'output_dir'),
    ('validate_scATAC_feature_space', 'genome_assembly'),
    ('build_replicate_pseudobulk', 'condition_key'),
    ('build_scATAC_cell_by_ccre', 'reference_manifest_sha256'),
    ('adopt_scATAC_cell_by_ccre', 'assembly'),
])
def test_paths_devices_design_columns_assemblies_and_digests_stay_excluded(registry, tool, name):
    from agent.application.dialogue_execution import _parameter_input, scientific_parameters
    from agent.application.turn_decisions import is_exact_string_argument

    registered = registry.get(tool)
    spec = registered.required_arguments.get(name, registered.optional_arguments.get(name))
    assert str in spec.accepted_types and not is_exact_string_argument(spec)
    assert not any((item['tool'], item['argument']) == (tool, name)
        for item in scientific_parameters(registry))
    with pytest.raises(IntentError, match='unsupported_intent'):
        _parameter_input(registry, tool, name)


@pytest.mark.parametrize('tool,name,_role', STRING_DECLARATIONS)
@pytest.mark.parametrize('literal', ['cell_type', 'Cell Type', '细胞类型_α', 'Cell.Type-α/β', '123',
                                    '7e-1', 'NaN', ' cell_type ', 'Cell_e\u0301'])
def test_exact_string_declarations_preserve_case_unicode_punctuation_and_spaces(registry, tool, name, _role, literal):
    utterance = '🧬 用所声明的列 "' + literal + '" 完成分析。'
    candidate = declaration(utterance, literal, tool, name)
    spec = registry.get(tool).required_arguments[name]
    value = admit_scientific_argument(candidate, utterance, spec)
    assert type(value) is str and value == literal
    assert candidate.start == len('🧬 用所声明的列 "')
    assert utterance[candidate.start:candidate.end] == literal


@pytest.mark.parametrize('left,right', [('"', '"'), ("'", "'"), ('“', '”'), ('‘', '’'), ('`', '`')])
def test_quoted_string_declaration_selects_only_the_exact_column_name(registry, left, right):
    literal = '细胞类型'
    utterance = 'Evaluate using column ' + left + literal + right + '.'
    candidate = declaration(utterance, literal, 'evaluate_cell_clustering', 'label_key')
    assert admit_scientific_argument(candidate, utterance,
        registry.get(candidate.tool).required_arguments[candidate.argument]) == literal


@pytest.mark.parametrize('literal', ['cell.type', 'cell-type', 'cell/type', 'cell+type', 'Cell_e\u0301'])
def test_unquoted_complete_string_tokens_preserve_exact_column_names(registry, literal):
    utterance = 'Evaluate using column ' + literal + '.'
    candidate = declaration(utterance, literal, 'evaluate_cell_clustering', 'label_key')
    assert admit_scientific_argument(candidate, utterance,
        registry.get(candidate.tool).required_arguments[candidate.argument]) == literal


@pytest.mark.parametrize('token,literal', [
    ('cell.type', 'cell'), ('cell.type', 'type'),
    ('cell-type', 'cell'), ('cell-type', 'type'),
    ('cell/type', 'cell'), ('cell/type', 'type'),
    ('cell+type', 'cell'), ('cell+type', 'type'),
    ('Cell_e\u0301', 'Cell_e'),
])
def test_unquoted_string_spans_reject_punctuation_and_combining_mark_truncation(registry, token, literal):
    utterance = 'Evaluate using column ' + token + '.'
    candidate = declaration(utterance, literal, 'evaluate_cell_clustering', 'label_key')
    with pytest.raises(IntentError, match='ungrounded_operand'):
        admit_scientific_argument(candidate, utterance,
            registry.get(candidate.tool).required_arguments[candidate.argument])


@pytest.mark.parametrize('utterance,literal', [
    ('Evaluate using column "cell_type".', 'cell'),
    ('Evaluate using column "cell_type".', 'type'),
    ('Evaluate using column "cell_type".', '"cell_type"'),
    ("Evaluate using column 'cell_type'.", "'cell_type'"),
    ('Evaluate using column “细胞类型”.', '“细胞类型”'),
])
def test_partial_or_quote_inclusive_string_literals_fail_closed(registry, utterance, literal):
    candidate = declaration(utterance, literal, 'evaluate_cell_clustering', 'label_key')
    with pytest.raises(IntentError):
        admit_scientific_argument(candidate, utterance,
            registry.get(candidate.tool).required_arguments[candidate.argument])


@pytest.mark.parametrize('token,literal', [
    ('Cell Type α', 'Type'),
    ('Cell Type α', 'Type α'),
    ('Cell / Type / α', 'Type'),
    ('Cell / Type / α', '/ Type /'),
])
def test_a_middle_word_or_punctuation_span_cannot_replace_the_complete_quoted_column(registry, token, literal):
    utterance = 'Evaluate using column "' + token + '".'
    candidate = declaration(utterance, literal, 'evaluate_cell_clustering', 'label_key')
    with pytest.raises(IntentError, match='ungrounded_operand'):
        admit_scientific_argument(candidate, utterance,
            registry.get(candidate.tool).required_arguments[candidate.argument])


def test_an_unmatched_other_quote_cannot_hide_truncation_of_the_declared_quoted_column(registry):
    utterance = 'For \'an unmatched quote, use column "Cell Type α".'
    candidate = declaration(utterance, 'Type', 'evaluate_cell_clustering', 'label_key')
    with pytest.raises(IntentError, match='ungrounded_operand'):
        admit_scientific_argument(candidate, utterance,
            registry.get(candidate.tool).required_arguments[candidate.argument])


@pytest.mark.parametrize('utterance,literal', [
    ('Evaluate using column "Cell / Type / α".', 'Cell / Type / α'),
    ('Use the patient\'s column "T cell".', 'T cell'),
    ('Use the dataset\'s column "T\'s cell".', "T's cell"),
    ('Use the request "Evaluate using column \'Cell Type\'."', 'Cell Type'),
])
def test_complete_quoted_columns_with_spaces_punctuation_and_contractions_remain_exact(registry, utterance, literal):
    candidate = declaration(utterance, literal, 'evaluate_cell_clustering', 'label_key')
    assert admit_scientific_argument(candidate, utterance,
        registry.get(candidate.tool).required_arguments[candidate.argument]) == literal


@pytest.mark.parametrize('literal', [' ', '\t', '\n'])
def test_blank_string_declarations_are_never_scientific_defaults(registry, literal):
    utterance = 'Evaluate using column "' + literal + '".'
    with pytest.raises(IntentError, match='invalid_parameter_value'):
        admit_scientific_argument(declaration(utterance, literal, 'evaluate_cell_clustering', 'label_key'),
            utterance, registry.get('evaluate_cell_clustering').required_arguments['label_key'])


def test_exact_string_utterance_identity_rejects_fabricated_and_utf8_byte_offsets(registry):
    utterance = '🧬 用列 "细胞类型" 评价聚类。'
    candidate = declaration(utterance, '细胞类型', 'evaluate_cell_clustering', 'label_key')
    spec = registry.get(candidate.tool).required_arguments[candidate.argument]
    byte_start = len(utterance[:candidate.start].encode('utf-8'))
    for invalid in (replace(candidate, literal='细胞类别'), replace(candidate, end=100),
                    replace(candidate, start=byte_start, end=byte_start + len(candidate.literal))):
        with pytest.raises(IntentError, match='ungrounded_operand'):
            admit_scientific_argument(invalid, utterance, spec)


def test_string_initial_and_pending_wires_reuse_scoped_argument_without_choice_normalization(registry):
    from agent.application.dialogue_execution import scientific_parameters

    utterance = 'Evaluate clustering using "Cell Type".'
    candidate = declaration(utterance, 'Cell Type', 'evaluate_cell_clustering', 'label_key')
    for kind in ('execute_plan', 'answer_parameters'):
        value = dict(kind=kind, arguments=[asdict(candidate)])
        if kind == 'execute_plan':
            value['target'] = candidate.tool
        else:
            value['pending'] = '@parameters'
        parsed = parse_decision(wire(value))
        assert parsed.arguments == (candidate,) and type(parsed.arguments[0]) is ScopedArgument

    value = public(pending=True)
    value['dialogue']['scientific_parameters'] = [item for item in scientific_parameters(registry)
        if (item['tool'], item['argument']) == (candidate.tool, candidate.argument)]
    variants = decision_schema(value)['properties']['decision']['anyOf']
    for kind in ('execute_plan', 'answer_parameters'):
        variant, = [item for item in variants if item['properties']['kind']['enum'] == [kind]]
        offered, = variant['properties']['arguments']['items']['anyOf']
        assert offered['properties']['tool']['enum'] == [candidate.tool]
        assert offered['properties']['argument']['enum'] == [candidate.argument]
        assert 'value' not in offered['properties']


def test_exact_strings_cannot_use_choice_normalization_to_substitute_another_column(registry):
    from agent.application.dialogue_execution import _bind_arguments

    utterance = 'Evaluate using column "Cell Type".'
    candidate = declaration(utterance, 'Cell Type', 'evaluate_cell_clustering', 'label_key')
    parsed = parse_decision(wire(dict(kind='execute_plan', target=candidate.tool,
        arguments=[asdict(candidate) | {'value': 'cell_type'}])))
    with pytest.raises(IntentError, match='invalid_reference'):
        _bind_arguments(registry, utterance, {}, parsed.arguments)


@pytest.mark.parametrize('tool,name,_role', STRING_DECLARATIONS)
def test_pending_string_schema_offers_only_exact_pairs_for_the_captured_operation(registry, tool, name, _role):
    from agent.application.dialogue_execution import scientific_parameters

    value = public(pending=True)
    value['dialogue']['scientific_parameters'] = scientific_parameters(registry)
    value['dialogue']['pending_prerequisite'].update(tool=tool, missing=[name])
    variants = decision_schema(value)['properties']['decision']['anyOf']
    answer, = [item for item in variants if item['properties']['kind']['enum'] == ['answer_parameters']]
    offered = answer['properties']['arguments']['items']['anyOf']
    pairs = {(item['properties']['tool']['enum'][0], item['properties']['argument']['enum'][0])
        for item in offered}
    expected = {(tool, name)}
    if tool == 'transfer_cell_labels':
        expected.update((tool, argument) for argument in ('n_neighbors', 'min_confidence'))
    assert pairs == expected
    assert all('value' not in item['properties'] for item in offered)


@pytest.mark.parametrize('pending_tool', ['inspect_scATAC', 'unregistered_tool'])
def test_pending_tool_without_eligible_parameters_never_emits_empty_argument_schema(registry, pending_tool):
    from agent.application.dialogue_execution import scientific_parameters

    value = public(pending=True)
    value['dialogue']['scientific_parameters'] = scientific_parameters(registry)
    value['dialogue']['pending_prerequisite']['tool'] = pending_tool
    assert all(item['properties']['kind']['enum'] != ['answer_parameters']
        for item in decision_schema(value)['properties']['decision']['anyOf'])


def test_unmatched_same_family_quote_cannot_conceal_a_truncated_column(registry):
    utterance = 'A stray " precedes column "Cell Type α".'
    candidate = declaration(utterance, 'Type', 'evaluate_cell_clustering', 'label_key')
    with pytest.raises(IntentError, match='ungrounded_operand'):
        admit_scientific_argument(candidate, utterance,
            registry.get(candidate.tool).required_arguments[candidate.argument])
