from dataclasses import asdict, replace
import json
from fractions import Fraction
import pytest
from agent.application.turn_decisions import *
from agent.orchestration import build_default_tool_registry


def test_strict_decision_parser_and_bounded_schema():
    assert parse_decision(json.dumps({'turn_schema_version':1,'decision':{'kind':'navigate','relation':'parent'}})) == Navigate('parent')
    for raw in ('{}','{"kind":"navigate","relation":"parent","revision_id":"invented"}',
                '{"kind":"clarify","reason":"oops"}', '{"kind":"navigate","kind":"clarify","relation":"parent"}'):
        with pytest.raises(IntentError): parse_decision('{"turn_schema_version":1,"decision":'+raw+'}')


def pending_public():
    return dict(bases={}, relations=[], dialogue=dict(outputs=[], tools=['epizoo_embed_cells'],
        pending_prerequisite=dict(handle='@species', field='species', choices=['human', 'mouse'],
            objective='Run EpiZoo embeddings, Leiden clustering and UMAP.',
            dataset='Selected registered H5AD')))


@pytest.mark.parametrize('species', ['human', 'mouse'])
def test_pending_species_answer_is_a_narrow_typed_decision(species):
    raw = json.dumps(dict(turn_schema_version=1,
        decision=dict(kind='answer_prerequisite', pending='@species', species=species)))
    assert parse_decision(raw) == SpeciesAnswer('@species', species)


@pytest.mark.parametrize('decision', [
    dict(kind='answer_prerequisite', pending='@species', species='Mouse'),
    dict(kind='answer_prerequisite', pending='@species', species='macaque'),
    dict(kind='answer_prerequisite', pending='@species', species=True),
    dict(kind='answer_prerequisite', pending='', species='mouse'),
    dict(kind='answer_prerequisite', pending='x' * 129, species='mouse'),
    dict(kind='answer_prerequisite', pending='@species', species='mouse', input_path='/tmp/other.h5ad'),
    dict(kind='answer_prerequisite', species='mouse'),
])
def test_pending_species_parser_rejects_unsupported_values_and_arbitrary_inputs(decision):
    with pytest.raises(IntentError, match='invalid_decision'):
        parse_decision(json.dumps(dict(turn_schema_version=1, decision=decision)))




@pytest.mark.parametrize('utterance,species', [
    ('Mouse.', 'mouse'), ("It's human.", 'human'), ('The dataset is from Mus musculus.', 'mouse'),
])
def test_species_meaning_comes_from_provider_in_captured_pending_context(utterance, species):
    class Model:
        def complete(self, *, prompt, response_schema):
            captured = json.loads(prompt)
            assert captured['utterance'] == utterance
            assert captured['dialogue']['pending_prerequisite'] == pending_public()['dialogue']['pending_prerequisite']
            assert any('same request and selected dataset' in i for i in captured['instructions'])
            assert any('never infer species' in i for i in captured['instructions'])
            assert any('never additional operations' in i for i in captured['instructions'])
            return json.dumps(dict(turn_schema_version=1,
                decision=dict(kind='answer_prerequisite', pending='@species', species=species)))
    assert interpret(Model(), utterance, pending_public()) == SpeciesAnswer('@species', species)


@pytest.mark.parametrize('reason', ['missing_species', 'ambiguous_species', 'unsupported_species',
                                    'conflicting_species', 'invalid_prerequisite'])
def test_species_clarifications_use_existing_clarify_contract(reason):
    assert parse_decision(json.dumps(dict(turn_schema_version=1,
        decision=dict(kind='clarify', reason=reason)))) == Clarify(reason)


def leiden_declaration(utterance, literal='0.7', *, occurrence=0):
    start = utterance.index(literal, occurrence)
    return ScopedArgument('cluster_cells', 'resolution', literal, start, start + len(literal))


def decision_wire(decision):
    return json.dumps(dict(turn_schema_version=1, decision=decision))


@pytest.mark.parametrize('kind', ['execute_plan', 'answer_prerequisite'])
def test_explicit_resolution_uses_shared_typed_arguments(kind):
    utterance = 'Cluster the selected cells using Leiden at resolution 0.7.'
    declaration = leiden_declaration(utterance)
    value = dict(kind=kind, arguments=[asdict(declaration)])
    if kind == 'execute_plan':
        value['target'] = 'cluster_cells'
        expected = Execute('cluster_cells', arguments=(declaration,))
    else:
        value.update(pending='@species', species='mouse')
        expected = SpeciesAnswer('@species', 'mouse', arguments=(declaration,))
    assert parse_decision(decision_wire(value)) == expected


@pytest.mark.parametrize('value,expected', [
    (dict(kind='execute_plan', target='cluster_cells'), Execute('cluster_cells')),
    (dict(kind='answer_prerequisite', pending='@species', species='mouse'), SpeciesAnswer('@species', 'mouse')),
])
def test_omitted_arguments_keep_their_existing_meaning(value, expected):
    assert parse_decision(decision_wire(value)) == expected


@pytest.mark.parametrize('argument', [None, {}, [], True, 'resolution=0.7',
    dict(tool='cluster_cells', argument='resolution', literal='0.7', start=0, end=3)])
def test_interpreter_rejects_retired_singular_parameter_adapter(argument):
    with pytest.raises(IntentError, match='invalid_decision'):
        parse_decision(decision_wire(dict(kind='execute_plan', target='cluster_cells', argument=argument)))


def test_parameter_schema_retires_the_singular_writer():
    variants = decision_schema()['properties']['decision']['anyOf']
    for variant in variants:
        kind = variant['properties']['kind']['enum'][0]
        assert 'argument' not in variant['properties']


@pytest.mark.parametrize('utterance,literal,expected', [
    ('Run EpiZoo, perform Leiden clustering with resolution 0.7, and generate UMAP.', '0.7', 0.7),
    ('用 Leiden 聚类，分辨率为 .7。', '.7', 0.7),
    ('Perform Leiden clustering at resolution 7e-1.', '7e-1', 0.7),
    ('Cluster at resolution +0.7.', '+0.7', 0.7),
    ('Leiden resolution=1.', '1', 1.0),
    ('Cluster at resolution 0.70.', '0.70', 0.7),
    ('Use the request "Cluster at resolution 0.7."', '0.7', 0.7),
])
def test_exact_decimal_source_span_delegates_value_validation_to_the_owner(utterance, literal, expected):
    argument = build_default_tool_registry().get('cluster_cells').optional_arguments['resolution']
    declaration = leiden_declaration(utterance, literal)
    value = admit_scientific_argument(declaration, utterance, argument)
    assert type(value) is float and value == expected


@pytest.mark.parametrize('utterance,literal', [
    ('Cluster at resolution 10.7.', '0.7'),
    ('Cluster at resolution 0.70.', '0.7'),
    ('Cluster at resolution -0.7.', '0.7'),
    ('Cluster at resolution 0.7e2.', '0.7'),
    ('Cluster at resolution 0.7/2.', '0.7'),
    ('Cluster at resolution 0.7%.', '0.7'),
    ('Cluster at resolution 0.7_0.', '0.7'),
    ('Cluster at resolution 0.7.0.', '0.7'),
])
def test_numeric_source_grounding_rejects_partial_values_units_and_numeric_suffixes(utterance, literal):
    argument = build_default_tool_registry().get('cluster_cells').optional_arguments['resolution']
    with pytest.raises(IntentError, match='ungrounded_operand'):
        admit_scientific_argument(leiden_declaration(utterance, literal), utterance, argument)


@pytest.mark.parametrize('literal', ['0', '-0.7', '1e309', 'NaN', 'Infinity', '0.7%', '7/10', '0_7', '0,7', 'seven'])
def test_explicit_invalid_value_never_becomes_a_default(literal):
    utterance = 'Cluster at resolution ' + literal + '.'
    argument = build_default_tool_registry().get('cluster_cells').optional_arguments['resolution']
    with pytest.raises(IntentError, match='invalid_parameter_value'):
        admit_scientific_argument(leiden_declaration(utterance, literal), utterance, argument)


def test_numeric_source_grounding_rejects_changed_text_and_out_of_utterance_spans():
    argument = build_default_tool_registry().get('cluster_cells').optional_arguments['resolution']
    utterance = 'Use resolution 0.7.'
    declaration = leiden_declaration(utterance)
    for invalid, text in ((replace(declaration, literal='0.8'), utterance),
                          (replace(declaration, end=100), utterance),
                          (declaration, 'Use resolution 0.8.')):
        with pytest.raises(IntentError, match='ungrounded_operand'):
            admit_scientific_argument(invalid, text, argument)


def test_assignment_association_and_execution_intent_belong_to_the_interpreter():
    utterance = '用 Leiden 聚类，分辨率为 0.7。'
    declaration = leiden_declaration(utterance)
    question = 'What does Leiden resolution 0.7 mean?'
    class Model:
        def complete(self, *, prompt, response_schema):
            value = json.loads(prompt)
            assert any('A number alone does not request an operation' in i for i in value['instructions'])
            assert any('default_when_omitted' in i for i in value['instructions'])
            if value['utterance'] == question:
                return decision_wire(dict(kind='answer_guidance',targets=[],candidate=None))
            return decision_wire(dict(kind='execute_plan', target='cluster_cells', arguments=[asdict(declaration)]))
    public = pending_public()
    public['dialogue']['tools'].append('cluster_cells')
    from agent.application.dialogue_execution import scientific_parameters
    public['dialogue']['scientific_parameters'] = scientific_parameters(build_default_tool_registry())
    interpreted = interpret(Model(), utterance, public)
    assert [asdict(d) for d in interpreted.arguments] == [asdict(declaration)]
    assert isinstance(interpret(Model(), question, pending_public()), Answer)


def test_parent_previous_distinction():
    snap={'relations':{'current':'R3','parent':'R1','previous_active':'R2'}}
    assert resolve_relation('parent',snap,'use parent version') == 'R1'
    assert resolve_relation('previous_active',snap,'use the version I was using before') == 'R2'
    with pytest.raises(IntentError): resolve_relation('previous',snap,'use the previous version')
    with pytest.raises(IntentError): resolve_relation('parent',snap,'use the previous version')
    with pytest.raises(IntentError): resolve_relation('parent',snap,'use the earlier version')
    with pytest.raises(IntentError): resolve_relation('previous',snap,'use the earlier version')


def test_additive_session_format_stays_closed_and_empty_compatible():
    from agent.application.session_state import AnalysisSession
    state=AnalysisSession('session')
    assert 'interactions' not in state.to_dict()
    assert AnalysisSession.from_dict(json.loads(json.dumps(state.to_dict()))) == state


def string_public(*, pending=False):
    from agent.application.dialogue_execution import scientific_parameters

    value = dict(bases={}, relations=[], dialogue=dict(outputs=[],
        tools=['evaluate_cell_clustering', 'transfer_cell_labels', ],
        scientific_parameters=scientific_parameters(build_default_tool_registry())))
    if pending:
        value['dialogue']['pending_prerequisite'] = dict(handle='@parameters', field='parameters',
            objective='Evaluate the fixed clustering against the supplied ground-truth dataset.',
            missing=['label_key'])
    return value


@pytest.mark.parametrize('tool,name,role,utterance,literal', [
    ('evaluate_cell_clustering', 'label_key', 'ground_truth',
     'Evaluate the clustering using the ground-truth column "cell_type".', 'cell_type'),
    ('transfer_cell_labels', 'reference_label_key', 'reference',
     'Transfer labels from the reference dataset using the column "major_type".', 'major_type'),

])
def test_interpreter_owns_operation_and_source_role_association_for_exact_strings(tool, name, role, utterance, literal):
    start = utterance.index(literal)
    argument = ScopedArgument(tool, name, literal, start, start + len(literal))

    class Model:
        def complete(self, *, prompt, response_schema):
            captured = json.loads(prompt)
            offered = captured['dialogue']['capabilities'][tool]['parameters'][name]
            assert offered['exact_string'] is True and offered['provenance_role'] == role
            rule, = [item for item in captured['instructions'] if item.startswith('For an offered exact_string')]
            for phrase in ('surrounding quotation marks excluded', 'Preserve case, Unicode',
                           'do not normalize or invent reference/query/ground-truth relationships', 'provenance_role',
                           'ambiguous or conflicting declarations require clarification'):
                assert phrase in rule
            return decision_wire(dict(kind='execute_plan', target=tool, arguments=[asdict(argument)]))

    parsed = interpret(Model(), utterance, string_public())
    assert parsed == Execute(tool, arguments=(argument,))


def test_pending_string_answer_uses_the_same_interpreter_and_keeps_original_scope_constraints():
    utterance = 'Use the exact column “细胞 类型”.'
    literal = '细胞 类型'
    start = utterance.index(literal)
    argument = ScopedArgument('evaluate_cell_clustering', 'label_key', literal, start, start + len(literal))

    class Model:
        def complete(self, *, prompt, response_schema):
            captured = json.loads(prompt)
            assert captured['utterance'] == utterance
            assert captured['dialogue']['pending_prerequisite'] == string_public(pending=True)['dialogue']['pending_prerequisite']
            rule, = [item for item in captured['instructions'] if item.startswith('Choose answer_parameters only')]
            assert 'numeric or exact_string literals' in rule
            assert 'never invent the remaining values or add operations' in rule
            assert any('original objective, inputs and operation scope remain fixed' in item
                for item in captured['instructions'])
            return decision_wire(dict(kind='answer_parameters', pending='@parameters', arguments=[asdict(argument)]))

    assert interpret(Model(), utterance, string_public(pending=True)) == ParameterAnswer('@parameters', (argument,))


def test_ambiguous_column_association_keeps_provider_authored_clarification():
    utterance = 'Transfer labels and evaluate them using either "cell_type" or "manual_type".'

    class Model:
        def complete(self, *, prompt, response_schema):
            assert json.loads(prompt)['utterance'] == utterance
            return decision_wire(dict(kind='clarify', reason='ambiguous_parameter'))

    assert interpret(Model(), utterance, string_public()) == Clarify('ambiguous_parameter')




def test_result_only_instructions_follow_the_existing_schema_offers():
    from copy import deepcopy
    from agent.application.dialogue_execution import scientific_parameters
    registry = build_default_tool_registry()
    public = dict(bases={}, relations=['current'], dialogue=dict(outputs=[],
        tools=list(registry.names()), scientific_parameters=scientific_parameters(registry)))
    prompts = []
    class Capture:
        def complete(self, *, prompt, response_schema):
            prompts.append(json.loads(prompt))
            return decision_wire(dict(kind='clarify', reason='ambiguous_subject'))
    interpret(Capture(), 'Explain the method.', public)
    populated = deepcopy(public)
    populated['dialogue']['result_groups'] = [dict(handle='g0', tool='cluster_cells',
        canonical_output='r0', outputs=['r0'], parameters=[dict(argument='resolution')], categorical={})]
    interpret(Capture(), 'Explain this result.', populated)
    absent, present = [p['instructions'] for p in prompts]
    prefixes = ('For a reusable H5AD', 'dialogue.result_groups groups', 'For execute_plan, references',
                'Use retain_parameters', 'Use answer_scientific.detail', 'Use section=category_counts')
    assert not any(rule.startswith(prefixes) for rule in absent)
    assert all(any(rule.startswith(prefix) for rule in present) for prefix in prefixes)
    assert [rule for rule in present if not rule.startswith(prefixes)] == absent
    populated['dialogue']['result_groups'][0].pop('parameters')
    interpret(Capture(), 'Use the earlier settings.', populated)
    assert any(rule.startswith('Use retain_parameters') and 'unknown historical settings' in rule
               for rule in prompts[-1]['instructions'])
    # Capability and parameter discovery remain complete in both contexts.
    for prompt in prompts:
        assert set(prompt['dialogue']['capabilities']) == set(registry.names())
        for row in scientific_parameters(registry):
            assert prompt['dialogue']['capabilities'][row['tool']]['parameters'][row['argument']] == {k:v for k,v in row.items() if k not in ('tool','argument')}
