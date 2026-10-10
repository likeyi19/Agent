from dataclasses import replace
import json
from fractions import Fraction
import pytest
from agent.application.turn_decisions import *
from agent.application.turn_context import parameter_specs
from agent.orchestration import build_default_tool_registry


@pytest.mark.parametrize('utterance,operation,literal,current,expected', [
    ('Set the TSS threshold to 7','set','7',6,7),
    ('Increase the TSS threshold by 1','add','1',6,7),
    ('Decrease it by 1','subtract','1',7,6),
    ('Increase TSS by 0.1','add','0.1','1/3','13/30'),
    ('try min_tss = 7','set','7',6,7),
])
def test_exact_grounded_arithmetic(utterance, operation, literal, current, expected):
    delta=IntentDelta('min_tss_enrichment',operation,literal,utterance)
    result=admit_delta(delta,utterance,{'min_tss_enrichment':current},
        parameter_specs(build_default_tool_registry()),focus='min_tss_enrichment')
    assert result == expected


@pytest.mark.parametrize('utterance,delta', [
    ('Make TSS stricter',IntentDelta('min_tss_enrichment','set','7','Make TSS stricter')),
    ('Set TSS to 7',IntentDelta('min_tss_enrichment','set','8','Set TSS to 7')),
    ('Increase TSS by 1',IntentDelta('min_tss_enrichment','set','1','Increase TSS by 1')),
    ('Decrease TSS by 9',IntentDelta('min_tss_enrichment','subtract','9','Decrease TSS by 9')),
    ('Do not set TSS to 7',IntentDelta('min_tss_enrichment','set','7','set TSS to 7')),
    ('Set threshold to 7',IntentDelta('min_tss_enrichment','set','7','Set threshold to 7')),
    ('Set foo_threshold to 7',IntentDelta('min_tss_enrichment','set','7','Set foo_threshold to 7')),
    ('Decrease it by 1',IntentDelta('min_tss_enrichment','subtract','1','Decrease it by 1')),
    ('Increase TSS by 10%',IntentDelta('min_tss_enrichment','add','10','Increase TSS by 10%')),
])
def test_unadmitted_changes(utterance,delta):
    with pytest.raises(IntentError):
        admit_delta(delta,utterance,{'min_tss_enrichment':6},parameter_specs(build_default_tool_registry()))


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


def test_species_answer_schema_is_offered_only_with_exact_pending_context():
    public = pending_public()
    variants = decision_schema(public)['properties']['decision']['anyOf']
    offered = [v for v in variants if v['properties']['kind']['enum'] == ['answer_prerequisite']]
    assert len(offered) == 1
    assert offered[0]['properties']['pending']['enum'] == ['@species']
    assert offered[0]['properties']['species']['enum'] == ['human', 'mouse']
    assert offered[0]['required'] == ['kind', 'pending', 'species', 'argument']
    assert offered[0]['additionalProperties'] is False
    public['dialogue'].pop('pending_prerequisite')
    assert all(v['properties']['kind']['enum'] != ['answer_prerequisite']
               for v in decision_schema(public)['properties']['decision']['anyOf'])


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
    return LeidenResolution('cluster_cells', 'resolution', literal, start, start + len(literal))


def decision_wire(decision):
    return json.dumps(dict(turn_schema_version=1, decision=decision))


@pytest.mark.parametrize('kind', ['execute_plan', 'answer_prerequisite'])
def test_explicit_leiden_resolution_is_one_closed_typed_argument(kind):
    utterance = 'Cluster the selected cells using Leiden at resolution 0.7.'
    declaration = leiden_declaration(utterance)
    value = dict(kind=kind, argument=asdict(declaration))
    if kind == 'execute_plan':
        value['target'] = 'cluster_cells'
        expected = Execute('current', 'plan', 'cluster_cells', None, declaration)
    else:
        value.update(pending='@species', species='mouse')
        expected = SpeciesAnswer('@species', 'mouse', declaration)
    assert parse_decision(decision_wire(value)) == expected


@pytest.mark.parametrize('value,expected', [
    (dict(kind='execute_plan', target='cluster_cells'), Execute('current', 'plan', 'cluster_cells', None)),
    (dict(kind='execute_plan', target='cluster_cells', argument=None), Execute('current', 'plan', 'cluster_cells', None)),
    (dict(kind='execute_plan', base='current', operation='plan', target='cluster_cells', delta=None),
     Execute('current', 'plan', 'cluster_cells', None)),
    (dict(kind='execute', base='current', operation='plan', target='cluster_cells', delta=None),
     Execute('current', 'plan', 'cluster_cells', None)),
    (dict(kind='answer_prerequisite', pending='@species', species='mouse'), SpeciesAnswer('@species', 'mouse')),
    (dict(kind='answer_prerequisite', pending='@species', species='mouse', argument=None), SpeciesAnswer('@species', 'mouse')),
])
def test_historical_and_omitted_argument_wires_keep_their_existing_meaning(value, expected):
    assert parse_decision(decision_wire(value)) == expected


@pytest.mark.parametrize('changes', [
    {'tool': 'compute_cell_umap'}, {'argument': 'min_dist'}, {'literal': ''},
    {'literal': '7' * 81}, {'start': True}, {'end': False}, {'start': -1},
    {'start': 3, 'end': 2}, {'end': 4097}, {'start': 0.5}, {'end': '3'},
    {'unexpected': 1}, {'literal': None}, {'tool': True},
])
def test_parameter_wire_rejects_unsupported_identity_shape_and_spans(changes):
    argument = asdict(leiden_declaration('Use Leiden 0.7.')) | changes
    with pytest.raises(IntentError, match='invalid_decision'):
        parse_decision(decision_wire(dict(kind='execute_plan', target='cluster_cells', argument=argument)))


@pytest.mark.parametrize('argument', [[], [{'tool': 'cluster_cells'}], True, 'resolution=0.7'])
def test_parameter_wire_never_admits_arbitrary_input_dictionaries_or_multiple_arguments(argument):
    with pytest.raises(IntentError, match='invalid_decision'):
        parse_decision(decision_wire(dict(kind='execute_plan', target='cluster_cells', argument=argument)))


def test_parameter_schema_is_narrow_nullable_and_not_offered_on_read_only_or_qc_decisions():
    variants = decision_schema(pending_public())['properties']['decision']['anyOf']
    for variant in variants:
        kind = variant['properties']['kind']['enum'][0]
        if kind not in ('execute_plan', 'answer_prerequisite'):
            assert 'argument' not in variant['properties']
            continue
        argument, null = variant['properties']['argument']['anyOf']
        assert null == {'type': 'null'}
        assert argument['properties']['tool']['enum'] == ['cluster_cells']
        assert argument['properties']['argument']['enum'] == ['resolution']
        assert argument['required'] == ['tool', 'argument', 'literal', 'start', 'end']
        assert argument['additionalProperties'] is False


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
    value = admit_leiden_resolution(declaration, utterance, argument)
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
        admit_leiden_resolution(leiden_declaration(utterance, literal), utterance, argument)


@pytest.mark.parametrize('literal', ['0', '-0.7', '1e309', 'NaN', 'Infinity', '0.7%', '7/10', '0_7', '0,7', 'seven'])
def test_explicit_invalid_value_never_becomes_a_default(literal):
    utterance = 'Cluster at resolution ' + literal + '.'
    argument = build_default_tool_registry().get('cluster_cells').optional_arguments['resolution']
    with pytest.raises(IntentError, match='invalid_parameter_value'):
        admit_leiden_resolution(leiden_declaration(utterance, literal), utterance, argument)


def test_numeric_source_grounding_rejects_changed_text_and_out_of_utterance_spans():
    argument = build_default_tool_registry().get('cluster_cells').optional_arguments['resolution']
    utterance = 'Use resolution 0.7.'
    declaration = leiden_declaration(utterance)
    for invalid, text in ((replace(declaration, literal='0.8'), utterance),
                          (replace(declaration, end=100), utterance),
                          (declaration, 'Use resolution 0.8.')):
        with pytest.raises(IntentError, match='ungrounded_operand'):
            admit_leiden_resolution(invalid, text, argument)


def test_assignment_association_and_execution_intent_belong_to_the_interpreter():
    utterance = '用 Leiden 聚类，分辨率为 0.7。'
    declaration = leiden_declaration(utterance)
    question = 'What does Leiden resolution 0.7 mean?'
    class Model:
        def complete(self, *, prompt, response_schema):
            value = json.loads(prompt)
            assert any('A number alone does not request an operation' in i for i in value['instructions'])
            assert any('existing default' in i for i in value['instructions'])
            if value['utterance'] == question:
                return decision_wire(dict(kind='answer', intent='unsupported', relation='current', technical=False))
            return decision_wire(dict(kind='execute_plan', target='cluster_cells', argument=asdict(declaration)))
    assert interpret(Model(), utterance, pending_public()).argument == declaration
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


def test_planner_effect_guard_preserves_parameters_sources_and_arbitrary_step_ids():
    from agent.application.turns import _check_plan
    from agent.schemas import AgentPlan, PlanStep, PriorOutputBinding, PriorOutputRef
    from agent.orchestration.planner import PlannerError
    binding=PriorOutputBinding('r','q','compute_scATAC_qc','barcode_qc','scatac_barcode_qc.v1','1'*64,'2'*64,'3'*64)
    parameters={'min_qc_fragment_records':3000,'min_tss_enrichment':7}
    admitted=dict(delta={},parameters=parameters,target='selection',operation={'source':{'name':'chosen'}})
    args=dict(parameters,barcode_qc_manifest_path=PriorOutputRef(binding,'manifest_path'),
        barcode_qc_manifest_sha256=PriorOutputRef(binding,'manifest_sha256'),output_dir='/tmp/test-intent')
    plan=AgentPlan('p','r','fixture',(PlanStep('arbitrary-output-step','select_scATAC_cells',args),))
    registry=build_default_tool_registry()
    assert _check_plan(plan,admitted,parameters,{'qc':binding},registry)[0].step_id=='arbitrary-output-step'
    for changed in (args|{'min_qc_fragment_records':3001},args|{'max_qc_fragment_records':9000},
                    args|{'barcode_qc_manifest_path':PriorOutputRef(replace(binding,run_id='other'),'manifest_path')}):
        with pytest.raises(PlannerError):
            _check_plan(replace(plan,steps=(replace(plan.steps[0],arguments=changed),)),admitted,parameters,{'qc':binding},registry)
