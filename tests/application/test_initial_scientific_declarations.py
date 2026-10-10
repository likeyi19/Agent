"""Initial finite scientific declarations use the existing scalar binding path.

Semantic responses are scripted. Registered source identity, resource resolution,
compiler, persisted binding and Session admission are ordinary implementations;
the existing recording executor withholds every scientific tool. Tiny H5ADs are
structural witnesses, not compatible human/mouse inference qualification.
"""
from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest

from agent.application.dialogue_execution import scientific_parameters
from agent.application.session_state import SessionError, digest
from agent.application.turn_decisions import (
    Clarify, Execute, IntentError, LeidenResolution, ScalarChoiceArgument, ScopedArgument, SpeciesAnswer,
    admit_scientific_argument, decision_schema, interpret, parse_decision,
)
from agent.orchestration import build_default_tool_registry
from agent.providers import PlanningModelFactoryRegistry
from agent.schemas.orchestration import _serialize
from agent.web.config import build_interactive_application
from test_human_epizoo_mapping import (
    MappingWitness, TOOLS, WithholdScience, configuration,
)
from test_service import _tiny_h5ad


EXPLICIT = ('Analyze these human scATAC-seq cells with EpiZoo, construct their neighbor graph, '
            'perform Leiden clustering, and compute UMAP.')
OMITTED = ('Analyze these scATAC-seq cells with EpiZoo, construct their neighbor graph, '
           'perform Leiden clustering, and compute UMAP.')
SUPPORTED = 'EpiZoo supports human and mouse. Analyze this dataset using EpiZoo.'


def choice(utterance, literal='human', value='human', *, tool='epizoo_embed_cells', argument='species'):
    start = utterance.index(literal)
    return ScalarChoiceArgument(tool, argument, literal, start, start + len(literal), value)


def numeric(utterance, literal='0.7'):
    start = utterance.index(literal)
    return ScopedArgument('cluster_cells', 'resolution', literal, start, start + len(literal))


def wire(decision):
    return json.dumps(dict(turn_schema_version=1, decision=decision))


def public():
    registry = build_default_tool_registry()
    return dict(bases={}, relations=[], dialogue=dict(outputs=[], tools=list(registry.names()),
        scientific_parameters=scientific_parameters(registry)))


class InitialWitness(MappingWitness):
    def __init__(self, calls, *, objective=EXPLICIT, declarations=(), species='human',
                 repeated=False, mutate=None, argument=None, initial_decision=None):
        super().__init__(species, calls, objective=objective)
        self.declarations, self.repeated, self.mutate = declarations, repeated, mutate
        self.argument = argument
        self.initial_decision = initial_decision

    def complete(self, *, prompt, response_schema):
        value = json.loads(prompt)
        if 'turn_schema_version' in value and value['utterance'] == self.objective:
            self.calls.append(value)
            if self.initial_decision is not None:
                return wire(self.initial_decision)
            decision = dict(kind='execute_plan', target='compute_cell_umap',
                arguments=[asdict(item) for item in self.declarations])
            if self.argument is not None:
                decision['argument'] = asdict(self.argument)
            return wire(decision)
        raw = super().complete(prompt=prompt, response_schema=response_schema)
        if self.repeated and 'semantic_prompt_version' in value:
            response = json.loads(raw)
            first = response['decision']['steps'][0]
            response['decision']['steps'].insert(1, first | dict(step_id='embed-again'))
            raw = json.dumps(response)
        if self.mutate is not None and 'output_selection_schema_version' in value:
            self.mutate()
        return raw


def harness(tmp_path, *, objective=EXPLICIT, declarations=None, species='human', typed=None,
            resources=None, repeated=False, mutate=None, plain=False, selected_id=None, argument=None,
            initial_decision=None):
    config = configuration(tmp_path)
    if resources is not None:
        config = replace(config, epizoo_resources=resources(config.epizoo_resources))
    registry = build_default_tool_registry()
    executor = WithholdScience(registry)
    calls = []
    declarations = (choice(objective, value=species),) if declarations is None else declarations
    factories = PlanningModelFactoryRegistry({'scripted': lambda _: InitialWitness(calls,
        objective=objective, declarations=declarations, species=species,
        repeated=repeated, mutate=mutate, argument=argument, initial_decision=initial_decision)})

    def application():
        return build_interactive_application(config, planning_model_factory_registry=factories,
            registry=registry, executor=executor)

    app = application()
    app.create_session('session')
    source = _tiny_h5ad(config.upload_root / 'structural-source.h5ad')
    registered = app.resources.register('source', source, label='Registered structural fixture',
        attribution='Exact source identity only; scientific compatibility is not asserted.')
    from agent.application.local_resources import select_epizoo_resource
    settings = dict(device='cuda:0') | (typed or {})
    settings, selection_error = select_epizoo_resource(settings, selected_id, config.epizoo_resources)
    binding = (app.resources.resolve(registered.resource_id, tool_name='inspect_scATAC') if plain else
        app.resources.compose_h5ad(registered.resource_id, settings, resource_selection_error=selection_error))
    return app, config, registered, binding, executor, calls, application


def submit(h, utterance=EXPLICIT, turn='initial'):
    app, _, _, binding, *_ = h
    return app.submit_turn('session', turn, utterance, expected_generation=0, registered_input=binding)


def assert_no_publication(h):
    app, _, _, _, _, _, _ = h
    state = app._application.sessions.load('session')
    assert not state.revisions and state.generation == 0
    return state


def assert_admitted(h, result, *, species='human', objective=EXPLICIT, extra=None):
    app, config, record, binding, executor, _, _ = h
    assert result.status == 'failed' and result.error.code == 'TEST_SCIENCE_WITHHELD', result
    assert result.revision_id is None and not result.steps
    plan, = executor.plans
    assert tuple(step.tool_name for step in plan.steps) == TOOLS
    state = assert_no_publication(h)
    interaction = state.interactions[-1]
    assert interaction.prerequisite is None
    received = _serialize(binding.execution_inputs)
    if 'continuation' not in interaction.admitted:
        assert interaction.submission['execution_inputs'] == received
    selected = interaction.admitted['inputs']['expected_resource_identity']['resource_id']
    resource = next(row for row in config.epizoo_resources if row.species == species and row.resource_id == selected)
    expected = received | dict(species=species) | resource.inputs() | (extra or {})
    assert _serialize(interaction.admitted['inputs']) == expected
    assert interaction.admitted['registered_input']['resource_id'] == record.resource_id
    assert interaction.admitted['registered_input']['record_sha256'] == record.record_sha256
    for key, value in expected.items():
        if key in ('input_path', 'species', 'device', 'checkpoint_path', 'expected_resource_identity'):
            assert _serialize(plan.steps[0].arguments[key]) == value
    run = app._application.run_store.load(result.run_id)
    assert run.request.prompt == objective and not run.steps
    assert not run.run_verification.passed
    return state


def test_old_numeric_wire_and_new_choice_wire_share_one_arguments_contract():
    utterance = EXPLICIT + ' Use resolution 0.7.'
    declarations = (choice(utterance), numeric(utterance))
    parsed = parse_decision(wire(dict(kind='execute_plan', target='compute_cell_umap',
        arguments=[asdict(item) for item in declarations])))
    assert parsed == Execute('current', 'plan', 'compute_cell_umap', None, arguments=declarations)
    assert parse_decision(wire(dict(kind='execute_plan', target='cluster_cells',
        arguments=[asdict(declarations[1])]))) == Execute('current', 'plan', 'cluster_cells', None,
            arguments=declarations[1:])
    assert parse_decision(wire(dict(kind='execute_plan', target='compute_cell_umap'))) == Execute(
        'current', 'plan', 'compute_cell_umap', None)


def test_only_reviewed_species_choice_is_offered_in_closed_schema():
    context = public()
    parameters = context['dialogue']['scientific_parameters']
    choices = [row for row in parameters if row.get('choices')]
    assert len(choices) == 1
    assert choices[0]['tool'] == 'epizoo_embed_cells' and choices[0]['argument'] == 'species'
    assert choices[0]['choices'] == ['human', 'mouse'] or choices[0]['choices'] == ('human', 'mouse')
    assert len(parameters) == 15
    execution = next(row for row in decision_schema(context)['properties']['decision']['anyOf']
        if row['properties']['kind']['enum'] == ['execute_plan'])
    members = execution['properties']['arguments']['items']['anyOf']
    categorical, = [row for row in members if 'value' in row['properties']]
    assert categorical['properties']['value']['enum'] == ['human', 'mouse']
    assert categorical['additionalProperties'] is False
    assert set(categorical['properties']) == {'tool', 'argument', 'literal', 'start', 'end', 'value'}
    assert all('value' not in row['properties'] for row in members if row is not categorical)


@pytest.mark.parametrize('utterance,literal,value', [
    ('Embed these Human cells.', 'Human', 'human'),
    ('Embed these cells from mice.', 'mice', 'mouse'),
    ('请分析 human cells。', 'human', 'human'),
    ('Embed these Homo sapiens cells.', 'Homo sapiens', 'human'),
])
def test_canonical_choice_is_provider_owned_and_span_is_exact(utterance, literal, value):
    declaration = choice(utterance, literal, value)
    spec = build_default_tool_registry().get('epizoo_embed_cells').required_arguments['species']
    assert admit_scientific_argument(declaration, utterance, spec) == value


@pytest.mark.parametrize('change', [dict(value=True), dict(value=1), dict(value=None),
                                  dict(extra='invented'), dict(start=-1), dict(end=5000)])
def test_choice_wire_remains_closed_and_strict(change):
    item = asdict(choice(EXPLICIT)) | change
    with pytest.raises(IntentError, match='invalid_decision'):
        parse_decision(wire(dict(kind='execute_plan', target='compute_cell_umap', arguments=[item])))


@pytest.mark.parametrize('literal', [' ', '\t\n'])
def test_whitespace_only_choice_literal_cannot_ground_a_scientific_declaration(literal):
    item = asdict(choice(EXPLICIT)) | dict(literal=literal, start=0, end=len(literal))
    with pytest.raises(IntentError, match='invalid_decision'):
        parse_decision(wire(dict(kind='execute_plan', target='compute_cell_umap', arguments=[item])))


@pytest.mark.parametrize('kind', ['answer_prerequisite', 'answer_parameters'])
def test_choice_argument_does_not_extend_existing_pending_answer_contracts(kind):
    decision = dict(kind=kind, pending='@species' if kind == 'answer_prerequisite' else '@parameters',
        arguments=[asdict(choice(EXPLICIT))])
    if kind == 'answer_prerequisite':
        decision['species'] = 'human'
    with pytest.raises(IntentError, match='invalid_decision'):
        parse_decision(wire(decision))
    assert parse_decision(wire(dict(kind='answer_prerequisite', pending='@species',
        species='human'))) == SpeciesAnswer('@species', 'human')


def test_choice_only_context_preserves_pending_schema_without_empty_numeric_union():
    context = public()
    context['dialogue']['scientific_parameters'] = [
        row for row in context['dialogue']['scientific_parameters'] if row.get('choices')]
    context['dialogue']['pending_prerequisite'] = dict(handle='@species', field='species',
        choices=['human', 'mouse'], objective=OMITTED)
    variants = decision_schema(context)['properties']['decision']['anyOf']
    answer, = [row for row in variants if row['properties']['kind']['enum'] == ['answer_prerequisite']]
    assert 'arguments' not in answer['properties']
    assert answer['properties']['species']['enum'] == ['human', 'mouse']
    assert 'argument' in answer['properties']
    context['dialogue']['pending_prerequisite'] = dict(handle='@parameters', field='parameters',
        objective='Select cells.', missing=['min_tss_enrichment'])
    assert all(row['properties']['kind']['enum'] != ['answer_parameters']
        for row in decision_schema(context)['properties']['decision']['anyOf'])


@pytest.mark.parametrize('change', [dict(literal='Human'), dict(start=0), dict(end=len(EXPLICIT) + 1),
                                  dict(value='macaque')])
def test_invalid_or_ungrounded_choice_never_becomes_a_default(change):
    item = replace(choice(EXPLICIT), **change)
    spec = build_default_tool_registry().get('epizoo_embed_cells').required_arguments['species']
    with pytest.raises(IntentError):
        admit_scientific_argument(item, EXPLICIT, spec)


def test_prompt_assigns_dataset_declaration_and_model_support_meaning_to_interpreter():
    class Model:
        def complete(self, *, prompt, response_schema):
            captured = json.loads(prompt)
            instructions = ' '.join(captured['instructions']).lower()
            assert 'model supporting human and mouse' in instructions and 'not a declaration' in instructions
            assert 'never infer species' in instructions
            assert captured['utterance'] == SUPPORTED
            return wire(dict(kind='execute_plan', target='compute_cell_umap', arguments=[]))
    assert interpret(Model(), SUPPORTED, public()).arguments == ()


def test_supported_new_request_with_omitted_declaration_reaches_existing_compiler_handoff():
    class Model:
        def complete(self, *, prompt, response_schema):
            captured = json.loads(prompt)
            instructions = ' '.join(captured['instructions'])
            assert 'otherwise supported new scientific command with a selected input' in instructions
            assert 'Choose execute_plan and omit that declaration from arguments' in instructions
            assert 'Planner/compiler determines the exact missing prerequisite' in instructions
            assert 'Application creates its durable clarification' in instructions
            assert 'never fabricate a pending request' in instructions
            return wire(dict(kind='execute_plan', target='compute_cell_umap', arguments=[]))
    decision = interpret(Model(), OMITTED, public())
    assert decision == Execute('current', 'plan', 'compute_cell_umap', None)


@pytest.mark.parametrize('pending_field,expected', [(None, False), ('species', True), ('parameters', False)])
def test_missing_species_schema_requires_existing_species_pending_context(pending_field, expected):
    context = public()
    if pending_field is not None:
        context['dialogue']['pending_prerequisite'] = dict(handle='@species' if pending_field == 'species' else '@parameters',
            field=pending_field, choices=['human', 'mouse'], objective=OMITTED)
    clarify, = [row for row in decision_schema(context)['properties']['decision']['anyOf']
        if row['properties']['kind']['enum'] == ['clarify']]
    reasons = clarify['properties']['reason']['enum']
    assert ('missing_species' in reasons) is expected
    assert {'ambiguous_species', 'unsupported_species', 'conflicting_species'} <= set(reasons)


def test_historical_missing_species_wire_remains_parseable_without_adding_pending_state():
    assert parse_decision(wire(dict(kind='clarify', reason='missing_species'))) == Clarify('missing_species')


@pytest.mark.parametrize('field,kind', [('species', 'answer_prerequisite'), ('parameters', 'answer_parameters')])
def test_existing_pending_numeric_schema_stays_exact_and_excludes_initial_choices(field, kind):
    context = public()
    context['dialogue']['pending_prerequisite'] = dict(handle='@species' if field == 'species' else '@parameters',
        field=field, choices=['human', 'mouse'], objective=OMITTED)
    answer, = [row for row in decision_schema(context)['properties']['decision']['anyOf']
        if row['properties']['kind']['enum'] == [kind]]
    arguments = answer['properties']['arguments']
    members = arguments['items']['anyOf']
    expected = {(row['tool'], row['argument']) for row in context['dialogue']['scientific_parameters']
        if 'choices' not in row}
    actual = {(row['properties']['tool']['enum'][0], row['properties']['argument']['enum'][0])
        for row in members}
    assert actual == expected and len(actual) == 14
    assert arguments['maxItems'] == 8
    assert all('value' not in row['properties'] and row['additionalProperties'] is False for row in members)
    if field == 'parameters':
        assert arguments['minItems'] == 1


@pytest.mark.parametrize('species,literal', [('human', 'human'), ('mouse', 'mouse')])
def test_initial_explicit_declaration_binds_selected_source_and_original_default_without_pending(
        tmp_path, species, literal):
    objective = EXPLICIT.replace('human', literal)
    h = harness(tmp_path, objective=objective, species=species,
        declarations=(choice(objective, literal, species),))
    result = submit(h, objective)
    state = assert_admitted(h, result, species=species, objective=objective)
    assert 'species' not in state.interactions[0].submission['execution_inputs']
    assert _serialize(state.interactions[0].admitted['arguments']) == [asdict(choice(objective, literal, species))]
    for step in h[4].plans[0].steps[1:]:
        assert not {'resolution', 'n_neighbors', 'min_dist', 'spread'} & set(step.arguments)


def test_matching_typed_companion_and_explicit_utterance_remain_one_binding(tmp_path):
    h = harness(tmp_path, typed=dict(species='human'))
    result = submit(h)
    state = assert_admitted(h, result)
    assert state.interactions[0].submission['execution_inputs']['species'] == 'human'
    assert len(state.interactions[0].admitted['arguments']) == 1


def test_plain_registered_upload_needs_no_selected_companion_for_initial_species_binding(tmp_path):
    h = harness(tmp_path, plain=True)
    assert h[3].composition is None and set(h[3].execution_inputs) == {'input_path'}
    result = submit(h)
    state = assert_admitted(h, result)
    assert state.interactions[0].admitted['registered_input']['composition'] == 'h5ad-science.v1'
    assert 'device' not in h[4].plans[0].steps[0].arguments


def test_explicit_original_human_resource_priority_is_preserved_after_initial_species_binding(tmp_path):
    def resources(rows):
        human = next(row for row in rows if row.species == 'human')
        return rows + (replace(human, resource_id='human-explicit', default=False),)
    h = harness(tmp_path, resources=resources, selected_id='human-explicit')
    result = submit(h)
    state = assert_admitted(h, result)
    assert state.interactions[0].admitted['inputs']['expected_resource_identity']['resource_id'] == 'human-explicit'


def test_explicit_mouse_resource_cannot_be_replaced_for_initial_human_declaration(tmp_path):
    h = harness(tmp_path, selected_id='mouse-original')
    result = submit(h)
    assert result.status == 'failed' and result.error.code == 'EPIZOO_RESOURCE_SELECTION_INVALID', result
    assert not h[4].plans
    assert_no_publication(h)


def test_conflicting_companion_rejects_initial_declaration_before_planner_or_science(tmp_path):
    h = harness(tmp_path, typed=dict(species='mouse'))
    result = submit(h)
    assert result.status == 'clarification' and result.revision_id is None, result
    assert result.response.clarification['reason'] in ('conflicting_species', 'conflicting_scientific_parameter')
    assert not h[4].plans and len(h[5]) == 1
    assert_no_publication(h)


@pytest.mark.parametrize('reason,objective', [
    ('ambiguous_species', 'Analyze these cells, which might be human or mouse, with EpiZoo.'),
    ('unsupported_species', 'Analyze these macaque cells with EpiZoo.'),
    ('conflicting_species', 'Analyze these human cells, which are mouse, with EpiZoo.'),
])
def test_fresh_semantic_species_refusal_preserves_reason_without_fabricating_pending_request(
        tmp_path, reason, objective):
    h = harness(tmp_path, objective=objective, declarations=(),
        initial_decision=dict(kind='clarify', reason=reason))
    result = submit(h, objective)
    assert result.status == 'clarification' and result.response.clarification['reason'] == reason
    assert tuple(result.response.clarification['choices']) == ('human', 'mouse')
    assert result.run_id is None and result.revision_id is None
    state = assert_no_publication(h)
    assert state.interactions[0].prerequisite is None
    assert state.turns[0].request_id is None and state.turns[0].run_id is None
    assert not h[4].plans and len(h[5]) == 1


@pytest.mark.parametrize('decision', [dict(kind='clarify', reason='missing_species'),
    dict(kind='answer_prerequisite', pending='@species', species='human')])
def test_fresh_missing_species_or_unoffered_answer_cannot_create_a_durable_prerequisite(tmp_path, decision):
    h = harness(tmp_path, objective=OMITTED, declarations=(), initial_decision=decision)
    result = submit(h, OMITTED)
    assert result.status == 'clarification' and result.response.clarification['reason'] == 'invalid_prerequisite'
    assert result.run_id is None and result.revision_id is None
    state = assert_no_publication(h)
    assert state.interactions[0].prerequisite is None
    assert not h[4].plans and len(h[5]) == 1


@pytest.mark.parametrize('objective', [OMITTED, SUPPORTED])
def test_omitted_or_model_support_only_species_keeps_durable_pending_and_original_resume(tmp_path, objective):
    h = harness(tmp_path, objective=objective, declarations=())
    initial = submit(h, objective)
    assert initial.status == 'clarification' and initial.response.clarification['reason'] == 'missing_species'
    assert not h[4].plans
    state = assert_no_publication(h)
    assert state.interactions[0].prerequisite['origin_turn_id'] == 'initial'
    before = len(h[5])
    restored = h[6]()
    assert restored.turn('session', 'initial') == initial
    restored.reopen_session('session')
    assert len(h[5]) == before
    answer = restored.submit_turn('session', 'answer', 'These are human cells.', expected_generation=0)
    state = assert_admitted(h, answer, objective=objective)
    assert state.interactions[-1].admitted['continuation']['origin_turn_id'] == 'initial'


def test_initial_species_and_numeric_resolution_share_exact_consumer_binding(tmp_path):
    objective = EXPLICIT + ' Use resolution 0.7.'
    h = harness(tmp_path, objective=objective, declarations=(choice(objective), numeric(objective)))
    result = submit(h, objective)
    state = assert_admitted(h, result, objective=objective, extra=dict(resolution=.7))
    assert state.interactions[0].admitted['inputs']['resolution'] == .7
    assert h[4].plans[0].steps[2].arguments['resolution'] == .7
    assert 'n_neighbors' not in h[4].plans[0].steps[1].arguments
    assert 'min_dist' not in h[4].plans[0].steps[3].arguments


def test_initial_choice_and_historical_singular_resolution_preserve_both_binding_records(tmp_path):
    objective = EXPLICIT + ' Use resolution 0.7.'
    legacy = LeidenResolution(**asdict(numeric(objective)))
    h = harness(tmp_path, objective=objective, declarations=(choice(objective),), argument=legacy)
    result = submit(h, objective)
    state = assert_admitted(h, result, objective=objective, extra=dict(resolution=.7))
    admitted = state.interactions[0].admitted
    assert _serialize(admitted['argument']) == asdict(legacy)
    assert _serialize(admitted['arguments']) == [asdict(choice(objective))]
    assert h[4].plans[0].steps[2].arguments['resolution'] == .7


@pytest.mark.parametrize('mode,code', [('missing', 'EPIZOO_RESOURCE_REQUIRED'),
                                     ('ambiguous', 'EPIZOO_RESOURCE_AMBIGUOUS')])
def test_initial_declaration_cannot_invent_or_choose_first_missing_or_ambiguous_resource(tmp_path, mode, code):
    def resources(rows):
        if mode == 'missing':
            return tuple(row for row in rows if row.species != 'human')
        human = next(row for row in rows if row.species == 'human')
        return rows + (replace(human, resource_id='human-other'),)
    h = harness(tmp_path, resources=resources)
    result = submit(h)
    assert result.status == 'failed' and result.error.code == code, result
    assert not h[4].plans and result.revision_id is None
    assert_no_publication(h)


def test_initial_choice_with_repeated_actual_consumers_never_binds_to_first_step(tmp_path):
    h = harness(tmp_path, repeated=True)
    result = submit(h)
    # Existing compiler scope rejection occurs before actual-consumer admission:
    # this candidate also repeats optional checkpoint/device/resource consumers.
    assert result.status == 'failed' and result.error.code == 'AMBIGUOUS_OPTIONAL_INPUT_SCOPE', result
    assert not h[4].plans and not any('output_selection_schema_version' in row for row in h[5])
    assert_no_publication(h)


def test_source_mutation_after_selection_fails_before_executor(tmp_path):
    captured = {}
    def mutate():
        captured['source'].write_bytes(captured['original'] + b'changed')
    h = harness(tmp_path, mutate=mutate)
    source = Path(h[2].source_path)
    original = source.read_bytes()
    captured.update(source=source, original=original)
    try:
        result = submit(h)
        assert result.status == 'failed' and result.error.code == 'LOCAL_RESOURCE_INTEGRITY_INVALID', result
        assert not h[4].plans
        assert_no_publication(h)
    finally:
        source.write_bytes(original)


def test_completed_initial_admission_reopens_and_retries_without_rebinding_or_provider_calls(tmp_path):
    h = harness(tmp_path)
    result = submit(h)
    assert_admitted(h, result)
    counts = len(h[5]), len(h[4].plans)
    restored = h[6]()
    assert restored.turn('session', 'initial') == result
    restored.reopen_session('session')
    assert restored.submit_turn('session', 'initial', EXPLICIT, expected_generation=0,
        registered_input=h[3]) == result
    assert counts == (len(h[5]), len(h[4].plans))


@pytest.mark.parametrize('mutation', ['declaration', 'value', 'source', 'device', 'pins', 'registration'])
def test_valid_checksum_cannot_conceal_changed_initial_declaration_binding(tmp_path, mutation):
    h = harness(tmp_path)
    result = submit(h)
    assert_admitted(h, result)
    sessions = h[0]._application.sessions
    path = sessions._store._path('session', '.json')
    original = path.read_bytes()
    envelope = json.loads(original)
    admitted = envelope['record']['interactions'][0]['admitted']
    if mutation == 'declaration':
        del admitted['arguments']
    elif mutation == 'value':
        admitted['arguments'][0]['value'] = 'mouse'
    elif mutation == 'source':
        admitted['inputs']['input_path'] = str(tmp_path / 'different.h5ad')
    elif mutation == 'device':
        admitted['inputs']['device'] = 'cpu'
    elif mutation == 'pins':
        admitted['inputs']['expected_resource_identity']['checkpoint_sha256'] = 'f' * 64
    else:
        admitted['registered_input']['record_sha256'] = 'f' * 64
    envelope['sha256'] = digest(envelope['record'])
    counts = len(h[5]), len(h[4].plans)
    try:
        path.write_text(json.dumps(envelope))
        with pytest.raises(SessionError):
            sessions.load('session')
    finally:
        path.write_bytes(original)
    assert counts == (len(h[5]), len(h[4].plans))
