"""Typed shared role questions and pure captured-source admission contracts."""
from dataclasses import FrozenInstanceError, asdict, replace
import json

import pytest

from agent.application.local_resources import (
    RegisteredInputGroup, ResourceAdmissionError, group_role_inputs, validate_group_roles,
)
from agent.application.turn_decisions import (
    Execute, InputRole, IntentError, RequestRoles, RoleAnswer, RoleRequirement,
    ScopedArgument, ScalarChoiceArgument, decision_schema, interpret, parse_decision,
)
from agent.orchestration.registry import build_default_tool_registry


def requirements(tool='evaluate_cell_clustering'):
    names = ('analysis', 'ground_truth_dataset') if tool == 'evaluate_cell_clustering' else (
        'reference_dataset', 'query_dataset')
    return [dict(tool=tool, port=port) for port in names]


def roles(tool='evaluate_cell_clustering'):
    return [requirement | dict(source=f'source.{index}')
            for index, requirement in enumerate(requirements(tool))]


def wire(value):
    return json.dumps(dict(turn_schema_version=1, decision=value))


def request(**changes):
    return dict(kind='request_roles', target='evaluate_cell_clustering',
                required_roles=requirements(), roles=[]) | changes


def selection():
    return RegisteredInputGroup([dict(resource_id='local-' + str(index)*64,
        record_sha256=str(index + 2)*64, source_path=f'/unopened/source-{index}.h5ad',
        input_type='h5ad', label=f'Selected {index + 1} 类型 Δ') for index in range(2)], {})


def public(*, selected=True, pending=False):
    value = dict(bases={}, relations=[], dialogue=dict(outputs=[],
        tools=['evaluate_cell_clustering', 'transfer_cell_labels'], scientific_parameters=[
            dict(tool='evaluate_cell_clustering', argument='label_key', required=True,
                 description='Explicit evaluation-only column.', exact_string=True),
            dict(tool='transfer_cell_labels', argument='reference_label_key', required=True,
                 description='Explicit reference column.', exact_string=True),
        ]))
    if selected:
        value['dialogue'].update(selected_sources=[dict(source=f'source.{index}', position=index + 1,
            label=f'Selected {index + 1} 类型 Δ', input_type='h5ad') for index in range(2)],
            input_roles=requirements() + requirements('transfer_cell_labels'))
    if pending:
        value['dialogue']['pending_prerequisite'] = dict(handle='@roles', field='roles',
            objective='Evaluate the selected fixed clustering against the selected ground truth.',
            required_roles=requirements(), roles=[])
    return value


def variants(value):
    return {variant['properties']['kind']['enum'][0]: variant
            for variant in decision_schema(value)['properties']['decision']['anyOf']}


def test_role_request_retains_exact_strings_and_is_not_an_executable_plan():
    utterance = 'Evaluate these datasets with column "Cell 类型 Δ".'
    literal = 'Cell 类型 Δ'
    start = utterance.index(literal)
    argument = ScopedArgument('evaluate_cell_clustering', 'label_key', literal, start, start + len(literal))
    value = parse_decision(wire(request(arguments=[asdict(argument)])))
    assert value == RequestRoles('evaluate_cell_clustering',
        tuple(RoleRequirement(**r) for r in requirements()), arguments=(argument,))
    assert not isinstance(value, Execute)
    with pytest.raises(FrozenInstanceError):
        value.required_roles = ()


@pytest.mark.parametrize('changes', [
    dict(required_roles=[]),
    dict(required_roles=requirements()*5), dict(required_roles=[requirements()[0]]*2),
    dict(required_roles=[requirements()[0] | dict(source='source.0'), requirements()[1]]),
    dict(target='cluster_cells'), dict(context='previous'), dict(roles=roles()),
    dict(roles=[dict(tool='transfer_cell_labels', port='reference_dataset', source='source.0')]),
    dict(required_roles=[dict(tool='illegal/name', port='analysis'), requirements()[1]]),
    dict(inputs={}),
])
def test_role_request_wire_is_closed_and_requires_a_genuinely_missing_association(changes):
    with pytest.raises(IntentError, match='invalid_decision'):
        parse_decision(wire(request(**changes)))


def test_role_answers_allow_partial_associations_or_an_exact_scientific_declaration():
    reply = dict(kind='answer_roles', pending='@roles', roles=roles()[:1])
    assert parse_decision(wire(reply)) == RoleAnswer('@roles', (InputRole(**roles()[0]),))
    declaration = dict(tool='evaluate_cell_clustering', argument='label_key',
                       literal='类型 Δ', start=1, end=5)
    assert parse_decision(wire(reply | dict(roles=[], arguments=[declaration]))) == RoleAnswer(
        '@roles', (), (ScopedArgument(**declaration),))


@pytest.mark.parametrize('changes', [
    dict(roles=[]), dict(roles=None), dict(pending=''), dict(pending='x'*129),
    dict(roles=[dict(tool='evaluate_cell_clustering', port='analysis', source='source.16')]),
    dict(roles=roles(), required_roles=requirements()),
    dict(arguments=[dict(tool='epizoo_embed_cells', argument='species', literal='mouse',
                         start=0, end=5, value={'untrusted': 'mouse'})]),
])
def test_role_answer_has_no_arbitrary_values_or_source_identity_wire(changes):
    with pytest.raises(IntentError, match='invalid_decision'):
        parse_decision(wire(dict(kind='answer_roles', pending='@roles', roles=roles()[:1]) | changes))


def test_role_answer_retains_reviewed_scalar_choice_wire():
    declaration = dict(tool='epizoo_embed_cells', argument='species', literal='mouse',
                       start=0, end=5, value='mouse')
    decision = parse_decision(wire(dict(kind='answer_roles', pending='@roles',
        roles=roles()[:1], arguments=[declaration])))
    assert decision.arguments == (ScalarChoiceArgument(**declaration),)


def test_schema_offers_new_decisions_only_in_exact_captured_role_context():
    assert 'request_roles' not in variants(public(selected=False))
    assert 'answer_roles' not in variants(public())
    question = variants(public())['request_roles']
    assert question['additionalProperties'] is False
    assert question['properties']['required_roles']['minItems'] == 2
    answer = variants(public(pending=True))['answer_roles']['properties']
    assert answer['pending']['enum'] == ['@roles']
    assert [entry['properties']['port']['enum'] for entry in answer['roles']['items']['anyOf']] == [
        ['analysis'], ['ground_truth_dataset']]
    argument_variants = answer['arguments']['items']['anyOf']
    entries = [branch for entry in argument_variants for branch in entry.get('anyOf', [entry])]
    assert {tuple(entry['properties']['argument']['enum']) for entry in entries} == {('label_key',)}
    assert 'answer_roles' not in variants(public(selected=False, pending=True))


@pytest.mark.parametrize('tool,ports', [
    ('inspect_scATAC', {'dataset'}), ('pca_embed_cells', {'dataset'}),
    ('transfer_cell_labels', {'reference_dataset', 'query_dataset'}),
    ('evaluate_cell_clustering', {'analysis', 'ground_truth_dataset'}),
])
def test_absent_files_are_registry_requirements_not_language_rules(tool, ports):
    from dataclasses import dataclass, field
    from agent.application.dialogue_execution import missing_file_request
    @dataclass
    class Capture:
        submission: dict
        snapshot: dict = field(default_factory=dict)
    registry = build_default_tool_registry()
    decision = parse_decision(wire(dict(kind='execute_plan', target=tool)))
    interaction = Capture(submission=dict(execution_inputs={}))
    result = missing_file_request(registry, interaction, decision)
    assert isinstance(result, RequestRoles) and result.target == tool
    assert {r.port for r in result.required_roles} == ports
    assert not result.roles
    # A supplied source or explicitly selected result is never replaced by an upload prompt.
    assert missing_file_request(registry, Capture(submission={
        'execution_inputs': {'input_path': '/selected/source'}}), decision) is None
    assert missing_file_request(registry, interaction, replace(decision, context='current')) is None
    assert missing_file_request(registry, interaction, replace(decision, target='not_registered')) is None
    with pytest.raises(IntentError, match='ambiguous_subject'):
        missing_file_request(registry, Capture(submission={
            'execution_inputs': {}, 'available_input_overflow': True}), decision)


def test_interpreter_gets_safe_original_roles_and_no_keyword_repair():
    class Model:
        def complete(self, *, prompt, response_schema):
            self.prompt = json.loads(prompt)
            self.schema = response_schema
            return wire(dict(kind='answer_roles', pending='@roles', roles=roles()))
    model = Model()
    answer = 'The first dataset is fixed analysis and the second is ground truth.'
    value = interpret(model, answer, public(pending=True))
    assert value == RoleAnswer('@roles', tuple(InputRole(**r) for r in roles()))
    assert model.prompt['utterance'] == answer
    assert model.prompt['dialogue']['pending_prerequisite'] == public(pending=True)['dialogue']['pending_prerequisite']
    instructions = ' '.join(model.prompt['instructions'])
    assert 'Previously admitted roles cannot be reassigned' in instructions
    assert 'only the original task' in instructions
    assert '/unopened/' not in json.dumps(model.prompt)


@pytest.mark.parametrize('tool', ['evaluate_cell_clustering', 'transfer_cell_labels'])
def test_pure_pending_admission_shares_registry_roles_without_opening_sources(tool, monkeypatch):
    registry, group = build_default_tool_registry(), selection()
    def forbidden(*args, **kwargs):
        raise AssertionError('Pure pending role admission opened a source.')
    monkeypatch.setattr('builtins.open', forbidden)
    required = tuple(RoleRequirement(**r) for r in requirements(tool))
    partial = validate_group_roles(group, roles(tool)[:1], registry, required)
    assert partial == (InputRole(**roles(tool)[0]),)
    completed = validate_group_roles(group, roles(tool), registry, required, complete=True)
    assert completed == tuple(InputRole(**r) for r in sorted(roles(tool), key=lambda r: (r['tool'], r['port'])))
    assert not group.roles and not group.execution_inputs
    assert len(group_role_inputs(group, [asdict(r) for r in completed], registry)) == 2


@pytest.mark.parametrize('attachment_candidates', [False, True])
def test_pending_distinct_consumers_can_explicitly_share_one_exact_source(monkeypatch, attachment_candidates):
    registry = build_default_tool_registry()
    group = selection()
    group = RegisteredInputGroup(group.members if attachment_candidates else group.members[:1], {},
                                 attachment_candidates=attachment_candidates)
    required = [dict(tool=tool, port='dataset') for tool in ('inspect_scATAC', 'pca_embed_cells')]
    assigned = [role | dict(source='source.0') for role in required]
    def forbidden(*args, **kwargs):
        raise AssertionError('Pure pending role admission opened a scientific source.')
    monkeypatch.setattr('builtins.open', forbidden)
    partial = validate_group_roles(group, assigned[:1], registry, required)
    assert partial == (InputRole(**assigned[0]),)
    completed = validate_group_roles(group, assigned, registry, required, complete=True)
    assert completed == tuple(InputRole(**role) for role in assigned)
    assert group_role_inputs(group, [asdict(role) for role in completed], registry) == {
        'input_path': group.members[0]['source_path']}


@pytest.mark.parametrize('complete', [False, True])
def test_pending_explicit_shared_alias_rejects_different_source_bindings(complete):
    registry, group = build_default_tool_registry(), selection()
    required = [dict(tool=tool, port='dataset') for tool in ('inspect_scATAC', 'pca_embed_cells')]
    assigned = [role | dict(source=f'source.{index}') for index, role in enumerate(required)]
    with pytest.raises(ResourceAdmissionError) as rejected:
        validate_group_roles(group, assigned, registry, required, complete=complete)
    assert rejected.value.code == 'MULTI_INPUT_ROLE_INVALID'


@pytest.mark.parametrize('variant', ['missing', 'count', 'duplicate_required', 'unsupported_port',
                                    'shared_alias', 'foreign_role', 'same_source', 'unoffered_source'])
def test_pending_role_compatibility_fails_closed(variant):
    registry, group = build_default_tool_registry(), selection()
    required, assigned = requirements(), roles()
    if variant == 'missing':
        assigned = assigned[:1]
    elif variant == 'count':
        required = required[:1]
    elif variant == 'duplicate_required':
        required = required[:1]*2
    elif variant == 'unsupported_port':
        required[0]['port'] = 'unreviewed'
    elif variant == 'shared_alias':
        required[1] = dict(tool='cluster_cells', port='analysis')
    elif variant == 'foreign_role':
        assigned[0] = dict(tool='transfer_cell_labels', port='reference_dataset', source='source.0')
    elif variant == 'same_source':
        assigned[1]['source'] = 'source.0'
    else:
        assigned[1]['source'] = 'source.7'
    with pytest.raises(ResourceAdmissionError) as rejected:
        validate_group_roles(group, assigned, registry, required, complete=True)
    assert rejected.value.code == 'MULTI_INPUT_ROLE_INVALID'


def test_existing_complete_group_binding_does_not_accept_partial_pending_associations():
    registry, group = build_default_tool_registry(), selection()
    validate_group_roles(group, roles()[:1], registry, requirements())
    with pytest.raises(ResourceAdmissionError) as rejected:
        group_role_inputs(group, roles()[:1], registry)
    assert rejected.value.code == 'MULTI_INPUT_ROLE_INVALID'
    assert parse_decision(wire(dict(kind='execute_plan', target='evaluate_cell_clustering'))) == Execute(
        'current', 'plan', 'evaluate_cell_clustering', None)


@pytest.mark.parametrize('association_stage', ['initial', 'answer'])
def test_partial_role_source_mutation_is_rejected_during_durable_session_decode(tmp_path, association_stage):
    from agent.application.session_state import AnalysisSession, SessionError
    from agent.schemas.orchestration import _serialize
    from test_role_declaration_continuation import (
        OBJECTIVE, REFERENCE, REFERENCE_ANSWER, transfer_continuation_harness, pending, request, submit,
    )
    decisions = {OBJECTIVE: request(roles=[REFERENCE])} if association_stage == 'initial' else None
    app, _, group, executor, _, _ = transfer_continuation_harness(tmp_path, decisions=decisions)
    first = submit(app, group=group)
    if association_stage == 'answer':
        pending(app, first)
        result = submit(app, turn='partial', utterance=REFERENCE_ANSWER)
        record = pending(app, result, turn='partial', missing=[
            dict(tool='transfer_cell_labels', port='query_dataset')])
    else:
        record = pending(app, first, missing=[dict(tool='transfer_cell_labels', port='query_dataset')])
    state = app._application.sessions.load('session')
    assert not state.turns and not state.revisions and not executor.plans
    assert _serialize(record.admitted['registered_input']['roles']) == [REFERENCE]
    assert not state.interactions[0].submission['registered_input']['roles']
    before = _serialize(state.to_dict())
    assert AnalysisSession.from_dict(before) == state
    forged = _serialize(state.to_dict())
    forged['interactions'][-1]['admitted']['roles'][0]['source'] = 'source.1'
    with pytest.raises(SessionError, match='pending scientific role binding'):
        AnalysisSession.from_dict(forged)
    assert _serialize(app._application.sessions.load('session').to_dict()) == before
    assert not executor.plans


@pytest.mark.parametrize('reselect_original', [False, True])
def test_reference_query_roles_then_column_answer_retains_exact_sources_and_resource_choice(tmp_path, reselect_original):
    from dataclasses import replace
    from agent.application.session_state import AnalysisSession, SessionError
    from agent.schemas.orchestration import _serialize
    from test_role_declaration_continuation import (
        LABEL, ROLES, ROLE_ANSWER, assert_withheld, label_declaration,
        transfer_continuation_harness, pending, request, submit,
    )
    objective = 'Transfer labels between these two selected datasets.'
    reply = f'Use the reference column "{LABEL}".'
    decisions = {objective: request(declarations=[]), reply: dict(kind='answer_parameters',
        pending='@parameters', arguments=[label_declaration(reply)])}
    app, records, group, executor, calls, companions = transfer_continuation_harness(tmp_path, decisions=decisions)
    token = dict(input_set_id='operator', definition_sha256='a'*64)
    group = replace(group, resource_choice=token)
    first = pending(app, submit(app, group=group, utterance=objective))
    middle_view = submit(app, turn='roles', utterance=ROLE_ANSWER)
    assert middle_view['response']['kind'] == 'clarify', middle_view
    state = app._application.sessions.load('session')
    middle = state.interactions[-1]
    assert middle.prerequisite['field'] == 'parameters'
    assert list(middle.prerequisite['missing']) == ['reference_label_key']
    assert not executor.plans and not state.revisions
    final_view = submit(app, turn='column', utterance=reply, group=group if reselect_original else None)
    assert_withheld(final_view, executor, records, companions)
    state = app._application.sessions.load('session')
    final = state.interactions[-1]
    assert state.interactions[0] == first and state.interactions[1] == middle
    assert _serialize(final.admitted['roles']) == ROLES
    assert final.admitted['registered_input']['resource_choice'] == token
    assert _serialize(final.admitted['registered_input']['members']) == _serialize(group.members)
    run = app._application.run_store.load(final_view['run_id'])
    assert run.request.prompt == objective and run.request.inputs['reference_label_key'] == LABEL
    decoded = _serialize(state.to_dict())
    assert AnalysisSession.from_dict(decoded) == state
    decoded['interactions'][-1]['admitted'].pop('registered_input')
    with pytest.raises(SessionError, match='pending scientific parameter binding'):
        AnalysisSession.from_dict(decoded)
    counts = len(calls), len(executor.plans)
    assert submit(app, turn='column', utterance=reply, group=group if reselect_original else None) == final_view
    assert counts == (len(calls), len(executor.plans))


def test_role_request_cannot_bypass_existing_candidate_selection_guard(tmp_path, monkeypatch):
    """A capturable role question remains an execution request at M17.2 admission.

    The existing predicate independently owns candidate-reference language. This
    seam supplies its positive classification and verifies the new decision does
    not evade the already-authoritative guard by delaying executable planning.
    """
    from test_role_declaration_continuation import (
        OBJECTIVE, transfer_continuation_harness, submit,
    )
    seen = []
    def candidate_reference(interaction, prior):
        seen.append((interaction.turn_id, interaction.utterance, prior))
        return True
    monkeypatch.setattr('agent.application.scientific_guidance.references_candidate', candidate_reference)
    app, _, group, executor, calls, _ = transfer_continuation_harness(tmp_path)
    result = submit(app, group=group)
    assert seen == [('first', OBJECTIVE, None)]
    assert result['status'] == 'failed'
    assert result['error']['code'] == 'INTERACTIVE_INTENT_BINDING_INVALID'
    state = app._application.sessions.load('session')
    assert state.interactions[-1].prerequisite is None
    assert all(turn.request_id is None and turn.run_id is None for turn in state.turns)
    assert not state.revisions and not executor.plans
    assert len(calls) == 1
