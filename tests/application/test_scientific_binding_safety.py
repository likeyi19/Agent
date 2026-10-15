"""Fail-closed persistence and exact-source checks for shared declarations."""
from contextlib import contextmanager
from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest

from agent.application import ResearchAgentApplication
from agent.application.dialogue_execution import (
    _bind_arguments, _check_parameter_plan, _parameter_input,
    _validate_argument_record, _validate_resolution_record,
)
from agent.application.session_state import (
    AnalysisSession, Interaction, SessionConflictError, SessionError, digest,
)
from agent.application.turn_decisions import IntentError, LeidenResolution, ScopedArgument
from agent.orchestration import build_default_tool_registry
from agent.orchestration.registry import (
    PlanningProvenanceRole, SemanticPortMember, SemanticRequestMember,
    SemanticRequestSourceSpec, ToolRegistry,
)
from agent.schemas import AgentPlan, PlanStep

from test_scientific_parameter_continuation import (
    BOTH, DEPTH, OBJECTIVE, PROFILE, REQUIRED, TOOL, TSS, SelectionModel,
    accepted_step, answer, argument, assert_pending, fixture_factory, qc_case,
    runtime, legacy_selection_harness as selection_harness, selection_inputs, selection_work, submit,
)


@contextmanager
def mutated_record(harness, mutation):
    sessions = harness.service._application.sessions
    path = sessions._store._path('session', '.json')
    original = path.read_bytes()
    envelope = json.loads(original)
    mutation(envelope['record'])
    envelope['sha256'] = digest(envelope['record'])
    path.write_text(json.dumps(envelope))
    try:
        yield sessions
    finally:
        path.write_bytes(original)


def initial_partial(harness, selection_inputs):
    utterance = OBJECTIVE + ' ' + DEPTH
    harness.decisions[utterance] = dict(kind='execute_plan', target=TOOL,
        arguments=[argument(utterance, REQUIRED[0], '1')])
    return submit(harness, utterance=utterance, inputs=selection_inputs)


def direct_sessions(harness, workspace):
    app = ResearchAgentApplication(workspace, registry=harness.service._application.registry,
        primary_planning_profile=PROFILE, planning_model_factory_registry=harness.service._factory)
    sessions = app.sessions
    sessions.create('session')
    return sessions, app.runtime.planner.model


@pytest.mark.parametrize('mutation', [
    'deleted_declaration', 'changed_value', 'boolean_equal_one', 'changed_identity', 'changed_literal',
])
def test_pending_initial_declaration_cannot_be_lost_or_reassigned_under_valid_checksum(
        tmp_path, selection_inputs, mutation):
    harness = selection_harness(tmp_path / 'application')
    harness.service.create_session('session')
    assert_pending(initial_partial(harness, selection_inputs), REQUIRED[1])

    def corrupt(record):
        admitted = record['interactions'][0]['admitted']
        if mutation == 'deleted_declaration':
            del admitted['arguments']
        elif mutation == 'changed_value':
            admitted['inputs'][REQUIRED[0]] = 2
        elif mutation == 'boolean_equal_one':
            admitted['inputs'][REQUIRED[0]] = True
        elif mutation == 'changed_identity':
            admitted['arguments'][0]['argument'] = 'min_tss_flank_evidence'
        else:
            admitted['arguments'][0]['literal'] = '2'

    with selection_work() as calls, mutated_record(harness, corrupt) as sessions:
        with pytest.raises(SessionError):
            sessions.load('session')
    assert not calls
    assert harness.service._application.sessions.load('session').interactions[0].admitted['inputs'][REQUIRED[0]] == 1


@pytest.mark.parametrize('mutation', ['deleted_declaration', 'changed_predecessor', 'changed_input', 'added_unscoped_value'])
def test_partial_answer_preserves_exact_predecessor_and_unrelated_inputs(
        tmp_path, selection_inputs, mutation):
    harness = selection_harness(tmp_path / 'application')
    harness.service.create_session('session')
    submit(harness, inputs=selection_inputs)
    assert_pending(submit(harness, 'partial', DEPTH), REQUIRED[1])

    def corrupt(record):
        admitted = record['interactions'][-1]['admitted']
        if mutation == 'deleted_declaration':
            del admitted['arguments']
        elif mutation == 'changed_predecessor':
            admitted['parameter_continuation']['binding_turn_id'] = 'partial'
        elif mutation == 'changed_input':
            admitted['inputs']['barcode_qc_manifest_sha256'] = 'f' * 64
        else:
            admitted['inputs']['min_tss_flank_evidence'] = 20

    with selection_work() as calls, mutated_record(harness, corrupt) as sessions:
        with pytest.raises(SessionError):
            sessions.load('session')
    assert not calls


def test_source_mutation_after_partial_answer_reaches_owner_rejection_without_selection_production(
        tmp_path, selection_inputs):
    harness = selection_harness(tmp_path / 'application')
    harness.service.create_session('session')
    submit(harness, inputs=selection_inputs)
    assert_pending(submit(harness, 'partial', DEPTH), REQUIRED[1])
    source = Path(selection_inputs['barcode_qc_manifest_path'])
    original = source.read_bytes()
    source.write_bytes(original + b'\n')
    try:
        with selection_work() as calls:
            view = submit(harness, 'remaining', TSS)
        assert view['revision_id'] is None and view['response']['status'] != 'activated', view
        assert not calls and not harness.service._application.sessions.load('session').revisions
    finally:
        source.write_bytes(original)


def test_canonical_tool_parameter_pair_cannot_be_inferred_from_shared_name():
    registry = build_default_tool_registry()
    assert _parameter_input(registry, TOOL, 'min_tss_enrichment')[0] == 'min_tss_enrichment'
    for tool, name in [('cluster_cells', 'min_tss_enrichment'), (TOOL, 'resolution'),
                       ('invented_selection', 'min_tss_enrichment')]:
        with pytest.raises(IntentError):
            _parameter_input(registry, tool, name)


@pytest.mark.parametrize('received', [None, {}, {'resolution': .7}])
def test_mixed_historical_and_generic_declarations_round_trip_through_shared_record_validation(received):
    utterance = 'Use Leiden resolution 0.7 and UMAP minimum distance 0.2.'
    first = utterance.index('0.7')
    second = utterance.index('0.2')
    legacy = LeidenResolution('cluster_cells', 'resolution', '0.7', first, first + 3)
    generic = ScopedArgument('compute_cell_umap', 'min_dist', '0.2', second, second + 3)
    admitted = dict(kind='execute', operation='plan', tool='compute_cell_umap',
        inputs=dict(resolution=.7, min_dist=.2), argument=asdict(legacy), arguments=[asdict(generic)])
    if received is not None:
        admitted.update(argument_received_inputs_sha256=digest(received), arguments_received_inputs_sha256=digest(received),
            arguments_added_inputs=[name for name in ('resolution', 'min_dist') if name not in received])
    interaction = Interaction('mixed', utterance, None, 0, dict(relations={}, bases={}),
        status='admitted', submission=None if received is not None else dict(execution_inputs={}), admitted=admitted)
    _validate_resolution_record(interaction)
    _validate_argument_record(interaction)


def test_direct_session_retry_returns_original_parameter_clarification_without_provider_calls(
        tmp_path, selection_inputs):
    harness = selection_harness(tmp_path / 'application')
    sessions, model = direct_sessions(harness, tmp_path / 'direct')
    initial = sessions.respond('session', 'first', OBJECTIVE, interpreter=model,
        expected_generation=0, execution_inputs=selection_inputs)
    assert initial.clarification.reason == 'missing_parameter_value'
    counts = len(harness.calls), len(harness.constructions)
    retry = sessions.respond('session', 'first', OBJECTIVE, interpreter=model,
        expected_generation=0, execution_inputs=selection_inputs)
    assert retry.clarification == initial.clarification
    assert counts == (len(harness.calls), len(harness.constructions))


def test_direct_partial_retry_rejects_changed_received_values_without_science(
        tmp_path, selection_inputs):
    harness = selection_harness(tmp_path / 'application')
    sessions, model = direct_sessions(harness, tmp_path / 'direct')
    sessions.respond('session', 'first', OBJECTIVE, interpreter=model,
        expected_generation=0, execution_inputs=selection_inputs)
    partial = sessions.respond('session', 'partial', DEPTH, interpreter=model, expected_generation=0)
    assert partial.clarification.choices == (REQUIRED[1],)
    counts = len(harness.calls), len(harness.constructions)
    with selection_work() as calls, pytest.raises(SessionConflictError):
        sessions.respond('session', 'partial', DEPTH, interpreter=model,
            expected_generation=0, execution_inputs={REQUIRED[0]: 2})
    assert not calls and counts == (len(harness.calls), len(harness.constructions))


def test_pending_received_parameter_cannot_silently_change_or_disappear_without_scoped_answer(
        tmp_path, selection_inputs):
    harness = selection_harness(tmp_path / 'application')
    harness.service.create_session('session')
    submit(harness, inputs=selection_inputs)
    submit(harness, 'partial', DEPTH)
    with selection_work() as calls:
        conflict = submit(harness, 'conflict', TSS, inputs={REQUIRED[0]: 2})
    assert conflict['response']['clarification']['reason'] == 'conflicting_scientific_parameter'
    assert not calls
    with selection_work() as calls:
        completed = submit(harness, 'remaining', TSS)
    assert calls == {'production': 1, 'verification': 1}
    accepted_step(harness, completed, depth=1)


STRING_PAIRS = (
    ('evaluate_cell_clustering', 'label_key'),
    ('transfer_cell_labels', 'reference_label_key'),
    ('evaluate_cell_annotation', 'ground_truth_label_key'),
)


def string_declaration(tool, name, literal='Cell Type_α'):
    utterance = '🧬 Use the declared column "' + literal + '".'
    start = utterance.index(literal)
    return utterance, ScopedArgument(tool, name, literal, start, start + len(literal))


def registry_with_tool(registry, changed):
    return ToolRegistry(tuple(changed if name == changed.name else registry.get(name)
        for name in registry.names()))


@pytest.mark.parametrize('tool,name', STRING_PAIRS)
def test_string_binding_adds_only_the_declared_canonical_request_input(tool, name):
    registry = build_default_tool_registry()
    utterance, candidate = string_declaration(tool, name)
    received = dict(reference_h5ad_path='/trusted/reference.h5ad',
        query_h5ad_path='/trusted/query.h5ad', ground_truth_h5ad_path='/trusted/truth.h5ad',
        resource_context_sha256='a' * 64)
    before = dict(received)
    assert _bind_arguments(registry, utterance, received, (candidate,)) == received | {name: candidate.literal}
    assert received == before
    with pytest.raises(IntentError, match='conflicting_scientific_parameter'):
        _bind_arguments(registry, utterance, received | {name: 'Different Column'}, (candidate,))


@pytest.mark.parametrize('tool,name', STRING_PAIRS)
def test_string_declarations_round_trip_exact_source_attribution_without_science(tool, name):
    utterance, candidate = string_declaration(tool, name)
    received = dict(reference_h5ad_path='/trusted/reference.h5ad', resource_context_sha256='a' * 64)
    admitted = dict(kind='execute', operation='plan', tool=tool,
        inputs=received | {name: candidate.literal}, arguments=[asdict(candidate)])
    interaction = Interaction('string', utterance, None, 0, dict(relations={}, bases={}),
        status='admitted', submission=dict(execution_inputs=received), admitted=admitted)
    state = AnalysisSession('session', interactions=(interaction,))
    recovered = AnalysisSession.from_dict(json.loads(json.dumps(state.to_dict())))
    record, = recovered.interactions
    assert record.utterance == utterance
    assert dict(record.admitted['arguments'][0]) == asdict(candidate)
    assert record.admitted['inputs'][name] == candidate.literal
    _validate_argument_record(record)


@pytest.mark.parametrize('mutation', ['literal', 'case', 'span', 'operation', 'source', 'resource_context'])
def test_string_record_replay_rejects_changed_literal_identity_and_unrelated_source_inputs(mutation):
    tool, name = STRING_PAIRS[0]
    utterance, candidate = string_declaration(tool, name)
    received = dict(reference_h5ad_path='/trusted/reference.h5ad', resource_context_sha256='a' * 64)
    admitted = dict(kind='execute', operation='plan', tool=tool,
        inputs=received | {name: candidate.literal}, arguments=[asdict(candidate)])
    if mutation == 'literal':
        admitted['arguments'][0]['literal'] = 'Different Column'
    elif mutation == 'case':
        admitted['inputs'][name] = candidate.literal.lower()
    elif mutation == 'span':
        admitted['arguments'][0]['start'] += 1
    elif mutation == 'operation':
        admitted['arguments'][0]['tool'] = 'transfer_cell_labels'
    elif mutation == 'source':
        admitted['inputs']['reference_h5ad_path'] = '/unrelated/source.h5ad'
    else:
        admitted['inputs']['resource_context_sha256'] = 'b' * 64
    interaction = Interaction('string', utterance, None, 0, dict(relations={}, bases={}),
        status='admitted', submission=dict(execution_inputs=received), admitted=admitted)
    with pytest.raises(SessionError, match='Invalid persisted scientific arguments'):
        _validate_argument_record(interaction)


def test_string_binding_uses_the_registered_request_identity_instead_of_argument_name():
    registry = build_default_tool_registry()
    tool, name = STRING_PAIRS[0]
    spec = registry.get(tool)
    ports = tuple(replace(port, request_sources=(SemanticRequestSourceSpec('declared_truth_column',
        (SemanticRequestMember(port.members[0].name, 'declared_truth_column'),)),))
        if port.name == name else port for port in spec.semantic_planning.consumer_ports)
    registry = registry_with_tool(registry, replace(spec,
        semantic_planning=replace(spec.semantic_planning, consumer_ports=ports)))
    utterance, candidate = string_declaration(tool, name)
    assert _parameter_input(registry, tool, name)[0] == 'declared_truth_column'
    assert _bind_arguments(registry, utterance, {}, (candidate,)) == {'declared_truth_column': candidate.literal}


@pytest.mark.parametrize('mode', ['alternate_input', 'grouped_source', 'upstream_alternative'])
def test_string_binding_requires_one_singleton_direct_consumer_source(mode):
    registry = build_default_tool_registry()
    tool, name = STRING_PAIRS[0]
    spec = registry.get(tool)
    port, = [p for p in spec.semantic_planning.consumer_ports if p.name == name]
    if mode == 'alternate_input':
        changed = replace(port, request_sources=port.request_sources + (SemanticRequestSourceSpec(
            'alternate_truth_column', (SemanticRequestMember(port.members[0].name, 'alternate_truth_column'),)),))
    elif mode == 'grouped_source':
        changed = replace(port, members=port.members + (SemanticPortMember('destination', 'output_dir'),),
            request_sources=(SemanticRequestSourceSpec(name,
                (SemanticRequestMember(port.members[0].name, name), SemanticRequestMember('destination', 'output_dir'))),))
    else:
        changed = replace(port, accepted_upstream_types=('unreviewed-label-column.v1',))
    ports = tuple(changed if p.name == name else p for p in spec.semantic_planning.consumer_ports
        if mode != 'grouped_source' or p.name != 'output_dir')
    registry = registry_with_tool(registry, replace(spec,
        semantic_planning=replace(spec.semantic_planning, consumer_ports=ports)))
    with pytest.raises(IntentError):
        _parameter_input(registry, tool, name)


@pytest.mark.parametrize('role', [None, PlanningProvenanceRole.QUERY, PlanningProvenanceRole.BRANCH_PRESERVING])
def test_string_binding_cannot_treat_query_or_unspecified_roles_as_reference_labels(role):
    registry = build_default_tool_registry()
    tool, name = STRING_PAIRS[1]
    spec = registry.get(tool)
    arguments = dict(spec.required_arguments)
    argument_spec = arguments[name]
    arguments[name] = replace(argument_spec, planning=replace(argument_spec.planning, provenance_role=role))
    registry = registry_with_tool(registry, replace(spec, required_arguments=arguments))
    with pytest.raises(IntentError, match='unsupported_intent'):
        _parameter_input(registry, tool, name)


@pytest.mark.parametrize('tool,name', [
    ('transfer_cell_labels', 'label_key'),
    ('evaluate_cell_clustering', 'reference_label_key'),
    ('evaluate_cell_annotation', 'reference_label_key'),
    ('transfer_cell_labels', 'ground_truth_label_key'),
])
def test_reference_and_ground_truth_argument_identities_are_never_interchanged(tool, name):
    utterance, candidate = string_declaration(tool, name)
    with pytest.raises(IntentError, match='unsupported_intent'):
        _bind_arguments(build_default_tool_registry(), utterance, {}, (candidate,))


@pytest.mark.parametrize('tool,name', STRING_PAIRS)
@pytest.mark.parametrize('count', [0, 2])
def test_string_plan_declaration_never_chooses_first_or_infers_missing_consumer(tool, name, count):
    utterance, candidate = string_declaration(tool, name)
    steps = tuple(PlanStep('consumer_' + str(index), tool, {name: candidate.literal}) for index in range(count))
    plan = AgentPlan('plan', 'request', 'Evaluate the captured scientific result.', steps)
    admitted = dict(inputs={name: candidate.literal}, arguments=[asdict(candidate)])
    with pytest.raises(IntentError, match='ambiguous_parameter'):
        _check_parameter_plan(build_default_tool_registry(), plan, admitted, ())


@pytest.mark.parametrize('tool,name', STRING_PAIRS)
def test_string_plan_consumer_requires_the_exact_accepted_column(tool, name):
    utterance, candidate = string_declaration(tool, name)
    accepted = PlanStep('consumer', tool, {name: candidate.literal})
    plan = AgentPlan('plan', 'request', 'Evaluate the captured scientific result.', (accepted,))
    admitted = dict(inputs={name: candidate.literal}, arguments=[asdict(candidate)])
    registry = build_default_tool_registry()
    _check_parameter_plan(registry, plan, admitted, ())
    with pytest.raises(IntentError, match='planning_failed'):
        _check_parameter_plan(registry, replace(plan, steps=(replace(accepted,
            arguments={name: candidate.literal.lower()}),)), admitted, ())


def test_shared_string_request_input_cannot_fan_out_to_reference_and_ground_truth_consumers():
    registry = build_default_tool_registry()
    transfer = registry.get('transfer_cell_labels')
    ports = tuple(replace(port, request_sources=(SemanticRequestSourceSpec('label_key',
        (SemanticRequestMember(port.members[0].name, 'label_key'),)),))
        if port.name == 'reference_label_key' else port for port in transfer.semantic_planning.consumer_ports)
    registry = registry_with_tool(registry, replace(transfer,
        semantic_planning=replace(transfer.semantic_planning, consumer_ports=ports)))
    utterance, candidate = string_declaration('evaluate_cell_clustering', 'label_key')
    plan = AgentPlan('plan', 'request', 'Evaluate and transfer the captured scientific result.', (
        PlanStep('evaluate', 'evaluate_cell_clustering', {'label_key': candidate.literal}),
        PlanStep('transfer', 'transfer_cell_labels', {'reference_label_key': candidate.literal}),
    ))
    admitted = dict(inputs={'label_key': candidate.literal}, arguments=[asdict(candidate)])
    with pytest.raises(IntentError, match='ambiguous_parameter'):
        _check_parameter_plan(registry, plan, admitted, ())
