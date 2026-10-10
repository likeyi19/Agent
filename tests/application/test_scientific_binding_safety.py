"""Fail-closed persistence and exact-source checks for shared numeric bindings."""
from contextlib import contextmanager
from dataclasses import asdict
import json
from pathlib import Path

import pytest

from agent.application import ResearchAgentApplication
from agent.application.dialogue_execution import (
    _parameter_input, _validate_argument_record, _validate_resolution_record,
)
from agent.application.session_state import Interaction, SessionConflictError, SessionError, digest
from agent.application.turn_decisions import IntentError, LeidenResolution, ScopedArgument
from agent.orchestration import build_default_tool_registry

from test_scientific_parameter_continuation import (
    BOTH, DEPTH, OBJECTIVE, PROFILE, REQUIRED, TOOL, TSS, SelectionModel,
    accepted_step, answer, argument, assert_pending, fixture_factory, qc_case,
    runtime, selection_harness, selection_inputs, selection_work, submit,
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
