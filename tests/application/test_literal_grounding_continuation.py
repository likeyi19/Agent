"""Mechanical quote ambiguity retains one exact objective without running it."""
import json
import pytest

from test_scientific_parameter_continuation import (
    TOOL, REQUIRED, OBJECTIVE, TSS, argument, assert_pending, accepted_step,
    selection_harness, selection_inputs, selection_work, submit,
    runtime, fixture_factory, qc_case,
)


def test_ambiguous_literal_survives_restart_and_explicit_correction(tmp_path, selection_inputs):
    text = OBJECTIVE + ' Use depth 1 and TSS 0.5; the note also says 0.5.'
    decision = dict(kind='execute_plan', target=TOOL, arguments=[
        dict(tool=TOOL, argument=REQUIRED[0], literal='1', evidence='depth 1'),
        dict(tool=TOOL, argument=REQUIRED[1], literal='0.5', evidence='0.5')])
    root = tmp_path / 'application'
    harness = selection_harness(root, decisions={text: decision})
    harness.service.create_session('session')
    with selection_work() as calls:
        result = submit(harness, utterance=text, inputs=selection_inputs)
    assert result['response']['clarification']['reason'] == 'ambiguous_parameter', result
    assert not calls and not harness.service._application.sessions.load('session').revisions
    interaction, = harness.service._application.sessions.load('session').interactions
    assert interaction.prerequisite['missing'] == (REQUIRED[1],)
    assert interaction.admitted['inputs'][REQUIRED[0]] == 1
    assert REQUIRED[1] not in interaction.admitted['inputs']
    assert not list((root / 'runs').glob('*/record.json'))
    restarted = selection_harness(root)
    assert restarted.service.turn('session', 'first').to_dict() == result
    assert not restarted.calls
    with selection_work() as calls:
        completed = submit(restarted, 'corrected', TSS)
    accepted_step(restarted, completed)
    assert calls == {'production': 1, 'verification': 1}
    run = restarted.service._application.run_store.load(completed['run_id'])
    assert run.request.prompt == text
    state = restarted.service._application.sessions.load('session')
    declaration, = state.interactions[-1].admitted['arguments']
    assert TSS[declaration['start']:declaration['end']] == '0.5'


def test_duplicate_identity_is_not_salvaged_into_pending(tmp_path, selection_inputs):
    text = OBJECTIVE + ' Use TSS 0.5 and TSS 0.5.'
    item = dict(tool=TOOL, argument=REQUIRED[1], literal='0.5', evidence='0.5')
    harness = selection_harness(tmp_path / 'application', decisions={text:
        dict(kind='execute_plan', target=TOOL, arguments=[item, item])})
    harness.service.create_session('session')
    with selection_work() as calls:
        result = submit(harness, utterance=text, inputs=selection_inputs)
    assert result['status'] == 'failed'
    assert result['response']['error']['code'] == 'INTERACTIVE_INTENT_BINDING_INVALID'
    assert result['response']['clarification'] is None
    state = harness.service._application.sessions.load('session')
    interaction, = state.interactions
    assert interaction.admitted is None and interaction.prerequisite is None
    assert not calls and not state.turns and not state.revisions
    assert not list(harness.service._application.run_store.root.glob('*.json'))


def test_invalid_numeric_declaration_can_be_corrected_without_repeating_goal(tmp_path, selection_inputs):
    text = OBJECTIVE + ' Use depth 1 and TSS -1.'
    harness = selection_harness(tmp_path / 'application', decisions={text:
        dict(kind='execute_plan', target=TOOL, arguments=[
            argument(text, REQUIRED[0], '1'), argument(text, REQUIRED[1], '-1')])})
    harness.service.create_session('session')
    with selection_work() as calls:
        result = submit(harness, utterance=text, inputs=selection_inputs)
    assert result['response']['clarification']['reason'] == 'invalid_parameter_value'
    assert not calls
    assert harness.service._application.sessions.load('session').interactions[0].prerequisite
    completed = submit(harness, 'corrected', TSS)
    accepted_step(harness, completed)
    assert harness.service._application.run_store.load(completed['run_id']).request.prompt == text


@pytest.mark.parametrize('field', ['inputs', 'parameter_rejection', 'missing'])
def test_rejected_declaration_capture_cannot_be_forged(tmp_path, selection_inputs, field):
    from test_scientific_binding_safety import mutated_record
    from agent.application.session_state import SessionError
    text = OBJECTIVE + ' Use depth 1 and TSS -1.'
    harness = selection_harness(tmp_path / 'application', decisions={text:
        dict(kind='execute_plan', target=TOOL, arguments=[
            argument(text, REQUIRED[0], '1'), argument(text, REQUIRED[1], '-1')])})
    harness.service.create_session('session')
    submit(harness, utterance=text, inputs=selection_inputs)
    def mutate(record):
        interaction = record['interactions'][0]
        if field == 'inputs':
            interaction['admitted']['inputs'][REQUIRED[1]] = 5
        elif field == 'parameter_rejection':
            interaction['admitted'][field] = '{}'
        else:
            interaction['prerequisite']['missing'] = ['min_tss_flank_evidence']
    with mutated_record(harness, mutate) as sessions, pytest.raises(SessionError):
        sessions.load('session')


def test_grounding_failure_cannot_bypass_guidance_candidate_guard(tmp_path, selection_inputs, monkeypatch):
    text = OBJECTIVE + ' Use TSS 0.5; note 0.5.'
    harness = selection_harness(tmp_path / 'application', decisions={text:
        dict(kind='execute_plan', target=TOOL, arguments=[dict(tool=TOOL,
            argument=REQUIRED[1], literal='0.5', evidence='0.5')])})
    harness.service.create_session('session')
    seen = []
    def candidate_reference(*args):
        seen.append(True)
        return True
    monkeypatch.setattr('agent.application.scientific_guidance.references_candidate', candidate_reference)
    result = submit(harness, utterance=text, inputs=selection_inputs)
    assert seen and result['revision_id'] is None
    assert harness.service._application.sessions.load('session').interactions[0].prerequisite is None


def test_exact_string_ambiguity_retains_original_evaluation(tmp_path, string_inputs):
    from test_string_scientific_execution import (
        CLUSTER_TOOL, CLUSTER_KEY, STRING_LITERAL, STRING_ANSWER,
        string_harness, submit_string, accepted_evaluation,
    )
    text = f'Evaluate clustering with truth column “{STRING_LITERAL}”; the note repeats “{STRING_LITERAL}”.'
    harness = string_harness(tmp_path / 'string', decisions={text:
        dict(kind='execute_plan', target=CLUSTER_TOOL, arguments=[dict(tool=CLUSTER_TOOL,
            argument=CLUSTER_KEY, literal=STRING_LITERAL, evidence=STRING_LITERAL)])})
    harness.service.create_session('session')
    pending = submit_string(harness, utterance=text, inputs=string_inputs)
    assert pending['response']['clarification']['reason'] == 'ambiguous_parameter'
    assert harness.service._application.sessions.load('session').interactions[0].prerequisite
    done = submit_string(harness, 'corrected', STRING_ANSWER)
    accepted_evaluation(harness, done, string_inputs)
    assert harness.service._application.run_store.load(done['run_id']).request.prompt == text


from test_string_scientific_execution import string_inputs
