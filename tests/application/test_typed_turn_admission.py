"""Typed generic decisions own intent; admission checks contracts before planning.

Scripted Execute decisions deliberately include neutral text and a result question.
These cases qualify removal of the second lexical veto, not interpreter quality.
Execution stops at the admitted request boundary; no scientific result is claimed.
"""
from dataclasses import replace

import pytest

from agent.application import SessionConflictError
from agent.application import dialogue_execution
from agent.application.session_state import Interaction, digest
from agent.application.turn_decisions import Execute, IntentDelta, IntentError
from agent.application.turns import TurnOutcome
from agent.orchestration import LLMPlanner
from agent.orchestration.executor import PlanExecutor
from agent.orchestration import semantic_compiler
from agent.schemas.orchestration import _serialize
from test_dialogue_evidence import app, accepted, forbid_work
from test_guidance_selection import Selection, offer
from test_scientific_dialogue import Model, ask, question


R2 = (
    'Plan and execute an inspection of the selected scATAC-seq input. '
    'Use the selected input as the dataset to inspect. '
    'Do not answer from prior scientific evidence. '
    'After the inspection executes successfully, summarize the verified dataset structure and contents.'
)
WORDINGS = (
    'Inspect the selected dataset.',
    'Run an inspection.',
    'Execute an inspection.',
    'Plan and execute an inspection.',
    'Please inspect the supplied data.',
    'Here is the selected dataset.',
    'Explain my previous QC result.',
    R2,
)


def stop_before_planning(app, monkeypatch):
    app = forbid_work(app, monkeypatch)
    reached, forbidden_calls = [], []

    def forbidden(*args, **kwargs):
        forbidden_calls.append(True)
        raise AssertionError('Admission entered planning, compilation or preflight')

    for name in ('plan', 'plan_with_diagnostics', 'plan_with_recovery', '_select_scope', '_plan_detailed'):
        monkeypatch.setattr(LLMPlanner, name, forbidden)
    monkeypatch.setattr(semantic_compiler, 'compile_semantic_plan', forbidden)
    monkeypatch.setattr(PlanExecutor, 'preflight', forbidden)

    def stop(sessions, interaction, admitted, model, *, epizoo_resources=()):
        reached.append((interaction, _serialize(admitted)))
        return TurnOutcome('execute', 'admitted')

    monkeypatch.setattr(dialogue_execution, 'execute', stop)
    return app, reached, forbidden_calls


@pytest.mark.parametrize('utterance', WORDINGS)
@pytest.mark.parametrize('minimal', [False, True])
def test_same_typed_execute_is_admitted_without_reclassifying_language(app, monkeypatch, utterance, minimal):
    app, reached, forbidden_calls = stop_before_planning(app, monkeypatch)
    inputs = {'input_path': '/operator-owned/provided-matrix.h5ad'}
    choice = dict(kind='execute_plan', target='inspect_scATAC') if minimal else dict(
        kind='execute', base='current', operation='plan', target='inspect_scATAC', delta=None)
    model = Model(choice)

    outcome = ask(app, 'first', utterance, model, expected_generation=0, execution_inputs=inputs)

    assert (outcome.kind, outcome.status) == ('execute', 'admitted')
    assert len(model.calls) == 1 and model.calls[0]['utterance'] == utterance
    interaction, admitted = reached.pop()
    assert admitted == dict(kind='execute', operation='plan', tool='inspect_scATAC', revision_id=None,
        request_id='turn-' + digest(dict(session='session', turn='first')), inputs=inputs)
    assert interaction.base_revision_id is None and interaction.base_generation == 0
    assert interaction.snapshot['relations'] == {} and interaction.snapshot['bases'] == {}
    assert interaction.snapshot['dialogue']['outputs'] == ()
    state = app.sessions.load('session')
    assert state.generation == 0 and state.active_revision_id is None
    assert not state.revisions and not state.turns and not reached and not forbidden_calls
    assert state.interactions[0].admitted == admitted
    assert state.interactions[0].status == 'admitted'
    assert not tuple(app.run_store.root.glob('*.json'))


@pytest.mark.parametrize('choice,kind,status', [
    (dict(kind='clarify', reason='unavailable_context'), 'clarify', 'clarification'),
    (dict(kind='answer', intent='unsupported', relation='current', technical=False), 'answer', 'unsupported'),
])
def test_execution_words_do_not_override_a_nonexecuting_typed_decision(app, monkeypatch, choice, kind, status):
    app, reached, forbidden_calls = stop_before_planning(app, monkeypatch)
    model = Model(choice)

    outcome = ask(app, 'first', R2, model, execution_inputs={'input_path': '/operator-owned/provided-matrix.h5ad'})

    assert (outcome.kind, outcome.status) == (kind, status)
    assert len(model.calls) == 1 and not reached and not forbidden_calls
    state = app.sessions.load('session')
    assert state.generation == 0 and not state.revisions
    assert not tuple(app.run_store.root.glob('*.json'))


@pytest.mark.parametrize('changes', [
    dict(target='unregistered_tool'),
    dict(base='parent'),
    dict(delta=IntentDelta('min_tss_enrichment', 'set', '5', 'Set TSS enrichment to 5.')),
])
def test_generic_admission_still_rejects_invalid_typed_contracts(app, changes):
    sessions = app.sessions
    sessions._interaction_session_id = 'session'
    interaction = Interaction('first', R2, None, 0, dict(relations={}, bases={}))
    decision = replace(Execute('current', 'plan', 'inspect_scATAC', None), **changes)

    with pytest.raises(IntentError, match='unsupported_intent'):
        dialogue_execution.admit(sessions, interaction, decision, {'input_path': '/provided.h5ad'})

    assert not app.sessions.load('session').interactions


@pytest.mark.parametrize('inputs', [[], 'input.h5ad', {1: 'value'}, {'input_path': object()}, {'x': float('nan')}])
def test_generic_admission_preserves_existing_json_mapping_validation(app, inputs):
    sessions = app.sessions
    sessions._interaction_session_id = 'session'
    interaction = Interaction('first', R2, None, 0, dict(relations={}, bases={}))

    with pytest.raises((TypeError, ValueError)):
        dialogue_execution.admit(sessions, interaction, Execute('current', 'plan', 'inspect_scATAC', None), inputs)

    assert not app.sessions.load('session').interactions


@pytest.mark.parametrize('inputs', [None, {'input_path': '/provided.h5ad', 'parameters': {'values': [1, None]}}])
def test_generic_admission_preserves_exact_inputs_without_inference(app, inputs):
    sessions = app.sessions
    sessions._interaction_session_id = 'session'
    interaction = Interaction('first', 'Here is the selected dataset.', None, 0, dict(relations={}, bases={}))

    admitted = dialogue_execution.admit(sessions, interaction, Execute('current', 'plan', 'inspect_scATAC', None), inputs)

    assert admitted['inputs'] == ({} if inputs is None else inputs)
    if inputs is not None:
        inputs['parameters']['values'].append(2)
        assert admitted['inputs']['parameters']['values'] == [1, None]
    assert admitted['revision_id'] is None and not app.sessions.load('session').interactions


@pytest.mark.parametrize('has_revision', [False, True])
def test_stale_generation_rejects_before_interpretation_or_admission(app, monkeypatch, has_revision):
    if has_revision:
        accepted(app, 'inspect_scATAC')
    app, reached, forbidden_calls = stop_before_planning(app, monkeypatch)
    before = app.sessions.load('session')
    model = Model(dict(kind='execute_plan', target='inspect_scATAC'))

    with pytest.raises(SessionConflictError, match='generation'):
        ask(app, 'first', R2, model, expected_generation=before.generation + 1,
            execution_inputs={'input_path': '/provided.h5ad'})

    assert not model.calls and not reached and not forbidden_calls
    assert app.sessions.load('session') == before


@pytest.mark.parametrize('utterance', ['Run option 1.', 'Execute option 1.'])
def test_generic_execute_cannot_bypass_existing_candidate_selection(app, monkeypatch, utterance):
    accepted(app, 'inspect_scATAC')
    offer(app)
    assert ask(app, 'discussion', 'What does this show?', Model(question('r0'))).status == 'answered'
    app, reached, forbidden_calls = stop_before_planning(app, monkeypatch)
    before = app.sessions.load('session')
    model = Model(dict(kind='execute_plan', target='inspect_scATAC'))

    outcome = ask(app, 'select', utterance, model, execution_inputs={'input_path': '/provided.h5ad'})

    assert outcome.status == 'clarification' and outcome.clarification.reason == 'unsupported_intent'
    assert len(model.calls) == 1 and not reached and not forbidden_calls
    state = app.sessions.load('session')
    assert state.revisions == before.revisions and state.generation == before.generation
    assert state.interactions[-1].admitted['kind'] == 'clarify'


@pytest.mark.parametrize('utterance', ['Run option 1.', 'Execute option 1.'])
def test_valid_execute_candidate_retains_existing_admission_route(app, monkeypatch, utterance):
    revision_id, _, _ = accepted(app, 'inspect_scATAC')
    candidate_id = offer(app)
    app, reached, forbidden_calls = stop_before_planning(app, monkeypatch)
    before = app.sessions.load('session')
    model = Selection(candidate_id, utterance)
    inputs = {'input_path': '/provided.h5ad'}

    outcome = ask(app, 'select', utterance, model, execution_inputs=inputs)

    assert (outcome.kind, outcome.status) == ('execute', 'admitted')
    assert len(model.calls) == 1 and not forbidden_calls
    _, admitted = reached.pop()
    assert admitted['operation'] == 'plan' and admitted['tool'] == 'inspect_scATAC'
    assert admitted['selected_candidate']['candidate_id'] == candidate_id
    assert admitted['command_evidence'] == utterance and admitted['inputs'] == inputs
    assert admitted['revision_id'] == revision_id
    state = app.sessions.load('session')
    assert state.revisions == before.revisions and state.generation == before.generation
    assert state.turns == before.turns and state.interactions[:-1] == before.interactions
    assert _serialize(state.interactions[-1].admitted) == admitted and not reached


@pytest.mark.parametrize('utterance', ['That sounds interesting.', 'Do not run option 1.', 'If ready, run option 1.'])
def test_candidate_command_evidence_keeps_existing_m17_2_rejections(app, monkeypatch, utterance):
    accepted(app, 'inspect_scATAC')
    candidate_id = offer(app)
    app, reached, forbidden_calls = stop_before_planning(app, monkeypatch)
    before = app.sessions.load('session')

    outcome = ask(app, 'select', utterance, Selection(candidate_id, utterance))

    assert outcome.status == 'clarification' and not reached and not forbidden_calls
    state = app.sessions.load('session')
    assert state.revisions == before.revisions and state.generation == before.generation
