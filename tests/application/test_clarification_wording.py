"""Generic current clarification text and immutable historical presentation."""
import pytest

from agent.application import responses
from test_interactive_boundary import boundary


@pytest.mark.parametrize('reason', ['invalid_parameter_value', 'ungrounded_operand'])
def test_fresh_invalid_value_does_not_assert_accepted_values_or_pending_task(tmp_path, reason):
    service, models = boundary(tmp_path, decision=dict(kind='clarify', reason=reason))
    service.create_session('session')
    view = service.submit_turn('session', 'first',
        'Cluster these cells using Leiden with resolution -0.7.', expected_generation=0,
        execution_inputs=dict(analysis_path='/supplied/neighbors.h5ad'))

    state = service._application.sessions.load('session')
    assert view.response.clarification['reason'] == reason
    assert view.response.clarification['value_required']
    assert 'valid explicit value' in view.response.text
    assert 'previously accepted' not in view.response.text.lower()
    assert 'pending' not in view.response.text.lower()
    assert not state.revisions and state.generation == 0 and view.run_id is None
    assert state.interactions[0].prerequisite is None
    assert state.interactions[0].submission['execution_inputs'] == dict(
        analysis_path='/supplied/neighbors.h5ad')
    assert len(models) == 1 and len(models[0].calls) == 1
    assert 'turn_schema_version' in models[0].calls[0]


def test_unsupported_computation_uses_generic_intent_wording(tmp_path):
    service, models = boundary(tmp_path,
        decision=dict(kind='clarify', reason='unsupported_intent'))
    service.create_session('session')
    view = service.submit_turn('session', 'first',
        'Run two independent Leiden clusterings on these cells; use resolution 0.7.',
        expected_generation=0, execution_inputs=dict(analysis_path='/supplied/neighbors.h5ad'))

    state = service._application.sessions.load('session')
    assert view.response.clarification['reason'] == 'unsupported_intent'
    assert not view.response.clarification['value_required']
    assert 'not supported' in view.response.text
    assert 'biological' not in view.response.text.lower()
    assert 'marker' not in view.response.text.lower()
    assert not state.revisions and state.generation == 0 and view.run_id is None
    assert state.interactions[0].prerequisite is None
    assert len(models) == 1 and len(models[0].calls) == 1
    # Existing bounded unsupported-answer wording remains specific to answers.
    assert 'biological or marker-based explanations' in responses.UNSUPPORTED


def test_old_clarification_stays_exact_on_reopen_retry_and_new_turn(tmp_path, monkeypatch):
    decision = dict(kind='clarify', reason='invalid_parameter_value')
    service, models = boundary(tmp_path, decision=decision)
    service.create_session('session')
    utterance = 'Cluster these cells using Leiden with resolution -0.7.'
    old_text = ('Please provide a valid explicit value for the scientific parameter. '
                'Previously accepted values are retained for the pending request.')
    with monkeypatch.context() as historical_renderer:
        historical_renderer.setattr(responses, 'clarification_text', lambda _: old_text)
        old_view = service.submit_turn('session', 'first', utterance, expected_generation=0)
    sessions = service._application.sessions
    path = sessions._store._path('session', '.json')
    old_bytes = path.read_bytes()
    old_interaction = sessions.load('session').interactions[0]

    reopened, reopened_models = boundary(tmp_path, decision=decision,
        workspace=service._application.workspace_root)
    assert reopened.reopen_session('session').turns[0] == old_view
    assert reopened.submit_turn('session', 'first', utterance, expected_generation=0) == old_view
    assert path.read_bytes() == old_bytes and not reopened_models
    new_view = reopened.submit_turn('session', 'second', utterance, expected_generation=0)
    assert 'pending' not in new_view.response.text.lower()
    assert new_view.response.clarification == old_view.response.clarification
    assert reopened._application.sessions.load('session').interactions[0] == old_interaction
    assert old_view.response.text == old_text
    assert len(models) == 1 and len(models[0].calls) == 1
