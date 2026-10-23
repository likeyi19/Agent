"""Offline interpretation diagnostics; no providers or scientific execution."""
import hashlib
import json
from uuid import UUID

import pytest

from agent.application.service import ResearchAgentApplication
from agent.application.interactive import InteractiveAgentApplication
from agent.application import turns, turn_decisions
from agent.application.turn_decisions import Clarify, interpret
from agent.orchestration.planning_model import PlanningModelError, PlanningModelProfile
from agent.orchestration.error_policy import safe_message_for
from agent.providers import PlanningModelFactoryRegistry

SID = str(UUID(int=1))
TID = str(UUID(int=2))
SECRET = 'Bearer sk-private-payload /private/scientific-input.h5ad'


class Model:
    def __init__(self, decision=None, *, error=None, raw=None):
        self.model_id = 'scripted:test/model'
        self.raw = raw if raw is not None else json.dumps(dict(turn_schema_version=1,
            decision=decision or dict(kind='clarify', reason='invalid_decision')))
        self.error, self.calls = error, []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.raw


def records(caplog):
    return [json.loads(r.getMessage().split('Agent operation diagnostic ', 1)[1])
            for r in caplog.records if r.msg == 'Agent operation diagnostic %s']


def invoke(tmp_path, model, **kwargs):
    app = ResearchAgentApplication(tmp_path / 'workspace')
    app.sessions.create(SID)
    result = app.sessions.respond(SID, TID, 'Please inspect the supplied data.',
        interpreter=model, expected_generation=0, **kwargs)
    state = app.sessions.load(SID)
    assert not state.turns and not state.revisions
    return result, state


@pytest.mark.parametrize('reason', ['invalid_decision', 'ambiguous_subject'])
def test_model_clarification_is_successfully_parsed_without_exception(tmp_path, caplog, reason):
    result, state = invoke(tmp_path, Model(dict(kind='clarify', reason=reason)))
    assert result.clarification == Clarify(reason)
    assert state.interactions[0].admitted == dict(kind='clarify', reason=reason, choices=(), value_required=False)
    rows = records(caplog)
    selected = next(r for r in rows if r['event'] == 'interpreter.parse_completed')
    assert selected['kind'] == 'Clarify'
    assert (selected['session_id'], selected['turn_id'], selected['reason']) == (SID, TID, reason)
    assert not any('exception_class' in r for r in rows)
    assert not any('diagnostic' in k for k in state.to_dict()['interactions'][0])


def test_parser_rejection_remains_semantic_clarification(tmp_path, caplog):
    result, state = invoke(tmp_path, Model(raw='{' + SECRET))
    assert result.clarification == Clarify('invalid_decision')
    assert result.text == ('I could not interpret or admit this request reliably. '
                           'Please restate the task and identify the data or result you mean.')
    rows = records(caplog)
    failed = next(r for r in rows if r['event'] == 'turn.boundary_failed')
    assert failed['exception_class'] == 'IntentError'
    assert failed['cause_class'] == 'JSONDecodeError'
    assert failed['stage'] == 'interpretation'
    assert not any(r['event'] == 'interpreter.parse_completed' for r in rows)
    assert SECRET not in caplog.text and '/private/supplied.h5ad' not in caplog.text
    assert 'Traceback' not in caplog.text


@pytest.mark.parametrize('code', ['PROVIDER_RATE_LIMITED', 'PROVIDER_TIMEOUT', 'PLANNING_PROVIDER_ERROR', SECRET])
def test_provider_failure_is_failed_interaction_with_safe_typed_error(tmp_path, caplog, code):
    model = Model(error=PlanningModelError(SECRET, code=code))
    result, state = invoke(tmp_path, model)
    expected = code if code != SECRET else 'PLANNING_PROVIDER_ERROR'
    assert result.kind == 'execute' and result.status == 'failed'
    assert result.clarification is None
    assert result.error.code == expected and result.text == safe_message_for(expected)
    interaction = state.interactions[0]
    assert interaction.status == 'failed' and interaction.admitted is None and interaction.prerequisite is None
    assert len(model.calls) == 1
    failed = next(r for r in records(caplog) if r['event'] == 'turn.boundary_failed')
    assert failed['exception_class'] == 'PlanningModelError' and failed['stage'] == 'interpretation'
    assert not any(r['event'] == 'interpreter.parse_completed' for r in records(caplog))
    assert SECRET not in result.text and SECRET not in caplog.text


@pytest.mark.parametrize('stage', ['preparation', 'admission'])
def test_unexpected_application_error_marks_failed_and_propagates(tmp_path, monkeypatch, caplog, stage):
    app = ResearchAgentApplication(tmp_path / 'workspace')
    app.sessions.create(SID)
    model = Model(dict(kind='execute_plan', target='inspect_scATAC'))
    def fail(*args, **kwargs):
        raise RuntimeError(SECRET)
    if stage == 'preparation':
        monkeypatch.setattr(turns, 'public_context', fail)
    else:
        from agent.application import dialogue_execution
        monkeypatch.setattr(dialogue_execution, 'admit', fail)
    with pytest.raises(RuntimeError, match='Bearer'):
        app.sessions.respond(SID, TID, 'Please inspect the supplied data.', interpreter=model,
            expected_generation=0, execution_inputs={'input_path': '/private/supplied.h5ad'})
    state = app.sessions.load(SID)
    assert state.interactions[0].status == 'failed' and state.interactions[0].admitted is None
    assert not state.turns and not state.revisions
    assert len(model.calls) == (stage == 'admission')
    failed = next(r for r in records(caplog) if r['event'] == 'turn.boundary_failed')
    assert failed['stage'] == stage and failed['exception_class'] == 'RuntimeError'
    assert SECRET not in caplog.text


@pytest.mark.parametrize('case', ['provider', 'model_clarification', 'malformed', 'application'])
def test_interactive_failure_and_clarification_survive_retry_and_reopen_without_replay(
        tmp_path, monkeypatch, case):
    from agent.orchestration import PlanExecutor
    def forbidden(*args, **kwargs):
        pytest.fail('Scientific execution or provider reconstruction during history read')
    monkeypatch.setattr(PlanExecutor, 'execute', forbidden)
    model = Model(dict(kind='clarify', reason='ambiguous_subject'))
    if case == 'provider':
        model = Model(error=PlanningModelError(SECRET, code='PROVIDER_RATE_LIMITED'))
    elif case == 'malformed':
        model = Model(raw='{' + SECRET)
    elif case == 'application':
        def fail(*args, **kwargs):
            raise RuntimeError(SECRET)
        monkeypatch.setattr(turns, 'public_context', fail)
    profile = PlanningModelProfile('offline', 'scripted', 'test/model')
    factories = []
    def factory(selected):
        factories.append(selected)
        return model
    workspace = tmp_path / 'workspace'
    app = InteractiveAgentApplication(workspace, model_profiles=(profile,), default_profile_id='offline',
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': factory}))
    app.create_session(SID)
    request = dict(expected_generation=0)
    result = app.submit_turn(SID, TID, 'Please inspect the supplied data.', **request)
    state = app._application.sessions.load(SID)
    assert not state.turns and not state.revisions
    assert state.generation == 0 and state.active_revision_id is None
    if case in {'provider', 'application'}:
        assert result.status == 'failed' and result.response.status == 'failed'
        assert result.response.clarification is None and state.interactions[0].admitted is None
        code = 'PROVIDER_RATE_LIMITED' if case == 'provider' else 'INTERACTIVE_APPLICATION_FAILED'
        assert result.error.code == code and result.response.error.code == code
        assert state.interactions[0].status == 'failed'
    else:
        assert result.status == 'clarification' and result.error is None
        reason = 'ambiguous_subject' if case == 'model_clarification' else 'invalid_decision'
        assert result.response.clarification['reason'] == reason
        assert result.response.kind == 'clarify' and result.response.error is None
    assert SECRET not in json.dumps(result.to_dict())
    assert app.submit_turn(SID, TID, 'Please inspect the supplied data.', **request) == result
    reopened = InteractiveAgentApplication(workspace, model_profiles=(profile,), default_profile_id='offline',
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': forbidden}))
    assert reopened.status(SID, TID) == result
    assert reopened.submit_turn(SID, TID, 'Please inspect the supplied data.', **request) == result
    assert len(factories) == 1 and len(model.calls) == (case != 'application')
    assert reopened._application.sessions.load(SID) == state


def test_interpreter_preparation_failure_does_not_call_model(caplog):
    model = Model()
    with pytest.raises(KeyError):
        interpret(model, SECRET, {}, diagnostic_context=(SID, TID))
    assert model.calls == []
    assert not records(caplog)
    assert SECRET not in caplog.text


def test_successful_read_only_admission_preserves_result(tmp_path, caplog):
    model = Model(dict(kind='answer', intent='unsupported', relation='current', technical=False))
    result, state = invoke(tmp_path, model)
    assert result.kind == 'answer' and result.status == 'unsupported'
    assert state.interactions[0].status == 'answered'
    assert len(model.calls) == 1
    assert next(r for r in records(caplog) if r['event'] == 'interpreter.parse_completed')['kind'] == 'Answer'
    assert not any('exception_class' in r for r in records(caplog))


def test_logging_failure_and_fingerprints_preserve_original_model_contract(monkeypatch, caplog):
    public = dict(bases={}, relations=['current'], dialogue=dict(outputs=[], tools=['inspect_scATAC']))
    model = Model()
    assert interpret(model, SECRET, public, diagnostic_context=(SID, TID)) == Clarify('invalid_decision')
    payload = model.calls[0]
    prepared = next(r for r in records(caplog) if r['event'] == 'interpreter.preparation_completed')
    assert prepared['prompt_sha256'] == hashlib.sha256(payload['prompt'].encode()).hexdigest()
    assert SECRET not in caplog.text
    def fail(*args, **kwargs):
        raise RuntimeError(SECRET)
    monkeypatch.setattr(turn_decisions.logger, 'warning', fail)
    model2 = Model()
    assert interpret(model2, SECRET, public, diagnostic_context=(SID, TID)) == Clarify('invalid_decision')
    assert model.calls == model2.calls
