"""M18.2 session-owned empty capture and immutable presentation contracts."""
from dataclasses import replace
import json
import os
import subprocess
import sys

import pytest

from agent.application import ResearchAgentApplication, SessionConflictError, SessionError
from agent.application.session_state import AnalysisSession, Interaction, canonical
from agent.orchestration import LLMPlanner
from test_service import _ScriptedPlanningModel, _semantic_planning_response, _tiny_h5ad, _counting_registry
from test_scientific_dialogue import Model


def presentation(outcome):
    return dict(version=1, kind=outcome.kind, status=outcome.status, text=outcome.text,
                clarification=None, scientific=None, guidance=None, error=None)


def test_empty_first_request_uses_interpreter_planner_compiler_and_runtime(tmp_path):
    source = _tiny_h5ad(tmp_path / 'tiny.h5ad')
    model = _ScriptedPlanningModel(
        json.dumps(dict(turn_schema_version=1, decision=dict(kind='execute_plan', target='inspect_scATAC'))),
        _semantic_planning_response(),
        json.dumps(dict(outputs=[dict(name='inspection', step_id='inspect', output_key='n_cells')])),
    )
    calls = []
    application = ResearchAgentApplication(tmp_path / 'workspace', planner=LLMPlanner(model),
                                          registry=_counting_registry(calls))
    application.sessions.create('empty')
    submission = dict(request=dict(inputs={'input_path': str(source)}), profile_id='scripted')
    outcome = application.sessions.respond('empty', 'first', 'Inspect the supplied matrix.',
        interpreter=model, expected_generation=0, execution_inputs={'input_path': str(source)},
        submission=submission)
    assert (outcome.kind, outcome.status) == ('execute', 'activated')
    assert calls == ['inspect_scATAC'] and model.calls == 3 and len(model.scope_calls) == 1
    state = application.sessions.load('empty')
    interaction, = state.interactions
    revision, = state.revisions
    assert interaction.base_revision_id is None and interaction.base_generation == 0
    assert interaction.snapshot['relations'] == {} and interaction.snapshot['bases'] == {}
    assert interaction.snapshot['dialogue']['outputs'] == ()
    assert revision.parent_revision_id is None and state.generation == 1
    assert application.sessions.evidence('empty', revision.revision_id, 'inspection').status == 'available'
    view = presentation(outcome)
    application.sessions.store_presentation('empty', 'first', view)
    restored = ResearchAgentApplication(application.workspace_root).sessions.load('empty')
    assert restored.interactions[0].presentation == view
    retry = application.sessions.respond('empty', 'first', 'Inspect the supplied matrix.',
        interpreter=model, expected_generation=0, execution_inputs={'input_path': str(source)},
        submission=submission)
    assert retry.status == 'activated' and calls == ['inspect_scATAC'] and model.calls == 3


@pytest.mark.parametrize('decision,expected', [
    (dict(kind='clarify', reason='missing_parameter_value'), 'missing_parameter_value'),
    (dict(kind='navigate', relation='current'), 'unavailable_context'),
    (dict(kind='execute', base='current', operation='op.0', target='selection', delta=None), 'unavailable_context'),
    (dict(kind='answer', intent='version', relation='current', technical=False), 'unavailable_context'),
    (dict(kind='answer_scientific', target=dict(output='@focus', subject=None), comparison=None, focus='question'), 'unavailable_context'),
    (dict(kind='answer_guidance', targets=[], candidate=None), 'unavailable_context'),
])
def test_empty_nonexecuting_context_is_persisted_without_revision(tmp_path, decision, expected):
    application = ResearchAgentApplication(tmp_path / 'workspace')
    application.sessions.create('empty')
    model = Model(decision)
    outcome = application.sessions.respond('empty', 'question', 'What can this result show?',
                                          interpreter=model, expected_generation=0)
    assert outcome.kind == 'clarify' and outcome.clarification.reason == expected
    state = application.sessions.load('empty')
    assert state.generation == 0 and state.active_revision_id is None and not state.revisions
    assert state.turn('question').status == 'clarification'
    assert state.interactions[0].base_revision_id is None and len(model.calls) == 1
    assert model.calls[0]['bases'] == {} and model.calls[0]['dialogue']['outputs'] == []
    assert 'inspect_scATAC' in model.calls[0]['dialogue']['tools']


def test_empty_unsupported_answer_and_exact_presentation_survive_process(tmp_path):
    application = ResearchAgentApplication(tmp_path / 'workspace')
    application.sessions.create('empty')
    model = Model(dict(kind='answer', intent='unsupported', relation='current', technical=False))
    outcome = application.sessions.respond('empty', 'first', 'Hello.', interpreter=model,
        expected_generation=0, submission={'profile_id': 'offered', 'inputs': {}})
    assert (outcome.kind, outcome.status) == ('answer', 'unsupported')
    view = presentation(outcome)
    view['text'] = outcome.text + ' Exactly displayed Unicode: λ.'
    application.sessions.store_presentation('empty', 'first', view)
    script = """
import json, sys
from agent.application import ResearchAgentApplication
from agent.schemas.orchestration import _serialize
app = ResearchAgentApplication(sys.argv[1])
def forbidden(*args, **kwargs): raise AssertionError('Unexpected model or science work')
app.run = app.resume = app._complete = forbidden
state = app.sessions.load('empty')
assert state.generation == 0 and not state.revisions
print(json.dumps(_serialize(state.interactions[0].presentation), ensure_ascii=False))
"""
    result = subprocess.run([sys.executable, '-B', '-c', script, str(application.workspace_root)],
                            env=os.environ.copy(), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == view and len(model.calls) == 1


def test_submission_capture_is_immutable_and_not_visible_to_interpreter(tmp_path):
    application = ResearchAgentApplication(tmp_path / 'workspace')
    application.sessions.create('empty')
    submission = {'profile_id': 'private-profile-sentinel', 'inputs': {'nested': [1, 2]}}
    model = Model(dict(kind='clarify', reason='unsupported_intent'))
    application.sessions.respond('empty', 'first', 'Start.', interpreter=model, submission=submission)
    assert 'submission' not in model.calls[0] and 'private-profile-sentinel' not in json.dumps(model.calls)
    submission['inputs']['nested'].append(3)
    state = application.sessions.load('empty')
    assert state.interactions[0].submission['inputs']['nested'] == (1, 2)
    with pytest.raises(TypeError):
        state.interactions[0].submission['profile_id'] = 'other'
    with pytest.raises(SessionConflictError, match='submission'):
        application.sessions.respond('empty', 'first', 'Start.', interpreter=model, submission=submission)
    assert len(model.calls) == 1
    with pytest.raises(SessionConflictError, match='Captured interaction'):
        application.sessions._store._update('empty', lambda current: replace(current,
            interactions=(replace(current.interactions[0], submission={'profile_id': 'changed'}),)))


def test_presentation_is_first_write_exact_and_terminal_metadata_stays_immutable(tmp_path):
    application = ResearchAgentApplication(tmp_path / 'workspace')
    application.sessions.create('empty')
    outcome = application.sessions.respond('empty', 'first', 'Start.',
        interpreter=Model(dict(kind='clarify', reason='unsupported_intent')))
    view = presentation(outcome)
    state = application.sessions.store_presentation('empty', 'first', view)
    raw_before = application.sessions._store._path('empty', '.json').read_bytes()
    assert application.sessions.store_presentation('empty', 'first', view) == state
    assert application.sessions._store._path('empty', '.json').read_bytes() == raw_before
    with pytest.raises(SessionConflictError, match='already exists'):
        application.sessions.store_presentation('empty', 'first', dict(view, text='Changed display.'))
    with pytest.raises(SessionConflictError, match='Terminal interaction'):
        application.sessions._store._update('empty', lambda current: replace(current,
            interactions=(replace(current.interactions[0], status='failed'),)))
    for invalid in (None, {'text': 'Partial.'}, dict(view, version=True), dict(view, unknown=1), dict(view, text='x' * 262145)):
        with pytest.raises((SessionError, ValueError)):
            application.sessions.store_presentation('empty', 'first', invalid)


def test_legacy_fields_remain_absent_and_roundtrip_exact(tmp_path):
    application = ResearchAgentApplication(tmp_path / 'workspace')
    application.sessions.create('empty')
    application.sessions.respond('empty', 'first', 'Start.',
        interpreter=Model(dict(kind='clarify', reason='unsupported_intent')))
    old = application.sessions.load('empty').to_dict()
    assert not {'presentation', 'submission', 'guidance_candidates'} & old['interactions'][0].keys()
    restored = AnalysisSession.from_dict(json.loads(canonical(old)))
    assert canonical(restored.to_dict()) == canonical(old)


def test_empty_capture_rejects_fabricated_revision_context():
    malformed = Interaction('turn', 'Start.', None, 0, {'relations': {'current': None}, 'bases': {}})
    with pytest.raises(SessionError, match='revision relations'):
        AnalysisSession('empty', interactions=(malformed,))


def test_processing_lease_blocks_other_process_and_releases_after_exception(tmp_path):
    application = ResearchAgentApplication(tmp_path / 'workspace')
    application.sessions.create('empty')
    code = """
import sys
from agent.application import ResearchAgentApplication, SessionConflictError
app = ResearchAgentApplication(sys.argv[1])
try:
    with app.sessions.processing_lease('empty', 'turn'): print('acquired')
except SessionConflictError:
    print('busy')
"""
    def check():
        result = subprocess.run([sys.executable, '-B', '-c', code, str(application.workspace_root)],
                                env=os.environ.copy(), capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        return result.stdout.strip()
    with pytest.raises(RuntimeError, match='abort'):
        with application.sessions.processing_lease('empty', 'turn'):
            assert check() == 'busy'
            with pytest.raises(SessionConflictError, match='already being processed'):
                with application.sessions.processing_lease('empty', 'turn'):
                    pytest.fail('Acquired duplicate lease')
            with application.sessions.processing_lease('empty', 'other'):
                pass
            raise RuntimeError('abort')
    assert check() == 'acquired'
