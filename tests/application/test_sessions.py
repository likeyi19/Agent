"""Session acceptance with synthetic accepted runs; no real scientific functions/providers."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from agent.application import (ResearchAgentApplication, ApplicationResult, ApplicationStatus,
                               ArtifactReference, OutputSelection, SessionError, SessionConflictError)
from agent.application.session_state import AnalysisSession, digest
from agent.application.session_store import FileSessionStore
from agent.orchestration import ToolRegistry, ToolSpec, ResultContract, ErrorClassification
from agent.schemas import AgentPlan, AgentRequest, PlanStep, ErrorCategory, RunMode

OUTPUTS = (OutputSelection('result', 'fixture', 'value'),)


class FixturePlanner:
    def plan(self, request, registry):
        return AgentPlan(request.request_id + ':plan', request.request_id, 'fixture',
                         (PlanStep('fixture', 'fixture', {}),))


@pytest.fixture
def app(tmp_path, monkeypatch):
    calls = []
    def science():
        calls.append('synthetic-tool')
        return {'value': 'synthetic-output'}
    spec = ToolSpec('fixture', science, {}, {}, ResultContract('Fixture', {'value': (str,)}),
                   lambda exc: ErrorClassification(ErrorCategory.TOOL_EXECUTION_ERROR, 'FIXTURE_ERROR'))
    application = ResearchAgentApplication(tmp_path / 'workspace', planner=FixturePlanner(),
                                            registry=ToolRegistry((spec,)))
    # Stub presentation only for this synthetic, non-scientific tool. Production
    # ApplicationResult and FileRunStore contracts remain exercised unchanged.
    def complete(result, run):
        if result.status.value != 'SUCCEEDED':
            status = ApplicationStatus(result.status.value)
            return application._base_result(result, run, status=status)
        refs = []
        for root, name in ((run.evidence, 'evidence.json'), (run.report, 'report.json')):
            path = root / name
            path.write_text(json.dumps({'run': result.run_id}))
            refs.append(ArtifactReference('fixture', str(path), hashlib.sha256(path.read_bytes()).hexdigest()))
        return ApplicationResult(result.request_id, result.run_id, ApplicationStatus.SUCCEEDED,
                                 result.status, str(run.root), result, evidence=refs[0], report=refs[1])
    monkeypatch.setattr(application, '_complete', complete)
    application.fixture_calls = calls
    return application


def request(name):
    return AgentRequest(name, 'Explicit synthetic fixture.', {})


def execute(app, name, generation):
    return app.sessions.run('session', name, request(name), OUTPUTS, expected_generation=generation)


def forbid_work(app, monkeypatch):
    def forbidden(*a, **kw):
        pytest.fail('Session operation attempted scientific/runtime/presentation work')
    monkeypatch.setattr(app, 'run', forbidden)
    monkeypatch.setattr(app, 'resume', forbidden)
    monkeypatch.setattr(app, '_complete', forbidden)
    from agent.orchestration.runtime import AgentRuntime
    from agent.orchestration.executor import PlanExecutor
    from agent.tools.data.authority_context import VerificationContext
    monkeypatch.setattr(AgentRuntime, 'run', forbidden)
    monkeypatch.setattr(AgentRuntime, 'resume', forbidden)
    monkeypatch.setattr(PlanExecutor, 'execute', forbidden)
    monkeypatch.setattr(VerificationContext, 'verify', forbidden)


def test_lazy_one_shot_parity(app):
    result = app.run(request('one-shot'))
    assert result.status is ApplicationStatus.SUCCEEDED
    assert not (app.workspace_root / 'sessions').exists()
    state = app.run_store.load(result.run_id)
    assert set(state.request.to_dict()) == {'request_id', 'prompt', 'inputs', 'mode'}
    from agent.schemas.run_state import PersistedRunState, fingerprint_plan
    assert PersistedRunState.from_dict(state.to_dict()) == state
    assert fingerprint_plan(state.plan) == state.plan_fingerprint
    assert app.resume(result.run_id) == result
    assert app.fixture_calls == ['synthetic-tool']
    assert not (app.workspace_root / 'sessions').exists()


def test_session_history_navigation_branch_restart(app, monkeypatch):
    app.sessions.create('session')
    one = execute(app, 'one', 0)
    r1 = one.active_revision_id
    two = execute(app, 'two', 1)
    r2 = two.active_revision_id
    assert two.revisions[1].parent_revision_id == r1
    app.sessions.switch('session', 'back', r1, expected_generation=2)
    three = execute(app, 'three', 3)
    assert three.generation == 4
    assert three.revisions[2].parent_revision_id == r1
    assert three.revisions[:2] == two.revisions
    assert {r.revision_id for r in three.revisions} >= {r1, r2}
    before = {p: p.read_bytes() for p in (app.workspace_root / 'run_state').glob('*.json')}
    forbid_work(app, monkeypatch)
    switched = app.sessions.switch('session', 'back-again', r2, expected_generation=4)
    assert switched.generation == 5
    assert switched.navigation[-1].from_revision_id == three.active_revision_id
    assert ResearchAgentApplication(app.workspace_root).sessions.load('session') == switched
    assert before == {p: p.read_bytes() for p in before}
    assert app.sessions.switch('session', 'back-again', r2, expected_generation=4) == switched
    assert app.fixture_calls == ['synthetic-tool'] * 3


def ready(app, name, generation):
    app.sessions.start_turn('session', name, request(name), OUTPUTS, expected_generation=generation)
    app.sessions.link_run('session', name)
    result = app.run(request(name))
    app.sessions._record_result('session', name, result)
    return result


def test_generation_aba_and_preserved_stale_revision(app, monkeypatch):
    app.sessions.create('session')
    r1 = execute(app, 'one', 0).active_revision_id
    r2 = execute(app, 'two', 1).active_revision_id
    app.sessions.switch('session', 'back', r1, expected_generation=2)
    ready(app, 'stale', 3)
    app.sessions.switch('session', 'forward', r2, expected_generation=3)
    app.sessions.switch('session', 'again', r1, expected_generation=4)
    forbid_work(app, monkeypatch)
    state = app.sessions.recover('session', 'stale')
    assert state.generation == 5 and state.active_revision_id == r1
    assert state.turn('stale').status == 'stale'
    assert state.revisions[-1].parent_revision_id == r1
    assert app.sessions.recover('session', 'stale') == state


def test_crash_after_completion_fresh_process_activation(app, monkeypatch):
    app.sessions.create('session')
    ready(app, 'one', 0)
    before = app.run_store.state_path('one:run').read_bytes()
    code = '''
import sys
from agent.application import ResearchAgentApplication
from agent.orchestration.runtime import AgentRuntime
from agent.orchestration.executor import PlanExecutor
from agent.tools.data.authority_context import VerificationContext
def forbidden(*a, **kw): raise AssertionError('unexpected computation')
AgentRuntime.run = AgentRuntime.resume = PlanExecutor.execute = VerificationContext.verify = forbidden
app = ResearchAgentApplication(sys.argv[1])
app._complete = forbidden
first = app.sessions.recover('session', 'one')
assert first.turn('one').status == 'activated'
assert app.sessions.recover('session', 'one') == first
assert len(first.revisions) == len(first.navigation) == first.generation == 1
print(first.active_revision_id)
'''
    completed = subprocess.run([sys.executable, '-c', code, str(app.workspace_root)],
                               env=os.environ.copy(), capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    forbid_work(app, monkeypatch)
    state = app.sessions.load('session')
    assert completed.stdout.strip() == state.active_revision_id
    assert before == app.run_store.state_path('one:run').read_bytes()
    assert app.sessions.recover('session', 'one') == state


def test_crash_before_presentation_boundary_requires_explicit_completion(app, monkeypatch):
    app.sessions.create('session')
    app.sessions.start_turn('session', 'one', request('one'), OUTPUTS, expected_generation=0)
    app.sessions.link_run('session', 'one')
    app.run(request('one'))  # crash before application completion was recorded
    with monkeypatch.context() as patch:
        forbid_work(app, patch)
        state = app.sessions.recover('session', 'one')
        assert state.turn('one').status == 'run_succeeded'
        assert state.active_revision_id is None
    state = app.sessions.complete_presentation('session', 'one')
    assert state.turn('one').status == 'activated'
    assert app.fixture_calls == ['synthetic-tool']


def test_navigation_concurrent_generation_conflict(app):
    app.sessions.create('session')
    r1 = execute(app, 'one', 0).active_revision_id
    execute(app, 'two', 1)
    def switch(name):
        try:
            return app.sessions.switch('session', name, r1, expected_generation=2)
        except SessionConflictError:
            return None
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(switch, ('a', 'b')))
    assert sum(r is not None for r in results) == 1
    assert app.sessions.load('session').generation == 3


def test_revision_immutable(app):
    app.sessions.create('session')
    state = execute(app, 'one', 0)
    with pytest.raises(SessionConflictError):
        app.sessions._store._update('session', lambda s: replace(s, revisions=(
            replace(s.revisions[0], outputs=(replace(s.revisions[0].outputs[0], accepted_step_sha256='0'*64),)),)))
    assert app.sessions.load('session') == state


@pytest.mark.parametrize('mutation', ['checksum', 'truncated', 'duplicate', 'generation', 'parent', 'unknown', 'version'])
def test_corruption_fails_closed(app, mutation):
    app.sessions.create('session')
    execute(app, 'one', 0)
    path = app.sessions._store._path('session', '.json')
    raw = path.read_bytes()
    value = json.loads(raw)
    if mutation == 'checksum': value['sha256'] = '0'*64
    elif mutation == 'truncated': path.write_bytes(raw[:12])
    elif mutation == 'duplicate': path.write_bytes(b'{"format":1,"format":2}')
    else:
        record = value['record']
        if mutation == 'generation': record['generation'] = 3
        if mutation == 'parent': record['revisions'][0]['parent_revision_id'] = 'missing'
        if mutation == 'unknown': record['extra'] = None
        if mutation == 'version': record['schema_version'] = True
        value['sha256'] = digest(record)
    if mutation not in {'truncated', 'duplicate'}: path.write_text(json.dumps(value))
    with pytest.raises(SessionError): app.sessions.load('session')


def test_presentation_mutation_prevents_activation(app):
    app.sessions.create('session')
    ready(app, 'one', 0)
    state = app.sessions.load('session')
    Path(state.turn('one').completion_files[0].path).write_text('mutated')
    with pytest.raises(SessionError): app.sessions.recover('session', 'one')
    assert app.sessions.load('session') == state


def test_missing_linked_run_never_reexecutes(app, monkeypatch):
    app.sessions.create('session')
    app.sessions.start_turn('session', 'one', request('one'), OUTPUTS, expected_generation=0)
    app.sessions.link_run('session', 'one')
    forbid_work(app, monkeypatch)
    assert execute(app, 'one', 0).turn('one').status == 'linked'
    assert not app.fixture_calls


def test_identity_reuse_and_output_errors(app):
    app.sessions.create('session')
    app.sessions.start_turn('session', 'one', request('one'), OUTPUTS, expected_generation=0)
    with pytest.raises(SessionConflictError):
        app.sessions.start_turn('session', 'one', request('other'), OUTPUTS, expected_generation=0)
    with pytest.raises(SessionConflictError):
        app.sessions.start_turn('session', 'other', request('one'), OUTPUTS, expected_generation=0)
    with pytest.raises(SessionError):
        app.sessions.start_turn('session', 'empty', request('empty'), (), expected_generation=0)
    with pytest.raises(SessionError):
        app.sessions.run('session', 'bad', request('bad'),
                         (OutputSelection('bad', 'absent', 'value'),), expected_generation=0)
    assert app.sessions.load('session').active_revision_id is None


def test_plan_only_nonactivation(app):
    app.sessions.create('session')
    before = execute(app, 'one', 0)
    planned = app.sessions.run('session', 'plan', replace(request('plan'), mode=RunMode.PLAN_ONLY),
                               OUTPUTS, expected_generation=1)
    assert planned.turn('plan').status == 'planned'
    assert planned.active_revision_id == before.active_revision_id
    assert app.fixture_calls == ['synthetic-tool']


def test_failed_science_nonactivation(app, monkeypatch):
    app.sessions.create('session')
    before = execute(app, 'one', 0)
    def fail(*args):
        from agent.orchestration import PlannerError
        raise PlannerError('FIXTURE', 'failed fixture')
    monkeypatch.setattr(app.runtime.planner, 'plan', fail)
    state = execute(app, 'failure', 1)
    assert state.turn('failure').status == 'failed'
    assert state.active_revision_id == before.active_revision_id


def test_symlinks_and_partial_write(app, monkeypatch, tmp_path):
    app.sessions.create('session')
    store = app.sessions._store
    original = store.load('session')
    def fail(*args): raise OSError('simulated replace failure')
    monkeypatch.setattr('agent.application.session_store.os.replace', fail)
    with pytest.raises(OSError):
        app.sessions.start_turn('session', 'one', request('one'), OUTPUTS, expected_generation=0)
    assert store.load('session') == original
    assert not list(store.root.glob('*.tmp'))
    path = store._path('session', '.json')
    path.unlink()
    path.symlink_to(tmp_path / 'outside')
    with pytest.raises(SessionError): store.load('session')


def test_cancelled_run_nonactivation(app, monkeypatch):
    app.sessions.create('session')
    previous = execute(app, 'one', 0)
    plan = app.runtime.planner.plan
    def cancelling(request, registry):
        app.cancel(request.request_id + ':run')
        return plan(request, registry)
    monkeypatch.setattr(app.runtime.planner, 'plan', cancelling)
    state = execute(app, 'cancel', 1)
    assert state.turn('cancel').status == 'cancelled'
    assert state.active_revision_id == previous.active_revision_id
    assert app.fixture_calls == ['synthetic-tool']


def test_presentation_failure_preserves_active_and_can_complete(app, monkeypatch):
    from agent.application import ApplicationError, ApplicationStage
    app.sessions.create('session')
    previous = execute(app, 'one', 0)
    complete = app._complete
    def failed(result, run):
        return app._base_result(result, run, status=ApplicationStatus.FAILED,
                               error=ApplicationError('FIXTURE', 'presentation failed', ApplicationStage.REPORT))
    monkeypatch.setattr(app, '_complete', failed)
    state = execute(app, 'two', 1)
    assert state.turn('two').status == 'run_succeeded'
    assert state.active_revision_id == previous.active_revision_id
    with monkeypatch.context() as patch:
        forbid_work(app, patch)
        assert app.sessions.recover('session', 'two') == state
    monkeypatch.setattr(app, '_complete', complete)
    state = app.sessions.complete_presentation('session', 'two')
    assert state.turn('two').status == 'activated'
    assert app.fixture_calls == ['synthetic-tool'] * 2


def test_concurrent_activation_idempotence(app, monkeypatch):
    app.sessions.create('session')
    ready(app, 'one', 0)
    forbid_work(app, monkeypatch)
    with ThreadPoolExecutor(2) as pool:
        states = list(pool.map(lambda _: app.sessions.recover('session', 'one'), range(2)))
    assert states[0] == states[1]
    assert len(states[0].revisions) == len(states[0].navigation) == 1


def test_no_run_outcomes(app, monkeypatch):
    app.sessions.create('session')
    forbid_work(app, monkeypatch)
    for outcome in ('clarification', 'failed', 'cancelled'):
        state = app.sessions.record_no_run_turn('session', outcome, outcome=outcome, expected_generation=0)
        assert state.turn(outcome).status == outcome
        assert state.generation == 0 and state.active_revision_id is None
        assert app.sessions.record_no_run_turn('session', outcome, outcome=outcome, expected_generation=0) == state


def test_existing_run_attach_exact_identity(app):
    app.sessions.create('session')
    result = app.run(request('one'))
    app.sessions.start_turn('session', 'attach', request('one'), OUTPUTS, expected_generation=0)
    app.sessions.link_run('session', 'attach')
    state = app.sessions.recover('session', 'attach')
    assert state.turn('attach').status == 'run_succeeded'
    state = app.sessions.complete_presentation('session', 'attach')
    assert state.revisions[0].run_id == result.run_id
    assert app.fixture_calls == ['synthetic-tool']


def test_existing_run_wrong_request_rejected(app):
    app.sessions.create('session')
    app.run(request('one'))
    altered = replace(request('one'), prompt='Different scientific intent')
    app.sessions.start_turn('session', 'attach', altered, OUTPUTS, expected_generation=0)
    app.sessions.link_run('session', 'attach')
    with pytest.raises(SessionConflictError): app.sessions.recover('session', 'attach')


def test_atomic_activation_commit_then_lost_acknowledgement(app, monkeypatch):
    app.sessions.create('session')
    ready(app, 'one', 0)
    store = app.sessions._store
    write = store._write
    def lost(state):
        write(state)
        raise OSError('lost acknowledgement after durable commit')
    monkeypatch.setattr(store, '_write', lost)
    sessions = app.sessions
    sessions._store = store
    with pytest.raises(OSError): sessions.recover('session', 'one')
    forbid_work(app, monkeypatch)
    state = app.sessions.recover('session', 'one')
    assert state.generation == 1
    assert len(state.revisions) == len(state.navigation) == 1


@pytest.mark.parametrize('failure', ['execution', 'verification'])
def test_execution_and_result_verification_failure_do_not_activate(app, monkeypatch, failure):
    app.sessions.create('session')
    previous = execute(app, 'one', 0)
    def bad():
        if failure == 'execution': raise ValueError('synthetic failure')
        return {'value': 42}  # violates the registered result contract
    original = app.registry.get('fixture')
    registry = ToolRegistry((replace(original, function=bad),))
    monkeypatch.setattr(app.runtime, '_registry', registry)
    monkeypatch.setattr(app.runtime.executor, '_registry', registry)
    state = execute(app, 'failure', 1)
    assert state.turn('failure').status == 'failed'
    assert state.active_revision_id == previous.active_revision_id
    assert state.generation == 1


def test_outside_session_symlink_rejected_without_write(app, tmp_path):
    outside = tmp_path / 'outside'
    outside.mkdir()
    (app.workspace_root / 'sessions').symlink_to(outside, target_is_directory=True)
    from agent.application import ApplicationWorkspaceError
    with pytest.raises(ApplicationWorkspaceError): app.sessions.create('session')
    assert list(outside.iterdir()) == []


def test_turn_roundtrip_and_no_authority_payload(app):
    app.sessions.create('session')
    state = execute(app, 'one', 0)
    value = json.loads(json.dumps(state.to_dict()))
    assert AnalysisSession.from_dict(value) == state
    assert set(value) == {'schema_version', 'session_id', 'active_revision_id', 'generation',
                          'revisions', 'turns', 'navigation'}
    assert set(value['revisions'][0]['outputs'][0]) == {
        'name', 'run_id', 'step_id', 'output_key', 'accepted_step_sha256'}
    assert 'artifact_authority' not in json.dumps(value)
    assert 'synthetic-output' not in json.dumps(value)


@pytest.mark.parametrize('mutation', ['stale_without_revision', 'ready_with_revision', 'duplicate_request'])
def test_incomplete_turn_outcomes_rejected(app, mutation):
    app.sessions.create('session')
    ready(app, 'one', 0)
    state = app.sessions.load('session')
    value = json.loads(json.dumps(state.to_dict()))
    turn = value['turns'][0]
    if mutation == 'stale_without_revision': turn['status'] = 'stale'
    elif mutation == 'ready_with_revision': turn['revision_id'] = 'unpublished'
    else: value['turns'].append(dict(turn, turn_id='another-turn'))
    with pytest.raises(SessionError): AnalysisSession.from_dict(value)


def test_configuration_failure_records_failed_submission(tmp_path):
    from agent.application import ApplicationServiceError
    application = ResearchAgentApplication(tmp_path / 'workspace')
    application.sessions.create('session')
    with pytest.raises(ApplicationServiceError):
        application.sessions.run('session', 'one', request('one'), OUTPUTS, expected_generation=0)
    state = application.sessions.load('session')
    assert state.turn('one').status == 'failed'
    assert state.active_revision_id is None and state.generation == 0
    assert not application.run_store.state_path('one:run').exists()
