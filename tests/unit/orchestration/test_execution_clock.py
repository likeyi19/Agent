"""Wall observations are audit data; revisions and sequence own causal order."""
from dataclasses import replace

import pytest

from agent.orchestration import (
    AgentRuntime, FileRunStore, RunAlreadyExistsError, RunLifecycleStatus,
    RunStateConflictError, StepStatus, ToolRegistry,
)
from agent.schemas import ExecutionTraceEvent, PersistedRunState, TraceEventType
from test_runtime import FixedPlanner, _plan, _request, _spec, PlanStep
from test_run_store import _planning_state


BEFORE = '2030-04-05T12:00:00+00:00'
EARLIER = '2030-04-05T11:00:00+00:00'
LATER = '2030-04-05T14:00:00+00:00'


def wall_clock(monkeypatch):
    observed = [BEFORE]
    for module in ('runtime', 'executor'):
        monkeypatch.setattr(f'agent.orchestration.{module}._utc_now', lambda: observed[0])
    return observed


@pytest.mark.parametrize('boundary', ['planning', 'step', 'forward_then_backward'])
def test_discontinuous_wall_observations_survive_execution_and_reopen(tmp_path, monkeypatch, boundary):
    clock = wall_clock(monkeypatch)
    ticks = iter((10.0, 10.25))
    monkeypatch.setattr('agent.orchestration.executor.perf_counter', lambda: next(ticks))
    calls = []

    def tool():
        calls.append(clock[0])
        if boundary != 'planning':
            clock[0] = EARLIER
        return {'value': 'verified'}

    class Planner(FixedPlanner):
        def plan(self, request, registry):
            clock[0] = EARLIER if boundary == 'planning' else LATER if boundary == 'forward_then_backward' else BEFORE
            return super().plan(request, registry)

    registry = ToolRegistry((_spec('tool', tool),))
    planner = Planner(_plan(PlanStep('step', 'tool', {})))
    store = FileRunStore(tmp_path/'state')
    result = AgentRuntime(planner=planner, registry=registry, run_store=store).run(_request())
    assert result.status.value == 'SUCCEEDED' and planner.calls == 1 and len(calls) == 1
    state = store.load(result.run_id)
    assert state.created_at == BEFORE and state.updated_at == EARLIER
    assert [e.sequence for e in state.trace] == list(range(len(state.trace)))
    assert any(b.timestamp < a.timestamp for a, b in zip(state.trace, state.trace[1:]))
    step, = state.steps
    assert step.finished_at == EARLIER
    assert step.started_at == (EARLIER if boundary == 'planning' else LATER if boundary == 'forward_then_backward' else BEFORE)
    assert step.duration_seconds == .25  # Elapsed time, never UTC subtraction or clamping.
    before = store.state_path(result.run_id).read_bytes()
    clock[0] = LATER
    reopened = AgentRuntime(registry=registry, run_store=FileRunStore(store.root))
    assert reopened.resume(result.run_id) == result
    assert store.state_path(result.run_id).read_bytes() == before
    assert len(calls) == 1
    with pytest.raises(RunAlreadyExistsError):
        AgentRuntime(planner=planner, registry=registry, run_store=store).run(_request())
    assert planner.calls == 1 and len(calls) == 1


def test_backward_update_retains_cas_creation_identity_and_exact_trace_prefix(tmp_path):
    event = ExecutionTraceEvent(0, TraceEventType.PLANNING, BEFORE, 'Captured planning.')
    store = FileRunStore(tmp_path)
    original = store.create(replace(_planning_state(), created_at=BEFORE, updated_at=BEFORE, trace=(event,)))
    later_event = ExecutionTraceEvent(1, TraceEventType.PLANNING, EARLIER, 'Next causal event.')
    updated = store.update(replace(original, revision=1, updated_at=EARLIER,
        trace=(event, later_event)), expected_revision=0)
    assert FileRunStore(tmp_path).load(original.run_id) == updated
    with pytest.raises(RunStateConflictError, match='not expected revision'):
        store.update(updated, expected_revision=0)
    with pytest.raises(RunStateConflictError, match='creation identity'):
        store.update(replace(updated, revision=2, created_at=EARLIER), expected_revision=1)
    with pytest.raises(RunStateConflictError, match='trace history'):
        store.update(replace(updated, revision=2, trace=(replace(event, timestamp=EARLIER), later_event)),
                     expected_revision=1)
    with pytest.raises(ValueError, match='contiguous'):
        replace(updated, trace=(event, replace(later_event, sequence=0)))
    assert store.load(original.run_id) == updated


def test_restart_after_clock_rollback_does_not_authorize_unknown_outcome_retry(tmp_path, monkeypatch):
    clock = wall_clock(monkeypatch)
    calls = []

    class ProcessExit(BaseException):
        pass

    class InterruptedStore(FileRunStore):
        def update(self, state, *, expected_revision):
            if any(s.status is StepStatus.SUCCEEDED for s in state.steps):
                raise ProcessExit()  # Tool returned; verified completion was not saved.
            return super().update(state, expected_revision=expected_revision)

    def tool():
        calls.append('once')
        return {'value': 'complete'}

    registry = ToolRegistry((_spec('tool', tool),))
    store = InterruptedStore(tmp_path)
    planner = FixedPlanner(_plan(PlanStep('step', 'tool', {})))
    with pytest.raises(ProcessExit):
        AgentRuntime(planner=planner, registry=registry, run_store=store).run(_request())
    old = store.load('request-1:run')
    assert old.steps[0].status is StepStatus.RUNNING and calls == ['once']
    clock[0] = EARLIER
    resumed = AgentRuntime(registry=registry, run_store=FileRunStore(tmp_path)).resume(old.run_id)
    saved = store.load(old.run_id)
    assert saved.lifecycle_status is RunLifecycleStatus.INTERRUPTED
    assert saved.trace[:len(old.trace)] == old.trace
    assert saved.updated_at == EARLIER and saved.steps[0].finished_at == EARLIER
    assert any(e.code == 'STEP_OUTCOME_UNKNOWN_AFTER_INTERRUPTION' for e in resumed.errors)
    assert calls == ['once'] and planner.calls == 1


@pytest.mark.parametrize('failure', ['invalid_result', 'tool_exception'])
def test_clock_rollback_never_turns_invalid_tool_outcome_into_success(tmp_path, monkeypatch, failure):
    clock = wall_clock(monkeypatch)
    calls = []

    def tool():
        calls.append('once')
        clock[0] = EARLIER
        if failure == 'tool_exception':
            raise ValueError('Rejected scientific input')
        return {'value': 37}  # Violates this tool's explicit result contract.

    registry = ToolRegistry((_spec('tool', tool),))
    store = FileRunStore(tmp_path)
    result = AgentRuntime(planner=FixedPlanner(_plan(PlanStep('step', 'tool', {}))),
        registry=registry, run_store=store).run(_request())
    assert result.status.value == 'FAILED' and result.errors
    saved = store.load(result.run_id)
    assert saved.updated_at == EARLIER and saved.steps[0].status is StepStatus.FAILED
    step, = saved.steps
    assert step.attempt_count == 1
    if failure == 'invalid_result':
        assert step.result == {'value': 37}  # Retain rejected bytes for audit.
        assert not step.verification.passed and step.verification.artifact_authority is None
    else:
        assert step.result is None and step.verification is None
    assert AgentRuntime(registry=registry, run_store=FileRunStore(tmp_path)).resume(result.run_id) == result
    assert calls == ['once']


@pytest.fixture
def completed_state(tmp_path, monkeypatch):
    wall_clock(monkeypatch)
    registry = ToolRegistry((_spec('tool', lambda: {'value': 'done'}),))
    store = FileRunStore(tmp_path)
    result = AgentRuntime(planner=FixedPlanner(_plan(PlanStep('step', 'tool', {}))),
        registry=registry, run_store=store).run(_request())
    assert result.status.value == 'SUCCEEDED'
    return store.load(result.run_id)


@pytest.mark.parametrize('part', ['updated_at', 'trace', 'step'])
def test_each_wall_timestamp_boundary_preserves_observations(completed_state, part):
    values = completed_state.to_dict()
    if part == 'updated_at':
        values[part] = EARLIER
    elif part == 'trace':
        values['trace'][-1]['timestamp'] = EARLIER
    else:
        values['steps'][0]['finished_at'] = EARLIER
    assert PersistedRunState.from_dict(values).to_dict() == values


@pytest.mark.parametrize('part', ['created_at', 'updated_at', 'trace', 'started_at', 'finished_at'])
@pytest.mark.parametrize('invalid', ['not-a-timestamp', '2030-04-05T12:00:00'])
def test_timestamp_syntax_and_timezone_remain_required(completed_state, part, invalid):
    values = completed_state.to_dict()
    if part in ('created_at', 'updated_at'):
        values[part] = invalid
    elif part == 'trace':
        values['trace'][0]['timestamp'] = invalid
    else:
        values['steps'][0][part] = invalid
    with pytest.raises(ValueError, match='timestamp|timezone'):
        PersistedRunState.from_dict(values)


@pytest.mark.parametrize('duration', [-1, float('nan'), float('inf'), True])
def test_wall_discontinuity_does_not_allow_invalid_elapsed_duration(completed_state, duration):
    with pytest.raises(ValueError, match='duration_seconds'):
        replace(completed_state.steps[0], duration_seconds=duration)
