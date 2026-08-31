"""Offline acceptance tests for durable cooperative run cancellation."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading
from unittest.mock import Mock

import pytest

from agent.orchestration import (
    AgentError,
    AgentPlan,
    AgentRequest,
    AgentRuntime,
    CancellationDisposition,
    CancellationRequestedError,
    CancellationStateCorruptionError,
    CancellationStateVersionError,
    ErrorCategory,
    ErrorClassification,
    FileRunStore,
    PersistedRunState,
    PlanExecutor,
    PlanStep,
    PlannerError,
    RecoveryPolicy,
    ResultContract,
    RunAlreadyActiveError,
    RunLifecycleStatus,
    RunMode,
    RunNotFoundError,
    RunStatus,
    RunStateVersionError,
    StepStatus,
    ToolRegistry,
    ToolSpec,
    TraceEventType,
)
from agent.schemas import RUN_STATE_SCHEMA_VERSION


class SimulatedProcessExit(BaseException):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _classify(
    exception: Exception, *, recoverable: bool = False
) -> ErrorClassification:
    return ErrorClassification(
        ErrorCategory.TOOL_EXECUTION_ERROR,
        "TRANSIENT" if recoverable else "FAKE_FAILURE",
    )


def _spec(name: str, function, *, recoverable: bool = False) -> ToolSpec:
    return ToolSpec(
        name,
        function,
        {},
        {},
        ResultContract(f"{name}Result", {"value": (str,)}),
        lambda exception: _classify(exception, recoverable=recoverable),
        frozenset({"TRANSIENT"}) if recoverable else frozenset(),
    )


def _request(*, mode: RunMode = RunMode.EXECUTE) -> AgentRequest:
    return AgentRequest("request-1", "fixed workflow", {}, mode)


def _one_step_plan() -> AgentPlan:
    return AgentPlan(
        "plan-1",
        "request-1",
        "fixed",
        (PlanStep("first", "first", {}),),
    )


def _two_step_plan() -> AgentPlan:
    return AgentPlan(
        "plan-1",
        "request-1",
        "fixed",
        (
            PlanStep("first", "first", {}),
            PlanStep("second", "second", {}, ("first",)),
        ),
    )


class FixedPlanner:
    def __init__(self, plan: AgentPlan) -> None:
        self.plan_value = plan
        self.calls = 0

    def plan(self, request: AgentRequest, registry: ToolRegistry) -> AgentPlan:
        self.calls += 1
        return self.plan_value


class CancelOnCreateStore(FileRunStore):
    def create(self, state: PersistedRunState) -> PersistedRunState:
        created = super().create(state)
        self.request_cancellation(state.run_id)
        return created


class CancelAfterPlanStore(FileRunStore):
    def update(
        self, state: PersistedRunState, *, expected_revision: int
    ) -> PersistedRunState:
        saved = super().update(state, expected_revision=expected_revision)
        if (
            saved.lifecycle_status is RunLifecycleStatus.PLANNING
            and saved.plan is not None
            and saved.preflight_verification is None
        ):
            self.request_cancellation(saved.run_id)
        return saved


class InterruptingStore(FileRunStore):
    def __init__(self, root: Path, predicate) -> None:
        super().__init__(root)
        self.predicate = predicate
        self.triggered = False

    def update(
        self, state: PersistedRunState, *, expected_revision: int
    ) -> PersistedRunState:
        saved = super().update(state, expected_revision=expected_revision)
        if not self.triggered and self.predicate(saved):
            self.triggered = True
            raise SimulatedProcessExit()
        return saved


class ForwardingCancellationStore:
    """Complete proxy that requests cancellation after step 1 is checkpointed."""

    def __init__(self, delegate: FileRunStore) -> None:
        self.delegate = delegate
        self.requested = False

    def execution_lease(self, run_id):
        return self.delegate.execution_lease(run_id)

    def create(self, state):
        return self.delegate.create(state)

    def load(self, run_id):
        return self.delegate.load(run_id)

    def request_cancellation(self, run_id):
        return self.delegate.request_cancellation(run_id)

    def load_cancellation(self, run_id):
        return self.delegate.load_cancellation(run_id)

    def update(self, state, *, expected_revision):
        saved = self.delegate.update(state, expected_revision=expected_revision)
        if (
            not self.requested
            and saved.lifecycle_status is RunLifecycleStatus.RUNNING
            and len(saved.steps) == 2
            and saved.steps[0].status is StepStatus.SUCCEEDED
            and saved.steps[1].status is StepStatus.PENDING
        ):
            self.delegate.request_cancellation(saved.run_id)
            self.requested = True
        return saved


def _runtime(
    store: FileRunStore,
    first,
    second=None,
    *,
    plan: AgentPlan | None = None,
    executor: PlanExecutor | None = None,
    first_recoverable: bool = False,
) -> tuple[AgentRuntime, FixedPlanner, ToolRegistry]:
    specs = [_spec("first", first, recoverable=first_recoverable)]
    if second is not None:
        specs.append(_spec("second", second))
    registry = executor.registry if executor is not None else ToolRegistry(tuple(specs))
    planner = FixedPlanner(plan or (_two_step_plan() if second else _one_step_plan()))
    return (
        AgentRuntime(
            planner=planner,
            registry=registry,
            executor=executor,
            run_store=store,
        ),
        planner,
        registry,
    )


def _planning_state() -> PersistedRunState:
    timestamp = _now()
    return PersistedRunState(
        RUN_STATE_SCHEMA_VERSION,
        0,
        "request-1:run",
        _request(),
        RunLifecycleStatus.PLANNING,
        timestamp,
        timestamp,
    )


def _literal_v1_error(
    *,
    category: str = "INTERNAL_AGENT_ERROR",
    code: str = "FAILED_FOR_TEST",
) -> dict[str, object]:
    return {
        "category": category,
        "code": code,
        "message": "legacy failure",
        "step_id": None,
        "tool_name": None,
        "exception_type": None,
        "recoverable": False,
        "attempt": None,
        "details": {},
    }


def _literal_v1_plan() -> dict[str, object]:
    return {
        "plan_id": "plan-1",
        "request_id": "request-1",
        "planner_name": "fixed",
        "steps": [
            {
                "step_id": "first",
                "tool_name": "first",
                "arguments": {},
                "depends_on": [],
                "description": None,
            }
        ],
    }


def _literal_v1_verification(
    target_type: str, target_id: str
) -> dict[str, object]:
    return {
        "passed": True,
        "target_type": target_type,
        "target_id": target_id,
        "checks": [],
        "error": None,
    }


def _literal_v1_pending_step() -> dict[str, object]:
    return {
        "step_id": "first",
        "tool_name": "first",
        "status": "PENDING",
        "attempt_count": 0,
        "resolved_arguments": {},
        "result": None,
        "verification": None,
        "error": None,
        "started_at": None,
        "finished_at": None,
        "duration_seconds": None,
    }


def _literal_v1_record(
    lifecycle_status: str,
    *,
    timestamp: str,
) -> dict[str, object]:
    plan = _literal_v1_plan()
    mode = "PLAN_ONLY" if lifecycle_status == "PLANNED" else "EXECUTE"
    record: dict[str, object] = {
        "schema_version": 1,
        "revision": 0,
        "run_id": "request-1:run",
        "request": {
            "request_id": "request-1",
            "prompt": "fixed workflow",
            "inputs": {},
            "mode": mode,
        },
        "lifecycle_status": lifecycle_status,
        "created_at": timestamp,
        "updated_at": timestamp,
        "plan": plan,
        "plan_fingerprint": hashlib.sha256(_canonical(plan)).hexdigest(),
        "preflight_verification": _literal_v1_verification("plan", "plan-1"),
        "steps": [],
        "run_verification": None,
        "errors": [],
        "trace": [],
    }
    if lifecycle_status == "SUCCEEDED":
        record["steps"] = [
            {
                **_literal_v1_pending_step(),
                "status": "SUCCEEDED",
                "attempt_count": 1,
                "result": {"value": "done"},
                "verification": _literal_v1_verification("step", "first"),
                "started_at": timestamp,
                "finished_at": timestamp,
                "duration_seconds": 0.0,
            }
        ]
        record["run_verification"] = _literal_v1_verification("run", "plan-1")
    elif lifecycle_status == "PLANNING":
        record["preflight_verification"] = None
    elif lifecycle_status == "FAILED":
        record["plan"] = None
        record["plan_fingerprint"] = None
        record["preflight_verification"] = None
        record["errors"] = [_literal_v1_error()]
    elif lifecycle_status == "INTERRUPTED":
        record["steps"] = [_literal_v1_pending_step()]
        record["errors"] = [
            _literal_v1_error(code="STEP_OUTCOME_UNKNOWN_AFTER_INTERRUPTION")
        ]
    return record


def _write_literal_v1_record(
    store: FileRunStore, record: dict[str, object]
) -> bytes:
    envelope = {
        "format": "agent.run-state",
        "schema_version": 1,
        "integrity": {
            "algorithm": "sha256",
            "digest": hashlib.sha256(_canonical(record)).hexdigest(),
        },
        "record": record,
    }
    payload = _canonical(envelope)
    store.state_path("request-1:run").write_bytes(payload)
    return payload


@pytest.mark.parametrize(
    "missing_method",
    ["request_cancellation", "load_cancellation"],
)
def test_runtime_rejects_store_missing_cancellation_capability(
    tmp_path: Path,
    missing_method: str,
) -> None:
    delegate = FileRunStore(tmp_path)

    class LegacyStore:
        def execution_lease(self, run_id):
            return delegate.execution_lease(run_id)

        def create(self, state):
            return delegate.create(state)

        def load(self, run_id):
            return delegate.load(run_id)

        def update(self, state, *, expected_revision):
            return delegate.update(state, expected_revision=expected_revision)

    methods = {
        "request_cancellation": delegate.request_cancellation,
        "load_cancellation": delegate.load_cancellation,
    }
    store = LegacyStore()
    for method_name, method in methods.items():
        if method_name != missing_method:
            setattr(store, method_name, method)
    planner = Mock()
    tool = Mock(return_value={"value": "must-not-run"})
    registry = ToolRegistry((_spec("first", tool),))

    with pytest.raises(TypeError, match="must implement RunStore"):
        AgentRuntime(planner=planner, registry=registry, run_store=store)

    planner.plan.assert_not_called()
    tool.assert_not_called()


def test_complete_forwarding_store_observes_cancel_before_second_step(
    tmp_path: Path,
) -> None:
    first = Mock(return_value={"value": "first"})
    second = Mock(return_value={"value": "unused"})
    delegate = FileRunStore(tmp_path)
    proxy = ForwardingCancellationStore(delegate)
    runtime, _, _ = _runtime(proxy, first, second)

    result = runtime.run(_request())

    assert proxy.requested
    assert result.status is RunStatus.CANCELLED
    assert delegate.load(result.run_id).lifecycle_status is RunLifecycleStatus.CANCELLED
    assert first.call_count == 1
    second.assert_not_called()


def test_sidecar_survives_reopen_is_hashed_and_does_not_change_revision(
    tmp_path: Path,
) -> None:
    store = FileRunStore(tmp_path)
    state = store.create(_planning_state())

    receipt = store.request_cancellation(state.run_id)
    reopened = FileRunStore(tmp_path)
    request = reopened.load_cancellation(state.run_id)

    assert receipt.disposition is CancellationDisposition.REQUESTED
    assert request is not None
    assert request.requested_at == receipt.requested_at
    assert reopened.load(state.run_id).revision == state.revision
    assert state.run_id not in store.cancellation_path(state.run_id).name
    assert store.cancellation_path(state.run_id).stat().st_mode & 0o777 == 0o600
    json.dumps(receipt.to_dict(), allow_nan=False)


def test_duplicate_cancel_is_idempotent_and_preserves_original_timestamp(
    tmp_path: Path,
) -> None:
    store = FileRunStore(tmp_path)
    state = store.create(_planning_state())

    first = store.request_cancellation(state.run_id)
    second = store.request_cancellation(state.run_id)

    assert first.disposition is CancellationDisposition.REQUESTED
    assert second.disposition is CancellationDisposition.ALREADY_REQUESTED
    assert second.requested_at == first.requested_at
    assert store.load(state.run_id).revision == 0


def test_concurrent_duplicate_cancel_creates_one_request(tmp_path: Path) -> None:
    store = FileRunStore(tmp_path)
    store.create(_planning_state())
    barrier = threading.Barrier(3)
    receipts = []

    def request() -> None:
        barrier.wait(timeout=10)
        receipts.append(FileRunStore(tmp_path).request_cancellation("request-1:run"))

    workers = [threading.Thread(target=request, daemon=True) for _ in range(2)]
    for worker in workers:
        worker.start()
    barrier.wait(timeout=10)
    for worker in workers:
        worker.join(timeout=10)

    assert not any(worker.is_alive() for worker in workers)
    assert {receipt.disposition for receipt in receipts} == {
        CancellationDisposition.REQUESTED,
        CancellationDisposition.ALREADY_REQUESTED,
    }
    assert receipts[0].requested_at == receipts[1].requested_at


def test_cancel_uses_state_lock_not_execution_lease(tmp_path: Path) -> None:
    store = FileRunStore(tmp_path)
    store.create(_planning_state())

    with store.execution_lease("request-1:run"):
        receipt = FileRunStore(tmp_path).request_cancellation("request-1:run")

    assert receipt.disposition is CancellationDisposition.REQUESTED


def test_unknown_run_and_runtime_without_store_raise_actionable_errors(
    tmp_path: Path,
) -> None:
    with pytest.raises(RunNotFoundError, match="not found"):
        FileRunStore(tmp_path).request_cancellation("missing:run")
    with pytest.raises(RuntimeError, match="requires an injected RunStore"):
        AgentRuntime().cancel("missing:run")


@pytest.mark.parametrize("mode", [RunMode.EXECUTE, RunMode.PLAN_ONLY])
def test_cancellation_before_planner_invocation_calls_neither_planner_nor_tool(
    tmp_path: Path, mode: RunMode
) -> None:
    tool = Mock(return_value={"value": "unused"})
    planner = FixedPlanner(_one_step_plan())
    registry = ToolRegistry((_spec("first", tool),))
    store = CancelOnCreateStore(tmp_path)

    result = AgentRuntime(
        planner=planner, registry=registry, run_store=store
    ).run(_request(mode=mode))

    assert result.status is RunStatus.CANCELLED
    assert result.planning_only is (mode is RunMode.PLAN_ONLY)
    assert result.plan is None
    assert planner.calls == 0
    tool.assert_not_called()


@pytest.mark.parametrize("mode", [RunMode.EXECUTE, RunMode.PLAN_ONLY])
def test_cancellation_after_plan_persistence_prevents_preflight_and_execution(
    tmp_path: Path, mode: RunMode
) -> None:
    tool = Mock(return_value={"value": "unused"})
    runtime, planner, _ = _runtime(CancelAfterPlanStore(tmp_path), tool)

    result = runtime.run(_request(mode=mode))

    assert result.status is RunStatus.CANCELLED
    assert result.plan == _one_step_plan()
    assert result.verification is None
    assert planner.calls == 1
    tool.assert_not_called()


@pytest.mark.parametrize("mode", [RunMode.EXECUTE, RunMode.PLAN_ONLY])
def test_cancellation_during_preflight_prevents_first_attempt(
    tmp_path: Path, mode: RunMode
) -> None:
    tool = Mock(return_value={"value": "unused"})
    store = FileRunStore(tmp_path)
    registry = ToolRegistry((_spec("first", tool),))

    class CancellingExecutor(PlanExecutor):
        def preflight(self, plan: AgentPlan):
            store.request_cancellation("request-1:run")
            return super().preflight(plan)

    executor = CancellingExecutor(registry)
    runtime = AgentRuntime(
        planner=FixedPlanner(_one_step_plan()),
        registry=registry,
        executor=executor,
        run_store=store,
    )

    result = runtime.run(_request(mode=mode))

    assert result.status is RunStatus.CANCELLED
    if mode is RunMode.EXECUTE:
        assert result.steps[0].status is StepStatus.SKIPPED
    else:
        assert result.steps == ()
    tool.assert_not_called()


def test_running_tool_finishes_checkpoints_success_and_stops_downstream(
    tmp_path: Path,
) -> None:
    entered = threading.Event()
    release = threading.Event()

    def blocking() -> dict[str, str]:
        entered.set()
        assert release.wait(timeout=10)
        return {"value": "verified"}

    first = Mock(side_effect=blocking)
    second = Mock(return_value={"value": "must-not-run"})
    store = FileRunStore(tmp_path)
    runtime, planner, _ = _runtime(store, first, second)
    outcomes = []
    failures: list[BaseException] = []

    def execute() -> None:
        try:
            outcomes.append(runtime.run(_request()))
        except BaseException as exc:
            failures.append(exc)

    worker = threading.Thread(target=execute, daemon=True)
    worker.start()
    assert entered.wait(timeout=10)
    active = store.load("request-1:run")
    receipt = AgentRuntime(run_store=FileRunStore(tmp_path)).cancel("request-1:run")
    assert receipt.disposition is CancellationDisposition.REQUESTED
    assert store.load("request-1:run").revision == active.revision
    assert worker.is_alive()
    with pytest.raises(RunAlreadyActiveError):
        AgentRuntime(
            planner=planner,
            registry=runtime.registry,
            run_store=FileRunStore(tmp_path),
        ).resume("request-1:run")
    release.set()
    worker.join(timeout=10)

    assert not worker.is_alive()
    assert not failures
    result = outcomes[0]
    assert result.status is RunStatus.CANCELLED
    assert [step.status for step in result.steps] == [
        StepStatus.SUCCEEDED,
        StepStatus.SKIPPED,
    ]
    assert store.load(result.run_id).steps[0].verification.passed
    assert first.call_count == 1
    second.assert_not_called()
    assert any(
        event.event_type is TraceEventType.STEP_SKIPPED for event in result.trace
    )


def test_cancellation_before_retry_preserves_failure_and_attempt_count(
    tmp_path: Path,
) -> None:
    store = FileRunStore(tmp_path)
    calls = 0

    def failing() -> dict[str, str]:
        nonlocal calls
        calls += 1
        store.request_cancellation("request-1:run")
        raise RuntimeError("retryable")

    runtime, _, _ = _runtime(
        store,
        failing,
        first_recoverable=True,
        executor=PlanExecutor(
            ToolRegistry((_spec("first", failing, recoverable=True),)),
            recovery_policy=RecoveryPolicy(max_attempts_per_step=2),
        ),
    )

    result = runtime.run(_request())

    assert result.status is RunStatus.CANCELLED
    assert calls == 1
    assert result.steps[0].status is StepStatus.FAILED
    assert result.steps[0].attempt_count == 1
    assert result.steps[0].error is not None
    assert result.steps[0].error.code == "TRANSIENT"


def test_cancellation_during_nonretryable_failure_wins_failed_checkpoint(
    tmp_path: Path,
) -> None:
    store = FileRunStore(tmp_path)

    def failing() -> dict[str, str]:
        store.request_cancellation("request-1:run")
        raise RuntimeError("permanent")

    runtime, _, _ = _runtime(store, failing)

    result = runtime.run(_request())

    assert result.status is RunStatus.CANCELLED
    assert result.steps[0].status is StepStatus.FAILED
    assert result.steps[0].attempt_count == 1
    assert any(error.code == "FAKE_FAILURE" for error in result.errors)
    assert store.load(result.run_id).lifecycle_status is RunLifecycleStatus.CANCELLED


def test_resume_with_cancellation_preserves_success_and_skips_pending(
    tmp_path: Path,
) -> None:
    first = Mock(return_value={"value": "done"})
    second = Mock(return_value={"value": "must-not-run"})
    store = InterruptingStore(
        tmp_path,
        lambda state: (
            len(state.steps) == 2
            and state.steps[0].status is StepStatus.SUCCEEDED
            and state.steps[1].status is StepStatus.PENDING
        ),
    )
    runtime, _, registry = _runtime(store, first, second)
    with pytest.raises(SimulatedProcessExit):
        runtime.run(_request())

    store.request_cancellation("request-1:run")
    resume_planner = FixedPlanner(_two_step_plan())
    result = AgentRuntime(
        planner=resume_planner,
        registry=registry,
        run_store=FileRunStore(tmp_path),
    ).resume("request-1:run")

    assert result.status is RunStatus.CANCELLED
    assert [step.status for step in result.steps] == [
        StepStatus.SUCCEEDED,
        StepStatus.SKIPPED,
    ]
    assert resume_planner.calls == 0
    assert first.call_count == 1
    second.assert_not_called()


def test_sidecar_survives_owner_crash_and_resume_uses_persisted_plan(
    tmp_path: Path,
) -> None:
    tool = Mock(return_value={"value": "must-not-run"})
    store = InterruptingStore(
        tmp_path,
        lambda state: (
            state.plan is not None
            and state.preflight_verification is None
            and state.lifecycle_status is RunLifecycleStatus.PLANNING
        ),
    )
    runtime, _, registry = _runtime(store, tool)
    with pytest.raises(SimulatedProcessExit):
        runtime.run(_request())
    store.request_cancellation("request-1:run")

    planner = FixedPlanner(_one_step_plan())
    result = AgentRuntime(
        planner=planner,
        registry=registry,
        run_store=FileRunStore(tmp_path),
    ).resume("request-1:run")

    assert result.status is RunStatus.CANCELLED
    assert result.plan == _one_step_plan()
    assert planner.calls == 0
    tool.assert_not_called()
    assert FileRunStore(tmp_path).load_cancellation(result.run_id) is not None


def test_cancellation_with_stale_running_preserves_interrupted(tmp_path: Path) -> None:
    tool = Mock(return_value={"value": "must-not-run"})
    store = InterruptingStore(
        tmp_path,
        lambda state: (
            state.steps and state.steps[0].status is StepStatus.RUNNING
        ),
    )
    runtime, _, registry = _runtime(store, tool)
    with pytest.raises(SimulatedProcessExit):
        runtime.run(_request())
    store.request_cancellation("request-1:run")

    result = AgentRuntime(
        planner=FixedPlanner(_one_step_plan()),
        registry=registry,
        run_store=FileRunStore(tmp_path),
    ).resume("request-1:run")
    state = store.load("request-1:run")

    assert result.status is RunStatus.FAILED
    assert state.lifecycle_status is RunLifecycleStatus.INTERRUPTED
    assert result.errors[0].code == "STEP_OUTCOME_UNKNOWN_AFTER_INTERRUPTION"
    assert any(
        event.event_type is TraceEventType.CANCELLATION_REQUESTED
        for event in result.trace
    )
    tool.assert_not_called()


def test_terminal_cancelled_resume_is_planner_and_tool_free(tmp_path: Path) -> None:
    tool = Mock(return_value={"value": "unused"})
    store = CancelOnCreateStore(tmp_path)
    runtime, planner, registry = _runtime(store, tool)
    first = runtime.run(_request())
    resume_planner = FixedPlanner(_one_step_plan())

    resumed = AgentRuntime(
        planner=resume_planner,
        registry=registry,
        run_store=FileRunStore(tmp_path),
    ).resume(first.run_id)

    assert resumed.to_dict() == first.to_dict()
    assert planner.calls == 0
    assert resume_planner.calls == 0
    tool.assert_not_called()


@pytest.mark.parametrize("terminal_kind", ["SUCCEEDED", "FAILED", "PLANNED"])
def test_cancel_normal_terminal_returns_already_terminal_without_sidecar(
    tmp_path: Path, terminal_kind: str
) -> None:
    if terminal_kind == "FAILED":
        tool = Mock(side_effect=RuntimeError("permanent"))
    else:
        tool = Mock(return_value={"value": "done"})
    store = FileRunStore(tmp_path)
    runtime, _, _ = _runtime(store, tool)
    mode = RunMode.PLAN_ONLY if terminal_kind == "PLANNED" else RunMode.EXECUTE
    result = runtime.run(_request(mode=mode))
    before = store.load(result.run_id)

    receipt = runtime.cancel(result.run_id)

    assert receipt.disposition is CancellationDisposition.ALREADY_TERMINAL
    assert receipt.terminal_status is RunLifecycleStatus[terminal_kind]
    assert store.load(result.run_id) == before
    assert not store.cancellation_path(result.run_id).exists()


def test_cancel_terminal_cancelled_returns_already_terminal(tmp_path: Path) -> None:
    store = CancelOnCreateStore(tmp_path)
    runtime, _, _ = _runtime(store, Mock(return_value={"value": "unused"}))
    result = runtime.run(_request())
    before = store.load(result.run_id)

    receipt = runtime.cancel(result.run_id)

    assert receipt.disposition is CancellationDisposition.ALREADY_TERMINAL
    assert receipt.terminal_status is RunLifecycleStatus.CANCELLED
    assert store.load(result.run_id) == before


def test_cancel_terminal_interrupted_returns_already_terminal(tmp_path: Path) -> None:
    tool = Mock(return_value={"value": "must-not-run"})
    store = InterruptingStore(
        tmp_path,
        lambda state: state.steps and state.steps[0].status is StepStatus.RUNNING,
    )
    runtime, _, registry = _runtime(store, tool)
    with pytest.raises(SimulatedProcessExit):
        runtime.run(_request())
    interrupted = AgentRuntime(
        planner=FixedPlanner(_one_step_plan()),
        registry=registry,
        run_store=FileRunStore(tmp_path),
    ).resume("request-1:run")

    receipt = FileRunStore(tmp_path).request_cancellation(interrupted.run_id)

    assert receipt.disposition is CancellationDisposition.ALREADY_TERMINAL
    assert receipt.terminal_status is RunLifecycleStatus.INTERRUPTED
    assert not store.cancellation_path(interrupted.run_id).exists()


def test_plan_only_cancellation_during_planner_executes_zero_tools(
    tmp_path: Path,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    tool = Mock(side_effect=AssertionError("PLAN_ONLY invoked a tool"))
    registry = ToolRegistry((_spec("first", tool),))

    class BlockingPlanner:
        calls = 0

        def plan(self, request, available_registry):
            self.calls += 1
            entered.set()
            assert release.wait(timeout=10)
            return _one_step_plan()

    planner = BlockingPlanner()
    store = FileRunStore(tmp_path)
    runtime = AgentRuntime(planner=planner, registry=registry, run_store=store)
    outcomes = []
    worker = threading.Thread(
        target=lambda: outcomes.append(runtime.run(_request(mode=RunMode.PLAN_ONLY))),
        daemon=True,
    )
    worker.start()
    assert entered.wait(timeout=10)
    runtime.cancel("request-1:run")
    release.set()
    worker.join(timeout=10)

    assert outcomes[0].status is RunStatus.CANCELLED
    assert outcomes[0].plan == _one_step_plan()
    assert planner.calls == 1
    tool.assert_not_called()


def test_terminal_planned_cancel_remains_planned(tmp_path: Path) -> None:
    tool = Mock(side_effect=AssertionError("PLAN_ONLY invoked a tool"))
    store = FileRunStore(tmp_path)
    runtime, _, _ = _runtime(store, tool)
    planned = runtime.run(_request(mode=RunMode.PLAN_ONLY))

    receipt = runtime.cancel(planned.run_id)
    resumed = runtime.resume(planned.run_id)

    assert receipt.disposition is CancellationDisposition.ALREADY_TERMINAL
    assert receipt.terminal_status is RunLifecycleStatus.PLANNED
    assert resumed.to_dict() == planned.to_dict()
    tool.assert_not_called()


@pytest.mark.parametrize("corruption", ["malformed", "bad_digest", "unsupported"])
def test_corrupt_cancellation_sidecar_fails_closed_before_tool(
    tmp_path: Path, corruption: str
) -> None:
    tool = Mock(return_value={"value": "must-not-run"})
    store = FileRunStore(tmp_path)
    registry = ToolRegistry((_spec("first", tool),))

    class CorruptingExecutor(PlanExecutor):
        def preflight(self, plan: AgentPlan):
            store.request_cancellation("request-1:run")
            path = store.cancellation_path("request-1:run")
            if corruption == "malformed":
                path.write_bytes(b"{")
            else:
                envelope = json.loads(path.read_text(encoding="utf-8"))
                if corruption == "bad_digest":
                    envelope["integrity"]["digest"] = "0" * 64
                else:
                    envelope["schema_version"] = 999
                path.write_bytes(_canonical(envelope))
            return super().preflight(plan)

    executor = CorruptingExecutor(registry)
    runtime = AgentRuntime(
        planner=FixedPlanner(_one_step_plan()),
        registry=registry,
        executor=executor,
        run_store=store,
    )

    expected_error = (
        CancellationStateVersionError
        if corruption == "unsupported"
        else CancellationStateCorruptionError
    )
    with pytest.raises(expected_error):
        runtime.run(_request())
    tool.assert_not_called()


def test_preexisting_cancel_wins_normal_failed_terminal_and_keeps_error(
    tmp_path: Path,
) -> None:
    store = FileRunStore(tmp_path)

    class CancellingFailingPlanner:
        def plan(self, request, registry):
            store.request_cancellation("request-1:run")
            raise PlannerError("UNSUPPORTED_REQUEST", "not supported")

    result = AgentRuntime(
        planner=CancellingFailingPlanner(),
        registry=ToolRegistry(()),
        run_store=store,
    ).run(_request())

    assert result.status is RunStatus.CANCELLED
    assert store.load(result.run_id).lifecycle_status is RunLifecycleStatus.CANCELLED
    assert any(error.code == "UNSUPPORTED_REQUEST" for error in result.errors)
    assert any(error.code == "RUN_CANCELLED" for error in result.errors)


def test_cancellation_after_run_verification_preserves_verification(
    tmp_path: Path, monkeypatch
) -> None:
    import agent.orchestration.runtime as runtime_module

    store = FileRunStore(tmp_path)
    tool = Mock(return_value={"value": "done"})
    original_verify_run = runtime_module.verify_run

    def verify_then_cancel(plan, steps):
        verification = original_verify_run(plan, steps)
        store.request_cancellation("request-1:run")
        return verification

    monkeypatch.setattr(runtime_module, "verify_run", verify_then_cancel)
    runtime, _, _ = _runtime(store, tool)

    result = runtime.run(_request())

    assert result.status is RunStatus.CANCELLED
    assert result.verification is not None
    assert result.verification.passed
    assert result.steps[0].status is StepStatus.SUCCEEDED
    assert tool.call_count == 1


def test_store_rejects_normal_terminal_commit_when_cancel_exists(
    tmp_path: Path,
) -> None:
    store = FileRunStore(tmp_path)
    state = store.create(_planning_state())
    store.request_cancellation(state.run_id)
    error = AgentError(
        ErrorCategory.INTERNAL_AGENT_ERROR,
        "FAILED_FOR_TEST",
        "failure",
    )
    terminal = replace(
        state,
        revision=1,
        updated_at=_now(),
        lifecycle_status=RunLifecycleStatus.FAILED,
        errors=(error,),
    )

    with pytest.raises(CancellationRequestedError):
        store.update(terminal, expected_revision=0)
    assert store.load(state.run_id).revision == 0


def test_cancellation_trace_records_request_observation_skip_and_terminal(
    tmp_path: Path,
) -> None:
    tool = Mock(return_value={"value": "unused"})
    runtime, _, _ = _runtime(CancelAfterPlanStore(tmp_path), tool)

    result = runtime.run(_request())
    event_types = [event.event_type for event in result.trace]

    assert TraceEventType.CANCELLATION_REQUESTED in event_types
    assert TraceEventType.CANCELLATION_OBSERVED in event_types
    assert TraceEventType.RUN_CANCELLED in event_types
    request_event = next(
        event
        for event in result.trace
        if event.event_type is TraceEventType.CANCELLATION_REQUESTED
    )
    terminal_event = next(
        event
        for event in result.trace
        if event.event_type is TraceEventType.RUN_CANCELLED
    )
    assert request_event.details["requested_at"] == terminal_event.details["requested_at"]


@pytest.mark.parametrize(
    "lifecycle_status",
    ["PLANNED", "SUCCEEDED", "FAILED", "INTERRUPTED"],
)
def test_literal_version_one_terminal_records_load_without_rewrite(
    tmp_path: Path,
    lifecycle_status: str,
) -> None:
    store = FileRunStore(tmp_path)
    timestamp = _now()
    record = _literal_v1_record(lifecycle_status, timestamp=timestamp)
    original = _write_literal_v1_record(store, record)
    path = store.state_path("request-1:run")

    loaded = store.load("request-1:run")

    assert loaded.schema_version == RUN_STATE_SCHEMA_VERSION == 3
    assert loaded.lifecycle_status.value == lifecycle_status
    assert loaded.lifecycle_status is not RunLifecycleStatus.CANCELLED
    assert path.read_bytes() == original


def test_version_one_record_rejects_cancelled_lifecycle(tmp_path: Path) -> None:
    store = FileRunStore(tmp_path)
    record = _literal_v1_record("FAILED", timestamp=_now())
    record["lifecycle_status"] = "CANCELLED"
    record["errors"] = [
        _literal_v1_error(category="CANCELLATION", code="RUN_CANCELLED")
    ]
    _write_literal_v1_record(store, record)

    with pytest.raises(
        RunStateVersionError,
        match="version 1 does not support lifecycle status 'CANCELLED'",
    ):
        store.load("request-1:run")


def test_version_one_record_rejects_cancellation_error_category(
    tmp_path: Path,
) -> None:
    store = FileRunStore(tmp_path)
    record = _literal_v1_record("FAILED", timestamp=_now())
    record["errors"] = [_literal_v1_error(category="CANCELLATION")]
    _write_literal_v1_record(store, record)

    with pytest.raises(
        RunStateVersionError,
        match="version 1 does not support error category 'CANCELLATION'",
    ):
        store.load("request-1:run")


def test_version_one_record_rejects_nested_cancellation_error_category(
    tmp_path: Path,
) -> None:
    store = FileRunStore(tmp_path)
    timestamp = _now()
    record = _literal_v1_record("INTERRUPTED", timestamp=timestamp)
    record["steps"] = [
        {
            **_literal_v1_pending_step(),
            "status": "SKIPPED",
            "error": _literal_v1_error(category="CANCELLATION"),
            "started_at": timestamp,
            "finished_at": timestamp,
            "duration_seconds": 0.0,
        }
    ]
    _write_literal_v1_record(store, record)

    with pytest.raises(
        RunStateVersionError,
        match="version 1 does not support error category 'CANCELLATION'",
    ):
        store.load("request-1:run")


@pytest.mark.parametrize(
    "event_type",
    ["CANCELLATION_REQUESTED", "CANCELLATION_OBSERVED", "RUN_CANCELLED"],
)
def test_version_one_record_rejects_cancellation_trace_events(
    tmp_path: Path,
    event_type: str,
) -> None:
    store = FileRunStore(tmp_path)
    timestamp = _now()
    record = _literal_v1_record("PLANNING", timestamp=timestamp)
    record["trace"] = [
        {
            "sequence": 0,
            "event_type": event_type,
            "timestamp": timestamp,
            "message": "v2-only cancellation event",
            "step_id": None,
            "attempt": None,
            "details": {},
        }
    ]
    _write_literal_v1_record(store, record)

    with pytest.raises(
        RunStateVersionError,
        match=f"version 1 does not support trace event type '{event_type}'",
    ):
        store.load("request-1:run")


def test_real_version_one_record_loads_without_rewrite_then_updates_to_v3(
    tmp_path: Path,
) -> None:
    store = FileRunStore(tmp_path)
    timestamp = _now()
    record = _literal_v1_record("PLANNING", timestamp=timestamp)
    record["request"]["mode"] = "PLAN_ONLY"
    record["trace"] = [
        {
            "sequence": 0,
            "event_type": "PLANNING",
            "timestamp": timestamp,
            "message": "legacy planning",
            "step_id": None,
            "attempt": None,
            "details": {"schema": 1},
        }
    ]
    original = _write_literal_v1_record(store, record)
    path = store.state_path("request-1:run")

    loaded = store.load("request-1:run")

    assert loaded.schema_version == RUN_STATE_SCHEMA_VERSION == 3
    assert path.read_bytes() == original
    original_fingerprint = loaded.plan_fingerprint
    original_trace = loaded.trace
    updated = replace(loaded, revision=1, updated_at=_now())
    store.update(updated, expected_revision=0)
    rewritten = json.loads(path.read_text(encoding="utf-8"))
    assert rewritten["schema_version"] == 3
    assert rewritten["record"]["schema_version"] == 3
    assert rewritten["record"]["revision"] == 1
    assert rewritten["record"]["run_id"] == "request-1:run"
    assert rewritten["record"]["plan_fingerprint"] == original_fingerprint
    assert rewritten["record"]["trace"] == [
        event.to_dict() for event in original_trace
    ]
    reloaded = store.load("request-1:run")
    assert reloaded.request == loaded.request
    assert reloaded.plan == loaded.plan
    assert reloaded.lifecycle_status is RunLifecycleStatus.PLANNING
