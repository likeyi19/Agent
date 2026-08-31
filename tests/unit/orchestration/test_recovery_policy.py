"""Offline acceptance tests for retry provenance and deterministic resume policy."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pytest

from agent.orchestration import (
    AgentPlan,
    AgentRequest,
    AgentRuntime,
    ArgumentSpec,
    ErrorCategory,
    ErrorClassification,
    FileRunStore,
    PlanExecutor,
    PlanStep,
    PersistedRunState,
    RecoveryPolicy,
    RecoveryPolicyIncompatibleError,
    RecoveryPolicyUnknownError,
    RecoveryDisposition,
    ResultContract,
    RunLifecycleStatus,
    RunMode,
    RunNotFoundError,
    RunStateConflictError,
    RunStateCorruptionError,
    StepExecutionResult,
    StepStatus,
    ToolRegistry,
    ToolSpec,
    TraceEventType,
)
from agent.orchestration.error_policy import build_recovery_policy_snapshot
from agent.schemas import AgentError, RUN_STATE_SCHEMA_VERSION, fingerprint_plan


class TransientFailure(RuntimeError):
    pass


def _classify(exception: Exception) -> ErrorClassification:
    code = "TRANSIENT" if isinstance(exception, TransientFailure) else "PERMANENT"
    return ErrorClassification(ErrorCategory.TOOL_EXECUTION_ERROR, code)


def _spec(
    function,
    *,
    retryable: frozenset[str] = frozenset(),
    version: str = "test-v1",
    arguments: dict[str, ArgumentSpec] | None = None,
) -> ToolSpec:
    return ToolSpec(
        "tool",
        function,
        arguments or {},
        {},
        ResultContract("ToolResult", {"value": (str,)}),
        _classify,
        retryable,
        version,
    )


def _plan(arguments: dict[str, object] | None = None) -> AgentPlan:
    return AgentPlan(
        "plan-1",
        "request-1",
        "fixed",
        (PlanStep("step", "tool", arguments or {}),),
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validated_state(
    store: FileRunStore,
    registry: ToolRegistry,
    executor: PlanExecutor,
) -> None:
    plan = _plan()
    timestamp = _now()
    store.create(
        state := PersistedRunState(
            schema_version=RUN_STATE_SCHEMA_VERSION,
            revision=0,
            run_id="request-1:run",
            request=AgentRequest("request-1", "fixed", {}),
            lifecycle_status=RunLifecycleStatus.VALIDATED,
            created_at=timestamp,
            updated_at=timestamp,
            plan=plan,
            plan_fingerprint=fingerprint_plan(plan),
            recovery_policy_snapshot=build_recovery_policy_snapshot(
                plan,
                registry,
                max_attempts_per_step=(
                    executor.recovery_policy.max_attempts_per_step
                ),
            ),
            preflight_verification=executor.preflight(plan),
            steps=(
                StepExecutionResult("step", "tool", StepStatus.PENDING),
            ),
        )
    )
    assert state.recovery_policy_snapshot is not None


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _rewrite_as_legacy(path: Path, version: int = 2) -> bytes:
    envelope = json.loads(path.read_text(encoding="utf-8"))
    record = envelope["record"]
    record["schema_version"] = version
    record.pop("recovery_policy_snapshot")
    envelope["schema_version"] = version
    envelope["integrity"]["digest"] = hashlib.sha256(_canonical(record)).hexdigest()
    payload = _canonical(envelope)
    path.write_bytes(payload)
    return payload


def test_retry_uses_fresh_canonical_equivalent_nested_arguments() -> None:
    observed: list[tuple[dict[str, object], type, type]] = []

    def tool(payload):
        observed.append((dict(payload["nested"]), type(payload), type(payload["nested"])))
        if len(observed) == 1:
            payload["nested"]["mutated"] = True
            raise TransientFailure("private transient detail")
        return {"value": "done"}

    spec = _spec(
        tool,
        retryable=frozenset({"TRANSIENT"}),
        arguments={"payload": ArgumentSpec((Mapping,))},
    )
    outcome = PlanExecutor(
        ToolRegistry((spec,)),
        recovery_policy=RecoveryPolicy(max_attempts_per_step=2),
    ).execute(_plan({"payload": {"nested": {"value": 1}}}))

    assert [item[0] for item in observed] == [{"value": 1}, {"value": 1}]
    assert observed[0][1:] == observed[1][1:] == (dict, dict)
    assert outcome.step_results[0].status is StepStatus.SUCCEEDED
    assert outcome.step_results[0].attempt_count == 2


def test_retry_exhaustion_preserves_error_and_records_policy_details() -> None:
    calls = 0

    def tool():
        nonlocal calls
        calls += 1
        raise TransientFailure("private backend detail")

    registry = ToolRegistry(
        (_spec(tool, retryable=frozenset({"TRANSIENT"})),)
    )
    outcome = PlanExecutor(
        registry,
        recovery_policy=RecoveryPolicy(max_attempts_per_step=2),
    ).execute(_plan())

    error = outcome.errors[0]
    assert calls == 2
    assert (error.category, error.code) == (
        ErrorCategory.TOOL_EXECUTION_ERROR,
        "TRANSIENT",
    )
    assert error.recoverable is True
    assert error.details["retry_exhausted"] is True
    assert error.details["attempts"] == 2
    assert error.details["max_attempts"] == 2
    assert isinstance(error.details["recovery_policy_fingerprint"], str)
    recovery = [
        event for event in outcome.trace if event.event_type is TraceEventType.RECOVERY
    ]
    assert recovery[-1].details["decision"] == "stop_attempt_limit"
    assert recovery[-1].details["reason"] == "attempt_limit_reached"
    assert recovery[-1].details["retry_exhausted"] is True
    assert "private backend detail" not in error.message


def test_nonretryable_tool_and_verification_failure_execute_once() -> None:
    permanent_calls = 0

    def permanent():
        nonlocal permanent_calls
        permanent_calls += 1
        raise RuntimeError("private")

    permanent_outcome = PlanExecutor(ToolRegistry((_spec(permanent),))).execute(
        _plan()
    )
    assert permanent_calls == 1
    assert permanent_outcome.step_results[0].attempt_count == 1

    verify_calls = 0

    def malformed():
        nonlocal verify_calls
        verify_calls += 1
        return {"wrong": "shape"}

    verify_outcome = PlanExecutor(ToolRegistry((_spec(malformed),))).execute(_plan())
    assert verify_calls == 1
    assert any(
        event.details.get("decision") == "stop_verification_failure"
        for event in verify_outcome.trace
        if event.event_type is TraceEventType.RECOVERY
    )


def test_durable_verifier_exception_message_is_sanitized(tmp_path: Path) -> None:
    secret = "Bearer durable-verifier-secret"
    calls = 0

    def tool():
        nonlocal calls
        calls += 1
        return {"value": "done"}

    def reject_result(result):
        raise ValueError(secret)

    spec = ToolSpec(
        "tool",
        tool,
        {},
        {},
        ResultContract(
            "ToolResult",
            {"value": (str,)},
            validator=reject_result,
        ),
        _classify,
        frozenset(),
        "test-v1",
    )
    registry = ToolRegistry((spec,))

    class Planner:
        def plan(self, request, registry):
            return _plan()

    store = FileRunStore(tmp_path)
    result = AgentRuntime(
        planner=Planner(),
        registry=registry,
        run_store=store,
    ).run(AgentRequest("request-1", "fixed", {}))
    state = store.load(result.run_id)
    persisted = store.state_path(result.run_id).read_text(encoding="utf-8")
    verification = result.steps[0].verification

    assert calls == 1
    assert result.status.value == "FAILED"
    assert result.steps[0].attempt_count == 1
    assert verification is not None and not verification.passed
    assert verification.error is not None
    assert verification.error.message == (
        "Result violates the authoritative registry contract."
    )
    assert result.errors[0].recoverable is False
    assert not any(
        event.details.get("decision") == "retry_same_arguments"
        for event in result.trace
    )
    assert secret not in json.dumps(verification.to_dict(), sort_keys=True)
    assert secret not in json.dumps(result.steps[0].to_dict(), sort_keys=True)
    assert secret not in json.dumps(result.errors[0].to_dict(), sort_keys=True)
    assert secret not in json.dumps(
        [event.to_dict() for event in result.trace], sort_keys=True
    )
    assert secret not in json.dumps(state.to_dict(), sort_keys=True)
    assert secret not in persisted


def test_cuda_oom_does_not_mutate_execution_arguments() -> None:
    observed: list[tuple[str, int, str]] = []
    cuda_oom_type = type(
        "OutOfMemoryError",
        (RuntimeError,),
        {"__module__": "torch.cuda"},
    )

    def tool(device: str, batch_size: int, dtype: str):
        observed.append((device, batch_size, dtype))
        raise cuda_oom_type("private allocator state")

    def classify(exception: Exception) -> ErrorClassification:
        assert type(exception).__name__ == "OutOfMemoryError"
        return ErrorClassification(ErrorCategory.RESOURCE_ERROR, "CUDA_OUT_OF_MEMORY")

    spec = ToolSpec(
        "tool",
        tool,
        {
            "device": ArgumentSpec((str,)),
            "batch_size": ArgumentSpec((int,)),
            "dtype": ArgumentSpec((str,)),
        },
        {},
        ResultContract("ToolResult", {"value": (str,)}),
        classify,
        frozenset(),
        "cuda-policy-v1",
    )
    plan = _plan(
        {"device": "cuda:0", "batch_size": 4, "dtype": "float32"}
    )

    outcome = PlanExecutor(ToolRegistry((spec,))).execute(plan)

    assert observed == [("cuda:0", 4, "float32")]
    assert outcome.errors[0].code == "CUDA_OUT_OF_MEMORY"
    assert outcome.errors[0].recoverable is False
    assert outcome.step_results[0].resolved_arguments == {
        "device": "cuda:0",
        "batch_size": 4,
        "dtype": "float32",
    }


def test_new_execute_run_persists_policy_before_tool_invocation(tmp_path: Path) -> None:
    store = FileRunStore(tmp_path)
    observed = []

    def tool():
        observed.append(store.load("request-1:run"))
        return {"value": "done"}

    registry = ToolRegistry((_spec(tool),))

    class Planner:
        def plan(self, request, registry):
            return _plan()

    result = AgentRuntime(
        planner=Planner(), registry=registry, run_store=store
    ).run(AgentRequest("request-1", "fixed", {}))

    assert result.status.value == "SUCCEEDED"
    assert len(observed) == 1
    assert observed[0].recovery_policy_snapshot is not None
    assert observed[0].lifecycle_status is RunLifecycleStatus.RUNNING


def test_policy_snapshot_fingerprint_is_deterministic() -> None:
    registry = ToolRegistry((_spec(lambda: {"value": "done"}),))

    first = build_recovery_policy_snapshot(
        _plan(), registry, max_attempts_per_step=2
    )
    second = build_recovery_policy_snapshot(
        _plan(), registry, max_attempts_per_step=2
    )

    assert first == second
    assert first.fingerprint == second.fingerprint


def test_compatible_v3_resume_succeeds(tmp_path: Path) -> None:
    calls = 0

    def tool():
        nonlocal calls
        calls += 1
        return {"value": "done"}

    registry = ToolRegistry((_spec(tool),))
    executor = PlanExecutor(registry)
    store = FileRunStore(tmp_path)
    _validated_state(store, registry, executor)

    result = AgentRuntime(
        registry=registry, executor=executor, run_store=store
    ).resume("request-1:run")

    assert result.status.value == "SUCCEEDED"
    assert calls == 1


@pytest.mark.parametrize("drift", ["attempts", "codes", "version"])
def test_resume_policy_drift_is_rejected_without_mutation_or_calls(
    tmp_path: Path,
    drift: str,
) -> None:
    original_spec = _spec(lambda: {"value": "original"})
    original_registry = ToolRegistry((original_spec,))
    original_executor = PlanExecutor(original_registry)
    store = FileRunStore(tmp_path)
    _validated_state(store, original_registry, original_executor)
    path = store.state_path("request-1:run")
    before = path.read_bytes()
    calls = 0

    def tool():
        nonlocal calls
        calls += 1
        return {"value": "changed"}

    if drift == "attempts":
        spec = _spec(tool)
        executor = PlanExecutor(
            ToolRegistry((spec,)),
            recovery_policy=RecoveryPolicy(max_attempts_per_step=3),
        )
    elif drift == "codes":
        spec = _spec(tool, retryable=frozenset({"TRANSIENT"}))
        executor = PlanExecutor(ToolRegistry((spec,)))
    else:
        spec = _spec(tool, version="test-v2")
        executor = PlanExecutor(ToolRegistry((spec,)))

    with pytest.raises(RecoveryPolicyIncompatibleError):
        AgentRuntime(
            registry=executor.registry,
            executor=executor,
            run_store=store,
        ).resume("request-1:run")

    assert calls == 0
    assert path.read_bytes() == before


def test_attempt_count_above_persisted_bound_is_corruption(tmp_path: Path) -> None:
    registry = ToolRegistry((_spec(lambda: {"value": "done"}),))
    executor = PlanExecutor(registry)
    store = FileRunStore(tmp_path)
    _validated_state(store, registry, executor)
    path = store.state_path("request-1:run")
    envelope = json.loads(path.read_text(encoding="utf-8"))
    step = envelope["record"]["steps"][0]
    step.update(
        {
            "status": "RUNNING",
            "attempt_count": 3,
            "started_at": _now(),
        }
    )
    envelope["record"]["lifecycle_status"] = "RUNNING"
    envelope["integrity"]["digest"] = hashlib.sha256(
        _canonical(envelope["record"])
    ).hexdigest()
    path.write_bytes(_canonical(envelope))

    with pytest.raises(RunStateCorruptionError, match="attempt count exceeds"):
        store.load("request-1:run")


def _failed_error_state(*, recoverable: bool = False) -> PersistedRunState:
    timestamp = _now()
    return PersistedRunState(
        schema_version=RUN_STATE_SCHEMA_VERSION,
        revision=0,
        run_id="error-state:run",
        request=AgentRequest("error-state", "fixed", {}),
        lifecycle_status=RunLifecycleStatus.FAILED,
        created_at=timestamp,
        updated_at=timestamp,
        errors=(
            AgentError(
                ErrorCategory.TOOL_EXECUTION_ERROR,
                "TRANSIENT" if recoverable else "PERMANENT",
                "A safe failure occurred.",
                recoverable=recoverable,
            ),
        ),
    )


@pytest.mark.parametrize(
    ("recoverable", "disposition"),
    [
        (True, RecoveryDisposition.NO_AUTOMATIC_RECOVERY.value),
        (False, RecoveryDisposition.SAME_STEP_RETRY_ELIGIBLE.value),
    ],
)
def test_v3_contradictory_error_recovery_fields_are_corruption(
    tmp_path: Path,
    recoverable: bool,
    disposition: str,
) -> None:
    store = FileRunStore(tmp_path)
    state = store.create(_failed_error_state())
    path = store.state_path(state.run_id)
    envelope = json.loads(path.read_text(encoding="utf-8"))
    error = envelope["record"]["errors"][0]
    error["recoverable"] = recoverable
    error["recovery_disposition"] = disposition
    envelope["integrity"]["digest"] = hashlib.sha256(
        _canonical(envelope["record"])
    ).hexdigest()
    path.write_bytes(_canonical(envelope))

    with pytest.raises(
        RunStateCorruptionError,
        match="recovery fields are inconsistent",
    ):
        store.load(state.run_id)


def test_v3_matching_error_recovery_fields_load_unchanged(tmp_path: Path) -> None:
    store = FileRunStore(tmp_path)
    original = store.create(_failed_error_state(recoverable=True))

    loaded = store.load(original.run_id)

    assert loaded.errors[0].recoverable is True
    assert (
        loaded.errors[0].recovery_disposition
        is RecoveryDisposition.SAME_STEP_RETRY_ELIGIBLE
    )
    assert loaded.errors[0] == original.errors[0]


@pytest.mark.parametrize("source_schema_version", [1, 2])
def test_create_rejects_spoofed_legacy_source_without_writing_state(
    tmp_path: Path,
    source_schema_version: int,
) -> None:
    store = FileRunStore(tmp_path)
    plan = _plan()
    timestamp = _now()
    state = PersistedRunState(
        schema_version=RUN_STATE_SCHEMA_VERSION,
        revision=0,
        run_id="request-1:run",
        request=AgentRequest("request-1", "fixed", {}),
        lifecycle_status=RunLifecycleStatus.PLANNING,
        created_at=timestamp,
        updated_at=timestamp,
        plan=plan,
        plan_fingerprint=fingerprint_plan(plan),
        source_schema_version=source_schema_version,
    )

    with pytest.raises(
        RunStateConflictError,
        match="current-schema provenance",
    ):
        store.create(state)

    assert not store.state_path(state.run_id).exists()
    with pytest.raises(RunNotFoundError):
        store.load(state.run_id)
    assert not store.state_path(state.run_id).exists()


@pytest.mark.parametrize("legacy_version", [1, 2])
def test_legacy_nonterminal_execute_resume_is_rejected_without_mutation(
    tmp_path: Path,
    legacy_version: int,
) -> None:
    calls = 0

    def tool():
        nonlocal calls
        calls += 1
        return {"value": "done"}

    registry = ToolRegistry((_spec(tool),))
    executor = PlanExecutor(registry)
    store = FileRunStore(tmp_path)
    _validated_state(store, registry, executor)
    path = store.state_path("request-1:run")
    before = _rewrite_as_legacy(path, legacy_version)

    with pytest.raises(RecoveryPolicyUnknownError):
        AgentRuntime(
            registry=registry, executor=executor, run_store=store
        ).resume("request-1:run")

    assert calls == 0
    assert path.read_bytes() == before


def test_terminal_legacy_v2_record_remains_readable_and_idempotent(
    tmp_path: Path,
) -> None:
    calls = 0

    def tool():
        nonlocal calls
        calls += 1
        return {"value": "done"}

    registry = ToolRegistry((_spec(tool),))

    class Planner:
        def plan(self, request, registry):
            return _plan()

    store = FileRunStore(tmp_path)
    runtime = AgentRuntime(planner=Planner(), registry=registry, run_store=store)
    original = runtime.run(AgentRequest("request-1", "fixed", {}))
    _rewrite_as_legacy(store.state_path(original.run_id))

    resumed = runtime.resume(original.run_id)

    assert resumed.status == original.status
    assert calls == 1


def test_plan_only_has_no_scientific_policy_and_remains_zero_tool(
    tmp_path: Path,
) -> None:
    calls = 0

    def tool():
        nonlocal calls
        calls += 1
        return {"value": "done"}

    registry = ToolRegistry((_spec(tool),))

    class Planner:
        def plan(self, request, registry):
            return _plan()

    store = FileRunStore(tmp_path)
    runtime = AgentRuntime(planner=Planner(), registry=registry, run_store=store)
    result = runtime.run(
        AgentRequest("request-1", "fixed", {}, RunMode.PLAN_ONLY)
    )
    state = store.load(result.run_id)
    assert state.recovery_policy_snapshot is None
    _rewrite_as_legacy(store.state_path(result.run_id))

    resumed = runtime.resume(result.run_id)

    assert resumed.status.value == "PLANNED"
    assert calls == 0
