"""Offline acceptance tests for production error classification semantics."""

from __future__ import annotations

import errno
import json
from pathlib import Path

import pytest

from agent.orchestration import (
    AgentError,
    AgentRequest,
    AgentRuntime,
    ErrorCategory,
    FileRunStore,
    RecoveryDisposition,
    LLMPlanner,
    PlannerError,
    PlanningModelError,
    RunMode,
    build_default_tool_registry,
    classify_provider_exception,
)
from agent.orchestration.error_policy import (
    BUILTIN_ERROR_CODES,
    ERROR_POLICY_CATALOG,
    classified_agent_error,
    policy_entry,
    recovery_disposition_for,
)


def test_every_builtin_policy_code_has_explicit_disposition() -> None:
    assert BUILTIN_ERROR_CODES == frozenset(ERROR_POLICY_CATALOG)
    assert BUILTIN_ERROR_CODES
    assert all(
        isinstance(entry.recovery_disposition, RecoveryDisposition)
        for entry in ERROR_POLICY_CATALOG.values()
    )


def test_unknown_code_fails_closed() -> None:
    entry = policy_entry("UNRECOGNIZED_FUTURE_CODE")

    assert entry.recovery_disposition is RecoveryDisposition.NO_AUTOMATIC_RECOVERY
    assert (
        recovery_disposition_for("UNRECOGNIZED_FUTURE_CODE")
        is RecoveryDisposition.NO_AUTOMATIC_RECOVERY
    )


def test_recoverable_means_only_static_same_step_eligibility() -> None:
    error = AgentError(
        ErrorCategory.TOOL_EXECUTION_ERROR,
        "TRANSIENT_TEST_FAILURE",
        "safe",
        recoverable=True,
    )

    assert error.recoverable is True
    assert (
        error.recovery_disposition
        is RecoveryDisposition.SAME_STEP_RETRY_ELIGIBLE
    )


def test_user_input_failure_requires_user_action_without_auto_retry() -> None:
    error = classified_agent_error(
        category=ErrorCategory.USER_INPUT_ERROR,
        code="INVALID_ARGUMENT",
    )

    assert error.recoverable is False
    assert error.recovery_disposition is RecoveryDisposition.USER_ACTION_REQUIRED


def test_runtime_user_input_failure_is_terminal_and_user_action_required() -> None:
    result = AgentRuntime().run(
        AgentRequest("request-1", "inspect this dataset", {}, RunMode.EXECUTE)
    )

    assert result.status.value == "FAILED"
    assert result.errors[0].code == "MISSING_REQUIRED_INPUT"
    assert result.errors[0].recoverable is False
    assert (
        result.errors[0].recovery_disposition
        is RecoveryDisposition.USER_ACTION_REQUIRED
    )


def test_arbitrary_tool_exception_text_is_not_persistable_message() -> None:
    secret = "Bearer scientific-secret; private backend payload"
    error = build_default_tool_registry().classify_exception(
        "epizoo_embed_cells",
        RuntimeError(secret),
        step_id="embed",
        attempt=1,
    )

    assert error.code == "TOOL_RUNTIME_ERROR"
    assert secret not in error.message
    assert "private backend" not in error.message
    assert error.exception_type == "RuntimeError"


def test_cuda_oom_is_safe_resource_error_and_not_retryable() -> None:
    cuda_oom_type = type(
        "OutOfMemoryError",
        (RuntimeError,),
        {"__module__": "torch.cuda"},
    )
    error = build_default_tool_registry().classify_exception(
        "epizoo_embed_cells",
        cuda_oom_type("secret allocator state"),
        step_id="embed",
        attempt=1,
    )

    assert error.category is ErrorCategory.RESOURCE_ERROR
    assert error.code == "CUDA_OUT_OF_MEMORY"
    assert error.recoverable is False
    assert error.recovery_disposition is RecoveryDisposition.USER_ACTION_REQUIRED
    assert "allocator" not in error.message


def test_host_memory_and_artifact_write_failures_are_safely_classified() -> None:
    registry = build_default_tool_registry()

    memory = registry.classify_exception("inspect_scATAC", MemoryError("secret"))
    write = registry.classify_exception(
        "epizoo_embed_cells",
        OSError(errno.EACCES, "private output location"),
    )
    disk = registry.classify_exception(
        "epizoo_embed_cells",
        OSError(errno.ENOSPC, "private filesystem detail"),
    )

    assert (memory.category, memory.code) == (
        ErrorCategory.RESOURCE_ERROR,
        "HOST_MEMORY_EXHAUSTED",
    )
    assert (write.category, write.code) == (
        ErrorCategory.RESOURCE_ERROR,
        "ARTIFACT_WRITE_FAILED",
    )
    assert disk.code == "DISK_FULL"
    assert "private" not in write.message
    assert "private" not in disk.message


@pytest.mark.parametrize(
    ("exception", "expected"),
    [
        (type("AuthenticationError", (RuntimeError,), {})(), "PROVIDER_AUTHENTICATION_FAILED"),
        (
            type("RequestTooLarge", (RuntimeError,), {"status_code": 413})(),
            "PROVIDER_REQUEST_TOO_LARGE",
        ),
        (type("RateLimitError", (RuntimeError,), {})(), "PROVIDER_RATE_LIMITED"),
        (TimeoutError(), "PROVIDER_TIMEOUT"),
        (ConnectionError(), "PROVIDER_CONNECTION_FAILED"),
        (type("ServerFailure", (RuntimeError,), {"status_code": 503})(), "PROVIDER_UNAVAILABLE"),
        (RuntimeError("unsafe body"), "PLANNING_PROVIDER_ERROR"),
    ],
)
def test_provider_classification_uses_structured_exception_information(
    exception: Exception,
    expected: str,
) -> None:
    code, message = classify_provider_exception(exception)

    assert code == expected
    assert "unsafe body" not in message


def test_provider_neutral_failure_reaches_runtime_with_safe_stable_code() -> None:
    class Model:
        model_id = "fake"

        def complete(self, *, prompt, response_schema):
            raise PlanningModelError(
                "Planning-provider authentication failed.",
                code="PROVIDER_AUTHENTICATION_FAILED",
            ) from RuntimeError("Bearer provider-secret")

    result = AgentRuntime(
        planner=LLMPlanner(Model()),
        registry=build_default_tool_registry(),
    ).run(
        AgentRequest(
            "request-1",
            "inspect input",
            {"input_path": "/data/example.h5ad"},
            RunMode.PLAN_ONLY,
        )
    )

    assert result.errors[0].code == "PROVIDER_AUTHENTICATION_FAILED"
    assert "provider-secret" not in result.errors[0].message
    assert (
        result.errors[0].recovery_disposition
        is RecoveryDisposition.USER_ACTION_REQUIRED
    )


class _UnknownFailurePlanner:
    def __init__(self, secret: str) -> None:
        self._secret = secret

    def plan(self, request, registry):
        raise PlannerError("CUSTOM_PROVIDER_FAILURE", self._secret)


def test_unknown_planner_error_is_sanitized_without_run_store() -> None:
    secret = "Bearer unknown-no-store-secret"

    result = AgentRuntime(planner=_UnknownFailurePlanner(secret)).run(
        AgentRequest("unknown-no-store", "inspect input", {})
    )
    serialized = json.dumps(result.to_dict(), sort_keys=True)

    assert result.errors[0].code == "CUSTOM_PROVIDER_FAILURE"
    assert result.errors[0].message == "The operation failed."
    assert result.errors[0].details == {}
    assert result.steps == ()
    assert secret not in serialized
    assert secret not in json.dumps(
        [event.to_dict() for event in result.trace], sort_keys=True
    )


def test_unknown_planner_error_is_sanitized_in_durable_state(
    tmp_path: Path,
) -> None:
    secret = "Bearer unknown-durable-secret"
    store = FileRunStore(tmp_path)

    result = AgentRuntime(
        planner=_UnknownFailurePlanner(secret),
        run_store=store,
    ).run(AgentRequest("unknown-durable", "inspect input", {}))
    state = store.load(result.run_id)
    persisted = store.state_path(result.run_id).read_text(encoding="utf-8")

    assert result.errors[0].code == "CUSTOM_PROVIDER_FAILURE"
    assert result.errors[0].message == "The operation failed."
    assert result.errors[0].details == {}
    assert result.steps == ()
    assert secret not in json.dumps(result.to_dict(), sort_keys=True)
    assert secret not in json.dumps(state.to_dict(), sort_keys=True)
    assert secret not in persisted
