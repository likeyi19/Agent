"""Strict, versioned contracts for durable orchestration run state."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import hashlib
import json
from typing import Mapping, Sequence, cast

from .orchestration import (
    AgentError,
    AgentPlan,
    AgentRequest,
    AgentRunResult,
    ErrorCategory,
    ExecutionTraceEvent,
    JsonValue,
    PlanStep,
    RunMode,
    RunStatus,
    StepExecutionResult,
    StepOutputRef,
    StepStatus,
    TraceEventType,
    VerificationCheck,
    VerificationResult,
)


RUN_STATE_SCHEMA_VERSION = 2
LEGACY_RUN_STATE_SCHEMA_VERSIONS = frozenset({1})
CANCELLATION_STATE_SCHEMA_VERSION = 1


class RunLifecycleStatus(str, Enum):
    """Durability-specific lifecycle states, including nonterminal execution."""

    PLANNING = "PLANNING"
    VALIDATED = "VALIDATED"
    RUNNING = "RUNNING"
    PLANNED = "PLANNED"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"
    CANCELLED = "CANCELLED"


_LEGACY_V1_LIFECYCLE_STATUSES = frozenset(
    {
        RunLifecycleStatus.PLANNING,
        RunLifecycleStatus.VALIDATED,
        RunLifecycleStatus.RUNNING,
        RunLifecycleStatus.PLANNED,
        RunLifecycleStatus.SUCCEEDED,
        RunLifecycleStatus.FAILED,
        RunLifecycleStatus.INTERRUPTED,
    }
)
_LEGACY_V1_ERROR_CATEGORIES = frozenset(
    {
        ErrorCategory.USER_INPUT_ERROR,
        ErrorCategory.RESOURCE_ERROR,
        ErrorCategory.ENVIRONMENT_ERROR,
        ErrorCategory.TOOL_EXECUTION_ERROR,
        ErrorCategory.VERIFICATION_ERROR,
        ErrorCategory.INTERNAL_AGENT_ERROR,
    }
)
_LEGACY_V1_TRACE_EVENT_TYPES = frozenset(
    {
        TraceEventType.PLANNING,
        TraceEventType.PLAN_VALIDATION,
        TraceEventType.STEP_EXECUTION,
        TraceEventType.VERIFICATION,
        TraceEventType.RECOVERY,
        TraceEventType.STEP_SKIPPED,
        TraceEventType.RUN_COMPLETION,
    }
)


class CancellationDisposition(str, Enum):
    """Outcome of one durable cancellation request."""

    REQUESTED = "REQUESTED"
    ALREADY_REQUESTED = "ALREADY_REQUESTED"
    ALREADY_TERMINAL = "ALREADY_TERMINAL"


@dataclass(frozen=True)
class CancellationRequest:
    """Minimal durable cancellation intent stored outside main run revisions."""

    schema_version: int
    run_id: str
    requested_at: str

    def __post_init__(self) -> None:
        if self.schema_version != CANCELLATION_STATE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported cancellation schema version {self.schema_version}."
            )
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("`run_id` must be a non-empty string.")
        _parse_aware_timestamp(self.requested_at, "requested_at")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "requested_at": self.requested_at,
        }

    @classmethod
    def from_dict(cls, value: object) -> CancellationRequest:
        mapping = _mapping(value, "cancellation request")
        _exact_keys(
            mapping,
            {"schema_version", "run_id", "requested_at"},
            "cancellation request",
        )
        return cls(
            schema_version=_integer(mapping["schema_version"], "schema_version"),
            run_id=_string(mapping["run_id"], "run_id"),
            requested_at=_string(mapping["requested_at"], "requested_at"),
        )


@dataclass(frozen=True)
class CancellationReceipt:
    """JSON-safe result of requesting cooperative cancellation."""

    run_id: str
    disposition: CancellationDisposition
    requested_at: str | None = None
    terminal_status: RunLifecycleStatus | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("`run_id` must be a non-empty string.")
        if not isinstance(self.disposition, CancellationDisposition):
            raise TypeError("`disposition` must be a CancellationDisposition.")
        if self.requested_at is not None:
            _parse_aware_timestamp(self.requested_at, "requested_at")
        if self.terminal_status is not None and not isinstance(
            self.terminal_status, RunLifecycleStatus
        ):
            raise TypeError("`terminal_status` must be a RunLifecycleStatus or None.")
        if self.disposition in {
            CancellationDisposition.REQUESTED,
            CancellationDisposition.ALREADY_REQUESTED,
        }:
            if self.requested_at is None or self.terminal_status is not None:
                raise ValueError(
                    "A requested cancellation requires a timestamp and no terminal status."
                )
        elif self.terminal_status is None:
            raise ValueError("ALREADY_TERMINAL requires a terminal status.")
        elif self.requested_at is not None:
            raise ValueError("ALREADY_TERMINAL cannot contain a request timestamp.")
        elif self.terminal_status not in {
            RunLifecycleStatus.PLANNED,
            RunLifecycleStatus.SUCCEEDED,
            RunLifecycleStatus.FAILED,
            RunLifecycleStatus.INTERRUPTED,
            RunLifecycleStatus.CANCELLED,
        }:
            raise ValueError("ALREADY_TERMINAL requires a terminal lifecycle status.")

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "disposition": self.disposition.value,
            "requested_at": self.requested_at,
            "terminal_status": (
                None if self.terminal_status is None else self.terminal_status.value
            ),
        }


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def fingerprint_plan(plan: AgentPlan) -> str:
    """Return the canonical SHA-256 identity of an AgentPlan."""

    if not isinstance(plan, AgentPlan):
        raise TypeError("`plan` must be an AgentPlan.")
    return hashlib.sha256(_canonical_json_bytes(plan.to_dict())).hexdigest()


def _parse_aware_timestamp(value: object, field_name: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"`{field_name}` must be a non-empty timestamp string.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"`{field_name}` must be a valid ISO timestamp.") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"`{field_name}` must include a timezone.")
    return parsed


def _require_aware_timestamp(value: object, field_name: str) -> str:
    _parse_aware_timestamp(value, field_name)
    assert isinstance(value, str)
    return value


def _step_payload_is_clean(result: StepExecutionResult) -> bool:
    if result.status is StepStatus.PENDING:
        return (
            result.attempt_count == 0
            and not result.resolved_arguments
            and result.result is None
            and result.verification is None
            and result.error is None
            and result.started_at is None
            and result.finished_at is None
            and result.duration_seconds is None
        )
    if result.status is StepStatus.RUNNING:
        return (
            result.attempt_count > 0
            and result.result is None
            and result.verification is None
            and result.error is None
            and result.started_at is not None
            and result.finished_at is None
            and result.duration_seconds is None
        )
    if result.status is StepStatus.SUCCEEDED:
        return (
            result.attempt_count > 0
            and result.result is not None
            and result.verification is not None
            and result.verification.passed
            and result.error is None
            and result.started_at is not None
            and result.finished_at is not None
            and result.duration_seconds is not None
        )
    if result.status is StepStatus.FAILED:
        return result.error is not None and result.finished_at is not None
    if result.status is StepStatus.SKIPPED:
        return (
            result.attempt_count == 0
            and result.result is None
            and result.error is not None
            and result.started_at is not None
            and result.finished_at is not None
        )
    return False


@dataclass(frozen=True)
class PersistedRunState:
    """Complete local checkpoint for one AgentRuntime run."""

    schema_version: int
    revision: int
    run_id: str
    request: AgentRequest
    lifecycle_status: RunLifecycleStatus
    created_at: str
    updated_at: str
    plan: AgentPlan | None = None
    plan_fingerprint: str | None = None
    preflight_verification: VerificationResult | None = None
    steps: tuple[StepExecutionResult, ...] = ()
    run_verification: VerificationResult | None = None
    errors: tuple[AgentError, ...] = ()
    trace: tuple[ExecutionTraceEvent, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != RUN_STATE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported run-state schema version {self.schema_version}."
            )
        if isinstance(self.revision, bool) or not isinstance(self.revision, int):
            raise TypeError("`revision` must be an integer.")
        if self.revision < 0:
            raise ValueError("`revision` must be nonnegative.")
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("`run_id` must be a non-empty string.")
        if not isinstance(self.request, AgentRequest):
            raise TypeError("`request` must be an AgentRequest.")
        if not isinstance(self.lifecycle_status, RunLifecycleStatus):
            raise TypeError("`lifecycle_status` must be a RunLifecycleStatus.")
        created_at = _parse_aware_timestamp(self.created_at, "created_at")
        updated_at = _parse_aware_timestamp(self.updated_at, "updated_at")
        if updated_at < created_at:
            raise ValueError("`updated_at` cannot precede `created_at`.")
        if self.run_id != f"{self.request.request_id}:run":
            raise ValueError("Persisted run identity does not match the request ID.")

        tuple_contracts = (
            ("steps", self.steps, StepExecutionResult),
            ("errors", self.errors, AgentError),
            ("trace", self.trace, ExecutionTraceEvent),
        )
        for field_name, values, expected_type in tuple_contracts:
            if not isinstance(values, tuple) or not all(
                isinstance(value, expected_type) for value in values
            ):
                raise TypeError(
                    f"`{field_name}` must be a tuple of {expected_type.__name__}."
                )
        if self.plan is not None and not isinstance(self.plan, AgentPlan):
            raise TypeError("`plan` must be an AgentPlan or None.")
        for field_name in ("preflight_verification", "run_verification"):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, VerificationResult):
                raise TypeError(f"`{field_name}` must be a VerificationResult or None.")

        self._validate_plan_identity()
        self._validate_trace()
        self._validate_steps()
        self._validate_lifecycle()

    def _validate_plan_identity(self) -> None:
        if self.plan is None:
            if self.plan_fingerprint is not None:
                raise ValueError("A run without a plan cannot have a plan fingerprint.")
            if self.preflight_verification is not None or self.steps:
                raise ValueError(
                    "A run without a plan cannot contain preflight or step state."
                )
            return
        if self.plan.request_id != self.request.request_id:
            raise ValueError("Persisted request and plan identities do not match.")
        if not isinstance(self.plan_fingerprint, str) or not self.plan_fingerprint:
            raise ValueError("A persisted plan requires a plan fingerprint.")
        if self.plan_fingerprint != fingerprint_plan(self.plan):
            raise ValueError("Persisted plan fingerprint does not match the plan.")
        if self.preflight_verification is not None and (
            self.preflight_verification.target_type != "plan"
            or self.preflight_verification.target_id != self.plan.plan_id
        ):
            raise ValueError("Persisted preflight verification targets the wrong plan.")
        if self.run_verification is not None and (
            self.run_verification.target_type != "run"
            or self.run_verification.target_id != self.plan.plan_id
        ):
            raise ValueError("Persisted run verification targets the wrong plan.")

    def _validate_trace(self) -> None:
        if tuple(event.sequence for event in self.trace) != tuple(
            range(len(self.trace))
        ):
            raise ValueError("Persisted trace sequences must be contiguous from zero.")
        previous: datetime | None = None
        for event in self.trace:
            timestamp = _parse_aware_timestamp(event.timestamp, "trace.timestamp")
            if previous is not None and timestamp < previous:
                raise ValueError("Persisted trace timestamps must be nondecreasing.")
            previous = timestamp

    def _validate_steps(self) -> None:
        if not all(_step_payload_is_clean(result) for result in self.steps):
            raise ValueError("Persisted step status and payload are inconsistent.")
        if self.plan is None:
            return
        plan_by_id = {step.step_id: step for step in self.plan.steps}
        if len({result.step_id for result in self.steps}) != len(self.steps):
            raise ValueError("Persisted step IDs must be unique.")
        for result in self.steps:
            step = plan_by_id.get(result.step_id)
            if step is None or step.tool_name != result.tool_name:
                raise ValueError("Persisted step identity does not match the plan.")
            if result.verification is not None and (
                result.verification.target_type != "step"
                or result.verification.target_id != result.step_id
            ):
                raise ValueError("Persisted step verification targets the wrong step.")
            started = (
                None
                if result.started_at is None
                else _parse_aware_timestamp(result.started_at, "step.started_at")
            )
            finished = (
                None
                if result.finished_at is None
                else _parse_aware_timestamp(result.finished_at, "step.finished_at")
            )
            if started is not None and finished is not None and finished < started:
                raise ValueError("Persisted step finish time precedes its start time.")
        if self.steps and tuple(result.step_id for result in self.steps) != tuple(
            step.step_id for step in self.plan.steps
        ):
            raise ValueError("Persisted steps must follow original plan order.")

    def _validate_lifecycle(self) -> None:
        status = self.lifecycle_status
        if status is RunLifecycleStatus.PLANNING:
            if self.preflight_verification is not None or self.steps:
                raise ValueError("PLANNING state cannot contain validated step state.")
            if self.run_verification is not None or self.errors:
                raise ValueError("PLANNING state cannot contain terminal outcomes.")
            return

        if status is RunLifecycleStatus.FAILED and self.plan is None:
            if not self.errors or self.run_verification is not None:
                raise ValueError(
                    "A planless FAILED state requires errors and no run verification."
                )
            return

        if status is RunLifecycleStatus.CANCELLED:
            if not any(
                error.category is ErrorCategory.CANCELLATION
                and error.code == "RUN_CANCELLED"
                for error in self.errors
            ):
                raise ValueError("CANCELLED state requires a cancellation reason.")
            if any(
                result.status in {StepStatus.PENDING, StepStatus.RUNNING}
                for result in self.steps
            ):
                raise ValueError("CANCELLED state cannot retain pending/running steps.")
            if self.plan is None:
                if (
                    self.steps
                    or self.preflight_verification is not None
                    or self.run_verification is not None
                ):
                    raise ValueError(
                        "A planless CANCELLED state cannot contain preflight or steps."
                    )
            elif self.steps and len(self.steps) != len(self.plan.steps):
                raise ValueError(
                    "A CANCELLED state with steps must cover the complete plan."
                )
            return

        if self.plan is None:
            raise ValueError(f"{status.value} state requires a persisted plan.")

        if status is RunLifecycleStatus.PLANNED:
            if self.request.mode is not RunMode.PLAN_ONLY:
                raise ValueError("PLANNED durable state requires PLAN_ONLY mode.")
            if (
                self.preflight_verification is None
                or not self.preflight_verification.passed
                or self.steps
                or self.run_verification is not None
                or self.errors
            ):
                raise ValueError("PLANNED state requires only a passed preflight.")
            return

        if status in {
            RunLifecycleStatus.VALIDATED,
            RunLifecycleStatus.RUNNING,
            RunLifecycleStatus.SUCCEEDED,
            RunLifecycleStatus.INTERRUPTED,
        }:
            if (
                self.preflight_verification is None
                or not self.preflight_verification.passed
            ):
                raise ValueError(f"{status.value} state requires a passed preflight.")

        if status is RunLifecycleStatus.VALIDATED:
            if self.request.mode is not RunMode.EXECUTE:
                raise ValueError("VALIDATED state requires EXECUTE mode.")
            if len(self.steps) != len(self.plan.steps) or any(
                result.status not in {StepStatus.PENDING, StepStatus.SUCCEEDED}
                for result in self.steps
            ):
                raise ValueError("VALIDATED state requires pending/verified plan steps.")
            if self.run_verification is not None or self.errors:
                raise ValueError("VALIDATED state cannot contain terminal outcomes.")
            return

        if status is RunLifecycleStatus.RUNNING:
            if self.request.mode is not RunMode.EXECUTE:
                raise ValueError("RUNNING state requires EXECUTE mode.")
            if len(self.steps) != len(self.plan.steps) or any(
                result.status not in {
                    StepStatus.PENDING,
                    StepStatus.RUNNING,
                    StepStatus.SUCCEEDED,
                }
                for result in self.steps
            ):
                raise ValueError("RUNNING state contains an invalid step lifecycle.")
            if self.run_verification is not None or self.errors:
                raise ValueError("RUNNING state cannot contain terminal outcomes.")
            return

        if status is RunLifecycleStatus.SUCCEEDED:
            if self.request.mode is not RunMode.EXECUTE:
                raise ValueError("SUCCEEDED state requires EXECUTE mode.")
            if len(self.steps) != len(self.plan.steps) or any(
                result.status is not StepStatus.SUCCEEDED for result in self.steps
            ):
                raise ValueError("SUCCEEDED state requires every step to succeed.")
            if (
                self.run_verification is None
                or not self.run_verification.passed
                or self.errors
            ):
                raise ValueError("SUCCEEDED state requires passed run verification.")
            return

        if status is RunLifecycleStatus.INTERRUPTED:
            if not self.errors or not any(
                error.code == "STEP_OUTCOME_UNKNOWN_AFTER_INTERRUPTION"
                for error in self.errors
            ):
                raise ValueError("INTERRUPTED state requires an interruption error.")
            if any(result.status is StepStatus.RUNNING for result in self.steps):
                raise ValueError("INTERRUPTED state cannot retain a RUNNING step.")
            return

        if status is RunLifecycleStatus.FAILED:
            if not self.errors:
                raise ValueError("FAILED state requires at least one error.")
            if any(result.status is StepStatus.RUNNING for result in self.steps):
                raise ValueError("FAILED state cannot retain a RUNNING step.")
            return

        raise ValueError(f"Unhandled durable lifecycle status {status.value}.")

    def to_dict(self) -> dict[str, object]:
        """Return a fresh deterministic JSON-compatible record mapping."""

        return {
            "schema_version": self.schema_version,
            "revision": self.revision,
            "run_id": self.run_id,
            "request": self.request.to_dict(),
            "lifecycle_status": self.lifecycle_status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "plan": None if self.plan is None else self.plan.to_dict(),
            "plan_fingerprint": self.plan_fingerprint,
            "preflight_verification": (
                None
                if self.preflight_verification is None
                else self.preflight_verification.to_dict()
            ),
            "steps": [step.to_dict() for step in self.steps],
            "run_verification": (
                None
                if self.run_verification is None
                else self.run_verification.to_dict()
            ),
            "errors": [error.to_dict() for error in self.errors],
            "trace": [event.to_dict() for event in self.trace],
        }

    @classmethod
    def from_dict(cls, value: object) -> PersistedRunState:
        mapping = _mapping(value, "run state")
        _exact_keys(
            mapping,
            {
                "schema_version",
                "revision",
                "run_id",
                "request",
                "lifecycle_status",
                "created_at",
                "updated_at",
                "plan",
                "plan_fingerprint",
                "preflight_verification",
                "steps",
                "run_verification",
                "errors",
                "trace",
            },
            "run state",
        )
        plan_value = mapping["plan"]
        preflight_value = mapping["preflight_verification"]
        run_verification_value = mapping["run_verification"]
        stored_schema_version = _integer(
            mapping["schema_version"], "schema_version"
        )
        if stored_schema_version not in {
            RUN_STATE_SCHEMA_VERSION,
            *LEGACY_RUN_STATE_SCHEMA_VERSIONS,
        }:
            raise ValueError(
                f"Unsupported run-state schema version {stored_schema_version}."
            )
        lifecycle_status = _enum(
            RunLifecycleStatus,
            mapping["lifecycle_status"],
            "lifecycle_status",
        )
        preflight_verification = (
            None
            if preflight_value is None
            else _decode_verification(preflight_value)
        )
        steps = tuple(
            _decode_step_result(item) for item in _sequence(mapping["steps"], "steps")
        )
        run_verification = (
            None
            if run_verification_value is None
            else _decode_verification(run_verification_value)
        )
        errors = tuple(
            _decode_error(item) for item in _sequence(mapping["errors"], "errors")
        )
        trace = tuple(
            _decode_trace(item) for item in _sequence(mapping["trace"], "trace")
        )
        if stored_schema_version == 1:
            _validate_legacy_v1_semantics(
                lifecycle_status=lifecycle_status,
                preflight_verification=preflight_verification,
                steps=steps,
                run_verification=run_verification,
                errors=errors,
                trace=trace,
            )
        return cls(
            schema_version=RUN_STATE_SCHEMA_VERSION,
            revision=_integer(mapping["revision"], "revision"),
            run_id=_string(mapping["run_id"], "run_id"),
            request=_decode_request(mapping["request"]),
            lifecycle_status=lifecycle_status,
            created_at=_string(mapping["created_at"], "created_at"),
            updated_at=_string(mapping["updated_at"], "updated_at"),
            plan=None if plan_value is None else _decode_plan(plan_value),
            plan_fingerprint=_optional_string(
                mapping["plan_fingerprint"], "plan_fingerprint"
            ),
            preflight_verification=preflight_verification,
            steps=steps,
            run_verification=run_verification,
            errors=errors,
            trace=trace,
        )

    def to_run_result(self) -> AgentRunResult:
        """Project a terminal durable state onto the accepted public result."""

        status_map = {
            RunLifecycleStatus.PLANNED: RunStatus.PLANNED,
            RunLifecycleStatus.SUCCEEDED: RunStatus.SUCCEEDED,
            RunLifecycleStatus.FAILED: RunStatus.FAILED,
            RunLifecycleStatus.INTERRUPTED: RunStatus.FAILED,
            RunLifecycleStatus.CANCELLED: RunStatus.CANCELLED,
        }
        try:
            public_status = status_map[self.lifecycle_status]
        except KeyError as exc:
            raise ValueError("Nonterminal durable state has no AgentRunResult.") from exc
        verification = (
            self.preflight_verification
            if self.request.mode is RunMode.PLAN_ONLY
            else self.run_verification
        )
        return AgentRunResult(
            run_id=self.run_id,
            request_id=self.request.request_id,
            status=public_status,
            planning_only=self.request.mode is RunMode.PLAN_ONLY,
            plan=self.plan,
            steps=self.steps,
            verification=verification,
            errors=self.errors,
            trace=self.trace,
        )


def _validate_legacy_v1_semantics(
    *,
    lifecycle_status: RunLifecycleStatus,
    preflight_verification: VerificationResult | None,
    steps: Sequence[StepExecutionResult],
    run_verification: VerificationResult | None,
    errors: Sequence[AgentError],
    trace: Sequence[ExecutionTraceEvent],
) -> None:
    """Reject current-only enum values from a legacy schema-v1 record."""

    if lifecycle_status not in _LEGACY_V1_LIFECYCLE_STATUSES:
        raise ValueError(
            "run-state schema version 1 does not support lifecycle status "
            f"{lifecycle_status.value!r}."
        )

    semantic_errors = list(errors)
    for verification in (preflight_verification, run_verification):
        if verification is not None and verification.error is not None:
            semantic_errors.append(verification.error)
    for step in steps:
        if step.error is not None:
            semantic_errors.append(step.error)
        if step.verification is not None and step.verification.error is not None:
            semantic_errors.append(step.verification.error)
    for error in semantic_errors:
        if error.category not in _LEGACY_V1_ERROR_CATEGORIES:
            raise ValueError(
                "run-state schema version 1 does not support error category "
                f"{error.category.value!r}."
            )

    for event in trace:
        if event.event_type not in _LEGACY_V1_TRACE_EVENT_TYPES:
            raise ValueError(
                "run-state schema version 1 does not support trace event type "
                f"{event.event_type.value!r}."
            )


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) for key in value
    ):
        raise TypeError(f"{path} must be an object with string keys.")
    return cast(Mapping[str, object], value)


def _exact_keys(value: Mapping[str, object], expected: set[str], path: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{path} fields do not match schema; missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}."
        )


def _sequence(value: object, path: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise TypeError(f"{path} must be a JSON array.")
    return value


def _string(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{path} must be a string.")
    return value


def _optional_string(value: object, path: str) -> str | None:
    if value is None:
        return None
    return _string(value, path)


def _integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{path} must be an integer.")
    return value


def _number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{path} must be a number.")
    return float(value)


def _boolean(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{path} must be a boolean.")
    return value


def _enum(enum_type: type[Enum], value: object, path: str):
    if not isinstance(value, str):
        raise TypeError(f"{path} must be a string enum value.")
    try:
        return enum_type(value)
    except ValueError as exc:
        raise ValueError(f"{path} has unsupported value {value!r}.") from exc


def _json_value(value: object, path: str) -> JsonValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, list):
        return tuple(_json_value(item, f"{path}[]") for item in value)
    if isinstance(value, Mapping):
        return {
            _string(key, f"{path} key"): _json_value(item, f"{path}.{key}")
            for key, item in value.items()
        }
    raise TypeError(f"{path} contains unsupported JSON value.")


def _json_mapping(value: object, path: str) -> Mapping[str, JsonValue]:
    mapping = _mapping(value, path)
    return {key: _json_value(item, f"{path}.{key}") for key, item in mapping.items()}


def _decode_request(value: object) -> AgentRequest:
    mapping = _mapping(value, "request")
    _exact_keys(mapping, {"request_id", "prompt", "inputs", "mode"}, "request")
    return AgentRequest(
        _string(mapping["request_id"], "request.request_id"),
        _string(mapping["prompt"], "request.prompt"),
        _json_mapping(mapping["inputs"], "request.inputs"),
        _enum(RunMode, mapping["mode"], "request.mode"),
    )


def _decode_reference(value: object) -> StepOutputRef | None:
    if not isinstance(value, Mapping) or set(value) != {"$ref"}:
        return None
    reference = _mapping(value["$ref"], "step reference")
    _exact_keys(reference, {"step_id", "output_key"}, "step reference")
    return StepOutputRef(
        _string(reference["step_id"], "reference.step_id"),
        _string(reference["output_key"], "reference.output_key"),
    )


def _decode_plan_step(value: object) -> PlanStep:
    mapping = _mapping(value, "plan step")
    _exact_keys(
        mapping,
        {"step_id", "tool_name", "arguments", "depends_on", "description"},
        "plan step",
    )
    arguments = _mapping(mapping["arguments"], "plan step arguments")
    decoded_arguments: dict[str, object] = {}
    for key, argument in arguments.items():
        reference = _decode_reference(argument)
        decoded_arguments[key] = (
            reference
            if reference is not None
            else _json_value(argument, f"plan step arguments.{key}")
        )
    description = mapping["description"]
    return PlanStep(
        _string(mapping["step_id"], "plan step.step_id"),
        _string(mapping["tool_name"], "plan step.tool_name"),
        decoded_arguments,
        tuple(
            _string(item, "plan step dependency")
            for item in _sequence(mapping["depends_on"], "plan step.depends_on")
        ),
        None if description is None else _string(description, "plan step.description"),
    )


def _decode_plan(value: object) -> AgentPlan:
    mapping = _mapping(value, "plan")
    _exact_keys(mapping, {"plan_id", "request_id", "planner_name", "steps"}, "plan")
    return AgentPlan(
        _string(mapping["plan_id"], "plan.plan_id"),
        _string(mapping["request_id"], "plan.request_id"),
        _string(mapping["planner_name"], "plan.planner_name"),
        tuple(_decode_plan_step(item) for item in _sequence(mapping["steps"], "plan.steps")),
    )


def _decode_error(value: object) -> AgentError:
    mapping = _mapping(value, "error")
    _exact_keys(
        mapping,
        {
            "category",
            "code",
            "message",
            "step_id",
            "tool_name",
            "exception_type",
            "recoverable",
            "attempt",
            "details",
        },
        "error",
    )
    attempt = mapping["attempt"]
    return AgentError(
        _enum(ErrorCategory, mapping["category"], "error.category"),
        _string(mapping["code"], "error.code"),
        _string(mapping["message"], "error.message"),
        _optional_string(mapping["step_id"], "error.step_id"),
        _optional_string(mapping["tool_name"], "error.tool_name"),
        _optional_string(mapping["exception_type"], "error.exception_type"),
        _boolean(mapping["recoverable"], "error.recoverable"),
        None if attempt is None else _integer(attempt, "error.attempt"),
        _json_mapping(mapping["details"], "error.details"),
    )


def _decode_check(value: object) -> VerificationCheck:
    mapping = _mapping(value, "verification check")
    _exact_keys(mapping, {"name", "passed", "message"}, "verification check")
    return VerificationCheck(
        _string(mapping["name"], "verification check.name"),
        _boolean(mapping["passed"], "verification check.passed"),
        _string(mapping["message"], "verification check.message"),
    )


def _decode_verification(value: object) -> VerificationResult:
    mapping = _mapping(value, "verification")
    _exact_keys(
        mapping,
        {"passed", "target_type", "target_id", "checks", "error"},
        "verification",
    )
    error = mapping["error"]
    return VerificationResult(
        _boolean(mapping["passed"], "verification.passed"),
        _string(mapping["target_type"], "verification.target_type"),
        _string(mapping["target_id"], "verification.target_id"),
        tuple(
            _decode_check(item)
            for item in _sequence(mapping["checks"], "verification.checks")
        ),
        None if error is None else _decode_error(error),
    )


def _decode_step_result(value: object) -> StepExecutionResult:
    mapping = _mapping(value, "step result")
    _exact_keys(
        mapping,
        {
            "step_id",
            "tool_name",
            "status",
            "attempt_count",
            "resolved_arguments",
            "result",
            "verification",
            "error",
            "started_at",
            "finished_at",
            "duration_seconds",
        },
        "step result",
    )
    result = mapping["result"]
    verification = mapping["verification"]
    error = mapping["error"]
    duration = mapping["duration_seconds"]
    return StepExecutionResult(
        _string(mapping["step_id"], "step result.step_id"),
        _string(mapping["tool_name"], "step result.tool_name"),
        _enum(StepStatus, mapping["status"], "step result.status"),
        _integer(mapping["attempt_count"], "step result.attempt_count"),
        _json_mapping(mapping["resolved_arguments"], "step result.resolved_arguments"),
        None if result is None else _json_mapping(result, "step result.result"),
        None if verification is None else _decode_verification(verification),
        None if error is None else _decode_error(error),
        _optional_string(mapping["started_at"], "step result.started_at"),
        _optional_string(mapping["finished_at"], "step result.finished_at"),
        None if duration is None else _number(duration, "step result.duration_seconds"),
    )


def _decode_trace(value: object) -> ExecutionTraceEvent:
    mapping = _mapping(value, "trace event")
    _exact_keys(
        mapping,
        {"sequence", "event_type", "timestamp", "message", "step_id", "attempt", "details"},
        "trace event",
    )
    attempt = mapping["attempt"]
    return ExecutionTraceEvent(
        _integer(mapping["sequence"], "trace.sequence"),
        _enum(TraceEventType, mapping["event_type"], "trace.event_type"),
        _string(mapping["timestamp"], "trace.timestamp"),
        _string(mapping["message"], "trace.message"),
        _optional_string(mapping["step_id"], "trace.step_id"),
        None if attempt is None else _integer(attempt, "trace.attempt"),
        _json_mapping(mapping["details"], "trace.details"),
    )


__all__ = [
    "CANCELLATION_STATE_SCHEMA_VERSION",
    "CancellationDisposition",
    "CancellationReceipt",
    "CancellationRequest",
    "LEGACY_RUN_STATE_SCHEMA_VERSIONS",
    "PersistedRunState",
    "RUN_STATE_SCHEMA_VERSION",
    "RunLifecycleStatus",
    "fingerprint_plan",
]
