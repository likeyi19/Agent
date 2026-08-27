"""Typed, JSON-safe contracts for Agent orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum
import math
from types import MappingProxyType
from typing import Any, Mapping, TypeAlias, cast


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | tuple["JsonValue", ...] | Mapping[str, "JsonValue"]


class RunMode(str, Enum):
    EXECUTE = "EXECUTE"
    PLAN_ONLY = "PLAN_ONLY"


class ErrorCategory(str, Enum):
    USER_INPUT_ERROR = "USER_INPUT_ERROR"
    RESOURCE_ERROR = "RESOURCE_ERROR"
    ENVIRONMENT_ERROR = "ENVIRONMENT_ERROR"
    TOOL_EXECUTION_ERROR = "TOOL_EXECUTION_ERROR"
    VERIFICATION_ERROR = "VERIFICATION_ERROR"
    INTERNAL_AGENT_ERROR = "INTERNAL_AGENT_ERROR"


class StepStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class RunStatus(str, Enum):
    PLANNED = "PLANNED"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class TraceEventType(str, Enum):
    PLANNING = "PLANNING"
    PLAN_VALIDATION = "PLAN_VALIDATION"
    STEP_EXECUTION = "STEP_EXECUTION"
    VERIFICATION = "VERIFICATION"
    RECOVERY = "RECOVERY"
    STEP_SKIPPED = "STEP_SKIPPED"
    RUN_COMPLETION = "RUN_COMPLETION"


def _require_non_empty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"`{field_name}` must be a non-empty string.")
    return value


def _freeze_json_value(value: object, path: str = "value") -> JsonValue:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"`{path}` must not contain non-finite floats.")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, JsonValue] = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise TypeError(f"`{path}` contains a non-string mapping key.")
            frozen[key] = _freeze_json_value(nested, f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json_value(nested, f"{path}[{index}]")
            for index, nested in enumerate(value)
        )
    raise TypeError(
        f"`{path}` contains unsupported non-JSON value {type(value).__name__}."
    )


def freeze_json_mapping(value: Mapping[str, object], field_name: str) -> Mapping[str, JsonValue]:
    if not isinstance(value, Mapping):
        raise TypeError(f"`{field_name}` must be a mapping.")
    frozen = _freeze_json_value(value, field_name)
    return cast(Mapping[str, JsonValue], frozen)


def _serialize(value: object) -> Any:
    if isinstance(value, StepOutputRef):
        return {
            "$ref": {
                "step_id": value.step_id,
                "output_key": value.output_key,
            }
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, _JsonModel):
        return {model_field.name: _serialize(getattr(value, model_field.name)) for model_field in fields(value)}
    if isinstance(value, Mapping):
        return {key: _serialize(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [_serialize(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Cannot serialize a non-finite float.")
        return value
    raise TypeError(f"Cannot serialize unsupported value {type(value).__name__}.")


class _JsonModel:
    def to_dict(self) -> dict[str, Any]:
        """Return a fresh deterministic JSON-compatible dictionary."""

        serialized = _serialize(self)
        if not isinstance(serialized, dict):  # pragma: no cover - model invariant
            raise TypeError("Serialized orchestration model must be a dictionary.")
        return serialized


@dataclass(frozen=True)
class StepOutputRef(_JsonModel):
    step_id: str
    output_key: str

    def __post_init__(self) -> None:
        _require_non_empty_string(self.step_id, "step_id")
        _require_non_empty_string(self.output_key, "output_key")


PlanArgument: TypeAlias = JsonValue | StepOutputRef


def _freeze_plan_arguments(
    value: Mapping[str, object],
) -> Mapping[str, PlanArgument]:
    if not isinstance(value, Mapping):
        raise TypeError("`arguments` must be a mapping.")
    frozen: dict[str, PlanArgument] = {}
    for key, argument in value.items():
        _require_non_empty_string(key, "argument name")
        frozen[key] = (
            argument
            if isinstance(argument, StepOutputRef)
            else _freeze_json_value(argument, f"arguments.{key}")
        )
    return MappingProxyType(frozen)


@dataclass(frozen=True)
class AgentRequest(_JsonModel):
    request_id: str
    prompt: str
    inputs: Mapping[str, JsonValue]
    mode: RunMode = RunMode.EXECUTE

    def __post_init__(self) -> None:
        _require_non_empty_string(self.request_id, "request_id")
        _require_non_empty_string(self.prompt, "prompt")
        if not isinstance(self.mode, RunMode):
            raise TypeError("`mode` must be a RunMode.")
        object.__setattr__(self, "inputs", freeze_json_mapping(self.inputs, "inputs"))


@dataclass(frozen=True)
class PlanStep(_JsonModel):
    step_id: str
    tool_name: str
    arguments: Mapping[str, PlanArgument]
    depends_on: tuple[str, ...] = ()
    description: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty_string(self.step_id, "step_id")
        _require_non_empty_string(self.tool_name, "tool_name")
        if not isinstance(self.depends_on, tuple):
            raise TypeError("`depends_on` must be a tuple.")
        for dependency in self.depends_on:
            _require_non_empty_string(dependency, "dependency")
        if len(set(self.depends_on)) != len(self.depends_on):
            raise ValueError("`depends_on` must not contain duplicate step IDs.")
        if self.step_id in self.depends_on:
            raise ValueError(f"Step {self.step_id!r} cannot depend on itself.")
        if self.description is not None and not isinstance(self.description, str):
            raise TypeError("`description` must be a string or None.")
        object.__setattr__(self, "arguments", _freeze_plan_arguments(self.arguments))


@dataclass(frozen=True)
class AgentPlan(_JsonModel):
    plan_id: str
    request_id: str
    planner_name: str
    steps: tuple[PlanStep, ...]

    def __post_init__(self) -> None:
        _require_non_empty_string(self.plan_id, "plan_id")
        _require_non_empty_string(self.request_id, "request_id")
        _require_non_empty_string(self.planner_name, "planner_name")
        if not isinstance(self.steps, tuple):
            raise TypeError("`steps` must be a tuple.")
        if not all(isinstance(step, PlanStep) for step in self.steps):
            raise TypeError("Every plan step must be a PlanStep.")

        step_ids = tuple(step.step_id for step in self.steps)
        if len(set(step_ids)) != len(step_ids):
            raise ValueError("AgentPlan step IDs must be unique.")
        known_ids = set(step_ids)
        for step in self.steps:
            missing = [dependency for dependency in step.depends_on if dependency not in known_ids]
            if missing:
                raise ValueError(
                    f"Step {step.step_id!r} references missing dependencies: {missing}."
                )
            for argument_name, argument in step.arguments.items():
                if not isinstance(argument, StepOutputRef):
                    continue
                if argument.step_id not in known_ids:
                    raise ValueError(
                        f"Argument {argument_name!r} in step {step.step_id!r} "
                        f"references missing step {argument.step_id!r}."
                    )
                if argument.step_id == step.step_id:
                    raise ValueError(
                        f"Step {step.step_id!r} cannot reference its own output."
                    )
                if argument.step_id not in step.depends_on:
                    raise ValueError(
                        f"Step {step.step_id!r} must declare referenced step "
                        f"{argument.step_id!r} in depends_on."
                    )
        self.stable_topological_steps()

    def stable_topological_steps(self) -> tuple[PlanStep, ...]:
        """Return dependency order, preserving plan order among ready steps."""

        emitted: set[str] = set()
        ordered: list[PlanStep] = []
        while len(ordered) < len(self.steps):
            ready = [
                step
                for step in self.steps
                if step.step_id not in emitted
                and all(dependency in emitted for dependency in step.depends_on)
            ]
            if not ready:
                raise ValueError("AgentPlan dependencies contain a cycle.")
            for step in ready:
                emitted.add(step.step_id)
                ordered.append(step)
        return tuple(ordered)


@dataclass(frozen=True)
class AgentError(_JsonModel):
    category: ErrorCategory
    code: str
    message: str
    step_id: str | None = None
    tool_name: str | None = None
    exception_type: str | None = None
    recoverable: bool = False
    attempt: int | None = None
    details: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.category, ErrorCategory):
            raise TypeError("`category` must be an ErrorCategory.")
        _require_non_empty_string(self.code, "code")
        _require_non_empty_string(self.message, "message")
        for field_name in ("step_id", "tool_name", "exception_type"):
            value = getattr(self, field_name)
            if value is not None:
                _require_non_empty_string(value, field_name)
        if not isinstance(self.recoverable, bool):
            raise TypeError("`recoverable` must be a boolean.")
        if self.attempt is not None and (
            isinstance(self.attempt, bool)
            or not isinstance(self.attempt, int)
            or self.attempt <= 0
        ):
            raise ValueError("`attempt` must be a positive integer or None.")
        object.__setattr__(self, "details", freeze_json_mapping(self.details, "details"))


@dataclass(frozen=True)
class VerificationCheck(_JsonModel):
    name: str
    passed: bool
    message: str

    def __post_init__(self) -> None:
        _require_non_empty_string(self.name, "name")
        if not isinstance(self.passed, bool):
            raise TypeError("`passed` must be a boolean.")
        _require_non_empty_string(self.message, "message")


@dataclass(frozen=True)
class VerificationResult(_JsonModel):
    passed: bool
    target_type: str
    target_id: str
    checks: tuple[VerificationCheck, ...] = ()
    error: AgentError | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.passed, bool):
            raise TypeError("`passed` must be a boolean.")
        _require_non_empty_string(self.target_type, "target_type")
        _require_non_empty_string(self.target_id, "target_id")
        if not isinstance(self.checks, tuple) or not all(
            isinstance(check, VerificationCheck) for check in self.checks
        ):
            raise TypeError("`checks` must be a tuple of VerificationCheck values.")
        if self.error is not None and not isinstance(self.error, AgentError):
            raise TypeError("`error` must be an AgentError or None.")
        if self.passed and (self.error is not None or any(not check.passed for check in self.checks)):
            raise ValueError("A passed verification cannot contain failures or an error.")


@dataclass(frozen=True)
class StepExecutionResult(_JsonModel):
    step_id: str
    tool_name: str
    status: StepStatus
    attempt_count: int = 0
    resolved_arguments: Mapping[str, JsonValue] = field(default_factory=dict)
    result: Mapping[str, JsonValue] | None = None
    verification: VerificationResult | None = None
    error: AgentError | None = None
    started_at: str | None = None
    finished_at: str | None = None
    duration_seconds: float | None = None

    def __post_init__(self) -> None:
        _require_non_empty_string(self.step_id, "step_id")
        _require_non_empty_string(self.tool_name, "tool_name")
        if not isinstance(self.status, StepStatus):
            raise TypeError("`status` must be a StepStatus.")
        if isinstance(self.attempt_count, bool) or not isinstance(self.attempt_count, int) or self.attempt_count < 0:
            raise ValueError("`attempt_count` must be a nonnegative integer.")
        object.__setattr__(
            self,
            "resolved_arguments",
            freeze_json_mapping(self.resolved_arguments, "resolved_arguments"),
        )
        if self.result is not None:
            object.__setattr__(self, "result", freeze_json_mapping(self.result, "result"))
        if self.verification is not None and not isinstance(self.verification, VerificationResult):
            raise TypeError("`verification` must be a VerificationResult or None.")
        if self.error is not None and not isinstance(self.error, AgentError):
            raise TypeError("`error` must be an AgentError or None.")
        for field_name in ("started_at", "finished_at"):
            value = getattr(self, field_name)
            if value is not None:
                _require_non_empty_string(value, field_name)
        if self.duration_seconds is not None and (
            isinstance(self.duration_seconds, bool)
            or not isinstance(self.duration_seconds, (int, float))
            or not math.isfinite(float(self.duration_seconds))
            or self.duration_seconds < 0
        ):
            raise ValueError("`duration_seconds` must be finite and nonnegative or None.")


@dataclass(frozen=True)
class ExecutionTraceEvent(_JsonModel):
    sequence: int
    event_type: TraceEventType
    timestamp: str
    message: str
    step_id: str | None = None
    attempt: int | None = None
    details: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence < 0:
            raise ValueError("`sequence` must be a nonnegative integer.")
        if not isinstance(self.event_type, TraceEventType):
            raise TypeError("`event_type` must be a TraceEventType.")
        _require_non_empty_string(self.timestamp, "timestamp")
        _require_non_empty_string(self.message, "message")
        if self.step_id is not None:
            _require_non_empty_string(self.step_id, "step_id")
        if self.attempt is not None and (
            isinstance(self.attempt, bool)
            or not isinstance(self.attempt, int)
            or self.attempt <= 0
        ):
            raise ValueError("`attempt` must be a positive integer or None.")
        object.__setattr__(self, "details", freeze_json_mapping(self.details, "details"))


@dataclass(frozen=True)
class AgentRunResult(_JsonModel):
    run_id: str
    request_id: str
    status: RunStatus
    planning_only: bool
    plan: AgentPlan | None = None
    steps: tuple[StepExecutionResult, ...] = ()
    verification: VerificationResult | None = None
    errors: tuple[AgentError, ...] = ()
    trace: tuple[ExecutionTraceEvent, ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty_string(self.run_id, "run_id")
        _require_non_empty_string(self.request_id, "request_id")
        if not isinstance(self.status, RunStatus):
            raise TypeError("`status` must be a RunStatus.")
        if not isinstance(self.planning_only, bool):
            raise TypeError("`planning_only` must be a boolean.")
        if self.plan is not None and not isinstance(self.plan, AgentPlan):
            raise TypeError("`plan` must be an AgentPlan or None.")
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
                    f"`{field_name}` must be a tuple of {expected_type.__name__} values."
                )
        if self.verification is not None and not isinstance(self.verification, VerificationResult):
            raise TypeError("`verification` must be a VerificationResult or None.")


__all__ = [
    "AgentError",
    "AgentPlan",
    "AgentRequest",
    "AgentRunResult",
    "ErrorCategory",
    "ExecutionTraceEvent",
    "JsonValue",
    "PlanStep",
    "RunMode",
    "RunStatus",
    "StepExecutionResult",
    "StepOutputRef",
    "StepStatus",
    "TraceEventType",
    "VerificationCheck",
    "VerificationResult",
]
