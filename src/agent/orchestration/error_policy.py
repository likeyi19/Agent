"""Central production error semantics and recovery-policy provenance.

Subsystems continue to own raw exception classification and stable error codes.
This module standardizes the safe public message, static recovery disposition,
and immutable policy identity applied around the existing executor retry loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping

from agent.schemas import (
    AgentError,
    AgentPlan,
    ErrorCategory,
    RecoveryDisposition,
    RecoveryPolicySnapshot,
    ToolRecoveryPolicySnapshot,
    fingerprint_recovery_policy,
)

if TYPE_CHECKING:
    from .registry import ToolRegistry


ERROR_POLICY_CATALOG_VERSION = "1"
UNREGISTERED_TOOL_POLICY_VERSION = "unregistered-v1"


@dataclass(frozen=True)
class ErrorPolicyEntry:
    message: str
    recovery_disposition: RecoveryDisposition


_NO = RecoveryDisposition.NO_AUTOMATIC_RECOVERY
_USER = RecoveryDisposition.USER_ACTION_REQUIRED
_RESUME = RecoveryDisposition.RESUME_WITH_COMPATIBLE_RUNTIME
_MANUAL = RecoveryDisposition.MANUAL_RECONCILIATION


# Safe messages and static recovery classes for production boundary codes.  An
# unknown code deliberately receives the fail-closed defaults below.
_ERROR_POLICY_CATALOG: dict[str, ErrorPolicyEntry] = {
    "RESOURCE_NOT_FOUND": ErrorPolicyEntry(
        "A required input or resource was not found.", _USER
    ),
    "OUTPUT_CONFLICT": ErrorPolicyEntry(
        "A requested scientific output already exists.", _USER
    ),
    "INVALID_ARGUMENT": ErrorPolicyEntry(
        "Scientific tool arguments are invalid.", _USER
    ),
    "DEPENDENCY_UNAVAILABLE": ErrorPolicyEntry(
        "A required runtime dependency is unavailable.", _USER
    ),
    "CUDA_UNAVAILABLE": ErrorPolicyEntry(
        "The requested CUDA runtime is unavailable.", _USER
    ),
    "CUDA_OUT_OF_MEMORY": ErrorPolicyEntry(
        "CUDA memory was exhausted during scientific execution.", _USER
    ),
    "HOST_MEMORY_EXHAUSTED": ErrorPolicyEntry(
        "Host memory was exhausted during scientific execution.", _USER
    ),
    "DISK_FULL": ErrorPolicyEntry(
        "Storage space was exhausted while writing scientific artifacts.", _USER
    ),
    "ARTIFACT_WRITE_FAILED": ErrorPolicyEntry(
        "Scientific artifacts could not be written safely.", _USER
    ),
    "TOOL_RUNTIME_ERROR": ErrorPolicyEntry(
        "The registered scientific tool failed during execution.", _NO
    ),
    "TOOL_EXCEPTION": ErrorPolicyEntry(
        "The registered scientific tool failed during execution.", _NO
    ),
    "PLANNING_PROVIDER_ERROR": ErrorPolicyEntry(
        "The planning provider failed to produce a response.", _USER
    ),
    "PROVIDER_AUTHENTICATION_FAILED": ErrorPolicyEntry(
        "Planning-provider authentication failed.", _USER
    ),
    "PROVIDER_RATE_LIMITED": ErrorPolicyEntry(
        "The planning provider rejected the request due to a rate limit.", _USER
    ),
    "PROVIDER_TIMEOUT": ErrorPolicyEntry(
        "The planning provider request timed out.", _USER
    ),
    "PROVIDER_CONNECTION_FAILED": ErrorPolicyEntry(
        "The planning provider could not be reached.", _USER
    ),
    "PROVIDER_UNAVAILABLE": ErrorPolicyEntry(
        "The planning provider is unavailable.", _USER
    ),
    "MISSING_REQUIRED_INPUT": ErrorPolicyEntry(
        "The request is missing a required structured input.", _USER
    ),
    "INVALID_REQUEST_INPUT": ErrorPolicyEntry(
        "A structured request input is invalid.", _USER
    ),
    "UNSUPPORTED_REQUEST": ErrorPolicyEntry(
        "The request is not supported by the registered scientific capabilities.",
        _USER,
    ),
    "STEP_OUTCOME_UNKNOWN_AFTER_INTERRUPTION": ErrorPolicyEntry(
        "A scientific step was interrupted with an unknown outcome.", _MANUAL
    ),
    "PLANNING_INTERRUPTED_BEFORE_PLAN_AVAILABLE": ErrorPolicyEntry(
        "Planning was interrupted before a resumable plan was persisted.", _USER
    ),
    "RECOVERY_POLICY_INCOMPATIBLE": ErrorPolicyEntry(
        "The run requires a runtime with its original recovery policy.", _RESUME
    ),
    "RECOVERY_POLICY_UNKNOWN": ErrorPolicyEntry(
        "The historical recovery policy is unavailable and requires manual reconciliation.",
        _MANUAL,
    ),
    "RUN_CANCELLED": ErrorPolicyEntry(
        "Run cancellation was observed at a cooperative checkpoint.", _NO
    ),
    "EXECUTION_CANCELLED": ErrorPolicyEntry(
        "The step was not started because run cancellation took effect.", _NO
    ),
}

_BUILTIN_NO_AUTOMATIC_CODES = frozenset(
    {
        "ARTIFACT_EMPTY",
        "ARTIFACT_MISSING",
        "CELL_COUNT_MISMATCH",
        "DEPENDENCY_FAILED",
        "DEPENDENCY_INCONSISTENT",
        "DEPENDENCY_RESULT_MISSING",
        "DUPLICATE_PERSISTED_STEP",
        "DUPLICATE_STEP_RESULT",
        "EXECUTION_ABORTED",
        "EXECUTION_ABORTED_AFTER_CHECKPOINT_FAILURE",
        "EXECUTION_ABORTED_AFTER_INTERRUPTION",
        "EXECUTION_STATE_UNAVAILABLE",
        "EXECUTION_STATE_UNAVAILABLE_DURING_CANCELLATION",
        "EXECUTOR_UNEXPECTED_ERROR",
        "INVALID_OUTPUT_REFERENCE",
        "INVALID_PLAN_STRUCTURE",
        "INVALID_TOOL_ARGUMENTS",
        "MISSING_STEP_RESULT",
        "PERSISTED_RESOLVED_ARGUMENTS_MISMATCH",
        "PERSISTED_STEP_ARGUMENT_RESOLUTION_FAILED",
        "PERSISTED_STEP_DEPENDENCY_MISSING",
        "PERSISTED_STEP_IDENTITY_MISMATCH",
        "PERSISTED_STEP_NOT_VERIFIED",
        "PERSISTED_STEP_RESULT_MISSING",
        "PERSISTED_STEP_REVALIDATION_FAILED",
        "PLAN_PREFLIGHT_FAILED",
        "PLANNER_BINDING_INVALID",
        "PLANNER_CATALOG_INVALID",
        "PLANNER_OUTPUT_INVALID",
        "PLANNER_REGISTRY_CONTRACT_MISMATCH",
        "PLANNER_STRUCTURE_INVALID",
        "PLANNER_UNEXPECTED_ERROR",
        "PREFLIGHT_UNEXPECTED_ERROR",
        "REFERENCE_DEPENDENCY_INVALID",
        "REFERENCE_OUTPUT_MISSING",
        "REFERENCE_RESULT_UNAVAILABLE",
        "REFERENCE_VALUE_INVALID",
        "RESOLVED_ARGUMENTS_INVALID",
        "RESULT_CONTRACT_INVALID",
        "RESULT_IDENTITY_MISMATCH",
        "RESULT_METADATA_INCONSISTENT",
        "RESULT_NOT_LIGHTWEIGHT",
        "RESULT_PATH_MISMATCH",
        "RESULT_STATUS_INVALID",
        "RESULT_VALUE_INVALID",
        "RUN_RESULT_INVALID",
        "RUN_VERIFICATION_UNEXPECTED_ERROR",
        "SKIPPED_AFTER_PREFLIGHT_FAILURE",
        "STEP_IDENTITY_MISMATCH",
        "STEP_NOT_SUCCEEDED",
        "STEP_STATUS_INCONSISTENT",
        "STEP_VERIFICATION_FAILED",
        "TOOL_IDENTITY_MISMATCH",
        "UNEXPECTED_STEP_RESULT",
        "UNKNOWN_TOOL",
    }
)
for _code in _BUILTIN_NO_AUTOMATIC_CODES:
    _ERROR_POLICY_CATALOG.setdefault(
        _code,
        ErrorPolicyEntry("The operation failed validation or execution.", _NO),
    )

ERROR_POLICY_CATALOG: Mapping[str, ErrorPolicyEntry] = MappingProxyType(
    _ERROR_POLICY_CATALOG
)
BUILTIN_ERROR_CODES = frozenset(ERROR_POLICY_CATALOG)


def policy_entry(code: str) -> ErrorPolicyEntry:
    """Return fail-closed public semantics for one stable error code."""

    return ERROR_POLICY_CATALOG.get(
        code,
        ErrorPolicyEntry("The operation failed.", _NO),
    )


def recovery_disposition_for(
    code: str,
    *,
    same_step_retry_eligible: bool = False,
) -> RecoveryDisposition:
    if same_step_retry_eligible:
        return RecoveryDisposition.SAME_STEP_RETRY_ELIGIBLE
    return policy_entry(code).recovery_disposition


def safe_message_for(code: str, *, fallback: str | None = None) -> str:
    entry = ERROR_POLICY_CATALOG.get(code)
    if entry is not None:
        return entry.message
    return fallback or "The operation failed."


def classified_agent_error(
    *,
    category: ErrorCategory,
    code: str,
    step_id: str | None = None,
    tool_name: str | None = None,
    exception_type: str | None = None,
    same_step_retry_eligible: bool = False,
    attempt: int | None = None,
    details: Mapping[str, object] | None = None,
    safe_fallback_message: str | None = None,
) -> AgentError:
    """Construct one sanitized AgentError from subsystem-owned classification."""

    return AgentError(
        category=category,
        code=code,
        message=safe_message_for(code, fallback=safe_fallback_message),
        step_id=step_id,
        tool_name=tool_name,
        exception_type=exception_type,
        recoverable=same_step_retry_eligible,
        attempt=attempt,
        details=details or {},
        recovery_disposition=recovery_disposition_for(
            code,
            same_step_retry_eligible=same_step_retry_eligible,
        ),
    )


def build_recovery_policy_snapshot(
    plan: AgentPlan,
    registry: ToolRegistry,
    *,
    max_attempts_per_step: int,
) -> RecoveryPolicySnapshot:
    """Build canonical non-executable recovery provenance for a plan."""

    tools: list[ToolRecoveryPolicySnapshot] = []
    for tool_name in sorted({step.tool_name for step in plan.steps}):
        if registry.contains(tool_name):
            spec = registry.get(tool_name)
            classifier_version = spec.recovery_policy_version
            retryable_codes = tuple(sorted(spec.retryable_error_codes))
        else:
            classifier_version = UNREGISTERED_TOOL_POLICY_VERSION
            retryable_codes = ()
        tools.append(
            ToolRecoveryPolicySnapshot(
                tool_name,
                classifier_version,
                retryable_codes,
            )
        )
    tool_tuple = tuple(tools)
    fingerprint = fingerprint_recovery_policy(
        ERROR_POLICY_CATALOG_VERSION,
        max_attempts_per_step,
        tool_tuple,
    )
    return RecoveryPolicySnapshot(
        ERROR_POLICY_CATALOG_VERSION,
        max_attempts_per_step,
        tool_tuple,
        fingerprint,
    )


@dataclass(frozen=True)
class RecoveryDecision:
    retry: bool
    decision: str
    reason: str
    exhausted: bool


def decide_same_step_recovery(
    error: AgentError,
    *,
    attempt: int,
    max_attempts: int,
) -> RecoveryDecision:
    """Apply the dynamic attempt bound to one statically classified error."""

    if not error.recoverable:
        return RecoveryDecision(False, "stop_nonretryable", "nonretryable", False)
    if attempt >= max_attempts:
        return RecoveryDecision(
            False,
            "stop_attempt_limit",
            "attempt_limit_reached",
            True,
        )
    return RecoveryDecision(
        True,
        "retry_same_arguments",
        "same_step_retry_eligible",
        False,
    )


__all__ = [
    "ERROR_POLICY_CATALOG",
    "ERROR_POLICY_CATALOG_VERSION",
    "BUILTIN_ERROR_CODES",
    "ErrorPolicyEntry",
    "RecoveryDecision",
    "build_recovery_policy_snapshot",
    "classified_agent_error",
    "decide_same_step_recovery",
    "policy_entry",
    "recovery_disposition_for",
    "safe_message_for",
]
