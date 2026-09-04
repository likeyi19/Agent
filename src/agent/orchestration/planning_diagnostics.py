"""Sanitized provider-neutral diagnostics for bounded LLM planning attempts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import re
from types import MappingProxyType
from typing import Mapping

from agent.schemas import AgentPlan, JsonValue


PLANNING_DIAGNOSTIC_SCHEMA_VERSION = 3
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$")
_OUTCOMES = frozenset({"started", "succeeded", "failed", "rejected"})


class PlanningDiagnosticStage(str, Enum):
    """Logical, non-semantic stages of structured LLM planning."""

    PROVIDER = "provider"
    PARSE = "parse"
    SCHEMA = "schema"
    TOOL_SELECTION = "tool_selection"
    ARGUMENT_BINDING = "argument_binding"
    DEPENDENCY_REFERENCE = "dependency_reference"
    UNSUPPORTED = "unsupported"
    CANDIDATE = "candidate"
    PREFLIGHT = "preflight"
    ACCEPTED = "accepted"
    RECOVERY = "recovery"


class PlanningAttemptKind(str, Enum):
    """Provider-neutral identity of one logical planning attempt."""

    INITIAL = "initial"
    TRANSPORT_RETRY = "transport_retry"
    REPAIR = "repair"
    FAILOVER = "failover"


def safe_diagnostic_identifier(value: object) -> str | None:
    """Return a bounded identifier only when it is intrinsically safe."""

    if isinstance(value, str) and _SAFE_IDENTIFIER.fullmatch(value):
        return value
    return None


@dataclass(frozen=True)
class PlanningDiagnosticContext:
    """Immutable non-secret identity shared by one logical planning attempt."""

    profile_id: str
    provider_id: str
    model_identity_digest: str
    catalog_fingerprint: str
    offered_tool_names: tuple[str, ...]
    planning_wire_schema_version: int
    attempt_kind: PlanningAttemptKind = PlanningAttemptKind.INITIAL
    logical_attempt_index: int = 1
    provider_call_index: int = 1

    def __post_init__(self) -> None:
        if safe_diagnostic_identifier(self.profile_id) != self.profile_id:
            raise ValueError("Diagnostic profile identity is not safe.")
        if safe_diagnostic_identifier(self.provider_id) != self.provider_id:
            raise ValueError("Diagnostic provider identity is not safe.")
        if not re.fullmatch(r"[0-9a-f]{64}", self.model_identity_digest):
            raise ValueError("Diagnostic model identity digest must be SHA-256.")
        if not re.fullmatch(r"[0-9a-f]{64}", self.catalog_fingerprint):
            raise ValueError("Diagnostic catalog fingerprint must be SHA-256.")
        if not isinstance(self.offered_tool_names, tuple) or not all(
            safe_diagnostic_identifier(name) == name
            for name in self.offered_tool_names
        ):
            raise ValueError("Diagnostic tool names must be safe identifiers.")
        if (
            isinstance(self.planning_wire_schema_version, bool)
            or not isinstance(self.planning_wire_schema_version, int)
            or self.planning_wire_schema_version < 1
        ):
            raise ValueError("Planning wire schema version must be positive.")
        if not isinstance(self.attempt_kind, PlanningAttemptKind):
            raise TypeError("Diagnostic attempt kind must be PlanningAttemptKind.")
        for field_name in ("logical_attempt_index", "provider_call_index"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"`{field_name}` must be a positive integer.")

    def diagnostic(
        self,
        stage: PlanningDiagnosticStage,
        code: str,
        outcome: str,
        *,
        response_byte_count: int | None = None,
        candidate_constructed: bool = False,
        candidate_preflight_passed: bool | None = None,
        step_index: int | None = None,
        argument_name: str | None = None,
        input_name: str | None = None,
        producer_step_index: int | None = None,
        output_key: str | None = None,
        tool_name: str | None = None,
        reason_code: str | None = None,
        retry_after_seconds: float | None = None,
        previous_failure_stage: PlanningDiagnosticStage | None = None,
        previous_failure_code: str | None = None,
        recovery_action: str | None = None,
        recovery_suppression_reason: str | None = None,
        total_provider_call_count: int | None = None,
        final_recovery_outcome: str | None = None,
        retry_used: bool | None = None,
        repair_used: bool | None = None,
        failover_used: bool | None = None,
        recovery_policy_fingerprint: str | None = None,
    ) -> PlanningDiagnostic:
        return PlanningDiagnostic(
            context=self,
            stage=stage,
            code=code,
            outcome=outcome,
            response_byte_count=response_byte_count,
            candidate_constructed=candidate_constructed,
            candidate_preflight_passed=candidate_preflight_passed,
            step_index=step_index,
            argument_name=safe_diagnostic_identifier(argument_name),
            input_name=safe_diagnostic_identifier(input_name),
            producer_step_index=producer_step_index,
            output_key=safe_diagnostic_identifier(output_key),
            tool_name=safe_diagnostic_identifier(tool_name),
            reason_code=safe_diagnostic_identifier(reason_code),
            retry_after_seconds=retry_after_seconds,
            previous_failure_stage=previous_failure_stage,
            previous_failure_code=safe_diagnostic_identifier(
                previous_failure_code
            ),
            recovery_action=safe_diagnostic_identifier(recovery_action),
            recovery_suppression_reason=safe_diagnostic_identifier(
                recovery_suppression_reason
            ),
            total_provider_call_count=total_provider_call_count,
            final_recovery_outcome=safe_diagnostic_identifier(
                final_recovery_outcome
            ),
            retry_used=retry_used,
            repair_used=repair_used,
            failover_used=failover_used,
            recovery_policy_fingerprint=recovery_policy_fingerprint,
        )


@dataclass(frozen=True)
class PlanningDiagnostic:
    """One immutable JSON-safe planning-stage observation."""

    context: PlanningDiagnosticContext
    stage: PlanningDiagnosticStage
    code: str
    outcome: str
    response_byte_count: int | None = None
    candidate_constructed: bool = False
    candidate_preflight_passed: bool | None = None
    step_index: int | None = None
    argument_name: str | None = None
    input_name: str | None = None
    producer_step_index: int | None = None
    output_key: str | None = None
    tool_name: str | None = None
    reason_code: str | None = None
    retry_after_seconds: float | None = None
    previous_failure_stage: PlanningDiagnosticStage | None = None
    previous_failure_code: str | None = None
    recovery_action: str | None = None
    recovery_suppression_reason: str | None = None
    total_provider_call_count: int | None = None
    final_recovery_outcome: str | None = None
    retry_used: bool | None = None
    repair_used: bool | None = None
    failover_used: bool | None = None
    recovery_policy_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.context, PlanningDiagnosticContext):
            raise TypeError("`context` must be a PlanningDiagnosticContext.")
        if not isinstance(self.stage, PlanningDiagnosticStage):
            raise TypeError("`stage` must be a PlanningDiagnosticStage.")
        if safe_diagnostic_identifier(self.code) != self.code:
            raise ValueError("Diagnostic code must be a safe stable identifier.")
        if self.outcome not in _OUTCOMES:
            raise ValueError("Diagnostic outcome is invalid.")
        for field_name in (
            "response_byte_count",
            "step_index",
            "producer_step_index",
            "total_provider_call_count",
        ):
            value = getattr(self, field_name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"`{field_name}` must be nonnegative or None.")
        if self.retry_after_seconds is not None and (
            isinstance(self.retry_after_seconds, bool)
            or not isinstance(self.retry_after_seconds, (int, float))
            or not math.isfinite(float(self.retry_after_seconds))
            or self.retry_after_seconds < 0
        ):
            raise ValueError("`retry_after_seconds` must be nonnegative or None.")
        if self.previous_failure_stage is not None and not isinstance(
            self.previous_failure_stage, PlanningDiagnosticStage
        ):
            raise TypeError(
                "`previous_failure_stage` must be PlanningDiagnosticStage or None."
            )
        for field_name in ("retry_used", "repair_used", "failover_used"):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"`{field_name}` must be boolean or None.")
        if self.recovery_policy_fingerprint is not None and not re.fullmatch(
            r"[0-9a-f]{64}", self.recovery_policy_fingerprint
        ):
            raise ValueError("Recovery policy fingerprint must be SHA-256 or None.")
        if not isinstance(self.candidate_constructed, bool):
            raise TypeError("`candidate_constructed` must be boolean.")
        if self.candidate_preflight_passed is not None and not isinstance(
            self.candidate_preflight_passed, bool
        ):
            raise TypeError("`candidate_preflight_passed` must be boolean or None.")
        for field_name in (
            "argument_name",
            "input_name",
            "output_key",
            "tool_name",
            "reason_code",
            "previous_failure_code",
            "recovery_action",
            "recovery_suppression_reason",
            "final_recovery_outcome",
        ):
            value = getattr(self, field_name)
            if value is not None and safe_diagnostic_identifier(value) != value:
                raise ValueError(f"`{field_name}` is not a safe identifier.")

    def to_details(self) -> Mapping[str, JsonValue]:
        details: dict[str, JsonValue] = {
            "diagnostic_schema_version": PLANNING_DIAGNOSTIC_SCHEMA_VERSION,
            "stage": self.stage.value,
            "code": self.code,
            "outcome": self.outcome,
            "attempt_kind": self.context.attempt_kind.value,
            "logical_attempt_index": self.context.logical_attempt_index,
            "provider_call_index": self.context.provider_call_index,
            "profile_id": self.context.profile_id,
            "provider_id": self.context.provider_id,
            "model_identity_digest": self.context.model_identity_digest,
            "planning_wire_schema_version": (
                self.context.planning_wire_schema_version
            ),
            "catalog_fingerprint": self.context.catalog_fingerprint,
            "offered_tool_names": self.context.offered_tool_names,
            "candidate_constructed": self.candidate_constructed,
            "candidate_preflight_passed": self.candidate_preflight_passed,
            "retry_used": (
                self.context.attempt_kind is PlanningAttemptKind.TRANSPORT_RETRY
                if self.retry_used is None
                else self.retry_used
            ),
            "repair_used": (
                self.context.attempt_kind is PlanningAttemptKind.REPAIR
                if self.repair_used is None
                else self.repair_used
            ),
            "failover_used": (
                self.context.attempt_kind is PlanningAttemptKind.FAILOVER
                if self.failover_used is None
                else self.failover_used
            ),
        }
        optional = {
            "response_byte_count": self.response_byte_count,
            "step_index": self.step_index,
            "argument_name": self.argument_name,
            "input_name": self.input_name,
            "producer_step_index": self.producer_step_index,
            "output_key": self.output_key,
            "tool_name": self.tool_name,
            "reason_code": self.reason_code,
            "retry_after_seconds": self.retry_after_seconds,
            "previous_failure_stage": (
                None
                if self.previous_failure_stage is None
                else self.previous_failure_stage.value
            ),
            "previous_failure_code": self.previous_failure_code,
            "recovery_action": self.recovery_action,
            "recovery_suppression_reason": self.recovery_suppression_reason,
            "total_provider_call_count": self.total_provider_call_count,
            "final_recovery_outcome": self.final_recovery_outcome,
            "recovery_policy_fingerprint": self.recovery_policy_fingerprint,
        }
        details.update(
            {key: value for key, value in optional.items() if value is not None}
        )
        return MappingProxyType(details)


@dataclass(frozen=True)
class DiagnosedPlanningAttempt:
    """A constructed plan and the diagnostics produced before runtime preflight."""

    plan: AgentPlan
    context: PlanningDiagnosticContext
    diagnostics: tuple[PlanningDiagnostic, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.plan, AgentPlan):
            raise TypeError("`plan` must be an AgentPlan.")
        if not isinstance(self.context, PlanningDiagnosticContext):
            raise TypeError("`context` must be a PlanningDiagnosticContext.")
        if not isinstance(self.diagnostics, tuple) or not all(
            isinstance(item, PlanningDiagnostic) for item in self.diagnostics
        ):
            raise TypeError("`diagnostics` must contain PlanningDiagnostic values.")


__all__ = [
    "DiagnosedPlanningAttempt",
    "PLANNING_DIAGNOSTIC_SCHEMA_VERSION",
    "PlanningAttemptKind",
    "PlanningDiagnostic",
    "PlanningDiagnosticContext",
    "PlanningDiagnosticStage",
    "safe_diagnostic_identifier",
]
