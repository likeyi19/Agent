"""Sanitized provider-neutral diagnostics for one LLM planning attempt."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from types import MappingProxyType
from typing import Mapping

from agent.schemas import AgentPlan, JsonValue


PLANNING_DIAGNOSTIC_SCHEMA_VERSION = 2
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
        ):
            value = getattr(self, field_name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"`{field_name}` must be nonnegative or None.")
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
            "attempt_kind": "initial",
            "attempt_number": 1,
            "provider_call_number": 1,
            "total_provider_call_count": 1,
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
            "repair_used": False,
            "retry_used": False,
            "fallback_used": False,
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
    "PlanningDiagnostic",
    "PlanningDiagnosticContext",
    "PlanningDiagnosticStage",
    "safe_diagnostic_identifier",
]
