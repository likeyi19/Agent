"""Shared planning protocol and typed failures for scientific planning."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from agent.schemas import (
    AgentPlan,
    AgentRequest,
    ErrorCategory,
)

from .planning_diagnostics import PlanningDiagnostic, PlanningDiagnosticStage
from .registry import ToolRegistry


@runtime_checkable
class Planner(Protocol):
    """Interface for components that produce plans without executing tools."""

    def plan(self, request: AgentRequest, registry: ToolRegistry) -> AgentPlan:
        """Produce a structurally valid plan using the registry vocabulary."""


class PlannerError(ValueError):
    """Stable planner failure suitable for future runtime classification."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        category: ErrorCategory = ErrorCategory.USER_INPUT_ERROR,
        diagnostics: tuple[PlanningDiagnostic, ...] = (),
        diagnostic_stage: PlanningDiagnosticStage | None = None,
        diagnostic_reason_code: str | None = None,
        diagnostic_fields: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.category = category
        if not isinstance(diagnostics, tuple) or not all(
            isinstance(item, PlanningDiagnostic) for item in diagnostics
        ):
            raise TypeError("`diagnostics` must contain PlanningDiagnostic values.")
        self.diagnostics = diagnostics
        if diagnostic_stage is not None and not isinstance(
            diagnostic_stage, PlanningDiagnosticStage
        ):
            raise TypeError("`diagnostic_stage` must be a PlanningDiagnosticStage.")
        self.diagnostic_stage = diagnostic_stage
        self.diagnostic_reason_code = diagnostic_reason_code
        self.diagnostic_fields = dict(diagnostic_fields or {})


__all__ = ["Planner", "PlannerError"]
