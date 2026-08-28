"""Provider-neutral text generation boundary for structured planning."""

from __future__ import annotations

from typing import Mapping, Protocol, runtime_checkable

from agent.schemas import JsonValue


@runtime_checkable
class PlanningModel(Protocol):
    """Minimal interface implemented by an external planning provider adapter."""

    @property
    def model_id(self) -> str:
        """Return a stable, non-secret model identity for plan provenance."""

    def complete(
        self,
        *,
        prompt: str,
        response_schema: Mapping[str, JsonValue],
    ) -> str:
        """Return exactly one text response containing the planning JSON."""


__all__ = ["PlanningModel"]
