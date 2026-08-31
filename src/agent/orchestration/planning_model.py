"""Provider-neutral text generation boundary for structured planning."""

from __future__ import annotations

from typing import Mapping, Protocol, runtime_checkable

from agent.schemas import JsonValue

from .error_policy import safe_message_for


class PlanningModelError(RuntimeError):
    """Sanitized provider-neutral planning failure with one stable code."""

    def __init__(
        self,
        message: str = "Planning provider request failed.",
        *,
        code: str = "PLANNING_PROVIDER_ERROR",
    ) -> None:
        self.code = code
        super().__init__(message)


def classify_provider_exception(exception: Exception) -> tuple[str, str]:
    """Classify only reliable structured provider exception information."""

    status_code = getattr(exception, "status_code", None)
    names = {base.__name__ for base in type(exception).__mro__}
    if status_code in {401, 403} or "AuthenticationError" in names:
        code = "PROVIDER_AUTHENTICATION_FAILED"
    elif status_code == 429 or "RateLimitError" in names:
        code = "PROVIDER_RATE_LIMITED"
    elif isinstance(exception, TimeoutError) or names.intersection(
        {"APITimeoutError", "ReadTimeout", "ConnectTimeout"}
    ):
        code = "PROVIDER_TIMEOUT"
    elif isinstance(exception, ConnectionError) or names.intersection(
        {"APIConnectionError", "ConnectError", "ConnectionError"}
    ):
        code = "PROVIDER_CONNECTION_FAILED"
    elif isinstance(status_code, int) and status_code >= 500:
        code = "PROVIDER_UNAVAILABLE"
    else:
        code = "PLANNING_PROVIDER_ERROR"
    return code, safe_message_for(code)


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


__all__ = [
    "PlanningModel",
    "PlanningModelError",
    "classify_provider_exception",
]
