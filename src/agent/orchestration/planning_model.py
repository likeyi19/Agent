"""Provider-neutral text generation boundary for structured planning."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Mapping, Protocol, runtime_checkable

from agent.schemas import JsonValue

from .error_policy import safe_message_for


_PROFILE_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_MODEL_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$")
_MAX_REQUEST_TIMEOUT_SECONDS = 3600.0


@dataclass(frozen=True)
class PlanningModelProfile:
    """Non-secret deployment configuration for one planning model."""

    profile_id: str
    provider_id: str
    model_id: str
    enabled: bool = True
    supports_structured_output: bool = True
    request_timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or not _PROFILE_IDENTIFIER.fullmatch(
            self.profile_id
        ):
            raise ValueError("`profile_id` must be a safe lowercase identifier.")
        if not isinstance(self.provider_id, str) or not _PROFILE_IDENTIFIER.fullmatch(
            self.provider_id
        ):
            raise ValueError("`provider_id` must be a safe lowercase identifier.")
        if not isinstance(self.model_id, str) or not _MODEL_IDENTIFIER.fullmatch(
            self.model_id
        ):
            raise ValueError("`model_id` must be a safe provider model identifier.")
        if not isinstance(self.enabled, bool):
            raise TypeError("`enabled` must be boolean.")
        if not isinstance(self.supports_structured_output, bool):
            raise TypeError("`supports_structured_output` must be boolean.")
        timeout = self.request_timeout_seconds
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
            or timeout > _MAX_REQUEST_TIMEOUT_SECONDS
        ):
            raise ValueError(
                "`request_timeout_seconds` must be positive, finite, and at most "
                f"{_MAX_REQUEST_TIMEOUT_SECONDS:g}."
            )
        object.__setattr__(self, "request_timeout_seconds", float(timeout))


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
    "PlanningModelProfile",
    "classify_provider_exception",
]
