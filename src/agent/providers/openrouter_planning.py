"""Optional OpenRouter Chat Completions transport for structured planning.

The default request permits at most 4,096 completion tokens and provider prices
of USD 2/10 per million prompt/completion tokens, with no per-request fee. These
are output and unit-price bounds, not an absolute request, turn or service budget.
An operator profile may explicitly configure finite nonnegative provider rates;
the completion-token maximum remains fixed. No spending
ledger, catalog discovery, model substitution or adapter retry is performed.
"""

from __future__ import annotations

import logging
import os
from types import MappingProxyType
from typing import Mapping

from agent.schemas import JsonValue
from agent.orchestration.planning_model import (
    PlanningModelError,
    PlanningModelProfile,
    _validate_provider_max_price,
    classify_provider_exception,
    normalized_retry_after_seconds,
)

from .openai_planning import _plain_json
from agent.orchestration.planning_diagnostics import log_operation_event

logger = logging.getLogger(__name__)


_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_MAX_COMPLETION_TOKENS = 4096
_DEFAULT_PROVIDER_PRICES = MappingProxyType({"prompt": 2, "completion": 10, "request": 0})


class OpenRouterPlanningError(PlanningModelError):
    """Sanitized failure at the OpenRouter planning-provider boundary."""


class OpenRouterPlanningDependencyError(OpenRouterPlanningError, ImportError):
    """Raised when the optional OpenAI SDK transport is unavailable."""


def _default_client() -> object:
    try:
        from openai import OpenAI
    except ImportError:
        raise OpenRouterPlanningDependencyError(
            "The optional OpenAI SDK is not installed. Install the `openai` "
            "package to use OpenRouterPlanningModel.",
            code="PLANNING_PROVIDER_DEPENDENCY_MISSING",
        ) from None
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key or not api_key.strip():
        raise OpenRouterPlanningError(
            "OpenRouter client initialization failed. Ensure OPENROUTER_API_KEY "
            "is configured in the server environment.",
            code="PLANNING_PROVIDER_CONFIGURATION_FAILED",
        )
    try:
        return OpenAI(api_key=api_key, base_url=_OPENROUTER_BASE_URL, max_retries=0)
    except Exception:
        raise OpenRouterPlanningError(
            "OpenRouter client initialization failed.",
            code="PLANNING_PROVIDER_CONFIGURATION_FAILED",
        ) from None


def _field(value: object, name: str, default: object = None) -> object:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _check_response_identity(response: object, model: str) -> None:
    returned_model = _field(response, "model")
    metadata = _field(response, "openrouter_metadata")
    # :floor selects capacity/price routing for the same underlying model.
    # Accept its exact base identity only with an explicit requested-slug echo;
    # other aliases, variants and missing routing evidence remain fail-closed.
    base, separator, variant = model.rpartition(":")
    floor_base = (separator and variant == "floor" and base and ":" not in base
                  and returned_model == base and isinstance(metadata, Mapping)
                  and _field(metadata, "requested") == model)
    if type(returned_model) is not str or (returned_model != model and not floor_base):
        raise OpenRouterPlanningError("OpenRouter returned an unexpected model identity.")
    if metadata is None:
        log_operation_event(logger, 'openrouter.routing_validated', requested_model=model,
                            observed_model=returned_model, verified_model=returned_model, reason='metadata_absent', transport=response)
        return  # Missing observations are not evidence of upstream routing behavior.
    if not isinstance(metadata, Mapping):
        raise OpenRouterPlanningError("OpenRouter returned invalid routing metadata.")
    requested = _field(metadata, "requested")
    pipeline = _field(metadata, "pipeline", [])
    attempt = _field(metadata, "attempt")
    if (
        requested is not None and requested != model
        or not isinstance(pipeline, (list, tuple))
        or bool(pipeline)
        or attempt is not None and (type(attempt) is not int or not 0 <= attempt <= 1)
    ):
        raise OpenRouterPlanningError("OpenRouter response violated the required routing policy.")
    log_operation_event(logger, 'openrouter.routing_validated', requested_model=model,
                        observed_model=returned_model, verified_model=returned_model, reason='metadata_checked', transport=response)


def _completed_content(response: object, model: str) -> str:
    if _field(response, "error") is not None:
        raise OpenRouterPlanningError("OpenRouter planning response reported a failure.")
    _check_response_identity(response, model)
    choices = _field(response, "choices")
    if not isinstance(choices, (list, tuple)) or len(choices) != 1:
        raise OpenRouterPlanningError(
            "OpenRouter did not return one completion.", code="PROVIDER_COMPLETION_INCOMPLETE"
        )
    choice = choices[0]
    message = _field(choice, "message")
    if _field(message, "refusal"):
        raise OpenRouterPlanningError("OpenRouter refused the request.", code="PROVIDER_REFUSED")
    content = _field(message, "content")
    if (
        _field(choice, "finish_reason") != "stop"
        or not isinstance(content, str)
        or not content.strip()
        or _field(message, "tool_calls")
    ):
        raise OpenRouterPlanningError(
            "OpenRouter completion was incomplete or lacked text.",
            code="PROVIDER_COMPLETION_INCOMPLETE",
        )
    return content


class OpenRouterPlanningModel:
    """One exact-model strict-schema request behind the shared PlanningModel API."""

    def __init__(
        self,
        *,
        model: str,
        timeout: float = 60.0,
        max_completion_tokens: int = _MAX_COMPLETION_TOKENS,
        provider_max_price: Mapping[str, float | int | str] | None = None,
        _client: object | None = None,
    ) -> None:
        profile = PlanningModelProfile(
            "openrouter-adapter", "openrouter", model, request_timeout_seconds=timeout
        )
        if type(max_completion_tokens) is not int or not 1 <= max_completion_tokens <= _MAX_COMPLETION_TOKENS:
            raise ValueError("Completion token limit must be an integer between 1 and 4096.")
        self._model = profile.model_id
        self._timeout = profile.request_timeout_seconds
        self._max_completion_tokens = max_completion_tokens
        self._provider_max_price = _validate_provider_max_price(provider_max_price) or _DEFAULT_PROVIDER_PRICES
        client = _default_client() if _client is None else _client
        if not callable(getattr(getattr(getattr(client, "chat", None), "completions", None), "create", None)):
            raise TypeError("OpenRouter planning client must provide chat.completions.create().")
        self._client = client

    @property
    def model_id(self) -> str:
        return f"openrouter:{self._model}"

    @property
    def model(self) -> str:
        return self._model

    @property
    def timeout(self) -> float:
        return self._timeout

    @property
    def max_completion_tokens(self) -> int:
        return self._max_completion_tokens

    @property
    def provider_max_price(self) -> Mapping[str, float | int]:
        return self._provider_max_price

    def complete(self, *, prompt: str, response_schema: Mapping[str, JsonValue]) -> str:
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("`prompt` must be a non-empty string.")
        if not isinstance(response_schema, Mapping):
            raise TypeError("`response_schema` must be a mapping.")
        schema = _plain_json(response_schema)
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_schema", "json_schema": {
                    "name": "agent_plan", "strict": True, "schema": schema,
                }},
                stream=False,
                timeout=self._timeout,
                max_completion_tokens=self._max_completion_tokens,
                extra_body={
                    "provider": {"require_parameters": True, "allow_fallbacks": False,
                                 "max_price": dict(self._provider_max_price)},
                    "plugins": [{"id": name, "enabled": False} for name in
                                ("response-healing", "context-compression", "web")],
                },
                extra_headers={"X-OpenRouter-Metadata": "enabled"},
            )
        except Exception as exc:
            code, message = classify_provider_exception(exc)
            log_operation_event(logger, 'openrouter.sdk_failed', requested_model=self._model,
                                error=exc, code=code, transport=exc)
            raise OpenRouterPlanningError(
                message, code=code, retry_after_seconds=normalized_retry_after_seconds(exc)
            ) from None
        try:
            content = _completed_content(response, self._model)
        except Exception as exc:
            log_operation_event(logger, 'openrouter.response_validation_failed', requested_model=self._model,
                                error=exc, transport=response)
            raise
        return content


__all__ = ["OpenRouterPlanningDependencyError", "OpenRouterPlanningError", "OpenRouterPlanningModel"]
