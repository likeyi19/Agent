"""Optional Alibaba Model Studio transport at the shared PlanningModel boundary.

The operator supplies the workspace endpoint through the server environment.
Requests use the unchanged Agent schema, non-thinking mode and a bounded 4,096
completion tokens. Only assistant content is consumed; private reasoning is
never inspected, persisted or logged. No adapter retry or model routing occurs.
"""
from __future__ import annotations

import logging
import os
from typing import Mapping
from urllib.parse import urlsplit

from agent.schemas import JsonValue
from agent.orchestration.planning_model import (
    PlanningModelError,
    PlanningModelProfile,
    classify_provider_exception,
    normalized_retry_after_seconds,
)
from agent.orchestration.planning_diagnostics import log_operation_event
from .openai_planning import _field, _plain_json

logger = logging.getLogger(__name__)

_MAX_COMPLETION_TOKENS = 4096


class AlibabaPlanningError(PlanningModelError):
    """Sanitized Model Studio transport or completion failure."""


class AlibabaPlanningDependencyError(AlibabaPlanningError, ImportError):
    """The optional OpenAI-compatible SDK transport is unavailable."""


def _default_client() -> object:
    try:
        from openai import OpenAI
    except ImportError:
        raise AlibabaPlanningDependencyError(
            "Install the optional openai SDK to use AlibabaPlanningModel.",
            code="PLANNING_PROVIDER_DEPENDENCY_MISSING",
        ) from None
    key = os.environ.get("DASHSCOPE_API_KEY")
    endpoint = os.environ.get("DASHSCOPE_BASE_URL")
    try:
        url = urlsplit(endpoint or "")
        valid = (
            key and key.strip() and endpoint and endpoint == endpoint.strip()
            and url.scheme == "https" and url.hostname and not url.username
            and not url.password and not url.query and not url.fragment
        )
    except ValueError:
        valid = False
    if not valid:
        raise AlibabaPlanningError(
            "Configure DASHSCOPE_API_KEY and the HTTPS DASHSCOPE_BASE_URL "
            "in the server environment.",
            code="PLANNING_PROVIDER_CONFIGURATION_FAILED",
        )
    try:
        return OpenAI(api_key=key, base_url=endpoint, max_retries=0)
    except Exception:
        raise AlibabaPlanningError(
            "Alibaba client initialization failed.",
            code="PLANNING_PROVIDER_CONFIGURATION_FAILED",
        ) from None


def _completed_content(response: object, model: str) -> str:
    if _field(response, "error") is not None or _field(response, "model") != model:
        raise AlibabaPlanningError("Alibaba returned a failed response or unexpected model identity.")
    choices = _field(response, "choices")
    if not isinstance(choices, (list, tuple)) or len(choices) != 1:
        raise AlibabaPlanningError(
            "Alibaba did not return one completion.", code="PROVIDER_COMPLETION_INCOMPLETE"
        )
    message = _field(choices[0], "message")
    if _field(message, "refusal"):
        raise AlibabaPlanningError("Alibaba refused the request.", code="PROVIDER_REFUSED")
    content = _field(message, "content")
    if (
        _field(choices[0], "finish_reason") != "stop"
        or _field(message, "role") != "assistant"
        or not isinstance(content, str) or not content.strip()
        or _field(message, "tool_calls") or _field(message, "function_call")
    ):
        raise AlibabaPlanningError(
            "Alibaba completion was incomplete or lacked assistant text.",
            code="PROVIDER_COMPLETION_INCOMPLETE",
        )
    return content


class AlibabaPlanningModel:
    """One explicitly selected model behind the ordinary PlanningModel API."""

    def __init__(self, *, model: str, timeout: float = 60.0, _client: object | None = None) -> None:
        profile = PlanningModelProfile(
            "alibaba-adapter", "alibaba", model, request_timeout_seconds=timeout
        )
        self._model = profile.model_id
        self._timeout = profile.request_timeout_seconds
        client = _default_client() if _client is None else _client
        if not callable(getattr(getattr(getattr(client, "chat", None), "completions", None), "create", None)):
            raise TypeError("Alibaba planning client must provide chat.completions.create().")
        self._client = client

    @property
    def model_id(self) -> str:
        return f"alibaba:{self._model}"

    @property
    def model(self) -> str:
        return self._model

    @property
    def timeout(self) -> float:
        return self._timeout

    def complete(self, *, prompt: str, response_schema: Mapping[str, JsonValue]) -> str:
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("`prompt` must be a nonempty string.")
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
                max_completion_tokens=_MAX_COMPLETION_TOKENS,
                extra_body={"enable_thinking": False},
            )
        except Exception as exc:
            code, message = classify_provider_exception(exc)
            log_operation_event(
                logger, "alibaba.sdk_failed", requested_model=self._model,
                error=exc, code=code, transport=exc,
            )
            raise AlibabaPlanningError(
                message, code=code, retry_after_seconds=normalized_retry_after_seconds(exc)
            ) from None
        try:
            content = _completed_content(response, self._model)
        except AlibabaPlanningError as exc:
            log_operation_event(
                logger, "alibaba.response_validation_failed", requested_model=self._model,
                error=exc, code=exc.code, transport=response,
            )
            raise
        log_operation_event(
            logger, "alibaba.completion_validated", requested_model=self._model,
            observed_model=_field(response, "model"), verified_model=self._model,
            transport=response,
        )
        return content


__all__ = ["AlibabaPlanningModel", "AlibabaPlanningError", "AlibabaPlanningDependencyError"]
