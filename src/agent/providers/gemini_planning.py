"""Optional Gemini Interactions API adapter for provider-neutral planning."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

from agent.schemas import JsonValue
from agent.orchestration.planning_model import (
    PlanningModelError,
    classify_provider_exception,
)


_DEFAULT_TIMEOUT_SECONDS = 60.0


class GeminiPlanningError(PlanningModelError):
    """Sanitized failure at the Gemini planning-provider boundary."""


class GeminiPlanningDependencyError(GeminiPlanningError, ImportError):
    """Raised when the optional official Google Gen AI SDK is unavailable."""


def _validate_model(model: object) -> str:
    if not isinstance(model, str) or not model.strip():
        raise ValueError("`model` must be a non-empty Gemini model name.")
    return model.strip()


def _validate_timeout(timeout: object) -> float:
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("`timeout` must be a positive finite number of seconds.")
    return float(timeout)


def _default_client() -> object:
    try:
        from google import genai
    except ImportError as exc:
        raise GeminiPlanningDependencyError(
            "The optional Google Gen AI SDK is not installed. Install the "
            "`google-genai` package to use GeminiPlanningModel."
        ) from exc
    try:
        return genai.Client()
    except Exception as exc:
        raise GeminiPlanningError(
            "Gemini client initialization failed. Ensure GEMINI_API_KEY is "
            "configured in the runtime environment."
        ) from exc


def _validate_client(client: object) -> object:
    interactions = getattr(client, "interactions", None)
    if not callable(getattr(interactions, "create", None)):
        raise TypeError(
            "Gemini planning client must provide interactions.create()."
        )
    return client


def _plain_json(value: object, *, path: str = "response_schema") -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"`{path}` must not contain non-finite numbers.")
        return value
    if isinstance(value, Mapping):
        copied: dict[str, object] = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise TypeError(f"`{path}` must contain only string keys.")
            copied[key] = _plain_json(nested, path=f"{path}.{key}")
        return copied
    if isinstance(value, (list, tuple)):
        return [
            _plain_json(nested, path=f"{path}[{index}]")
            for index, nested in enumerate(value)
        ]
    if isinstance(value, Path):
        return str(value)
    raise TypeError(
        f"`{path}` contains unsupported value {type(value).__name__}."
    )


def _field(value: object, name: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _completed_output_text(interaction: object) -> str:
    if _field(interaction, "error") is not None:
        raise GeminiPlanningError("Gemini planning interaction reported a failure.")
    if _field(interaction, "status") != "completed":
        raise GeminiPlanningError("Gemini planning interaction was not completed.")
    try:
        output_text = _field(interaction, "output_text")
    except Exception as exc:
        raise GeminiPlanningError(
            "Gemini planning interaction text could not be read."
        ) from exc
    if not isinstance(output_text, str) or not output_text.strip():
        raise GeminiPlanningError(
            "Gemini planning interaction did not contain output text."
        )
    return output_text


class GeminiPlanningModel:
    """PlanningModel adapter using one Gemini Interactions API request."""

    def __init__(
        self,
        *,
        model: str,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        _client: object | None = None,
    ) -> None:
        self._model = _validate_model(model)
        self._timeout = _validate_timeout(timeout)
        self._client = _validate_client(
            _default_client() if _client is None else _client
        )

    @property
    def model_id(self) -> str:
        return f"gemini:{self._model}"

    @property
    def model(self) -> str:
        return self._model

    @property
    def timeout(self) -> float:
        return self._timeout

    def complete(
        self,
        *,
        prompt: str,
        response_schema: Mapping[str, JsonValue],
    ) -> str:
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("`prompt` must be a non-empty string.")
        if not isinstance(response_schema, Mapping):
            raise TypeError("`response_schema` must be a mapping.")
        schema = _plain_json(response_schema)
        if not isinstance(schema, dict):  # pragma: no cover - mapping invariant
            raise TypeError("`response_schema` must serialize to a JSON object.")

        try:
            interaction = self._client.interactions.create(
                model=self._model,
                input=prompt,
                response_format={
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": schema,
                },
                store=False,
                background=False,
                timeout=self._timeout,
            )
        except Exception as exc:
            code, message = classify_provider_exception(exc)
            if code == "PLANNING_PROVIDER_ERROR":
                message = "Gemini planning request failed."
            raise GeminiPlanningError(
                message,
                code=code,
            ) from exc
        return _completed_output_text(interaction)


__all__ = [
    "GeminiPlanningDependencyError",
    "GeminiPlanningError",
    "GeminiPlanningModel",
]
