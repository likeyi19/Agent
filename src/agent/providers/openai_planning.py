"""Optional OpenAI Responses API adapter for provider-neutral planning."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

from agent.schemas import JsonValue


_DEFAULT_TIMEOUT_SECONDS = 60.0
_RESPONSE_FORMAT_NAME = "agent_plan"


class OpenAIPlanningError(RuntimeError):
    """Sanitized failure at the OpenAI planning-provider boundary."""


class OpenAIPlanningDependencyError(OpenAIPlanningError, ImportError):
    """Raised when the optional official OpenAI SDK is unavailable."""


def _validate_model(model: object) -> str:
    if not isinstance(model, str) or not model.strip():
        raise ValueError("`model` must be a non-empty OpenAI model name.")
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
        from openai import OpenAI
    except ImportError as exc:
        raise OpenAIPlanningDependencyError(
            "The optional OpenAI SDK is not installed. Install the `openai` "
            "package to use OpenAIPlanningModel."
        ) from exc
    try:
        return OpenAI()
    except Exception as exc:
        raise OpenAIPlanningError(
            "OpenAI client initialization failed. Ensure OPENAI_API_KEY is "
            "configured in the runtime environment."
        ) from exc


def _validate_client(client: object) -> object:
    responses = getattr(client, "responses", None)
    if not callable(getattr(responses, "create", None)):
        raise TypeError(
            "OpenAI planning client must provide responses.create()."
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


def _contains_refusal(response: object) -> bool:
    output = _field(response, "output", ())
    if not isinstance(output, (list, tuple)):
        return False
    for item in output:
        if _field(item, "type") != "message":
            continue
        content = _field(item, "content", ())
        if not isinstance(content, (list, tuple)):
            continue
        if any(_field(part, "type") == "refusal" for part in content):
            return True
    return False


def _completed_output_text(response: object) -> str:
    if _field(response, "error") is not None:
        raise OpenAIPlanningError("OpenAI planning response reported a failure.")
    if _field(response, "status") != "completed":
        raise OpenAIPlanningError("OpenAI planning response was not completed.")
    if _contains_refusal(response):
        raise OpenAIPlanningError("OpenAI planning response was refused.")
    try:
        output_text = _field(response, "output_text")
    except Exception as exc:
        raise OpenAIPlanningError(
            "OpenAI planning response text could not be read."
        ) from exc
    if not isinstance(output_text, str) or not output_text.strip():
        raise OpenAIPlanningError(
            "OpenAI planning response did not contain output text."
        )
    return output_text


class OpenAIPlanningModel:
    """PlanningModel adapter using one OpenAI Responses API request."""

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
        return f"openai:{self._model}"

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
            response = self._client.responses.create(
                model=self._model,
                input=prompt,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": _RESPONSE_FORMAT_NAME,
                        "strict": True,
                        "schema": schema,
                    }
                },
                store=False,
                background=False,
                timeout=self._timeout,
            )
        except Exception as exc:
            raise OpenAIPlanningError(
                "OpenAI planning request failed."
            ) from exc
        return _completed_output_text(response)


__all__ = [
    "OpenAIPlanningDependencyError",
    "OpenAIPlanningError",
    "OpenAIPlanningModel",
]
