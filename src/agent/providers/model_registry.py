"""Immutable non-routing factory registry for configured planning models."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from types import MappingProxyType
from typing import Callable, Mapping

from agent.orchestration.planning_model import PlanningModel, PlanningModelProfile


PlanningModelFactory = Callable[[PlanningModelProfile], PlanningModel]
BUILTIN_PLANNING_PROVIDER_IDS = ("openai", "gemini", "groq")


class PlanningModelFactoryError(ValueError):
    """Stable configuration failure before a planning provider is invoked."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True, init=False)
class PlanningModelFactoryRegistry:
    """Resolve one explicitly selected profile through immutable factories."""

    _factories: Mapping[str, PlanningModelFactory]

    def __init__(self, factories: Mapping[str, PlanningModelFactory]) -> None:
        if not isinstance(factories, Mapping):
            raise TypeError("`factories` must be a mapping.")
        copied: dict[str, PlanningModelFactory] = {}
        for provider_id, factory in factories.items():
            try:
                PlanningModelProfile(
                    profile_id="registry-validation",
                    provider_id=provider_id,
                    model_id="validation-model",
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Factory provider identifiers must be safe lowercase identifiers."
                ) from exc
            if not callable(factory):
                raise TypeError("Every planning-model factory must be callable.")
            copied[provider_id] = factory
        object.__setattr__(self, "_factories", MappingProxyType(copied))

    @property
    def provider_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))

    def create(self, profile: PlanningModelProfile) -> PlanningModel:
        """Construct one model; never select, route, retry, repair, or fallback."""

        if not isinstance(profile, PlanningModelProfile):
            raise TypeError("`profile` must be a PlanningModelProfile.")
        if not profile.enabled:
            raise PlanningModelFactoryError(
                "PLANNING_MODEL_PROFILE_DISABLED",
                "The selected planning-model profile is disabled.",
            )
        if not profile.supports_structured_output:
            raise PlanningModelFactoryError(
                "PLANNING_MODEL_CAPABILITY_UNSUPPORTED",
                "The selected planning model lacks required structured output.",
            )
        factory = self._factories.get(profile.provider_id)
        if factory is None:
            raise PlanningModelFactoryError(
                "PLANNING_MODEL_PROVIDER_UNKNOWN",
                "The selected planning-model provider is not registered.",
            )
        model = factory(profile)
        if not isinstance(model, PlanningModel):
            raise TypeError("Planning-model factory returned an invalid adapter.")
        return model


def build_planning_model_profile(
    provider_id: str,
    model_id: str,
    *,
    request_timeout_seconds: float = 60.0,
) -> PlanningModelProfile:
    """Build a stable internal profile for direct provider/model selection."""

    validated = PlanningModelProfile(
        profile_id="planning-provisional",
        provider_id=provider_id,
        model_id=model_id,
        request_timeout_seconds=request_timeout_seconds,
    )
    identity = (
        f"{validated.provider_id}\0{validated.model_id}\0"
        f"{validated.request_timeout_seconds.hex()}"
    ).encode("utf-8")
    profile_id = f"planning-{hashlib.sha256(identity).hexdigest()[:24]}"
    return PlanningModelProfile(
        profile_id=profile_id,
        provider_id=validated.provider_id,
        model_id=validated.model_id,
        request_timeout_seconds=validated.request_timeout_seconds,
    )


def _openai_factory(profile: PlanningModelProfile) -> PlanningModel:
    from .openai_planning import OpenAIPlanningModel

    return OpenAIPlanningModel(
        model=profile.model_id,
        timeout=profile.request_timeout_seconds,
    )


def _gemini_factory(profile: PlanningModelProfile) -> PlanningModel:
    from .gemini_planning import GeminiPlanningModel

    return GeminiPlanningModel(
        model=profile.model_id,
        timeout=profile.request_timeout_seconds,
    )


def _groq_factory(profile: PlanningModelProfile) -> PlanningModel:
    from .groq_planning import GroqPlanningModel

    return GroqPlanningModel(
        model=profile.model_id,
        timeout=profile.request_timeout_seconds,
    )


def build_default_planning_model_factory_registry(
) -> PlanningModelFactoryRegistry:
    """Return the built-in provider factories without constructing any SDK client."""

    return PlanningModelFactoryRegistry(
        {
            "openai": _openai_factory,
            "gemini": _gemini_factory,
            "groq": _groq_factory,
        }
    )


__all__ = [
    "BUILTIN_PLANNING_PROVIDER_IDS",
    "PlanningModelFactory",
    "PlanningModelFactoryError",
    "PlanningModelFactoryRegistry",
    "build_default_planning_model_factory_registry",
    "build_planning_model_profile",
]
