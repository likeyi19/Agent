"""Offline tests for immutable planning-model profiles and factories."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json

import pytest

import agent.application.cli as cli_module
import agent.providers.gemini_planning as gemini_module
import agent.providers.groq_planning as groq_module
import agent.providers.openai_planning as openai_module
import benchmarks.planner.run_benchmark as benchmark_cli
from agent.orchestration import (
    DeterministicPlanner,
    PlanningModel,
    PlanningModelProfile,
)
from agent.providers import (
    BUILTIN_PLANNING_PROVIDER_IDS,
    PlanningModelFactoryError,
    PlanningModelFactoryRegistry,
    build_default_planning_model_factory_registry,
    build_planning_model_profile,
)


class StubPlanningModel:
    def __init__(self, model_id: str = "custom:stub-model") -> None:
        self.model_id = model_id

    def complete(self, *, prompt: str, response_schema) -> str:
        del prompt, response_schema
        return json.dumps(
            {
                "schema_version": 3,
                "status": "unsupported",
                "steps": [],
                "reason": "stub",
            }
        )


def _profile(**changes: object) -> PlanningModelProfile:
    values: dict[str, object] = {
        "profile_id": "primary-planner",
        "provider_id": "custom",
        "model_id": "organization/model-v1",
        "enabled": True,
        "supports_structured_output": True,
        "request_timeout_seconds": 30.0,
    }
    values.update(changes)
    return PlanningModelProfile(**values)  # type: ignore[arg-type]


def test_profile_is_validated_normalized_and_immutable() -> None:
    profile = _profile(request_timeout_seconds=30)

    assert profile.request_timeout_seconds == 30.0
    assert profile.enabled
    assert profile.supports_structured_output
    with pytest.raises(FrozenInstanceError):
        profile.model_id = "replacement"  # type: ignore[misc]


@pytest.mark.parametrize(
    "changes",
    [
        {"profile_id": "Uppercase"},
        {"profile_id": "path/profile"},
        {"provider_id": "Provider"},
        {"provider_id": "provider.value"},
        {"model_id": " model"},
        {"model_id": "model with spaces"},
        {"enabled": 1},
        {"supports_structured_output": 1},
        {"request_timeout_seconds": True},
        {"request_timeout_seconds": 0},
        {"request_timeout_seconds": float("inf")},
        {"request_timeout_seconds": 3600.1},
    ],
)
def test_invalid_profile_configuration_is_rejected(
    changes: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        _profile(**changes)


@pytest.mark.parametrize(
    ("changes", "code"),
    [
        (
            {"enabled": False},
            "PLANNING_MODEL_PROFILE_DISABLED",
        ),
        (
            {"supports_structured_output": False},
            "PLANNING_MODEL_CAPABILITY_UNSUPPORTED",
        ),
    ],
)
def test_disabled_or_incapable_profile_is_rejected_before_factory(
    changes: dict[str, object],
    code: str,
) -> None:
    calls = 0

    def factory(profile: PlanningModelProfile) -> PlanningModel:
        nonlocal calls
        calls += 1
        return StubPlanningModel(profile.model_id)

    registry = PlanningModelFactoryRegistry({"custom": factory})

    with pytest.raises(PlanningModelFactoryError) as raised:
        registry.create(_profile(**changes))

    assert raised.value.code == code
    assert calls == 0


def test_unknown_provider_is_rejected_without_fallback() -> None:
    registry = PlanningModelFactoryRegistry({"custom": lambda _: StubPlanningModel()})

    with pytest.raises(PlanningModelFactoryError) as raised:
        registry.create(_profile(provider_id="unknown"))

    assert raised.value.code == "PLANNING_MODEL_PROVIDER_UNKNOWN"


@pytest.mark.parametrize(
    ("provider_id", "module", "attribute"),
    [
        ("openai", openai_module, "OpenAIPlanningModel"),
        ("gemini", gemini_module, "GeminiPlanningModel"),
        ("groq", groq_module, "GroqPlanningModel"),
    ],
)
def test_builtin_factory_constructs_only_the_selected_adapter(
    provider_id: str,
    module: object,
    attribute: str,
    monkeypatch,
) -> None:
    calls: list[dict[str, object]] = []

    def constructor(**kwargs: object) -> StubPlanningModel:
        calls.append(dict(kwargs))
        return StubPlanningModel(f"{provider_id}:{kwargs['model']}")

    monkeypatch.setattr(module, attribute, constructor)
    profile = _profile(provider_id=provider_id)

    model = build_default_planning_model_factory_registry().create(profile)

    assert isinstance(model, PlanningModel)
    assert calls == [
        {"model": profile.model_id, "timeout": profile.request_timeout_seconds}
    ]


def test_custom_factory_injection_and_registry_are_immutable() -> None:
    received: list[PlanningModelProfile] = []

    def factory(profile: PlanningModelProfile) -> PlanningModel:
        received.append(profile)
        return StubPlanningModel(f"custom:{profile.model_id}")

    factories = {"custom": factory}
    registry = PlanningModelFactoryRegistry(factories)
    factories["later"] = factory
    profile = _profile()

    model = registry.create(profile)

    assert model.model_id == f"custom:{profile.model_id}"
    assert received == [profile]
    assert registry.provider_ids == ("custom",)
    with pytest.raises(TypeError):
        registry._factories["later"] = factory  # type: ignore[index]


def test_building_registry_does_not_construct_provider_clients(
    monkeypatch,
) -> None:
    def fail() -> object:
        raise AssertionError("provider client must remain lazy")

    monkeypatch.setattr(openai_module, "_default_client", fail)
    monkeypatch.setattr(gemini_module, "_default_client", fail)
    monkeypatch.setattr(groq_module, "_default_client", fail)

    registry = build_default_planning_model_factory_registry()

    assert registry.provider_ids == tuple(sorted(BUILTIN_PLANNING_PROVIDER_IDS))


def test_derived_profiles_are_stable_and_distinguish_models() -> None:
    first = build_planning_model_profile("groq", "model-a", request_timeout_seconds=60)
    equivalent = build_planning_model_profile(
        "groq", "model-a", request_timeout_seconds=60.0
    )
    second = build_planning_model_profile("groq", "model-b")

    assert first == equivalent
    assert first.profile_id != second.profile_id
    assert first.model_id != second.model_id


def test_cli_and_benchmark_use_the_same_profile_factory_contract(
    monkeypatch,
) -> None:
    class RecordingRegistry:
        def __init__(self) -> None:
            self.profiles: list[PlanningModelProfile] = []

        def create(self, profile: PlanningModelProfile) -> PlanningModel:
            self.profiles.append(profile)
            return StubPlanningModel(f"{profile.provider_id}:{profile.model_id}")

    registry = RecordingRegistry()
    monkeypatch.setattr(
        cli_module,
        "build_default_planning_model_factory_registry",
        lambda: registry,
    )
    monkeypatch.setattr(
        benchmark_cli,
        "build_default_planning_model_factory_registry",
        lambda: registry,
    )

    planner = cli_module._planner("groq", "model-a")
    benchmark_profile, benchmark_model = benchmark_cli._live_model(
        "groq", "model-a", 60.0
    )

    assert planner.profile == benchmark_profile
    assert benchmark_model.model_id == "groq:model-a"
    assert registry.profiles == [benchmark_profile, benchmark_profile]
    assert isinstance(cli_module._planner("deterministic", None), DeterministicPlanner)


def test_profile_and_registry_never_receive_environment_credentials(
    monkeypatch,
) -> None:
    secret = "provider-api-secret"
    monkeypatch.setenv("CUSTOM_API_KEY", secret)
    received: list[PlanningModelProfile] = []

    def factory(profile: PlanningModelProfile) -> PlanningModel:
        received.append(profile)
        return StubPlanningModel()

    profile = _profile()
    registry = PlanningModelFactoryRegistry({"custom": factory})
    registry.create(profile)

    assert received == [profile]
    assert secret not in repr(profile)
    assert secret not in repr(registry)
    assert set(profile.__dict__) == {
        "profile_id",
        "provider_id",
        "model_id",
        "enabled",
        "supports_structured_output",
        "request_timeout_seconds",
    }
