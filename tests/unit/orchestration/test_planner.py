"""Offline tests for the deliberately narrow deterministic planner."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import Mock

import pytest

from agent.orchestration import (
    DeterministicPlanner,
    ErrorCategory,
    Planner,
    PlannerError,
    RunMode,
    StepOutputRef,
    ToolRegistry,
    build_default_tool_registry,
)
from agent.schemas import AgentRequest


@pytest.fixture
def registry() -> ToolRegistry:
    return build_default_tool_registry()


def _request(
    prompt: str,
    inputs: dict[str, object],
    *,
    mode: RunMode = RunMode.EXECUTE,
) -> AgentRequest:
    return AgentRequest("request-1", prompt, inputs, mode)


def test_deterministic_planner_satisfies_protocol() -> None:
    assert isinstance(DeterministicPlanner(), Planner)


def test_inspection_request_produces_exact_one_step_plan(registry) -> None:
    plan = DeterministicPlanner().plan(
        _request("Inspect this scATAC-seq dataset", {"input_path": "/data/in.h5ad"}),
        registry,
    )

    assert plan.plan_id == "request-1:inspection"
    assert plan.request_id == "request-1"
    assert plan.planner_name == "deterministic"
    assert len(plan.steps) == 1
    assert plan.steps[0].step_id == "inspect"
    assert plan.steps[0].tool_name == "inspect_scATAC"
    assert dict(plan.steps[0].arguments) == {"path": "/data/in.h5ad"}
    assert plan.steps[0].depends_on == ()


def test_embedding_request_produces_exact_two_step_plan(registry) -> None:
    plan = DeterministicPlanner().plan(
        _request(
            "Compute EpiZoo embeddings",
            {
                "input_path": "/data/in.h5ad",
                "output_dir": "/output",
                "species": "mouse",
            },
        ),
        registry,
    )

    assert plan.plan_id == "request-1:epizoo-embedding"
    assert tuple(step.step_id for step in plan.steps) == ("inspect", "embed")
    assert tuple(step.tool_name for step in plan.steps) == (
        "inspect_scATAC",
        "epizoo_embed_cells",
    )
    assert plan.steps[1].depends_on == ("inspect",)
    assert plan.steps[1].arguments["input_path"] == StepOutputRef(
        "inspect", "input_path"
    )


def test_repeated_planning_is_deterministic(registry) -> None:
    request = _request(
        "embed this dataset",
        {
            "input_path": "/data/in.h5ad",
            "output_dir": "/output",
            "species": "human",
        },
    )
    planner = DeterministicPlanner()

    assert planner.plan(request, registry) == planner.plan(request, registry)
    assert planner.plan(request, registry).to_dict() == planner.plan(
        request, registry
    ).to_dict()


def test_embedding_intent_takes_precedence_over_inspection(registry) -> None:
    plan = DeterministicPlanner().plan(
        _request(
            "Inspect the dataset and compute EpiZoo embeddings",
            {
                "input_path": "/data/in.h5ad",
                "output_dir": "/output",
                "species": "mouse",
            },
        ),
        registry,
    )

    assert tuple(step.step_id for step in plan.steps) == ("inspect", "embed")


@pytest.mark.parametrize(
    ("prompt", "inputs", "missing_name"),
    [
        ("inspect dataset", {}, "input_path"),
        (
            "compute embeddings",
            {"output_dir": "/output", "species": "mouse"},
            "input_path",
        ),
        (
            "compute embeddings",
            {"input_path": "/data/in.h5ad", "species": "mouse"},
            "output_dir",
        ),
        (
            "compute embeddings",
            {"input_path": "/data/in.h5ad", "output_dir": "/output"},
            "species",
        ),
    ],
)
def test_workflows_require_structured_inputs(
    registry, prompt, inputs, missing_name
) -> None:
    with pytest.raises(PlannerError) as raised:
        DeterministicPlanner().plan(_request(prompt, inputs), registry)

    assert raised.value.code == "MISSING_REQUIRED_INPUT"
    assert raised.value.category is ErrorCategory.USER_INPUT_ERROR
    assert missing_name in str(raised.value)


@pytest.mark.parametrize("species", ["rat", "", 1, None])
def test_invalid_species_is_rejected(registry, species) -> None:
    with pytest.raises(PlannerError) as raised:
        DeterministicPlanner().plan(
            _request(
                "compute embedding",
                {
                    "input_path": "/data/in.h5ad",
                    "output_dir": "/output",
                    "species": species,
                },
            ),
            registry,
        )

    assert raised.value.code == "INVALID_REQUEST_INPUT"


def test_empty_path_input_is_rejected(registry) -> None:
    with pytest.raises(PlannerError, match="non-empty path"):
        DeterministicPlanner().plan(
            _request("inspect", {"input_path": "  "}), registry
        )


def test_optional_embedding_arguments_are_forwarded(registry) -> None:
    plan = DeterministicPlanner().plan(
        _request(
            "EpiZoo embedding",
            {
                "input_path": "/data/in.h5ad",
                "output_dir": "/output",
                "species": " Human ",
                "checkpoint_path": "/models/model.pth",
                "device": "cuda:0",
                "overwrite": True,
                "unregistered_option": "ignored",
            },
        ),
        registry,
    )

    arguments = plan.steps[1].arguments
    assert arguments["species"] == "human"
    assert arguments["checkpoint_path"] == "/models/model.pth"
    assert arguments["device"] == "cuda:0"
    assert arguments["overwrite"] is True
    assert "unregistered_option" not in arguments


def test_generated_plan_and_arguments_validate(registry) -> None:
    plan = DeterministicPlanner().plan(
        _request(
            "summarize this dataset and make embeddings",
            {
                "input_path": "/data/in.h5ad",
                "output_dir": "/output",
                "species": "mouse",
            },
        ),
        registry,
    )

    assert tuple(step.step_id for step in plan.stable_topological_steps()) == (
        "inspect",
        "embed",
    )
    for step in plan.steps:
        registry.validate_arguments(step.tool_name, step.arguments)


@pytest.mark.parametrize(
    "prompt",
    ["cluster the cells", "tell me about chromatin", "this dataset is inspectable"],
)
def test_unsupported_or_unrelated_prompt_is_rejected(registry, prompt) -> None:
    with pytest.raises(PlannerError) as raised:
        DeterministicPlanner().plan(
            _request(prompt, {"input_path": "/data/in.h5ad"}), registry
        )

    assert raised.value.code == "UNSUPPORTED_REQUEST"


def test_registry_contract_mismatch_is_an_internal_planner_error(registry) -> None:
    embedding_only = ToolRegistry((registry.get("epizoo_embed_cells"),))

    with pytest.raises(PlannerError) as raised:
        DeterministicPlanner().plan(
            _request("inspect", {"input_path": "/data/in.h5ad"}),
            embedding_only,
        )

    assert raised.value.code == "PLANNER_REGISTRY_CONTRACT_MISMATCH"
    assert raised.value.category is ErrorCategory.INTERNAL_AGENT_ERROR


def test_planner_never_executes_registered_tools(registry) -> None:
    inspection_callable = Mock(side_effect=AssertionError("must not execute"))
    embedding_callable = Mock(side_effect=AssertionError("must not execute"))
    guarded_registry = ToolRegistry(
        (
            replace(registry.get("inspect_scATAC"), function=inspection_callable),
            replace(
                registry.get("epizoo_embed_cells"), function=embedding_callable
            ),
        )
    )

    DeterministicPlanner().plan(
        _request(
            "inspect and embed",
            {
                "input_path": "/data/in.h5ad",
                "output_dir": "/output",
                "species": "mouse",
            },
        ),
        guarded_registry,
    )

    inspection_callable.assert_not_called()
    embedding_callable.assert_not_called()


def test_run_mode_does_not_change_scientific_plan(registry) -> None:
    inputs = {
        "input_path": "/data/in.h5ad",
        "output_dir": "/output",
        "species": "mouse",
    }
    planner = DeterministicPlanner()

    execute_plan = planner.plan(
        _request("embedding", inputs, mode=RunMode.EXECUTE), registry
    )
    plan_only = planner.plan(
        _request("embedding", inputs, mode=RunMode.PLAN_ONLY), registry
    )

    assert execute_plan == plan_only


def test_planner_requires_no_provider_or_network_configuration(registry) -> None:
    plan = DeterministicPlanner().plan(
        _request("inspection", {"input_path": "/data/in.h5ad"}), registry
    )

    assert plan.steps[0].tool_name == "inspect_scATAC"
