"""Narrow deterministic planning for the initial orchestration workflows."""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from agent.schemas import (
    AgentPlan,
    AgentRequest,
    ErrorCategory,
    PlanStep,
    StepOutputRef,
)

from .registry import ToolArgumentError, ToolRegistry, UnknownToolError


@runtime_checkable
class Planner(Protocol):
    """Interface for components that produce plans without executing tools."""

    def plan(self, request: AgentRequest, registry: ToolRegistry) -> AgentPlan:
        """Produce a structurally valid plan using the registry vocabulary."""


class PlannerError(ValueError):
    """Stable planner failure suitable for future runtime classification."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        category: ErrorCategory = ErrorCategory.USER_INPUT_ERROR,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.category = category


_EMBEDDING_INTENT = re.compile(r"\b(?:embed|embedding|embeddings)\b")
_INSPECTION_INTENT = re.compile(
    r"\b(?:inspect|inspection)\b|\bsummarize(?:\s+this)?\s+dataset\b"
)
_EMBEDDING_REQUIRED_INPUTS = ("input_path", "output_dir", "species")
_EMBEDDING_OPTIONAL_INPUTS = ("checkpoint_path", "device", "overwrite")


def _normalized_prompt(prompt: str) -> str:
    return " ".join(prompt.casefold().split())


def _require_inputs(request: AgentRequest, names: tuple[str, ...]) -> None:
    missing = [name for name in names if name not in request.inputs]
    if missing:
        raise PlannerError(
            "MISSING_REQUIRED_INPUT",
            f"Request is missing required structured inputs: {missing}.",
        )


def _path_input(request: AgentRequest, name: str) -> str:
    value = request.inputs[name]
    if not isinstance(value, str) or not value.strip():
        raise PlannerError(
            "INVALID_REQUEST_INPUT",
            f"Structured input {name!r} must be a non-empty path string.",
        )
    return value


def _embedding_inputs(request: AgentRequest) -> dict[str, object]:
    _require_inputs(request, _EMBEDDING_REQUIRED_INPUTS)
    arguments: dict[str, object] = {
        "output_dir": _path_input(request, "output_dir"),
    }

    species = request.inputs["species"]
    if not isinstance(species, str) or species.strip().casefold() not in {
        "human",
        "mouse",
    }:
        raise PlannerError(
            "INVALID_REQUEST_INPUT",
            "Structured input 'species' must be 'human' or 'mouse'.",
        )
    arguments["species"] = species.strip().casefold()

    for name in _EMBEDDING_OPTIONAL_INPUTS:
        if name not in request.inputs:
            continue
        value = request.inputs[name]
        if name in {"checkpoint_path", "device"}:
            if not isinstance(value, str) or not value.strip():
                raise PlannerError(
                    "INVALID_REQUEST_INPUT",
                    f"Structured input {name!r} must be a non-empty string.",
                )
        elif not isinstance(value, bool):
            raise PlannerError(
                "INVALID_REQUEST_INPUT",
                "Structured input 'overwrite' must be a boolean.",
            )
        arguments[name] = value
    return arguments


def _validate_generated_steps(
    registry: ToolRegistry, steps: tuple[PlanStep, ...]
) -> None:
    try:
        for step in steps:
            registry.validate_arguments(step.tool_name, step.arguments)
    except (UnknownToolError, ToolArgumentError) as exc:
        raise PlannerError(
            "PLANNER_REGISTRY_CONTRACT_MISMATCH",
            f"Deterministic planner generated a registry-invalid step: {exc}",
            category=ErrorCategory.INTERNAL_AGENT_ERROR,
        ) from exc


class DeterministicPlanner:
    """Bootstrap planner for explicit inspection and EpiZoo workflows."""

    name = "deterministic"

    def plan(self, request: AgentRequest, registry: ToolRegistry) -> AgentPlan:
        if not isinstance(request, AgentRequest):
            raise TypeError("`request` must be an AgentRequest.")
        if not isinstance(registry, ToolRegistry):
            raise TypeError("`registry` must be a ToolRegistry.")

        prompt = _normalized_prompt(request.prompt)
        embedding_intent = _EMBEDDING_INTENT.search(prompt) is not None
        inspection_intent = _INSPECTION_INTENT.search(prompt) is not None

        if embedding_intent:
            _require_inputs(request, ("input_path",))
            input_path = _path_input(request, "input_path")
            embedding_arguments = _embedding_inputs(request)
            embedding_arguments["input_path"] = StepOutputRef(
                step_id="inspect", output_key="input_path"
            )
            steps = (
                PlanStep(
                    step_id="inspect",
                    tool_name="inspect_scATAC",
                    arguments={"path": input_path},
                    description="Inspect the input scATAC dataset.",
                ),
                PlanStep(
                    step_id="embed",
                    tool_name="epizoo_embed_cells",
                    arguments=embedding_arguments,
                    depends_on=("inspect",),
                    description="Compute and persist EpiZoo cell embeddings.",
                ),
            )
            workflow = "epizoo-embedding"
        elif inspection_intent:
            _require_inputs(request, ("input_path",))
            steps = (
                PlanStep(
                    step_id="inspect",
                    tool_name="inspect_scATAC",
                    arguments={"path": _path_input(request, "input_path")},
                    description="Inspect the input scATAC dataset.",
                ),
            )
            workflow = "inspection"
        else:
            raise PlannerError(
                "UNSUPPORTED_REQUEST",
                "DeterministicPlanner supports only explicit scATAC inspection "
                "or EpiZoo embedding requests.",
            )

        _validate_generated_steps(registry, steps)
        return AgentPlan(
            plan_id=f"{request.request_id}:{workflow}",
            request_id=request.request_id,
            planner_name=self.name,
            steps=steps,
        )


__all__ = ["DeterministicPlanner", "Planner", "PlannerError"]
