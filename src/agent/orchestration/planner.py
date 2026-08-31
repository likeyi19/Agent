"""Narrow deterministic planning for the initial orchestration workflows."""

from __future__ import annotations

import re
import math
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
_DOWNSTREAM_INTENT = re.compile(
    r"\b(?:neighbors?|cluster(?:ing)?|umap)\b|\bfull(?:\s+scatac)?\s+analysis\b"
)
_CLUSTERING_EVALUATION_INTENT = re.compile(
    r"\b(?:evaluate|evaluation|benchmark|metrics?)\b.*\bcluster(?:s|ing)?\b"
    r"|\bcluster(?:s|ing)?\b.*\b(?:evaluate|evaluation|benchmark|metrics?)\b"
)
_EMBEDDING_REQUIRED_INPUTS = ("input_path", "output_dir", "species")
_EMBEDDING_OPTIONAL_INPUTS = ("checkpoint_path", "device", "overwrite")
_DOWNSTREAM_OPTIONAL_INPUTS = (
    "n_neighbors",
    "metric",
    "random_seed",
    "resolution",
    "min_dist",
    "spread",
    "overwrite",
)


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


def _downstream_inputs(request: AgentRequest) -> dict[str, object]:
    values: dict[str, object] = {}
    for name in _DOWNSTREAM_OPTIONAL_INPUTS:
        if name not in request.inputs:
            continue
        value = request.inputs[name]
        if name in {"n_neighbors", "random_seed"}:
            if isinstance(value, bool) or not isinstance(value, int):
                raise PlannerError(
                    "INVALID_REQUEST_INPUT",
                    f"Structured input {name!r} must be an integer.",
                )
            if name == "random_seed" and value < 0:
                raise PlannerError(
                    "INVALID_REQUEST_INPUT",
                    "Structured input 'random_seed' must be nonnegative.",
                )
        elif name == "metric":
            if value not in {"euclidean", "cosine"}:
                raise PlannerError(
                    "INVALID_REQUEST_INPUT",
                    "Structured input 'metric' must be 'euclidean' or 'cosine'.",
                )
        elif name == "overwrite":
            if not isinstance(value, bool):
                raise PlannerError(
                    "INVALID_REQUEST_INPUT",
                    "Structured input 'overwrite' must be a boolean.",
                )
        else:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise PlannerError(
                    "INVALID_REQUEST_INPUT",
                    f"Structured input {name!r} must be a finite number.",
                )
        values[name] = value

    effective_resolution = float(values.get("resolution", 1.0))
    effective_min_dist = float(values.get("min_dist", 0.5))
    effective_spread = float(values.get("spread", 1.0))
    if effective_resolution <= 0:
        raise PlannerError(
            "INVALID_REQUEST_INPUT",
            "Structured input 'resolution' must be strictly positive.",
        )
    if effective_min_dist < 0 or effective_spread <= 0:
        raise PlannerError(
            "INVALID_REQUEST_INPUT",
            "Structured UMAP inputs require min_dist >= 0 and spread > 0.",
        )
    if effective_min_dist > effective_spread:
        raise PlannerError(
            "INVALID_REQUEST_INPUT",
            "Structured input 'min_dist' must not exceed 'spread'.",
        )
    return values


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
        downstream_intent = _DOWNSTREAM_INTENT.search(prompt) is not None
        evaluation_intent = _CLUSTERING_EVALUATION_INTENT.search(prompt) is not None

        if evaluation_intent:
            _require_inputs(
                request, ("input_path", "output_dir", "species", "label_key")
            )
            input_path = _path_input(request, "input_path")
            output_dir = _path_input(request, "output_dir")
            label_key = request.inputs["label_key"]
            if not isinstance(label_key, str) or not label_key.strip():
                raise PlannerError(
                    "INVALID_REQUEST_INPUT",
                    "Structured input 'label_key' must be a non-empty string.",
                )
            embedding_arguments = _embedding_inputs(request)
            embedding_arguments["input_path"] = StepOutputRef("inspect", "input_path")
            downstream_inputs = _downstream_inputs(request)
            neighbors_arguments: dict[str, object] = {
                "embedding_path": StepOutputRef("embed", "embedding_path"),
                "cell_ids_path": StepOutputRef("embed", "cell_ids_path"),
                "output_dir": output_dir,
            }
            clustering_arguments: dict[str, object] = {
                "analysis_path": StepOutputRef("neighbors", "analysis_path"),
                "output_dir": output_dir,
            }
            evaluation_arguments: dict[str, object] = {
                "analysis_path": StepOutputRef("cluster", "analysis_path"),
                "reference_h5ad_path": StepOutputRef("inspect", "input_path"),
                "label_key": label_key,
                "output_dir": output_dir,
            }
            for name in ("n_neighbors", "metric", "random_seed", "overwrite"):
                if name in downstream_inputs:
                    neighbors_arguments[name] = downstream_inputs[name]
            for name in ("resolution", "random_seed", "overwrite"):
                if name in downstream_inputs:
                    clustering_arguments[name] = downstream_inputs[name]
            if "overwrite" in request.inputs:
                evaluation_arguments["overwrite"] = request.inputs["overwrite"]
            if "cluster_key" in request.inputs:
                cluster_key = request.inputs["cluster_key"]
                if not isinstance(cluster_key, str) or not cluster_key.strip():
                    raise PlannerError(
                        "INVALID_REQUEST_INPUT",
                        "Structured input 'cluster_key' must be a non-empty string.",
                    )
                evaluation_arguments["cluster_key"] = cluster_key
            steps = (
                PlanStep("inspect", "inspect_scATAC", {"path": input_path}, description="Inspect the input scATAC dataset."),
                PlanStep("embed", "epizoo_embed_cells", embedding_arguments, ("inspect",), "Compute and persist EpiZoo cell embeddings."),
                PlanStep("neighbors", "build_cell_neighbors", neighbors_arguments, ("embed",), "Build and persist a sparse cell-neighbor graph."),
                PlanStep("cluster", "cluster_cells", clustering_arguments, ("neighbors",), "Cluster cells with fixed-setting Leiden."),
                PlanStep(
                    "evaluate",
                    "evaluate_cell_clustering",
                    evaluation_arguments,
                    ("cluster", "inspect"),
                    "Evaluate the fixed clustering against reference annotations.",
                ),
            )
            workflow = "epizoo-clustering-evaluation"
        elif downstream_intent:
            _require_inputs(request, ("input_path",))
            input_path = _path_input(request, "input_path")
            embedding_arguments = _embedding_inputs(request)
            embedding_arguments["input_path"] = StepOutputRef(
                step_id="inspect", output_key="input_path"
            )
            downstream_inputs = _downstream_inputs(request)
            output_dir = _path_input(request, "output_dir")
            neighbors_arguments: dict[str, object] = {
                "embedding_path": StepOutputRef("embed", "embedding_path"),
                "cell_ids_path": StepOutputRef("embed", "cell_ids_path"),
                "output_dir": output_dir,
            }
            clustering_arguments: dict[str, object] = {
                "analysis_path": StepOutputRef("neighbors", "analysis_path"),
                "output_dir": output_dir,
            }
            umap_arguments: dict[str, object] = {
                "analysis_path": StepOutputRef("cluster", "analysis_path"),
                "output_dir": output_dir,
            }
            for name in ("n_neighbors", "metric", "random_seed", "overwrite"):
                if name in downstream_inputs:
                    neighbors_arguments[name] = downstream_inputs[name]
            for name in ("resolution", "random_seed", "overwrite"):
                if name in downstream_inputs:
                    clustering_arguments[name] = downstream_inputs[name]
            for name in ("min_dist", "spread", "random_seed", "overwrite"):
                if name in downstream_inputs:
                    umap_arguments[name] = downstream_inputs[name]
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
                PlanStep(
                    step_id="neighbors",
                    tool_name="build_cell_neighbors",
                    arguments=neighbors_arguments,
                    depends_on=("embed",),
                    description="Build and persist a sparse cell-neighbor graph.",
                ),
                PlanStep(
                    step_id="cluster",
                    tool_name="cluster_cells",
                    arguments=clustering_arguments,
                    depends_on=("neighbors",),
                    description="Cluster cells with fixed-setting Leiden.",
                ),
                PlanStep(
                    step_id="umap",
                    tool_name="compute_cell_umap",
                    arguments=umap_arguments,
                    depends_on=("cluster",),
                    description="Compute and persist a two-dimensional UMAP.",
                ),
            )
            workflow = "epizoo-downstream-analysis"
        elif embedding_intent:
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
                "or EpiZoo embedding/downstream-analysis requests.",
            )

        _validate_generated_steps(registry, steps)
        return AgentPlan(
            plan_id=f"{request.request_id}:{workflow}",
            request_id=request.request_id,
            planner_name=self.name,
            steps=steps,
        )


__all__ = ["DeterministicPlanner", "Planner", "PlannerError"]
