"""Offline real-tool acceptance for Milestone 6.1 downstream analysis."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import Mock

import anndata as ad
import numpy as np
from scipy import sparse

from agent.orchestration import (
    AgentPlan,
    AgentRequest,
    AgentRuntime,
    DeterministicPlanner,
    PlanExecutor,
    PlanStep,
    RunMode,
    RunStatus,
    StepOutputRef,
    ToolRegistry,
    build_default_tool_registry,
)


def _embedding_artifacts(tmp_path: Path) -> tuple[Path, Path, tuple[str, ...]]:
    rng = np.random.default_rng(23)
    embeddings = np.vstack(
        (
            rng.normal(-1.5, 0.2, size=(16, 512)),
            rng.normal(1.5, 0.2, size=(16, 512)),
        )
    ).astype(np.float32)
    cell_ids = tuple(f"cell-{index:02d}" for index in range(32))
    embedding_path = tmp_path / "synthetic.epizoo_embeddings.npy"
    cell_ids_path = tmp_path / "synthetic.epizoo_obs_names.txt"
    np.save(embedding_path, embeddings, allow_pickle=False)
    cell_ids_path.write_text(
        "".join(f"{cell_id}\n" for cell_id in cell_ids), encoding="utf-8"
    )
    return embedding_path, cell_ids_path, cell_ids


def test_real_downstream_tools_execute_through_registry_and_references(
    tmp_path: Path,
) -> None:
    embedding_path, cell_ids_path, cell_ids = _embedding_artifacts(tmp_path)
    output_dir = tmp_path / "outputs"
    plan = AgentPlan(
        "milestone6-real",
        "request-1",
        "integration",
        (
            PlanStep(
                "neighbors",
                "build_cell_neighbors",
                {
                    "embedding_path": str(embedding_path),
                    "cell_ids_path": str(cell_ids_path),
                    "output_dir": str(output_dir),
                },
            ),
            PlanStep(
                "cluster",
                "cluster_cells",
                {
                    "analysis_path": StepOutputRef("neighbors", "analysis_path"),
                    "output_dir": str(output_dir),
                },
                ("neighbors",),
            ),
            PlanStep(
                "umap",
                "compute_cell_umap",
                {
                    "analysis_path": StepOutputRef("cluster", "analysis_path"),
                    "output_dir": str(output_dir),
                },
                ("cluster",),
            ),
        ),
    )
    executor = PlanExecutor(build_default_tool_registry())
    assert executor.preflight(plan).passed
    outcome = executor.execute(plan)

    assert not outcome.errors
    assert tuple(result.status.value for result in outcome.step_results) == (
        "SUCCEEDED",
        "SUCCEEDED",
        "SUCCEEDED",
    )
    final_result = outcome.step_results[-1].result
    assert final_result is not None
    json.dumps(outcome.step_results[-1].to_dict(), allow_nan=False)
    assert "coordinates" not in final_result

    artifact = ad.read_h5ad(final_result["analysis_path"])
    assert artifact.shape == (32, 0)
    assert tuple(artifact.obs_names) == cell_ids
    assert artifact.obsm["X_epizoo"].shape == (32, 512)
    assert artifact.obsm["X_umap"].shape == (32, 2)
    assert np.isfinite(artifact.obsm["X_umap"]).all()
    assert artifact.obs["leiden"].notna().all()
    assert sparse.issparse(artifact.obsp["distances"])
    assert sparse.issparse(artifact.obsp["connectivities"])


def test_full_five_step_plan_only_preflights_and_invokes_zero_tools(
    tmp_path: Path,
) -> None:
    default = build_default_tool_registry()
    guarded: list[Mock] = []
    specs = []
    for name in default.names():
        function = Mock(side_effect=AssertionError(f"PLAN_ONLY invoked {name}"))
        guarded.append(function)
        specs.append(replace(default.get(name), function=function))
    registry = ToolRegistry(tuple(specs))
    request = AgentRequest(
        "request-1",
        "Build neighbors, cluster cells, and compute UMAP",
        {
            "input_path": str(tmp_path / "not-opened.h5ad"),
            "output_dir": str(tmp_path / "not-created"),
            "species": "mouse",
        },
        RunMode.PLAN_ONLY,
    )
    result = AgentRuntime(
        planner=DeterministicPlanner(), registry=registry
    ).run(request)
    assert result.status is RunStatus.PLANNED
    assert result.planning_only is True
    assert result.verification is not None and result.verification.passed
    assert tuple(step.tool_name for step in result.plan.steps) == default.names()  # type: ignore[union-attr]
    assert not (tmp_path / "not-created").exists()
    for function in guarded:
        function.assert_not_called()


def test_invalid_later_downstream_step_prevents_earlier_execution(tmp_path: Path) -> None:
    default = build_default_tool_registry()
    neighbor_call = Mock(side_effect=AssertionError("preflight invoked neighbors"))
    registry = ToolRegistry(
        (
            replace(default.get("build_cell_neighbors"), function=neighbor_call),
            default.get("cluster_cells"),
        )
    )
    plan = AgentPlan(
        "invalid-later",
        "request-1",
        "integration",
        (
            PlanStep(
                "neighbors",
                "build_cell_neighbors",
                {
                    "embedding_path": str(tmp_path / "embedding.npy"),
                    "cell_ids_path": str(tmp_path / "ids.txt"),
                    "output_dir": str(tmp_path),
                },
            ),
            PlanStep(
                "cluster",
                "cluster_cells",
                {
                    "analysis_path": StepOutputRef("neighbors", "analysis_path"),
                    "output_dir": str(tmp_path),
                    "resolution": "invented",
                },
                ("neighbors",),
            ),
        ),
    )
    outcome = PlanExecutor(registry).execute(plan)
    assert outcome.errors[0].code == "INVALID_TOOL_ARGUMENTS"
    neighbor_call.assert_not_called()
