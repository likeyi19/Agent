"""Durability, recovery, and cancellation checks for Milestone 6.1 tools."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from agent.orchestration import (
    AgentPlan,
    AgentRequest,
    AgentRuntime,
    FileRunStore,
    PlanStep,
    RecoveryPolicyIncompatibleError,
    RunLifecycleStatus,
    RunStatus,
    StepOutputRef,
    StepStatus,
    ToolRegistry,
    build_default_tool_registry,
)
from agent.tools import build_cell_neighbors


class SimulatedProcessExit(BaseException):
    pass


class FixedPlanner:
    def __init__(self, plan: AgentPlan) -> None:
        self.plan_value = plan
        self.calls = 0

    def plan(self, request, registry):
        self.calls += 1
        return self.plan_value


class InterruptingStore:
    def __init__(self, delegate: FileRunStore) -> None:
        self.delegate = delegate
        self.triggered = False

    def create(self, state):
        return self.delegate.create(state)

    def execution_lease(self, run_id):
        return self.delegate.execution_lease(run_id)

    def load(self, run_id):
        return self.delegate.load(run_id)

    def request_cancellation(self, run_id):
        return self.delegate.request_cancellation(run_id)

    def load_cancellation(self, run_id):
        return self.delegate.load_cancellation(run_id)

    def update(self, state, *, expected_revision):
        saved = self.delegate.update(state, expected_revision=expected_revision)
        if (
            not self.triggered
            and saved.lifecycle_status is RunLifecycleStatus.RUNNING
            and len(saved.steps) == 3
            and saved.steps[0].status is StepStatus.SUCCEEDED
            and saved.steps[1].status is StepStatus.PENDING
        ):
            self.triggered = True
            raise SimulatedProcessExit()
        return saved


def _embedding_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(31)
    embedding_path = tmp_path / "embedding.npy"
    cell_ids_path = tmp_path / "ids.txt"
    np.save(
        embedding_path,
        rng.normal(size=(32, 512)).astype(np.float32),
        allow_pickle=False,
    )
    cell_ids_path.write_text(
        "".join(f"cell-{index}\n" for index in range(32)), encoding="utf-8"
    )
    return embedding_path, cell_ids_path


def _plan(tmp_path: Path, embedding_path: Path, cell_ids_path: Path) -> AgentPlan:
    output_dir = tmp_path / "outputs"
    return AgentPlan(
        "plan-1",
        "request-1",
        "fixed",
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


def _interrupted_after_neighbors(tmp_path: Path):
    embedding_path, cell_ids_path = _embedding_artifacts(tmp_path)
    plan = _plan(tmp_path, embedding_path, cell_ids_path)
    default = build_default_tool_registry()
    neighbor_call = Mock(wraps=default.get("build_cell_neighbors").function)
    registry = ToolRegistry(
        tuple(
            replace(default.get(name), function=neighbor_call)
            if name == "build_cell_neighbors"
            else default.get(name)
            for name in default.names()
        )
    )
    store = FileRunStore(tmp_path / "store")
    interrupting = InterruptingStore(store)
    with pytest.raises(SimulatedProcessExit):
        AgentRuntime(
            planner=FixedPlanner(plan), registry=registry, run_store=interrupting
        ).run(AgentRequest("request-1", "fixed", {}))
    return plan, registry, store, neighbor_call


def test_completed_neighbors_is_reused_and_analysis_reference_restored_on_resume(
    tmp_path: Path,
) -> None:
    plan, registry, store, neighbor_call = _interrupted_after_neighbors(tmp_path)
    resume_planner = FixedPlanner(plan)
    result = AgentRuntime(
        planner=resume_planner, registry=registry, run_store=store
    ).resume("request-1:run")
    assert result.status is RunStatus.SUCCEEDED
    assert neighbor_call.call_count == 1
    assert resume_planner.calls == 0
    assert result.steps[1].resolved_arguments["analysis_path"] == (
        result.steps[0].result["analysis_path"]  # type: ignore[index]
    )
    assert result.steps[2].resolved_arguments["analysis_path"] == (
        result.steps[1].result["analysis_path"]  # type: ignore[index]
    )


def test_missing_neighbors_checkpoint_is_rejected_before_downstream_resume(
    tmp_path: Path,
) -> None:
    plan, registry, store, neighbor_call = _interrupted_after_neighbors(tmp_path)
    state = store.load("request-1:run")
    neighbor_path = Path(state.steps[0].result["analysis_path"])  # type: ignore[index]
    neighbor_path.unlink()
    default = registry
    cluster_call = Mock(side_effect=AssertionError("corrupt resume invoked clustering"))
    guarded = ToolRegistry(
        tuple(
            replace(default.get(name), function=cluster_call)
            if name == "cluster_cells"
            else default.get(name)
            for name in default.names()
        )
    )
    result = AgentRuntime(
        planner=FixedPlanner(plan), registry=guarded, run_store=store
    ).resume("request-1:run")
    assert result.status is RunStatus.FAILED
    assert any(
        error.code == "PERSISTED_STEP_REVALIDATION_FAILED"
        for error in result.errors
    )
    assert neighbor_call.call_count == 1
    cluster_call.assert_not_called()


def test_new_tool_policy_snapshot_and_version_drift_are_authoritative(
    tmp_path: Path,
) -> None:
    plan, registry, store, _ = _interrupted_after_neighbors(tmp_path)
    state = store.load("request-1:run")
    assert state.recovery_policy_snapshot is not None
    assert tuple(
        (tool.tool_name, tool.classifier_version, tool.retryable_error_codes)
        for tool in state.recovery_policy_snapshot.tools
    ) == (
        ("build_cell_neighbors", "build-cell-neighbors-v1", ()),
        ("cluster_cells", "cluster-cells-v1", ()),
        ("compute_cell_umap", "compute-cell-umap-v1", ()),
    )
    guarded = Mock(side_effect=AssertionError("policy drift invoked a tool"))
    incompatible = ToolRegistry(
        tuple(
            replace(
                registry.get(name),
                function=guarded,
                recovery_policy_version="build-cell-neighbors-v2",
            )
            if name == "build_cell_neighbors"
            else replace(registry.get(name), function=guarded)
            for name in registry.names()
        )
    )
    with pytest.raises(RecoveryPolicyIncompatibleError):
        AgentRuntime(
            planner=FixedPlanner(plan), registry=incompatible, run_store=store
        ).resume("request-1:run")
    guarded.assert_not_called()


def test_cancellation_after_neighbors_prevents_next_downstream_stage(
    tmp_path: Path,
) -> None:
    embedding_path, cell_ids_path = _embedding_artifacts(tmp_path)
    plan = _plan(tmp_path, embedding_path, cell_ids_path)
    prepared = build_cell_neighbors(
        embedding_path, cell_ids_path, tmp_path / "prepared"
    )
    store = FileRunStore(tmp_path / "store")

    def cancel_after_neighbor(**kwargs):
        store.request_cancellation("request-1:run")
        return prepared

    default = build_default_tool_registry()
    cluster_call = Mock(side_effect=AssertionError("cancelled run invoked clustering"))
    umap_call = Mock(side_effect=AssertionError("cancelled run invoked UMAP"))
    registry = ToolRegistry(
        tuple(
            replace(default.get(name), function=cancel_after_neighbor)
            if name == "build_cell_neighbors"
            else replace(default.get(name), function=cluster_call)
            if name == "cluster_cells"
            else replace(default.get(name), function=umap_call)
            if name == "compute_cell_umap"
            else default.get(name)
            for name in default.names()
        )
    )
    result = AgentRuntime(
        planner=FixedPlanner(plan), registry=registry, run_store=store
    ).run(AgentRequest("request-1", "fixed", {}))
    assert result.status is RunStatus.CANCELLED
    assert tuple(step.status for step in result.steps) == (
        StepStatus.SUCCEEDED,
        StepStatus.SKIPPED,
        StepStatus.SKIPPED,
    )
    cluster_call.assert_not_called()
    umap_call.assert_not_called()
