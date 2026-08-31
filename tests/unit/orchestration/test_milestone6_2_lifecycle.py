"""Durability, recovery, and cancellation checks for Milestone 6.2."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

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
from agent.tools import build_cell_neighbors, cluster_cells
from agent.tools.analysis.embedding_analysis import PROVENANCE_KEY, _cell_order_digest


class SimulatedProcessExit(BaseException):
    pass


class FixedPlanner:
    def __init__(self, plan: AgentPlan) -> None:
        self.plan_value = plan
        self.calls = 0

    def plan(self, request, registry):
        self.calls += 1
        return self.plan_value


class InterruptAfterEvaluationStore:
    def __init__(self, delegate: FileRunStore) -> None:
        self.delegate = delegate
        self.triggered = False

    def __getattr__(self, name: str):
        return getattr(self.delegate, name)

    def update(self, state, *, expected_revision):
        saved = self.delegate.update(state, expected_revision=expected_revision)
        if (
            not self.triggered
            and saved.lifecycle_status is RunLifecycleStatus.RUNNING
            and len(saved.steps) == 2
            and saved.steps[0].status is StepStatus.SUCCEEDED
            and saved.steps[1].status is StepStatus.PENDING
        ):
            self.triggered = True
            raise SimulatedProcessExit()
        return saved


def _evaluation_sources(tmp_path: Path) -> tuple[Path, Path]:
    ids = tuple(f"cell-{index}" for index in range(8))
    labels = ("a", "a", "b", "b", "c", "c", "d", "d")
    reference = tmp_path / "reference.h5ad"
    ad.AnnData(
        X=sparse.csr_matrix((8, 10), dtype=np.float32),
        obs=pd.DataFrame({"celltype": labels}, index=pd.Index(ids)),
    ).write_h5ad(reference)
    artifact = ad.AnnData(obs=pd.DataFrame({"leiden": labels}, index=pd.Index(ids)))
    artifact.obsm["X_epizoo"] = np.zeros((8, 512), dtype=np.float32)
    graph = sparse.eye(8, format="csr", dtype=np.float32)
    artifact.obsp["distances"] = graph
    artifact.obsp["connectivities"] = graph.copy()
    artifact.uns["neighbors"] = {
        "distances_key": "distances",
        "connectivities_key": "connectivities",
    }
    artifact.uns[PROVENANCE_KEY] = {
        "schema_version": 1,
        "stage": "clustering",
        "cell_order_sha256": _cell_order_digest(ids),
        "source_analysis_path": str((tmp_path / "upstream.h5ad").resolve()),
        "parameters": {
            "neighbors": {
                "n_neighbors": 2,
                "metric": "euclidean",
                "method": "umap",
                "transformer": "none",
                "random_seed": 0,
                "use_rep": "X_epizoo",
            },
            "clustering": {
                "algorithm": "leiden",
                "resolution": 1.0,
                "flavor": "igraph",
                "n_iterations": 2,
                "directed": False,
                "use_weights": True,
                "random_seed": 0,
                "key_added": "leiden",
            }
        },
        "software_versions": {},
    }
    analysis = tmp_path / "clustered.h5ad"
    artifact.write_h5ad(analysis)
    return analysis, reference


def _evaluation_plan(tmp_path: Path, analysis: Path, reference: Path) -> AgentPlan:
    return AgentPlan(
        "plan-evaluation",
        "request-1",
        "fixed",
        (
            PlanStep(
                "evaluate",
                "evaluate_cell_clustering",
                {
                    "analysis_path": str(analysis),
                    "reference_h5ad_path": str(reference),
                    "label_key": "celltype",
                    "output_dir": str(tmp_path / "output"),
                },
            ),
            PlanStep(
                "inspect",
                "inspect_scATAC",
                {"path": str(reference)},
                ("evaluate",),
            ),
        ),
    )


def _interrupted_after_evaluation(tmp_path: Path):
    analysis, reference = _evaluation_sources(tmp_path)
    plan = _evaluation_plan(tmp_path, analysis, reference)
    default = build_default_tool_registry()
    evaluation_call = Mock(wraps=default.get("evaluate_cell_clustering").function)
    registry = ToolRegistry(
        tuple(
            replace(default.get(name), function=evaluation_call)
            if name == "evaluate_cell_clustering"
            else default.get(name)
            for name in default.names()
        )
    )
    store = FileRunStore(tmp_path / "store")
    with pytest.raises(SimulatedProcessExit):
        AgentRuntime(
            planner=FixedPlanner(plan),
            registry=registry,
            run_store=InterruptAfterEvaluationStore(store),
        ).run(AgentRequest("request-1", "fixed", {}))
    return plan, registry, store, evaluation_call, analysis


def test_completed_evaluation_is_revalidated_and_reused_on_resume(tmp_path: Path) -> None:
    plan, registry, store, evaluation_call, _ = _interrupted_after_evaluation(tmp_path)
    planner = FixedPlanner(plan)
    result = AgentRuntime(planner=planner, registry=registry, run_store=store).resume(
        "request-1:run"
    )
    assert result.status is RunStatus.SUCCEEDED
    assert evaluation_call.call_count == 1
    assert planner.calls == 0


@pytest.mark.parametrize("missing", ["report", "analysis"])
def test_missing_evaluation_evidence_blocks_resume_before_downstream_execution(
    tmp_path: Path, missing: str
) -> None:
    plan, registry, store, evaluation_call, analysis = _interrupted_after_evaluation(tmp_path)
    state = store.load("request-1:run")
    if missing == "report":
        Path(state.steps[0].result["report_path"]).unlink()  # type: ignore[index]
    else:
        analysis.unlink()
    inspect_call = Mock(side_effect=AssertionError("invalid resume invoked downstream"))
    guarded = ToolRegistry(
        tuple(
            replace(registry.get(name), function=inspect_call)
            if name == "inspect_scATAC"
            else registry.get(name)
            for name in registry.names()
        )
    )
    result = AgentRuntime(
        planner=FixedPlanner(plan), registry=guarded, run_store=store
    ).resume("request-1:run")
    assert result.status is RunStatus.FAILED
    assert any(error.code == "PERSISTED_STEP_REVALIDATION_FAILED" for error in result.errors)
    evaluation_call.assert_called_once()
    inspect_call.assert_not_called()


def test_evaluation_recovery_identity_and_drift_are_authoritative(tmp_path: Path) -> None:
    plan, registry, store, evaluation_call, _ = _interrupted_after_evaluation(tmp_path)
    state = store.load("request-1:run")
    assert state.recovery_policy_snapshot is not None
    identities = {
        tool.tool_name: tool.classifier_version
        for tool in state.recovery_policy_snapshot.tools
    }
    assert identities["evaluate_cell_clustering"] == "evaluate-cell-clustering-v1"
    guarded_call = Mock(side_effect=AssertionError("policy drift invoked a tool"))
    incompatible = ToolRegistry(
        tuple(
            replace(
                registry.get(name),
                function=guarded_call,
                recovery_policy_version="evaluate-cell-clustering-v2",
            )
            if name == "evaluate_cell_clustering"
            else replace(registry.get(name), function=guarded_call)
            for name in registry.names()
        )
    )
    with pytest.raises(RecoveryPolicyIncompatibleError):
        AgentRuntime(
            planner=FixedPlanner(plan), registry=incompatible, run_store=store
        ).resume("request-1:run")
    assert evaluation_call.call_count == 1
    guarded_call.assert_not_called()


def test_cancellation_after_clustering_prevents_evaluation(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    embeddings = tmp_path / "embedding.npy"
    ids_path = tmp_path / "ids.txt"
    ids = tuple(f"cell-{index}" for index in range(20))
    np.save(embeddings, rng.normal(size=(20, 512)).astype(np.float32))
    ids_path.write_text("".join(f"{value}\n" for value in ids), encoding="utf-8")
    neighbors = build_cell_neighbors(embeddings, ids_path, tmp_path / "prepared")
    clustered = cluster_cells(neighbors["analysis_path"], tmp_path / "prepared")
    reference = tmp_path / "reference.h5ad"
    ad.AnnData(
        X=sparse.csr_matrix((20, 4), dtype=np.float32),
        obs=pd.DataFrame(
            {"celltype": ["a"] * 10 + ["b"] * 10}, index=pd.Index(ids)
        ),
    ).write_h5ad(reference)
    plan = AgentPlan(
        "cancel-plan",
        "request-1",
        "fixed",
        (
            PlanStep(
                "cluster",
                "cluster_cells",
                {
                    "analysis_path": neighbors["analysis_path"],
                    "output_dir": str(tmp_path / "runtime"),
                },
            ),
            PlanStep(
                "evaluate",
                "evaluate_cell_clustering",
                {
                    "analysis_path": StepOutputRef("cluster", "analysis_path"),
                    "reference_h5ad_path": str(reference),
                    "label_key": "celltype",
                    "output_dir": str(tmp_path / "runtime"),
                },
                ("cluster",),
            ),
        ),
    )
    store = FileRunStore(tmp_path / "store")

    def cancel_after_cluster(**kwargs):
        store.request_cancellation("request-1:run")
        return clustered

    default = build_default_tool_registry()
    evaluation_call = Mock(side_effect=AssertionError("cancelled run invoked evaluation"))
    registry = ToolRegistry(
        tuple(
            replace(default.get(name), function=cancel_after_cluster)
            if name == "cluster_cells"
            else replace(default.get(name), function=evaluation_call)
            if name == "evaluate_cell_clustering"
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
    )
    evaluation_call.assert_not_called()


def test_terminal_resume_keeps_existing_idempotent_semantics(tmp_path: Path) -> None:
    analysis, reference = _evaluation_sources(tmp_path)
    plan = AgentPlan(
        "terminal-plan",
        "request-1",
        "fixed",
        (_evaluation_plan(tmp_path, analysis, reference).steps[0],),
    )
    store = FileRunStore(tmp_path / "store")
    runtime = AgentRuntime(
        planner=FixedPlanner(plan), registry=build_default_tool_registry(), run_store=store
    )
    result = runtime.run(AgentRequest("request-1", "fixed", {}))
    assert result.status is RunStatus.SUCCEEDED
    Path(result.steps[0].result["report_path"]).unlink()  # type: ignore[index]
    assert runtime.resume(result.run_id) == result
