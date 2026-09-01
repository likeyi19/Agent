"""Durability, recovery, and cancellation checks for Milestone 6.3."""

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
    StepStatus,
    ToolRegistry,
    build_default_tool_registry,
)
from agent.tools import inspect_scATAC


class SimulatedProcessExit(BaseException):
    pass


class FixedPlanner:
    def __init__(self, plan: AgentPlan) -> None:
        self.plan_value = plan
        self.calls = 0

    def plan(self, request, registry):
        self.calls += 1
        return self.plan_value


class InterruptAfterTransferStore:
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


def _write_ids(path: Path, values: tuple[str, ...]) -> None:
    path.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")


def _transfer_sources(tmp_path: Path) -> dict[str, object]:
    reference_ids = tuple(f"reference-{index}" for index in range(4))
    query_ids = tuple(f"query-{index}" for index in range(2))
    reference = np.zeros((4, 512), dtype=np.float32)
    reference[:, 0] = (0.0, 0.2, 9.8, 10.0)
    query = np.zeros((2, 512), dtype=np.float32)
    query[:, 0] = (0.1, 9.9)
    reference_embedding = tmp_path / "reference.npy"
    query_embedding = tmp_path / "query.npy"
    np.save(reference_embedding, reference, allow_pickle=False)
    np.save(query_embedding, query, allow_pickle=False)
    reference_ids_path = tmp_path / "reference.txt"
    query_ids_path = tmp_path / "query.txt"
    _write_ids(reference_ids_path, reference_ids)
    _write_ids(query_ids_path, query_ids)
    reference_h5ad = tmp_path / "reference.h5ad"
    query_h5ad = tmp_path / "query.h5ad"
    ad.AnnData(
        X=sparse.csr_matrix((4, 3), dtype=np.float32),
        obs=pd.DataFrame(
            {"celltype": pd.Categorical(["T", "T", "B", "B"])},
            index=pd.Index(reference_ids),
        ),
    ).write_h5ad(reference_h5ad)
    ad.AnnData(
        X=sparse.csr_matrix((2, 3), dtype=np.float32),
        obs=pd.DataFrame(index=pd.Index(query_ids)),
    ).write_h5ad(query_h5ad)
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"test checkpoint provenance")
    return {
        "reference_embedding_path": str(reference_embedding),
        "reference_cell_ids_path": str(reference_ids_path),
        "reference_h5ad_path": str(reference_h5ad),
        "reference_label_key": "celltype",
        "query_embedding_path": str(query_embedding),
        "query_cell_ids_path": str(query_ids_path),
        "query_h5ad_path": str(query_h5ad),
        "output_dir": str(tmp_path / "output"),
        "reference_species": "mouse",
        "query_species": "mouse",
        "reference_checkpoint_path": str(checkpoint),
        "query_checkpoint_path": str(checkpoint),
        "n_neighbors": 2,
    }


def _transfer_plan(arguments: dict[str, object]) -> AgentPlan:
    return AgentPlan(
        "plan-transfer",
        "request-1",
        "fixed",
        (
            PlanStep("transfer", "transfer_cell_labels", arguments),
            PlanStep(
                "inspect",
                "inspect_scATAC",
                {"path": arguments["query_h5ad_path"]},
                ("transfer",),
            ),
        ),
    )


def _interrupted_after_transfer(tmp_path: Path):
    arguments = _transfer_sources(tmp_path)
    plan = _transfer_plan(arguments)
    default = build_default_tool_registry()
    transfer_call = Mock(wraps=default.get("transfer_cell_labels").function)
    registry = ToolRegistry(
        tuple(
            replace(default.get(name), function=transfer_call)
            if name == "transfer_cell_labels"
            else default.get(name)
            for name in default.names()
        )
    )
    store = FileRunStore(tmp_path / "store")
    with pytest.raises(SimulatedProcessExit):
        AgentRuntime(
            planner=FixedPlanner(plan),
            registry=registry,
            run_store=InterruptAfterTransferStore(store),
        ).run(AgentRequest("request-1", "fixed", {}))
    return arguments, plan, registry, store, transfer_call


def test_completed_transfer_is_revalidated_and_reused_on_resume(tmp_path: Path) -> None:
    _, plan, registry, store, transfer_call = _interrupted_after_transfer(tmp_path)
    planner = FixedPlanner(plan)
    result = AgentRuntime(planner=planner, registry=registry, run_store=store).resume(
        "request-1:run"
    )
    assert result.status is RunStatus.SUCCEEDED
    assert transfer_call.call_count == 1
    assert planner.calls == 0


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_annotation",
        "corrupt_annotation",
        "reference_labels",
        "reference_embedding",
        "query_embedding",
        "reference_ids",
        "query_ids",
    ),
)
def test_changed_transfer_evidence_blocks_resume_before_downstream_execution(
    tmp_path: Path, mutation: str
) -> None:
    arguments, plan, registry, store, transfer_call = _interrupted_after_transfer(
        tmp_path
    )
    state = store.load("request-1:run")
    annotation = Path(state.steps[0].result["annotation_path"])  # type: ignore[index]
    if mutation == "missing_annotation":
        annotation.unlink()
    elif mutation == "corrupt_annotation":
        annotation.write_bytes(b"not-hdf5")
    elif mutation == "reference_labels":
        source = ad.read_h5ad(arguments["reference_h5ad_path"])
        source.obs.loc[source.obs_names[0], "celltype"] = "B"
        source.write_h5ad(arguments["reference_h5ad_path"])
    elif mutation in {"reference_embedding", "query_embedding"}:
        key = f"{mutation}_path"
        path = Path(arguments[key])
        values = np.load(path, allow_pickle=False)
        values[0, 1] = 1.0
        np.save(path, values, allow_pickle=False)
    else:
        key = "reference_cell_ids_path" if mutation == "reference_ids" else "query_cell_ids_path"
        path = Path(arguments[key])
        values = path.read_text(encoding="utf-8").splitlines()
        path.write_text("".join(f"{value}\n" for value in reversed(values)), encoding="utf-8")

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
    transfer_call.assert_called_once()
    inspect_call.assert_not_called()


def test_changed_model_configuration_blocks_resume(monkeypatch, tmp_path: Path) -> None:
    _, plan, registry, store, transfer_call = _interrupted_after_transfer(tmp_path)
    monkeypatch.setattr(
        "agent.tools.analysis.label_transfer._model_config_digest",
        lambda: "0" * 64,
    )
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
    transfer_call.assert_called_once()
    inspect_call.assert_not_called()


def test_transfer_recovery_identity_and_drift_are_authoritative(tmp_path: Path) -> None:
    _, plan, registry, store, transfer_call = _interrupted_after_transfer(tmp_path)
    state = store.load("request-1:run")
    assert state.recovery_policy_snapshot is not None
    identities = {
        tool.tool_name: tool.classifier_version
        for tool in state.recovery_policy_snapshot.tools
    }
    assert identities["transfer_cell_labels"] == "transfer-cell-labels-v1"
    guarded_call = Mock(side_effect=AssertionError("policy drift invoked a tool"))
    incompatible = ToolRegistry(
        tuple(
            replace(
                registry.get(name),
                function=guarded_call,
                recovery_policy_version="transfer-cell-labels-v2",
            )
            if name == "transfer_cell_labels"
            else replace(registry.get(name), function=guarded_call)
            for name in registry.names()
        )
    )
    with pytest.raises(RecoveryPolicyIncompatibleError):
        AgentRuntime(
            planner=FixedPlanner(plan), registry=incompatible, run_store=store
        ).resume("request-1:run")
    transfer_call.assert_called_once()
    guarded_call.assert_not_called()


def test_cancellation_before_transfer_prevents_invocation(tmp_path: Path) -> None:
    arguments = _transfer_sources(tmp_path)
    plan = AgentPlan(
        "cancel-transfer",
        "request-1",
        "fixed",
        (
            PlanStep(
                "inspect",
                "inspect_scATAC",
                {"path": arguments["query_h5ad_path"]},
            ),
            PlanStep(
                "transfer",
                "transfer_cell_labels",
                arguments,
                ("inspect",),
            ),
        ),
    )
    store = FileRunStore(tmp_path / "store")

    def inspect_then_cancel(**kwargs):
        result = inspect_scATAC(**kwargs)
        store.request_cancellation("request-1:run")
        return result

    default = build_default_tool_registry()
    transfer_call = Mock(side_effect=AssertionError("cancelled run invoked transfer"))
    registry = ToolRegistry(
        tuple(
            replace(default.get(name), function=inspect_then_cancel)
            if name == "inspect_scATAC"
            else replace(default.get(name), function=transfer_call)
            if name == "transfer_cell_labels"
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
    transfer_call.assert_not_called()


def test_terminal_resume_keeps_existing_idempotent_semantics(tmp_path: Path) -> None:
    arguments = _transfer_sources(tmp_path)
    plan = AgentPlan(
        "terminal-transfer",
        "request-1",
        "fixed",
        (PlanStep("transfer", "transfer_cell_labels", arguments),),
    )
    store = FileRunStore(tmp_path / "store")
    default = build_default_tool_registry()
    transfer_call = Mock(wraps=default.get("transfer_cell_labels").function)
    registry = ToolRegistry(
        tuple(
            replace(default.get(name), function=transfer_call)
            if name == "transfer_cell_labels"
            else default.get(name)
            for name in default.names()
        )
    )
    runtime = AgentRuntime(
        planner=FixedPlanner(plan), registry=registry, run_store=store
    )
    result = runtime.run(AgentRequest("request-1", "fixed", {}))
    assert result.status is RunStatus.SUCCEEDED
    Path(result.steps[0].result["annotation_path"]).unlink()  # type: ignore[index]
    assert runtime.resume(result.run_id) == result
    transfer_call.assert_called_once()
