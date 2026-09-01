"""Durability, recovery, and cancellation checks for Milestone 6.4."""

from __future__ import annotations

from dataclasses import replace
import json
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
from agent.tools import inspect_scATAC, transfer_cell_labels


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


def _sources(tmp_path: Path) -> tuple[dict[str, object], Path, Path]:
    reference_ids = [f"reference-{index}" for index in range(4)]
    query_ids = ["query-0", "query-1"]
    reference = np.zeros((4, 512), dtype=np.float32)
    reference[:, 0] = [0.0, 0.2, 9.8, 10.0]
    query = np.zeros((2, 512), dtype=np.float32)
    query[:, 0] = [0.1, 9.9]
    reference_embedding = tmp_path / "reference.npy"
    query_embedding = tmp_path / "query.npy"
    np.save(reference_embedding, reference, allow_pickle=False)
    np.save(query_embedding, query, allow_pickle=False)
    reference_ids_path = tmp_path / "reference.txt"
    query_ids_path = tmp_path / "query.txt"
    reference_ids_path.write_text(
        "".join(f"{value}\n" for value in reference_ids), encoding="utf-8"
    )
    query_ids_path.write_text(
        "".join(f"{value}\n" for value in query_ids), encoding="utf-8"
    )
    reference_h5ad = tmp_path / "reference.h5ad"
    query_h5ad = tmp_path / "query.h5ad"
    ad.AnnData(
        obs=pd.DataFrame(
            {"celltype": pd.Categorical(["A", "A", "B", "B"])},
            index=reference_ids,
        )
    ).write_h5ad(reference_h5ad)
    ad.AnnData(obs=pd.DataFrame(index=query_ids)).write_h5ad(query_h5ad)
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint")
    transfer = transfer_cell_labels(
        reference_embedding,
        reference_ids_path,
        reference_h5ad,
        "celltype",
        query_embedding,
        query_ids_path,
        query_h5ad,
        tmp_path / "transfer",
        reference_species="mouse",
        query_species="mouse",
        reference_checkpoint_path=checkpoint,
        query_checkpoint_path=checkpoint,
        n_neighbors=2,
    )
    truth_path = tmp_path / "truth.h5ad"
    ad.AnnData(
        X=sparse.csr_matrix((2, 1), dtype=np.float32),
        obs=pd.DataFrame(
            {"truth": pd.Categorical(["A", "B"])}, index=query_ids
        ),
        var=pd.DataFrame(index=["placeholder"]),
    ).write_h5ad(truth_path)
    arguments = {
        "annotation_path": transfer["annotation_path"],
        "ground_truth_h5ad_path": str(truth_path),
        "ground_truth_label_key": "truth",
        "output_dir": str(tmp_path / "evaluation"),
    }
    return arguments, Path(transfer["annotation_path"]), truth_path


def _plan(arguments: dict[str, object]) -> AgentPlan:
    return AgentPlan(
        "plan-evaluation",
        "request-1",
        "fixed",
        (
            PlanStep(
                "evaluate_annotation", "evaluate_cell_annotation", arguments
            ),
            PlanStep(
                "inspect",
                "inspect_scATAC",
                {"path": arguments["ground_truth_h5ad_path"]},
                ("evaluate_annotation",),
            ),
        ),
    )


def _interrupted_after_evaluation(tmp_path: Path):
    arguments, annotation, truth = _sources(tmp_path)
    plan = _plan(arguments)
    default = build_default_tool_registry()
    evaluation_call = Mock(wraps=default.get("evaluate_cell_annotation").function)
    registry = ToolRegistry(
        tuple(
            replace(default.get(name), function=evaluation_call)
            if name == "evaluate_cell_annotation"
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
    return arguments, annotation, truth, plan, registry, store, evaluation_call


def test_completed_evaluation_is_revalidated_and_reused_on_resume(
    tmp_path: Path,
) -> None:
    _, _, _, plan, registry, store, evaluation_call = (
        _interrupted_after_evaluation(tmp_path)
    )
    planner = FixedPlanner(plan)
    result = AgentRuntime(planner=planner, registry=registry, run_store=store).resume(
        "request-1:run"
    )
    assert result.status is RunStatus.SUCCEEDED
    assert evaluation_call.call_count == 1
    assert planner.calls == 0


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_report",
        "corrupt_report",
        "report_metric",
        "report_confusion",
        "report_per_class",
        "annotation_confidence",
        "annotation_provenance",
        "ground_truth_label",
    ),
)
def test_changed_evaluation_evidence_blocks_nonterminal_resume(
    tmp_path: Path, mutation: str
) -> None:
    arguments, annotation, truth, plan, registry, store, evaluation_call = (
        _interrupted_after_evaluation(tmp_path)
    )
    state = store.load("request-1:run")
    report_path = Path(state.steps[0].result["report_path"])  # type: ignore[index]
    if mutation == "missing_report":
        report_path.unlink()
    elif mutation == "corrupt_report":
        report_path.write_text("not-json", encoding="utf-8")
    elif mutation.startswith("report_"):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if mutation == "report_metric":
            report["metrics"]["overall_accuracy"] = 0.123
        elif mutation == "report_confusion":
            report["confusion"]["counts"][0][0] = 0
        else:
            report["per_class"][0]["recall"] = 0.123
        report_path.write_text(json.dumps(report), encoding="utf-8")
    elif mutation.startswith("annotation_"):
        artifact = ad.read_h5ad(annotation)
        if mutation == "annotation_confidence":
            artifact.obs["prediction_confidence"] = [0.2, 1.0]
        else:
            provenance = dict(artifact.uns["agent_milestone6_label_transfer"])
            provenance["stage"] = "changed"
            artifact.uns["agent_milestone6_label_transfer"] = provenance
        artifact.write_h5ad(annotation)
    else:
        source = ad.read_h5ad(truth)
        source.obs["truth"] = pd.Categorical(["B", "A"])
        source.write_h5ad(truth)

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
    assert any(
        error.code == "PERSISTED_STEP_REVALIDATION_FAILED"
        for error in result.errors
    )
    evaluation_call.assert_called_once()
    inspect_call.assert_not_called()


def test_annotation_evaluation_recovery_identity_and_drift(tmp_path: Path) -> None:
    _, _, _, plan, registry, store, evaluation_call = _interrupted_after_evaluation(
        tmp_path
    )
    state = store.load("request-1:run")
    assert state.recovery_policy_snapshot is not None
    identities = {
        tool.tool_name: tool.classifier_version
        for tool in state.recovery_policy_snapshot.tools
    }
    assert identities["evaluate_cell_annotation"] == "evaluate-cell-annotation-v1"
    guarded_call = Mock(side_effect=AssertionError("policy drift invoked a tool"))
    incompatible = ToolRegistry(
        tuple(
            replace(
                registry.get(name),
                function=guarded_call,
                recovery_policy_version="evaluate-cell-annotation-v2",
            )
            if name == "evaluate_cell_annotation"
            else replace(registry.get(name), function=guarded_call)
            for name in registry.names()
        )
    )
    with pytest.raises(RecoveryPolicyIncompatibleError):
        AgentRuntime(
            planner=FixedPlanner(plan), registry=incompatible, run_store=store
        ).resume("request-1:run")
    evaluation_call.assert_called_once()
    guarded_call.assert_not_called()


def test_cancellation_before_evaluation_prevents_invocation(tmp_path: Path) -> None:
    arguments, _, truth = _sources(tmp_path)
    plan = AgentPlan(
        "cancel-evaluation",
        "request-1",
        "fixed",
        (
            PlanStep("inspect", "inspect_scATAC", {"path": str(truth)}),
            PlanStep(
                "evaluate_annotation",
                "evaluate_cell_annotation",
                arguments,
                ("inspect",),
            ),
        ),
    )
    store = FileRunStore(tmp_path / "cancel-store")

    def inspect_then_cancel(**kwargs):
        result = inspect_scATAC(**kwargs)
        store.request_cancellation("request-1:run")
        return result

    default = build_default_tool_registry()
    evaluation_call = Mock(
        side_effect=AssertionError("cancelled run invoked annotation evaluation")
    )
    registry = ToolRegistry(
        tuple(
            replace(default.get(name), function=inspect_then_cancel)
            if name == "inspect_scATAC"
            else replace(default.get(name), function=evaluation_call)
            if name == "evaluate_cell_annotation"
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


def test_terminal_resume_remains_idempotent(tmp_path: Path) -> None:
    arguments, _, _ = _sources(tmp_path)
    plan = AgentPlan(
        "terminal-evaluation",
        "request-1",
        "fixed",
        (
            PlanStep(
                "evaluate_annotation", "evaluate_cell_annotation", arguments
            ),
        ),
    )
    store = FileRunStore(tmp_path / "terminal-store")
    default = build_default_tool_registry()
    call = Mock(wraps=default.get("evaluate_cell_annotation").function)
    registry = ToolRegistry(
        tuple(
            replace(default.get(name), function=call)
            if name == "evaluate_cell_annotation"
            else default.get(name)
            for name in default.names()
        )
    )
    runtime = AgentRuntime(
        planner=FixedPlanner(plan), registry=registry, run_store=store
    )
    result = runtime.run(AgentRequest("request-1", "fixed", {}))
    assert result.status is RunStatus.SUCCEEDED
    Path(result.steps[0].result["report_path"]).unlink()  # type: ignore[index]
    assert runtime.resume(result.run_id) == result
    call.assert_called_once()
