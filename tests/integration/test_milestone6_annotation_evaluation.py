"""Offline production-path acceptance for Milestone 6.4 evaluation."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import Mock

import anndata as ad
import numpy as np
import pandas as pd

from agent.orchestration import (
    AgentPlan,
    AgentRequest,
    AgentRuntime,
    DeterministicPlanner,
    FileRunStore,
    PlanStep,
    RunMode,
    RunStatus,
    ToolRegistry,
    build_default_tool_registry,
)
from agent.tools import transfer_cell_labels


class FixedPlanner:
    def __init__(self, plan: AgentPlan) -> None:
        self.plan_value = plan

    def plan(self, request, registry):
        return self.plan_value


def _fixed_annotation_and_truth(tmp_path: Path) -> tuple[Path, Path]:
    reference_ids = [f"reference-{index}" for index in range(4)]
    query_ids = [f"query-{index}" for index in range(3)]
    reference = np.zeros((4, 512), dtype=np.float32)
    reference[:, 0] = [0.0, 0.2, 9.8, 10.0]
    query = np.zeros((3, 512), dtype=np.float32)
    query[:, 0] = [0.1, 9.9, 0.15]
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
    checkpoint.write_bytes(b"checkpoint provenance")
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
        obs=pd.DataFrame(
            {"truth": pd.Categorical(["A", "B", "B"])}, index=query_ids
        )
    ).write_h5ad(truth_path)
    return Path(transfer["annotation_path"]), truth_path


def test_real_annotation_evaluation_through_registry_verifier_and_runtime(
    tmp_path: Path,
) -> None:
    annotation_path, truth_path = _fixed_annotation_and_truth(tmp_path)
    arguments = {
        "annotation_path": str(annotation_path),
        "ground_truth_h5ad_path": str(truth_path),
        "ground_truth_label_key": "truth",
        "output_dir": str(tmp_path / "evaluation"),
    }
    plan = AgentPlan(
        "milestone6-4-real",
        "request-1",
        "integration",
        (PlanStep("evaluate_annotation", "evaluate_cell_annotation", arguments),),
    )
    runtime = AgentRuntime(
        planner=FixedPlanner(plan),
        registry=build_default_tool_registry(),
        run_store=FileRunStore(tmp_path / "store"),
    )
    result = runtime.run(AgentRequest("request-1", "evaluate annotation", {}))
    assert result.status is RunStatus.SUCCEEDED
    step = result.steps[0]
    assert step.verification is not None and step.verification.passed
    payload = step.result
    assert payload is not None
    assert payload["n_cells"] == 3
    assert payload["assigned_count"] == 3
    assert payload["correct_assigned_count"] == 2
    assert payload["overall_accuracy"] == 2 / 3
    assert payload["assigned_accuracy"] == 2 / 3
    assert payload["finite"] is True
    forbidden = {
        "ground_truth_labels",
        "predicted_labels",
        "prediction_status",
        "prediction_confidence",
        "cell_ids",
        "confusion",
        "per_class",
    }
    assert forbidden.isdisjoint(payload)
    json.dumps(result.to_dict(), allow_nan=False)

    report = json.loads(Path(payload["report_path"]).read_text(encoding="utf-8"))
    assert report["artifact_type"] == "agent.cell-annotation-evaluation"
    assert report["schema_version"] == 1
    assert sum(map(sum, report["confusion"]["counts"])) == 3
    assert [entry["label"] for entry in report["per_class"]] == ["A", "B"]
    assert runtime.resume(result.run_id) == result


def test_annotation_evaluation_plan_only_invokes_zero_tools(tmp_path: Path) -> None:
    default = build_default_tool_registry()
    guarded = {
        name: Mock(side_effect=AssertionError(f"PLAN_ONLY invoked {name}"))
        for name in default.names()
    }
    registry = ToolRegistry(
        tuple(
            replace(default.get(name), function=guarded[name])
            for name in default.names()
        )
    )
    result = AgentRuntime(planner=DeterministicPlanner(), registry=registry).run(
        AgentRequest(
            "request-plan-only",
            "Evaluate this fixed annotation",
            {
                "annotation_path": str(tmp_path / "annotation.h5ad"),
                "ground_truth_h5ad_path": str(tmp_path / "truth.h5ad"),
                "ground_truth_label_key": "celltype",
                "output_dir": str(tmp_path / "not-created"),
            },
            RunMode.PLAN_ONLY,
        )
    )
    assert result.status is RunStatus.PLANNED
    assert tuple(step.step_id for step in result.plan.steps) == (  # type: ignore[union-attr]
        "evaluate_annotation",
    )
    assert not (tmp_path / "not-created").exists()
    for function in guarded.values():
        function.assert_not_called()
