"""Offline production-path acceptance for Milestone 6.3 label transfer."""

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


class FixedPlanner:
    def __init__(self, plan: AgentPlan) -> None:
        self.plan_value = plan

    def plan(self, request, registry):
        return self.plan_value


def _artifacts(tmp_path: Path) -> tuple[dict[str, object], tuple[str, ...]]:
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
            {"celltype": pd.Categorical(["T cell", "T cell", "B cell", "B cell"])},
            index=pd.Index(reference_ids),
        )
    ).write_h5ad(reference_h5ad)
    ad.AnnData(obs=pd.DataFrame(index=pd.Index(query_ids))).write_h5ad(query_h5ad)
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"test checkpoint provenance")
    arguments: dict[str, object] = {
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
    return arguments, query_ids


def test_real_transfer_through_registry_verifier_and_durable_runtime(
    tmp_path: Path,
) -> None:
    arguments, query_ids = _artifacts(tmp_path)
    plan = AgentPlan(
        "milestone6-3-real",
        "request-1",
        "integration",
        (PlanStep("transfer", "transfer_cell_labels", arguments),),
    )
    runtime = AgentRuntime(
        planner=FixedPlanner(plan),
        registry=build_default_tool_registry(),
        run_store=FileRunStore(tmp_path / "store"),
    )
    result = runtime.run(AgentRequest("request-1", "transfer labels", {}))
    assert result.status is RunStatus.SUCCEEDED
    transfer = result.steps[0]
    assert transfer.verification is not None and transfer.verification.passed
    payload = transfer.result
    assert payload is not None
    assert payload["n_query_cells"] == len(query_ids)
    assert payload["assigned_count"] == len(query_ids)
    assert payload["unassigned_count"] == 0
    assert payload["assignment_rate"] == 1.0
    assert payload["embedding_dim"] == 512
    assert payload["embedding_dtype"] == "float32"
    assert "predictions" not in payload
    json.dumps(result.to_dict(), allow_nan=False)

    artifact = ad.read_h5ad(payload["annotation_path"])
    assert artifact.X is None
    assert artifact.n_vars == 0
    assert tuple(str(value) for value in artifact.obs_names) == query_ids
    assert artifact.obs["predicted_label"].tolist() == ["T cell", "B cell"]
    assert artifact.obs["prediction_status"].tolist() == ["assigned", "assigned"]
    assert artifact.obs["prediction_confidence"].tolist() == [1.0, 1.0]

    persisted = runtime.resume(result.run_id)
    assert persisted == result


def test_label_transfer_plan_only_invokes_zero_scientific_tools(tmp_path: Path) -> None:
    default = build_default_tool_registry()
    guarded = {
        name: Mock(side_effect=AssertionError(f"PLAN_ONLY invoked {name}"))
        for name in default.names()
    }
    registry = ToolRegistry(
        tuple(replace(default.get(name), function=guarded[name]) for name in default.names())
    )
    result = AgentRuntime(planner=DeterministicPlanner(), registry=registry).run(
        AgentRequest(
            "request-plan-only",
            "Use the annotated reference to annotate the query cells",
            {
                "reference_input_path": str(tmp_path / "reference.h5ad"),
                "query_input_path": str(tmp_path / "query.h5ad"),
                "output_dir": str(tmp_path / "not-created"),
                "species": "mouse",
                "reference_label_key": "celltype",
                "checkpoint_path": str(tmp_path / "checkpoint.pth"),
                "device": "cuda:0",
            },
            RunMode.PLAN_ONLY,
        )
    )
    assert result.status is RunStatus.PLANNED
    assert tuple(step.step_id for step in result.plan.steps) == (  # type: ignore[union-attr]
        "inspect_reference",
        "embed_reference",
        "inspect_query",
        "embed_query",
        "transfer",
    )
    assert not (tmp_path / "not-created").exists()
    for function in guarded.values():
        function.assert_not_called()
