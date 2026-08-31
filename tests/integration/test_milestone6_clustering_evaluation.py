"""Offline production-path acceptance for Milestone 6.2 evaluation."""

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
    DeterministicPlanner,
    FileRunStore,
    PlanStep,
    RunStatus,
    RunMode,
    StepOutputRef,
    build_default_tool_registry,
    ToolRegistry,
    verify_step,
)
from agent.tools import evaluate_cell_clustering
from agent.tools.analysis.embedding_analysis import PROVENANCE_KEY, _cell_order_digest


class FixedPlanner:
    def __init__(self, plan: AgentPlan) -> None:
        self.plan_value = plan

    def plan(self, request, registry):
        return self.plan_value


def _artifacts(tmp_path: Path) -> tuple[Path, Path, tuple[str, ...]]:
    ids = tuple(f"cell-{index}" for index in range(8))
    labels = ("a", "a", "b", "b", "c", "c", "d", "d")
    reference_path = tmp_path / "reference.h5ad"
    ad.AnnData(
        X=sparse.csr_matrix((8, 100), dtype=np.float32),
        obs=pd.DataFrame({"celltype": labels}, index=pd.Index(ids)),
    ).write_h5ad(reference_path)

    analysis = ad.AnnData(
        obs=pd.DataFrame({"leiden": labels}, index=pd.Index(ids))
    )
    analysis.obsm["X_epizoo"] = np.zeros((8, 512), dtype=np.float32)
    graph = sparse.eye(8, format="csr", dtype=np.float32)
    analysis.obsp["distances"] = graph
    analysis.obsp["connectivities"] = graph.copy()
    analysis.uns["neighbors"] = {
        "distances_key": "distances",
        "connectivities_key": "connectivities",
    }
    analysis.uns[PROVENANCE_KEY] = {
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
    analysis_path = tmp_path / "clustered.h5ad"
    analysis.write_h5ad(analysis_path)
    return reference_path, analysis_path, ids


def test_real_evaluation_through_registry_verifier_and_durable_runtime(
    tmp_path: Path,
) -> None:
    reference_path, analysis_path, ids = _artifacts(tmp_path)
    output_dir = tmp_path / "outputs"
    plan = AgentPlan(
        "milestone6-2-real",
        "request-1",
        "integration",
        (
            PlanStep("inspect", "inspect_scATAC", {"path": str(reference_path)}),
            PlanStep(
                "evaluate",
                "evaluate_cell_clustering",
                {
                    "analysis_path": str(analysis_path),
                    "reference_h5ad_path": StepOutputRef("inspect", "input_path"),
                    "label_key": "celltype",
                    "output_dir": str(output_dir),
                },
                ("inspect",),
            ),
        ),
    )
    runtime = AgentRuntime(
        planner=FixedPlanner(plan),
        registry=build_default_tool_registry(),
        run_store=FileRunStore(tmp_path / "store"),
    )
    result = runtime.run(AgentRequest("request-1", "evaluate clustering", {}))
    assert result.status is RunStatus.SUCCEEDED
    evaluation = result.steps[-1]
    assert evaluation.verification is not None and evaluation.verification.passed
    payload = evaluation.result
    assert payload is not None
    assert payload["n_cells"] == len(ids)
    assert all(payload[name] == 1.0 for name in ("nmi", "ari", "ami", "homogeneity"))
    report = json.loads(Path(payload["report_path"]).read_text(encoding="utf-8"))
    assert report["counts"]["n_cells"] == len(ids)
    assert report["validation"] == {"cell_order_preserved": True, "finite": True}
    assert set(report["metrics"]) == {"nmi", "ari", "ami", "homogeneity"}
    json.dumps(result.to_dict(), allow_nan=False)


def test_evaluation_plan_only_invokes_zero_scientific_tools(tmp_path: Path) -> None:
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
    result = AgentRuntime(
        planner=DeterministicPlanner(), registry=registry
    ).run(
        AgentRequest(
            "request-plan-only",
            "Analyze with EpiZoo and evaluate the clustering",
            {
                "input_path": str(tmp_path / "not-opened.h5ad"),
                "output_dir": str(tmp_path / "not-created"),
                "species": "mouse",
                "label_key": "celltype",
            },
            RunMode.PLAN_ONLY,
        )
    )
    assert result.status is RunStatus.PLANNED
    assert tuple(step.step_id for step in result.plan.steps) == (  # type: ignore[union-attr]
        "inspect", "embed", "neighbors", "cluster", "evaluate"
    )
    assert not (tmp_path / "not-created").exists()
    for function in guarded.values():
        function.assert_not_called()


def _verification_case(tmp_path: Path):
    reference, analysis, _ = _artifacts(tmp_path)
    output = tmp_path / "output"
    result = evaluate_cell_clustering(analysis, reference, "celltype", output)
    arguments = {
        "analysis_path": str(analysis),
        "reference_h5ad_path": str(reference),
        "label_key": "celltype",
        "output_dir": str(output),
    }
    step = PlanStep("evaluate", "evaluate_cell_clustering", arguments)
    return reference, analysis, result, arguments, step


def test_verifier_independently_accepts_recomputed_metrics(tmp_path: Path) -> None:
    _, _, result, arguments, step = _verification_case(tmp_path)
    verification = verify_step(
        step, arguments, result, build_default_tool_registry()
    )
    assert verification.passed
    assert any(
        check.name == "evaluation_metrics_recomputed" and check.passed
        for check in verification.checks
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "metric",
        "report_provenance",
        "missing_report",
        "corrupt_report",
        "reference_labels",
        "predicted_labels",
        "analysis_provenance",
        "missing_analysis",
        "corrupt_analysis",
        "missing_reference",
    ],
)
def test_verifier_rejects_stale_corrupt_or_missing_evaluation_evidence(
    tmp_path: Path, mutation: str
) -> None:
    reference, analysis, result, arguments, step = _verification_case(tmp_path)
    report_path = Path(result["report_path"])
    if mutation in {"metric", "report_provenance"}:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if mutation == "metric":
            report["metrics"]["nmi"] = 0.25
        else:
            report["provenance"]["reference_labels_sha256"] = "0" * 64
        report_path.write_text(json.dumps(report), encoding="utf-8")
    elif mutation == "missing_report":
        report_path.unlink()
    elif mutation == "corrupt_report":
        report_path.write_text("not-json", encoding="utf-8")
    elif mutation == "reference_labels":
        source = ad.read_h5ad(reference)
        source.obs.loc[source.obs_names[0], "celltype"] = "b"
        source.write_h5ad(reference)
    elif mutation in {"predicted_labels", "analysis_provenance"}:
        source = ad.read_h5ad(analysis)
        if mutation == "predicted_labels":
            source.obs.loc[source.obs_names[0], "leiden"] = "b"
        else:
            source.uns[PROVENANCE_KEY]["parameters"]["clustering"]["flavor"] = "invalid"
        source.write_h5ad(analysis)
    elif mutation == "missing_analysis":
        analysis.unlink()
    elif mutation == "corrupt_analysis":
        analysis.write_text("not-hdf5")
    elif mutation == "missing_reference":
        reference.unlink()
    verification = verify_step(
        step, arguments, result, build_default_tool_registry()
    )
    assert not verification.passed
