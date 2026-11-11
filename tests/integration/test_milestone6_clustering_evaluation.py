"""Offline production-path acceptance for Milestone 6.2 evaluation."""

from __future__ import annotations

import json
from pathlib import Path

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
    RunStatus,
    StepOutputRef,
    build_default_tool_registry,
    verify_step,
)
from agent.tools import evaluate_cell_clustering



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
    analysis.uns['agent_native_invocation'] = dict(function='snapatac2.tl.leiden', parameters={'resolution':1.},source_sha256='a'*64)
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
            source.uns['agent_native_invocation']['parameters']['resolution'] = 7.
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
