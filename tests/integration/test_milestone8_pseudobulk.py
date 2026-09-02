from __future__ import annotations

from pathlib import Path
from dataclasses import replace
from unittest.mock import Mock

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from agent.application import ApplicationStatus, ResearchAgentApplication
from agent.orchestration import (
    AgentRequest,
    AgentRuntime,
    PlanStep,
    RunStatus,
    StepOutputRef,
    ToolRegistry,
    build_default_tool_registry,
    verify_step,
)
from agent.report import (
    build_analysis_evidence,
    build_analysis_report,
    get_supported_visualization_kinds,
    verify_analysis_evidence,
    verify_analysis_report,
)
from agent.tools import build_replicate_pseudobulk, validate_scATAC_feature_space


def _raw(path: Path) -> Path:
    ad.AnnData(
        X=sparse.csr_matrix(
            [[1, 0, 2], [0, 3, 0], [4, 0, 5], [0, 6, 0]], dtype="int32"
        ),
        obs=pd.DataFrame(
            {
                "cell_type": ["A", "A", "B", "B"],
                "donor": ["d1", "d1", "d2", "d2"],
                "condition": ["control", "control", "treated", "treated"],
            },
            index=["c1", "c2", "c3", "c4"],
        ),
        var=pd.DataFrame(index=["p1", "p2", "p3"]),
    ).write_h5ad(path)
    return path


def _inputs(source: Path, *, include_output: Path | None = None) -> dict[str, object]:
    values: dict[str, object] = {
        "input_path": str(source),
        "matrix_source": "X",
        "matrix_semantics": "fragment_counts",
        "species": "human",
        "genome_assembly": "hg38",
        "coordinate_source": "none",
        "replicate_key": "donor",
        "group_key": "cell_type",
        "condition_key": "condition",
        "group_source": "raw_obs",
    }
    if include_output is not None:
        values["output_dir"] = str(include_output)
    return values


def test_runtime_evidence_and_figureless_report_complete_for_m81(tmp_path: Path) -> None:
    source = _raw(tmp_path / "raw.h5ad")
    registry = build_default_tool_registry()
    run = AgentRuntime(registry=registry).run(
        AgentRequest(
            "m81-runtime",
            "Build replicate-aware pseudobulk accessibility",
            _inputs(source, include_output=tmp_path / "scientific"),
        )
    )
    assert run.status is RunStatus.SUCCEEDED
    assert tuple(step.tool_name for step in run.steps) == (
        "validate_scATAC_feature_space",
        "build_replicate_pseudobulk",
    )

    evidence = build_analysis_evidence(run, tmp_path / "evidence", registry=registry)
    assert evidence["schema_version"] == 1
    assert verify_analysis_evidence(run, evidence, registry=registry).passed
    assert get_supported_visualization_kinds(run, evidence, registry=registry) == ()
    report = build_analysis_report(run, evidence, tmp_path / "report", registry=registry)
    assert report["schema_version"] == 1
    assert verify_analysis_report(run, evidence, report, registry=registry).passed
    document = Path(report["report_path"]).read_text(encoding="utf-8")
    assert "## Regulatory Feature Space" in document
    assert "## Replicate-aware Pseudobulk" in document


def test_application_completes_and_terminal_resume_reuses_m81_outputs(tmp_path: Path) -> None:
    source = _raw(tmp_path / "raw.h5ad")
    application = ResearchAgentApplication(tmp_path / "workspace")
    result = application.run(
        AgentRequest(
            "m81-application",
            "Build replicate-aware pseudobulk accessibility",
            _inputs(source),
        )
    )
    assert result.status is ApplicationStatus.SUCCEEDED
    assert result.evidence is not None
    assert result.visualization is None
    assert result.report is not None

    resumed = application.resume(result.run_id)
    assert resumed.status is ApplicationStatus.SUCCEEDED
    assert resumed.run_result == result.run_result
    assert resumed.evidence == result.evidence
    assert resumed.visualization is None
    assert resumed.report == result.report


def test_realistic_backed_sparse_acceptance_without_matrix_densification(
    tmp_path: Path, monkeypatch
) -> None:
    rng = np.random.default_rng(81)
    n_cells = 1_024
    n_features = 50_000
    n_entries = 75_000
    matrix = sparse.coo_matrix(
        (
            rng.integers(1, 4, size=n_entries, dtype=np.int32),
            (
                rng.integers(0, n_cells, size=n_entries),
                rng.integers(0, n_features, size=n_entries),
            ),
        ),
        shape=(n_cells, n_features),
    ).tocsc()
    matrix.sum_duplicates()
    source = tmp_path / "realistic.h5ad"
    ad.AnnData(
        X=matrix,
        obs=pd.DataFrame(
            {
                "group": [f"celltype-{index % 8}" for index in range(n_cells)],
                "replicate": [f"donor-{index % 32}" for index in range(n_cells)],
                "condition": [
                    "treated" if index % 64 >= 32 else "control"
                    for index in range(n_cells)
                ],
            },
            index=[f"cell-{index}" for index in range(n_cells)],
        ),
        var=pd.DataFrame(index=[f"peak-{index}" for index in range(n_features)]),
    ).write_h5ad(source)

    def reject_dense(*args, **kwargs):
        raise AssertionError("M8.1 attempted complete sparse-matrix densification")

    monkeypatch.setattr(sparse.csr_matrix, "toarray", reject_dense)
    monkeypatch.setattr(sparse.csc_matrix, "toarray", reject_dense)
    feature_arguments = {
        "input_path": str(source),
        "output_dir": str(tmp_path / "feature"),
        "matrix_source": "X",
        "matrix_semantics": "fragment_counts",
        "species": "human",
        "genome_assembly": "hg38",
        "coordinate_source": "none",
    }
    feature = validate_scATAC_feature_space(**feature_arguments)
    pseudobulk_arguments = {
        "feature_space_path": feature["feature_space_path"],
        "replicate_key": "replicate",
        "group_key": "group",
        "condition_key": "condition",
        "output_dir": str(tmp_path / "pseudobulk"),
        "group_source": "raw_obs",
    }
    pseudobulk = build_replicate_pseudobulk(**pseudobulk_arguments)

    feature_step = PlanStep(
        "feature", "validate_scATAC_feature_space", feature_arguments
    )
    pseudobulk_step = PlanStep(
        "pseudobulk",
        "build_replicate_pseudobulk",
        {
            **pseudobulk_arguments,
            "feature_space_path": StepOutputRef("feature", "feature_space_path"),
        },
        ("feature",),
    )
    default = build_default_tool_registry()
    guard = Mock(side_effect=AssertionError("verifier invoked a scientific callable"))
    registry = ToolRegistry(
        tuple(
            replace(default.get(name), function=guard)
            if name in {
                "validate_scATAC_feature_space",
                "build_replicate_pseudobulk",
            }
            else default.get(name)
            for name in default.names()
        )
    )
    assert verify_step(feature_step, feature_arguments, feature, registry).passed
    assert verify_step(
        pseudobulk_step,
        pseudobulk_arguments,
        pseudobulk,
        registry,
        dependency_results={"feature": feature},
    ).passed
    guard.assert_not_called()

    artifact = ad.read_h5ad(pseudobulk["pseudobulk_path"], backed="r")
    try:
        assert artifact.shape == (64, n_features)
        assert isinstance(artifact.X, ad.abc.CSRDataset)
        assert artifact.X.dtype == np.dtype(np.int64)
        assert pseudobulk["n_cells"] == n_cells
        assert pseudobulk["all_cells_accounted_for"] is True
    finally:
        artifact.file.close()
