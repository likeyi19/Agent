from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import agent.report.evidence as evidence_module
from agent.orchestration import (
    ResultContract,
    ToolRegistry,
    build_default_tool_registry,
    verify_step,
    verify_run,
)
from agent.report import (
    ANALYSIS_EVIDENCE_ARTIFACT_TYPE,
    ANALYSIS_EVIDENCE_SCHEMA_VERSION,
    AnalysisEvidenceError,
    build_analysis_evidence,
    verify_analysis_evidence,
)
from agent.schemas import (
    AgentError,
    AgentPlan,
    AgentRunResult,
    ErrorCategory,
    PlanStep,
    RunStatus,
    StepExecutionResult,
    StepOutputRef,
    StepStatus,
    VerificationCheck,
    VerificationResult,
)
from agent.tools import (
    build_cell_neighbors,
    cluster_cells,
    evaluate_cell_clustering,
    transfer_cell_labels,
)


def _passed(target_type: str, target_id: str) -> VerificationResult:
    return VerificationResult(
        True,
        target_type,
        target_id,
        (VerificationCheck("accepted", True, "Accepted."),),
    )


def _inspection_result(path: str = "/data/input.h5ad") -> dict[str, object]:
    return {
        "input_path": path,
        "n_cells": 2,
        "n_features": 3,
        "x_storage_type": "scipy.sparse.csr_matrix",
        "x_is_sparse": True,
        "x_dtype": "float32",
        "nnz": 2,
        "density": 1.0 / 3.0,
        "obs_columns": ["batch"],
        "var_columns": ["feature_type"],
        "obs_names_sample": ["cell-1", "cell-2"],
        "var_names_sample": ["peak-1", "peak-2", "peak-3"],
    }


def _inspection_run(
    *,
    result: dict[str, object] | None = None,
    path: str = "/data/input.h5ad",
) -> AgentRunResult:
    plan = AgentPlan(
        "plan-1",
        "request-1",
        "test-planner",
        (PlanStep("inspect", "inspect_scATAC", {"path": path}),),
    )
    step = StepExecutionResult(
        "inspect",
        "inspect_scATAC",
        StepStatus.SUCCEEDED,
        attempt_count=1,
        resolved_arguments={"path": path},
        result=result or _inspection_result(path),
        verification=_passed("step", "inspect"),
    )
    return AgentRunResult(
        "request-1:run",
        "request-1",
        RunStatus.SUCCEEDED,
        False,
        plan=plan,
        steps=(step,),
        verification=verify_run(plan, (step,)),
    )


def _guarded_registry() -> ToolRegistry:
    default = build_default_tool_registry()

    def forbidden(**_: object) -> object:
        raise AssertionError("Evidence processing invoked a scientific callable.")

    return ToolRegistry(
        tuple(replace(default.get(name), function=forbidden) for name in default.names())
    )


def _all_tool_run(*, extra_embedding_field: bool = False) -> AgentRunResult:
    digest = "a" * 64
    versions = {"backend": "1.0"}
    definitions: list[tuple[str, str, dict[str, object], dict[str, object]]] = [
        (
            "inspect",
            "inspect_scATAC",
            {"path": "/data/input.h5ad"},
            _inspection_result(),
        ),
        (
            "embed",
            "epizoo_embed_cells",
            {
                "input_path": "/data/input.h5ad",
                "output_dir": "/output",
                "species": "mouse",
            },
            {
                "status": "success",
                "input_path": "/data/input.h5ad",
                "embedding_path": "/output/embedding.npy",
                "cell_ids_path": "/output/cells.txt",
                "n_cells": 2,
                "embedding_dim": 512,
                "embedding_dtype": "float32",
                "finite": True,
                "cell_order_preserved": True,
                "backend": "EpiZoo",
                "species": "mouse",
                "checkpoint_path": "/models/epizoo.pth",
                "device": "cuda:0",
            },
        ),
        (
            "neighbors",
            "build_cell_neighbors",
            {
                "embedding_path": "/output/embedding.npy",
                "cell_ids_path": "/output/cells.txt",
                "output_dir": "/output",
            },
            {
                "status": "success",
                "embedding_path": "/output/embedding.npy",
                "cell_ids_path": "/output/cells.txt",
                "analysis_path": "/output/neighbors.h5ad",
                "n_cells": 2,
                "embedding_dim": 512,
                "n_neighbors": 15,
                "metric": "euclidean",
                "neighbors_method": "umap",
                "transformer": "none",
                "random_seed": 0,
                "connectivities_nnz": 2,
                "distances_nnz": 2,
                "finite": True,
                "cell_order_preserved": True,
                "backend": "Scanpy",
                "software_versions": versions,
            },
        ),
        (
            "cluster",
            "cluster_cells",
            {"analysis_path": "/output/neighbors.h5ad", "output_dir": "/output"},
            {
                "status": "success",
                "input_analysis_path": "/output/neighbors.h5ad",
                "analysis_path": "/output/clustered.h5ad",
                "n_cells": 2,
                "n_clusters": 2,
                "cluster_key": "leiden",
                "algorithm": "leiden",
                "resolution": 1.0,
                "random_seed": 0,
                "cell_order_preserved": True,
                "backend": "Scanpy",
                "software_versions": versions,
            },
        ),
        (
            "umap",
            "compute_cell_umap",
            {"analysis_path": "/output/clustered.h5ad", "output_dir": "/output"},
            {
                "status": "success",
                "input_analysis_path": "/output/clustered.h5ad",
                "analysis_path": "/output/umap.h5ad",
                "n_cells": 2,
                "n_components": 2,
                "umap_key": "X_umap",
                "coordinate_dtype": "float32",
                "finite": True,
                "min_dist": 0.5,
                "spread": 1.0,
                "random_seed": 0,
                "cell_order_preserved": True,
                "backend": "Scanpy",
                "software_versions": versions,
            },
        ),
        (
            "cluster_eval",
            "evaluate_cell_clustering",
            {
                "analysis_path": "/output/clustered.h5ad",
                "reference_h5ad_path": "/data/reference.h5ad",
                "label_key": "celltype",
                "output_dir": "/output",
            },
            {
                "status": "success",
                "analysis_path": "/output/clustered.h5ad",
                "reference_h5ad_path": "/data/reference.h5ad",
                "report_path": "/output/clustering_metrics.json",
                "label_key": "celltype",
                "cluster_key": "leiden",
                "n_cells": 2,
                "n_reference_classes": 2,
                "n_predicted_clusters": 2,
                "nmi": 1.0,
                "ari": 1.0,
                "ami": 1.0,
                "homogeneity": 1.0,
                "finite": True,
                "cell_order_preserved": True,
                "metric_backend": "scikit-learn",
                "average_method": "arithmetic",
                "report_schema_version": 1,
                "software_versions": versions,
            },
        ),
        (
            "transfer",
            "transfer_cell_labels",
            {
                "reference_embedding_path": "/output/ref.npy",
                "reference_cell_ids_path": "/output/ref.txt",
                "reference_h5ad_path": "/data/ref.h5ad",
                "reference_label_key": "celltype",
                "query_embedding_path": "/output/query.npy",
                "query_cell_ids_path": "/output/query.txt",
                "query_h5ad_path": "/data/query.h5ad",
                "output_dir": "/output",
                "reference_species": "mouse",
                "query_species": "mouse",
                "reference_checkpoint_path": "/models/epizoo.pth",
                "query_checkpoint_path": "/models/epizoo.pth",
            },
            {
                "status": "success",
                "annotation_path": "/output/annotation.h5ad",
                "annotation_sha256": digest,
                "reference_embedding_path": "/output/ref.npy",
                "reference_cell_ids_path": "/output/ref.txt",
                "reference_h5ad_path": "/data/ref.h5ad",
                "query_embedding_path": "/output/query.npy",
                "query_cell_ids_path": "/output/query.txt",
                "query_h5ad_path": "/data/query.h5ad",
                "checkpoint_path": "/models/epizoo.pth",
                "reference_label_key": "celltype",
                "n_reference_cells": 2,
                "n_query_cells": 2,
                "n_reference_classes": 2,
                "assigned_count": 2,
                "unassigned_count": 0,
                "assignment_rate": 1.0,
                "embedding_dim": 512,
                "embedding_dtype": "float32",
                "n_neighbors": 20,
                "metric": "euclidean",
                "voting_method": "uniform_plurality",
                "min_confidence": 0.0,
                "backend": "scikit-learn exact pairwise distances",
                "species": "mouse",
                "species_compatible": True,
                "checkpoint_compatible": True,
                "cell_order_preserved": True,
                "finite": True,
                "reference_embedding_sha256": digest,
                "query_embedding_sha256": digest,
                "reference_cell_ids_sha256": digest,
                "query_cell_ids_sha256": digest,
                "reference_labels_sha256": digest,
                "model_config_sha256": digest,
                "artifact_schema_version": 1,
                "software_versions": versions,
            },
        ),
        (
            "annotation_eval",
            "evaluate_cell_annotation",
            {
                "annotation_path": "/output/annotation.h5ad",
                "ground_truth_h5ad_path": "/data/truth.h5ad",
                "ground_truth_label_key": "celltype",
                "output_dir": "/output",
            },
            {
                "status": "success",
                "annotation_path": "/output/annotation.h5ad",
                "annotation_sha256": digest,
                "ground_truth_h5ad_path": "/data/truth.h5ad",
                "report_path": "/output/annotation_evaluation.json",
                "ground_truth_label_key": "celltype",
                "n_cells": 2,
                "n_ground_truth_classes": 2,
                "n_assigned_predicted_classes": 2,
                "assigned_count": 2,
                "unassigned_count": 0,
                "assignment_rate": 1.0,
                "correct_assigned_count": 2,
                "incorrect_assigned_count": 0,
                "overall_accuracy": 1.0,
                "assigned_accuracy": 1.0,
                "macro_f1": 1.0,
                "median_confidence": 1.0,
                "median_assigned_confidence": 1.0,
                "median_correct_assigned_confidence": 1.0,
                "median_incorrect_assigned_confidence": None,
                "finite": True,
                "cell_order_preserved": True,
                "metric_backend": "scikit-learn",
                "macro_average": "macro",
                "zero_division": 0,
                "report_schema_version": 1,
                "software_versions": versions,
            },
        ),
    ]
    if extra_embedding_field:
        definitions[1][3]["future_unapproved_payload"] = ["must", "not", "leak"]
    steps = tuple(
        PlanStep(step_id, tool_name, arguments)
        for step_id, tool_name, arguments, _ in definitions
    )
    plan = AgentPlan("all-tools-plan", "all-tools", "test-planner", steps)
    results = tuple(
        StepExecutionResult(
            step_id,
            tool_name,
            StepStatus.SUCCEEDED,
            attempt_count=1,
            resolved_arguments=arguments,
            result=result,
            verification=_passed("step", step_id),
        )
        for step_id, tool_name, arguments, result in definitions
    )
    return AgentRunResult(
        "all-tools:run",
        "all-tools",
        RunStatus.SUCCEEDED,
        False,
        plan=plan,
        steps=results,
        verification=verify_run(plan, results),
    )


def _accept_fresh_step(step, *_args, **_kwargs) -> VerificationResult:
    return VerificationResult(
        True,
        "step",
        step.step_id,
        (VerificationCheck("fresh", True, "Freshly verified."),),
    )


def _single_step_run(
    step: PlanStep,
    resolved_arguments: dict[str, object],
    result: dict[str, object],
    registry: ToolRegistry,
    *,
    request_id: str,
) -> AgentRunResult:
    plan = AgentPlan(
        f"{request_id}-plan", request_id, "test-planner", (step,)
    )
    stored_verification = verify_step(
        step, resolved_arguments, result, registry
    )
    assert stored_verification.passed
    step_result = StepExecutionResult(
        step.step_id,
        step.tool_name,
        StepStatus.SUCCEEDED,
        1,
        resolved_arguments,
        result,
        stored_verification,
    )
    return AgentRunResult(
        f"{request_id}:run",
        request_id,
        RunStatus.SUCCEEDED,
        False,
        plan=plan,
        steps=(step_result,),
        verification=verify_run(plan, (step_result,)),
    )


def test_builds_strict_deterministic_inspection_evidence(tmp_path: Path) -> None:
    run = _inspection_run()
    registry = _guarded_registry()
    first = build_analysis_evidence(run, tmp_path / "first", registry=registry)
    second = build_analysis_evidence(run, tmp_path / "second", registry=registry)

    assert first["evidence_sha256"] == second["evidence_sha256"]
    assert first["tool_names"] == ["inspect_scATAC"]
    assert first["all_steps_verified"] is True
    first_bytes = Path(first["evidence_path"]).read_bytes()
    second_bytes = Path(second["evidence_path"]).read_bytes()
    assert first_bytes == second_bytes
    assert hashlib.sha256(first_bytes).hexdigest() == first["evidence_sha256"]
    payload = json.loads(first_bytes)
    assert payload["schema_version"] == ANALYSIS_EVIDENCE_SCHEMA_VERSION
    assert payload["artifact_type"] == ANALYSIS_EVIDENCE_ARTIFACT_TYPE
    assert payload["validation"] == {
        "fresh_run_verification_passed": True,
        "all_steps_freshly_verified": True,
        "all_dependencies_resolved_from_verified_results": True,
        "prohibited_large_scientific_payloads_included": False,
        "scientific_tools_invoked_during_evidence_processing": False,
    }
    assert verify_analysis_evidence(run, first, registry=registry).passed


def test_projection_covers_all_eight_tools_and_artifact_integrity_modes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(evidence_module, "verify_step", _accept_fresh_step)
    run = _all_tool_run()
    result = build_analysis_evidence(
        run, tmp_path, registry=build_default_tool_registry()
    )
    payload = json.loads(Path(result["evidence_path"]).read_text())

    assert result["n_steps"] == 8
    assert len(payload["steps"]) == 8
    assert {step["tool_name"] for step in payload["steps"]} == {
        "inspect_scATAC",
        "epizoo_embed_cells",
        "build_cell_neighbors",
        "cluster_cells",
        "compute_cell_umap",
        "evaluate_cell_clustering",
        "transfer_cell_labels",
        "evaluate_cell_annotation",
    }
    artifacts = {value["artifact_kind"]: value for value in payload["artifacts"]}
    assert len(artifacts) == 8
    assert artifacts["cell_label_transfer_h5ad"]["integrity"][
        "authoritative_digest"
    ]["value"] == "a" * 64
    assert artifacts["epizoo_embedding_npy"]["integrity"][
        "authoritative_digest"
    ] is None
    assert "fresh_existing_verifier" in artifacts["cell_umap_h5ad"][
        "integrity"
    ]["verification_basis"]


def test_unapproved_future_result_field_is_not_projected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(evidence_module, "verify_step", _accept_fresh_step)
    result = build_analysis_evidence(
        _all_tool_run(extra_embedding_field=True),
        tmp_path,
        registry=build_default_tool_registry(),
    )
    text = Path(result["evidence_path"]).read_text()
    assert "future_unapproved_payload" not in text
    assert '"must"' not in text


def test_contract_drift_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(evidence_module, "verify_step", _accept_fresh_step)
    base = build_default_tool_registry()
    inspect = base.get("inspect_scATAC")
    fields = dict(inspect.result_contract.required_fields)
    fields["future_required_field"] = (str,)
    changed = replace(
        inspect,
        result_contract=ResultContract(
            "FutureInspection", fields, inspect.result_contract.validator
        ),
    )
    registry = ToolRegistry(
        (changed,) + tuple(base.get(name) for name in base.names()[1:])
    )
    with pytest.raises(AnalysisEvidenceError, match="incompatible") as caught:
        build_analysis_evidence(_inspection_run(), tmp_path, registry=registry)
    assert caught.value.code == "EVIDENCE_TOOL_SCHEMA_INCOMPATIBLE"


@pytest.mark.parametrize("status", [RunStatus.FAILED, RunStatus.CANCELLED, RunStatus.PLANNED])
def test_rejects_non_successful_source_runs(status: RunStatus, tmp_path: Path) -> None:
    run = replace(_inspection_run(), status=status)
    with pytest.raises(AnalysisEvidenceError) as caught:
        build_analysis_evidence(run, tmp_path, registry=build_default_tool_registry())
    assert caught.value.code == "EVIDENCE_SOURCE_RUN_NOT_SUCCEEDED"


def test_rejects_plan_only_source(tmp_path: Path) -> None:
    run = replace(_inspection_run(), planning_only=True)
    with pytest.raises(AnalysisEvidenceError):
        build_analysis_evidence(run, tmp_path, registry=build_default_tool_registry())


def test_rejects_inconsistent_source_identity_or_success_errors(tmp_path: Path) -> None:
    run = _inspection_run()
    registry = build_default_tool_registry()
    with pytest.raises(AnalysisEvidenceError) as identity:
        build_analysis_evidence(
            replace(run, run_id="different:run"), tmp_path / "identity", registry=registry
        )
    assert identity.value.code == "EVIDENCE_SOURCE_RUN_IDENTITY_INVALID"

    source_error = AgentError(
        ErrorCategory.INTERNAL_AGENT_ERROR,
        "UNEXPECTED_SUCCESS_ERROR",
        "A successful source run cannot retain errors.",
    )
    with pytest.raises(AnalysisEvidenceError) as errors:
        build_analysis_evidence(
            replace(run, errors=(source_error,)), tmp_path / "errors", registry=registry
        )
    assert errors.value.code == "EVIDENCE_SOURCE_RUN_IDENTITY_INVALID"


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "mismatched"])
def test_rejects_invalid_step_sets(mutation: str, tmp_path: Path) -> None:
    run = _inspection_run()
    if mutation == "missing":
        steps = ()
    elif mutation == "duplicate":
        steps = (run.steps[0], run.steps[0])
    else:
        steps = (replace(run.steps[0], tool_name="epizoo_embed_cells"),)
    with pytest.raises(AnalysisEvidenceError):
        build_analysis_evidence(
            replace(run, steps=steps),
            tmp_path,
            registry=build_default_tool_registry(),
        )


def test_rejects_failed_stored_step_verification(tmp_path: Path) -> None:
    run = _inspection_run()
    failed = VerificationResult(
        False,
        "step",
        "inspect",
        (VerificationCheck("stored", False, "Rejected."),),
        AgentError(ErrorCategory.VERIFICATION_ERROR, "REJECTED", "Rejected."),
    )
    altered = replace(run, steps=(replace(run.steps[0], verification=failed),))
    with pytest.raises(AnalysisEvidenceError) as caught:
        build_analysis_evidence(
            altered, tmp_path, registry=build_default_tool_registry()
        )
    assert caught.value.code == "EVIDENCE_SOURCE_STEP_NOT_VERIFIED"


def test_unsupported_tool_identity_fails_closed(tmp_path: Path) -> None:
    plan = AgentPlan(
        "future-plan",
        "future-request",
        "future-planner",
        (PlanStep("future", "future_scientific_tool", {}),),
    )
    step = StepExecutionResult(
        "future",
        "future_scientific_tool",
        StepStatus.SUCCEEDED,
        attempt_count=1,
        result={"status": "success"},
        verification=_passed("step", "future"),
    )
    run = AgentRunResult(
        "future-request:run",
        "future-request",
        RunStatus.SUCCEEDED,
        False,
        plan=plan,
        steps=(step,),
        verification=verify_run(plan, (step,)),
    )
    with pytest.raises(AnalysisEvidenceError) as caught:
        build_analysis_evidence(run, tmp_path, registry=build_default_tool_registry())
    assert caught.value.code == "EVIDENCE_TOOL_UNSUPPORTED"


def test_fresh_verification_uses_topological_dependencies_without_tool_calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    first = PlanStep("first", "inspect_scATAC", {"path": "/data/input.h5ad"})
    second = PlanStep(
        "second",
        "inspect_scATAC",
        {"path": StepOutputRef("first", "input_path")},
        ("first",),
    )
    plan = AgentPlan("dependency-plan", "dependency", "test", (second, first))
    first_result = StepExecutionResult(
        "first",
        "inspect_scATAC",
        StepStatus.SUCCEEDED,
        1,
        {"path": "/data/input.h5ad"},
        _inspection_result(),
        _passed("step", "first"),
    )
    second_result = StepExecutionResult(
        "second",
        "inspect_scATAC",
        StepStatus.SUCCEEDED,
        1,
        {"path": "/data/input.h5ad"},
        _inspection_result(),
        _passed("step", "second"),
    )
    run = AgentRunResult(
        "dependency:run",
        "dependency",
        RunStatus.SUCCEEDED,
        False,
        plan=plan,
        steps=(second_result, first_result),
        verification=verify_run(plan, (second_result, first_result)),
    )
    calls: list[tuple[str, tuple[str, ...]]] = []
    original = evidence_module.verify_step

    def recording(step, *args, dependency_results=None, **kwargs):
        calls.append((step.step_id, tuple((dependency_results or {}).keys())))
        return original(
            step,
            *args,
            dependency_results=dependency_results,
            **kwargs,
        )

    monkeypatch.setattr(evidence_module, "verify_step", recording)
    result = build_analysis_evidence(run, tmp_path, registry=_guarded_registry())
    assert calls == [("first", ()), ("second", ("first",))]
    assert verify_analysis_evidence(run, result, registry=_guarded_registry()).passed


def test_overwrite_and_atomic_failure_behavior(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run = _inspection_run()
    registry = build_default_tool_registry()
    build_analysis_evidence(run, tmp_path / "existing", registry=registry)
    with pytest.raises(FileExistsError):
        build_analysis_evidence(run, tmp_path / "existing", registry=registry)
    replaced = build_analysis_evidence(
        run, tmp_path / "existing", registry=registry, overwrite=True
    )
    assert Path(replaced["evidence_path"]).is_file()

    def fail_replace(*_args: object) -> None:
        raise OSError("simulated atomic publication failure")

    monkeypatch.setattr(evidence_module.os, "replace", fail_replace)
    failed_dir = tmp_path / "failed"
    with pytest.raises(AnalysisEvidenceError) as caught:
        build_analysis_evidence(run, failed_dir, registry=registry)
    assert caught.value.code == "EVIDENCE_WRITE_FAILED"
    assert not (failed_dir / "analysis_evidence.json").exists()
    assert not tuple(failed_dir.glob("*.tmp"))


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        (lambda value: value.__setitem__("schema_version", 2), "EVIDENCE_SCHEMA_UNSUPPORTED"),
        (lambda value: value.__setitem__("artifact_type", "wrong"), "EVIDENCE_ARTIFACT_TYPE_INVALID"),
        (lambda value: value["run"].__setitem__("plan_id", "tampered"), "EVIDENCE_CONTENT_MISMATCH"),
        (lambda value: value["steps"][0]["facts"].__setitem__("n_cells", 999), "EVIDENCE_CONTENT_MISMATCH"),
        (lambda value: value["validation"].__setitem__("all_steps_freshly_verified", False), "EVIDENCE_CONTENT_MISMATCH"),
    ],
)
def test_evidence_content_tampering_is_rejected(
    mutation, expected_code: str, tmp_path: Path
) -> None:
    run = _inspection_run()
    registry = build_default_tool_registry()
    result = build_analysis_evidence(run, tmp_path, registry=registry)
    path = Path(result["evidence_path"])
    payload = json.loads(path.read_text())
    mutation(payload)
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    verification = verify_analysis_evidence(run, path, registry=registry)
    assert not verification.passed
    assert verification.error is not None
    assert verification.error.code == expected_code


def test_duplicate_keys_and_nonfinite_json_are_rejected(tmp_path: Path) -> None:
    run = _inspection_run()
    registry = build_default_tool_registry()
    result = build_analysis_evidence(run, tmp_path, registry=registry)
    path = Path(result["evidence_path"])
    path.write_text(
        '{"schema_version":1,"schema_version":1,"artifact_type":"agent.analysis-evidence","status":"success"}',
        encoding="utf-8",
    )
    duplicate = verify_analysis_evidence(run, path, registry=registry)
    assert not duplicate.passed
    assert duplicate.error is not None
    assert duplicate.error.code == "EVIDENCE_ARTIFACT_MALFORMED"

    path.write_text(
        '{"schema_version":1,"artifact_type":"agent.analysis-evidence","status":"success","bad":NaN}',
        encoding="utf-8",
    )
    nonfinite = verify_analysis_evidence(run, path, registry=registry)
    assert not nonfinite.passed
    assert nonfinite.error is not None
    assert nonfinite.error.code == "EVIDENCE_ARTIFACT_MALFORMED"


def test_analysis_evidence_result_digest_is_authoritative(tmp_path: Path) -> None:
    run = _inspection_run()
    registry = build_default_tool_registry()
    result = build_analysis_evidence(run, tmp_path, registry=registry)
    altered = dict(result)
    altered["evidence_sha256"] = "0" * 64
    verification = verify_analysis_evidence(run, altered, registry=registry)
    assert not verification.passed
    assert verification.error is not None
    assert verification.error.code == "EVIDENCE_SHA256_MISMATCH"

    wrong_run = dict(result)
    wrong_run["run_id"] = "different:run"
    identity = verify_analysis_evidence(run, wrong_run, registry=registry)
    assert not identity.passed
    assert identity.error is not None
    assert identity.error.code == "EVIDENCE_RESULT_INVALID"


def test_missing_verified_source_artifact_prevents_evidence_creation(
    tmp_path: Path,
) -> None:
    embedding_path = tmp_path / "embedding.npy"
    cell_ids_path = tmp_path / "cells.txt"
    embedding_path.write_bytes(b"nonempty")
    cell_ids_path.write_text("cell-1\n", encoding="utf-8")
    arguments = {
        "input_path": str(tmp_path / "input.h5ad"),
        "output_dir": str(tmp_path),
        "species": "mouse",
    }
    plan = AgentPlan(
        "embedding-plan",
        "embedding-request",
        "test-planner",
        (PlanStep("embed", "epizoo_embed_cells", arguments),),
    )
    result = {
        "status": "success",
        "input_path": arguments["input_path"],
        "embedding_path": str(embedding_path),
        "cell_ids_path": str(cell_ids_path),
        "n_cells": 1,
        "embedding_dim": 512,
        "embedding_dtype": "float32",
        "finite": True,
        "cell_order_preserved": True,
        "backend": "EpiZoo",
        "species": "mouse",
        "checkpoint_path": "/models/epizoo.pth",
        "device": "cuda:0",
    }
    step = StepExecutionResult(
        "embed",
        "epizoo_embed_cells",
        StepStatus.SUCCEEDED,
        1,
        arguments,
        result,
        _passed("step", "embed"),
    )
    run = AgentRunResult(
        "embedding-request:run",
        "embedding-request",
        RunStatus.SUCCEEDED,
        False,
        plan=plan,
        steps=(step,),
        verification=verify_run(plan, (step,)),
    )
    registry = _guarded_registry()
    before = build_analysis_evidence(run, tmp_path / "before", registry=registry)
    evidence_path = Path(before["evidence_path"])
    evidence_payload = json.loads(evidence_path.read_text())
    evidence_payload["artifacts"][0]["artifact_path"] = "/tampered/embedding.npy"
    evidence_path.write_text(
        json.dumps(evidence_payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    artifact_metadata = verify_analysis_evidence(run, evidence_path, registry=registry)
    assert not artifact_metadata.passed
    assert artifact_metadata.error is not None
    assert artifact_metadata.error.code == "EVIDENCE_CONTENT_MISMATCH"

    embedding_path.unlink()
    with pytest.raises(AnalysisEvidenceError) as caught:
        build_analysis_evidence(run, tmp_path / "after", registry=registry)
    assert caught.value.code == "EVIDENCE_SOURCE_STEP_REVALIDATION_FAILED"


def test_analysis_stage_tampering_fails_through_existing_verifier(
    tmp_path: Path,
) -> None:
    embedding_path = tmp_path / "embedding.npy"
    cell_ids_path = tmp_path / "cells.txt"
    embeddings = np.zeros((5, 512), dtype=np.float32)
    embeddings[:, 0] = np.arange(5, dtype=np.float32)
    np.save(embedding_path, embeddings, allow_pickle=False)
    cell_ids_path.write_text(
        "cell-1\ncell-2\ncell-3\ncell-4\ncell-5\n", encoding="utf-8"
    )
    arguments = {
        "embedding_path": str(embedding_path),
        "cell_ids_path": str(cell_ids_path),
        "output_dir": str(tmp_path / "analysis"),
        "n_neighbors": 2,
    }
    result = build_cell_neighbors(**arguments)
    registry = build_default_tool_registry()
    run = _single_step_run(
        PlanStep("neighbors", "build_cell_neighbors", arguments),
        arguments,
        result,
        registry,
        request_id="neighbors-evidence",
    )
    build_analysis_evidence(run, tmp_path / "before-tamper", registry=registry)

    artifact = ad.read_h5ad(result["analysis_path"])
    artifact.uns["agent_milestone6"]["stage"] = "tampered"
    artifact.write_h5ad(result["analysis_path"])
    with pytest.raises(AnalysisEvidenceError) as caught:
        build_analysis_evidence(run, tmp_path / "after-tamper", registry=registry)
    assert caught.value.code == "EVIDENCE_SOURCE_STEP_REVALIDATION_FAILED"


def test_label_transfer_digest_tampering_fails_through_existing_verifier(
    tmp_path: Path,
) -> None:
    reference_embedding = tmp_path / "reference.npy"
    query_embedding = tmp_path / "query.npy"
    reference_ids = tmp_path / "reference.txt"
    query_ids = tmp_path / "query.txt"
    reference_h5ad = tmp_path / "reference.h5ad"
    query_h5ad = tmp_path / "query.h5ad"
    checkpoint = tmp_path / "checkpoint.pth"
    reference_values = np.zeros((2, 512), dtype=np.float32)
    reference_values[1, 0] = 10.0
    query_values = np.zeros((1, 512), dtype=np.float32)
    np.save(reference_embedding, reference_values, allow_pickle=False)
    np.save(query_embedding, query_values, allow_pickle=False)
    reference_ids.write_text("ref-1\nref-2\n", encoding="utf-8")
    query_ids.write_text("query-1\n", encoding="utf-8")
    reference = ad.AnnData(obs={"celltype": ["A", "B"]})
    reference.obs_names = ["ref-1", "ref-2"]
    reference.write_h5ad(reference_h5ad)
    query = ad.AnnData(obs=pd.DataFrame(index=["query-1"]))
    query.write_h5ad(query_h5ad)
    checkpoint.write_bytes(b"checkpoint-identity")
    arguments = {
        "reference_embedding_path": str(reference_embedding),
        "reference_cell_ids_path": str(reference_ids),
        "reference_h5ad_path": str(reference_h5ad),
        "reference_label_key": "celltype",
        "query_embedding_path": str(query_embedding),
        "query_cell_ids_path": str(query_ids),
        "query_h5ad_path": str(query_h5ad),
        "output_dir": str(tmp_path / "transfer"),
        "reference_species": "mouse",
        "query_species": "mouse",
        "reference_checkpoint_path": str(checkpoint),
        "query_checkpoint_path": str(checkpoint),
        "n_neighbors": 1,
    }
    result = transfer_cell_labels(**arguments)
    registry = build_default_tool_registry()
    run = _single_step_run(
        PlanStep("transfer", "transfer_cell_labels", arguments),
        arguments,
        result,
        registry,
        request_id="transfer-evidence",
    )
    build_analysis_evidence(run, tmp_path / "before-transfer-tamper", registry=registry)
    annotation = Path(result["annotation_path"])
    annotation.write_bytes(annotation.read_bytes() + b"tamper")
    with pytest.raises(AnalysisEvidenceError) as caught:
        build_analysis_evidence(
            run, tmp_path / "after-transfer-tamper", registry=registry
        )
    assert caught.value.code == "EVIDENCE_SOURCE_STEP_REVALIDATION_FAILED"


def test_evaluation_report_tampering_fails_through_existing_verifier(
    tmp_path: Path,
) -> None:
    embedding_path = tmp_path / "eval_embedding.npy"
    cell_ids_path = tmp_path / "eval_cells.txt"
    ids = tuple(f"cell-{index}" for index in range(8))
    embeddings = np.zeros((8, 512), dtype=np.float32)
    embeddings[:4, 0] = np.arange(4, dtype=np.float32) * 0.01
    embeddings[4:, 0] = 10.0 + np.arange(4, dtype=np.float32) * 0.01
    np.save(embedding_path, embeddings, allow_pickle=False)
    cell_ids_path.write_text("\n".join(ids) + "\n", encoding="utf-8")
    neighbors = build_cell_neighbors(
        embedding_path,
        cell_ids_path,
        tmp_path / "eval-analysis",
        n_neighbors=3,
    )
    clustered = cluster_cells(
        neighbors["analysis_path"], tmp_path / "eval-analysis"
    )
    reference_path = tmp_path / "evaluation_reference.h5ad"
    reference = ad.AnnData(obs={"celltype": ["A"] * 4 + ["B"] * 4})
    reference.obs_names = list(ids)
    reference.write_h5ad(reference_path)
    arguments = {
        "analysis_path": clustered["analysis_path"],
        "reference_h5ad_path": str(reference_path),
        "label_key": "celltype",
        "output_dir": str(tmp_path / "evaluation"),
    }
    result = evaluate_cell_clustering(**arguments)
    registry = build_default_tool_registry()
    run = _single_step_run(
        PlanStep("evaluate", "evaluate_cell_clustering", arguments),
        arguments,
        result,
        registry,
        request_id="clustering-evaluation-evidence",
    )
    build_analysis_evidence(run, tmp_path / "before-report-tamper", registry=registry)
    report_path = Path(result["report_path"])
    report = json.loads(report_path.read_text())
    report["metrics"]["nmi"] = 0.123
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(AnalysisEvidenceError) as caught:
        build_analysis_evidence(run, tmp_path / "after-report-tamper", registry=registry)
    assert caught.value.code == "EVIDENCE_SOURCE_STEP_REVALIDATION_FAILED"


def test_production_registry_membership_is_exactly_eleven_after_milestone8_2() -> None:
    assert build_default_tool_registry().names() == (
        "inspect_scATAC",
        "epizoo_embed_cells",
        "build_cell_neighbors",
        "cluster_cells",
        "compute_cell_umap",
        "evaluate_cell_clustering",
        "transfer_cell_labels",
        "evaluate_cell_annotation",
        "validate_scATAC_feature_space",
        "build_replicate_pseudobulk",
        "run_replicate_differential_accessibility",
    )
