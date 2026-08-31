"""Synthetic tests for lightweight orchestration verification."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import anndata as ad
import numpy as np
import pytest

from agent.orchestration import (
    AgentError,
    AgentPlan,
    ErrorCategory,
    PlanStep,
    StepExecutionResult,
    StepStatus,
    ToolRegistry,
    VerificationCheck,
    VerificationResult,
    build_default_tool_registry,
    verify_run,
    verify_step,
)
from agent.tools import build_cell_neighbors, cluster_cells, compute_cell_umap


@pytest.fixture
def registry() -> ToolRegistry:
    return build_default_tool_registry()


def _inspection_result(path: str, *, n_cells: int = 2) -> dict[str, object]:
    n_features = 4
    nnz = n_cells * 2
    return {
        "input_path": path,
        "n_cells": n_cells,
        "n_features": n_features,
        "x_storage_type": "anndata._core.sparse_dataset._CSRDataset",
        "x_is_sparse": True,
        "x_dtype": "float32",
        "nnz": nnz,
        "density": nnz / (n_cells * n_features),
        "obs_columns": [],
        "var_columns": [],
        "obs_names_sample": ["cell-1", "cell-2"],
        "var_names_sample": ["feature-1", "feature-2"],
    }


def _embedding_result(
    input_path: str,
    embedding_path: str,
    cell_ids_path: str,
    *,
    n_cells: int = 2,
) -> dict[str, object]:
    return {
        "status": "success",
        "input_path": input_path,
        "embedding_path": embedding_path,
        "cell_ids_path": cell_ids_path,
        "n_cells": n_cells,
        "embedding_dim": 512,
        "embedding_dtype": "float32",
        "finite": True,
        "cell_order_preserved": True,
        "backend": "EpiZoo",
        "species": "mouse",
        "checkpoint_path": "/models/epizoo.pth",
        "device": "cuda:0",
    }


def _inspection_step() -> PlanStep:
    return PlanStep("inspect", "inspect_scATAC", {"path": "/data/input.h5ad"})


def _embedding_step(*, with_dependency: bool = False) -> PlanStep:
    return PlanStep(
        "embed",
        "epizoo_embed_cells",
        {
            "input_path": "/data/input.h5ad",
            "output_dir": "/output",
            "species": "mouse",
        },
        ("inspect",) if with_dependency else (),
    )


def _artifacts(tmp_path):
    embedding_path = tmp_path / "embeddings.npy"
    cell_ids_path = tmp_path / "obs_names.txt"
    embedding_path.write_bytes(b"npy artifact")
    cell_ids_path.write_text("cell-1\ncell-2\n", encoding="utf-8")
    return embedding_path, cell_ids_path


def _downstream_artifacts(tmp_path: Path):
    rng = np.random.default_rng(11)
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
    neighbors = build_cell_neighbors(
        embedding_path, cell_ids_path, tmp_path / "neighbors"
    )
    clustering = cluster_cells(
        neighbors["analysis_path"], tmp_path / "clustering"
    )
    umap = compute_cell_umap(clustering["analysis_path"], tmp_path / "umap")
    return embedding_path, cell_ids_path, neighbors, clustering, umap


def test_valid_downstream_artifacts_pass_explicit_verification(
    registry, tmp_path: Path
) -> None:
    embedding_path, cell_ids_path, neighbors, clustering, umap = (
        _downstream_artifacts(tmp_path)
    )
    neighbors_step = PlanStep(
        "neighbors",
        "build_cell_neighbors",
        {
            "embedding_path": str(embedding_path),
            "cell_ids_path": str(cell_ids_path),
            "output_dir": str(tmp_path / "neighbors"),
        },
    )
    cluster_step = PlanStep(
        "cluster",
        "cluster_cells",
        {
            "analysis_path": neighbors["analysis_path"],
            "output_dir": str(tmp_path / "clustering"),
        },
    )
    umap_step = PlanStep(
        "umap",
        "compute_cell_umap",
        {
            "analysis_path": clustering["analysis_path"],
            "output_dir": str(tmp_path / "umap"),
        },
    )
    assert verify_step(
        neighbors_step, neighbors_step.arguments, neighbors, registry
    ).passed
    assert verify_step(cluster_step, cluster_step.arguments, clustering, registry).passed
    assert verify_step(umap_step, umap_step.arguments, umap, registry).passed


def test_corrupted_umap_artifact_fails_explicit_verification(
    registry, tmp_path: Path
) -> None:
    _, _, _, clustering, umap = _downstream_artifacts(tmp_path)
    artifact = ad.read_h5ad(umap["analysis_path"])
    artifact.obsm["X_umap"][0, 0] = np.nan
    artifact.write_h5ad(umap["analysis_path"])
    step = PlanStep(
        "umap",
        "compute_cell_umap",
        {
            "analysis_path": clustering["analysis_path"],
            "output_dir": str(tmp_path / "umap"),
        },
    )
    result = verify_step(step, step.arguments, umap, registry)
    assert not result.passed
    assert result.error is not None
    assert "umap_artifact_structure" in result.error.details["failed_checks"]


def test_valid_inspection_result_passes(registry) -> None:
    result = verify_step(
        _inspection_step(),
        {"path": "/data/input.h5ad"},
        _inspection_result("/data/input.h5ad"),
        registry,
    )

    assert result.passed
    assert result.error is None
    assert {check.name for check in result.checks} >= {
        "tool_registered",
        "result_contract",
        "input_path_matches",
        "sparse_metadata_coherent",
    }


def test_malformed_inspection_result_fails_contract(registry) -> None:
    malformed = _inspection_result("/data/input.h5ad")
    del malformed["x_dtype"]

    result = verify_step(
        _inspection_step(), {"path": "/data/input.h5ad"}, malformed, registry
    )

    assert not result.passed
    assert result.error is not None
    assert result.error.code == "RESULT_CONTRACT_INVALID"


def test_inspection_path_mismatch_fails(registry) -> None:
    result = verify_step(
        _inspection_step(),
        {"path": "/data/expected.h5ad"},
        _inspection_result("/data/other.h5ad"),
        registry,
    )

    assert not result.passed
    assert result.error is not None
    assert result.error.code == "RESULT_PATH_MISMATCH"


@pytest.mark.parametrize(
    ("field", "value"),
    [("n_cells", 0), ("n_features", -1), ("nnz", -1), ("density", 1.1)],
)
def test_invalid_inspection_counts_or_density_fail(registry, field, value) -> None:
    inspection = _inspection_result("/data/input.h5ad")
    inspection[field] = value

    result = verify_step(
        _inspection_step(),
        {"path": "/data/input.h5ad"},
        inspection,
        registry,
    )

    assert not result.passed
    assert result.error is not None
    assert result.error.category is ErrorCategory.VERIFICATION_ERROR


def test_valid_embedding_result_passes(registry, tmp_path) -> None:
    embedding_path, cell_ids_path = _artifacts(tmp_path)
    result = verify_step(
        _embedding_step(),
        {"input_path": "/data/input.h5ad"},
        _embedding_result(
            "/data/input.h5ad", str(embedding_path), str(cell_ids_path)
        ),
        registry,
    )

    assert result.passed


def test_embedding_input_path_mismatch_fails(registry, tmp_path) -> None:
    embedding_path, cell_ids_path = _artifacts(tmp_path)
    result = verify_step(
        _embedding_step(),
        {"input_path": "/data/expected.h5ad"},
        _embedding_result(
            "/data/other.h5ad", str(embedding_path), str(cell_ids_path)
        ),
        registry,
    )

    assert not result.passed
    assert result.error is not None
    assert result.error.code == "RESULT_PATH_MISMATCH"


def test_embedding_species_identity_mismatch_fails(registry, tmp_path) -> None:
    embedding_path, cell_ids_path = _artifacts(tmp_path)
    embedding = _embedding_result(
        "/data/input.h5ad", str(embedding_path), str(cell_ids_path)
    )
    embedding["species"] = "human"

    result = verify_step(
        _embedding_step(),
        {"input_path": "/data/input.h5ad", "species": "mouse"},
        embedding,
        registry,
    )

    assert not result.passed
    assert result.error is not None
    assert result.error.code == "RESULT_IDENTITY_MISMATCH"


@pytest.mark.parametrize("missing_field", ["embedding_path", "cell_ids_path"])
def test_missing_embedding_artifact_fails(
    registry, tmp_path, missing_field
) -> None:
    embedding_path, cell_ids_path = _artifacts(tmp_path)
    paths = {
        "embedding_path": str(embedding_path),
        "cell_ids_path": str(cell_ids_path),
    }
    missing_path = tmp_path / f"missing-{missing_field}"
    paths[missing_field] = str(missing_path)
    result = verify_step(
        _embedding_step(),
        {"input_path": "/data/input.h5ad"},
        _embedding_result(
            "/data/input.h5ad", paths["embedding_path"], paths["cell_ids_path"]
        ),
        registry,
    )

    assert not result.passed
    assert result.error is not None
    assert result.error.code == "ARTIFACT_MISSING"


@pytest.mark.parametrize("empty_field", ["embedding_path", "cell_ids_path"])
def test_empty_artifact_fails(registry, tmp_path, empty_field) -> None:
    embedding_path, cell_ids_path = _artifacts(tmp_path)
    path = embedding_path if empty_field == "embedding_path" else cell_ids_path
    path.write_bytes(b"")

    result = verify_step(
        _embedding_step(),
        {"input_path": "/data/input.h5ad"},
        _embedding_result(
            "/data/input.h5ad", str(embedding_path), str(cell_ids_path)
        ),
        registry,
    )

    assert not result.passed
    assert result.error is not None
    assert result.error.code == "ARTIFACT_EMPTY"


@pytest.mark.parametrize("field", ["finite", "cell_order_preserved"])
def test_embedding_boolean_invariant_failure(registry, tmp_path, field) -> None:
    embedding_path, cell_ids_path = _artifacts(tmp_path)
    embedding = _embedding_result(
        "/data/input.h5ad", str(embedding_path), str(cell_ids_path)
    )
    embedding[field] = False

    result = verify_step(
        _embedding_step(),
        {"input_path": "/data/input.h5ad"},
        embedding,
        registry,
    )

    assert not result.passed


def test_invalid_embedding_contract_fails(registry, tmp_path) -> None:
    embedding_path, cell_ids_path = _artifacts(tmp_path)
    embedding = _embedding_result(
        "/data/input.h5ad", str(embedding_path), str(cell_ids_path)
    )
    del embedding["device"]

    result = verify_step(
        _embedding_step(),
        {"input_path": "/data/input.h5ad"},
        embedding,
        registry,
    )

    assert not result.passed
    assert result.error is not None
    assert result.error.code == "RESULT_CONTRACT_INVALID"


def test_forbidden_embedding_payload_fails_registry_contract(registry, tmp_path) -> None:
    embedding_path, cell_ids_path = _artifacts(tmp_path)
    embedding = _embedding_result(
        "/data/input.h5ad", str(embedding_path), str(cell_ids_path)
    )
    embedding["embeddings"] = [[0.0] * 512]

    result = verify_step(
        _embedding_step(),
        {"input_path": "/data/input.h5ad"},
        embedding,
        registry,
    )

    assert not result.passed
    assert result.error is not None
    assert result.error.code == "RESULT_CONTRACT_INVALID"


def test_valid_inspect_to_embed_cross_step_verification(registry, tmp_path) -> None:
    embedding_path, cell_ids_path = _artifacts(tmp_path)
    inspection = _inspection_result("/data/input.h5ad")
    result = verify_step(
        _embedding_step(with_dependency=True),
        {"input_path": "/data/input.h5ad"},
        _embedding_result(
            "/data/input.h5ad", str(embedding_path), str(cell_ids_path)
        ),
        registry,
        dependency_results={"inspect": inspection},
    )

    assert result.passed


def test_cross_step_cell_count_mismatch_fails(registry, tmp_path) -> None:
    embedding_path, cell_ids_path = _artifacts(tmp_path)
    result = verify_step(
        _embedding_step(with_dependency=True),
        {"input_path": "/data/input.h5ad"},
        _embedding_result(
            "/data/input.h5ad",
            str(embedding_path),
            str(cell_ids_path),
            n_cells=3,
        ),
        registry,
        dependency_results={
            "inspect": _inspection_result("/data/input.h5ad", n_cells=2)
        },
    )

    assert not result.passed
    assert result.error is not None
    assert result.error.code == "CELL_COUNT_MISMATCH"


def test_cross_step_path_mismatch_fails(registry, tmp_path) -> None:
    embedding_path, cell_ids_path = _artifacts(tmp_path)
    result = verify_step(
        _embedding_step(with_dependency=True),
        {"input_path": "/data/embed.h5ad"},
        _embedding_result(
            "/data/embed.h5ad", str(embedding_path), str(cell_ids_path)
        ),
        registry,
        dependency_results={"inspect": _inspection_result("/data/inspect.h5ad")},
    )

    assert not result.passed
    assert result.error is not None
    assert result.error.code == "RESULT_PATH_MISMATCH"


def test_unknown_tool_fails_safely(registry) -> None:
    result = verify_step(
        PlanStep("unknown", "arbitrary_python", {}), {}, {}, registry
    )

    assert not result.passed
    assert result.error is not None
    assert result.error.category is ErrorCategory.VERIFICATION_ERROR
    assert result.error.code == "UNKNOWN_TOOL"


def test_failure_contains_named_checks_and_structured_error(registry) -> None:
    result = verify_step(
        _inspection_step(),
        {"path": "/expected.h5ad"},
        _inspection_result("/other.h5ad"),
        registry,
    )

    assert any(check.name == "input_path_matches" for check in result.checks)
    assert all(isinstance(check, VerificationCheck) for check in result.checks)
    assert isinstance(result.error, AgentError)
    assert result.error.category is ErrorCategory.VERIFICATION_ERROR


def _plan() -> AgentPlan:
    return AgentPlan(
        "plan-1",
        "request-1",
        "deterministic",
        (_inspection_step(), _embedding_step(with_dependency=True)),
    )


def _passed_verification(step_id: str) -> VerificationResult:
    return VerificationResult(
        True,
        "step",
        step_id,
        (VerificationCheck("contract", True, "Passed."),),
    )


def _successful_step(step_id: str, tool_name: str) -> StepExecutionResult:
    return StepExecutionResult(
        step_id,
        tool_name,
        StepStatus.SUCCEEDED,
        attempt_count=1,
        result={"status": "success"},
        verification=_passed_verification(step_id),
    )


def _failed_step(step_id: str, tool_name: str) -> StepExecutionResult:
    return StepExecutionResult(
        step_id,
        tool_name,
        StepStatus.FAILED,
        attempt_count=1,
        error=AgentError(
            ErrorCategory.TOOL_EXECUTION_ERROR,
            "TOOL_RUNTIME_ERROR",
            "Tool failed.",
            step_id=step_id,
            tool_name=tool_name,
        ),
    )


def test_valid_run_level_verification_passes() -> None:
    result = verify_run(
        _plan(),
        (
            _successful_step("inspect", "inspect_scATAC"),
            _successful_step("embed", "epizoo_embed_cells"),
        ),
    )

    assert result.passed
    assert result.error is None


def test_missing_step_result_fails_run_verification() -> None:
    result = verify_run(
        _plan(), (_successful_step("inspect", "inspect_scATAC"),)
    )

    assert not result.passed
    assert result.error is not None
    assert result.error.code == "MISSING_STEP_RESULT"


@pytest.mark.parametrize("mode", ["duplicate", "unexpected"])
def test_duplicate_or_unexpected_step_result_fails(mode) -> None:
    inspect_result = _successful_step("inspect", "inspect_scATAC")
    results = [inspect_result, _successful_step("embed", "epizoo_embed_cells")]
    if mode == "duplicate":
        results.append(inspect_result)
    else:
        results.append(_successful_step("extra", "inspect_scATAC"))

    result = verify_run(_plan(), tuple(results))

    assert not result.passed
    assert result.error is not None
    assert result.error.code in {"DUPLICATE_STEP_RESULT", "UNEXPECTED_STEP_RESULT"}


def test_failed_step_causes_run_verification_failure() -> None:
    result = verify_run(
        _plan(),
        (
            _failed_step("inspect", "inspect_scATAC"),
            StepExecutionResult(
                "embed", "epizoo_embed_cells", StepStatus.SKIPPED
            ),
        ),
    )

    assert not result.passed
    assert any(not check.passed for check in result.checks)


def test_successful_step_without_passed_verification_fails() -> None:
    inspect_result = StepExecutionResult(
        "inspect",
        "inspect_scATAC",
        StepStatus.SUCCEEDED,
        result={"ok": True},
    )
    result = verify_run(
        _plan(),
        (inspect_result, _successful_step("embed", "epizoo_embed_cells")),
    )

    assert not result.passed
    assert any(
        check.name == "inspect_verification_passed" and not check.passed
        for check in result.checks
    )


def test_failed_verification_may_preserve_lightweight_result() -> None:
    verification_error = AgentError(
        ErrorCategory.VERIFICATION_ERROR,
        "RESULT_CONTRACT_INVALID",
        "Result verification failed.",
        step_id="inspect",
        tool_name="inspect_scATAC",
    )
    failed_verification = VerificationResult(
        False,
        "step",
        "inspect",
        (VerificationCheck("contract", False, "Failed."),),
        verification_error,
    )
    inspect_result = StepExecutionResult(
        "inspect",
        "inspect_scATAC",
        StepStatus.FAILED,
        attempt_count=1,
        result={"lightweight": True},
        verification=failed_verification,
        error=verification_error,
    )

    result = verify_run(
        _plan(),
        (inspect_result, _successful_step("embed", "epizoo_embed_cells")),
    )

    assert any(
        check.name == "inspect_status_consistent" and check.passed
        for check in result.checks
    )


def test_dependency_completion_inconsistency_fails() -> None:
    result = verify_run(
        _plan(),
        (
            _failed_step("inspect", "inspect_scATAC"),
            _successful_step("embed", "epizoo_embed_cells"),
        ),
    )

    assert not result.passed
    assert any(
        check.name == "embed_dependencies_consistent" and not check.passed
        for check in result.checks
    )


def test_verifier_never_invokes_scientific_tools(registry, tmp_path) -> None:
    inspect_callable = Mock(side_effect=AssertionError("must not execute"))
    embedding_callable = Mock(side_effect=AssertionError("must not execute"))
    guarded_registry = ToolRegistry(
        (
            replace(registry.get("inspect_scATAC"), function=inspect_callable),
            replace(
                registry.get("epizoo_embed_cells"), function=embedding_callable
            ),
        )
    )
    embedding_path, cell_ids_path = _artifacts(tmp_path)

    verify_step(
        _inspection_step(),
        {"path": "/data/input.h5ad"},
        _inspection_result("/data/input.h5ad"),
        guarded_registry,
    )
    verify_step(
        _embedding_step(),
        {"input_path": "/data/input.h5ad"},
        _embedding_result(
            "/data/input.h5ad", str(embedding_path), str(cell_ids_path)
        ),
        guarded_registry,
    )

    inspect_callable.assert_not_called()
    embedding_callable.assert_not_called()
