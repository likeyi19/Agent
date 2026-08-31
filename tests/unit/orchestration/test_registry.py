"""Unit tests for the explicit scientific-tool allowlist."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from agent.orchestration import (
    ErrorCategory,
    StepOutputRef,
    ToolArgumentError,
    ToolResultContractError,
    UnknownToolError,
    build_default_tool_registry,
)


@pytest.fixture
def registry():
    return build_default_tool_registry()


def test_default_registry_contains_exact_allowlist(registry) -> None:
    assert registry.names() == (
        "inspect_scATAC",
        "epizoo_embed_cells",
        "build_cell_neighbors",
        "cluster_cells",
        "compute_cell_umap",
        "evaluate_cell_clustering",
    )
    assert registry.contains("inspect_scATAC")
    assert registry.contains("epizoo_embed_cells")
    assert registry.contains("build_cell_neighbors")
    assert registry.contains("cluster_cells")
    assert registry.contains("compute_cell_umap")
    assert registry.contains("evaluate_cell_clustering")
    assert not registry.contains("arbitrary_python")


def test_unknown_tool_is_rejected(registry) -> None:
    with pytest.raises(UnknownToolError, match="executable allowlist"):
        registry.get("arbitrary_python")


def test_valid_inspection_arguments(registry) -> None:
    validated = registry.validate_arguments(
        "inspect_scATAC", {"path": Path("input.h5ad")}
    )
    assert validated == {"path": Path("input.h5ad")}


def test_missing_inspection_argument(registry) -> None:
    with pytest.raises(ToolArgumentError, match="missing required"):
        registry.validate_arguments("inspect_scATAC", {})


def test_unknown_inspection_argument(registry) -> None:
    with pytest.raises(ToolArgumentError, match="unknown arguments"):
        registry.validate_arguments(
            "inspect_scATAC", {"path": "input.h5ad", "execute": "shell"}
        )


def test_valid_embedding_arguments(registry) -> None:
    validated = registry.validate_arguments(
        "epizoo_embed_cells",
        {
            "input_path": "input.h5ad",
            "output_dir": "/tmp/output",
            "species": "mouse",
        },
    )
    assert validated["species"] == "mouse"


def test_missing_required_embedding_arguments(registry) -> None:
    with pytest.raises(ToolArgumentError, match="output_dir"):
        registry.validate_arguments(
            "epizoo_embed_cells",
            {"input_path": "input.h5ad", "species": "mouse"},
        )


def test_invalid_species(registry) -> None:
    with pytest.raises(ToolArgumentError, match="must be one of"):
        registry.validate_arguments(
            "epizoo_embed_cells",
            {
                "input_path": "input.h5ad",
                "output_dir": "/tmp/output",
                "species": "rat",
            },
        )


def test_wrong_primitive_argument_type(registry) -> None:
    with pytest.raises(ToolArgumentError, match="must have type"):
        registry.validate_arguments("inspect_scATAC", {"path": 123})
    with pytest.raises(ToolArgumentError, match="must have type bool"):
        registry.validate_arguments(
            "epizoo_embed_cells",
            {
                "input_path": "input.h5ad",
                "output_dir": "/tmp/output",
                "species": "human",
                "overwrite": 1,
            },
        )


def test_optional_embedding_arguments(registry) -> None:
    validated = registry.validate_arguments(
        "epizoo_embed_cells",
        {
            "input_path": "input.h5ad",
            "output_dir": "/tmp/output",
            "species": "human",
            "checkpoint_path": Path("checkpoint.pth"),
            "device": "cuda:0",
            "overwrite": False,
        },
    )
    assert validated["device"] == "cuda:0"


def test_step_output_ref_is_accepted_before_resolution(registry) -> None:
    reference = StepOutputRef("inspect", "input_path")
    validated = registry.validate_arguments(
        "epizoo_embed_cells",
        {
            "input_path": reference,
            "output_dir": "/tmp/output",
            "species": "mouse",
        },
    )
    assert validated["input_path"] is reference


def test_valid_downstream_arguments_and_choices(registry) -> None:
    neighbors = registry.validate_arguments(
        "build_cell_neighbors",
        {
            "embedding_path": "embedding.npy",
            "cell_ids_path": "ids.txt",
            "output_dir": "output",
            "n_neighbors": 15,
            "metric": "cosine",
            "random_seed": 0,
            "overwrite": False,
        },
    )
    assert neighbors["metric"] == "cosine"
    registry.validate_arguments(
        "cluster_cells",
        {"analysis_path": "neighbors.h5ad", "output_dir": "output", "resolution": 1},
    )
    registry.validate_arguments(
        "compute_cell_umap",
        {
            "analysis_path": "clustered.h5ad",
            "output_dir": "output",
            "min_dist": 0.5,
            "spread": 1,
        },
    )
    evaluation = registry.validate_arguments(
        "evaluate_cell_clustering",
        {
            "analysis_path": StepOutputRef("cluster", "analysis_path"),
            "reference_h5ad_path": StepOutputRef("inspect", "input_path"),
            "label_key": "celltype",
            "output_dir": "output",
        },
    )
    assert "cluster_key" not in evaluation
    spec = registry.get("evaluate_cell_clustering")
    assert spec.recovery_policy_version == "evaluate-cell-clustering-v1"
    assert spec.retryable_error_codes == frozenset()
    with pytest.raises(ToolArgumentError, match="must be one of"):
        registry.validate_arguments(
            "build_cell_neighbors",
            {
                "embedding_path": "embedding.npy",
                "cell_ids_path": "ids.txt",
                "output_dir": "output",
                "metric": "manhattan",
            },
        )


def test_registry_contracts_match_current_public_signatures(registry) -> None:
    for name in registry.names():
        spec = registry.get(name)
        parameters = inspect.signature(spec.function).parameters
        assert set(parameters) == set(spec.required_arguments) | set(
            spec.optional_arguments
        )
        expected_required = {
            argument_name
            for argument_name, parameter in parameters.items()
            if parameter.default is inspect.Parameter.empty
        }
        assert expected_required == set(spec.required_arguments)


def test_result_contracts_validate_lightweight_real_shapes(registry) -> None:
    inspection = {
        "input_path": "/data/input.h5ad",
        "n_cells": 2,
        "n_features": 3,
        "x_storage_type": "anndata._core.sparse_dataset._CSRDataset",
        "x_is_sparse": True,
        "x_dtype": "float32",
        "nnz": 4,
        "density": 2 / 3,
        "obs_columns": [],
        "var_columns": [],
        "obs_names_sample": ["cell-1", "cell-2"],
        "var_names_sample": ["peak-1", "peak-2", "peak-3"],
    }
    embedding = {
        "status": "success",
        "input_path": "/data/input.h5ad",
        "embedding_path": "/output/embeddings.npy",
        "cell_ids_path": "/output/obs_names.txt",
        "n_cells": 2,
        "embedding_dim": 512,
        "embedding_dtype": "float32",
        "finite": True,
        "cell_order_preserved": True,
        "backend": "EpiZoo",
        "species": "mouse",
        "checkpoint_path": "/models/epizoo.pth",
        "device": "cuda:0",
    }
    versions = {"scanpy": "1.11.5"}
    neighbors = {
        "status": "success",
        "embedding_path": "/output/embeddings.npy",
        "cell_ids_path": "/output/obs_names.txt",
        "analysis_path": "/output/neighbors.h5ad",
        "n_cells": 2,
        "embedding_dim": 512,
        "n_neighbors": 1,
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
    }
    clustering = {
        "status": "success",
        "input_analysis_path": "/output/neighbors.h5ad",
        "analysis_path": "/output/clustered.h5ad",
        "n_cells": 2,
        "n_clusters": 1,
        "cluster_key": "leiden",
        "algorithm": "leiden",
        "resolution": 1.0,
        "random_seed": 0,
        "cell_order_preserved": True,
        "backend": "Scanpy",
        "software_versions": versions,
    }
    umap = {
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
    }
    evaluation = {
        "status": "success",
        "analysis_path": "/output/clustered.h5ad",
        "reference_h5ad_path": "/data/input.h5ad",
        "report_path": "/output/clustered.clustering_metrics.json",
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
        "software_versions": {"scikit_learn": "1.9.0"},
    }

    registry.validate_result("inspect_scATAC", inspection)
    registry.validate_result("epizoo_embed_cells", embedding)
    registry.validate_result("build_cell_neighbors", neighbors)
    registry.validate_result("cluster_cells", clustering)
    registry.validate_result("compute_cell_umap", umap)
    registry.validate_result("evaluate_cell_clustering", evaluation)
    assert registry.get("inspect_scATAC").result_contract.name == "ScATACInspection"
    assert (
        registry.get("epizoo_embed_cells").result_contract.name
        == "EpiZooEmbeddingToolResult"
    )
    assert registry.get("build_cell_neighbors").result_contract.name == (
        "CellNeighborsToolResult"
    )


def test_result_contract_rejects_missing_or_heavy_embedding_result(registry) -> None:
    with pytest.raises(ToolResultContractError, match="missing required fields"):
        registry.validate_result("inspect_scATAC", {})

    result = {
        key: ({"status": "success"}.get(key, expected_types[0]()))
        for key, expected_types in registry.get(
            "epizoo_embed_cells"
        ).result_contract.required_fields.items()
    }
    result.update(
        {
            "status": "success",
            "n_cells": 1,
            "embedding_dim": 512,
            "finite": True,
            "cell_order_preserved": True,
            "embeddings": [[0.0]],
        }
    )
    with pytest.raises(ToolResultContractError, match="must not return"):
        registry.validate_result("epizoo_embed_cells", result)


def test_evaluation_result_contract_rejects_invalid_metric_range(registry) -> None:
    result = {
        key: expected_types[0]()
        for key, expected_types in registry.get(
            "evaluate_cell_clustering"
        ).result_contract.required_fields.items()
    }
    result.update(
        {
            "status": "success",
            "n_cells": 2,
            "n_reference_classes": 2,
            "n_predicted_clusters": 1,
            "nmi": 1.1,
            "ari": 0.0,
            "ami": 0.0,
            "homogeneity": 0.0,
            "finite": True,
            "cell_order_preserved": True,
            "metric_backend": "scikit-learn",
            "average_method": "arithmetic",
            "report_schema_version": 1,
        }
    )
    with pytest.raises(ToolResultContractError, match="valid range"):
        registry.validate_result("evaluate_cell_clustering", result)


@pytest.mark.parametrize(
    ("exception", "category", "code"),
    [
        (FileNotFoundError("missing"), ErrorCategory.RESOURCE_ERROR, "RESOURCE_NOT_FOUND"),
        (FileExistsError("exists"), ErrorCategory.USER_INPUT_ERROR, "OUTPUT_CONFLICT"),
        (TypeError("bad type"), ErrorCategory.TOOL_EXECUTION_ERROR, "TOOL_EXCEPTION"),
        (ValueError("bad value"), ErrorCategory.TOOL_EXECUTION_ERROR, "TOOL_EXCEPTION"),
        (
            RuntimeError(
                "CUDA device cuda:0 was requested but CUDA is not available in this runtime."
            ),
            ErrorCategory.ENVIRONMENT_ERROR,
            "CUDA_UNAVAILABLE",
        ),
        (
            RuntimeError("scientific execution failed"),
            ErrorCategory.TOOL_EXECUTION_ERROR,
            "TOOL_RUNTIME_ERROR",
        ),
        (KeyError("backend bug"), ErrorCategory.TOOL_EXECUTION_ERROR, "TOOL_EXCEPTION"),
    ],
)
def test_builtin_exception_classification(registry, exception, category, code) -> None:
    error = registry.classify_exception(
        "epizoo_embed_cells", exception, step_id="embed", attempt=1
    )

    assert error.category is category
    assert error.code == code
    assert error.tool_name == "epizoo_embed_cells"
    assert error.step_id == "embed"
    assert error.recoverable is False


@pytest.mark.parametrize("exception", [TypeError("bad type"), ValueError("bad value")])
def test_analysis_validation_exception_is_classified_as_invalid_argument(
    registry, exception
) -> None:
    error = registry.classify_exception("build_cell_neighbors", exception)
    assert error.category is ErrorCategory.USER_INPUT_ERROR
    assert error.code == "INVALID_ARGUMENT"
    assert error.recoverable is False


def test_real_tools_are_non_retryable(registry) -> None:
    assert all(
        not registry.get(name).retryable_error_codes for name in registry.names()
    )


def test_default_allowlist_cannot_be_mutated(registry) -> None:
    assert not hasattr(registry, "register")
    with pytest.raises(AttributeError):
        registry._specs = {}  # type: ignore[attr-defined]
    with pytest.raises(TypeError):
        registry.get("inspect_scATAC").required_arguments["extra"] = None  # type: ignore[index]
