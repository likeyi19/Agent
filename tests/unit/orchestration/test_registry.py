"""Unit tests for the explicit scientific-tool allowlist."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from agent.tools.analysis.differential_accessibility import M82ScientificError
from agent.orchestration import (
    ErrorCategory,
    RecoveryDisposition,
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
        "invoke_native_function",
        "inspect_scATAC",
        "compute_motif_accessibility",
        "run_motifmatchr",
        "run_chromvar",
        "run_episcanpy",
        "run_snapatac2",
        "run_signac",
        "run_scanpy",
        "run_chromap",
        "run_macs3",
        "run_cicero",
        "run_cicero_cds",
        "run_monalisa",
        "run_archr",
        "call_scATAC_peaks",
        "epizoo_embed_cells",
        "evaluate_cell_clustering",
        "transfer_cell_labels",

        "validate_scATAC_feature_space",
        "build_replicate_pseudobulk",
        "run_replicate_differential_accessibility",
        "inspect_raw_scATAC",
        "inspect_scATAC_fragments",
        "adopt_scATAC_cell_by_ccre",
        "annotate_scATAC_cell_types",


        "adopt_scATAC_cell_by_features",
        "adapt_epizoo_species",
        "posttrain_epizoo_model",
        "predict_regulatory_sequences",
        "evaluate_regulatory_predictions",
        "optimize_regulatory_sequences",
        "generate_regulatory_sequences",
        "prepare_scatac_regulatory_sequences",
    )
    assert registry.contains("inspect_scATAC")
    assert registry.contains("epizoo_embed_cells")
    assert not registry.contains("build_cell_neighbors")
    assert not registry.contains("cluster_cells")
    assert not registry.contains("compute_cell_umap")
    assert registry.contains("evaluate_cell_clustering")
    assert registry.contains("transfer_cell_labels")
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


def test_embedding_resource_pins_are_optional_bounded_and_not_output_refs(registry) -> None:
    pins = {"resource_id": "reviewed-mouse", "checkpoint_sha256": "a" * 64,
            "frequencies_sha256": "b" * 64, "filter_indices_sha256": "c" * 64}
    argument = registry.get("epizoo_embed_cells").optional_arguments["expected_resource_identity"]
    argument.validate("expected_resource_identity", pins)
    argument.validate("expected_resource_identity", None)
    for invalid in (dict(pins, path="/private"), dict(pins, checkpoint_sha256="A" * 64),
                    StepOutputRef("inspect", "input_path")):
        with pytest.raises(ToolArgumentError):
            argument.validate("expected_resource_identity", invalid)


def test_embedding_typed_pin_errors_have_safe_central_classification(registry) -> None:
    from agent.tools.analysis.epizoo_embedding import EpiZooResourceIdentityError
    for code, category in (("EPIZOO_RESOURCE_IDENTITY_INVALID", ErrorCategory.USER_INPUT_ERROR),
                           ("EPIZOO_RESOURCE_IDENTITY_MISMATCH", ErrorCategory.VERIFICATION_ERROR)):
        classified = registry.classify_exception("epizoo_embed_cells", EpiZooResourceIdentityError(code))
        assert classified.code == code
        assert classified.category is category
        assert classified.recoverable is False
        assert "/" not in classified.message


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
    for fn in ('pp.knn', 'tl.leiden', 'tl.umap'):
        registry.validate_arguments('run_snapatac2', dict(input_path='source.h5ad', output_dir='out', function_name=fn))
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


def test_label_transfer_registry_contract_and_recovery_identity(registry) -> None:
    validated = registry.validate_arguments(
        "transfer_cell_labels",
        {
            "reference_embedding_path": StepOutputRef("ref", "embedding_path"),
            "reference_cell_ids_path": StepOutputRef("ref", "cell_ids_path"),
            "reference_h5ad_path": StepOutputRef("inspect-ref", "input_path"),
            "reference_label_key": "celltype",
            "query_embedding_path": StepOutputRef("query", "embedding_path"),
            "query_cell_ids_path": StepOutputRef("query", "cell_ids_path"),
            "query_h5ad_path": StepOutputRef("inspect-query", "input_path"),
            "output_dir": "/output",
            "reference_species": StepOutputRef("ref", "species"),
            "query_species": StepOutputRef("query", "species"),
            "reference_checkpoint_path": StepOutputRef("ref", "checkpoint_path"),
            "query_checkpoint_path": StepOutputRef("query", "checkpoint_path"),
        },
    )
    assert "n_neighbors" not in validated
    spec = registry.get("transfer_cell_labels")
    assert spec.recovery_policy_version == "transfer-cell-labels-v1"
    assert spec.retryable_error_codes == frozenset()


def test_label_transfer_addition_preserves_existing_recovery_identities(registry) -> None:
    assert {
        name: registry.get(name).recovery_policy_version for name in registry.names()
    } == {
        "invoke_native_function": "native-callable-v1",
        "compute_motif_accessibility": "chromvar-native-v1",
        "run_motifmatchr": "motifmatchr-native-v1",
        "run_chromvar": "chromvar-native-statistics-v1",
        "run_episcanpy": "native-episcanpy-matrix-v1",
        "run_snapatac2": "native-snapatac2-matrix-v1",
        "run_signac": "native-signac-matrix-v1",
        "run_scanpy": "native-scanpy-matrix-v1",
        "run_macs3": "native-macs3-table-v1",
        "run_cicero": "native-cicero-table-v1",
        "run_cicero_cds": "native-cicero-cds-v1",
        "run_monalisa": "native-monalisa-enrichment-v1",
        "run_archr": "native-archr-project-v1",
        "run_chromap": "native-chromap-v1",
        "adopt_scATAC_cell_by_ccre": "adopt-scatac-cell-by-ccre-v1",
        "call_scATAC_peaks": "macs3-frag-unit-record-v1",
        "predict_regulatory_sequences": "regulatory-sequence-native-v1",
        "optimize_regulatory_sequences": "regulatory-sequence-optimization-native-v1",
        "generate_regulatory_sequences": "regulatory-sequence-generation-native-v1",
        "prepare_scatac_regulatory_sequences": "scatac-regulatory-sequences-v1",
        "evaluate_regulatory_predictions": "regulatory-sequence-native-v1",
        "inspect_scATAC": "inspect-scatac-v2",
        "inspect_raw_scATAC": "inspect-raw-scatac-v1",
        "inspect_scATAC_fragments": "inspect-fragment-structure-v1",
        "annotate_scATAC_cell_types": "hierarchical-annotation-v1",


        "adopt_scATAC_cell_by_features": "adopt-scatac-cell-by-features-v1",
        "adapt_epizoo_species": "adapt-epizoo-species-v1",
        "posttrain_epizoo_model": "adapt-epizoo-species-v1",
        "epizoo_embed_cells": "epizoo-embed-cells-v2",
        "evaluate_cell_clustering": "evaluate-cell-clustering-v1",
        "transfer_cell_labels": "transfer-cell-labels-v1",

        "validate_scATAC_feature_space": "validate-scatac-feature-space-v1",
        "build_replicate_pseudobulk": "build-replicate-pseudobulk-v1",
        "run_replicate_differential_accessibility": (
            "run-replicate-differential-accessibility-edger-ql-v1"
        ),
    }


def test_differential_accessibility_registry_contract_is_exact_and_nonretryable(
    registry,
) -> None:
    spec = registry.get("run_replicate_differential_accessibility")
    assert tuple(spec.required_arguments) == (
        "pseudobulk_path",
        "group_value",
        "condition_key",
        "numerator_condition",
        "denominator_condition",
        "design_type",
        "output_dir",
    )
    assert tuple(spec.optional_arguments) == ("covariates", "overwrite")
    assert spec.recovery_policy_version == (
        "run-replicate-differential-accessibility-edger-ql-v1"
    )
    assert spec.retryable_error_codes == frozenset()
    validated = registry.validate_arguments(
        spec.name,
        {
            "pseudobulk_path": StepOutputRef("pseudobulk", "pseudobulk_path"),
            "group_value": "T",
            "condition_key": "condition",
            "numerator_condition": "treated",
            "denominator_condition": "control",
            "design_type": "paired",
            "output_dir": "/output",
            "covariates": ({"key": "age", "kind": "numeric"},),
        },
    )
    assert validated["design_type"] == "paired"


@pytest.mark.parametrize(
    ("code", "category"),
    [
        ("DA_CONDITION_KEY_MISMATCH", ErrorCategory.USER_INPUT_ERROR),
        ("DA_DESIGN_RANK_DEFICIENT", ErrorCategory.USER_INPUT_ERROR),
        ("DA_NO_FEATURES_AFTER_FILTER", ErrorCategory.TOOL_EXECUTION_ERROR),
        ("R_BACKEND_PROTOCOL_INVALID", ErrorCategory.TOOL_EXECUTION_ERROR),
        ("EDGER_VERSION_UNSUPPORTED", ErrorCategory.ENVIRONMENT_ERROR),
        ("HOST_MEMORY_EXHAUSTED", ErrorCategory.RESOURCE_ERROR),
        ("DISK_FULL", ErrorCategory.RESOURCE_ERROR),
        ("SOURCE_CHANGED_DURING_READ", ErrorCategory.VERIFICATION_ERROR),
    ],
)
def test_differential_accessibility_error_classification_is_stable_and_sanitized(
    registry, code, category
) -> None:
    error = registry.classify_exception(
        "run_replicate_differential_accessibility",
        M82ScientificError(code, "/secret/path must not persist"),
        step_id="da",
        attempt=1,
    )
    assert error.code == code
    assert error.category is category
    assert error.recoverable is False
    assert "/secret/path" not in error.message


@pytest.mark.parametrize(
    "code",
    [
        "RSCRIPT_UNAVAILABLE",
        "EDGER_PACKAGE_UNAVAILABLE",
        "EDGER_VERSION_UNSUPPORTED",
        "R_PACKAGE_VERSION_INCOMPATIBLE",
    ],
)
def test_differential_accessibility_backend_compatibility_requires_compatible_runtime(
    registry, code
) -> None:
    error = registry.classify_exception(
        "run_replicate_differential_accessibility",
        M82ScientificError(code, "sensitive backend diagnostic"),
        step_id="da",
        attempt=1,
    )
    assert error.category is ErrorCategory.ENVIRONMENT_ERROR
    assert error.recoverable is False
    assert (
        error.recovery_disposition
        is RecoveryDisposition.RESUME_WITH_COMPATIBLE_RUNTIME
    )




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
        "analysis_path": "/output/embeddings.h5ad",
        "analysis_sha256": "a" * 64,
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
    evaluation = {
        "status": "success",
        "analysis_path": "/output/clustered.h5ad",
            "analysis_sha256": "a" * 64,
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
    registry.validate_result("evaluate_cell_clustering", evaluation)

    assert registry.get("inspect_scATAC").result_contract.name == "ScATACInspection"
    assert (
        registry.get("epizoo_embed_cells").result_contract.name
        == "EpiZooEmbeddingToolResult"
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
