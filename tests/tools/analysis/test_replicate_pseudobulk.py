from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from agent.tools.analysis.replicate_pseudobulk import (
    FEATURE_SPACE_ARTIFACT_TYPE,
    M81ScientificError,
    PSEUDOBULK_ARTIFACT_TYPE,
    PSEUDOBULK_PROVENANCE_KEY,
    build_replicate_pseudobulk,
    validate_scATAC_feature_space,
)
from agent.tools.analysis.label_transfer import transfer_cell_labels
from agent.orchestration import (
    PlanStep,
    StepOutputRef,
    ToolRegistry,
    build_default_tool_registry,
    verify_step,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(path: Path, *, coordinates: bool = False, layer: bool = False) -> Path:
    matrix = sparse.csr_matrix(
        np.asarray(
            [
                [1, 0, 2, 0],
                [0, 3, 0, 0],
                [4, 0, 5, 6],
                [0, 7, 0, 8],
                [9, 0, 1, 0],
            ],
            dtype=np.int32,
        )
    )
    obs = pd.DataFrame(
        {
            "cell_type": ["T", "T", "B", "T", "B"],
            "donor": ["d1", "d1", "d2", "d2", "d1"],
            "condition": ["control", "control", "treated", "treated", "treated"],
            "sex": ["F", "F", "M", "M", "F"],
            "age": [10, 10, 12, 12, 10],
        },
        index=[f"cell-{index}" for index in range(5)],
    )
    var = pd.DataFrame(index=[f"peak-{index}" for index in range(4)])
    if coordinates:
        var["chromosome"] = ["chr1", "chr1", "chr2", "chr2"]
        var["start"] = [0, 100, 0, 100]
        var["end"] = [50, 150, 50, 150]
    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    if layer:
        adata.layers["counts"] = matrix.copy()
        adata.X = sparse.csr_matrix(matrix.shape, dtype=np.float32)
    adata.uns["matrix_semantics"] = "fragment_counts"
    adata.write_h5ad(path)
    return path


def _feature(
    source: Path,
    output: Path,
    *,
    matrix_source: str = "X",
    layer_key: str | None = None,
    coordinates: bool = False,
) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "matrix_source": matrix_source,
        "layer_key": layer_key,
        "matrix_semantics": "fragment_counts",
        "semantics_metadata_key": "matrix_semantics",
        "species": "human",
        "genome_assembly": "hg38",
        "coordinate_source": "none",
    }
    if coordinates:
        kwargs.update(
            {
                "coordinate_source": "var_columns",
                "feature_chrom_key": "chromosome",
                "feature_start_key": "start",
                "feature_end_key": "end",
                "coordinate_system": "zero_based_half_open",
            }
        )
    return validate_scATAC_feature_space(source, output, **kwargs)


def test_feature_space_manifest_binds_source_semantics_and_optional_coordinates(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path / "source.h5ad", coordinates=True)
    before = _sha(source)

    result = _feature(source, tmp_path / "feature", coordinates=True)

    assert _sha(source) == before
    assert result["status"] == "success"
    assert result["source_h5ad_sha256"] == before
    assert result["semantics_assertion_source"] == "structured_request_and_raw_uns"
    assert result["coordinate_source"] == "var_columns"
    assert result["coordinates_sha256"] is not None
    manifest_path = Path(str(result["feature_space_path"]))
    assert _sha(manifest_path) == result["feature_space_sha256"]
    manifest = json.loads(manifest_path.read_text())
    assert manifest["artifact_type"] == FEATURE_SPACE_ARTIFACT_TYPE
    assert manifest["coordinates"]["available"] is True
    assert "cell_ids" not in manifest and "feature_ids" not in manifest


def test_feature_space_supports_explicit_sparse_layer(tmp_path: Path) -> None:
    source = _source(tmp_path / "layered.h5ad", layer=True)
    result = _feature(
        source,
        tmp_path / "feature",
        matrix_source="layer",
        layer_key="counts",
    )
    assert result["matrix_source"] == "layer"
    assert result["layer_key"] == "counts"
    assert result["nnz"] == 10


def test_pseudobulk_exact_sum_and_first_occurrence_order(tmp_path: Path) -> None:
    source = _source(tmp_path / "source.h5ad")
    feature = _feature(source, tmp_path / "feature")

    result = build_replicate_pseudobulk(
        str(feature["feature_space_path"]),
        "donor",
        "cell_type",
        "condition",
        tmp_path / "pseudobulk",
        group_source="raw_obs",
        covariate_keys=["sex", "age"],
    )

    assert result["status"] == "success"
    assert result["n_pseudobulks"] == 4
    assert result["n_groups"] == 2
    assert result["n_replicates"] == 2
    assert result["n_conditions"] == 2
    assert result["minimum_cells_per_pseudobulk"] == 1
    assert result["maximum_cells_per_pseudobulk"] == 2
    artifact_path = Path(str(result["pseudobulk_path"]))
    assert _sha(artifact_path) == result["pseudobulk_sha256"]
    artifact = ad.read_h5ad(artifact_path)
    assert sparse.isspmatrix_csr(artifact.X)
    assert artifact.X.dtype == np.dtype(np.int64)
    assert artifact.obs[["group", "replicate", "condition"]].astype(str).values.tolist() == [
        ["T", "d1", "control"],
        ["B", "d2", "treated"],
        ["T", "d2", "treated"],
        ["B", "d1", "treated"],
    ]
    np.testing.assert_array_equal(
        artifact.X.toarray(),
        np.asarray(
            [
                [1, 3, 2, 0],
                [4, 0, 5, 6],
                [0, 7, 0, 8],
                [9, 0, 1, 0],
            ],
            dtype=np.int64,
        ),
    )
    assert artifact.obs["n_cells"].tolist() == [2, 1, 1, 1]
    assert artifact.obs["first_cell_index"].tolist() == [0, 2, 3, 4]
    assert artifact.obs["library_size"].tolist() == [6, 15, 15, 10]
    assert all(value.startswith("pb-") and len(value) == 67 for value in artifact.obs_names)
    provenance = artifact.uns[PSEUDOBULK_PROVENANCE_KEY]
    assert provenance["artifact_type"] == PSEUDOBULK_ARTIFACT_TYPE
    assert provenance["aggregation"]["method"] == "sum"
    assert provenance["validation"]["normalization_performed"] is False
    assert len(artifact.layers) == len(artifact.obsm) == len(artifact.obsp) == 0


def test_pseudobulk_ids_are_scoped_to_feature_space_identity(tmp_path: Path) -> None:
    first_source = _source(tmp_path / "first.h5ad")
    second_source = _source(tmp_path / "second.h5ad")
    second_adata = ad.read_h5ad(second_source)
    second_adata.uns["source_identity_marker"] = "second"
    second_adata.write_h5ad(second_source)
    first_feature = _feature(first_source, tmp_path / "first-feature")
    second_feature = _feature(second_source, tmp_path / "second-feature")
    first = build_replicate_pseudobulk(
        str(first_feature["feature_space_path"]), "donor", "cell_type", "condition",
        tmp_path / "first-output", group_source="raw_obs",
    )
    second = build_replicate_pseudobulk(
        str(second_feature["feature_space_path"]), "donor", "cell_type", "condition",
        tmp_path / "second-output", group_source="raw_obs",
    )
    first_ids = tuple(ad.read_h5ad(first["pseudobulk_path"]).obs_names)
    second_ids = tuple(ad.read_h5ad(second["pseudobulk_path"]).obs_names)
    assert first_feature["feature_space_identity_sha256"] != second_feature["feature_space_identity_sha256"]
    assert first_ids != second_ids


@pytest.mark.parametrize(
    ("species", "assembly"),
    [("human", "mm10"), ("mouse", "hg38"), ("rat", "rn7")],
)
def test_feature_space_rejects_unsupported_species_assembly(
    tmp_path: Path, species: str, assembly: str
) -> None:
    source = _source(tmp_path / "source.h5ad")
    with pytest.raises(M81ScientificError, match="human/hg38") as failure:
        validate_scATAC_feature_space(
            source,
            tmp_path / "out",
            matrix_source="X",
            matrix_semantics="fragment_counts",
            species=species,
            genome_assembly=assembly,
            coordinate_source="none",
        )
    assert failure.value.code == "SPECIES_ASSEMBLY_INVALID"


def test_feature_space_rejects_normalized_dense_fractional_and_semantics_mismatch(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path / "source.h5ad")
    common = dict(
        input_path=source,
        output_dir=tmp_path / "out",
        matrix_source="X",
        species="human",
        genome_assembly="hg38",
        coordinate_source="none",
    )
    with pytest.raises(M81ScientificError) as normalized:
        validate_scATAC_feature_space(**common, matrix_semantics="normalized_continuous")
    assert normalized.value.code == "MATRIX_SEMANTICS_UNSUPPORTED"

    with pytest.raises(M81ScientificError) as mismatch:
        validate_scATAC_feature_space(
            **common,
            matrix_semantics="insertion_counts",
            semantics_metadata_key="matrix_semantics",
        )
    assert mismatch.value.code == "MATRIX_SEMANTICS_UNSUPPORTED"

    dense = tmp_path / "dense.h5ad"
    ad.AnnData(np.ones((2, 2), dtype=np.int64)).write_h5ad(dense)
    with pytest.raises(M81ScientificError) as dense_failure:
        validate_scATAC_feature_space(
            dense, tmp_path / "dense-out", matrix_source="X",
            matrix_semantics="fragment_counts", species="human",
            genome_assembly="hg38", coordinate_source="none",
        )
    assert dense_failure.value.code == "MATRIX_STORAGE_UNSUPPORTED"

    fractional = tmp_path / "fractional.h5ad"
    ad.AnnData(sparse.csr_matrix([[0.5, 0.0]])).write_h5ad(fractional)
    with pytest.raises(M81ScientificError) as fractional_failure:
        validate_scATAC_feature_space(
            fractional, tmp_path / "fractional-out", matrix_source="X",
            matrix_semantics="fragment_counts", species="human",
            genome_assembly="hg38", coordinate_source="none",
        )
    assert fractional_failure.value.code == "MATRIX_VALUES_INVALID"


def test_pseudobulk_rejects_nonconstant_covariate_and_does_not_drop_cells(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path / "source.h5ad")
    adata = ad.read_h5ad(source)
    adata.obs.loc["cell-1", "age"] = 99
    adata.write_h5ad(source)
    feature = _feature(source, tmp_path / "feature")
    with pytest.raises(M81ScientificError) as failure:
        build_replicate_pseudobulk(
            str(feature["feature_space_path"]), "donor", "cell_type", "condition",
            tmp_path / "out", group_source="raw_obs", covariate_keys=["age"],
        )
    assert failure.value.code == "COVARIATE_NOT_CONSTANT"


def test_pseudobulk_rejects_library_size_overflow_without_wrapping(
    tmp_path: Path,
) -> None:
    source = tmp_path / "overflow.h5ad"
    maximum = np.iinfo(np.int64).max
    ad.AnnData(
        X=sparse.csr_matrix([[maximum, maximum, maximum]], dtype=np.int64),
        obs=pd.DataFrame(
            {"group": ["A"], "replicate": ["r1"], "condition": ["control"]},
            index=["cell-0"],
        ),
        var=pd.DataFrame(index=["p1", "p2", "p3"]),
    ).write_h5ad(source)
    feature = validate_scATAC_feature_space(
        source,
        tmp_path / "feature",
        matrix_source="X",
        matrix_semantics="fragment_counts",
        species="human",
        genome_assembly="hg38",
        coordinate_source="none",
    )
    with pytest.raises(M81ScientificError) as caught:
        build_replicate_pseudobulk(
            feature["feature_space_path"],
            "replicate",
            "group",
            "condition",
            tmp_path / "pseudobulk",
            group_source="raw_obs",
        )
    assert caught.value.code == "INTEGER_SUM_OVERFLOW"


def test_binary_accessibility_has_distinct_output_semantics(tmp_path: Path) -> None:
    path = tmp_path / "binary.h5ad"
    adata = ad.AnnData(
        X=sparse.csr_matrix([[1, 0], [1, 1]], dtype=np.int8),
        obs=pd.DataFrame(
            {"group": ["A", "A"], "replicate": ["r1", "r1"], "condition": ["c", "c"]},
            index=["c1", "c2"],
        ),
        var=pd.DataFrame(index=["f1", "f2"]),
    )
    adata.write_h5ad(path)
    feature = validate_scATAC_feature_space(
        path, tmp_path / "feature", matrix_source="X",
        matrix_semantics="binary_accessibility", species="human",
        genome_assembly="hg38", coordinate_source="none",
    )
    result = build_replicate_pseudobulk(
        feature["feature_space_path"], "replicate", "group", "condition",
        tmp_path / "out", group_source="raw_obs",
    )
    assert result["output_value_semantics"] == "accessible_cell_count"
    np.testing.assert_array_equal(ad.read_h5ad(result["pseudobulk_path"]).X.toarray(), [[2, 1]])


def _verified_annotation(
    tmp_path: Path,
    query_source: Path,
    query_ids: list[str],
    *,
    tie: bool = False,
) -> dict[str, object]:
    reference_ids = ["reference-0", "reference-1", "reference-2", "reference-3"]
    reference_embedding = np.zeros((4, 512), dtype=np.float32)
    reference_embedding[:, 0] = [-2.0, -1.0, 1.0, 2.0] if tie else [0.0, 0.2, 9.8, 10.0]
    query_embedding = np.zeros((len(query_ids), 512), dtype=np.float32)
    if not tie:
        query_embedding[:, 0] = [0.0, 0.1, 9.9, 10.0, 0.2][: len(query_ids)]
    reference_embedding_path = tmp_path / "reference.npy"
    query_embedding_path = tmp_path / "query.npy"
    np.save(reference_embedding_path, reference_embedding, allow_pickle=False)
    np.save(query_embedding_path, query_embedding, allow_pickle=False)
    reference_ids_path = tmp_path / "reference.txt"
    query_ids_path = tmp_path / "query.txt"
    reference_ids_path.write_text(
        "".join(f"{value}\n" for value in reference_ids), encoding="utf-8"
    )
    query_ids_path.write_text(
        "".join(f"{value}\n" for value in query_ids), encoding="utf-8"
    )
    reference_h5ad = tmp_path / "reference.h5ad"
    ad.AnnData(
        obs=pd.DataFrame(
            {"celltype": pd.Categorical(["A", "A", "B", "B"])},
            index=reference_ids,
        )
    ).write_h5ad(reference_h5ad)
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint")
    return transfer_cell_labels(
        reference_embedding_path,
        reference_ids_path,
        reference_h5ad,
        "celltype",
        query_embedding_path,
        query_ids_path,
        query_source,
        tmp_path / "annotation",
        reference_species="human",
        query_species="human",
        reference_checkpoint_path=checkpoint,
        query_checkpoint_path=checkpoint,
        n_neighbors=4 if tie else 2,
    )


def test_verified_m63_annotation_is_the_only_nonraw_group_source(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path / "source.h5ad")
    annotation = _verified_annotation(
        tmp_path, source, [f"cell-{index}" for index in range(5)]
    )
    feature = _feature(source, tmp_path / "feature")
    result = build_replicate_pseudobulk(
        feature["feature_space_path"],
        "donor",
        "predicted_label",
        "condition",
        tmp_path / "pseudobulk",
        group_source="verified_annotation",
        group_annotation_path=annotation["annotation_path"],
    )
    artifact = ad.read_h5ad(result["pseudobulk_path"])
    assert artifact.obs["group"].astype(str).tolist() == ["A", "B", "A"]
    provenance = artifact.uns[PSEUDOBULK_PROVENANCE_KEY]
    assert provenance["metadata"]["group_source"] == "verified_annotation"
    assert provenance["metadata"]["group_annotation_sha256"] == annotation["annotation_sha256"]


@pytest.mark.parametrize("failure", ["unassigned", "cell_order"])
def test_verified_annotation_fails_closed_on_invalid_cell_assignment(
    tmp_path: Path, failure: str
) -> None:
    source = _source(tmp_path / "source.h5ad")
    source_ids = [f"cell-{index}" for index in range(5)]
    annotation_ids = list(reversed(source_ids)) if failure == "cell_order" else source_ids
    annotation_source = source
    if failure == "cell_order":
        annotation_source = tmp_path / "annotation-query.h5ad"
        ad.AnnData(
            obs=pd.DataFrame(index=annotation_ids)
        ).write_h5ad(annotation_source)
    annotation = _verified_annotation(
        tmp_path, annotation_source, annotation_ids, tie=failure == "unassigned"
    )
    feature = _feature(source, tmp_path / "feature")
    with pytest.raises(M81ScientificError) as caught:
        build_replicate_pseudobulk(
            feature["feature_space_path"],
            "donor",
            "predicted_label",
            "condition",
            tmp_path / "pseudobulk",
            group_source="verified_annotation",
            group_annotation_path=annotation["annotation_path"],
        )
    assert caught.value.code in {"GROUP_ANNOTATION_INVALID", "CELL_IDENTITY_MISMATCH"}


def test_independent_verifier_passes_without_invoking_scientific_callables(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path / "source.h5ad")
    feature = _feature(source, tmp_path / "feature")
    feature_arguments = {
        "input_path": str(source),
        "output_dir": str(tmp_path / "feature"),
        "matrix_source": "X",
        "matrix_semantics": "fragment_counts",
        "species": "human",
        "genome_assembly": "hg38",
        "coordinate_source": "none",
        "semantics_metadata_key": "matrix_semantics",
    }
    feature_step = PlanStep(
        "feature", "validate_scATAC_feature_space", feature_arguments
    )
    pseudobulk_arguments = {
        "feature_space_path": str(feature["feature_space_path"]),
        "replicate_key": "donor",
        "group_key": "cell_type",
        "condition_key": "condition",
        "output_dir": str(tmp_path / "pseudobulk"),
        "group_source": "raw_obs",
        "covariate_keys": ["sex", "age"],
    }
    pseudobulk = build_replicate_pseudobulk(**pseudobulk_arguments)
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
    guards = {
        name: Mock(side_effect=AssertionError("verifier invoked a scientific tool"))
        for name in ("validate_scATAC_feature_space", "build_replicate_pseudobulk")
    }
    registry = ToolRegistry(
        tuple(
            replace(spec, function=guards[spec.name]) if spec.name in guards else spec
            for spec in (default.get(name) for name in default.names())
        )
    )

    assert verify_step(feature_step, feature_arguments, feature, registry).passed
    verification = verify_step(
        pseudobulk_step,
        pseudobulk_arguments,
        pseudobulk,
        registry,
        dependency_results={"feature": feature},
    )
    assert verification.passed
    assert all(guard.call_count == 0 for guard in guards.values())


def test_independent_verifier_rejects_exact_sum_tampering(tmp_path: Path) -> None:
    source = _source(tmp_path / "source.h5ad")
    feature = _feature(source, tmp_path / "feature")
    arguments = {
        "feature_space_path": str(feature["feature_space_path"]),
        "replicate_key": "donor",
        "group_key": "cell_type",
        "condition_key": "condition",
        "output_dir": str(tmp_path / "pseudobulk"),
        "group_source": "raw_obs",
    }
    result = build_replicate_pseudobulk(**arguments)
    artifact_path = Path(result["pseudobulk_path"])
    artifact = ad.read_h5ad(artifact_path)
    artifact.X[0, 0] += 1
    artifact.write_h5ad(artifact_path)
    tampered = dict(result)
    tampered["pseudobulk_sha256"] = _sha(artifact_path)
    step = PlanStep("pseudobulk", "build_replicate_pseudobulk", arguments)

    verification = verify_step(
        step, arguments, tampered, build_default_tool_registry()
    )

    assert not verification.passed
    assert verification.error is not None
    assert verification.error.code == "PSEUDOBULK_AGGREGATION_MISMATCH"


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("metadata", "PSEUDOBULK_METADATA_MISMATCH"),
        ("feature", "PSEUDOBULK_FEATURE_MISMATCH"),
        ("provenance", "PSEUDOBULK_PROVENANCE_MISMATCH"),
        ("artifact_digest", "ARTIFACT_SHA256_MISMATCH"),
        ("source", "FEATURE_SPACE_SOURCE_MISMATCH"),
    ],
)
def test_independent_verifier_rejects_identity_and_provenance_tampering(
    tmp_path: Path, mutation: str, expected_code: str
) -> None:
    source = _source(tmp_path / "source.h5ad")
    feature = _feature(source, tmp_path / "feature")
    arguments = {
        "feature_space_path": str(feature["feature_space_path"]),
        "replicate_key": "donor",
        "group_key": "cell_type",
        "condition_key": "condition",
        "output_dir": str(tmp_path / "pseudobulk"),
        "group_source": "raw_obs",
    }
    result = build_replicate_pseudobulk(**arguments)
    artifact_path = Path(result["pseudobulk_path"])
    tampered = dict(result)
    if mutation == "source":
        raw = ad.read_h5ad(source)
        raw.uns["changed"] = True
        raw.write_h5ad(source)
    else:
        artifact = ad.read_h5ad(artifact_path)
        if mutation == "metadata":
            artifact.obs["group"] = pd.Categorical(["changed"] * artifact.n_obs)
        elif mutation == "feature":
            artifact.var_names = [f"changed-{index}" for index in range(artifact.n_vars)]
        elif mutation == "provenance":
            provenance = dict(artifact.uns[PSEUDOBULK_PROVENANCE_KEY])
            provenance["stage"] = "changed"
            artifact.uns[PSEUDOBULK_PROVENANCE_KEY] = provenance
        else:
            artifact.obs["library_size"] = artifact.obs["library_size"] + 1
        artifact.write_h5ad(artifact_path)
        if mutation != "artifact_digest":
            tampered["pseudobulk_sha256"] = _sha(artifact_path)

    verification = verify_step(
        PlanStep("pseudobulk", "build_replicate_pseudobulk", arguments),
        arguments,
        tampered,
        build_default_tool_registry(),
    )
    assert not verification.passed
    assert verification.error is not None
    assert verification.error.code == expected_code
