"""Milestone 8.2-A deterministic statistical-preparation tests."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from agent.tools.analysis import differential_accessibility as da
from agent.tools.analysis.differential_accessibility import (
    DA_LOW_REPLICATION_WARNING,
    DA_ONE_CELL_PSEUDOBULK_WARNING,
    M82ScientificError,
    prepare_replicate_differential_accessibility,
)
from agent.tools.analysis.replicate_pseudobulk import (
    M81ScientificError,
    build_replicate_pseudobulk,
    validate_scATAC_feature_space,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _independent_units(
    numerator: int = 3,
    denominator: int = 3,
    *,
    group: str = "T",
) -> list[dict[str, object]]:
    return [
        {"group": group, "replicate": f"control-{index}", "condition": "control"}
        for index in range(denominator)
    ] + [
        {"group": group, "replicate": f"treated-{index}", "condition": "treated"}
        for index in range(numerator)
    ]


def _paired_units(count: int = 3) -> list[dict[str, object]]:
    units: list[dict[str, object]] = []
    for index in range(count):
        units.extend(
            (
                {"group": "T", "replicate": f"donor-{index}", "condition": "control"},
                {"group": "T", "replicate": f"donor-{index}", "condition": "treated"},
            )
        )
    return units


def _pseudobulk(
    tmp_path: Path,
    units: list[dict[str, object]],
    *,
    semantics: str = "fragment_counts",
    covariate_keys: tuple[str, ...] = (),
) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    observations: list[dict[str, object]] = []
    rows: list[list[int]] = []
    cell_ids: list[str] = []
    for unit_index, unit in enumerate(units):
        n_cells = int(unit.get("n_cells", 2))
        zero_library = bool(unit.get("zero_library", False))
        low_depth = bool(unit.get("low_depth", False))
        for cell_index in range(n_cells):
            observation = {
                "group": unit["group"],
                "replicate": unit["replicate"],
                "condition": unit["condition"],
            }
            for key in covariate_keys:
                observation[key] = unit[key]
            observations.append(observation)
            cell_ids.append(f"cell-{unit_index:03d}-{cell_index:03d}")
            if zero_library:
                rows.append([0, 0, 0])
            elif semantics == "binary_accessibility":
                rows.append([1, 0, 1])
            elif low_depth:
                rows.append([1 if cell_index == 0 else 0, 0, 0])
            else:
                rows.append([unit_index + 1, cell_index + 1, 1])
    source = tmp_path / "source.h5ad"
    ad.AnnData(
        X=sparse.csr_matrix(np.asarray(rows, dtype=np.int64)),
        obs=pd.DataFrame(observations, index=cell_ids),
        var=pd.DataFrame(index=["peak-z", "peak-a", "peak-m"]),
    ).write_h5ad(source)
    feature = validate_scATAC_feature_space(
        source,
        tmp_path / "feature",
        matrix_source="X",
        matrix_semantics=semantics,
        species="human",
        genome_assembly="hg38",
        coordinate_source="none",
    )
    pseudobulk = build_replicate_pseudobulk(
        feature["feature_space_path"],
        "replicate",
        "group",
        "condition",
        tmp_path / "pseudobulk",
        group_source="raw_obs",
        covariate_keys=list(covariate_keys),
    )
    return source, Path(pseudobulk["pseudobulk_path"])


def _prepare(
    path: Path,
    *,
    design_type: str = "independent",
    covariates: tuple[dict[str, str], ...] = (),
):
    return prepare_replicate_differential_accessibility(
        path,
        "T",
        "condition",
        "treated",
        "control",
        design_type,
        covariates=covariates,
    )


def _assert_error(code: str, callback) -> None:
    with pytest.raises(M82ScientificError) as caught:
        callback()
    assert caught.value.code == code


def test_valid_independent_two_vs_two_warns_and_codes_numerator_minus_denominator(
    tmp_path: Path,
) -> None:
    _, path = _pseudobulk(tmp_path, _independent_units(2, 2))

    prepared = _prepare(path)

    assert prepared.design_columns == ("intercept", "condition_numerator")
    np.testing.assert_array_equal(
        prepared.design_matrix,
        np.asarray([[1, 0], [1, 0], [1, 1], [1, 1]], dtype=np.float64),
    )
    np.testing.assert_array_equal(prepared.contrast, [0.0, 1.0])
    assert prepared.residual_degrees_of_freedom == 2
    assert [warning.code for warning in prepared.warnings] == [
        DA_LOW_REPLICATION_WARNING
    ]
    assert dict(prepared.warnings[0].metadata) == {
        "numerator_replicates": 2,
        "denominator_replicates": 2,
        "recommended_minimum_per_condition": 3,
    }
    assert not prepared.design_matrix.flags.writeable
    assert not prepared.contrast.flags.writeable


def test_valid_independent_three_plus_and_unequal_replication(
    tmp_path: Path,
) -> None:
    _, path = _pseudobulk(tmp_path, _independent_units(4, 3))
    prepared = _prepare(path)
    assert len(prepared.numerator_replicates) == 4
    assert len(prepared.denominator_replicates) == 3
    assert prepared.warnings == ()
    assert prepared.design_rank == 2
    assert prepared.residual_degrees_of_freedom == 5


@pytest.mark.parametrize(("numerator", "denominator"), [(1, 3), (3, 1)])
def test_independent_rejects_fewer_than_two_replicates(
    tmp_path: Path, numerator: int, denominator: int
) -> None:
    _, path = _pseudobulk(
        tmp_path, _independent_units(numerator, denominator)
    )
    _assert_error("DA_REPLICATION_INSUFFICIENT", lambda: _prepare(path))


def test_independent_rejects_replicate_overlap(tmp_path: Path) -> None:
    units = _independent_units()
    units[3]["replicate"] = units[0]["replicate"]
    _, path = _pseudobulk(tmp_path, units)
    _assert_error("DA_PAIRING_INVALID", lambda: _prepare(path))


def test_valid_paired_design_uses_first_occurrence_block_order(tmp_path: Path) -> None:
    units = [
        {"group": "T", "replicate": "r2", "condition": "control"},
        {"group": "T", "replicate": "r1", "condition": "treated"},
        {"group": "T", "replicate": "r3", "condition": "treated"},
        {"group": "T", "replicate": "r2", "condition": "treated"},
        {"group": "T", "replicate": "r1", "condition": "control"},
        {"group": "T", "replicate": "r3", "condition": "control"},
    ]
    _, path = _pseudobulk(tmp_path, units)

    prepared = _prepare(path, design_type="paired")

    assert prepared.replicate_order == ("r2", "r1", "r3")
    assert prepared.design_columns == (
        "intercept",
        "condition_numerator",
        "replicate_001",
        "replicate_002",
    )
    np.testing.assert_array_equal(
        prepared.design_matrix,
        np.asarray(
            [
                [1, 0, 0, 0],
                [1, 1, 1, 0],
                [1, 1, 0, 1],
                [1, 1, 0, 0],
                [1, 0, 1, 0],
                [1, 0, 0, 1],
            ],
            dtype=np.float64,
        ),
    )
    np.testing.assert_array_equal(prepared.contrast, [0, 1, 0, 0])
    assert prepared.residual_degrees_of_freedom == 2


def test_valid_larger_paired_design(tmp_path: Path) -> None:
    _, path = _pseudobulk(tmp_path, _paired_units(5))
    prepared = _prepare(path, design_type="paired")
    assert prepared.design_matrix.shape == (10, 6)
    assert prepared.design_rank == 6
    assert prepared.residual_degrees_of_freedom == 4


def test_paired_rejects_fewer_than_three_pairs(tmp_path: Path) -> None:
    _, path = _pseudobulk(tmp_path, _paired_units(2))
    _assert_error(
        "DA_REPLICATION_INSUFFICIENT", lambda: _prepare(path, design_type="paired")
    )


@pytest.mark.parametrize("mutation", ["incomplete", "mismatched"])
def test_paired_rejects_incomplete_or_mismatched_sets(
    tmp_path: Path, mutation: str
) -> None:
    units = _paired_units(3)
    if mutation == "incomplete":
        units.pop()
    else:
        units[-1]["replicate"] = "other"
    _, path = _pseudobulk(tmp_path, units)
    _assert_error("DA_PAIRING_INVALID", lambda: _prepare(path, design_type="paired"))


def test_paired_duplicate_observation_fails_defensive_internal_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, path = _pseudobulk(tmp_path, _paired_units(3))
    snapshot = da._verified_pseudobulk(path)
    duplicate = replace(
        snapshot,
        row_ids=(*snapshot.row_ids, "defensive-duplicate"),
        groups=(*snapshot.groups, snapshot.groups[0]),
        replicates=(*snapshot.replicates, snapshot.replicates[0]),
        conditions=(*snapshot.conditions, snapshot.conditions[0]),
        n_cells=(*snapshot.n_cells, snapshot.n_cells[0]),
        library_sizes=(*snapshot.library_sizes, snapshot.library_sizes[0]),
        covariate_values=(*snapshot.covariate_values, snapshot.covariate_values[0]),
    )
    monkeypatch.setattr(da, "_verified_pseudobulk", lambda _: duplicate)
    _assert_error("DA_PAIRING_INVALID", lambda: _prepare(path, design_type="paired"))


def test_selected_zero_library_fails_but_positive_one_cell_warns(
    tmp_path: Path,
) -> None:
    zero_units = _independent_units()
    zero_units[0]["zero_library"] = True
    _, zero_path = _pseudobulk(tmp_path / "zero", zero_units)
    _assert_error("DA_ZERO_LIBRARY", lambda: _prepare(zero_path))

    one_cell_units = _independent_units()
    one_cell_units[2]["n_cells"] = 1
    _, one_cell_path = _pseudobulk(tmp_path / "one", one_cell_units)
    prepared = _prepare(one_cell_path)
    warning = next(
        warning
        for warning in prepared.warnings
        if warning.code == DA_ONE_CELL_PSEUDOBULK_WARNING
    )
    metadata = dict(warning.metadata)
    assert metadata["pseudobulk_count"] == 1
    assert metadata["cell_counts"] == (1,)
    assert metadata["pseudobulk_ids"] == (prepared.source_row_ids[2],)


def test_positive_low_depth_is_not_excluded(tmp_path: Path) -> None:
    units = _independent_units()
    units[0]["low_depth"] = True
    _, path = _pseudobulk(tmp_path, units)
    prepared = _prepare(path)
    assert prepared.row_eligibility[0].library_size == 1
    assert prepared.row_eligibility[0].included
    assert len(prepared.included_source_positions) == 6


def test_categorical_numeric_and_ordered_covariate_encoding(tmp_path: Path) -> None:
    units = [
        {"group": "T", "replicate": "c1", "condition": "control", "batch": "b2", "age": 10},
        {"group": "T", "replicate": "t1", "condition": "treated", "batch": "b1", "age": 15},
        {"group": "T", "replicate": "c2", "condition": "control", "batch": "b1", "age": 20},
        {"group": "T", "replicate": "t2", "condition": "treated", "batch": "b2", "age": 25},
        {"group": "T", "replicate": "c3", "condition": "control", "batch": "b2", "age": 30},
        {"group": "T", "replicate": "t3", "condition": "treated", "batch": "b1", "age": 40},
    ]
    _, path = _pseudobulk(tmp_path, units, covariate_keys=("batch", "age"))

    prepared = _prepare(
        path,
        covariates=(
            {"key": "batch", "kind": "categorical"},
            {"key": "age", "kind": "numeric"},
        ),
    )

    assert prepared.design_columns == (
        "intercept",
        "condition_numerator",
        "covariate_000_level_001",
        "covariate_001_numeric",
    )
    batch, age = prepared.covariate_encodings
    assert [(level.value, level.design_column) for level in batch.categorical_levels] == [
        ("b2", None),
        ("b1", "covariate_000_level_001"),
    ]
    assert age.values == (10.0, 15.0, 20.0, 25.0, 30.0, 40.0)
    np.testing.assert_array_equal(prepared.design_matrix[:, 2], [0, 1, 1, 0, 0, 1])
    np.testing.assert_array_equal(prepared.design_matrix[:, 3], age.values)


@pytest.mark.parametrize(
    "covariates",
    [
        ({"key": "age", "kind": "numeric"}, {"key": "age", "kind": "numeric"}),
        ({"key": "absent", "kind": "numeric"},),
        ({"key": "age", "kind": "other"},),
        ({"key": "age"},),
    ],
)
def test_invalid_covariate_specifications_fail(
    tmp_path: Path, covariates: tuple[dict[str, str], ...]
) -> None:
    units = _independent_units()
    for index, unit in enumerate(units):
        unit["age"] = index
    _, path = _pseudobulk(tmp_path, units, covariate_keys=("age",))
    _assert_error(
        "DA_COVARIATE_INVALID", lambda: _prepare(path, covariates=covariates)
    )


@pytest.mark.parametrize(("kind", "value"), [("categorical", "same"), ("numeric", 10)])
def test_invariant_covariates_fail(
    tmp_path: Path, kind: str, value: object
) -> None:
    units = _independent_units()
    for unit in units:
        unit["covariate"] = value
    _, path = _pseudobulk(tmp_path, units, covariate_keys=("covariate",))
    _assert_error(
        "DA_COVARIATE_INVARIANT",
        lambda: _prepare(
            path, covariates=({"key": "covariate", "kind": kind},)
        ),
    )


def test_condition_confounded_covariate_makes_contrast_nonestimable(
    tmp_path: Path,
) -> None:
    units = _independent_units()
    for unit in units:
        unit["batch"] = unit["condition"]
    _, path = _pseudobulk(tmp_path, units, covariate_keys=("batch",))
    _assert_error(
        "DA_CONTRAST_NOT_ESTIMABLE",
        lambda: _prepare(
            path, covariates=({"key": "batch", "kind": "categorical"},)
        ),
    )


def test_redundant_covariates_and_paired_block_covariate_fail_rank(
    tmp_path: Path,
) -> None:
    independent = _independent_units()
    for index, unit in enumerate(independent):
        unit["first"] = index + (index % 2)
        unit["second"] = unit["first"]
    _, independent_path = _pseudobulk(
        tmp_path / "independent",
        independent,
        covariate_keys=("first", "second"),
    )
    _assert_error(
        "DA_DESIGN_RANK_DEFICIENT",
        lambda: _prepare(
            independent_path,
            covariates=(
                {"key": "first", "kind": "numeric"},
                {"key": "second", "kind": "numeric"},
            ),
        ),
    )

    paired = _paired_units(3)
    ages = {"donor-0": 10, "donor-1": 20, "donor-2": 30}
    for unit in paired:
        unit["age"] = ages[str(unit["replicate"])]
    _, paired_path = _pseudobulk(
        tmp_path / "paired", paired, covariate_keys=("age",)
    )
    _assert_error(
        "DA_DESIGN_RANK_DEFICIENT",
        lambda: _prepare(
            paired_path,
            design_type="paired",
            covariates=({"key": "age", "kind": "numeric"},),
        ),
    )


@pytest.mark.parametrize("invalid", [None, np.nan, np.inf, "not-numeric", True])
def test_numeric_covariate_rejects_missing_nonfinite_or_non_numeric_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid: object
) -> None:
    units = _independent_units()
    for index, unit in enumerate(units):
        unit["age"] = index + 1
    _, path = _pseudobulk(tmp_path, units, covariate_keys=("age",))
    snapshot = da._verified_pseudobulk(path)
    values = list(snapshot.covariate_values)
    values[0] = (invalid,)
    monkeypatch.setattr(
        da,
        "_verified_pseudobulk",
        lambda _: replace(snapshot, covariate_values=tuple(values)),
    )
    _assert_error(
        "DA_COVARIATE_INVALID",
        lambda: _prepare(
            path, covariates=({"key": "age", "kind": "numeric"},)
        ),
    )


def test_full_rank_design_with_one_residual_df_fails(tmp_path: Path) -> None:
    units = _independent_units(2, 2)
    for unit, batch in zip(units, ("a", "b", "a", "b"), strict=True):
        unit["batch"] = batch
    _, path = _pseudobulk(tmp_path, units, covariate_keys=("batch",))
    _assert_error(
        "DA_RESIDUAL_DF_INSUFFICIENT",
        lambda: _prepare(
            path, covariates=({"key": "batch", "kind": "categorical"},)
        ),
    )


def test_selection_preserves_all_row_states_source_and_feature_order(
    tmp_path: Path,
) -> None:
    units = [
        {
            "group": "B",
            "replicate": "b1",
            "condition": "control",
            "zero_library": True,
        },
        {"group": "T", "replicate": "c1", "condition": "control"},
        {"group": "T", "replicate": "x1", "condition": "other"},
        {"group": "T", "replicate": "t1", "condition": "treated"},
        {"group": "T", "replicate": "c2", "condition": "control"},
        {"group": "T", "replicate": "t2", "condition": "treated"},
    ]
    source, path = _pseudobulk(tmp_path, units)
    source_before, artifact_before = _sha(source), _sha(path)

    prepared = _prepare(path)

    assert [row.reason for row in prepared.row_eligibility] == [
        "group_not_selected",
        "included",
        "condition_not_selected",
        "included",
        "included",
        "included",
    ]
    assert prepared.included_source_positions == (1, 3, 4, 5)
    assert prepared.row_eligibility[0].library_size == 0
    assert prepared.included_pseudobulk_ids == tuple(
        prepared.source_row_ids[index] for index in (1, 3, 4, 5)
    )
    assert prepared.feature_ids == ("peak-z", "peak-a", "peak-m")
    assert _sha(source) == source_before
    assert _sha(path) == artifact_before


@pytest.mark.parametrize("semantics", ["fragment_counts", "insertion_counts"])
def test_count_semantics_are_eligible_and_deterministic(
    tmp_path: Path, semantics: str
) -> None:
    _, path = _pseudobulk(tmp_path, _independent_units(), semantics=semantics)
    first = _prepare(path)
    second = _prepare(path)
    assert first.matrix_semantics == semantics
    assert first.output_value_semantics == semantics
    assert first.inclusion_sha256 == second.inclusion_sha256
    assert first.design_sha256 == second.design_sha256
    assert first.contrast_sha256 == second.contrast_sha256
    assert first.preparation_sha256 == second.preparation_sha256
    np.testing.assert_array_equal(first.design_matrix, second.design_matrix)
    np.testing.assert_array_equal(first.contrast, second.contrast)


def test_binary_accessibility_pseudobulk_is_ineligible(tmp_path: Path) -> None:
    _, path = _pseudobulk(
        tmp_path, _independent_units(), semantics="binary_accessibility"
    )
    _assert_error("DA_MATRIX_SEMANTICS_INELIGIBLE", lambda: _prepare(path))


@pytest.mark.parametrize("mutation", ["source", "artifact"])
def test_m81_source_or_pseudobulk_tampering_fails_before_design(
    tmp_path: Path, mutation: str
) -> None:
    source, path = _pseudobulk(tmp_path, _independent_units())
    if mutation == "source":
        raw = ad.read_h5ad(source)
        raw.uns["changed"] = True
        raw.write_h5ad(source)
    else:
        artifact = ad.read_h5ad(path)
        artifact.X[0, 0] += 1
        artifact.write_h5ad(path)
    with pytest.raises(M81ScientificError) as caught:
        _prepare(path)
    assert getattr(caught.value, "code", None) in {
        "FEATURE_SPACE_SOURCE_MISMATCH",
        "PSEUDOBULK_AGGREGATION_MISMATCH",
        "PSEUDOBULK_PROVENANCE_MISMATCH",
    }


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        ("group", "DA_GROUP_NOT_FOUND"),
        ("key", "DA_CONDITION_KEY_MISMATCH"),
        ("condition", "DA_CONDITION_NOT_FOUND"),
        ("same_condition", "DA_CONDITION_NOT_FOUND"),
        ("design", "DA_DESIGN_INVALID"),
    ],
)
def test_structured_comparison_contract_fails_closed(
    tmp_path: Path, mutation: str, code: str
) -> None:
    _, path = _pseudobulk(tmp_path, _independent_units())
    arguments = {
        "pseudobulk_path": path,
        "group_value": "T",
        "condition_key": "condition",
        "numerator_condition": "treated",
        "denominator_condition": "control",
        "design_type": "independent",
    }
    if mutation == "group":
        arguments["group_value"] = "absent"
    elif mutation == "key":
        arguments["condition_key"] = "other_key"
    elif mutation == "condition":
        arguments["numerator_condition"] = "absent"
    elif mutation == "same_condition":
        arguments["numerator_condition"] = "control"
    else:
        arguments["design_type"] = "automatic"
    _assert_error(
        code, lambda: prepare_replicate_differential_accessibility(**arguments)
    )
