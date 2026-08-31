"""Focused offline tests for the Milestone 6.1 Scanpy analysis tools."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pytest
from scipy import sparse

from agent.tools.analysis.embedding_analysis import (
    PROVENANCE_KEY,
    build_cell_neighbors,
    cluster_cells,
    compute_cell_umap,
)


@pytest.fixture
def embedding_artifacts(tmp_path: Path) -> tuple[Path, Path, tuple[str, ...]]:
    rng = np.random.default_rng(17)
    first = rng.normal(loc=-1.0, scale=0.25, size=(16, 512))
    second = rng.normal(loc=1.0, scale=0.25, size=(16, 512))
    embeddings = np.vstack((first, second)).astype(np.float32)
    cell_ids = tuple(f"cell-{index:02d}" for index in range(32))
    embedding_path = tmp_path / "cells.epizoo_embeddings.npy"
    cell_ids_path = tmp_path / "cells.epizoo_obs_names.txt"
    np.save(embedding_path, embeddings, allow_pickle=False)
    cell_ids_path.write_text(
        "".join(f"{cell_id}\n" for cell_id in cell_ids), encoding="utf-8"
    )
    return embedding_path, cell_ids_path, cell_ids


@pytest.fixture
def neighbors_result(
    embedding_artifacts: tuple[Path, Path, tuple[str, ...]], tmp_path: Path
) -> dict[str, object]:
    embedding_path, cell_ids_path, _ = embedding_artifacts
    return build_cell_neighbors(embedding_path, cell_ids_path, tmp_path / "neighbors")


@pytest.fixture
def clustering_result(
    neighbors_result: dict[str, object], tmp_path: Path
) -> dict[str, object]:
    return cluster_cells(
        str(neighbors_result["analysis_path"]), tmp_path / "clustering"
    )


def test_successful_neighbors_is_sparse_ordered_and_lightweight(
    neighbors_result: dict[str, object],
    embedding_artifacts: tuple[Path, Path, tuple[str, ...]],
) -> None:
    _, _, cell_ids = embedding_artifacts
    json.dumps(neighbors_result, allow_nan=False)
    assert neighbors_result["status"] == "success"
    assert neighbors_result["n_cells"] == 32
    assert neighbors_result["embedding_dim"] == 512
    assert neighbors_result["finite"] is True
    assert neighbors_result["cell_order_preserved"] is True
    assert not any(
        isinstance(value, np.ndarray) for value in neighbors_result.values()
    )

    artifact = ad.read_h5ad(str(neighbors_result["analysis_path"]))
    assert artifact.shape == (32, 0)
    assert tuple(artifact.obs_names) == cell_ids
    assert artifact.obsm["X_epizoo"].shape == (32, 512)
    assert sparse.issparse(artifact.obsp["distances"])
    assert sparse.issparse(artifact.obsp["connectivities"])
    assert np.isfinite(artifact.obsp["distances"].data).all()
    assert np.isfinite(artifact.obsp["connectivities"].data).all()
    assert artifact.uns[PROVENANCE_KEY]["stage"] == "neighbors"
    assert artifact.uns[PROVENANCE_KEY]["parameters"]["neighbors"]["use_rep"] == (
        "X_epizoo"
    )


def test_successful_leiden_preserves_graph_and_order(
    clustering_result: dict[str, object],
    embedding_artifacts: tuple[Path, Path, tuple[str, ...]],
) -> None:
    _, _, cell_ids = embedding_artifacts
    json.dumps(clustering_result, allow_nan=False)
    assert clustering_result["status"] == "success"
    assert clustering_result["algorithm"] == "leiden"
    assert clustering_result["n_clusters"] >= 1
    assert clustering_result["cell_order_preserved"] is True
    artifact = ad.read_h5ad(str(clustering_result["analysis_path"]))
    assert tuple(artifact.obs_names) == cell_ids
    assert artifact.obs["leiden"].notna().all()
    assert artifact.obs["leiden"].nunique() == clustering_result["n_clusters"]
    assert sparse.issparse(artifact.obsp["distances"])
    assert sparse.issparse(artifact.obsp["connectivities"])
    assert artifact.uns[PROVENANCE_KEY]["stage"] == "clustering"


def test_successful_umap_preserves_labels_graph_order_and_is_lightweight(
    clustering_result: dict[str, object],
    embedding_artifacts: tuple[Path, Path, tuple[str, ...]],
    tmp_path: Path,
) -> None:
    _, _, cell_ids = embedding_artifacts
    result = compute_cell_umap(
        str(clustering_result["analysis_path"]), tmp_path / "umap"
    )
    json.dumps(result, allow_nan=False)
    assert result["status"] == "success"
    assert result["n_components"] == 2
    assert result["finite"] is True
    assert result["cell_order_preserved"] is True
    assert "coordinates" not in result
    artifact = ad.read_h5ad(result["analysis_path"])
    assert tuple(artifact.obs_names) == cell_ids
    assert "leiden" in artifact.obs
    assert sparse.issparse(artifact.obsp["distances"])
    assert sparse.issparse(artifact.obsp["connectivities"])
    assert artifact.obsm["X_umap"].shape == (32, 2)
    assert np.isfinite(artifact.obsm["X_umap"]).all()
    assert artifact.uns[PROVENANCE_KEY]["stage"] == "umap"


def test_repeated_pipeline_is_deterministic(
    embedding_artifacts: tuple[Path, Path, tuple[str, ...]], tmp_path: Path
) -> None:
    embedding_path, cell_ids_path, _ = embedding_artifacts
    outputs: list[ad.AnnData] = []
    for name in ("first", "second"):
        neighbors = build_cell_neighbors(
            embedding_path, cell_ids_path, tmp_path / name, random_seed=0
        )
        clusters = cluster_cells(
            neighbors["analysis_path"], tmp_path / name, random_seed=0
        )
        umap = compute_cell_umap(
            clusters["analysis_path"], tmp_path / name, random_seed=0
        )
        outputs.append(ad.read_h5ad(umap["analysis_path"]))

    graph_difference = (
        outputs[0].obsp["connectivities"] - outputs[1].obsp["connectivities"]
    )
    assert graph_difference.nnz == 0
    assert outputs[0].obs["leiden"].tolist() == outputs[1].obs["leiden"].tolist()
    np.testing.assert_allclose(
        outputs[0].obsm["X_umap"], outputs[1].obsm["X_umap"], rtol=0, atol=0
    )


@pytest.mark.parametrize(
    ("array", "message"),
    [
        (np.ones((32, 511), dtype=np.float32), "dimension"),
        (np.ones((32, 512), dtype=np.float64), "dtype"),
        (np.ones((32,), dtype=np.float32), "two dimensions"),
    ],
)
def test_invalid_embedding_shapes_and_dtype_are_rejected(
    tmp_path: Path, array: np.ndarray, message: str
) -> None:
    embedding_path = tmp_path / "bad.npy"
    ids_path = tmp_path / "ids.txt"
    np.save(embedding_path, array, allow_pickle=False)
    ids_path.write_text(
        "".join(f"cell-{index}\n" for index in range(32)), encoding="utf-8"
    )
    with pytest.raises(ValueError, match=message):
        build_cell_neighbors(embedding_path, ids_path, tmp_path / "output")


def test_nonfinite_embedding_is_rejected(
    embedding_artifacts: tuple[Path, Path, tuple[str, ...]], tmp_path: Path
) -> None:
    embedding_path, ids_path, _ = embedding_artifacts
    array = np.load(embedding_path)
    array[0, 0] = np.nan
    np.save(embedding_path, array, allow_pickle=False)
    with pytest.raises(ValueError, match="non-finite"):
        build_cell_neighbors(embedding_path, ids_path, tmp_path / "output")


@pytest.mark.parametrize("contents", ["cell-0\ncell-0\n", "cell-0\n\ncell-2\n"])
def test_duplicate_or_empty_cell_ids_are_rejected(
    tmp_path: Path, contents: str
) -> None:
    embedding_path = tmp_path / "embedding.npy"
    ids_path = tmp_path / "ids.txt"
    np.save(embedding_path, np.ones((len(contents.splitlines()), 512), np.float32))
    ids_path.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate|empty"):
        build_cell_neighbors(embedding_path, ids_path, tmp_path / "output")


def test_embedding_row_and_id_count_mismatch_is_rejected(
    embedding_artifacts: tuple[Path, Path, tuple[str, ...]], tmp_path: Path
) -> None:
    embedding_path, ids_path, _ = embedding_artifacts
    ids_path.write_text("only-one\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly equal"):
        build_cell_neighbors(embedding_path, ids_path, tmp_path / "output")


@pytest.mark.parametrize("n_neighbors", [0, 1, 32, 33])
def test_invalid_neighbor_counts_are_rejected(
    embedding_artifacts: tuple[Path, Path, tuple[str, ...]],
    tmp_path: Path,
    n_neighbors: int,
) -> None:
    embedding_path, ids_path, _ = embedding_artifacts
    with pytest.raises(ValueError, match="2 <= n_neighbors < n_cells"):
        build_cell_neighbors(
            embedding_path,
            ids_path,
            tmp_path / "output",
            n_neighbors=n_neighbors,
        )


def test_invalid_metric_is_rejected(
    embedding_artifacts: tuple[Path, Path, tuple[str, ...]], tmp_path: Path
) -> None:
    embedding_path, ids_path, _ = embedding_artifacts
    with pytest.raises(ValueError, match="metric"):
        build_cell_neighbors(
            embedding_path, ids_path, tmp_path / "output", metric="manhattan"  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("random_seed", [-1, 0.5, True])
def test_invalid_random_seed_is_rejected(
    embedding_artifacts: tuple[Path, Path, tuple[str, ...]],
    tmp_path: Path,
    random_seed: object,
) -> None:
    embedding_path, ids_path, _ = embedding_artifacts
    with pytest.raises((TypeError, ValueError), match="random_seed"):
        build_cell_neighbors(
            embedding_path,
            ids_path,
            tmp_path / "output",
            random_seed=random_seed,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("resolution", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_resolution_is_rejected(
    neighbors_result: dict[str, object], tmp_path: Path, resolution: float
) -> None:
    with pytest.raises(ValueError, match="resolution"):
        cluster_cells(
            str(neighbors_result["analysis_path"]),
            tmp_path / "clustering",
            resolution=resolution,
        )


@pytest.mark.parametrize(
    ("min_dist", "spread"),
    [(-0.1, 1.0), (0.5, 0.0), (2.0, 1.0), (float("nan"), 1.0)],
)
def test_invalid_umap_parameters_are_rejected(
    clustering_result: dict[str, object],
    tmp_path: Path,
    min_dist: float,
    spread: float,
) -> None:
    with pytest.raises(ValueError, match="min_dist|spread"):
        compute_cell_umap(
            str(clustering_result["analysis_path"]),
            tmp_path / "umap",
            min_dist=min_dist,
            spread=spread,
        )


def test_malformed_upstream_artifact_is_rejected(tmp_path: Path) -> None:
    malformed = tmp_path / "unrelated.h5ad"
    ad.AnnData(np.ones((32, 4), dtype=np.float32)).write_h5ad(malformed)
    with pytest.raises(ValueError, match="compact|provenance"):
        cluster_cells(malformed, tmp_path / "output")


def test_existing_output_requires_explicit_overwrite(
    embedding_artifacts: tuple[Path, Path, tuple[str, ...]], tmp_path: Path
) -> None:
    embedding_path, ids_path, _ = embedding_artifacts
    output_dir = tmp_path / "output"
    build_cell_neighbors(embedding_path, ids_path, output_dir)
    with pytest.raises(FileExistsError, match="overwrite=True"):
        build_cell_neighbors(embedding_path, ids_path, output_dir)


def test_failed_computation_leaves_no_completed_artifact(
    embedding_artifacts: tuple[Path, Path, tuple[str, ...]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scanpy as sc

    embedding_path, ids_path, _ = embedding_artifacts
    output_dir = tmp_path / "output"

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(sc.pp, "neighbors", fail)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        build_cell_neighbors(embedding_path, ids_path, output_dir)
    assert not (output_dir / f"{embedding_path.stem}.neighbors.h5ad").exists()


def test_missing_inputs_are_actionable(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        build_cell_neighbors(
            tmp_path / "missing.npy",
            tmp_path / "missing.txt",
            tmp_path / "output",
        )
