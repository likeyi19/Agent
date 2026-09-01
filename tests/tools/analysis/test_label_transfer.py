from __future__ import annotations

import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from agent.tools.analysis.label_transfer import (
    LABEL_TRANSFER_PROVENANCE_KEY,
    _deterministic_neighbor_indices,
    _embedding_content_digest,
    _file_sha256,
    _model_config_digest,
    _ordered_values_digest,
    _strict_reference_labels,
    transfer_cell_labels,
)


def _write_ids(path: Path, values: list[str]) -> Path:
    path.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")
    return path


def _write_h5ad(path: Path, ids: list[str], labels: object | None = None) -> Path:
    obs = pd.DataFrame(index=pd.Index(ids, dtype="object"))
    if labels is not None:
        obs["celltype"] = labels
    ad.AnnData(obs=obs).write_h5ad(path)
    return path


def _sources(
    tmp_path: Path,
    *,
    reference: np.ndarray | None = None,
    query: np.ndarray | None = None,
    labels: object | None = None,
    reference_ids: list[str] | None = None,
    query_ids: list[str] | None = None,
) -> dict[str, object]:
    if reference is None:
        reference = np.zeros((4, 512), dtype=np.float32)
        reference[0, 0] = 0.0
        reference[1, 0] = 0.2
        reference[2, 0] = 9.8
        reference[3, 0] = 10.0
    if query is None:
        query = np.zeros((2, 512), dtype=np.float32)
        query[0, 0] = 0.1
        query[1, 0] = 9.9
    if labels is None:
        labels = pd.Categorical(["T cell", "T cell", "B cell", "B cell"])
    if reference_ids is None:
        reference_ids = [f"reference-{index}" for index in range(reference.shape[0])]
    if query_ids is None:
        query_ids = [f"query-{index}" for index in range(query.shape[0])]
    reference_embedding = tmp_path / "reference.npy"
    query_embedding = tmp_path / "query.npy"
    np.save(reference_embedding, reference, allow_pickle=False)
    np.save(query_embedding, query, allow_pickle=False)
    reference_ids_path = _write_ids(tmp_path / "reference.txt", reference_ids)
    query_ids_path = _write_ids(tmp_path / "query.txt", query_ids)
    reference_h5ad = _write_h5ad(
        tmp_path / "reference.h5ad", reference_ids, labels
    )
    query_h5ad = _write_h5ad(tmp_path / "query.h5ad", query_ids)
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"test checkpoint provenance")
    return {
        "reference_embedding_path": reference_embedding,
        "reference_cell_ids_path": reference_ids_path,
        "reference_h5ad_path": reference_h5ad,
        "reference_label_key": "celltype",
        "query_embedding_path": query_embedding,
        "query_cell_ids_path": query_ids_path,
        "query_h5ad_path": query_h5ad,
        "output_dir": tmp_path / "output",
        "reference_species": "mouse",
        "query_species": "mouse",
        "reference_checkpoint_path": checkpoint,
        "query_checkpoint_path": checkpoint,
        "n_neighbors": 2,
    }


def _run(arguments: dict[str, object], **changes: object) -> dict[str, object]:
    return transfer_cell_labels(**(arguments | changes))  # type: ignore[arg-type]


def test_separable_transfer_creates_compact_ordered_artifact(tmp_path: Path) -> None:
    arguments = _sources(tmp_path)
    result = _run(arguments)

    assert result["status"] == "success"
    assert result["n_reference_cells"] == 4
    assert result["n_query_cells"] == 2
    assert result["n_reference_classes"] == 2
    assert result["assigned_count"] == 2
    assert result["unassigned_count"] == 0
    assert result["assignment_rate"] == 1.0
    assert result["embedding_dim"] == 512
    assert result["embedding_dtype"] == "float32"
    assert result["cell_order_preserved"] is True
    assert result["annotation_sha256"] == _file_sha256(Path(result["annotation_path"]))
    json.dumps(result, allow_nan=False)
    forbidden = {"predictions", "confidences", "cell_ids", "embeddings", "distances"}
    assert forbidden.isdisjoint(result)

    artifact = ad.read_h5ad(result["annotation_path"])
    assert artifact.X is None
    assert artifact.n_vars == 0
    assert list(artifact.obs_names) == ["query-0", "query-1"]
    assert artifact.obs["predicted_label"].tolist() == ["T cell", "B cell"]
    assert artifact.obs["prediction_confidence"].tolist() == [1.0, 1.0]
    assert artifact.obs["prediction_status"].tolist() == ["assigned", "assigned"]
    assert "annotation_sha256" not in artifact.uns[LABEL_TRANSFER_PROVENANCE_KEY]


def test_multiclass_transfer_preserves_label_text(tmp_path: Path) -> None:
    reference = np.zeros((6, 512), dtype=np.float32)
    reference[:, 0] = [0, 0.1, 5, 5.1, 10, 10.1]
    query = np.zeros((3, 512), dtype=np.float32)
    query[:, 0] = [0.05, 5.05, 10.05]
    labels = pd.Categorical(["alpha", "alpha", "Beta cell", "Beta cell", "γ", "γ"])
    result = _run(_sources(tmp_path, reference=reference, query=query, labels=labels))
    artifact = ad.read_h5ad(result["annotation_path"])
    assert artifact.obs["predicted_label"].tolist() == ["alpha", "Beta cell", "γ"]


def test_repeated_transfer_is_deterministic(tmp_path: Path) -> None:
    arguments = _sources(tmp_path)
    first = _run(arguments)
    second = _run(arguments, output_dir=tmp_path / "second")
    first_artifact = ad.read_h5ad(first["annotation_path"])
    second_artifact = ad.read_h5ad(second["annotation_path"])
    pd.testing.assert_frame_equal(first_artifact.obs, second_artifact.obs)


def test_prediction_is_invariant_to_other_query_cells(tmp_path: Path) -> None:
    arguments = _sources(tmp_path)
    full = _run(arguments)
    full_artifact = ad.read_h5ad(full["annotation_path"])

    query = np.zeros((1, 512), dtype=np.float32)
    query[0, 0] = 0.1
    single_dir = tmp_path / "single"
    single_dir.mkdir()
    single = _run(_sources(single_dir, query=query, query_ids=["query-0"]))
    single_artifact = ad.read_h5ad(single["annotation_path"])
    assert single_artifact.obs.iloc[0].to_dict() == full_artifact.obs.iloc[0].to_dict()


def test_exact_vote_tie_is_unassigned_and_retains_confidence(tmp_path: Path) -> None:
    reference = np.zeros((4, 512), dtype=np.float32)
    reference[:, 0] = [-2, -1, 1, 2]
    query = np.zeros((1, 512), dtype=np.float32)
    labels = pd.Categorical(["A", "A", "B", "B"])
    result = _run(
        _sources(tmp_path, reference=reference, query=query, labels=labels),
        n_neighbors=4,
    )
    artifact = ad.read_h5ad(result["annotation_path"])
    assert pd.isna(artifact.obs["predicted_label"].iloc[0])
    assert artifact.obs["prediction_status"].iloc[0] == "unassigned"
    assert artifact.obs["prediction_confidence"].iloc[0] == 0.5


def test_threshold_rejection_and_default_plurality(tmp_path: Path) -> None:
    reference = np.zeros((5, 512), dtype=np.float32)
    reference[:, 0] = [0, 0.1, 0.2, 0.3, 0.4]
    query = np.zeros((1, 512), dtype=np.float32)
    query[0, 0] = 0.2
    labels = pd.Categorical(["A", "A", "B", "C", "D"])
    arguments = _sources(tmp_path, reference=reference, query=query, labels=labels)
    default = _run(arguments, n_neighbors=5)
    assert ad.read_h5ad(default["annotation_path"]).obs["prediction_status"].iloc[0] == "assigned"
    rejected = _run(
        arguments,
        output_dir=tmp_path / "threshold",
        n_neighbors=5,
        min_confidence=0.5,
    )
    artifact = ad.read_h5ad(rejected["annotation_path"])
    assert artifact.obs["prediction_status"].iloc[0] == "unassigned"
    assert artifact.obs["prediction_confidence"].iloc[0] == 0.4


def test_kth_distance_boundary_uses_reference_index() -> None:
    distances = np.asarray([0.0, 1.0, 1.0, 1.0, 2.0])
    assert _deterministic_neighbor_indices(distances, 3).tolist() == [0, 1, 2]


@pytest.mark.parametrize(
    ("labels", "message"),
    [
        (["A", None, "B", "B"], "missing"),
        (["A", "", "B", "B"], "blank"),
        (["A", " B", "B", "B"], "whitespace"),
        ([True, True, False, False], "string"),
        ([1, 1, 2, 2], "string"),
        (["A", "A", "A", "A"], "two distinct"),
    ],
)
def test_invalid_reference_labels_are_rejected(
    tmp_path: Path, labels: object, message: str
) -> None:
    arguments = _sources(tmp_path, labels=labels)
    with pytest.raises(ValueError, match=message):
        _run(arguments)


def test_arbitrary_object_reference_label_is_rejected() -> None:
    series = pd.Series(["A", object(), "B"])
    with pytest.raises(ValueError, match="string"):
        _strict_reference_labels(series, "celltype")


def test_missing_label_key_is_rejected(tmp_path: Path) -> None:
    arguments = _sources(tmp_path)
    with pytest.raises(ValueError, match="lacks label key"):
        _run(arguments, reference_label_key="missing")


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ({"n_neighbors": 0}, ValueError),
        ({"n_neighbors": 5}, ValueError),
        ({"n_neighbors": True}, TypeError),
        ({"metric": "manhattan"}, ValueError),
        ({"min_confidence": True}, TypeError),
        ({"min_confidence": -0.1}, ValueError),
        ({"min_confidence": 1.1}, ValueError),
        ({"min_confidence": float("nan")}, ValueError),
    ],
)
def test_invalid_scientific_parameters(
    tmp_path: Path, change: dict[str, object], error: type[Exception]
) -> None:
    with pytest.raises(error):
        _run(_sources(tmp_path), **change)


def test_embedding_validation_failures(tmp_path: Path) -> None:
    cases = [
        np.zeros((4, 511), dtype=np.float32),
        np.zeros((4, 512), dtype=np.float64),
        np.full((4, 512), np.nan, dtype=np.float32),
    ]
    for index, reference in enumerate(cases):
        case = tmp_path / str(index)
        case.mkdir()
        with pytest.raises(ValueError):
            _run(_sources(case, reference=reference))


@pytest.mark.parametrize("target", ["reference", "query"])
def test_cosine_rejects_zero_norm_embeddings(tmp_path: Path, target: str) -> None:
    arguments = _sources(tmp_path)
    path = Path(arguments[f"{target}_embedding_path"])
    array = np.load(path)
    array[0] = 0
    np.save(path, array)
    with pytest.raises(ValueError, match="zero-norm"):
        _run(arguments, metric="cosine")


def test_cell_id_validation_and_count_mismatch(tmp_path: Path) -> None:
    arguments = _sources(tmp_path)
    Path(arguments["reference_cell_ids_path"]).write_text(
        "reference-0\nreference-0\nreference-2\nreference-3\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="duplicate"):
        _run(arguments)

    other = tmp_path / "other"
    other.mkdir()
    arguments = _sources(other)
    Path(arguments["query_cell_ids_path"]).write_text("query-0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="count"):
        _run(arguments)

    empty = tmp_path / "empty"
    empty.mkdir()
    arguments = _sources(empty)
    Path(arguments["query_cell_ids_path"]).write_text(
        "query-0\n\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="empty identifier"):
        _run(arguments)


def test_malformed_embedding_and_h5ad_artifacts_are_rejected(tmp_path: Path) -> None:
    arguments = _sources(tmp_path)
    Path(arguments["query_embedding_path"]).write_bytes(b"not-npy")
    with pytest.raises(ValueError, match="embedding npy"):
        _run(arguments)

    other = tmp_path / "other"
    other.mkdir()
    arguments = _sources(other)
    Path(arguments["reference_h5ad_path"]).write_bytes(b"not-hdf5")
    with pytest.raises(ValueError, match="reference AnnData"):
        _run(arguments)


def test_species_checkpoint_and_self_transfer_rejections(tmp_path: Path) -> None:
    arguments = _sources(tmp_path)
    with pytest.raises(ValueError, match="species"):
        _run(arguments, query_species="human")
    other_checkpoint = tmp_path / "other.pth"
    other_checkpoint.write_bytes(b"test checkpoint provenance")
    with pytest.raises(ValueError, match="checkpoint paths"):
        _run(arguments, query_checkpoint_path=other_checkpoint)
    with pytest.raises(ValueError, match="embedding paths"):
        _run(
            arguments,
            query_embedding_path=arguments["reference_embedding_path"],
        )
    with pytest.raises(ValueError, match="raw h5ad paths"):
        _run(arguments, query_h5ad_path=arguments["reference_h5ad_path"])
    with pytest.raises(ValueError, match="sidecar paths"):
        _run(arguments, query_cell_ids_path=arguments["reference_cell_ids_path"])


def test_identical_content_and_ids_rejected_but_partial_overlap_allowed(
    tmp_path: Path,
) -> None:
    reference = np.zeros((4, 512), dtype=np.float32)
    reference[:, 0] = [0, 1, 2, 3]
    identical = tmp_path / "identical"
    identical.mkdir()
    arguments = _sources(
        identical,
        reference=reference,
        query=reference.copy(),
        query_ids=["reference-0", "reference-1", "reference-2", "reference-3"],
    )
    with pytest.raises(ValueError, match="self-transfer"):
        _run(arguments, n_neighbors=2)

    overlap = tmp_path / "overlap"
    overlap.mkdir()
    arguments = _sources(
        overlap,
        query=reference[:2].copy(),
        query_ids=["reference-0", "new-query"],
    )
    assert _run(arguments)["status"] == "success"


def test_output_conflict_occurs_before_knn(monkeypatch, tmp_path: Path) -> None:
    arguments = _sources(tmp_path)
    first = _run(arguments)
    monkeypatch.setattr(
        "agent.tools.analysis.label_transfer._transfer_predictions",
        lambda _: (_ for _ in ()).throw(AssertionError("kNN should not run")),
    )
    with pytest.raises(FileExistsError):
        _run(arguments)
    assert Path(first["annotation_path"]).is_file()


def test_failed_write_leaves_no_completed_artifact(monkeypatch, tmp_path: Path) -> None:
    arguments = _sources(tmp_path)
    monkeypatch.setattr(ad.AnnData, "write_h5ad", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("fail")))
    with pytest.raises(OSError):
        _run(arguments)
    output = Path(arguments["output_dir"])
    assert not (output / "query.label_transfer.h5ad").exists()
    assert not list(output.glob("*.tmp.h5ad"))


def test_source_and_model_digests_are_authoritative(tmp_path: Path) -> None:
    arguments = _sources(tmp_path)
    before = {
        name: hashlib.sha256(Path(arguments[name]).read_bytes()).hexdigest()
        for name in (
            "reference_embedding_path",
            "reference_cell_ids_path",
            "reference_h5ad_path",
            "query_embedding_path",
            "query_cell_ids_path",
            "query_h5ad_path",
        )
    }
    result = _run(arguments)
    reference = np.load(arguments["reference_embedding_path"], mmap_mode="r")
    query = np.load(arguments["query_embedding_path"], mmap_mode="r")
    assert result["reference_embedding_sha256"] == _embedding_content_digest(reference)
    assert result["query_embedding_sha256"] == _embedding_content_digest(query)
    assert result["reference_cell_ids_sha256"] == _ordered_values_digest(
        ("reference-0", "reference-1", "reference-2", "reference-3"),
        domain="agent.cell-ids.v1",
    )
    assert result["query_cell_ids_sha256"] == _ordered_values_digest(
        ("query-0", "query-1"), domain="agent.cell-ids.v1"
    )
    assert result["reference_labels_sha256"] == _ordered_values_digest(
        ("T cell", "T cell", "B cell", "B cell"),
        domain="agent.reference-labels.v1",
    )
    assert result["model_config_sha256"] == _model_config_digest()
    after = {
        name: hashlib.sha256(Path(arguments[name]).read_bytes()).hexdigest()
        for name in before
    }
    assert after == before
