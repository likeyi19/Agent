"""Focused offline tests for quantitative clustering evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    homogeneity_score,
    normalized_mutual_info_score,
)

from agent.tools.analysis.clustering_evaluation import (
    _normalized_labels,
    evaluate_cell_clustering,
)
from agent.tools.analysis.embedding_analysis import PROVENANCE_KEY, _cell_order_digest


IDS = tuple(f"cell-{index}" for index in range(8))
TRUE_LABELS = ("a", "a", "b", "b", "c", "c", "d", "d")
PREDICTED = ("0", "0", "1", "1", "2", "3", "3", "3")


def _write_analysis(
    path: Path,
    *,
    ids: tuple[str, ...] = IDS,
    labels: tuple[object, ...] = PREDICTED,
    stage: str = "clustering",
) -> Path:
    obs = pd.DataFrame({"leiden": list(labels)}, index=pd.Index(ids, dtype="object"))
    artifact = ad.AnnData(obs=obs)
    artifact.obsm["X_epizoo"] = np.zeros((len(ids), 512), dtype=np.float32)
    graph = sparse.eye(len(ids), format="csr", dtype=np.float32)
    artifact.obsp["distances"] = graph
    artifact.obsp["connectivities"] = graph.copy()
    artifact.uns["neighbors"] = {
        "distances_key": "distances",
        "connectivities_key": "connectivities",
    }
    parameters: dict[str, object] = {
        "neighbors": {
            "n_neighbors": 2,
            "metric": "euclidean",
            "method": "umap",
            "transformer": "none",
            "random_seed": 0,
            "use_rep": "X_epizoo",
        },
        "clustering": {
            "algorithm": "leiden",
            "resolution": 1.0,
            "flavor": "igraph",
            "n_iterations": 2,
            "directed": False,
            "use_weights": True,
            "random_seed": 0,
            "key_added": "leiden",
        }
    }
    if stage == "umap":
        artifact.obsm["X_umap"] = np.zeros((len(ids), 2), dtype=np.float32)
        parameters["umap"] = {
            "min_dist": 0.5,
            "spread": 1.0,
            "n_components": 2,
            "init_pos": "spectral",
            "random_seed": 0,
            "key_added": "X_umap",
        }
    artifact.uns[PROVENANCE_KEY] = {
        "schema_version": 1,
        "stage": stage,
        "cell_order_sha256": _cell_order_digest(tuple(str(value) for value in ids)),
        "source_analysis_path": str(path.with_name("upstream.h5ad").resolve()),
        "parameters": parameters,
        "software_versions": {},
    }
    artifact.write_h5ad(path)
    return path


def _write_reference(
    path: Path,
    *,
    ids: tuple[str, ...] = IDS,
    labels: tuple[object, ...] = TRUE_LABELS,
    key: str = "celltype",
) -> Path:
    reference = ad.AnnData(
        X=sparse.csr_matrix((len(ids), 100), dtype=np.float32),
        obs=pd.DataFrame({key: list(labels)}, index=pd.Index(ids, dtype="object")),
    )
    reference.write_h5ad(path)
    return path


@pytest.fixture
def sources(tmp_path: Path) -> tuple[Path, Path]:
    return (
        _write_analysis(tmp_path / "clustered.h5ad"),
        _write_reference(tmp_path / "reference.h5ad"),
    )


def test_perfect_clustering_produces_strict_lightweight_report(tmp_path: Path) -> None:
    analysis = _write_analysis(tmp_path / "perfect.h5ad", labels=TRUE_LABELS)
    reference = _write_reference(tmp_path / "reference.h5ad")
    result = evaluate_cell_clustering(analysis, reference, "celltype", tmp_path / "out")
    assert (result["nmi"], result["ari"], result["ami"], result["homogeneity"]) == (
        1.0,
        1.0,
        1.0,
        1.0,
    )
    assert result["finite"] is True and result["cell_order_preserved"] is True
    json.dumps(result, allow_nan=False)
    report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
    json.dumps(report, allow_nan=False)
    assert set(report["metrics"]) == {"nmi", "ari", "ami", "homogeneity"}
    assert "reference_labels" not in result and "predicted_labels" not in result
    assert "reference_labels" not in report and "predicted_labels" not in report
    assert "cell-0" not in json.dumps({"result": result, "report": report})


def test_known_imperfect_scores_match_sklearn(sources: tuple[Path, Path], tmp_path: Path) -> None:
    analysis, reference = sources
    result = evaluate_cell_clustering(analysis, reference, "celltype", tmp_path / "out")
    expected = {
        "nmi": normalized_mutual_info_score(TRUE_LABELS, PREDICTED, average_method="arithmetic"),
        "ari": adjusted_rand_score(TRUE_LABELS, PREDICTED),
        "ami": adjusted_mutual_info_score(TRUE_LABELS, PREDICTED, average_method="arithmetic"),
        "homogeneity": homogeneity_score(TRUE_LABELS, PREDICTED),
    }
    for name, value in expected.items():
        assert result[name] == pytest.approx(value, rel=0, abs=1e-15)


def test_one_predicted_cluster_is_valid(tmp_path: Path) -> None:
    analysis = _write_analysis(tmp_path / "one.h5ad", labels=("0",) * len(IDS))
    reference = _write_reference(tmp_path / "reference.h5ad")
    result = evaluate_cell_clustering(analysis, reference, "celltype", tmp_path / "out")
    assert result["n_predicted_clusters"] == 1
    assert result["homogeneity"] == 0.0


@pytest.mark.parametrize("stage", ["clustering", "umap"])
def test_accepted_milestone6_stages(stage: str, tmp_path: Path) -> None:
    analysis = _write_analysis(tmp_path / f"{stage}.h5ad", stage=stage)
    reference = _write_reference(tmp_path / "reference.h5ad")
    result = evaluate_cell_clustering(analysis, reference, "celltype", tmp_path / "out")
    report = json.loads(Path(result["report_path"]).read_text())
    assert report["provenance"]["analysis_stage"] == stage


@pytest.mark.parametrize(
    ("label_key", "cluster_key", "message"),
    [("", "leiden", "label_key"), ("celltype", "", "cluster_key"), ("missing", "leiden", "lacks label"), ("celltype", "missing", "lacks cluster")],
)
def test_invalid_or_missing_keys(
    sources: tuple[Path, Path], tmp_path: Path, label_key: str, cluster_key: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate_cell_clustering(
            sources[0], sources[1], label_key, tmp_path / "out", cluster_key=cluster_key
        )


@pytest.mark.parametrize(
    ("reference_labels", "predicted_labels", "message"),
    [
        (("a", "a", None, "b", "c", "c", "d", "d"), PREDICTED, "missing"),
        (("a", "a", " ", "b", "c", "c", "d", "d"), PREDICTED, "blank"),
        (TRUE_LABELS, ("0", "0", None, "1", "2", "3", "3", "3"), "invalid|missing"),
        (TRUE_LABELS, ("0", "0", " ", "1", "2", "3", "3", "3"), "blank"),
        (("a",) * len(IDS), PREDICTED, "at least two"),
        ((True, False, True, False, True, False, True, False), PREDICTED, "boolean"),
    ],
)
def test_invalid_labels_are_rejected(
    tmp_path: Path,
    reference_labels: tuple[object, ...],
    predicted_labels: tuple[object, ...],
    message: str,
) -> None:
    analysis = _write_analysis(tmp_path / "analysis.h5ad", labels=predicted_labels)
    reference = _write_reference(tmp_path / "reference.h5ad", labels=reference_labels)
    with pytest.raises(ValueError, match=message):
        evaluate_cell_clustering(analysis, reference, "celltype", tmp_path / "out")


def test_unhashable_label_value_is_rejected() -> None:
    series = pd.Series([["a"], ["b"]], dtype="object")
    with pytest.raises(ValueError, match="unsupported or unhashable"):
        _normalized_labels(series, source="test labels")


@pytest.mark.parametrize(
    ("ids", "message"),
    [
        (IDS[:-1], "cell counts"),
        (IDS[:-1] + ("other",), "identities"),
        (tuple(reversed(IDS)), "order"),
        (IDS[:-1] + (IDS[0],), "unique"),
        (IDS[:-1] + ("",), "nonempty"),
    ],
)
def test_reference_cell_contract_is_strict(
    sources: tuple[Path, Path], tmp_path: Path, ids: tuple[str, ...], message: str
) -> None:
    analysis, _ = sources
    reference = _write_reference(
        tmp_path / "changed.h5ad", ids=ids, labels=TRUE_LABELS[: len(ids)]
    )
    with pytest.raises(ValueError, match=message):
        evaluate_cell_clustering(analysis, reference, "celltype", tmp_path / "out")


def test_malformed_inputs_and_invalid_stage_are_rejected(tmp_path: Path) -> None:
    malformed_reference = tmp_path / "bad-reference.h5ad"
    malformed_reference.write_text("not hdf5")
    analysis = _write_analysis(tmp_path / "analysis.h5ad")
    with pytest.raises(ValueError, match="reference AnnData"):
        evaluate_cell_clustering(analysis, malformed_reference, "celltype", tmp_path / "one")
    malformed_analysis = tmp_path / "bad-analysis.h5ad"
    malformed_analysis.write_text("not hdf5")
    reference = _write_reference(tmp_path / "reference.h5ad")
    with pytest.raises(ValueError, match="Milestone 6 analysis"):
        evaluate_cell_clustering(malformed_analysis, reference, "celltype", tmp_path / "two")
    wrong_stage = _write_analysis(tmp_path / "wrong.h5ad", stage="neighbors")
    with pytest.raises(ValueError, match="unexpected Milestone 6 stage"):
        evaluate_cell_clustering(wrong_stage, reference, "celltype", tmp_path / "three")


def test_existing_report_requires_overwrite(sources: tuple[Path, Path], tmp_path: Path) -> None:
    output = tmp_path / "out"
    evaluate_cell_clustering(*sources, "celltype", output)
    with pytest.raises(FileExistsError, match="overwrite=True"):
        evaluate_cell_clustering(*sources, "celltype", output)
    assert evaluate_cell_clustering(*sources, "celltype", output, overwrite=True)["status"] == "success"


@pytest.mark.parametrize("failure", [RuntimeError("metric failed"), float("nan")])
def test_metric_failure_leaves_no_completed_report(
    sources: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: object
) -> None:
    import agent.tools.analysis.clustering_evaluation as module

    if isinstance(failure, Exception):
        def fail(*args: object, **kwargs: object) -> float:
            raise failure
        replacement = fail
    else:
        replacement = lambda *args, **kwargs: failure
    monkeypatch.setattr(module, "normalized_mutual_info_score", replacement)
    output = tmp_path / "out"
    with pytest.raises((RuntimeError, ValueError)):
        evaluate_cell_clustering(*sources, "celltype", output)
    assert not (output / f"{sources[0].stem}.clustering_metrics.json").exists()
    assert not list(output.glob("*.tmp"))


def test_inputs_unchanged_and_reference_is_opened_backed(
    sources: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import agent.tools.analysis.clustering_evaluation as module

    before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in sources]
    original = module.ad.read_h5ad
    modes: list[object] = []

    def checked(path: object, *args: object, **kwargs: object):
        modes.append(kwargs.get("backed"))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(module.ad, "read_h5ad", checked)
    evaluate_cell_clustering(*sources, "celltype", tmp_path / "out")
    assert modes[-1] == "r" and modes.count("r") == 1
    after = [hashlib.sha256(path.read_bytes()).hexdigest() for path in sources]
    assert after == before
