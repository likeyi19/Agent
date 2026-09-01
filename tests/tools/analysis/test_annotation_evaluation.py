from __future__ import annotations

import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from agent.tools.analysis.annotation_evaluation import (
    _confidence_digest,
    _load_annotation_evaluation_report,
    _ordered_strings_digest,
    _predicted_labels_digest,
    _strict_biological_labels,
    evaluate_cell_annotation,
)
from agent.tools.analysis.label_transfer import (
    LABEL_TRANSFER_PROVENANCE_KEY,
    transfer_cell_labels,
)


_VOCABULARY = ["A", "B", "external-X", "unassigned"]


def _write_ids(path: Path, values: list[str]) -> Path:
    path.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")
    return path


def _make_annotation(
    tmp_path: Path,
    predictions: list[str | None],
    statuses: list[str],
    confidences: list[float],
    *,
    cell_ids: list[str] | None = None,
) -> Path:
    n_cells = len(predictions)
    if cell_ids is None:
        cell_ids = [f"query-{index}" for index in range(n_cells)]
    reference = np.zeros((8, 512), dtype=np.float32)
    reference[:, 0] = np.arange(8, dtype=np.float32)
    query = np.zeros((n_cells, 512), dtype=np.float32)
    query[:, 0] = np.arange(n_cells, dtype=np.float32)
    reference_ids = [f"reference-{index}" for index in range(8)]
    reference_labels = pd.Categorical(
        [label for label in _VOCABULARY for _ in range(2)],
        categories=_VOCABULARY,
    )
    reference_embedding = tmp_path / "reference.npy"
    query_embedding = tmp_path / "query.npy"
    np.save(reference_embedding, reference, allow_pickle=False)
    np.save(query_embedding, query, allow_pickle=False)
    reference_ids_path = _write_ids(tmp_path / "reference.txt", reference_ids)
    query_ids_path = _write_ids(tmp_path / "query.txt", cell_ids)
    reference_h5ad = tmp_path / "reference.h5ad"
    ad.AnnData(
        obs=pd.DataFrame(
            {"celltype": reference_labels},
            index=pd.Index(reference_ids, dtype="object"),
        )
    ).write_h5ad(reference_h5ad)
    query_h5ad = tmp_path / "query.h5ad"
    ad.AnnData(obs=pd.DataFrame(index=pd.Index(cell_ids, dtype="object"))).write_h5ad(
        query_h5ad
    )
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint")
    result = transfer_cell_labels(
        reference_embedding,
        reference_ids_path,
        reference_h5ad,
        "celltype",
        query_embedding,
        query_ids_path,
        query_h5ad,
        tmp_path / "transfer",
        reference_species="mouse",
        query_species="mouse",
        reference_checkpoint_path=checkpoint,
        query_checkpoint_path=checkpoint,
        n_neighbors=1,
    )
    annotation_path = Path(result["annotation_path"])
    annotation = ad.read_h5ad(annotation_path)
    annotation.obs["predicted_label"] = pd.Categorical(
        predictions, categories=_VOCABULARY
    )
    annotation.obs["prediction_confidence"] = np.asarray(
        confidences, dtype=np.float64
    )
    annotation.obs["prediction_status"] = pd.Categorical(
        statuses, categories=["assigned", "unassigned"]
    )
    provenance = dict(annotation.uns[LABEL_TRANSFER_PROVENANCE_KEY])
    counts = dict(provenance["counts"])
    assigned_count = statuses.count("assigned")
    counts["assigned_count"] = assigned_count
    counts["unassigned_count"] = n_cells - assigned_count
    provenance["counts"] = counts
    annotation.uns[LABEL_TRANSFER_PROVENANCE_KEY] = provenance
    annotation.write_h5ad(annotation_path)
    return annotation_path


def _make_ground_truth(
    tmp_path: Path,
    labels: object,
    *,
    cell_ids: list[str] | None = None,
    name: str = "ground_truth.h5ad",
) -> Path:
    if cell_ids is None:
        cell_ids = [f"query-{index}" for index in range(len(labels))]
    path = tmp_path / name
    obs = pd.DataFrame(index=pd.Index(cell_ids, dtype="object"))
    obs["truth"] = labels
    ad.AnnData(obs=obs).write_h5ad(path)
    return path


def _evaluate(
    tmp_path: Path,
    truth: object,
    predictions: list[str | None],
    statuses: list[str],
    confidences: list[float],
) -> tuple[dict[str, object], dict[str, object], Path, Path]:
    annotation_path = _make_annotation(
        tmp_path, predictions, statuses, confidences
    )
    truth_path = _make_ground_truth(tmp_path, truth)
    result = evaluate_cell_annotation(
        annotation_path, truth_path, "truth", tmp_path / "evaluation"
    )
    report_path = Path(result["report_path"])
    report = _load_annotation_evaluation_report(report_path)
    return result, report, annotation_path, truth_path


def test_perfect_evaluation_is_lightweight_and_json_safe(tmp_path: Path) -> None:
    result, report, annotation_path, _ = _evaluate(
        tmp_path,
        pd.Categorical(["A", "A", "B", "B"]),
        ["A", "A", "B", "B"],
        ["assigned"] * 4,
        [0.8, 0.9, 1.0, 0.7],
    )
    assert result["status"] == "success"
    assert result["n_cells"] == 4
    assert result["n_ground_truth_classes"] == 2
    assert result["n_assigned_predicted_classes"] == 2
    assert result["assigned_count"] == 4
    assert result["unassigned_count"] == 0
    assert result["correct_assigned_count"] == 4
    assert result["incorrect_assigned_count"] == 0
    assert result["overall_accuracy"] == 1.0
    assert result["assigned_accuracy"] == 1.0
    assert result["macro_f1"] == 1.0
    assert result["finite"] is True
    assert result["cell_order_preserved"] is True
    assert result["annotation_sha256"] == hashlib.sha256(
        annotation_path.read_bytes()
    ).hexdigest()
    json.dumps(result, allow_nan=False)
    forbidden = {
        "ground_truth_labels",
        "predicted_labels",
        "prediction_status",
        "prediction_confidence",
        "cell_ids",
        "confusion",
        "per_class",
    }
    assert forbidden.isdisjoint(result)
    assert set(report) == {
        "schema_version",
        "artifact_type",
        "status",
        "inputs",
        "counts",
        "metrics",
        "confidence_diagnostics",
        "per_class",
        "confusion",
        "metric_backend",
        "validation",
        "provenance",
        "software_versions",
    }
    serialized = json.dumps(report, allow_nan=False)
    for forbidden_name in ("query-0", "prediction_vector", "confidence_vector"):
        assert forbidden_name not in serialized


def test_mixed_assignment_metrics_confusion_and_medians(tmp_path: Path) -> None:
    result, report, _, _ = _evaluate(
        tmp_path,
        ["B", "A", "B", "A", "B"],
        ["B", "B", None, "external-X", "B"],
        ["assigned", "assigned", "unassigned", "assigned", "assigned"],
        [0.9, 0.4, 0.2, 0.6, 1.0],
    )
    assert result["n_ground_truth_classes"] == 2
    assert result["n_assigned_predicted_classes"] == 2
    assert result["assigned_count"] == 4
    assert result["unassigned_count"] == 1
    assert result["assignment_rate"] == pytest.approx(0.8)
    assert result["correct_assigned_count"] == 2
    assert result["incorrect_assigned_count"] == 2
    assert result["overall_accuracy"] == pytest.approx(0.4)
    assert result["assigned_accuracy"] == pytest.approx(0.5)
    assert result["median_confidence"] == pytest.approx(0.6)
    assert result["median_assigned_confidence"] == pytest.approx(0.75)
    assert result["median_correct_assigned_confidence"] == pytest.approx(0.95)
    assert result["median_incorrect_assigned_confidence"] == pytest.approx(0.5)
    assert [entry["label"] for entry in report["per_class"]] == ["B", "A"]
    assert [entry["support"] for entry in report["per_class"]] == [3, 2]
    assert [entry["true_positive"] for entry in report["per_class"]] == [2, 0]
    assert report["confusion"] == {
        "rows": {
            "kind": "ground_truth_biological_label",
            "labels": ["B", "A"],
        },
        "columns": [
            {"kind": "biological_label", "label": "B"},
            {"kind": "biological_label", "label": "external-X"},
            {"kind": "structural_unassigned", "label": None},
        ],
        "counts": [[2, 0, 1], [1, 1, 0]],
    }
    assert sum(map(sum, report["confusion"]["counts"])) == 5


@pytest.mark.parametrize(
    ("truth", "predictions", "statuses", "expected"),
    [
        (["A", "B"], ["A", None], ["assigned", "unassigned"], 0.5),
        (["A", "B"], ["external-X", "B"], ["assigned", "assigned"], 0.5),
        (["A", "A"], ["A", None], ["assigned", "unassigned"], 2 / 3),
        (["A", "B"], [None, None], ["unassigned", "unassigned"], 0.0),
    ],
)
def test_required_macro_f1_examples(
    tmp_path: Path,
    truth: list[str],
    predictions: list[str | None],
    statuses: list[str],
    expected: float,
) -> None:
    result, _, _, _ = _evaluate(
        tmp_path, truth, predictions, statuses, [0.5] * len(truth)
    )
    assert result["macro_f1"] == pytest.approx(expected)


def test_all_unassigned_uses_json_null_for_undefined_values(tmp_path: Path) -> None:
    result, report, _, _ = _evaluate(
        tmp_path,
        ["A", "B"],
        [None, None],
        ["unassigned", "unassigned"],
        [0.4, 0.6],
    )
    assert result["assigned_accuracy"] is None
    assert result["median_assigned_confidence"] is None
    assert result["median_correct_assigned_confidence"] is None
    assert result["median_incorrect_assigned_confidence"] is None
    assert result["median_confidence"] == pytest.approx(0.5)
    text = json.dumps(report, allow_nan=False)
    assert "NaN" not in text and "Infinity" not in text
    assert report["metrics"]["assigned_accuracy"] is None


def test_literal_unassigned_label_is_not_structural_status(tmp_path: Path) -> None:
    result, report, _, _ = _evaluate(
        tmp_path,
        ["unassigned", "A"],
        ["unassigned", None],
        ["assigned", "unassigned"],
        [0.9, 0.2],
    )
    assert result["correct_assigned_count"] == 1
    assert report["confusion"]["columns"] == [
        {"kind": "biological_label", "label": "unassigned"},
        {"kind": "structural_unassigned", "label": None},
    ]


def test_semantic_provenance_digests_match_ordered_values(tmp_path: Path) -> None:
    predictions = ["A", None, "external-X"]
    statuses = ["assigned", "unassigned", "assigned"]
    confidences = np.asarray([0.9, 0.3, 0.6], dtype=np.float64)
    _, report, _, _ = _evaluate(
        tmp_path, ["A", "B", "B"], predictions, statuses, confidences.tolist()
    )
    provenance = report["provenance"]
    assert provenance["query_cell_ids_sha256"] == _ordered_strings_digest(
        ["query-0", "query-1", "query-2"],
        domain="agent.annotation-query-cell-ids.v1",
    )
    assert provenance["ground_truth_labels_sha256"] == _ordered_strings_digest(
        ["A", "B", "B"], domain="agent.annotation-ground-truth-labels.v1"
    )
    assert provenance["predicted_labels_sha256"] == _predicted_labels_digest(
        predictions
    )
    assert provenance["prediction_status_sha256"] == _ordered_strings_digest(
        statuses, domain="agent.annotation-prediction-status.v1"
    )
    assert provenance["prediction_confidence_sha256"] == _confidence_digest(
        confidences
    )


@pytest.mark.parametrize(
    ("cell_ids", "message"),
    [
        (["query-1", "query-0"], "different exact order"),
        (["query-0", "other"], "identities"),
        (["query-0"], "identical cell counts"),
        (["query-0", "query-0"], "unique"),
        (["query-0", ""], "nonempty"),
    ],
)
def test_ground_truth_cell_identity_failures(
    tmp_path: Path, cell_ids: list[str], message: str
) -> None:
    annotation = _make_annotation(
        tmp_path, ["A", "B"], ["assigned", "assigned"], [1.0, 1.0]
    )
    truth = _make_ground_truth(tmp_path, ["A"] * len(cell_ids), cell_ids=cell_ids)
    with pytest.raises(ValueError, match=message):
        evaluate_cell_annotation(annotation, truth, "truth", tmp_path / "output")


@pytest.mark.parametrize(
    ("labels", "message"),
    [
        (["A", None], "missing"),
        (["A", ""], "blank"),
        (["A", " B"], "whitespace"),
        ([True, False], "string"),
        ([1, 2], "string"),
    ],
)
def test_invalid_ground_truth_labels(
    tmp_path: Path, labels: object, message: str
) -> None:
    annotation = _make_annotation(
        tmp_path, ["A", "B"], ["assigned", "assigned"], [1.0, 1.0]
    )
    truth = _make_ground_truth(tmp_path, labels)
    with pytest.raises(ValueError, match=message):
        evaluate_cell_annotation(annotation, truth, "truth", tmp_path / "output")


def test_missing_or_blank_ground_truth_key(tmp_path: Path) -> None:
    annotation = _make_annotation(
        tmp_path, ["A"], ["assigned"], [1.0]
    )
    truth = _make_ground_truth(tmp_path, ["A"])
    with pytest.raises(ValueError, match="nonempty"):
        evaluate_cell_annotation(annotation, truth, "", tmp_path / "blank")
    with pytest.raises(ValueError, match="lacks label key"):
        evaluate_cell_annotation(annotation, truth, "absent", tmp_path / "missing")


def test_arbitrary_object_ground_truth_label_is_rejected_directly() -> None:
    with pytest.raises(ValueError, match="string"):
        _strict_biological_labels(
            pd.Series(["A", object()], dtype="object"), source="Ground truth"
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("schema", "schema, type, or stage"),
        ("artifact_type", "schema, type, or stage"),
        ("stage", "schema, type, or stage"),
        ("extra_column", "observation columns"),
        ("missing_column", "observation columns"),
        ("assigned_missing", "Assigned"),
        ("unassigned_label", "Unassigned"),
        ("nonfinite_confidence", "finite"),
        ("outside_confidence", r"\[0, 1\]"),
        ("invalid_status", "status"),
    ],
)
def test_malformed_annotation_is_rejected(
    tmp_path: Path, mutation: str, message: str
) -> None:
    annotation_path = _make_annotation(
        tmp_path, ["A", "B"], ["assigned", "assigned"], [0.9, 0.8]
    )
    annotation = ad.read_h5ad(annotation_path)
    provenance = dict(annotation.uns[LABEL_TRANSFER_PROVENANCE_KEY])
    if mutation in {"schema", "artifact_type", "stage"}:
        key = {"schema": "schema_version"}.get(mutation, mutation)
        provenance[key] = "invalid"
        annotation.uns[LABEL_TRANSFER_PROVENANCE_KEY] = provenance
    elif mutation == "extra_column":
        annotation.obs["extra"] = 1
    elif mutation == "missing_column":
        del annotation.obs["prediction_confidence"]
    elif mutation == "assigned_missing":
        annotation.obs["predicted_label"] = pd.Categorical(
            [None, "B"], categories=_VOCABULARY
        )
    elif mutation == "unassigned_label":
        annotation.obs["prediction_status"] = pd.Categorical(
            ["unassigned", "assigned"], categories=["assigned", "unassigned"]
        )
    elif mutation == "nonfinite_confidence":
        annotation.obs["prediction_confidence"] = [np.nan, 0.8]
    elif mutation == "outside_confidence":
        annotation.obs["prediction_confidence"] = [1.1, 0.8]
    elif mutation == "invalid_status":
        annotation.obs["prediction_status"] = pd.Categorical(
            ["invalid", "assigned"],
            categories=["assigned", "unassigned", "invalid"],
        )
    annotation.write_h5ad(annotation_path)
    truth = _make_ground_truth(tmp_path, ["A", "B"])
    with pytest.raises(ValueError, match=message):
        evaluate_cell_annotation(annotation_path, truth, "truth", tmp_path / "output")


def test_output_conflict_and_source_files_unchanged(tmp_path: Path) -> None:
    annotation = _make_annotation(
        tmp_path, ["A", "B"], ["assigned", "assigned"], [1.0, 1.0]
    )
    truth = _make_ground_truth(tmp_path, ["A", "B"])
    annotation_before = annotation.read_bytes()
    truth_before = truth.read_bytes()
    first = evaluate_cell_annotation(annotation, truth, "truth", tmp_path / "output")
    with pytest.raises(FileExistsError, match="overwrite=True"):
        evaluate_cell_annotation(annotation, truth, "truth", tmp_path / "output")
    second = evaluate_cell_annotation(
        annotation, truth, "truth", tmp_path / "output", overwrite=True
    )
    assert first["report_path"] == second["report_path"]
    assert annotation.read_bytes() == annotation_before
    assert truth.read_bytes() == truth_before


def test_duplicate_json_keys_and_nonfinite_constants_are_rejected(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    with pytest.raises(ValueError, match="Unable to read"):
        _load_annotation_evaluation_report(duplicate)
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value":NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="Unable to read"):
        _load_annotation_evaluation_report(nonfinite)


def test_atomic_replace_failure_leaves_no_final_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    annotation = _make_annotation(
        tmp_path, ["A"], ["assigned"], [1.0]
    )
    truth = _make_ground_truth(tmp_path, ["A"])
    import agent.tools.analysis.annotation_evaluation as module

    def fail_replace(source: object, destination: object) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(module.os, "replace", fail_replace)
    output_dir = tmp_path / "output"
    with pytest.raises(OSError, match="replace failure"):
        evaluate_cell_annotation(annotation, truth, "truth", output_dir)
    assert not (output_dir / f"{annotation.stem}.annotation_evaluation.json").exists()
    assert list(output_dir.glob("*.tmp")) == []
