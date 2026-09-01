"""Deterministic scientific visualization downstream of verified evidence."""

from __future__ import annotations

from dataclasses import dataclass
import colorsys
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import struct
import tempfile
from typing import Literal, Mapping, Sequence, TypedDict

import anndata as ad
from matplotlib import ft2font, font_manager, rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np

from agent.orchestration.registry import ToolRegistry
from agent.schemas import (
    AgentError,
    AgentRunResult,
    ErrorCategory,
    JsonValue,
    VerificationCheck,
    VerificationResult,
)
from agent.tools.analysis.annotation_evaluation import (
    _load_annotation_evaluation_report,
)

from .evidence import (
    ANALYSIS_EVIDENCE_ARTIFACT_TYPE,
    ANALYSIS_EVIDENCE_SCHEMA_VERSION,
    AnalysisEvidenceResult,
    verify_analysis_evidence,
)


ANALYSIS_VISUALIZATION_SCHEMA_VERSION = 1
ANALYSIS_VISUALIZATION_ARTIFACT_TYPE = "agent.analysis-visualizations"
ANALYSIS_VISUALIZATION_BUNDLE_DIRNAME = "analysis_visualizations"
ANALYSIS_VISUALIZATION_MANIFEST_FILENAME = "visualization_manifest.json"
PLOTTING_SPEC_VERSION = 1

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_DPI = 200
_FONT_FAMILY = "DejaVu Sans"
_UMAP_ALPHA = 0.8
_CONFUSION_TEXT_CELL_LIMIT = 400
_PRIMARY_PALETTE = (
    "#1f77b4",
    "#aec7e8",
    "#ff7f0e",
    "#ffbb78",
    "#2ca02c",
    "#98df8a",
    "#d62728",
    "#ff9896",
    "#9467bd",
    "#c5b0d5",
    "#8c564b",
    "#c49c94",
    "#e377c2",
    "#f7b6d2",
    "#7f7f7f",
    "#c7c7c7",
    "#bcbd22",
    "#dbdb8d",
    "#17becf",
    "#9edae5",
)
_METRIC_ORDER = ("NMI", "ARI", "AMI", "Homogeneity")
_METRIC_FACT_KEYS = ("nmi", "ari", "ami", "homogeneity")
_METRIC_COLORS = ("#4c78a8", "#f58518", "#54a24b", "#b279a2")
_RC_PARAMS: Mapping[str, object] = {
    "font.family": _FONT_FAMILY,
    "font.size": 9.0,
    "axes.titlesize": 11.0,
    "axes.labelsize": 10.0,
    "axes.linewidth": 0.8,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.transparent": False,
    "path.simplify": False,
    "agg.path.chunksize": 0,
}


class VisualizationFigureResult(TypedDict):
    """Compact identity for one rendered visualization figure."""

    figure_id: str
    figure_kind: str
    figure_path: str
    png_sha256: str


class AnalysisVisualizationResult(TypedDict):
    """Lightweight result for a completed visualization bundle."""

    status: Literal["success"]
    manifest_path: str
    manifest_sha256: str
    bundle_path: str
    schema_version: int
    artifact_type: str
    evidence_path: str
    evidence_sha256: str
    run_id: str
    request_id: str
    plan_id: str
    n_figures: int
    figures: list[VisualizationFigureResult]


class AnalysisVisualizationError(ValueError):
    """Fail-closed visualization error with a stable public code."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class _EvidenceSnapshot:
    path: Path
    sha256: str
    payload: Mapping[str, object]
    run_id: str
    request_id: str
    plan_id: str


@dataclass(frozen=True)
class _UmapData:
    cell_ids: tuple[str, ...]
    coordinates: np.ndarray
    labels: tuple[str, ...]
    categories: tuple[str, ...]
    palette: tuple[str, ...]
    point_size: float


@dataclass(frozen=True)
class _MetricData:
    names: tuple[str, ...]
    values: tuple[float, ...]


@dataclass(frozen=True)
class _ConfusionData:
    row_labels: tuple[str, ...]
    column_labels: tuple[str, ...]
    counts: np.ndarray
    annotate_cells: bool


@dataclass(frozen=True)
class _FigureProjection:
    figure_id: str
    figure_kind: str
    filename: str
    source_step_ids: tuple[str, ...]
    source_artifacts: tuple[Mapping[str, JsonValue], ...]
    plotting_data_sha256: str
    width_px: int
    height_px: int
    dpi: int
    presentation: Mapping[str, JsonValue]
    data: _UmapData | _MetricData | _ConfusionData


class _DuplicateJsonKey(ValueError):
    pass


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKey(f"Duplicate JSON key {key!r}.")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ValueError(f"Non-finite JSON constant {value!r} is not permitted.")


def _plain_json(value: object) -> JsonValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AnalysisVisualizationError(
                "VISUALIZATION_VALUE_INVALID",
                "Visualization metadata cannot contain non-finite values.",
            )
        return value
    if isinstance(value, Mapping):
        copied: dict[str, JsonValue] = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise AnalysisVisualizationError(
                    "VISUALIZATION_VALUE_INVALID",
                    "Visualization metadata requires string mapping keys.",
                )
            copied[key] = _plain_json(nested)
        return copied
    if isinstance(value, (list, tuple)):
        return tuple(_plain_json(nested) for nested in value)
    raise AnalysisVisualizationError(
        "VISUALIZATION_VALUE_INVALID",
        "Visualization metadata contains an unsupported value type.",
    )


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise AnalysisVisualizationError(
            "VISUALIZATION_NOT_JSON_SAFE",
            "Visualization metadata is not strict JSON-safe data.",
        ) from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(chunk_size):
                digest.update(chunk)
    except OSError as exc:
        raise AnalysisVisualizationError(
            "VISUALIZATION_ARTIFACT_UNAVAILABLE",
            "A visualization artifact could not be read.",
        ) from exc
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _domain_digest(domain: str, value: object) -> str:
    return _sha256_bytes(
        _canonical_json_bytes({"domain": domain, "value": _plain_json(value)})
    )


def _ordered_strings_digest(values: Sequence[str], *, domain: str) -> str:
    digest = hashlib.sha256()
    prefix = _canonical_json_bytes({"domain": domain, "count": len(values)})
    digest.update(len(prefix).to_bytes(8, "big"))
    digest.update(prefix)
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _umap_plot_digest(
    cell_ids: tuple[str, ...], coordinates: np.ndarray, labels: tuple[str, ...]
) -> str:
    contiguous = np.ascontiguousarray(coordinates)
    digest = hashlib.sha256()
    header = _canonical_json_bytes(
        {
            "domain": "agent.visualization.umap-leiden-data.v1",
            "shape": tuple(int(value) for value in contiguous.shape),
            "dtype": contiguous.dtype.str,
            "cell_ids_sha256": _ordered_strings_digest(
                cell_ids, domain="agent.visualization.umap-cell-ids.v1"
            ),
            "labels_sha256": _ordered_strings_digest(
                labels, domain="agent.visualization.umap-leiden-labels.v1"
            ),
        }
    )
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _evidence_path(value: str | Path | AnalysisEvidenceResult) -> Path:
    if isinstance(value, Mapping):
        path = value.get("evidence_path")
        if not isinstance(path, str) or not path:
            raise AnalysisVisualizationError(
                "VISUALIZATION_EVIDENCE_INVALID",
                "AnalysisEvidenceResult lacks a valid evidence path.",
            )
        return Path(path).expanduser().resolve()
    if not isinstance(value, (str, Path)):
        raise TypeError("`evidence` must be a path or AnalysisEvidenceResult mapping.")
    return Path(value).expanduser().resolve()


def _verified_evidence_snapshot(
    run_result: AgentRunResult,
    evidence: str | Path | AnalysisEvidenceResult,
    registry: ToolRegistry,
) -> _EvidenceSnapshot:
    verification = verify_analysis_evidence(run_result, evidence, registry=registry)
    if not verification.passed:
        code = (
            verification.error.code
            if verification.error is not None
            else "VISUALIZATION_EVIDENCE_INVALID"
        )
        raise AnalysisVisualizationError(
            code,
            "Visualization requires valid freshly verified AnalysisEvidence.",
        )
    path = _evidence_path(evidence)
    try:
        payload_bytes = path.read_bytes()
        parsed = json.loads(
            payload_bytes.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except Exception as exc:
        raise AnalysisVisualizationError(
            "VISUALIZATION_EVIDENCE_INVALID",
            "Verified analysis evidence could not be strictly loaded.",
        ) from exc
    if not isinstance(parsed, dict):
        raise AnalysisVisualizationError(
            "VISUALIZATION_EVIDENCE_INVALID",
            "Verified analysis evidence must contain one JSON object.",
        )
    run = parsed.get("run")
    if (
        parsed.get("schema_version") != ANALYSIS_EVIDENCE_SCHEMA_VERSION
        or parsed.get("artifact_type") != ANALYSIS_EVIDENCE_ARTIFACT_TYPE
        or parsed.get("status") != "success"
        or not isinstance(run, dict)
        or run.get("run_id") != run_result.run_id
        or run.get("request_id") != run_result.request_id
        or run_result.plan is None
        or run.get("plan_id") != run_result.plan.plan_id
    ):
        raise AnalysisVisualizationError(
            "VISUALIZATION_EVIDENCE_INVALID",
            "Verified analysis evidence has inconsistent source identity.",
        )
    digest = _sha256_bytes(payload_bytes)
    if isinstance(evidence, Mapping) and digest != evidence.get("evidence_sha256"):
        raise AnalysisVisualizationError(
            "EVIDENCE_SHA256_MISMATCH",
            "Verified analysis evidence differs from its authoritative digest.",
        )
    return _EvidenceSnapshot(
        path=path,
        sha256=digest,
        payload=parsed,
        run_id=run_result.run_id,
        request_id=run_result.request_id,
        plan_id=run_result.plan.plan_id,
    )


def _first_occurrence(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _rgb_hex(red: float, green: float, blue: float) -> str:
    channels = tuple(
        min(255, max(0, int(round(component * 255.0))))
        for component in (red, green, blue)
    )
    return "#" + "".join(f"{value:02x}" for value in channels)


def _categorical_palette(count: int) -> tuple[str, ...]:
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("Palette size must be a positive integer.")
    colors = list(_PRIMARY_PALETTE[:count])
    golden_ratio = 0.6180339887498949
    for extension_index in range(max(0, count - len(_PRIMARY_PALETTE))):
        hue = (0.08333333333333333 + extension_index * golden_ratio) % 1.0
        saturation = (0.68, 0.82, 0.58)[extension_index % 3]
        value = (0.88, 0.76)[(extension_index // 3) % 2]
        colors.append(_rgb_hex(*colorsys.hsv_to_rgb(hue, saturation, value)))
    return tuple(colors)


def _umap_point_size(n_cells: int) -> float:
    if isinstance(n_cells, bool) or not isinstance(n_cells, int) or n_cells <= 0:
        raise ValueError("UMAP point sizing requires a positive cell count.")
    return float(max(1.0, min(12.0, 12000.0 / n_cells)))


def _source_artifact(
    artifacts: Sequence[object], *, step_id: str, kind: str, tool_name: str
) -> Mapping[str, JsonValue]:
    matches = [
        value
        for value in artifacts
        if isinstance(value, Mapping)
        and value.get("producing_step_id") == step_id
        and value.get("artifact_kind") == kind
    ]
    if len(matches) != 1:
        raise AnalysisVisualizationError(
            "VISUALIZATION_SOURCE_BINDING_INVALID",
            "A supported figure requires exactly one explicit evidence artifact.",
        )
    artifact = matches[0]
    required = {
        "producing_step_id",
        "tool_name",
        "result_field",
        "artifact_kind",
        "artifact_path",
        "integrity",
    }
    if (
        set(artifact) != required
        or artifact.get("tool_name") != tool_name
        or not isinstance(artifact.get("artifact_path"), str)
        or not artifact.get("artifact_path")
        or not isinstance(artifact.get("integrity"), Mapping)
    ):
        raise AnalysisVisualizationError(
            "VISUALIZATION_SOURCE_BINDING_INVALID",
            "A supported figure has invalid evidence artifact metadata.",
        )
    copied = _plain_json(artifact)
    assert isinstance(copied, Mapping)
    return copied


def _read_umap_presentation(path: Path, *, expected_n_cells: object) -> _UmapData:
    try:
        artifact = ad.read_h5ad(path, backed="r")
    except Exception as exc:
        raise AnalysisVisualizationError(
            "VISUALIZATION_SOURCE_INVALID",
            "The verified UMAP artifact could not be presentation-read.",
        ) from exc
    try:
        cell_ids = tuple(str(value) for value in artifact.obs_names)
        if (
            not cell_ids
            or len(set(cell_ids)) != len(cell_ids)
            or any(not value for value in cell_ids)
            or expected_n_cells != len(cell_ids)
        ):
            raise ValueError("Invalid UMAP cell identifiers.")
        coordinates = np.asarray(artifact.obsm["X_umap"])
        if coordinates.shape != (len(cell_ids), 2) or not np.isfinite(coordinates).all():
            raise ValueError("Invalid UMAP coordinates.")
        coordinates = np.array(coordinates, copy=True, order="C")
        label_series = artifact.obs["leiden"]
        if len(label_series) != len(cell_ids) or label_series.isna().any():
            raise ValueError("Invalid Leiden labels.")
        labels = tuple(str(value) for value in label_series.tolist())
        if any(not value for value in labels):
            raise ValueError("Invalid Leiden labels.")
    except Exception as exc:
        raise AnalysisVisualizationError(
            "VISUALIZATION_SOURCE_INVALID",
            "The verified UMAP artifact lacks valid presentation data.",
        ) from exc
    finally:
        file_manager = getattr(artifact, "file", None)
        if file_manager is not None:
            file_manager.close()
    categories = _first_occurrence(labels)
    return _UmapData(
        cell_ids=cell_ids,
        coordinates=coordinates,
        labels=labels,
        categories=categories,
        palette=_categorical_palette(len(categories)),
        point_size=_umap_point_size(len(cell_ids)),
    )


def _metric_data(facts: Mapping[str, object]) -> _MetricData:
    values: list[float] = []
    for name, key in zip(_METRIC_ORDER, _METRIC_FACT_KEYS, strict=True):
        value = facts.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AnalysisVisualizationError(
                "VISUALIZATION_SOURCE_INVALID",
                "Clustering evaluation evidence lacks a required metric.",
            )
        score = float(value)
        lower = -1.0 if name in {"ARI", "AMI"} else 0.0
        if not math.isfinite(score) or not lower <= score <= 1.0:
            raise AnalysisVisualizationError(
                "VISUALIZATION_SOURCE_INVALID",
                "Clustering evaluation evidence contains an invalid metric.",
            )
        values.append(score)
    return _MetricData(_METRIC_ORDER, tuple(values))


def _read_confusion_presentation(path: Path) -> _ConfusionData:
    try:
        report = _load_annotation_evaluation_report(path)
        confusion = report["confusion"]
        assert isinstance(confusion, Mapping)
        rows = confusion["rows"]
        columns = confusion["columns"]
        matrix = confusion["counts"]
        assert isinstance(rows, Mapping) and isinstance(columns, list)
        row_labels = tuple(str(value) for value in rows["labels"])
        column_labels: list[str] = []
        for descriptor in columns:
            assert isinstance(descriptor, Mapping)
            if descriptor.get("kind") == "structural_unassigned":
                column_labels.append("Unassigned (structural)")
            else:
                label = descriptor.get("label")
                if not isinstance(label, str):
                    raise ValueError("Invalid confusion column label.")
                column_labels.append(label)
        counts = np.asarray(matrix, dtype=np.int64)
        if (
            counts.shape != (len(row_labels), len(column_labels))
            or np.any(counts < 0)
        ):
            raise ValueError("Invalid confusion matrix.")
        counts = np.array(counts, copy=True, order="C")
    except Exception as exc:
        raise AnalysisVisualizationError(
            "VISUALIZATION_SOURCE_INVALID",
            "The verified annotation-evaluation report lacks valid confusion data.",
        ) from exc
    return _ConfusionData(
        row_labels=row_labels,
        column_labels=tuple(column_labels),
        counts=counts,
        annotate_cells=counts.size <= _CONFUSION_TEXT_CELL_LIMIT,
    )


def _figure_identity(ordinal: int, kind: str, step_id: str) -> tuple[str, str]:
    step_hash = hashlib.sha256(step_id.encode("utf-8")).hexdigest()[:12]
    figure_id = f"{ordinal:03d}_{kind}_{step_hash}"
    return figure_id, f"{figure_id}.png"


def _derive_figure_projections(snapshot: _EvidenceSnapshot) -> tuple[_FigureProjection, ...]:
    workflow = snapshot.payload.get("workflow")
    steps = snapshot.payload.get("steps")
    artifacts = snapshot.payload.get("artifacts")
    if (
        not isinstance(workflow, Mapping)
        or not isinstance(workflow.get("ordered_steps"), list)
        or not isinstance(steps, list)
        or not isinstance(artifacts, list)
    ):
        raise AnalysisVisualizationError(
            "VISUALIZATION_EVIDENCE_INVALID",
            "Verified evidence lacks its workflow projection.",
        )
    step_records: dict[str, Mapping[str, object]] = {}
    for value in steps:
        if not isinstance(value, Mapping) or not isinstance(value.get("step_id"), str):
            raise AnalysisVisualizationError(
                "VISUALIZATION_EVIDENCE_INVALID",
                "Verified evidence has invalid step metadata.",
            )
        step_id = str(value["step_id"])
        if step_id in step_records:
            raise AnalysisVisualizationError(
                "VISUALIZATION_EVIDENCE_INVALID",
                "Verified evidence has duplicate step metadata.",
            )
        step_records[step_id] = value

    projections: list[_FigureProjection] = []
    for ordered_step in workflow["ordered_steps"]:
        if not isinstance(ordered_step, Mapping):
            raise AnalysisVisualizationError(
                "VISUALIZATION_EVIDENCE_INVALID",
                "Verified evidence has invalid workflow metadata.",
            )
        step_id = ordered_step.get("step_id")
        tool_name = ordered_step.get("tool_name")
        if not isinstance(step_id, str) or not isinstance(tool_name, str):
            raise AnalysisVisualizationError(
                "VISUALIZATION_EVIDENCE_INVALID",
                "Verified evidence has invalid workflow identity.",
            )
        if tool_name not in {
            "compute_cell_umap",
            "evaluate_cell_clustering",
            "evaluate_cell_annotation",
        }:
            continue
        step = step_records.get(step_id)
        if step is None or step.get("tool_name") != tool_name:
            raise AnalysisVisualizationError(
                "VISUALIZATION_EVIDENCE_INVALID",
                "Verified evidence workflow and step metadata disagree.",
            )
        facts = step.get("facts")
        if not isinstance(facts, Mapping):
            raise AnalysisVisualizationError(
                "VISUALIZATION_EVIDENCE_INVALID",
                "Verified evidence lacks supported step facts.",
            )
        ordinal = len(projections) + 1
        if tool_name == "compute_cell_umap":
            kind = "umap_leiden"
            source = _source_artifact(
                artifacts,
                step_id=step_id,
                kind="cell_umap_h5ad",
                tool_name=tool_name,
            )
            path = Path(str(source["artifact_path"])).expanduser().resolve()
            data = _read_umap_presentation(path, expected_n_cells=facts.get("n_cells"))
            digest = _umap_plot_digest(data.cell_ids, data.coordinates, data.labels)
            legend_columns = max(1, math.ceil(len(data.categories) / 20))
            width_px = int((8.0 + 1.5 * (legend_columns - 1)) * _DPI)
            height_px = 6 * _DPI
            presentation: Mapping[str, JsonValue] = {
                "n_cells": len(data.cell_ids),
                "coordinate_shape": tuple(int(v) for v in data.coordinates.shape),
                "coordinate_dtype": str(data.coordinates.dtype),
                "cell_order_sha256": _ordered_strings_digest(
                    data.cell_ids, domain="agent.visualization.umap-cell-ids.v1"
                ),
                "leiden_labels_sha256": _ordered_strings_digest(
                    data.labels, domain="agent.visualization.umap-leiden-labels.v1"
                ),
                "category_order": data.categories,
                "palette": tuple(
                    {"category": category, "color": color}
                    for category, color in zip(
                        data.categories, data.palette, strict=True
                    )
                ),
                "palette_version": 1,
                "point_size": data.point_size,
                "alpha": _UMAP_ALPHA,
                "jitter": False,
                "subsampling": False,
                "coordinate_transform": False,
                "legend_columns": legend_columns,
            }
        elif tool_name == "evaluate_cell_clustering":
            kind = "clustering_metrics"
            source = _source_artifact(
                artifacts,
                step_id=step_id,
                kind="clustering_evaluation_json",
                tool_name=tool_name,
            )
            data = _metric_data(facts)
            digest = _domain_digest(
                "agent.visualization.clustering-metrics-data.v1",
                tuple(
                    {"name": name, "value": value}
                    for name, value in zip(data.names, data.values, strict=True)
                ),
            )
            width_px = 7 * _DPI
            height_px = int(4.5 * _DPI)
            presentation = {
                "metric_order": data.names,
                "metric_values": data.values,
                "y_axis": (-1.0, 1.0),
                "zero_line": True,
                "ranking": False,
                "selection": False,
            }
        else:
            kind = "annotation_confusion"
            source = _source_artifact(
                artifacts,
                step_id=step_id,
                kind="annotation_evaluation_json",
                tool_name=tool_name,
            )
            path = Path(str(source["artifact_path"])).expanduser().resolve()
            data = _read_confusion_presentation(path)
            digest = _domain_digest(
                "agent.visualization.annotation-confusion-data.v1",
                {
                    "rows": data.row_labels,
                    "columns": data.column_labels,
                    "counts": data.counts.tolist(),
                },
            )
            width_inches = min(20.0, max(8.0, 3.0 + 0.45 * len(data.column_labels)))
            height_inches = min(20.0, max(7.0, 3.0 + 0.4 * len(data.row_labels)))
            width_px = int(width_inches * _DPI)
            height_px = int(height_inches * _DPI)
            presentation = {
                "row_order": data.row_labels,
                "column_order": data.column_labels,
                "matrix_shape": tuple(int(v) for v in data.counts.shape),
                "raw_count_total": int(data.counts.sum()),
                "raw_counts": True,
                "normalization": False,
                "structural_unassigned_last": True,
                "cell_text_annotation_limit": _CONFUSION_TEXT_CELL_LIMIT,
                "cell_text_annotations_drawn": data.annotate_cells,
                "colormap": "Blues",
            }
        figure_id, filename = _figure_identity(ordinal, kind, step_id)
        projections.append(
            _FigureProjection(
                figure_id=figure_id,
                figure_kind=kind,
                filename=filename,
                source_step_ids=(step_id,),
                source_artifacts=(source,),
                plotting_data_sha256=digest,
                width_px=width_px,
                height_px=height_px,
                dpi=_DPI,
                presentation=presentation,
                data=data,
            )
        )
    if not projections:
        raise AnalysisVisualizationError(
            "VISUALIZATION_NO_SUPPORTED_FIGURES",
            "Verified evidence contains no supported Milestone 7.2 figure source.",
        )
    return tuple(projections)


def _renderer_contract() -> Mapping[str, JsonValue]:
    try:
        font_path = Path(
            font_manager.findfont(
                font_manager.FontProperties(family=[_FONT_FAMILY]),
                fallback_to_default=False,
            )
        ).resolve()
        font_sha256 = _sha256_file(font_path)
        matplotlib_version = importlib.metadata.version("matplotlib")
    except Exception as exc:
        raise AnalysisVisualizationError(
            "VISUALIZATION_RENDERER_UNAVAILABLE",
            "The fixed Matplotlib renderer/font contract is unavailable.",
        ) from exc
    return {
        "backend": "Agg",
        "matplotlib_version": matplotlib_version,
        "numpy_version": np.__version__,
        "freetype_version": str(ft2font.__freetype_version__),
        "font_family": _FONT_FAMILY,
        "font_path": str(font_path),
        "font_sha256": font_sha256,
        "png_metadata": {"Software": f"Agent plotting spec v{PLOTTING_SPEC_VERSION}"},
        "reproducibility_scope": "recorded_renderer_font_software_contract",
    }


def _render_umap(figure: Figure, data: _UmapData, *, legend_columns: int) -> None:
    axes = figure.add_axes((0.08, 0.12, 0.62, 0.78))
    color_by_category = dict(zip(data.categories, data.palette, strict=True))
    colors = [color_by_category[label] for label in data.labels]
    axes.scatter(
        data.coordinates[:, 0],
        data.coordinates[:, 1],
        c=colors,
        s=data.point_size,
        alpha=_UMAP_ALPHA,
        linewidths=0.0,
        edgecolors="none",
    )
    axes.set_xlabel("UMAP 1")
    axes.set_ylabel("UMAP 2")
    axes.set_xticks(())
    axes.set_yticks(())
    axes.set_title("UMAP by Leiden cluster")
    handles = [
        Line2D(
            (),
            (),
            marker="o",
            linestyle="none",
            markerfacecolor=color,
            markeredgewidth=0.0,
            markersize=5.0,
            label=category,
        )
        for category, color in zip(data.categories, data.palette, strict=True)
    ]
    figure.legend(
        handles=handles,
        labels=list(data.categories),
        loc="center left",
        bbox_to_anchor=(0.72, 0.5),
        frameon=False,
        ncol=legend_columns,
        fontsize=7.0,
        title="Leiden",
    )


def _render_metrics(figure: Figure, data: _MetricData) -> None:
    axes = figure.add_axes((0.1, 0.16, 0.86, 0.74))
    positions = np.arange(len(data.names))
    bars = axes.bar(positions, data.values, color=_METRIC_COLORS, width=0.68)
    axes.axhline(0.0, color="#333333", linewidth=0.8)
    axes.set_ylim(-1.0, 1.0)
    axes.set_xticks(positions, data.names)
    axes.set_ylabel("Score")
    axes.set_title("Clustering evaluation metrics")
    axes.grid(axis="y", color="#dddddd", linewidth=0.6)
    axes.set_axisbelow(True)
    for bar, value in zip(bars, data.values, strict=True):
        offset = 0.035 if value >= 0 else -0.035
        vertical = "bottom" if value >= 0 else "top"
        axes.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + offset,
            f"{value:.3f}",
            ha="center",
            va=vertical,
            fontsize=8.0,
        )


def _render_confusion(figure: Figure, data: _ConfusionData) -> None:
    axes = figure.add_axes((0.18, 0.22, 0.68, 0.68))
    maximum = max(1, int(data.counts.max(initial=0)))
    image = axes.imshow(
        data.counts,
        cmap="Blues",
        interpolation="nearest",
        aspect="auto",
        vmin=0,
        vmax=maximum,
    )
    axes.set_xticks(np.arange(len(data.column_labels)), data.column_labels)
    axes.set_yticks(np.arange(len(data.row_labels)), data.row_labels)
    axes.tick_params(axis="x", labelrotation=90, labelsize=7.0)
    axes.tick_params(axis="y", labelsize=7.0)
    axes.set_xlabel("Predicted label")
    axes.set_ylabel("Ground-truth label")
    axes.set_title("Annotation evaluation confusion matrix (raw counts)")
    if data.annotate_cells:
        threshold = maximum / 2.0
        for row in range(data.counts.shape[0]):
            for column in range(data.counts.shape[1]):
                count = int(data.counts[row, column])
                axes.text(
                    column,
                    row,
                    str(count),
                    ha="center",
                    va="center",
                    fontsize=6.0,
                    color="white" if count > threshold else "black",
                )
    colorbar_axes = figure.add_axes((0.88, 0.22, 0.025, 0.68))
    figure.colorbar(image, cax=colorbar_axes, label="Cell count")


def _render_figure(projection: _FigureProjection, path: Path) -> None:
    figure = Figure(
        figsize=(projection.width_px / projection.dpi, projection.height_px / projection.dpi),
        dpi=projection.dpi,
    )
    FigureCanvasAgg(figure)
    with rc_context(_RC_PARAMS):
        if isinstance(projection.data, _UmapData):
            legend_columns = projection.presentation.get("legend_columns")
            if not isinstance(legend_columns, int):  # pragma: no cover - internal invariant
                raise RuntimeError("Invalid UMAP legend specification.")
            _render_umap(figure, projection.data, legend_columns=legend_columns)
        elif isinstance(projection.data, _MetricData):
            _render_metrics(figure, projection.data)
        else:
            _render_confusion(figure, projection.data)
        try:
            figure.savefig(
                path,
                format="png",
                dpi=projection.dpi,
                metadata={"Software": f"Agent plotting spec v{PLOTTING_SPEC_VERSION}"},
            )
        finally:
            figure.clear()
    try:
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
    except OSError as exc:
        raise AnalysisVisualizationError(
            "VISUALIZATION_RENDER_FAILED",
            "A visualization figure could not be safely persisted.",
        ) from exc


def _png_dimensions(path: Path) -> tuple[int, int]:
    try:
        with path.open("rb") as handle:
            header = handle.read(24)
    except OSError as exc:
        raise AnalysisVisualizationError(
            "VISUALIZATION_FIGURE_INVALID",
            "A visualization figure could not be read.",
        ) from exc
    if (
        len(header) != 24
        or header[:8] != _PNG_SIGNATURE
        or header[12:16] != b"IHDR"
        or struct.unpack(">I", header[8:12])[0] != 13
    ):
        raise AnalysisVisualizationError(
            "VISUALIZATION_FIGURE_INVALID",
            "A visualization figure is not a valid PNG artifact.",
        )
    return struct.unpack(">II", header[16:24])


def _figure_manifest_record(
    projection: _FigureProjection, png_sha256: str
) -> Mapping[str, object]:
    return {
        "figure_id": projection.figure_id,
        "figure_kind": projection.figure_kind,
        "relative_path": f"figures/{projection.filename}",
        "source_step_ids": projection.source_step_ids,
        "source_artifacts": projection.source_artifacts,
        "plotting_data_sha256": projection.plotting_data_sha256,
        "plotting_spec_version": PLOTTING_SPEC_VERSION,
        "png_sha256": png_sha256,
        "format": "png",
        "width_px": projection.width_px,
        "height_px": projection.height_px,
        "dpi": projection.dpi,
        "presentation": projection.presentation,
    }


def _manifest_payload(
    snapshot: _EvidenceSnapshot,
    projections: Sequence[_FigureProjection],
    png_digests: Mapping[str, str],
    renderer: Mapping[str, JsonValue],
) -> Mapping[str, object]:
    return {
        "schema_version": ANALYSIS_VISUALIZATION_SCHEMA_VERSION,
        "artifact_type": ANALYSIS_VISUALIZATION_ARTIFACT_TYPE,
        "status": "success",
        "source": {
            "run_id": snapshot.run_id,
            "request_id": snapshot.request_id,
            "plan_id": snapshot.plan_id,
            "evidence_path": str(snapshot.path),
            "evidence_sha256": snapshot.sha256,
            "evidence_schema_version": ANALYSIS_EVIDENCE_SCHEMA_VERSION,
            "evidence_artifact_type": ANALYSIS_EVIDENCE_ARTIFACT_TYPE,
        },
        "figure_set": {
            "n_figures": len(projections),
            "expected_figure_ids": tuple(value.figure_id for value in projections),
            "figure_order": "verified_workflow_topological_order",
        },
        "figures": tuple(
            _figure_manifest_record(value, png_digests[value.figure_id])
            for value in projections
        ),
        "plotting": {
            "spec_version": PLOTTING_SPEC_VERSION,
            "renderer": renderer,
        },
        "publication": {
            "bundle_directory": ANALYSIS_VISUALIZATION_BUNDLE_DIRNAME,
            "completion_marker": ANALYSIS_VISUALIZATION_MANIFEST_FILENAME,
            "strategy": "staged_directory_with_backup_rollback",
            "nonempty_directory_exchange_universally_atomic": False,
        },
        "validation": {
            "fresh_evidence_verified_before_extraction": True,
            "fresh_evidence_verified_before_publication": True,
            "explicit_evidence_artifacts_only": True,
            "scientific_tools_invoked": False,
            "raw_scatac_matrix_accessed": False,
            "transferred_label_umap_included": False,
            "large_scientific_payloads_in_manifest": False,
        },
    }


def _load_strict_manifest(path: Path) -> tuple[Mapping[str, object], bytes]:
    try:
        payload = path.read_bytes()
        parsed = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except Exception as exc:
        raise AnalysisVisualizationError(
            "VISUALIZATION_MANIFEST_MALFORMED",
            "Visualization manifest is not strict valid JSON.",
        ) from exc
    if not isinstance(parsed, dict):
        raise AnalysisVisualizationError(
            "VISUALIZATION_MANIFEST_MALFORMED",
            "Visualization manifest must contain one JSON object.",
        )
    if (
        parsed.get("schema_version") != ANALYSIS_VISUALIZATION_SCHEMA_VERSION
        or parsed.get("artifact_type") != ANALYSIS_VISUALIZATION_ARTIFACT_TYPE
        or parsed.get("status") != "success"
    ):
        raise AnalysisVisualizationError(
            "VISUALIZATION_MANIFEST_IDENTITY_INVALID",
            "Visualization manifest identity is invalid.",
        )
    return parsed, payload


def _manifest_png_digests(
    manifest: Mapping[str, object], projections: Sequence[_FigureProjection]
) -> Mapping[str, str]:
    figures = manifest.get("figures")
    if not isinstance(figures, list) or len(figures) != len(projections):
        raise AnalysisVisualizationError(
            "VISUALIZATION_MANIFEST_SCHEMA_INVALID",
            "Visualization manifest has an invalid figure set.",
        )
    digests: dict[str, str] = {}
    for record, projection in zip(figures, projections, strict=True):
        if not isinstance(record, Mapping):
            raise AnalysisVisualizationError(
                "VISUALIZATION_MANIFEST_SCHEMA_INVALID",
                "Visualization manifest has an invalid figure record.",
            )
        figure_id = record.get("figure_id")
        digest = record.get("png_sha256")
        if figure_id != projection.figure_id or not _is_sha256(digest):
            raise AnalysisVisualizationError(
                "VISUALIZATION_MANIFEST_SCHEMA_INVALID",
                "Visualization manifest has invalid figure identity or digest metadata.",
            )
        digests[projection.figure_id] = str(digest)
    return digests


def _verify_bundle_contents(
    bundle: Path,
    manifest: Mapping[str, object],
    projections: Sequence[_FigureProjection],
    expected_manifest: Mapping[str, object],
    manifest_bytes: bytes,
) -> None:
    expected_bytes = _canonical_json_bytes(expected_manifest)
    if manifest_bytes != expected_bytes or manifest != json.loads(expected_bytes):
        raise AnalysisVisualizationError(
            "VISUALIZATION_MANIFEST_CONTENT_MISMATCH",
            "Visualization manifest differs from freshly derived plotting metadata.",
        )
    try:
        root_entries = {value.name for value in bundle.iterdir()}
        figures_dir = bundle / "figures"
        if (
            root_entries
            != {"figures", ANALYSIS_VISUALIZATION_MANIFEST_FILENAME}
            or not figures_dir.is_dir()
            or figures_dir.is_symlink()
        ):
            raise ValueError("Invalid bundle entries.")
        expected_names = {projection.filename for projection in projections}
        actual_names = {value.name for value in figures_dir.iterdir()}
        if actual_names != expected_names:
            raise ValueError("Invalid figure entries.")
        records = manifest["figures"]
        assert isinstance(records, list)
        by_id = {
            str(record["figure_id"]): record
            for record in records
            if isinstance(record, Mapping)
        }
        for projection in projections:
            path = figures_dir / projection.filename
            if not path.is_file() or path.is_symlink():
                raise ValueError("Invalid figure file.")
            record = by_id[projection.figure_id]
            if _sha256_file(path) != record["png_sha256"]:
                raise AnalysisVisualizationError(
                    "VISUALIZATION_FIGURE_SHA256_MISMATCH",
                    "A visualization PNG differs from its manifest digest.",
                )
            if _png_dimensions(path) != (projection.width_px, projection.height_px):
                raise AnalysisVisualizationError(
                    "VISUALIZATION_FIGURE_DIMENSIONS_INVALID",
                    "A visualization PNG has unexpected dimensions.",
                )
    except AnalysisVisualizationError:
        raise
    except Exception as exc:
        raise AnalysisVisualizationError(
            "VISUALIZATION_BUNDLE_INVALID",
            "Visualization bundle contains missing, renamed, or unexpected artifacts.",
        ) from exc


def _resolve_output_dir(value: str | Path) -> Path:
    if not isinstance(value, (str, Path)):
        raise TypeError("`output_dir` must be a string or pathlib.Path.")
    path = Path(value).expanduser().resolve()
    if path.exists() and not path.is_dir():
        raise ValueError(f"Visualization output path is not a directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_bundle(
    staging: Path, destination: Path, *, overwrite: bool
) -> None:
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Visualization bundle already exists: {destination}. "
            "Use overwrite=True to replace it."
        )
    backup: Path | None = None
    parent = destination.parent
    staged_installed = False
    try:
        if destination.exists():
            backup = Path(
                tempfile.mkdtemp(
                    dir=parent,
                    prefix=".analysis_visualizations.",
                    suffix=".backup",
                )
            )
            backup.rmdir()
            os.replace(destination, backup)
        try:
            os.replace(staging, destination)
            staged_installed = True
            _fsync_directory(parent)
        except Exception as publication_error:
            try:
                if backup is not None:
                    if destination.exists():
                        shutil.rmtree(destination)
                    os.replace(backup, destination)
                    backup = None
                    _fsync_directory(parent)
                elif staged_installed and destination.exists():
                    shutil.rmtree(destination)
                    _fsync_directory(parent)
            except Exception as rollback_error:
                raise AnalysisVisualizationError(
                    "VISUALIZATION_ROLLBACK_FAILED",
                    "Visualization publication failed and rollback requires manual recovery.",
                ) from rollback_error
            raise publication_error
        if backup is not None:
            shutil.rmtree(backup)
            backup = None
            _fsync_directory(parent)
    except FileExistsError:
        raise
    except AnalysisVisualizationError:
        raise
    except Exception as exc:
        raise AnalysisVisualizationError(
            "VISUALIZATION_PUBLISH_FAILED",
            "Visualization bundle could not be published safely.",
        ) from exc


def _visualization_reference(
    visualization: str | Path | AnalysisVisualizationResult,
) -> tuple[Path, str | None]:
    if isinstance(visualization, Mapping):
        required = {
            "status",
            "manifest_path",
            "manifest_sha256",
            "bundle_path",
            "schema_version",
            "artifact_type",
            "evidence_path",
            "evidence_sha256",
            "run_id",
            "request_id",
            "plan_id",
            "n_figures",
            "figures",
        }
        if set(visualization) != required:
            raise AnalysisVisualizationError(
                "VISUALIZATION_RESULT_INVALID",
                "AnalysisVisualizationResult has an invalid schema.",
            )
        path = visualization.get("manifest_path")
        digest = visualization.get("manifest_sha256")
        if (
            visualization.get("status") != "success"
            or visualization.get("schema_version")
            != ANALYSIS_VISUALIZATION_SCHEMA_VERSION
            or visualization.get("artifact_type")
            != ANALYSIS_VISUALIZATION_ARTIFACT_TYPE
            or not isinstance(path, str)
            or not path
            or not _is_sha256(digest)
        ):
            raise AnalysisVisualizationError(
                "VISUALIZATION_RESULT_INVALID",
                "AnalysisVisualizationResult contains invalid identity metadata.",
            )
        return Path(path).expanduser().resolve(), str(digest)
    if not isinstance(visualization, (str, Path)):
        raise TypeError(
            "`visualization` must be a manifest path or AnalysisVisualizationResult."
        )
    return Path(visualization).expanduser().resolve(), None


def build_analysis_visualizations(
    run_result: AgentRunResult,
    evidence: str | Path | AnalysisEvidenceResult,
    output_dir: str | Path,
    *,
    registry: ToolRegistry,
    overwrite: bool = False,
) -> AnalysisVisualizationResult:
    """Build a deterministic PNG bundle from freshly verified analysis evidence."""

    if not isinstance(run_result, AgentRunResult):
        raise TypeError("`run_result` must be an AgentRunResult.")
    if not isinstance(registry, ToolRegistry):
        raise TypeError("`registry` must be a ToolRegistry.")
    if not isinstance(overwrite, bool):
        raise TypeError("`overwrite` must be a boolean.")
    output = _resolve_output_dir(output_dir)
    destination = output / ANALYSIS_VISUALIZATION_BUNDLE_DIRNAME
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Visualization bundle already exists: {destination}. "
            "Use overwrite=True to replace it."
        )

    first = _verified_evidence_snapshot(run_result, evidence, registry)
    projections = _derive_figure_projections(first)
    renderer = _renderer_contract()

    staging = Path(
        tempfile.mkdtemp(
            dir=output,
            prefix=".analysis_visualizations.",
            suffix=".tmp",
        )
    )
    published = False
    try:
        figures_dir = staging / "figures"
        figures_dir.mkdir()
        png_digests: dict[str, str] = {}
        for projection in projections:
            figure_path = figures_dir / projection.filename
            _render_figure(projection, figure_path)
            if _png_dimensions(figure_path) != (
                projection.width_px,
                projection.height_px,
            ):
                raise AnalysisVisualizationError(
                    "VISUALIZATION_RENDER_FAILED",
                    "A rendered PNG has unexpected dimensions.",
                )
            png_digests[projection.figure_id] = _sha256_file(figure_path)
        _fsync_directory(figures_dir)
        second = _verified_evidence_snapshot(run_result, evidence, registry)
        if (
            first.path != second.path
            or first.sha256 != second.sha256
            or first.payload != second.payload
        ):
            raise AnalysisVisualizationError(
                "VISUALIZATION_EVIDENCE_CHANGED",
                "Analysis evidence changed during visualization preparation.",
            )
        second_projections = _derive_figure_projections(second)
        placeholder_digests = {
            projection.figure_id: "0" * 64 for projection in projections
        }
        second_placeholder_digests = {
            projection.figure_id: "0" * 64 for projection in second_projections
        }
        if _canonical_json_bytes(
            tuple(
                _figure_manifest_record(
                    projection, placeholder_digests[projection.figure_id]
                )
                for projection in projections
            )
        ) != _canonical_json_bytes(
            tuple(
                _figure_manifest_record(
                    projection,
                    second_placeholder_digests[projection.figure_id],
                )
                for projection in second_projections
            )
        ):
            raise AnalysisVisualizationError(
                "VISUALIZATION_SOURCE_CHANGED",
                "A visualization source changed during visualization preparation.",
            )
        manifest = _manifest_payload(first, projections, png_digests, renderer)
        manifest_bytes = _canonical_json_bytes(manifest)
        manifest_path = staging / ANALYSIS_VISUALIZATION_MANIFEST_FILENAME
        with manifest_path.open("wb") as handle:
            handle.write(manifest_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        loaded, loaded_bytes = _load_strict_manifest(manifest_path)
        _verify_bundle_contents(
            staging, loaded, projections, manifest, loaded_bytes
        )
        _fsync_directory(staging)
        _publish_bundle(staging, destination, overwrite=overwrite)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)

    final_manifest = destination / ANALYSIS_VISUALIZATION_MANIFEST_FILENAME
    final_bytes = final_manifest.read_bytes()
    figure_results: list[VisualizationFigureResult] = []
    for projection in projections:
        path = destination / "figures" / projection.filename
        figure_results.append(
            {
                "figure_id": projection.figure_id,
                "figure_kind": projection.figure_kind,
                "figure_path": str(path),
                "png_sha256": _sha256_file(path),
            }
        )
    return {
        "status": "success",
        "manifest_path": str(final_manifest),
        "manifest_sha256": _sha256_bytes(final_bytes),
        "bundle_path": str(destination),
        "schema_version": ANALYSIS_VISUALIZATION_SCHEMA_VERSION,
        "artifact_type": ANALYSIS_VISUALIZATION_ARTIFACT_TYPE,
        "evidence_path": str(first.path),
        "evidence_sha256": first.sha256,
        "run_id": first.run_id,
        "request_id": first.request_id,
        "plan_id": first.plan_id,
        "n_figures": len(projections),
        "figures": figure_results,
    }


def verify_analysis_visualizations(
    run_result: AgentRunResult,
    evidence: str | Path | AnalysisEvidenceResult,
    visualization: str | Path | AnalysisVisualizationResult,
    *,
    registry: ToolRegistry,
) -> VerificationResult:
    """Freshly verify source evidence, plotting metadata, and persisted PNGs."""

    target_id = (
        run_result.run_id
        if isinstance(run_result, AgentRunResult)
        else "analysis-visualizations"
    )
    checks: list[VerificationCheck] = []
    try:
        if not isinstance(run_result, AgentRunResult):
            raise TypeError("`run_result` must be an AgentRunResult.")
        if not isinstance(registry, ToolRegistry):
            raise TypeError("`registry` must be a ToolRegistry.")
        snapshot = _verified_evidence_snapshot(run_result, evidence, registry)
        projections = _derive_figure_projections(snapshot)
        renderer = _renderer_contract()
    except (AnalysisVisualizationError, TypeError) as exc:
        code = getattr(exc, "code", "VISUALIZATION_SOURCE_INVALID")
        checks.append(
            VerificationCheck(
                "source_evidence_freshly_verified",
                False,
                "Visualization source evidence failed fresh verification.",
            )
        )
        return VerificationResult(
            passed=False,
            target_type="analysis_visualizations",
            target_id=target_id,
            checks=tuple(checks),
            error=AgentError(
                ErrorCategory.VERIFICATION_ERROR,
                code,
                "Visualization source evidence failed verification.",
            ),
        )
    checks.append(
        VerificationCheck(
            "source_evidence_freshly_verified",
            True,
            "Source evidence and scientific artifacts passed fresh verification.",
        )
    )

    try:
        manifest_path, expected_manifest_digest = _visualization_reference(
            visualization
        )
        bundle = manifest_path.parent
        if manifest_path.name != ANALYSIS_VISUALIZATION_MANIFEST_FILENAME:
            raise AnalysisVisualizationError(
                "VISUALIZATION_RESULT_INVALID",
                "Visualization manifest has an unexpected filename.",
            )
        manifest, manifest_bytes = _load_strict_manifest(manifest_path)
        actual_manifest_digest = _sha256_bytes(manifest_bytes)
        if (
            expected_manifest_digest is not None
            and actual_manifest_digest != expected_manifest_digest
        ):
            raise AnalysisVisualizationError(
                "VISUALIZATION_MANIFEST_SHA256_MISMATCH",
                "Visualization manifest differs from its authoritative digest.",
            )
        if isinstance(visualization, Mapping):
            figures = visualization.get("figures")
            if (
                visualization.get("bundle_path") != str(bundle)
                or visualization.get("evidence_path") != str(snapshot.path)
                or visualization.get("evidence_sha256") != snapshot.sha256
                or visualization.get("run_id") != snapshot.run_id
                or visualization.get("request_id") != snapshot.request_id
                or visualization.get("plan_id") != snapshot.plan_id
                or visualization.get("n_figures") != len(projections)
                or not isinstance(figures, list)
                or len(figures) != len(projections)
            ):
                raise AnalysisVisualizationError(
                    "VISUALIZATION_RESULT_INVALID",
                    "AnalysisVisualizationResult does not match verified sources.",
                )
        png_digests = _manifest_png_digests(manifest, projections)
        if isinstance(visualization, Mapping):
            expected_figures = [
                {
                    "figure_id": projection.figure_id,
                    "figure_kind": projection.figure_kind,
                    "figure_path": str(bundle / "figures" / projection.filename),
                    "png_sha256": png_digests[projection.figure_id],
                }
                for projection in projections
            ]
            if visualization.get("figures") != expected_figures:
                raise AnalysisVisualizationError(
                    "VISUALIZATION_RESULT_INVALID",
                    "AnalysisVisualizationResult figure metadata is invalid.",
                )
        expected_manifest = _manifest_payload(
            snapshot, projections, png_digests, renderer
        )
        _verify_bundle_contents(
            bundle, manifest, projections, expected_manifest, manifest_bytes
        )
    except (AnalysisVisualizationError, OSError, TypeError) as exc:
        code = getattr(exc, "code", "VISUALIZATION_BUNDLE_INVALID")
        checks.append(
            VerificationCheck(
                "visualization_bundle_valid",
                False,
                "Visualization bundle failed strict verification.",
            )
        )
        return VerificationResult(
            passed=False,
            target_type="analysis_visualizations",
            target_id=target_id,
            checks=tuple(checks),
            error=AgentError(
                ErrorCategory.VERIFICATION_ERROR,
                code,
                "Visualization bundle failed verification.",
            ),
        )
    checks.append(
        VerificationCheck(
            "visualization_bundle_valid",
            True,
            "Manifest, plotting metadata, and all expected PNGs passed verification.",
        )
    )
    if expected_manifest_digest is not None:
        checks.append(
            VerificationCheck(
                "visualization_manifest_sha256_matches",
                True,
                "Visualization manifest matches its authoritative result digest.",
            )
        )
    return VerificationResult(
        passed=True,
        target_type="analysis_visualizations",
        target_id=target_id,
        checks=tuple(checks),
    )


__all__ = [
    "ANALYSIS_VISUALIZATION_ARTIFACT_TYPE",
    "ANALYSIS_VISUALIZATION_BUNDLE_DIRNAME",
    "ANALYSIS_VISUALIZATION_MANIFEST_FILENAME",
    "ANALYSIS_VISUALIZATION_SCHEMA_VERSION",
    "PLOTTING_SPEC_VERSION",
    "AnalysisVisualizationError",
    "AnalysisVisualizationResult",
    "VisualizationFigureResult",
    "build_analysis_visualizations",
    "verify_analysis_visualizations",
]
