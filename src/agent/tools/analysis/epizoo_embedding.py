"""File-based scientific tool for validated EpiZoo cell embeddings."""

from __future__ import annotations

import os
from pathlib import Path
import re
import tempfile
from typing import Literal, Mapping, TypedDict

import anndata as ad
import numpy as np

from agent.tools.data import inspect_scATAC
from agent.tools.models import epizoo as epizoo_backend
from agent.tools.models.epizoo_cache import get_cached_epizoo_model


_RESOURCE_IDENTITY_FIELDS = frozenset({
    "resource_id", "checkpoint_sha256", "frequencies_sha256",
    "filter_indices_sha256",
})
_INFERENCE_SETTING_FIELDS = frozenset({
    "batch_size", "max_length", "random_sample", "random_seed", "num_workers",
    "device", "dtype", "amp_requested", "amp_enabled", "show_progress", "overwrite",
})
RESOURCE_PROVENANCE_SCHEMA = "epizoo-resource-provenance.v1"


class EpiZooResourceIdentityError(ValueError):
    """A scoped, safe failure of an explicitly pinned EpiZoo invocation."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__("EpiZoo resource identity validation failed.")


def validate_expected_resource_identity(value: object) -> dict[str, str]:
    """Validate the one public bounded resource-pin contract without reading files."""
    if not isinstance(value, Mapping) or set(value) != _RESOURCE_IDENTITY_FIELDS:
        raise EpiZooResourceIdentityError("EPIZOO_RESOURCE_IDENTITY_INVALID")
    if not isinstance(value["resource_id"], str) or re.fullmatch(
        r"[a-z][a-z0-9_-]{0,63}", value["resource_id"]
    ) is None:
        raise EpiZooResourceIdentityError("EPIZOO_RESOURCE_IDENTITY_INVALID")
    for field in _RESOURCE_IDENTITY_FIELDS - {"resource_id"}:
        if not isinstance(value[field], str) or re.fullmatch(
            r"[0-9a-f]{64}", value[field]
        ) is None:
            raise EpiZooResourceIdentityError("EPIZOO_RESOURCE_IDENTITY_INVALID")
    return dict(value)


def validate_resource_provenance(
    value: object,
    *,
    expected_resource_identity: object | None = None,
    species: str | None = None,
    device: str | None = None,
    overwrite: bool | None = None,
) -> dict[str, object]:
    """Validate owner-produced pinned provenance; this does not reproduce inference."""
    def invalid() -> None:
        raise EpiZooResourceIdentityError("EPIZOO_RESOURCE_IDENTITY_MISMATCH")

    if not isinstance(value, Mapping) or set(value) != {
        "schema", "expected_resource_identity", "actual_resource_identity",
        "species", "inference_settings",
    } or value.get("schema") != RESOURCE_PROVENANCE_SCHEMA:
        invalid()
    expected = validate_expected_resource_identity(value["expected_resource_identity"])
    actual = validate_expected_resource_identity(value["actual_resource_identity"])
    if expected != actual or (expected_resource_identity is not None and expected !=
                             validate_expected_resource_identity(expected_resource_identity)):
        invalid()
    if not isinstance(value["species"], str) or value["species"] not in {"human", "mouse"} or (
        species is not None and (not isinstance(species, str) or value["species"] != species.strip().lower())
    ):
        invalid()
    settings = value["inference_settings"]
    if not isinstance(settings, Mapping) or set(settings) != _INFERENCE_SETTING_FIELDS:
        invalid()
    fixed = {"batch_size": 4, "max_length": 8192, "random_seed": 0, "num_workers": 0}
    if any(type(settings[name]) is not int or settings[name] != number
           for name, number in fixed.items()):
        invalid()
    for name, required in {"random_sample": True, "amp_requested": True,
                           "show_progress": False}.items():
        if settings[name] is not required:
            invalid()
    if (type(settings["amp_enabled"]) is not bool
        or type(settings["overwrite"]) is not bool
        or settings["dtype"] != "float32"
        or not isinstance(settings["device"], str)
        or re.fullmatch(r"(?:cpu|cuda)(?::[0-9]+)?", settings["device"]) is None
        or settings["amp_enabled"] != settings["device"].startswith("cuda")
        or (device is not None and settings["device"] != device)
        or (overwrite is not None and settings["overwrite"] is not overwrite)):
        invalid()
    return {"schema": RESOURCE_PROVENANCE_SCHEMA, "expected_resource_identity": expected,
            "actual_resource_identity": actual, "species": value["species"],
            "inference_settings": dict(settings)}


def _pinned_resource_paths(checkpoint_path: str | Path, species: str) -> dict[str, Path]:
    configuration = epizoo_backend._normalize_species(species)
    directory = epizoo_backend.DEFAULT_RESOURCES_DIR.expanduser().resolve()
    return {"checkpoint_sha256": Path(checkpoint_path).expanduser().resolve(),
            "frequencies_sha256": directory / configuration.frequency_filename,
            "filter_indices_sha256": directory / configuration.filter_filename}


def _check_resource_files(
    paths: Mapping[str, Path], expected: Mapping[str, str],
) -> dict[str, tuple[int, ...]]:
    snapshots: dict[str, tuple[int, ...]] = {}
    for field, path in paths.items():
        try:
            snapshot = epizoo_backend._resource_file_snapshot(path)
            matches = epizoo_backend._sha256_file(path) == expected[field]
            epizoo_backend._check_resource_snapshot(path, snapshot)
        except (OSError, epizoo_backend.EpiZooResourceChangedError) as exc:
            raise EpiZooResourceIdentityError("EPIZOO_RESOURCE_IDENTITY_MISMATCH") from exc
        if not matches:
            raise EpiZooResourceIdentityError("EPIZOO_RESOURCE_IDENTITY_MISMATCH")
        snapshots[field] = snapshot
    return snapshots


def _pinned_provenance(
    metadata: Mapping[str, object], expected: Mapping[str, str], *,
    species: str, device: str, overwrite: bool, paths: Mapping[str, Path],
) -> dict[str, object]:
    try:
        checkpoint, resources, amp = metadata["checkpoint"], metadata["resources"], metadata["amp"]
        actual = {"resource_id": expected["resource_id"],
                  "checkpoint_sha256": checkpoint["sha256"],
                  "frequencies_sha256": resources["frequencies"]["sha256"],
                  "filter_indices_sha256": resources["filter_indices"]["sha256"]}
        actual_paths = {"checkpoint_sha256": checkpoint["path"],
                        "frequencies_sha256": resources["frequencies"]["path"],
                        "filter_indices_sha256": resources["filter_indices"]["path"]}
        if any(not isinstance(actual_paths[key], str) or
               Path(actual_paths[key]).expanduser().resolve() != path
               for key, path in paths.items()):
            raise EpiZooResourceIdentityError("EPIZOO_RESOURCE_IDENTITY_MISMATCH")
        settings = {name: metadata[name] for name in (
            "batch_size", "max_length", "random_sample", "random_seed", "num_workers", "device", "dtype")}
        settings.update(amp_requested=amp["requested"], amp_enabled=amp["enabled"],
                        show_progress=False, overwrite=overwrite)
        provenance = {"schema": RESOURCE_PROVENANCE_SCHEMA,
                      "expected_resource_identity": dict(expected), "actual_resource_identity": actual,
                      "species": metadata["species"]["name"], "inference_settings": settings}
        return validate_resource_provenance(provenance, expected_resource_identity=expected,
                                            species=species, device=device, overwrite=overwrite)
    except (KeyError, TypeError, ValueError) as exc:
        raise EpiZooResourceIdentityError("EPIZOO_RESOURCE_IDENTITY_MISMATCH") from exc


class _OptionalEpiZooEmbeddingFields(TypedDict, total=False):
    resource_provenance: dict[str, object]


class EpiZooEmbeddingToolResult(_OptionalEpiZooEmbeddingFields):
    """Lightweight, JSON-serializable result for an EpiZoo embedding run."""

    status: Literal["success"]
    input_path: str
    embedding_path: str
    cell_ids_path: str
    n_cells: int
    embedding_dim: int
    embedding_dtype: str
    finite: bool
    cell_order_preserved: bool
    backend: str
    species: str
    checkpoint_path: str
    device: str


def _resolve_output_dir(output_dir: str | Path) -> Path:
    if not isinstance(output_dir, (str, Path)):
        raise TypeError("`output_dir` must be a string or pathlib.Path.")

    resolved = Path(output_dir).expanduser().resolve()
    if resolved.exists() and not resolved.is_dir():
        raise ValueError(f"EpiZoo output path is not a directory: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _artifact_paths(input_path: Path, output_dir: Path) -> tuple[Path, Path]:
    stem = input_path.stem
    return (
        output_dir / f"{stem}.epizoo_embeddings.npy",
        output_dir / f"{stem}.epizoo_obs_names.txt",
    )


def _ensure_outputs_available(
    embedding_path: Path, cell_ids_path: Path, *, overwrite: bool
) -> None:
    existing = [path for path in (embedding_path, cell_ids_path) if path.exists()]
    if existing and not overwrite:
        shown = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"EpiZoo output artifact already exists: {shown}. "
            "Use overwrite=True to replace existing artifacts."
        )


def _all_finite(embeddings: np.ndarray, rows_per_chunk: int = 8192) -> bool:
    for start in range(0, embeddings.shape[0], rows_per_chunk):
        if not np.isfinite(embeddings[start : start + rows_per_chunk]).all():
            return False
    return True


def _validate_backend_result(
    embeddings: np.ndarray,
    result_obs_names: tuple[str, ...],
    input_obs_names: tuple[str, ...],
) -> tuple[bool, bool]:
    if embeddings.ndim != 2:
        raise RuntimeError(
            f"EpiZoo returned a {embeddings.ndim}-dimensional embedding array; "
            "expected a 2-dimensional [cells, embedding_dim] array."
        )
    if embeddings.shape[0] != len(input_obs_names):
        raise RuntimeError(
            f"EpiZoo returned {embeddings.shape[0]} embedding rows for "
            f"{len(input_obs_names)} input cells."
        )

    expected_dim = int(epizoo_backend.MODEL_CONFIG["emb_dim"])
    if embeddings.shape[1] != expected_dim:
        raise RuntimeError(
            f"EpiZoo returned embedding dimension {embeddings.shape[1]}; "
            f"expected {expected_dim}."
        )
    if embeddings.dtype != np.dtype(np.float32):
        raise RuntimeError(
            f"EpiZoo returned embedding dtype {embeddings.dtype}; expected float32."
        )

    order_preserved = result_obs_names == input_obs_names
    if not order_preserved:
        raise RuntimeError(
            "EpiZoo output cell identifiers do not exactly match the input cell order."
        )

    finite = _all_finite(embeddings)
    if not finite:
        raise RuntimeError("EpiZoo returned non-finite cell embeddings.")
    return finite, order_preserved


def _validate_cell_ids_for_text(cell_ids: tuple[str, ...]) -> None:
    invalid = [cell_id for cell_id in cell_ids if "\n" in cell_id or "\r" in cell_id]
    if invalid:
        raise ValueError(
            "Cell identifiers containing newline characters cannot be written to the "
            "one-cell-per-line sidecar artifact."
        )


def _write_artifacts(
    embeddings: np.ndarray,
    cell_ids: tuple[str, ...],
    embedding_path: Path,
    cell_ids_path: Path,
    *,
    overwrite: bool,
) -> None:
    embedding_temp: Path | None = None
    cell_ids_temp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=embedding_path.parent,
            prefix=f".{embedding_path.stem}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            embedding_temp = Path(handle.name)
            np.save(handle, embeddings, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=cell_ids_path.parent,
            prefix=f".{cell_ids_path.stem}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            cell_ids_temp = Path(handle.name)
            for cell_id in cell_ids:
                handle.write(cell_id)
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        _ensure_outputs_available(
            embedding_path, cell_ids_path, overwrite=overwrite
        )
        os.replace(cell_ids_temp, cell_ids_path)
        cell_ids_temp = None
        os.replace(embedding_temp, embedding_path)
        embedding_temp = None
    finally:
        if embedding_temp is not None:
            embedding_temp.unlink(missing_ok=True)
        if cell_ids_temp is not None:
            cell_ids_temp.unlink(missing_ok=True)


def epizoo_embed_cells(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    species: Literal["human", "mouse"],
    checkpoint_path: str | Path = epizoo_backend.DEFAULT_CHECKPOINT_PATH,
    device: str = "cuda:0",
    overwrite: bool = False,
    expected_resource_identity: Mapping[str, str] | None = None,
) -> EpiZooEmbeddingToolResult:
    """Embed a raw scATAC ``.h5ad`` file and persist ordered artifacts.

    This function delegates all scientific preprocessing and inference to the
    validated Milestone 1 EpiZoo wrapper. The full embedding matrix is written
    to ``.npy`` and is never included in the returned dictionary.
    """

    if not isinstance(species, str) or species.strip().lower() not in {
        "human",
        "mouse",
    }:
        raise ValueError("`species` must be 'human' or 'mouse'.")
    normalized_species = species.strip().lower()
    if not isinstance(overwrite, bool):
        raise TypeError("`overwrite` must be a boolean.")
    expected = (None if expected_resource_identity is None else
                validate_expected_resource_identity(expected_resource_identity))
    pinned_paths = None
    if expected is not None:
        pinned_paths = _pinned_resource_paths(checkpoint_path, normalized_species)
        pinned_snapshots = _check_resource_files(pinned_paths, expected)

    inspection = inspect_scATAC(input_path)
    resolved_input = Path(inspection["input_path"])
    if not inspection["x_is_sparse"]:
        raise TypeError(
            "EpiZoo requires a sparse scATAC X matrix; dense input files are rejected."
        )

    resolved_output_dir = _resolve_output_dir(output_dir)
    embedding_path, cell_ids_path = _artifact_paths(
        resolved_input, resolved_output_dir
    )
    _ensure_outputs_available(
        embedding_path, cell_ids_path, overwrite=overwrite
    )

    try:
        adata = ad.read_h5ad(resolved_input)
    except Exception as exc:
        raise ValueError(
            f"Unable to load sparse scATAC AnnData file {resolved_input}: {exc}"
        ) from exc

    input_obs_names = tuple(str(name) for name in adata.obs_names)
    try:
        model = get_cached_epizoo_model(
            checkpoint_path=checkpoint_path,
            device=device,
        )
        if expected is not None:
            if (getattr(model, "_agent_checkpoint_sha256", None) != expected["checkpoint_sha256"]
                or getattr(model, "_agent_checkpoint_source_snapshot", None)
                != pinned_snapshots["checkpoint_sha256"]):
                raise EpiZooResourceIdentityError("EPIZOO_RESOURCE_IDENTITY_MISMATCH")
            for field, path in pinned_paths.items():
                epizoo_backend._check_resource_snapshot(path, pinned_snapshots[field])
    except epizoo_backend.EpiZooResourceChangedError as exc:
        raise EpiZooResourceIdentityError("EPIZOO_RESOURCE_IDENTITY_MISMATCH") from exc
    pinned_kwargs = ({} if expected is None else
                     {"resources_dir": epizoo_backend.DEFAULT_RESOURCES_DIR})
    try:
        backend_result = epizoo_backend.embed_cells(
            model,
            adata,
            species=normalized_species,
            device=device,
            batch_size=4,
            max_length=8192,
            random_sample=True,
            random_seed=0,
            use_amp=True,
            num_workers=0,
            show_progress=False,
            **pinned_kwargs,
        )
    except epizoo_backend.EpiZooResourceChangedError as exc:
        raise EpiZooResourceIdentityError("EPIZOO_RESOURCE_IDENTITY_MISMATCH") from exc

    embeddings = np.asarray(backend_result.embeddings)
    result_obs_names = tuple(str(name) for name in backend_result.obs_names)
    finite, order_preserved = _validate_backend_result(
        embeddings, result_obs_names, input_obs_names
    )
    _validate_cell_ids_for_text(input_obs_names)

    metadata = backend_result.metadata
    resource_provenance = None
    if expected is not None:
        if _check_resource_files(pinned_paths, expected) != pinned_snapshots:
            raise EpiZooResourceIdentityError("EPIZOO_RESOURCE_IDENTITY_MISMATCH")
        resource_provenance = _pinned_provenance(
            metadata, expected, species=normalized_species, device=device,
            overwrite=overwrite, paths=pinned_paths,
        )

    _write_artifacts(
        embeddings,
        input_obs_names,
        embedding_path,
        cell_ids_path,
        overwrite=overwrite,
    )

    result: EpiZooEmbeddingToolResult = {
        "status": "success",
        "input_path": str(resolved_input),
        "embedding_path": str(embedding_path),
        "cell_ids_path": str(cell_ids_path),
        "n_cells": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "embedding_dtype": str(embeddings.dtype),
        "finite": finite,
        "cell_order_preserved": order_preserved,
        "backend": "EpiZoo",
        "species": str(metadata["species"]["name"]),
        "checkpoint_path": str(metadata["checkpoint"]["path"]),
        "device": str(metadata["device"]),
    }
    if resource_provenance is not None:
        result["resource_provenance"] = resource_provenance
    return result


__all__ = ["EpiZooEmbeddingToolResult", "EpiZooResourceIdentityError", "epizoo_embed_cells",
           "validate_expected_resource_identity", "validate_resource_provenance"]
