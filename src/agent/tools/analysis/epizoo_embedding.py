"""File-based scientific tool for validated EpiZoo cell embeddings."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Literal, TypedDict

import anndata as ad
import numpy as np

from agent.tools.data import inspect_scATAC
from agent.tools.models import epizoo as epizoo_backend


class EpiZooEmbeddingToolResult(TypedDict):
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
    model = epizoo_backend.load_model(
        checkpoint_path=checkpoint_path,
        device=device,
    )
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
    )

    embeddings = np.asarray(backend_result.embeddings)
    result_obs_names = tuple(str(name) for name in backend_result.obs_names)
    finite, order_preserved = _validate_backend_result(
        embeddings, result_obs_names, input_obs_names
    )
    _validate_cell_ids_for_text(input_obs_names)

    _write_artifacts(
        embeddings,
        input_obs_names,
        embedding_path,
        cell_ids_path,
        overwrite=overwrite,
    )

    metadata = backend_result.metadata
    return {
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


__all__ = ["EpiZooEmbeddingToolResult", "epizoo_embed_cells"]
