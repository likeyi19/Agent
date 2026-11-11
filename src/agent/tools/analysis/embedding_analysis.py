"""Shared embedding file I/O and exact ordered-cell identities."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
from typing import Callable

import anndata as ad
import numpy as np


EPIZOO_EMBEDDING_DIM = 512
EPIZOO_REPRESENTATION_KEY = "X_epizoo"
EPIZOO_REPRESENTATION = dict(method='EpiZoo', representation=EPIZOO_REPRESENTATION_KEY,
                             embedding_dim=EPIZOO_EMBEDDING_DIM, embedding_dtype='float32')


def _ensure_output_available(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Analysis output artifact already exists: {path}. "
            "Use overwrite=True to replace it."
        )


def _cell_order_digest(cell_ids: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for cell_id in cell_ids:
        encoded = cell_id.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _all_finite(array: np.ndarray, *, rows_per_chunk: int = 8192) -> bool:
    for start in range(0, array.shape[0], rows_per_chunk):
        if not np.isfinite(array[start : start + rows_per_chunk]).all():
            return False
    return True


def _load_embeddings(path: Path) -> np.ndarray:
    try:
        embeddings = np.load(path, mmap_mode="r", allow_pickle=False)
    except Exception as exc:
        raise ValueError(f"Unable to read embedding npy artifact: {path}") from exc
    if not isinstance(embeddings, np.ndarray):
        raise ValueError("Embedding artifact did not contain a NumPy array.")
    if embeddings.ndim != 2:
        raise ValueError("Embedding artifact must have exactly two dimensions.")
    if embeddings.shape[0] <= 0:
        raise ValueError("Embedding artifact must contain at least one cell.")
    if embeddings.shape[1] != EPIZOO_EMBEDDING_DIM:
        raise ValueError(
            f"Embedding dimension must be {EPIZOO_EMBEDDING_DIM}; "
            f"received {embeddings.shape[1]}."
        )
    if embeddings.dtype != np.dtype(np.float32):
        raise ValueError(
            f"Embedding dtype must be float32; received {embeddings.dtype}."
        )
    if not _all_finite(embeddings):
        raise ValueError("Embedding artifact contains non-finite values.")
    return embeddings


def _load_cell_ids(path: Path) -> tuple[str, ...]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            cell_ids = tuple(line.rstrip("\n\r") for line in handle)
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Unable to read cell-ID sidecar artifact: {path}") from exc
    if not cell_ids:
        raise ValueError("Cell-ID sidecar must contain at least one identifier.")
    if any(not cell_id for cell_id in cell_ids):
        raise ValueError("Cell-ID sidecar contains an empty identifier.")
    if len(set(cell_ids)) != len(cell_ids):
        raise ValueError("Cell-ID sidecar contains duplicate identifiers.")
    return cell_ids


def _read_analysis(path: Path) -> ad.AnnData:
    try:
        return ad.read_h5ad(path)
    except Exception as exc:
        raise ValueError(f"Unable to read analysis artifact: {path}") from exc


def _atomic_write_h5ad(
    adata: ad.AnnData,
    output_path: Path,
    *,
    overwrite: bool,
    validator: Callable[[ad.AnnData], object],
) -> None:
    _ensure_output_available(output_path, overwrite=overwrite)
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=output_path.parent,
            prefix=f".{output_path.stem}.",
            suffix=".tmp.h5ad",
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        adata.write_h5ad(temporary_path)
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        written = _read_analysis(temporary_path)
        try:
            validator(written)
        finally:
            if getattr(written, "file", None) is not None:
                written.file.close()
        _ensure_output_available(output_path, overwrite=overwrite)
        os.replace(temporary_path, output_path)
        temporary_path = None
        directory_fd = os.open(output_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _analysis_digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()
