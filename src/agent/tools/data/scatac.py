"""Lightweight, read-only inspection of scATAC-seq AnnData files."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import anndata as ad
import scipy.sparse as sp


_NAME_SAMPLE_SIZE = 5


def read_feature_names(path: str | Path) -> list[str]:
    """Read the bounded, ordered feature index, without consuming X or annotations.

    This is a vocabulary observation, not scientific matrix qualification. Legacy
    AnnData dataframe 0.1.0 indexes and current 0.2.0 indexes have the same meaning.
    Only consumed HDF5 objects are checked; unrelated annotations impose no gate.
    """
    import h5py

    if Path(path).stat().st_size > 1024 ** 3:
        raise ValueError('H5AD exceeds the supported vocabulary admission size.')
    with h5py.File(path, 'r') as handle:
        if not isinstance(handle.get('var', getlink=True), h5py.HardLink):
            raise ValueError('The feature axis must be stored in the supplied H5AD.')
        var = handle['var']
        if (not isinstance(var, h5py.Group) or var.attrs.get('encoding-type') != 'dataframe'
                or var.attrs.get('encoding-version') not in ('0.1.0', '0.2.0')):
            raise ValueError('Unsupported H5AD feature axis encoding.')
        key = var.attrs.get('_index')
        if (not isinstance(key, str) or not key or '/' in key or key in ('.', '..')
                or not isinstance(var.get(key, getlink=True), h5py.HardLink)):
            raise ValueError('The feature index must be a local dataset.')
        index = var[key]
        if (not isinstance(index, h5py.Dataset) or index.ndim != 1
                or not 1 <= len(index) <= 2_000_000
                or h5py.check_string_dtype(index.dtype) is None
                or index.is_virtual or bool(index.external)):
            raise ValueError('Unsupported H5AD feature index storage or size.')
        creation = index.id.get_create_plist()
        if any(creation.get_filter(i)[0] not in (1, 2, 3)
               for i in range(creation.get_nfilters())):
            raise ValueError('Unsupported H5AD feature index compression.')
        names, total = [], 0
        # Fixed-width reads bound variable-length HDF5 strings before allocation.
        reader = index.astype('S257')
        for left in range(0, len(index), 1024):
            for offset, value in enumerate(reader[left:left + 1024]):
                raw = bytes(value)
                if len(raw) > 256:
                    raw = bytes(index.astype('S65537')[left + offset])
                total += len(raw)
                if not raw or len(raw) > 65_536 or total > 2 * 1024 ** 3:
                    raise ValueError('Feature names exceed the supported decoding bound.')
                names.append(raw.decode('utf-8', errors='strict'))
    if len(set(names)) != len(names):
        raise ValueError('The feature vocabulary must be unique.')
    return names


class ScATACInspection(TypedDict):
    """JSON-serializable summary returned by :func:`inspect_scATAC`."""

    input_path: str
    n_cells: int
    n_features: int
    x_storage_type: str
    x_is_sparse: bool
    x_dtype: str
    nnz: int | None
    density: float | None
    obs_columns: list[str]
    var_columns: list[str]
    obs_names_sample: list[str]
    var_names_sample: list[str]


def _resolve_h5ad_path(path: str | Path) -> Path:
    if not isinstance(path, (str, Path)):
        raise TypeError("`path` must be a string or pathlib.Path.")

    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"scATAC AnnData file not found: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"scATAC AnnData path is not a file: {resolved}")
    if resolved.suffix.lower() != ".h5ad":
        raise ValueError(
            f"Expected a .h5ad AnnData file, but received: {resolved}"
        )
    return resolved


def _is_sparse_matrix(matrix: object) -> bool:
    return sp.issparse(matrix) or isinstance(
        matrix, (ad.abc.CSRDataset, ad.abc.CSCDataset)
    )


def _sparse_nnz(matrix: object) -> int:
    if sp.issparse(matrix):
        return int(matrix.nnz)

    if isinstance(matrix, (ad.abc.CSRDataset, ad.abc.CSCDataset)):
        group = matrix.group
        if "data" not in group:
            raise ValueError(
                "Sparse AnnData X encoding is missing its required `data` array."
            )
        return int(group["data"].shape[0])

    raise TypeError("Cannot calculate nnz for a non-sparse matrix.")


def _storage_type(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def inspect_scATAC(path: str | Path) -> ScATACInspection:
    """Inspect a scATAC-seq ``.h5ad`` file without loading or densifying ``X``.

    The file is opened with AnnData's read-only backed mode. Dense on-disk
    matrices are reported as dense, but are not scanned to calculate ``nnz`` or
    density.
    """

    resolved = _resolve_h5ad_path(path)
    try:
        adata = ad.read_h5ad(resolved, backed="r")
    except Exception as exc:
        raise ValueError(
            f"Unable to read scATAC AnnData file {resolved}: {exc}"
        ) from exc

    try:
        matrix = adata.X
        if matrix is None:
            raise ValueError(
                f"AnnData file {resolved} does not contain an X feature matrix."
            )

        is_sparse = _is_sparse_matrix(matrix)
        nnz = _sparse_nnz(matrix) if is_sparse else None
        element_count = int(adata.n_obs) * int(adata.n_vars)
        density = (
            float(nnz / element_count)
            if nnz is not None and element_count > 0
            else None
        )

        return {
            "input_path": str(resolved),
            "n_cells": int(adata.n_obs),
            "n_features": int(adata.n_vars),
            "x_storage_type": _storage_type(matrix),
            "x_is_sparse": is_sparse,
            "x_dtype": str(matrix.dtype),
            "nnz": nnz,
            "density": density,
            "obs_columns": [str(name) for name in adata.obs.columns],
            "var_columns": [str(name) for name in adata.var.columns],
            "obs_names_sample": [
                str(name) for name in adata.obs_names[:_NAME_SAMPLE_SIZE]
            ],
            "var_names_sample": [
                str(name) for name in adata.var_names[:_NAME_SAMPLE_SIZE]
            ],
        }
    finally:
        adata.file.close()


__all__ = ["ScATACInspection", "inspect_scATAC"]
