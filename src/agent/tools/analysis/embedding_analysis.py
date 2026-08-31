"""Copy-on-write Scanpy analysis tools for EpiZoo cell embeddings."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
from importlib import metadata
import math
import os
from pathlib import Path
import tempfile
from typing import Callable, Literal, TypedDict

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


EPIZOO_EMBEDDING_DIM = 512
EPIZOO_REPRESENTATION_KEY = "X_epizoo"
LEIDEN_KEY = "leiden"
UMAP_KEY = "X_umap"
PROVENANCE_KEY = "agent_milestone6"
PROVENANCE_SCHEMA_VERSION = 1


class CellNeighborsToolResult(TypedDict):
    """Lightweight result for a persisted Scanpy neighbor graph."""

    status: Literal["success"]
    embedding_path: str
    cell_ids_path: str
    analysis_path: str
    n_cells: int
    embedding_dim: int
    n_neighbors: int
    metric: str
    neighbors_method: str
    transformer: str
    random_seed: int
    connectivities_nnz: int
    distances_nnz: int
    finite: bool
    cell_order_preserved: bool
    backend: str
    software_versions: dict[str, str]


class CellClusteringToolResult(TypedDict):
    """Lightweight result for persisted Leiden cell clustering."""

    status: Literal["success"]
    input_analysis_path: str
    analysis_path: str
    n_cells: int
    n_clusters: int
    cluster_key: str
    algorithm: str
    resolution: float
    random_seed: int
    cell_order_preserved: bool
    backend: str
    software_versions: dict[str, str]


class CellUMAPToolResult(TypedDict):
    """Lightweight result for a persisted two-dimensional UMAP."""

    status: Literal["success"]
    input_analysis_path: str
    analysis_path: str
    n_cells: int
    n_components: int
    umap_key: str
    coordinate_dtype: str
    finite: bool
    min_dist: float
    spread: float
    random_seed: int
    cell_order_preserved: bool
    backend: str
    software_versions: dict[str, str]


def _software_versions() -> dict[str, str]:
    distributions = {
        "scanpy": "scanpy",
        "anndata": "anndata",
        "numpy": "numpy",
        "scipy": "scipy",
        "scikit_learn": "scikit-learn",
        "umap_learn": "umap-learn",
        "igraph": "igraph",
    }
    versions: dict[str, str] = {}
    for key, distribution in distributions.items():
        try:
            versions[key] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[key] = "unavailable"
    return versions


def _resolve_existing_file(
    value: str | Path, argument_name: str, *, suffix: str
) -> Path:
    if not isinstance(value, (str, Path)):
        raise TypeError(f"`{argument_name}` must be a string or pathlib.Path.")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Required input file does not exist: {path}")
    if path.suffix.casefold() != suffix:
        raise ValueError(f"`{argument_name}` must identify a {suffix} file: {path}")
    return path


def _resolve_output_dir(value: str | Path) -> Path:
    if not isinstance(value, (str, Path)):
        raise TypeError("`output_dir` must be a string or pathlib.Path.")
    path = Path(value).expanduser().resolve()
    if path.exists() and not path.is_dir():
        raise ValueError(f"Analysis output path is not a directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _validate_overwrite(overwrite: bool) -> None:
    if not isinstance(overwrite, bool):
        raise TypeError("`overwrite` must be a boolean.")


def _validate_seed(random_seed: int) -> None:
    if isinstance(random_seed, bool) or not isinstance(random_seed, int):
        raise TypeError("`random_seed` must be a nonnegative integer.")
    if random_seed < 0:
        raise ValueError("`random_seed` must be a nonnegative integer.")


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"`{name}` must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"`{name}` must be a finite number.")
    return result


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


def _provenance(adata: ad.AnnData) -> Mapping[str, object]:
    value = adata.uns.get(PROVENANCE_KEY)
    if not isinstance(value, Mapping):
        raise ValueError("Analysis artifact lacks Milestone 6 provenance.")
    if value.get("schema_version") != PROVENANCE_SCHEMA_VERSION:
        raise ValueError("Analysis artifact uses unsupported Milestone 6 provenance.")
    return value


def _cell_ids_from_adata(adata: ad.AnnData) -> tuple[str, ...]:
    cell_ids = tuple(str(value) for value in adata.obs_names)
    if len(cell_ids) != adata.n_obs or not cell_ids:
        raise ValueError("Analysis artifact has invalid cell identifiers.")
    if any(not value for value in cell_ids) or len(set(cell_ids)) != len(cell_ids):
        raise ValueError("Analysis artifact cell identifiers must be nonempty and unique.")
    return cell_ids


def _validate_sparse_graph(adata: ad.AnnData, key: str) -> sparse.spmatrix:
    if key not in adata.obsp:
        raise ValueError(f"Analysis artifact lacks sparse graph {key!r}.")
    graph = adata.obsp[key]
    if not sparse.issparse(graph):
        raise ValueError(f"Analysis graph {key!r} must remain sparse.")
    if graph.shape != (adata.n_obs, adata.n_obs):
        raise ValueError(f"Analysis graph {key!r} has an invalid shape.")
    if not np.isfinite(graph.data).all():
        raise ValueError(f"Analysis graph {key!r} contains non-finite values.")
    return graph


def _validate_neighbors_artifact(
    adata: ad.AnnData, *, expected_stages: frozenset[str]
) -> tuple[tuple[str, ...], sparse.spmatrix, sparse.spmatrix]:
    if adata.n_obs <= 0 or adata.n_vars != 0 or adata.X is not None:
        raise ValueError("Analysis artifact is not a compact embedding-only AnnData.")
    cell_ids = _cell_ids_from_adata(adata)
    provenance = _provenance(adata)
    if provenance.get("stage") not in expected_stages:
        raise ValueError("Analysis artifact has an unexpected Milestone 6 stage.")
    if provenance.get("cell_order_sha256") != _cell_order_digest(cell_ids):
        raise ValueError("Analysis artifact cell-order provenance is inconsistent.")
    if EPIZOO_REPRESENTATION_KEY not in adata.obsm:
        raise ValueError("Analysis artifact lacks the EpiZoo representation.")
    embeddings = np.asarray(adata.obsm[EPIZOO_REPRESENTATION_KEY])
    if embeddings.shape != (adata.n_obs, EPIZOO_EMBEDDING_DIM):
        raise ValueError("Analysis artifact has an invalid EpiZoo representation shape.")
    if embeddings.dtype != np.dtype(np.float32) or not _all_finite(embeddings):
        raise ValueError("Analysis artifact has an invalid EpiZoo representation.")
    neighbors = adata.uns.get("neighbors")
    if not isinstance(neighbors, Mapping):
        raise ValueError("Analysis artifact lacks Scanpy neighbors metadata.")
    if neighbors.get("connectivities_key") != "connectivities":
        raise ValueError("Analysis artifact has invalid connectivities metadata.")
    if neighbors.get("distances_key") != "distances":
        raise ValueError("Analysis artifact has invalid distances metadata.")
    distances = _validate_sparse_graph(adata, "distances")
    connectivities = _validate_sparse_graph(adata, "connectivities")
    return cell_ids, distances, connectivities


def _validate_cluster_labels(adata: ad.AnnData) -> int:
    if LEIDEN_KEY not in adata.obs:
        raise ValueError("Clustered analysis artifact lacks Leiden labels.")
    labels = adata.obs[LEIDEN_KEY]
    if len(labels) != adata.n_obs or labels.isna().any():
        raise ValueError("Clustered analysis artifact has invalid Leiden labels.")
    n_clusters = int(labels.nunique(dropna=True))
    if n_clusters <= 0:
        raise ValueError("Clustered analysis artifact has no Leiden clusters.")
    return n_clusters


def _read_analysis(path: Path) -> ad.AnnData:
    try:
        return ad.read_h5ad(path)
    except Exception as exc:
        raise ValueError(f"Unable to read Milestone 6 analysis artifact: {path}") from exc


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


def build_cell_neighbors(
    embedding_path: str | Path,
    cell_ids_path: str | Path,
    output_dir: str | Path,
    *,
    n_neighbors: int = 15,
    metric: Literal["euclidean", "cosine"] = "euclidean",
    random_seed: int = 0,
    overwrite: bool = False,
) -> CellNeighborsToolResult:
    """Build and persist a sparse Scanpy neighbor graph from EpiZoo embeddings."""

    _validate_overwrite(overwrite)
    _validate_seed(random_seed)
    if isinstance(n_neighbors, bool) or not isinstance(n_neighbors, int):
        raise TypeError("`n_neighbors` must be an integer.")
    if metric not in {"euclidean", "cosine"}:
        raise ValueError("`metric` must be 'euclidean' or 'cosine'.")

    resolved_embedding = _resolve_existing_file(
        embedding_path, "embedding_path", suffix=".npy"
    )
    resolved_cell_ids = _resolve_existing_file(
        cell_ids_path, "cell_ids_path", suffix=".txt"
    )
    embeddings = _load_embeddings(resolved_embedding)
    cell_ids = _load_cell_ids(resolved_cell_ids)
    if len(cell_ids) != embeddings.shape[0]:
        raise ValueError(
            "Cell-ID count must exactly equal the number of embedding rows."
        )
    if not 2 <= n_neighbors < embeddings.shape[0]:
        raise ValueError("`n_neighbors` must satisfy 2 <= n_neighbors < n_cells.")

    resolved_output_dir = _resolve_output_dir(output_dir)
    output_path = resolved_output_dir / f"{resolved_embedding.stem}.neighbors.h5ad"
    _ensure_output_available(output_path, overwrite=overwrite)

    adata = ad.AnnData(obs=pd.DataFrame(index=pd.Index(cell_ids, dtype="object")))
    adata.obsm[EPIZOO_REPRESENTATION_KEY] = np.asarray(embeddings)
    versions = _software_versions()
    adata.uns[PROVENANCE_KEY] = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "stage": "neighbors",
        "cell_order_sha256": _cell_order_digest(cell_ids),
        "source_embedding_path": str(resolved_embedding),
        "source_cell_ids_path": str(resolved_cell_ids),
        "parameters": {
            "neighbors": {
                "n_neighbors": n_neighbors,
                "metric": metric,
                "method": "umap",
                "transformer": "none",
                "random_seed": random_seed,
                "use_rep": EPIZOO_REPRESENTATION_KEY,
            }
        },
        "software_versions": versions,
    }

    import scanpy as sc

    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        use_rep=EPIZOO_REPRESENTATION_KEY,
        metric=metric,
        method="umap",
        transformer=None,
        random_state=random_seed,
    )
    output_cell_ids, distances, connectivities = _validate_neighbors_artifact(
        adata, expected_stages=frozenset({"neighbors"})
    )
    order_preserved = output_cell_ids == cell_ids
    if not order_preserved:
        raise RuntimeError("Neighbor construction changed the ordered cell identifiers.")
    _atomic_write_h5ad(
        adata,
        output_path,
        overwrite=overwrite,
        validator=lambda written: _validate_neighbors_artifact(
            written, expected_stages=frozenset({"neighbors"})
        ),
    )
    return {
        "status": "success",
        "embedding_path": str(resolved_embedding),
        "cell_ids_path": str(resolved_cell_ids),
        "analysis_path": str(output_path),
        "n_cells": int(adata.n_obs),
        "embedding_dim": EPIZOO_EMBEDDING_DIM,
        "n_neighbors": n_neighbors,
        "metric": metric,
        "neighbors_method": "umap",
        "transformer": "none",
        "random_seed": random_seed,
        "connectivities_nnz": int(connectivities.nnz),
        "distances_nnz": int(distances.nnz),
        "finite": True,
        "cell_order_preserved": order_preserved,
        "backend": "Scanpy",
        "software_versions": versions,
    }


def cluster_cells(
    analysis_path: str | Path,
    output_dir: str | Path,
    *,
    resolution: float = 1.0,
    random_seed: int = 0,
    overwrite: bool = False,
) -> CellClusteringToolResult:
    """Run fixed-setting Leiden clustering on a neighbors analysis artifact."""

    _validate_overwrite(overwrite)
    _validate_seed(random_seed)
    normalized_resolution = _finite_number(resolution, "resolution")
    if normalized_resolution <= 0:
        raise ValueError("`resolution` must be strictly positive.")
    resolved_input = _resolve_existing_file(
        analysis_path, "analysis_path", suffix=".h5ad"
    )
    resolved_output_dir = _resolve_output_dir(output_dir)
    output_path = resolved_output_dir / f"{resolved_input.stem}.clustered.h5ad"
    if output_path == resolved_input:
        raise ValueError("Cluster output must differ from its input artifact.")
    _ensure_output_available(output_path, overwrite=overwrite)

    adata = _read_analysis(resolved_input)
    input_cell_ids, _, _ = _validate_neighbors_artifact(
        adata, expected_stages=frozenset({"neighbors"})
    )

    import scanpy as sc

    sc.tl.leiden(
        adata,
        resolution=normalized_resolution,
        flavor="igraph",
        n_iterations=2,
        directed=False,
        use_weights=True,
        random_state=random_seed,
        key_added=LEIDEN_KEY,
    )
    n_clusters = _validate_cluster_labels(adata)
    output_cell_ids, _, _ = _validate_neighbors_artifact(
        adata, expected_stages=frozenset({"neighbors"})
    )
    order_preserved = output_cell_ids == input_cell_ids
    if not order_preserved:
        raise RuntimeError("Leiden clustering changed the ordered cell identifiers.")
    provenance = deepcopy(dict(_provenance(adata)))
    parameters = deepcopy(dict(provenance.get("parameters", {})))
    parameters["clustering"] = {
        "algorithm": "leiden",
        "resolution": normalized_resolution,
        "flavor": "igraph",
        "n_iterations": 2,
        "directed": False,
        "use_weights": True,
        "random_seed": random_seed,
        "key_added": LEIDEN_KEY,
    }
    provenance["parameters"] = parameters
    provenance["stage"] = "clustering"
    provenance["source_analysis_path"] = str(resolved_input)
    provenance["software_versions"] = _software_versions()
    adata.uns[PROVENANCE_KEY] = provenance

    def validate_written(written: ad.AnnData) -> None:
        _validate_neighbors_artifact(
            written, expected_stages=frozenset({"clustering"})
        )
        _validate_cluster_labels(written)

    _atomic_write_h5ad(
        adata,
        output_path,
        overwrite=overwrite,
        validator=validate_written,
    )
    versions = _software_versions()
    return {
        "status": "success",
        "input_analysis_path": str(resolved_input),
        "analysis_path": str(output_path),
        "n_cells": int(adata.n_obs),
        "n_clusters": n_clusters,
        "cluster_key": LEIDEN_KEY,
        "algorithm": "leiden",
        "resolution": normalized_resolution,
        "random_seed": random_seed,
        "cell_order_preserved": order_preserved,
        "backend": "Scanpy",
        "software_versions": versions,
    }


def compute_cell_umap(
    analysis_path: str | Path,
    output_dir: str | Path,
    *,
    min_dist: float = 0.5,
    spread: float = 1.0,
    random_seed: int = 0,
    overwrite: bool = False,
) -> CellUMAPToolResult:
    """Compute and persist a fixed two-dimensional UMAP from a clustered artifact."""

    _validate_overwrite(overwrite)
    _validate_seed(random_seed)
    normalized_min_dist = _finite_number(min_dist, "min_dist")
    normalized_spread = _finite_number(spread, "spread")
    if normalized_min_dist < 0:
        raise ValueError("`min_dist` must be nonnegative.")
    if normalized_spread <= 0:
        raise ValueError("`spread` must be strictly positive.")
    if normalized_min_dist > normalized_spread:
        raise ValueError("`min_dist` must not exceed `spread`.")
    resolved_input = _resolve_existing_file(
        analysis_path, "analysis_path", suffix=".h5ad"
    )
    resolved_output_dir = _resolve_output_dir(output_dir)
    output_path = resolved_output_dir / f"{resolved_input.stem}.umap.h5ad"
    if output_path == resolved_input:
        raise ValueError("UMAP output must differ from its input artifact.")
    _ensure_output_available(output_path, overwrite=overwrite)

    adata = _read_analysis(resolved_input)
    input_cell_ids, _, _ = _validate_neighbors_artifact(
        adata, expected_stages=frozenset({"clustering"})
    )
    _validate_cluster_labels(adata)

    import scanpy as sc

    sc.tl.umap(
        adata,
        min_dist=normalized_min_dist,
        spread=normalized_spread,
        n_components=2,
        init_pos="spectral",
        random_state=random_seed,
    )
    if UMAP_KEY not in adata.obsm:
        raise RuntimeError("Scanpy did not produce X_umap coordinates.")
    coordinates = np.asarray(adata.obsm[UMAP_KEY])
    if coordinates.shape != (adata.n_obs, 2) or not _all_finite(coordinates):
        raise RuntimeError("Scanpy produced invalid UMAP coordinates.")
    output_cell_ids, _, _ = _validate_neighbors_artifact(
        adata, expected_stages=frozenset({"clustering"})
    )
    _validate_cluster_labels(adata)
    order_preserved = output_cell_ids == input_cell_ids
    if not order_preserved:
        raise RuntimeError("UMAP computation changed the ordered cell identifiers.")
    provenance = deepcopy(dict(_provenance(adata)))
    parameters = deepcopy(dict(provenance.get("parameters", {})))
    parameters["umap"] = {
        "min_dist": normalized_min_dist,
        "spread": normalized_spread,
        "n_components": 2,
        "init_pos": "spectral",
        "random_seed": random_seed,
        "key_added": UMAP_KEY,
    }
    provenance["parameters"] = parameters
    provenance["stage"] = "umap"
    provenance["source_analysis_path"] = str(resolved_input)
    provenance["software_versions"] = _software_versions()
    adata.uns[PROVENANCE_KEY] = provenance

    def validate_written(written: ad.AnnData) -> None:
        _validate_neighbors_artifact(written, expected_stages=frozenset({"umap"}))
        _validate_cluster_labels(written)
        written_coordinates = np.asarray(written.obsm[UMAP_KEY])
        if written_coordinates.shape != (written.n_obs, 2):
            raise ValueError("Written UMAP coordinates have an invalid shape.")
        if not _all_finite(written_coordinates):
            raise ValueError("Written UMAP coordinates contain non-finite values.")

    _atomic_write_h5ad(
        adata,
        output_path,
        overwrite=overwrite,
        validator=validate_written,
    )
    versions = _software_versions()
    return {
        "status": "success",
        "input_analysis_path": str(resolved_input),
        "analysis_path": str(output_path),
        "n_cells": int(adata.n_obs),
        "n_components": 2,
        "umap_key": UMAP_KEY,
        "coordinate_dtype": str(coordinates.dtype),
        "finite": True,
        "min_dist": normalized_min_dist,
        "spread": normalized_spread,
        "random_seed": random_seed,
        "cell_order_preserved": order_preserved,
        "backend": "Scanpy",
        "software_versions": versions,
    }


__all__ = [
    "CellClusteringToolResult",
    "CellNeighborsToolResult",
    "CellUMAPToolResult",
    "build_cell_neighbors",
    "cluster_cells",
    "compute_cell_umap",
]
