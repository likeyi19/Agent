"""Structured scientific tools."""

from .analysis import (
    CellClusteringToolResult,
    CellNeighborsToolResult,
    CellUMAPToolResult,
    EpiZooEmbeddingToolResult,
    build_cell_neighbors,
    cluster_cells,
    compute_cell_umap,
    epizoo_embed_cells,
)
from .data import ScATACInspection, inspect_scATAC

__all__ = [
    "CellClusteringToolResult",
    "CellNeighborsToolResult",
    "CellUMAPToolResult",
    "EpiZooEmbeddingToolResult",
    "ScATACInspection",
    "build_cell_neighbors",
    "cluster_cells",
    "compute_cell_umap",
    "epizoo_embed_cells",
    "inspect_scATAC",
]
