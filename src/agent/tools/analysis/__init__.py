"""Scientific analysis tools."""

from .clustering_evaluation import (
    CellClusteringEvaluationToolResult,
    evaluate_cell_clustering,
)
from .embedding_analysis import (
    CellClusteringToolResult,
    CellNeighborsToolResult,
    CellUMAPToolResult,
    build_cell_neighbors,
    cluster_cells,
    compute_cell_umap,
)
from .epizoo_embedding import EpiZooEmbeddingToolResult, epizoo_embed_cells
from .label_transfer import CellLabelTransferToolResult, transfer_cell_labels
from .replicate_pseudobulk import (
    ReplicatePseudobulkToolResult,
    ScATACFeatureSpaceToolResult,
    build_replicate_pseudobulk,
    validate_scATAC_feature_space,
)
from .differential_accessibility_backend import (
    ReplicateDifferentialAccessibilityToolResult,
    run_replicate_differential_accessibility,
)

from .scatac_annotation import CellTypeAnnotationResult, annotate_scATAC_cell_types

__all__ = [
    "CellTypeAnnotationResult",
    "annotate_scATAC_cell_types",
    "CellClusteringEvaluationToolResult",
    "CellClusteringToolResult",
    "CellNeighborsToolResult",
    "CellLabelTransferToolResult",
    "CellUMAPToolResult",
    "EpiZooEmbeddingToolResult",
    "ReplicatePseudobulkToolResult",
    "ReplicateDifferentialAccessibilityToolResult",
    "ScATACFeatureSpaceToolResult",
    "build_cell_neighbors",
    "cluster_cells",
    "compute_cell_umap",
    "epizoo_embed_cells",
    "evaluate_cell_clustering",
    "transfer_cell_labels",
    "build_replicate_pseudobulk",
    "run_replicate_differential_accessibility",
    "validate_scATAC_feature_space",
]
