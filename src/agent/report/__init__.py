"""Verified result boundaries for deterministic scientific reporting."""

from .evidence import (
    ANALYSIS_EVIDENCE_ARTIFACT_TYPE,
    ANALYSIS_EVIDENCE_FILENAME,
    ANALYSIS_EVIDENCE_SCHEMA_VERSION,
    AnalysisEvidenceError,
    AnalysisEvidenceResult,
    build_analysis_evidence,
    verify_analysis_evidence,
)
from .visualization import (
    ANALYSIS_VISUALIZATION_ARTIFACT_TYPE,
    ANALYSIS_VISUALIZATION_BUNDLE_DIRNAME,
    ANALYSIS_VISUALIZATION_MANIFEST_FILENAME,
    ANALYSIS_VISUALIZATION_SCHEMA_VERSION,
    PLOTTING_SPEC_VERSION,
    AnalysisVisualizationError,
    AnalysisVisualizationResult,
    VisualizationFigureResult,
    build_analysis_visualizations,
    verify_analysis_visualizations,
)

__all__ = [
    "ANALYSIS_EVIDENCE_ARTIFACT_TYPE",
    "ANALYSIS_EVIDENCE_FILENAME",
    "ANALYSIS_EVIDENCE_SCHEMA_VERSION",
    "AnalysisEvidenceError",
    "AnalysisEvidenceResult",
    "build_analysis_evidence",
    "verify_analysis_evidence",
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
