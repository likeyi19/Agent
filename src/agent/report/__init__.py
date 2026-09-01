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

__all__ = [
    "ANALYSIS_EVIDENCE_ARTIFACT_TYPE",
    "ANALYSIS_EVIDENCE_FILENAME",
    "ANALYSIS_EVIDENCE_SCHEMA_VERSION",
    "AnalysisEvidenceError",
    "AnalysisEvidenceResult",
    "build_analysis_evidence",
    "verify_analysis_evidence",
]
