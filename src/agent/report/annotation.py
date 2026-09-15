"""Bounded annotation facts; optional validation never implies annotation accuracy."""
from agent.tools.analysis.scatac_annotation import CellTypeAnnotationResult
CONTRACT_FIELDS = frozenset(CellTypeAnnotationResult.__annotations__)
FACT_FIELDS = tuple(k for k in CellTypeAnnotationResult.__annotations__ if k not in
                    ('status','manifest_path','manifest_sha256','annotated_h5ad_path'))
REPORT_FIELDS = tuple(k for k in FACT_FIELDS if k not in ('artifact_type','artifact_schema_version'))
