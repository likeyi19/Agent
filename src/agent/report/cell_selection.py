"""Explicit compact authority for verified QC-selected candidate cells."""
from pathlib import Path
from agent.tools.data.raw_history import CellSelectionResult

CONTRACT_FIELDS=frozenset(CellSelectionResult.__annotations__)
FACT_FIELDS=tuple(k for k in CellSelectionResult.__annotations__ if k not in ('status','manifest_path','manifest_sha256'))
REPORT_FIELDS=(*(k for k in FACT_FIELDS if k not in ('artifact_type','artifact_schema_version')),'effective_thresholds','reason_counts','qc_lineage','selection_profile_sha256')
