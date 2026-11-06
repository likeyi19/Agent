"""Explicit compact authority for verified barcode metrics, never selected cells."""
from pathlib import Path
from agent.tools.data.raw_history import BarcodeQCResult

CONTRACT_FIELDS=frozenset(BarcodeQCResult.__annotations__)
FACT_FIELDS=tuple(k for k in BarcodeQCResult.__annotations__ if k not in ('status','manifest_path','manifest_sha256'))
REPORT_FIELDS=(*(k for k in FACT_FIELDS if k not in ('artifact_type','artifact_schema_version')),'tss_method','producer_authority','qc_summary')
