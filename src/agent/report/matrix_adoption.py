"""Whitelisted external conservation facts; no inferred biological provenance."""
from agent.tools.data.scatac_matrix_adoption import AdoptedCellByCCREResult
CONTRACT_FIELDS=frozenset(AdoptedCellByCCREResult.__annotations__)
FACT_FIELDS=tuple(k for k in CONTRACT_FIELDS if k not in ('status','manifest_path','manifest_sha256','matrix_path'))
# Stable field order is part of deterministic reporting.
FACT_FIELDS=tuple(k for k in AdoptedCellByCCREResult.__annotations__ if k in FACT_FIELDS)
REPORT_FIELDS=tuple(k for k in FACT_FIELDS if k not in ('artifact_type','artifact_schema_version'))
