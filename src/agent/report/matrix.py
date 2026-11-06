"""Bounded facts from the freshly independently verified M11.5 matrix."""
from agent.tools.data.raw_history import CellByCCREResult

CONTRACT_FIELDS=frozenset(CellByCCREResult.__annotations__)
FACT_FIELDS=tuple(k for k in CellByCCREResult.__annotations__ if k not in
    ('status','manifest_path','manifest_sha256','matrix_path'))
REPORT_FIELDS=tuple(k for k in FACT_FIELDS if k not in ('artifact_type','artifact_schema_version'))+('upstream_identities',)
