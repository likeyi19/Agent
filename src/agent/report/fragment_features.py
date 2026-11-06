"""Bounded, verified fragment-derived facts without QC or readiness inference."""
from dataclasses import asdict
from pathlib import Path
from .matrix import CONTRACT_FIELDS, FACT_FIELDS
from agent.tools.data.regulatory_feature_reference import load_regulatory_feature_reference
from agent.tools.data.scatac_fragments_v2 import load_fragments_manifest_v2

DERIVED_FIELDS = ('upstream_identities','feature_category','feature_provenance','fragment_source_scope','execution_identity','publication_identity')
REPORT_FIELDS = tuple(k for k in FACT_FIELDS if k not in ('artifact_type','artifact_schema_version')) + DERIVED_FIELDS
