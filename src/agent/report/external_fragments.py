"""Explicit external-adoption evidence authority; no raw rows/header prose."""
from pathlib import Path

from agent.tools.data import external_fragment_manifest as m
RECOVERY_POLICY = 'import-scatac-fragments-external-v1'

FACT_FIELDS = ('artifact_type', 'artifact_schema_version', 'contract_version', 'source_profile',
    'source_encoding', 'species', 'assembly', 'namespace', 'n_libraries', 'n_fragment_records',
    'total_support', 'strand_mode')
CONTRACT_FIELDS = frozenset(('status', 'manifest_path', 'manifest_sha256', *FACT_FIELDS))
REPORT_FIELDS = ('route', 'source_profile', 'source_encoding', 'species', 'assembly',
    'reference_identity_sha256', 'namespace', 'n_libraries', 'source_record_count',
    'n_fragment_records', 'total_support', 'n_distinct_fragment_barcodes', 'strand_mode',
    'source_selection', 'source_index_supplied', 'contract_version', 'support_definition',
    'transformation_policy', 'source_content_verification', 'conservation_verification', 'producer_history')
