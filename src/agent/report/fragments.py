"""Explicit compact projection of independently verified canonical fragments."""
from pathlib import Path


FACT_FIELDS = (
    'artifact_type', 'artifact_schema_version', 'contract_version', 'species',
    'assembly', 'n_libraries', 'n_fragment_records', 'total_support',
)
CONTRACT_FIELDS = frozenset(('status', 'manifest_path', 'manifest_sha256', *FACT_FIELDS))
SUPPORT_DEFINITION = (
    'Exact number of Chromap-generated paired mappings assigned to the accepted '
    'cell-level duplicate group under chromap-atac-agent-support-v1, including '
    'the retained representative.'
)
LIBRARY_SUMMARY_LIMIT = 50
