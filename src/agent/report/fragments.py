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


def project_fragments(arguments, result, step_id):
    """Reverify before deriving facts; no production execution or raw payloads."""
    from agent.tools.data.scatac_fragments import verify_public_result
    manifest = verify_public_result(arguments, result)
    from agent.tools.data.fastq_fragment_manifest import record_for_manifest
    record = record_for_manifest(manifest, Path(result['manifest_path']).parent)
    libraries = manifest['libraries']
    selected = sorted(libraries, key=lambda entry: entry['namespace'])[:LIBRARY_SUMMARY_LIMIT]
    facts = {
        'input_kind': 'FASTQ',
        'backend_policy': record['backend']['backend_policy'],
        'upstream_version': record['backend']['upstream_version'],
        'upstream_commit': record['backend']['upstream_commit'],
        'support_definition': SUPPORT_DEFINITION,
        'library_summary': [dict(namespace=entry['namespace'],
            n_fragment_records=entry['n_fragment_records'], total_support=entry['sum_support'],
            n_distinct_fragment_barcodes=entry['n_distinct_barcodes'],
            max_support=entry['max_support']) for entry in selected],
        'n_libraries_omitted_from_summary': len(libraries) - len(selected),
    }
    artifacts = []
    root = Path(result['manifest_path']).parent
    # All artifact identities remain represented (the manifest caps libraries at
    # 4096). Only the human-facing summary is truncated, explicitly and stably.
    for ordinal, entry in enumerate(libraries):
        for kind in ('bgzf', 'tabix'):
            field = f'libraries[{ordinal}].{kind}'
            artifacts.append({
                'producing_step_id': step_id, 'tool_name': 'prepare_scATAC_fragments',
                'result_field': 'manifest_path', 'source_manifest_field': field,
                'namespace': entry['namespace'],
                'artifact_kind': 'scatac_fragments_bgzf' if kind == 'bgzf' else 'scatac_fragments_tabix',
                'artifact_path': str(root / entry[kind]['path']),
                'integrity': {
                    'authoritative_digest': {'algorithm': 'sha256', 'value': entry[kind]['sha256'],
                        'source_manifest_field': field + '.sha256'},
                    'verification_basis': (
                        'independent_fragments_verification', 'full_artifact_sha256',
                        'full_bgzf_crc_and_canonical_record_stream_verification' if kind == 'bgzf'
                        else 'tabix_contig_inventory_full_contig_and_narrow_overlap_queries',
                    ),
                },
            })
    for key, kind in (('profile', 'fastq_fragment_profile'), ('producer_record', 'fastq_fragment_production_record')):
        p = libraries[0]['provenance']
        resource = p['profile']['resource'] if key == 'profile' else p['producer_record']
        field = 'libraries[0].provenance.' + ('profile.resource' if key == 'profile' else 'producer_record')
        artifacts.append(dict(producing_step_id=step_id, tool_name='prepare_scATAC_fragments',
            result_field='manifest_path', source_manifest_field=field, namespace=libraries[0]['namespace'],
            artifact_kind=kind, artifact_path=resource['path'], integrity={
                'authoritative_digest': {'algorithm': 'sha256', 'value': resource['sha256'],
                    'source_manifest_field': field + '.sha256'},
                'verification_basis': ('independent_fragments_verification', 'full_artifact_sha256')}))
    return facts, artifacts
