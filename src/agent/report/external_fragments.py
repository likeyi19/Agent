"""Explicit external-adoption evidence authority; no raw rows/header prose."""
from pathlib import Path

from agent.tools.data import external_fragment_manifest as m
from agent.tools.data.scatac_fragment_import import RECOVERY_POLICY

FACT_FIELDS = ('artifact_type', 'artifact_schema_version', 'contract_version', 'source_profile',
    'source_encoding', 'species', 'assembly', 'namespace', 'n_libraries', 'n_fragment_records',
    'total_support', 'strand_mode')
CONTRACT_FIELDS = frozenset(('status', 'manifest_path', 'manifest_sha256', *FACT_FIELDS))
REPORT_FIELDS = ('route', 'source_profile', 'source_encoding', 'species', 'assembly',
    'reference_identity_sha256', 'namespace', 'n_libraries', 'source_record_count',
    'n_fragment_records', 'total_support', 'n_distinct_fragment_barcodes', 'strand_mode',
    'source_selection', 'source_index_supplied', 'contract_version', 'support_definition',
    'transformation_policy', 'source_content_verification', 'conservation_verification', 'producer_history')


def project_external_fragments(arguments, result, step_id):
    from agent.tools.data.scatac_fragment_import import verify_public_result
    verified = verify_public_result(arguments, result)
    manifest = verified.fragments.manifest; entry = manifest['libraries'][0]; p = entry['provenance']
    record = m.load_adoption_record(p['producer_record']['path'], p['producer_record']['sha256'])
    facts = dict(route='external_fragment_adoption', reference_identity_sha256=manifest['reference']['reference_identity_sha256'],
        source_record_count=record['source']['n_records'], n_distinct_fragment_barcodes=entry['n_distinct_barcodes'],
        source_selection=record['source_selection'], source_index_supplied=record['source_index'] is not None,
        support_definition=m.SUPPORT['definition'], transformation_policy=m.TRANSFORMATION,
        source_content_verification=verified.source_content, conservation_verification=verified.canonicalization_conservation,
        producer_history=verified.producer_history)
    root = Path(result['manifest_path']).parent
    bindings = [
        ('libraries[0].bgzf', 'scatac_fragments_bgzf', entry['bgzf'] | {'path': str(root / entry['bgzf']['path'])},
         'complete_bgzf_crc_canonical_stream_and_source_conservation'),
        ('libraries[0].tabix', 'scatac_fragments_tabix', entry['tabix'] | {'path': str(root / entry['tabix']['path'])},
         'tabix_contig_inventory_full_contig_and_narrow_overlap_queries'),
        ('libraries[0].provenance.profile.resource', 'external_fragment_profile', p['profile']['resource'], 'exact_reviewed_profile_identity'),
        ('libraries[0].provenance.producer_record', 'external_fragment_adoption_record', p['producer_record'], 'independent_complete_source_reparse_and_exact_conservation'),
        ('libraries[0].provenance.sources[0].resource', 'external_fragment_source', record['source']['resource'], 'complete_encoded_hash_decoding_and_structural_validation'),
    ]
    if record['source_index'] is not None:
        bindings.append(('adoption.source_index', 'external_fragment_source_tabix', record['source_index'], 'explicit_source_pair_full_and_narrow_functional_queries'))
    artifacts = []
    for field, kind, resource, basis in bindings:
        artifacts.append(dict(producing_step_id=step_id, tool_name='import_scATAC_fragments', result_field='manifest_path',
            source_manifest_field=field, namespace=entry['namespace'], artifact_kind=kind, artifact_path=resource['path'],
            integrity={'authoritative_digest': {'algorithm': 'sha256', 'value': resource['sha256'], 'source_manifest_field': field + '.sha256'},
                       'verification_basis': ('fresh_independent_external_adoption_verification', 'full_artifact_sha256', basis)}))
    return facts, artifacts
