"""Bounded, freshly verified BAM evidence facts and artifact authority."""
from pathlib import Path
from agent.tools.data import bam_fragment_manifest as m
from agent.tools.data.scatac_bam_fragments import BamFragmentsResult

FACT_FIELDS=tuple(k for k in BamFragmentsResult.__annotations__ if k not in ('status','manifest_path','manifest_sha256'))
CONTRACT_FIELDS=frozenset(BamFragmentsResult.__annotations__)
REPORT_FIELDS=('route','source_profile','source_bam_sha256','species','assembly','reference_identity_sha256',
    'namespace','n_libraries','n_templates','n_decoded_records','eligible_pairs','exclusion_accounting',
    'n_fragment_records','total_support','n_distinct_fragment_barcodes','support_definition',
    'mapq_policy','tn5_policy','strand_mode','source_content_verification','bam_transformation_verification',
    'output_verification','source_history','producer_history','contract_version')


def project_bam_fragments(arguments,result,step_id):
    from agent.tools.data.scatac_bam_fragments import verify_public_result
    verified=verify_public_result(arguments,result);manifest=verified.fragments.manifest
    entry=manifest['libraries'][0];p=entry['provenance'];root=Path(result['manifest_path']).parent
    record=m.load_record(p['producer_record']['path'],p['producer_record']['sha256'])
    facts=dict(route='bam_fragment_production',source_bam_sha256=record['source']['sha256'],
        reference_identity_sha256=manifest['reference']['reference_identity_sha256'],
        n_decoded_records=record['qualification']['n_records'],exclusion_accounting=record['qualification']['exclusions'],
        n_distinct_fragment_barcodes=entry['n_distinct_barcodes'],support_definition=m.SUPPORT['definition'],
        mapq_policy=m.PROFILE['mapq'],tn5_policy=m.PROFILE['tn5'],source_history=m.HISTORY,
        source_content_verification=verified.source_content,bam_transformation_verification=verified.transformation,
        output_verification='verified',producer_history=verified.producer_history)
    resources=[('libraries[0].'+k,'scatac_fragments_'+k,entry[k]|{'path':str(root/entry[k]['path'])}) for k in ('bgzf','tabix')]
    resources += [('libraries[0].provenance.profile.resource','bam_fragment_profile',p['profile']['resource']),
        ('libraries[0].provenance.producer_record','bam_fragment_production_record',p['producer_record'])]
    resources += [(f'libraries[0].provenance.sources[{i}].resource','bam_fragment_'+s['role'],s['resource']) for i,s in enumerate(p['sources'])]
    artifacts=[dict(producing_step_id=step_id,tool_name='prepare_scATAC_bam_fragments',result_field='manifest_path',
        source_manifest_field=field,namespace=entry['namespace'],artifact_kind=kind,artifact_path=r['path'],
        integrity={'authoritative_digest':{'algorithm':'sha256','value':r['sha256'],'source_manifest_field':field+'.sha256'},
        'verification_basis':('independent_bam_transformation_verification','full_artifact_sha256','generic_v2_artifact_verification')})
        for field,kind,r in resources]
    return facts,artifacts
