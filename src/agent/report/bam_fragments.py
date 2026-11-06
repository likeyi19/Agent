"""Bounded, freshly verified BAM evidence facts and artifact authority."""
from pathlib import Path
from agent.tools.data import bam_fragment_manifest as m
from agent.tools.data.raw_history import BamFragmentsResult

FACT_FIELDS=tuple(k for k in BamFragmentsResult.__annotations__ if k not in ('status','manifest_path','manifest_sha256'))
CONTRACT_FIELDS=frozenset(BamFragmentsResult.__annotations__)
REPORT_FIELDS=('route','source_profile','source_bam_sha256','species','assembly','reference_identity_sha256',
    'namespace','n_libraries','n_templates','n_decoded_records','eligible_pairs','exclusion_accounting',
    'n_fragment_records','total_support','n_distinct_fragment_barcodes','support_definition',
    'mapq_policy','tn5_policy','strand_mode','source_content_verification','bam_transformation_verification',
    'output_verification','source_history','producer_history','contract_version')
