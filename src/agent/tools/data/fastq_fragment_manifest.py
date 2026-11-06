"""Frozen accepted FASTQ method identity; no producer or verifier remains."""
from . import _chromap as c
from ._fragments_common import MAX_SUPPORT, MAX_TOTAL, PACKAGING_POLICY, canonical

ARTIFACT_TYPE = "agent.fastq-fragment-production"
CONTRACT_VERSION = "fastq-fragment-production.v1"
SEMANTICS = {
    'coordinates': '0-based-half-open',
    'order': 'fai-rank,start-numeric,end-numeric,barcode-ascii.v1',
    'support': c.SUPPORT_SEMANTICS,
    'tn5': 'Chromap-only:+4-start,-5-end;no-second-shift',
    'distinct_barcodes': 'observed-accepted-fragment-tokens;not-called-cells',
    'individual_support_max': MAX_SUPPORT,
    'aggregate_support_max': MAX_TOTAL,
    'stream_identity': 'sha256(canonical-utf8-five-tab-fields-with-LF-in-order)',
}

PROFILE = dict(artifact_type="agent.fastq-fragment-profile", schema_version=1,
    profile_id=c.BACKEND_POLICY, backend_policy=c.BACKEND_POLICY,
    executable_sha256=c.QUALIFIED_EXECUTABLE_SHA256, mapping_policy_sha256=c.policy_sha256(),
    index_policy=c.INDEX_POLICY, semantics=SEMANTICS, packaging=PACKAGING_POLICY)
PROFILE_BYTES = canonical(PROFILE)
