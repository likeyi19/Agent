"""Bounded, verified fragment-derived facts without QC or readiness inference."""
from dataclasses import asdict
from pathlib import Path
from .matrix import CONTRACT_FIELDS, FACT_FIELDS
from agent.tools.data.fragment_feature_tool import verify_public_result, _publication
from agent.tools.data.regulatory_feature_reference import load_regulatory_feature_reference
from agent.tools.data.scatac_fragments_v2 import load_fragments_manifest_v2

DERIVED_FIELDS = ('upstream_identities','feature_category','feature_provenance','fragment_source_scope','execution_identity','publication_identity')
REPORT_FIELDS = tuple(k for k in FACT_FIELDS if k not in ('artifact_type','artifact_schema_version')) + DERIVED_FIELDS


def project(arguments, result, execution_identity):
    value = verify_public_result(arguments, result)
    ref = value['upstream']['reference']; fr = value['upstream']['fragments']
    destination = Path(result['manifest_path']).parent.parent
    _, expected_destination, token = _publication(arguments, execution_identity)
    if destination != expected_destination:
        raise ValueError('Matrix publication differs from durable execution.')
    _, reference, _ = load_regulatory_feature_reference(ref['manifest_path'],expected_sha256=ref['manifest_sha256'])
    fragments = load_fragments_manifest_v2(fr['manifest_path'],expected_sha256=fr['manifest_sha256'])
    return dict(execution_identity=execution_identity,
        publication_identity=token, upstream_identities={k:{f:p[f] for f in ('manifest_sha256','identity_sha256','contract_version')}
        for k,p in value['upstream'].items()}, feature_category=reference.features.category,
        feature_provenance=asdict(reference.features.bed.provenance),
        fragment_source_scope=sorted({p['provenance']['kind'] for p in fragments['libraries']})), []
