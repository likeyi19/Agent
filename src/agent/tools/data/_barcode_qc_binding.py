"""Fresh source authority and operator-owned runtime/resource qualification.

Producer dispatch establishes existing verification authority only. The metric
algorithms receive one common v2 reader and never branch on producer origin.
"""
import json
import os
from dataclasses import dataclass

from . import scatac_qc_reference as qr
from . import scatac_fragments_v2 as v2
from .scatac_fragments_v2_verifier import take_snapshots, check_snapshots
from .scatac_qc_profile import fail




def resource_qualification(bundle):
    if bundle.annotation.qualification=='synthetic_only':
        if (os.environ.get('AGENT_QC_ALLOW_SYNTHETIC')!='1' or bundle.tss_rows>1000
            or sum(c.length for c in bundle.contigs)>1_000_000): fail('QC_RESOURCE_UNQUALIFIED')
        return dict(mode='synthetic_only',catalog_sha256=None,basis='explicit_runtime_synthetic_opt_in')
    from agent.application.shared_resources import current_scientific_library
    library = current_scientific_library()
    if library is not None:
        # Installation owns qualification; exact installed bytes own reuse.
        # A configured library is authoritative, never a catalog fallback.
        for entry in library.entries():
            if entry['kind'] != 'qc':
                continue
            resource = entry['resource']
            _, installed, _ = qr.load_scatac_qc_reference_bundle(
                resource['qc_reference_manifest_path'],
                expected_sha256=resource['qc_reference_manifest_sha256'])
            if installed == bundle:
                library.check(entry)
                return dict(mode='library_qualified', resource_id=entry['id'],
                            basis='scientific_resource_library_installation')
        fail('QC_RESOURCE_UNQUALIFIED')
    configured=os.environ.get('AGENT_QC_RESOURCE_CATALOG')
    if not configured: fail('QC_RESOURCE_UNQUALIFIED')
    v2.absolute_path(configured)
    before=take_snapshots([configured])
    with open(configured,'rb') as f: raw=f.read(65537)
    if len(raw)>65536: fail('QC_RESOURCE_UNQUALIFIED')
    catalog=json.loads(raw,object_pairs_hook=v2._pairs,parse_constant=lambda _:fail('QC_RESOURCE_UNQUALIFIED'))
    v2.shape(catalog,('artifact_type','schema_version','resources'))
    if (catalog['artifact_type']!='agent.qc-resource-qualification-catalog'
        or type(catalog['schema_version']) is not int or catalog['schema_version']!=1
        or type(catalog['resources']) is not list or not 1<=len(catalog['resources'])<=16): fail('QC_RESOURCE_UNQUALIFIED')
    seen=set(); matches=[]
    for r in catalog['resources']:
        v2.shape(r,('resource_identity_sha256','parent_reference_identity_sha256','annotation_sha256','annotation_release','qualification_basis'))
        for k in ('resource_identity_sha256','parent_reference_identity_sha256','annotation_sha256'): v2.sha(r[k])
        v2.token(r['annotation_release']);v2.token(r['qualification_basis'])
        if r['resource_identity_sha256'] in seen: fail('QC_RESOURCE_UNQUALIFIED')
        seen.add(r['resource_identity_sha256'])
        if (r['resource_identity_sha256']==bundle.resource_identity_sha256
            and r['parent_reference_identity_sha256']==bundle.parent_reference_identity_sha256
            and r['annotation_sha256']==bundle.annotation.resource.sha256
            and r['annotation_release']==bundle.annotation.release): matches.append(r)
    if len(matches)!=1: fail('QC_RESOURCE_UNQUALIFIED')
    check_snapshots(before)
    import hashlib
    return dict(mode='operator_qualified',catalog_sha256=hashlib.sha256(raw).hexdigest(),basis=matches[0]['qualification_basis'])




@dataclass(frozen=True)
class BoundQC:
    fragments: object
    reference: qr.ScATACQCReferenceBundle
    producer_authority: dict
    resource_qualification: dict
    executable: str
    backend_identity: dict
    snapshots: tuple
    fragments_authority: object = None

    def unchanged(self):
        self.fragments.verification.check_unchanged();check_snapshots(self.snapshots)
        if self.fragments_authority is not None:
            self.fragments_authority.validate(self.fragments.verification.manifest_path,
                                               self.fragments.verification.manifest_sha256)
        if resource_qualification(self.reference)!=self.resource_qualification: fail('QC_RESOURCE_UNQUALIFIED')
