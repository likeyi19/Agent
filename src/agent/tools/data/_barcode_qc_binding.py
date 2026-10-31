"""Fresh source authority and operator-owned runtime/resource qualification.

Producer dispatch establishes existing verification authority only. The metric
algorithms receive one common v2 reader and never branch on producer origin.
"""
import json
import os
from dataclasses import dataclass

from . import _barcode_qc_contract as m, scatac_qc_reference as qr
from . import scatac_fragments_v2 as v2
from .scatac_fragments_v2_verifier import take_snapshots, check_snapshots
from .scatac_qc_profile import fail
from . import neutral_qc_reference as neutral
from ._qc_bedtools import verify_qc_bedtools
from agent.tools._cancellation import cancellation_checkpoint


def backend_runtime():
    executable=os.environ.get('AGENT_QC_BEDTOOLS')
    if not executable: fail('QC_BEDTOOLS_UNAVAILABLE')
    pinned=verify_qc_bedtools(executable)
    return executable,dict(profile=pinned['profile'],executable_sha256=pinned['sha256'],
        runtime_sha256=m.digest(pinned),compression='gzip-mtime0-level6.v1')


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


def qualified_fragments(path, sha, *, fragments_authority=None):
    from .scatac_fragment_reader import qualified_fragments as qualify
    value = v2.load_fragments_manifest_v2(path, expected_sha256=sha)
    if (sum(e['n_fragment_records'] for e in value['libraries']) > m.MAX_RECORDS
            or sum(e['n_distinct_barcodes'] for e in value['libraries']) > m.MAX_BARCODES):
        fail('QC_RESOURCE_LIMIT')
    return qualify(path, sha, fragments_authority=fragments_authority)


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


def bind(arguments, *, fragments_authority=None):
    args=m.validate_arguments(arguments);cancellation_checkpoint()
    snapshots=take_snapshots([args['fragments_manifest_path'],args['qc_reference_manifest_path']])
    executable,backend=backend_runtime()
    _,bundle,_=qr.load_scatac_qc_reference_bundle(args['qc_reference_manifest_path'],expected_sha256=args['qc_reference_manifest_sha256'])
    generic = type(bundle) is neutral.NeutralQCReference
    resources = neutral.resources(bundle) if generic else (bundle.parent_manifest,bundle.annotation.resource,bundle.tss,bundle.lineage)
    snapshots=tuple(sorted(snapshots+take_snapshots([r.path for r in resources])))
    qualification=resource_qualification(bundle)
    if generic:
        neutral.verify_integrity(bundle)
        if qualification['mode'] == 'synthetic_only':
            neutral.verify_normalized(bundle)
    elif qualification['mode'] != 'library_qualified':
        # Installed resources were reconstructed at installation and their
        # complete pinned closure was just checked. Legacy standalone catalogs
        # do not carry that qualification closure.
        qr.reinspect_scatac_qc_reference_bundle_sources(bundle)
    view,producer=qualified_fragments(args['fragments_manifest_path'],args['fragments_manifest_sha256'],
                                    fragments_authority=fragments_authority)
    reference=view.manifest['reference']
    if (reference['reference_identity_sha256']!=bundle.parent_reference_identity_sha256
        or reference['manifest_sha256']!=bundle.parent_manifest.sha256
        or tuple(view.contigs)!=tuple((c.name,c.length) for c in bundle.contigs)): fail('QC_REFERENCE_MISMATCH')
    if generic:
        from dataclasses import asdict
        if reference['species'] != asdict(bundle.species) or reference['assembly'] != bundle.assembly:
            fail('QC_REFERENCE_MISMATCH')
        neutral.check_fragment_scope(bundle, view.manifest)
    if qualification['mode']=='synthetic_only' and sum(x.n_fragment_records for x in view.libraries)>10000: fail('QC_RESOURCE_LIMIT')
    bound=BoundQC(view,bundle,producer,qualification,executable,backend,snapshots,fragments_authority)
    bound.unchanged();cancellation_checkpoint()
    return bound
