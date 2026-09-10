"""Fresh source authority and operator-owned runtime/resource qualification.

Producer dispatch establishes existing verification authority only. The metric
algorithms receive one common v2 reader and never branch on producer origin.
"""
import json
import os
from pathlib import Path
from dataclasses import dataclass

from . import _barcode_qc_contract as m, scatac_qc_reference as qr
from . import scatac_fragments_v2 as v2
from .scatac_fragment_reader import open_verified_fragments
from .scatac_fragments_v2_verifier import FragmentVerificationRuntime, take_snapshots, check_snapshots
from .scatac_qc_profile import fail
from ._qc_bedtools import verify_qc_bedtools
from agent.tools._cancellation import cancellation_checkpoint


def packaging_runtime(): return FragmentVerificationRuntime()


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


def qualified_fragments(path,sha):
    # The declared producer kind selects its reviewed verifier, never a QC algorithm.
    value=v2.load_fragments_manifest_v2(path,expected_sha256=sha)
    if (sum(e['n_fragment_records'] for e in value['libraries'])>m.MAX_RECORDS
        or sum(e['n_distinct_barcodes'] for e in value['libraries'])>m.MAX_BARCODES): fail('QC_RESOURCE_LIMIT')
    kinds={e['provenance']['kind'] for e in value['libraries']}
    if len(kinds)!=1: fail('QC_PRODUCER_UNQUALIFIED')
    kind=next(iter(kinds)); runtime=packaging_runtime()
    if kind=='fastq_fragment_production':
        from .fastq_fragments_verifier import verify_fragments
        from .scatac_fragments import verification_runtime
        verify_fragments(path,expected_sha256=sha,runtime=verification_runtime())
        basis='qualified_fastq_source_and_producer_record';history='alignment_not_rerun'
    elif kind=='bam_fragment_production':
        from .bam_fragments_verifier import verify_bam_fragments
        verify_bam_fragments(path,expected_sha256=sha,runtime=runtime)
        basis='independent_bam_transformation';history='upstream_history_declared'
    elif kind=='external_fragment_adoption':
        from .external_fragments_verifier import verify_external_fragments
        verify_external_fragments(path,expected_sha256=sha,runtime=runtime)
        basis='independent_external_conservation';history='upstream_history_declared'
    else: fail('QC_PRODUCER_UNQUALIFIED')
    view=open_verified_fragments(path,expected_sha256=sha,runtime=runtime)
    if view.manifest!=value: fail('QC_SOURCE_CHANGED')
    return view,dict(kind=kind,profile_ids=sorted({e['provenance']['profile']['id'] for e in value['libraries']}),
        verification_basis=basis,history=history)


@dataclass(frozen=True)
class BoundQC:
    fragments: object
    reference: qr.ScATACQCReferenceBundle
    producer_authority: dict
    resource_qualification: dict
    executable: str
    backend_identity: dict
    snapshots: tuple

    def unchanged(self):
        self.fragments.verification.check_unchanged();check_snapshots(self.snapshots)
        if resource_qualification(self.reference)!=self.resource_qualification: fail('QC_RESOURCE_UNQUALIFIED')


def bind(arguments):
    args=m.validate_arguments(arguments);cancellation_checkpoint()
    snapshots=take_snapshots([args['fragments_manifest_path'],args['qc_reference_manifest_path']])
    executable,backend=backend_runtime()
    _,bundle,_=qr.load_scatac_qc_reference_bundle(args['qc_reference_manifest_path'],expected_sha256=args['qc_reference_manifest_sha256'])
    snapshots=tuple(sorted(snapshots+take_snapshots([r.path for r in (bundle.parent_manifest,bundle.annotation.resource,bundle.tss,bundle.lineage)])))
    qualification=resource_qualification(bundle)
    qr.reinspect_scatac_qc_reference_bundle_sources(bundle)
    view,producer=qualified_fragments(args['fragments_manifest_path'],args['fragments_manifest_sha256'])
    reference=view.manifest['reference']
    if (reference['reference_identity_sha256']!=bundle.parent_reference_identity_sha256
        or reference['manifest_sha256']!=bundle.parent_manifest.sha256
        or tuple(view.contigs)!=tuple((c.name,c.length) for c in bundle.contigs)): fail('QC_REFERENCE_MISMATCH')
    if qualification['mode']=='synthetic_only' and sum(x.n_fragment_records for x in view.libraries)>10000: fail('QC_RESOURCE_LIMIT')
    bound=BoundQC(view,bundle,producer,qualification,executable,backend,snapshots)
    bound.unchanged();cancellation_checkpoint()
    return bound
