"""Neutral dependency binding for the shared matrix engines; no counting code."""
from dataclasses import dataclass
from pathlib import Path

from . import explicit_cells as cells, regulatory_matrix_contract as reference
from . import fragment_feature_matrix_contract as contract, scatac_matrix_contract as m
from .scatac_fragment_reader import open_verified_fragments
from .scatac_fragments_v2_verifier import FragmentVerificationRuntime, take_snapshots, check_snapshots


@dataclass
class BoundFeatures:
    fragments: object
    selection: dict  # mechanical row-count/order interface, no QC selection claim
    reference: object
    upstream: dict
    snapshots: tuple
    contract: object = contract

    def unchanged(self):
        check_snapshots(self.snapshots)
        self.fragments.verification.check_unchanged()


def bind(upstream, limits):
    m.shape(upstream, ('fragments', 'cells', 'reference'))
    for pointer in upstream.values():
        m.absolute_path(pointer['manifest_path']); m.sha(pointer['manifest_sha256'])
    fp, cp, rp = (upstream[k] for k in ('fragments', 'cells', 'reference'))
    snapshots = take_snapshots(p['manifest_path'] for p in upstream.values())
    _, ref, _ = reference.load_reference(rp['manifest_path'], expected_sha256=rp['manifest_sha256'])
    declared = cells.load_manifest(cp['manifest_path'], cp['manifest_sha256'])
    if ref.ccre.feature_count > limits.max_features or declared['selected_count'] > limits.max_selected:
        m.fail('MATRIX_RESOURCE_LIMIT')
    snapshots += take_snapshots([ref.genome.fasta.path, ref.genome.fai.path, ref.ccre.bed.path,
                                Path(cp['manifest_path']).parent / 'cells.tsv.gz'])
    reference.reinspect_reference(ref)
    selected = cells.verify_explicit_cells(cp['manifest_path'], expected_sha256=cp['manifest_sha256'])
    from .external_fragments_verifier import verify_external_fragments
    from .external_fragment_manifest import load_adoption_record
    from ._external_fragment_io import resource
    from .scatac_fragments_v2 import load_fragments_manifest_v2
    manifest = load_fragments_manifest_v2(fp['manifest_path'], expected_sha256=fp['manifest_sha256'])
    kinds = {e['provenance']['kind'] for e in manifest['libraries']}
    if kinds == {'external_fragment_adoption'}:
        verified = verify_external_fragments(fp['manifest_path'], expected_sha256=fp['manifest_sha256'],
                                             runtime=FragmentVerificationRuntime())
        record_pointer = verified.fragments.manifest['libraries'][0]['provenance']['producer_record']
        record = load_adoption_record(record_pointer['path'], record_pointer['sha256'])
        if 'primary_preparation' in record:
            from .primary_fragment_preparation import verify_preparation, dependencies
            pointer = record['primary_preparation']
            prep = verify_preparation(pointer['path'], pointer['sha256'],
                reference_path=rp['manifest_path'], reference_sha256=rp['manifest_sha256'], deep=False)
            snapshots += take_snapshots([pointer['path'], *[r['path'] for r in dependencies(prep)]])
        for source in (record['source']['resource'], record['source_index']):
            if source is not None:
                snapshots += take_snapshots([source['path']])
                if resource(source['path']) != source: m.fail('MATRIX_LINEAGE_INVALID')
    elif kinds == {'bam_fragment_production'}:
        from .bam_fragments_verifier import verify_bam_fragments
        from .bam_fragment_manifest import NEUTRAL_PROFILE_ID, load_record
        from .neutral_bam import load, dependencies
        from .primary_contigs import verify_scope
        if any(e['provenance']['profile']['id'] != NEUTRAL_PROFILE_ID for e in manifest['libraries']):
            m.fail('MATRIX_REFERENCE_MISMATCH')
        verified = verify_bam_fragments(fp['manifest_path'], expected_sha256=fp['manifest_sha256'],
                                       runtime=FragmentVerificationRuntime())
        pointer = verified.fragments.manifest['libraries'][0]['provenance']['producer_record']
        record = load_record(pointer['path'], pointer['sha256'])
        pointer = record['input_binding']; spec = load(pointer['path'], pointer['sha256'])
        verify_scope(spec['primary_scope']['path'], spec['primary_scope']['sha256'],
            reference_path=rp['manifest_path'], reference_sha256=rp['manifest_sha256'])
        resources = dependencies(spec)
        snapshots += take_snapshots([pointer['path'], *[r['path'] for r in resources]])
        for source in resources:
            if resource(source['path']) != source: m.fail('MATRIX_LINEAGE_INVALID')
    else:
        m.fail('MATRIX_REFERENCE_MISMATCH')
    fragments = open_verified_fragments(fp['manifest_path'], expected_sha256=fp['manifest_sha256'],
                                        runtime=FragmentVerificationRuntime())
    fr = fragments.manifest['reference']
    if (type(fr['species']) is not dict or fr['species'] != ref.species
            or fr['assembly'] != ref.target_assembly):
        m.fail('MATRIX_REFERENCE_MISMATCH')
    _, source_ref, _ = reference.load_reference(fr['manifest_path'], expected_sha256=fr['manifest_sha256'])
    # Feature vocabularies may differ; the exact genome/assembly may not.
    def genome(r):
        return (r.genome.fasta.sha256, r.genome.fai.sha256,
                r.genome.ordered_contig_sha256, r.genome.contig_count)
    if genome(source_ref) != genome(ref): m.fail('MATRIX_REFERENCE_MISMATCH')
    namespaces = {library.namespace for library in fragments.libraries}
    for _, namespace, _, _ in cells.iter_rows(Path(cp['manifest_path']).parent / 'cells.tsv.gz'):
        if namespace not in namespaces: m.fail('MATRIX_NAMESPACE_MISMATCH')
    ids = (fragments.manifest['fragments_identity_sha256'], selected['identity_sha256'], ref.reference_identity_sha256)
    contracts = ('scatac-fragments.v2', cells.CONTRACT, reference.REFERENCE_CONTRACT)
    pointers = {k: dict(manifest_path=upstream[k]['manifest_path'], manifest_sha256=upstream[k]['manifest_sha256'],
                        identity_sha256=i, contract_version=c)
                for k, i, c in zip(('fragments', 'cells', 'reference'), ids, contracts)}
    result = BoundFeatures(fragments, selected, ref, pointers, tuple(sorted(set(snapshots))))
    result.unchanged()
    return result
