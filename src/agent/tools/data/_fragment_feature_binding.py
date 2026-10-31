"""Neutral dependency binding for the shared matrix engines; no counting code."""
from dataclasses import dataclass
from pathlib import Path
import json

from . import explicit_cells as cells, regulatory_matrix_contract as reference
from . import fragment_feature_matrix_contract as contract, scatac_matrix_contract as m
from .scatac_fragments_v2_verifier import take_snapshots, check_snapshots


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


def cell_input(pointer):
    """Dispatch only explicit closed artifact contracts, never filenames or order."""
    from . import _cell_selection_contract as selection
    from . import selected_feature_matrix_contract as selected_contract
    from .scatac_fragments_v2 import _pairs
    with open(pointer['manifest_path'], 'rb') as stream:
        raw = stream.read(selection.qc.MAX_MANIFEST + 1)
    if len(raw) > selection.qc.MAX_MANIFEST: m.fail('MATRIX_RESOURCE_LIMIT')
    value = json.loads(raw, object_pairs_hook=_pairs)
    if value.get('contract_version') == cells.CONTRACT:
        return cells.load_manifest(pointer['manifest_path'], pointer['manifest_sha256']), contract
    if value.get('contract_version') == selection.CONTRACT:
        return selection.load_manifest(pointer['manifest_path'], pointer['manifest_sha256']).to_dict(), selected_contract
    m.fail('MATRIX_LINEAGE_INVALID')


def bind(upstream, limits):
    m.shape(upstream, ('fragments', 'cells', 'reference'))
    for pointer in upstream.values():
        m.absolute_path(pointer['manifest_path']); m.sha(pointer['manifest_sha256'])
    fp, cp, rp = (upstream[k] for k in ('fragments', 'cells', 'reference'))
    snapshots = take_snapshots(p['manifest_path'] for p in upstream.values())
    _, ref, _ = reference.load_reference(rp['manifest_path'], expected_sha256=rp['manifest_sha256'])
    declared, selected_contract = cell_input(cp)
    explicit = selected_contract is contract
    if ref.ccre.feature_count > limits.max_features or declared['selected_count'] > limits.max_selected:
        m.fail('MATRIX_RESOURCE_LIMIT')
    snapshots += take_snapshots([ref.genome.fasta.path, ref.genome.fai.path, ref.ccre.bed.path,
                                Path(cp['manifest_path']).parent / ('cells.tsv.gz' if explicit else 'selected.tsv.gz')])
    reference.reinspect_reference(ref)
    if explicit:
        selected = cells.verify_explicit_cells(cp['manifest_path'], expected_sha256=cp['manifest_sha256'])
    else:
        from .cell_selection_verifier import verify_cell_selection
        from ._barcode_qc_contract import load_manifest as load_qc
        from .scatac_qc_reference import load_scatac_qc_reference_bundle
        from .neutral_qc_reference import NeutralQCReference, resources as qc_resources
        selected = verify_cell_selection(cp['manifest_path'], expected_sha256=cp['manifest_sha256']).to_dict()
        args = selected['qc_lineage']['arguments']
        if (args['fragments_manifest_path'] != fp['manifest_path']
                or args['fragments_manifest_sha256'] != fp['manifest_sha256']): m.fail('MATRIX_LINEAGE_INVALID')
        _, qc_reference, _ = load_scatac_qc_reference_bundle(args['qc_reference_manifest_path'],
            expected_sha256=args['qc_reference_manifest_sha256'])
        qa = selected['arguments']; qp = Path(qa['barcode_qc_manifest_path'])
        qvalue = load_qc(qp, qa['barcode_qc_manifest_sha256']).to_dict()
        resources = (qc_resources(qc_reference) if type(qc_reference) is NeutralQCReference else
                     (qc_reference.parent_manifest, qc_reference.annotation.resource, qc_reference.tss, qc_reference.lineage))
        snapshots += take_snapshots([qp, args['qc_reference_manifest_path'],
            *[r.path for r in resources],
            *[qp.parent / qvalue[k]['path'] for k in ('table', 'histogram')],
            Path(cp['manifest_path']).parent / 'decisions.tsv.gz'])
    # The producer owns source history and original scope; downstream features
    # need exact genome compatibility, not the producer's feature vocabulary.
    from .scatac_fragment_reader import qualified_fragments
    fragments, producer = qualified_fragments(fp['manifest_path'], fp['manifest_sha256'])
    # A new matrix binds current sources, even when the producer science was
    # already verified in this operation. Accepted matrix reads use their proof.
    from .authority_context import current, VerificationContext
    from .fastq_verification_authority import file_closure
    context = current() or VerificationContext()
    description, _ = context.describe(producer['kind'], Path(fp['manifest_path']), fp['manifest_sha256'])
    sources = description['historical_sources']
    if file_closure(r['path'] for r in sources) != sources:
        m.fail('MATRIX_LINEAGE_INVALID')
    snapshots += take_snapshots(r['path'] for r in sources)
    fr = fragments.manifest['reference']
    from .scatac_reference import species_identity
    species = fr['species'] if type(fr['species']) is dict else species_identity(fr['species'])
    if (species != ref.species or fr['assembly'] != ref.target_assembly):
        m.fail('MATRIX_REFERENCE_MISMATCH')
    from ._fragment_reference import load
    source_ref = load(fr['manifest_path'], fr['manifest_sha256'])
    # Feature vocabularies may differ; the exact genome/assembly may not.
    def genome(r):
        return (r.genome.fasta.sha256, r.genome.fai.sha256,
                r.genome.ordered_contig_sha256, r.genome.contig_count)
    if genome(source_ref) != genome(ref): m.fail('MATRIX_REFERENCE_MISMATCH')
    namespaces = {library.namespace for library in fragments.libraries}
    for _, namespace, _, _ in cells.iter_rows(Path(cp['manifest_path']).parent / ('cells.tsv.gz' if explicit else 'selected.tsv.gz')):
        if namespace not in namespaces: m.fail('MATRIX_NAMESPACE_MISMATCH')
    ids = (fragments.manifest['fragments_identity_sha256'], selected['identity_sha256'], ref.reference_identity_sha256)
    contracts = ('scatac-fragments.v2', selected_contract.CELL_CONTRACT, reference.REFERENCE_CONTRACT)
    pointers = {k: dict(manifest_path=upstream[k]['manifest_path'], manifest_sha256=upstream[k]['manifest_sha256'],
                        identity_sha256=i, contract_version=c)
                for k, i, c in zip(('fragments', 'cells', 'reference'), ids, contracts)}
    result = BoundFeatures(fragments, selected, ref, pointers, tuple(sorted(set(snapshots))),
                           selected_contract)
    result.unchanged()
    return result
