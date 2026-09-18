"""Explicit neutral BAM/reference binding for the existing paired-ATAC owner.

No alignment parser, pairing, transformation or aggregation lives here. This
versioned data-layer declaration replaces legacy M10 target selection, not its
public intake contract. The shared producer still fully validates every record.
"""
from pathlib import Path
from types import SimpleNamespace

from . import bam_fragment_manifest as m, primary_contigs as scope
from . import regulatory_matrix_contract as ref, scatac_fragments_v2 as v2
from . import _bam_fragment_io as io, _external_fragment_io as physical

CONTRACT = 'neutral-bam-inputs.v1'
LIBRARY = dict(barcode_tag='CB', barcode_interpretation='corrected_identifier',
               correction_policy='already_corrected', assay='paired_atac')


def load(path, sha):
    value = scope.read(path, sha)
    v2.shape(value, ('contract_version', 'source', 'source_index', 'source_species',
        'source_assembly', 'source_reference', 'primary_scope', 'library', 'source_history'))
    if value['contract_version'] != CONTRACT or value['source_history'] != m.HISTORY: m.fail()
    for key in ('source', 'source_reference', 'primary_scope'): v2.resource(value[key])
    if value['source_index'] is not None: v2.resource(value['source_index'])
    ref.validate_binding(value['source_species'], value['source_assembly'])
    library = value['library']; v2.shape(library, (*LIBRARY, 'namespace', 'source_library_id'))
    if any(library[k] != v for k,v in LIBRARY.items()): m.fail('BAM_FRAGMENTS_BARCODE_POLICY_UNSUPPORTED')
    v2.token(library['namespace'])
    if library['source_library_id'] is not None: v2.text(library['source_library_id'])
    return value


def dependencies(value):
    """Complete current binding closure, shared by owner and authority description."""
    primary = scope.verify_scope(value['primary_scope']['path'], value['primary_scope']['sha256'],
        reference_path=value['source_reference']['path'], reference_sha256=value['source_reference']['sha256'])
    return [value[k] for k in ('source', 'source_reference', 'primary_scope')]+[
        primary['reference'], primary['classification_source']]+([value['source_index']] if value['source_index'] else [])


def bind(arguments):
    a = m.validate_arguments(arguments)
    value = load(a['input_spec_path'], a['input_spec_sha256'])
    pointer = value['source_reference']
    reference, contigs, paths = physical.reference(pointer['path'], pointer['sha256'])
    _, bundle, _ = ref.load_reference(pointer['path'], expected_sha256=pointer['sha256'])
    if value['source_species'] != reference['species'] or value['source_assembly'] != reference['assembly']:
        m.fail('BAM_FRAGMENTS_REFERENCE_MISMATCH')
    resources = dependencies(value)
    snapshots = io.take_snapshots([a['input_spec_path'], *paths, *[r['path'] for r in resources]])
    for resource in resources:
        if io.resource(resource['path'], resource['sha256']) != resource: m.fail('BAM_FRAGMENTS_SOURCE_CHANGED')
    primary = scope.verify_scope(value['primary_scope']['path'], value['primary_scope']['sha256'],
        reference_path=pointer['path'], reference_sha256=pointer['sha256'])
    if value['source_index'] is not None:
        check_index(value['source']['path'], value['source_index']['path'])
    io.check_snapshots(snapshots)
    return dict(reference=reference, contigs=contigs, bundle=bundle,
        library=SimpleNamespace(source_library_id=value['library']['source_library_id']),
        namespace=value['library']['namespace'], source=value['source'],
        input_binding=io.resource(a['input_spec_path'], a['input_spec_sha256']),
        primary_contigs=frozenset(c['name'] for c in primary['contigs'] if c['classification']=='primary_nuclear'),
        snapshots=snapshots)


def check_header_declarations(header, binding):
    """Exact optional SQ declarations; never normalize aliases or infer assembly."""
    reference = binding['reference']
    for entry in header.get('SQ', []):
        if ('AS' in entry and entry['AS'] != reference['assembly']) or (
                'SP' in entry and entry['SP'] != reference['species']['scientific_name']):
            m.fail('BAM_FRAGMENTS_REFERENCE_MISMATCH')


def check_index(source, index):
    """Optional exact index binding, checked against a full sequential decode.

    No index is needed by transformation. When explicitly supplied, every
    reference's indexed record stream must equal its sequential record stream.
    Adjacent undisclosed indexes are not execution inputs and are never opened.
    """
    import hashlib
    ps = io._raw_bam._backend()
    def add(digest, record):
        data = record.to_string().encode('utf-8')
        digest.update(len(data).to_bytes(8, 'big')); digest.update(data)
    try:
        # A file object suppresses htslib's implicit adjacent-index discovery.
        with Path(source).open('rb') as stream, ps.AlignmentFile(stream, 'rb') as bam:
            if len(bam.references)>4096: m.fail('BAM_FRAGMENTS_HEADER_INVALID')
            names = bam.references; expected = [hashlib.sha256() for _ in names]
            for record in bam.fetch(until_eof=True):
                if record.reference_id >= 0: add(expected[record.reference_id], record)
        with ps.AlignmentFile(source, 'rb', index_filename=index, require_index=True) as bam:
            if not bam.check_index(): m.fail('BAM_FRAGMENTS_INDEX_MISMATCH')
            for name, digest in zip(names, expected, strict=True):
                actual = hashlib.sha256()
                for record in bam.fetch(name): add(actual, record)
                if actual.digest() != digest.digest(): m.fail('BAM_FRAGMENTS_INDEX_MISMATCH')
    except (OSError, ValueError, IndexError):
        m.fail('BAM_FRAGMENTS_INDEX_MISMATCH')


def prepare_neutral_bam_fragments(*, input_spec_path, input_spec_sha256, output_dir, execution_identity=None):
    """Data-layer entry point; shared lifecycle and operation-local owner proof."""
    from . import scatac_bam_fragments as owner
    from .authority_context import current, authority_operation, VerificationContext
    args = dict(input_spec_path=str(input_spec_path), input_spec_sha256=input_spec_sha256,
                output_dir=str(output_dir), source_profile=m.NEUTRAL_PROFILE_ID)
    if current() is not None: return owner.execute_bam_fragments(args, execution_identity)
    with authority_operation(VerificationContext()):
        return owner.execute_bam_fragments(args, execution_identity)


def recover_neutral_bam_fragments(arguments, execution_identity=None):
    from . import scatac_bam_fragments as owner
    return owner.recover_bam_fragments(arguments, execution_identity)
