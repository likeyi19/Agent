"""Read-only exact dependencies of accepted historical neutral BAM inputs."""
from . import bam_fragment_manifest as m, primary_contigs as scope
from . import regulatory_matrix_contract as ref, scatac_fragments_v2 as v2

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
