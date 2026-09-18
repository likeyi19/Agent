import hashlib
import json
from pathlib import Path
import pytest

from agent.tools.data import neutral_fragment_import as adoption, scatac_fragments_v2 as v2
from agent.tools.data import fragment_feature_matrix as matrix
from agent.tools.data import scientific_authority
from agent.tools.data.scatac_fragments_v2_verifier import FragmentVerificationRuntime
from agent.tools.data.external_fragments_verifier import verify_external_fragments
from agent.tools.data.authority_context import VerificationContext, authority_operation


@pytest.mark.parametrize('encoding,indexed', [('plain',False), ('gzip',False), ('bgzf',False), ('bgzf',True)])
@pytest.mark.parametrize('strand', [False,True])
def test_existing_source_formats(factory, encoding, indexed, strand):
    args, _, source_args = factory(encoding=encoding,indexed=indexed,strand=strand)
    verified = verify_external_fragments(args['fragments_manifest_path'],
        expected_sha256=args['fragments_manifest_sha256'], runtime=FragmentVerificationRuntime())
    value = verified.fragments.manifest
    assert value['reference']['contract_version'] == 'regulatory-feature-reference.v1'
    assert value['reference']['species'] == {'scientific_name':'Danio rerio','taxonomy_id':7955}
    assert value['libraries'][0]['sum_support'] == 8000
    assert value['libraries'][0]['n_fragment_records'] == 8
    assert value['libraries'][0]['strand']['mode'] == ('present' if strand else 'absent')
    assert adoption.recover_external_fragments(source_args, None)['manifest_sha256'] == args['fragments_manifest_sha256']


@pytest.mark.parametrize('source', ['source_path','source_index_path'])
def test_source_index_mutation_invalidates_reuse(factory, source):
    args, _, source_args = factory(encoding='bgzf', indexed=True)
    with authority_operation(VerificationContext()):
        matrix.build_cell_by_features(**args)
        path = Path(source_args[source]); path.write_bytes(path.read_bytes()+b'changed')
        with pytest.raises(ValueError): matrix.recover_matrix(args,None)


def test_legacy_registered_boundary_stays_legacy(factory):
    _, _, args = factory()
    args['output_dir'] += '-legacy'
    from agent.tools.data.scatac_fragment_import import import_scATAC_fragments
    with pytest.raises(ValueError): import_scATAC_fragments(**args)
    assert not Path(args['output_dir']).exists()


@pytest.mark.parametrize('kind', ['fastq_fragment_production','bam_fragment_production'])
def test_neutral_reference_cannot_qualify_legacy_producer(factory, kind):
    args, _, _ = factory()
    value = json.loads(Path(args['fragments_manifest_path']).read_bytes())
    value['libraries'][0]['provenance']['kind'] = kind
    value['fragments_identity_sha256'] = v2.fragments_identity(value)
    with pytest.raises(ValueError): v2.validate_fragments_manifest_v2(value)


def test_wrong_source_contig_and_duplicates_rejected(factory):
    _, _, args = factory()
    source = Path(args['source_path']); args['output_dir'] += '-bad'
    for raw in (b'unknown\t0\t1\tA\t1\n', b'chr1\t0\t1\tA\t1\n'*2,
                b'chr1\t0\t101\tA\t1\n'):
        source.write_bytes(raw); args['source_sha256'] = hashlib.sha256(raw).hexdigest()
        with pytest.raises(ValueError): adoption.import_external_fragments(**args)


def test_rehashed_wrong_genome_binding_fails(factory):
    args, _, _ = factory()
    value = json.loads(Path(args['fragments_manifest_path']).read_bytes())
    value['reference']['assembly'] = 'other-assembly'
    value['fragments_identity_sha256'] = v2.fragments_identity(value)
    path = Path(args['fragments_manifest_path']); path.write_bytes(v2.canonical_fragments_manifest_v2_bytes(value))
    with pytest.raises(ValueError):
        verify_external_fragments(path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), runtime=FragmentVerificationRuntime())


def test_neutral_fragment_owner_and_preexisting_source_proof(factory):
    _, _, source_args = factory()
    source_args['output_dir'] += '-owned'
    context = VerificationContext(); execution = 'e'*64
    context.register_execution('external_fragment_adoption', source_args['output_dir'], execution)
    with authority_operation(context):
        result = adoption.import_external_fragments(**source_args, execution_identity=execution)
        authority = scientific_authority.issue(context, 'import_external_fragments', source_args, result, execution)
        assert authority.record['schema_version'] == 2
        assert authority.record['producer_qualification']['kind'] == 'external_fragment_adoption'
        assert adoption.recover_external_fragments(source_args, execution) == result


def test_mutated_source_cannot_enter_matrix_from_cached_producer(factory):
    with authority_operation(VerificationContext()):
        args, _, source_args = factory()
        source = Path(source_args['source_path']); source.write_bytes(source.read_bytes()+b'changed')
        with pytest.raises(ValueError): matrix.build_cell_by_features(**args)
