"""BAM-specific scientific declarations use the shared task machinery."""
from dataclasses import asdict, replace
from pathlib import Path
import pytest
from agent.application.raw_handoff import BamDeclaration, FastqDeclaration, declaration, merge, schema
from agent.application.turn_decisions import RequestRoles, RoleAnswer
from agent.tools.data import raw_scatac, raw_scatac_manifest, _raw_bam
from agent.orchestration.raw_scatac_verifier import verify_raw_scatac_intake
from agent.orchestration import build_default_tool_registry
import sys
sys.path.insert(0, str(Path(__file__).parents[1]))
from bam_fragments.conftest import bam_factory, pair

@pytest.mark.parametrize('changes', [dict(source='source.16'), dict(source='/file'), dict(assay='RNA'),
    dict(species='macaque'), dict(species='human', assembly='mm10'), dict(barcode_policy='raw_CB'),
    dict(source_profile='shifted'), dict(source_profile='duplicates_removed'), dict(resource='path'), dict(namespace='../sample')])
def test_invalid_bam_declarations(changes):
    with pytest.raises(ValueError): BamDeclaration(**changes)


def test_closed_bam_declaration_and_immutable_merge():
    initial = BamDeclaration(source='source.0', species='human', namespace='sample')
    final = merge(initial, BamDeclaration(assembly='hg38', barcode_policy='corrected_cb_already_corrected'))
    assert declaration(asdict(final), 'bam') == final
    for answer in (BamDeclaration(source='source.1'), BamDeclaration(species='mouse'), FastqDeclaration()):
        with pytest.raises(ValueError): merge(initial, answer)
    with pytest.raises(ValueError): declaration(asdict(final) | dict(proven=True), 'bam')
    with pytest.raises(ValueError): RequestRoles('prepare_scATAC_fragments', (), bam=initial)
    with pytest.raises(ValueError): RoleAnswer('@roles', (), fastq=FastqDeclaration(), bam=initial)


@pytest.mark.parametrize('species', ['human','mouse'])
def test_original_opaque_bam_uses_same_owner_and_verification(bam_factory, tmp_path, species):
    args = bam_factory(species=species)
    source = Path(args['source_path']); opaque = source.with_suffix('.attachment'); source.rename(opaque)
    rows = [dict(path=str(opaque), input_kind='BAM')]
    arguments = dict(raw_input_paths=rows, output_dir=str(tmp_path/'intake'), species=species,
                     raw_assay='SCATAC', source_genome_assembly='hg38' if species=='human' else 'mm10')
    result = raw_scatac.inspect_raw_scATAC(**arguments)
    verify_raw_scatac_intake(arguments, result, build_default_tool_registry())
    _, manifest, _ = raw_scatac_manifest.load_raw_intake_manifest(result['manifest_path'])
    assert manifest.files[0].path == str(opaque)
    assert _raw_bam.observe_bam_source(manifest.files[0]).header is not None
    legacy = _raw_bam.inspect_bam_inputs(rows, species=species, assay=_raw_bam.BamAssay.SCATAC,
                                       source_genome_assembly=arguments['source_genome_assembly'])
    assert raw_scatac_manifest.canonical_manifest_bytes(legacy) == raw_scatac_manifest.canonical_manifest_bytes(manifest)
    opaque.write_bytes(b'mutation')
    with pytest.raises(ValueError): verify_raw_scatac_intake(arguments, result, build_default_tool_registry())


@pytest.mark.parametrize('failure', ['mixed', 'unknown', 'extra', 'duplicate', 'directory','symlink', 'traversal'])
def test_closed_original_bam_source_shape(tmp_path, failure):
    source=tmp_path/'source.attachment';source.write_bytes(b'not yet qualified')
    row=dict(path=str(source),input_kind='BAM'); rows=[row]
    if failure=='mixed': rows.append(str(source))
    elif failure=='unknown': row['input_kind']='SAM'
    elif failure=='extra': row['authority']='yes'
    elif failure=='duplicate': rows.append(dict(row))
    elif failure=='directory': row['path']=str(tmp_path)
    elif failure=='symlink':
        link=tmp_path/'link';link.symlink_to(source);row['path']=str(link)
    else: row['path']=str(tmp_path)+'/sub/../source.attachment'
    with pytest.raises(ValueError): raw_scatac.inspect_raw_scATAC(rows,tmp_path/'intake')


def test_declared_bam_format_does_not_qualify_arbitrary_bytes(tmp_path):
    path=tmp_path/'opaque.attachment';path.write_bytes(b'not a BAM')
    result=raw_scatac.inspect_raw_scATAC([dict(path=str(path),input_kind='BAM')],tmp_path/'intake',
        species='human', raw_assay='SCATAC', source_genome_assembly='hg38')
    assert result['readiness'] != 'READY'


@pytest.mark.parametrize('case', ['retry','foreign','forged','mutation'])
def test_bam_metadata_binding_retains_original_integrity(tmp_path, case):
    from agent.application.local_resources import LocalResourceAdmission, ResourceAdmissionError
    owner=LocalResourceAdmission(tmp_path/'workspace',approved_source_roots=(tmp_path,))
    path=tmp_path/'original.attachment';path.write_bytes(b'storage-only fixture')
    original=owner.register('source',path,input_type='attachment',label='Source',attribution='Simulated',session_id='session')
    source=original
    if case=='forged': source=replace(source,source_sha256='0'*64)
    if case=='mutation': path.write_bytes(b'changed')
    def bind():return owner.bind_raw_source('bam',source,input_type='bam',session_id='other' if case=='foreign' else 'session')
    if case!='retry':
        with pytest.raises(ResourceAdmissionError):bind()
        assert not owner._path(owner.registration_id('bam')).exists()
    else:
        before=path.stat()
        typed=bind();assert bind()==typed and owner.load(original.resource_id)==original
        assert typed.source_path==original.source_path and typed.source_sha256==original.source_sha256
        assert (before.st_ino,before.st_blocks)==(path.stat().st_ino,path.stat().st_blocks)
        assert all(p.suffix in ('.json','.lock') for p in owner._workspace.root.rglob('*') if p.is_file())
