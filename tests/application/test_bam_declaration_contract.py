"""BAM-specific scientific declarations use the shared task machinery."""
from dataclasses import asdict, replace
from pathlib import Path
import pytest
from agent.application.raw_handoff import BamDeclaration, FastqDeclaration, declaration, merge, schema
from agent.application.turn_decisions import RequestRoles, RoleAnswer
import sys
sys.path.insert(0, str(Path(__file__).parents[1]))

@pytest.mark.parametrize('changes', [dict(source='source.16'), dict(source='/file'), dict(assay='RNA'),
    dict(species='human', assembly='mm10'), dict(barcode_policy='raw_CB'),
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
    with pytest.raises(ValueError): RequestRoles('run_chromap', (), bam=initial)
    with pytest.raises(ValueError): RoleAnswer('@roles', (), fastq=FastqDeclaration(), bam=initial)








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
