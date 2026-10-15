"""Typed raw declarations retain exact membership and owner validation."""
from dataclasses import asdict, replace
from types import SimpleNamespace
import json

import pytest

from agent.application.raw_handoff import FastqDeclaration, FastqMember, declaration, merge, requirements
from agent.application.turn_decisions import IntentError, parse_decision
from agent.application.local_resources import LocalResourceAdmission, ResourceAdmissionError
from agent.application.input_sets import ScientificInputSet
from agent.application.session_state import digest

LAYOUT = 'tenx-atac-r1-r2-r3.v1'


def members(*roles):
    return tuple(FastqMember(f'source.{i}', role, '001', '001') for i, role in enumerate(roles))


@pytest.mark.parametrize('change', [dict(layout='unresolved'), dict(layout='unsupported-role-set'),
    dict(barcode_policy='already_corrected'), dict(assay='SCATAC'), dict(resource='/tmp/context.json'),
    dict(namespace='a/b'), dict(namespace='x'*65), dict(species='guessed-human'),
    dict(members=members('R1','R1'))])
def test_closed_declarations(change):
    with pytest.raises(ValueError):
        FastqDeclaration(**change)


@pytest.mark.parametrize('field,value', [('layout', 'tenx-atac-r1-i2-r2.v1'), ('namespace', 'other'),
    ('species', 'mouse'), ('resource', 'resource.other')])
def test_captured_declarations_cannot_be_replaced(field, value):
    original = FastqDeclaration(layout=LAYOUT, namespace='sample', species='human', resource='resource.reference')
    with pytest.raises(ValueError, match='cannot be replaced'):
        merge(original, FastqDeclaration(**{field: value}))


def test_no_member_reassignment_or_duplicate_slot():
    original = FastqDeclaration(members=members('R1', 'R2'))
    for added in (FastqMember('source.0','R3','001','001'), FastqMember('source.2','R2','001','001')):
        with pytest.raises(ValueError):
            merge(original, FastqDeclaration(members=(added,)))


def test_owner_layout_completeness_and_forged_handle():
    binding = SimpleNamespace(members=({}, {}, {}))
    base = FastqDeclaration(layout=LAYOUT, namespace='sample', members=members('R1','R2'))
    missing, _ = requirements(binding, base, ())
    assert any('R3 (genomic-read-2)' in item for item in missing)
    missing, _ = requirements(binding, replace(base, layout=None), ())
    assert not any('R3' in item for item in missing)
    with pytest.raises(ResourceAdmissionError):
        requirements(binding, replace(base, members=members('R1','I2')), ())
    with pytest.raises(ValueError, match='outside captured'):
        requirements(binding, replace(base, members=(FastqMember('source.15','R3','001','001'),)), ())


def test_wire_remains_closed_and_non_executable():
    value = dict(kind='request_roles', target='prepare_scATAC_fragments', required_roles=[], roles=[],
        arguments=[], fastq=asdict(FastqDeclaration()))
    wire = lambda decision: json.dumps(dict(turn_schema_version=1, decision=decision))
    assert parse_decision(wire(value)).fastq == FastqDeclaration()
    for change in (dict(target='prepare_scATAC_bam_fragments'), dict(fastq=value['fastq'] | {'shell': 'run'}),
                   dict(required_roles=[dict(tool='prepare_scATAC_fragments',port='intake')])):
        with pytest.raises(IntentError):
            parse_decision(wire(value | change))


def test_legacy_inputset_choice_and_catalog_identity_unchanged():
    from agent.application.resource_continuation import catalog
    choice = ScientificInputSet('legacy','Legacy',dict(species='human'),fastq_companion=True)
    from agent.schemas.orchestration import _serialize
    old = {k: _serialize(getattr(choice,k)) for k in choice.__dataclass_fields__ if k != 'fastq_whitelist'}
    assert choice.resource_choice()['definition_sha256'] == digest(old)
    assert catalog((choice,)) == [old]


def test_fastq_variant_does_not_admit_unrelated_epizoo_declarations():
    from agent.application.turn_decisions import RequestRoles, ScopedArgument
    with pytest.raises(ValueError, match='registered scientific consumers'):
        RequestRoles('prepare_scATAC_fragments', (), fastq=FastqDeclaration(),
            arguments=(ScopedArgument('epizoo_embed_cells','species','human',0,5),))



def source_binding_fixture(tmp_path):
    owner = LocalResourceAdmission(tmp_path/'workspace', approved_source_roots=(tmp_path,))
    path = tmp_path/'source.attachment'
    path.write_bytes(b'@read\nACGT\n+\nIIII\n')
    source = owner.register('source', path, input_type='attachment', label='opaque',
        attribution='simulated', session_id='session')
    declared = dict(library_id='library', fastq_layout=LAYOUT, role='R1', lane='001', chunk='001')
    def bind(**changes):
        return owner.bind_raw_source('bound', source, input_type='fastq', declaration=declared | changes, session_id='session')
    return owner, path, source, bind


def test_roles_bind_original_bytes_without_source_materialization(tmp_path):
    owner, path, source, bind = source_binding_fixture(tmp_path)
    original = path.read_bytes(); inode = path.stat().st_ino; blocks = path.stat().st_blocks
    bound = bind()
    assert bound.source_path == source.source_path == str(path)
    assert bound.source_sha256 == source.source_sha256 and bound.session_id == source.session_id
    assert source.resource_id in bound.attribution and source.record_sha256 in bound.attribution
    assert owner.load(source.resource_id) == source and source.input_type == 'attachment'
    assert bind() == bound and path.stat().st_ino == inode and path.stat().st_blocks == blocks
    assert path.read_bytes() == original
    assert not list(owner._workspace.root.rglob('*.fastq*'))
    assert all(p.suffix in ('.json','.lock') for p in owner._workspace.root.rglob('*') if p.is_file())
    with pytest.raises(ResourceAdmissionError):
        bind(library_id='another')
    assert owner.load(bound.resource_id) == bound and path.read_bytes() == original


@pytest.mark.parametrize('case', ['foreign', 'forged', 'tampered'])
def test_original_binding_rejects_substitution(tmp_path, case):
    owner, path, source, bind = source_binding_fixture(tmp_path)
    declared = dict(library_id='library',fastq_layout=LAYOUT,role='R1',lane='001',chunk='001')
    if case == 'tampered':
        path.write_bytes(path.read_bytes().replace(b'ACGT', b'TGCA'))
    if case == 'forged':
        other = tmp_path/'other';other.write_bytes(path.read_bytes())
        source = replace(source, source_path=str(other))
    with pytest.raises(ResourceAdmissionError):
        owner.bind_raw_source('bad',source,input_type='fastq', declaration=declared,
            session_id='other' if case == 'foreign' else 'session')
    assert not owner._path(owner.registration_id('bad')).exists()


def test_failed_metadata_publication_leaves_original_intact_and_can_retry(tmp_path, monkeypatch):
    import os
    owner,path,source,bind = source_binding_fixture(tmp_path)
    data=path.read_bytes();link=os.link
    def interrupted(*args,**kwargs):
        raise OSError('simulated publication interruption')
    monkeypatch.setattr(os,'link',interrupted)
    with pytest.raises(ResourceAdmissionError):
        bind()
    assert path.read_bytes()==data and owner.load(source.resource_id)==source
    assert not owner._path(owner.registration_id('bound')).exists()
    assert not list(owner._workspace.root.rglob('*.tmp'))
    monkeypatch.setattr(os,'link',link)
    assert bind().source_path==str(path)


@pytest.mark.parametrize('layout,roles', [(LAYOUT, ('R1','R2','R3')),
    ('tenx-atac-r1-i2-r2.v1', ('R1','I2','R2'))])
@pytest.mark.parametrize('compressed', [False, True])
def test_original_sources_reach_owner_with_exact_bytes_for_both_layouts(tmp_path, layout, roles, compressed):
    import gzip
    from pathlib import Path
    from agent.tools.data.raw_scatac import inspect_raw_scATAC
    from agent.orchestration.raw_scatac_verifier import verify_raw_scatac_intake
    from agent.orchestration.registry import build_default_tool_registry
    owner = LocalResourceAdmission(tmp_path/'workspace', approved_source_roots=(tmp_path,),
        registry=build_default_tool_registry())
    records = []
    for role in roles:
        plain = b'@read\nACGTACGTACGTACGT\n+\nIIIIIIIIIIIIIIII\n'
        data = gzip.compress(plain, mtime=0) if compressed else plain
        path = tmp_path/(role+'.attachment'); path.write_bytes(data)
        source = owner.register(role, path, input_type='attachment', label='opaque', attribution='simulated', session_id='session')
        record = owner.bind_raw_source('typed-'+role, source, input_type='fastq',
            declaration=dict(library_id='library', fastq_layout=layout, role=role, lane='001', chunk='001'), session_id='session')
        assert record.fastq_member['role'] == role and record.fastq_member['fastq_layout'] == layout
        assert record.source_path == str(path) and path.read_bytes() == data
        assert owner.load(source.resource_id) == source
        records.append(record)
    arguments = dict(raw_input_paths=owner._fastq_sources(records), output_dir=str(tmp_path/'intake'), species='human',
        raw_assay='TENX_ATAC', fastq_layout=layout)
    result = inspect_raw_scATAC(**arguments)
    verify_raw_scatac_intake(arguments, result, owner._registry)
    assert result['readiness'] == 'READY' and result['n_files'] == 3 and result['n_groups'] == 1


def test_legacy_intake_cannot_substitute_filename_roles_for_explicit_task_roles(tmp_path):
    from agent.tools.data.raw_scatac import inspect_raw_scATAC
    owner = LocalResourceAdmission(tmp_path/'workspace', approved_source_roots=(tmp_path,))
    records=[]
    for role in ('R1','R2','R3'):
        path=tmp_path/f'Library_S1_L001_{role}_001.fastq'
        path.write_bytes(b'@read\nACGT\n+\nIIII\n')
        records.append(owner.register(role,path,input_type='fastq',label=role,attribution='simulated',
            fastq_member=dict(library_id='library',fastq_layout=LAYOUT,role=role,lane='001',chunk='001',compression='plain')))
    result=inspect_raw_scATAC([r.source_path for r in records],tmp_path/'intake',
        species='human',raw_assay='TENX_ATAC',fastq_layout=LAYOUT)
    values=dict(intake_manifest_path=result['manifest_path'],intake_manifest_sha256=result['manifest_sha256'])
    owner._validate_fastq_intake(records,values)
    swapped=[replace(r,fastq_member=dict(r.fastq_member,role={'R1':'R3','R3':'R1'}.get(r.fastq_member['role'],r.fastq_member['role']))) for r in records]
    owner._fastq_groups(swapped)  # Structurally complete, but contradicts this legacy intake.
    with pytest.raises(ResourceAdmissionError):
        owner._validate_fastq_intake(swapped,values)
