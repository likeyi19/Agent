"""Exact FASTQ admission; existing intake and library owners retain science."""
from dataclasses import FrozenInstanceError, replace
import hashlib
import itertools
import json
import os
from pathlib import Path

import pytest

from agent.application import local_resources as resources
from agent.application.local_resources import (
    LocalResourceAdmission, RegisteredInput, RegisteredInputCollection,
    ResourceAdmissionError,
)
from agent.application.session_state import canonical, digest
from agent.orchestration.registry import ToolRegistry, build_default_tool_registry
from agent.schemas.orchestration import _serialize
from agent.tools.data import _raw_fastq, raw_scatac_manifest, scatac_library_context
from agent.tools.data.raw_scatac import inspect_raw_scATAC


def assert_code(code, call):
    with pytest.raises(ResourceAdmissionError) as failed:
        call()
    assert failed.value.code == failed.value.error.code == code
    assert failed.value.message == failed.value.error.message
    assert '/' not in failed.value.message


def register(owner, source, key=None, **changes):
    return owner.register(key or source.name, source,
        **(dict(input_type='fastq', label='Supplied reads',
                attribution='Explicit operator selection; suitability unverified') | changes))


def references(records):
    return [dict(resource_id=record.resource_id, record_sha256=record.record_sha256)
            for record in records]


def submission(binding):
    return dict(registered_input=binding.attribution(),
                execution_inputs=_serialize(binding.execution_inputs))


@pytest.fixture
def fastqs(tmp_path):
    approved = tmp_path / 'approved'
    approved.mkdir()
    owner = LocalResourceAdmission(tmp_path / 'workspace', approved_source_roots=(approved,))
    records = []
    for role in ('R1', 'R2', 'R3'):
        path = approved / f'sample_S1_L001_{role}_001.fastq'
        path.write_bytes(b'deliberately invalid sequencing reads: ' + role.encode())
        records.append(register(owner, path))
    return owner, records


@pytest.mark.parametrize('suffix', (*_raw_fastq.SUFFIXES, '.declared', ''))
def test_registration_hashes_declared_sources_without_parsing(tmp_path, monkeypatch, suffix):
    source = tmp_path / ('reads' + suffix)
    payload = b'\xff\x00not FASTQ or gzip\n'
    source.write_bytes(payload)
    owner = LocalResourceAdmission(tmp_path / 'workspace', approved_source_roots=(tmp_path,))
    def forbidden(*args, **kwargs):
        pytest.fail('Registration entered the Registry or FASTQ scientific owner.')
    import agent.orchestration.registry as registry
    import agent.tools.data.raw_scatac as intake
    import agent.tools.data.fastq_fragments as producer
    monkeypatch.setattr(_raw_fastq, 'inspect_fastq_inputs', forbidden)
    monkeypatch.setattr(intake, 'inspect_raw_scATAC', forbidden)
    monkeypatch.setattr(producer, 'prepare_fastq_fragments', forbidden)
    monkeypatch.setattr(registry, 'build_default_tool_registry', forbidden)
    record = register(owner, source)
    assert record.source_path == str(source)
    assert record.source_sha256 == hashlib.sha256(payload).hexdigest()
    assert record.size_bytes == len(payload)
    assert record.public() == dict(resource_id=record.resource_id, input_type='fastq',
                                  label='Supplied reads', status='registered')
    assert str(source) not in json.dumps(record.public())
    assert not (owner._workspace.root / 'sessions').exists()
    assert not list((owner._workspace.root / 'runs').iterdir())
    stored = owner._path(record.resource_id).read_bytes()
    assert register(owner, source) == record
    reopened = LocalResourceAdmission(owner._workspace.root)
    assert reopened.load(record.resource_id) == record
    assert reopened._path(record.resource_id).read_bytes() == stored


@pytest.mark.parametrize('kind', ['outside', 'missing', 'directory', 'symlink', 'ancestor_symlink', 'url'])
def test_fastq_registration_preserves_canonical_approved_source_policy(fastqs, tmp_path, kind):
    owner, records = fastqs
    source = Path(records[0].source_path)
    if kind == 'outside':
        supplied = tmp_path / 'outside.fastq'
        supplied.write_bytes(b'bytes')
    elif kind == 'missing':
        supplied = source.with_name('missing.fastq')
    elif kind == 'directory':
        supplied = source.parent
    elif kind == 'symlink':
        supplied = source.with_name('alias.fastq')
        supplied.symlink_to(source)
    elif kind == 'ancestor_symlink':
        alias = tmp_path / 'alias'
        alias.symlink_to(source.parent, target_is_directory=True)
        supplied = alias / source.name
    else:
        supplied = 'https://example.invalid/source.fastq'
    original = {record.resource_id: owner._path(record.resource_id).read_bytes() for record in records}
    assert_code('LOCAL_RESOURCE_ACCESS_INVALID', lambda: register(owner, supplied, 'new-key'))
    assert original == {record.resource_id: owner._path(record.resource_id).read_bytes() for record in records}


def test_collection_permutations_have_identical_immutable_attribution(fastqs):
    owner, records = fastqs
    supplied = references(records)
    original = {record.resource_id: owner._path(record.resource_id).read_bytes() for record in records}
    bindings = [owner.resolve_collection(list(order), tool_name='inspect_raw_scATAC')
                for order in itertools.permutations(supplied)]
    binding = bindings[0]
    expected_members = sorted(references(records), key=lambda member: member['resource_id'])
    expected_inputs = dict(raw_input_paths=sorted(record.source_path for record in records))
    assert all(candidate == binding for candidate in bindings)
    assert _serialize(binding.members) == expected_members
    assert owner.validate_binding(binding) == expected_inputs
    assert binding.attribution() == dict(members=expected_members,
        collection_sha256=binding.collection_sha256, tool_name='inspect_raw_scATAC')
    assert binding.collection_sha256 == digest(dict(format='agent.local-resource-collection.v1',
                                                   members=expected_members))
    assert original == {record.resource_id: owner._path(record.resource_id).read_bytes() for record in records}
    assert len(list((owner._workspace.root / 'local_resources').glob('*.json'))) == len(records)
    supplied[0]['record_sha256'] = '0' * 64
    assert _serialize(binding.members) == expected_members
    with pytest.raises(FrozenInstanceError):
        binding.collection_sha256 = '0' * 64
    with pytest.raises(TypeError):
        binding.members[0]['record_sha256'] = '0' * 64
    with pytest.raises(TypeError):
        binding.execution_inputs['raw_input_paths'][0] = '/unregistered.fastq'
    captured = binding.attribution()
    captured['members'].clear()
    assert _serialize(binding.members) == expected_members


def test_collection_identity_changes_with_membership_but_not_declarations(fastqs):
    owner, records = fastqs
    first = owner.resolve_collection(references(records), tool_name='inspect_raw_scATAC')
    declared = owner.resolve_collection(references(records), tool_name='inspect_raw_scATAC',
        scientific_inputs=dict(species='human', raw_assay='TENX_ATAC',
                               fastq_layout='tenx-atac-r1-r2-r3.v1'))
    subset = owner.resolve_collection(references(records[:2]), tool_name='inspect_raw_scATAC')
    assert first.collection_sha256 == declared.collection_sha256
    assert first.execution_inputs != declared.execution_inputs
    assert subset.collection_sha256 != first.collection_sha256
    changed = register(owner, Path(records[0].source_path), 'another-registration-key')
    other_identity = owner.resolve_collection(references([changed, *records[1:]]),
                                             tool_name='inspect_raw_scATAC')
    assert other_identity.collection_sha256 != first.collection_sha256


@pytest.mark.parametrize('kind', ['empty', 'scalar_id', 'none', 'string_member', 'record',
    'missing_digest', 'extra_field', 'bad_id', 'bad_digest', 'boolean_digest', 'duplicate', 'conflicting'])
def test_membership_requires_closed_exact_nonduplicated_references(fastqs, kind):
    owner, records = fastqs
    members = references(records)
    if kind == 'empty':
        members = []
    elif kind == 'scalar_id':
        members = records[0].resource_id
    elif kind == 'none':
        members = None
    elif kind == 'string_member':
        members[0] = records[0].resource_id
    elif kind == 'record':
        members[0] = records[0]
    elif kind == 'missing_digest':
        members[0].pop('record_sha256')
    elif kind == 'extra_field':
        members[0]['source_path'] = records[0].source_path
    elif kind == 'bad_id':
        members[0]['resource_id'] = 'invalid'
    elif kind == 'bad_digest':
        members[0]['record_sha256'] = 'wrong'
    elif kind == 'boolean_digest':
        members[0]['record_sha256'] = True
    elif kind == 'duplicate':
        members.append(dict(members[0]))
    else:
        members.append(dict(members[0], record_sha256='0' * 64))
    assert_code('LOCAL_RESOURCE_BINDING_INVALID', lambda: owner.resolve_collection(members,
        tool_name='inspect_raw_scATAC'))


@pytest.mark.parametrize('input_type', ['h5ad', 'external_fragments', 'bam'])
def test_nonfastq_collection_member_cannot_be_reclassified(fastqs, input_type):
    owner, records = fastqs
    declared = register(owner, Path(records[0].source_path), 'other-input-type', input_type=input_type)
    assert_code('LOCAL_RESOURCE_BINDING_INVALID', lambda: owner.resolve_collection(
        references([declared, *records[1:]]), tool_name='inspect_raw_scATAC'))


def test_two_registration_ids_cannot_select_same_source_twice(fastqs):
    owner, records = fastqs
    alias = register(owner, Path(records[0].source_path), 'second-resource-for-same-source')
    assert alias.resource_id != records[0].resource_id
    assert_code('LOCAL_RESOURCE_BINDING_INVALID', lambda: owner.resolve_collection(
        references([*records, alias]), tool_name='inspect_raw_scATAC'))


@pytest.mark.parametrize('kind,code', [('missing', 'LOCAL_RESOURCE_UNAVAILABLE'),
    ('stale_digest', 'LOCAL_RESOURCE_BINDING_INVALID'), ('truncated', 'LOCAL_RESOURCE_RECORD_INVALID'),
    ('checksum', 'LOCAL_RESOURCE_RECORD_INVALID'), ('extra_field', 'LOCAL_RESOURCE_RECORD_INVALID')])
def test_any_missing_stale_or_corrupt_member_prevents_binding(fastqs, kind, code):
    owner, records = fastqs
    members = references(records)
    record = records[1]
    path = owner._path(record.resource_id)
    if kind == 'missing':
        path.unlink()
    elif kind == 'stale_digest':
        members[1]['record_sha256'] = '0' * 64
    elif kind == 'truncated':
        path.write_bytes(b'{"record":')
    else:
        envelope = json.loads(path.read_bytes())
        if kind == 'checksum':
            envelope['record']['label'] = 'tampered label'
        else:
            envelope['record']['ready'] = True
            envelope['sha256'] = digest(envelope['record'])
        path.write_bytes(canonical(envelope))
    assert_code(code, lambda: owner.resolve_collection(members, tool_name='inspect_raw_scATAC'))


@pytest.mark.parametrize('kind', ['changed', 'same_size_mtime', 'missing', 'symlink'])
def test_new_consumption_checks_every_source_and_historical_metadata_survives(fastqs, kind):
    owner, records = fastqs
    binding = owner.resolve_collection(references(records), tool_name='inspect_raw_scATAC')
    source = Path(records[-1].source_path)
    original = source.stat()
    if kind == 'changed':
        source.write_bytes(b'changed reads')
    elif kind == 'same_size_mtime':
        source.write_bytes(b'x' * original.st_size)
        os.utime(source, ns=(original.st_atime_ns, original.st_mtime_ns))
    elif kind == 'missing':
        source.unlink()
    else:
        moved = source.with_name('moved.fastq')
        source.rename(moved)
        source.symlink_to(moved)
    reopened = LocalResourceAdmission(owner._workspace.root)
    assert reopened.validate_binding(binding, verify_source=False) == _serialize(binding.execution_inputs)
    reopened.validate_submission(submission(binding), verify_source=False)
    assert_code('LOCAL_RESOURCE_INTEGRITY_INVALID', lambda: reopened.validate_binding(binding))
    assert_code('LOCAL_RESOURCE_INTEGRITY_INVALID', lambda: reopened.resolve_collection(
        references(records), tool_name='inspect_raw_scATAC'))


@pytest.mark.parametrize('mutation', ['different_bytes', 'same_bytes_replacement'])
def test_collection_rechecks_earlier_member_after_later_member_hashing(fastqs, monkeypatch, mutation):
    owner, records = fastqs
    binding = owner.resolve_collection(references(records), tool_name='inspect_raw_scATAC')
    by_id = {record.resource_id: record for record in records}
    first_record = by_id[binding.members[0]['resource_id']]
    first_source = Path(first_record.source_path)
    original_bytes = first_source.read_bytes()
    original_stat = first_source.stat()
    real_source = owner._source
    hashed = []
    def source_changed_between_members(source_path, **options):
        if len(hashed) == 1:
            if mutation == 'different_bytes':
                first_source.write_bytes(b'@new\nACGT\n+\nIIII\n')
            else:
                replacement = first_source.with_name('replacement.fastq')
                replacement.write_bytes(original_bytes)
                os.utime(replacement, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
                replacement.replace(first_source)
        identity = real_source(source_path, **options)
        record = next(record for record in records if record.source_path == str(source_path))
        assert identity[:3] == (record.source_path, record.source_sha256, record.size_bytes)
        hashed.append(record.resource_id)
        return identity
    monkeypatch.setattr(owner, '_source', source_changed_between_members)
    assert_code('LOCAL_RESOURCE_INTEGRITY_INVALID', lambda: owner.validate_binding(binding))
    assert hashed == [member['resource_id'] for member in binding.members]
    if mutation == 'same_bytes_replacement':
        assert first_source.read_bytes() == original_bytes
        assert first_source.stat().st_mtime_ns == original_stat.st_mtime_ns
        assert first_source.stat().st_ino != original_stat.st_ino


def test_collection_count_and_metadata_bounds_are_enforced(fastqs, monkeypatch):
    owner, records = fastqs
    assert resources.MAX_COLLECTION_MEMBERS == 128
    assert resources.MAX_COLLECTION_BYTES == 32768
    assert owner.resolve_collection(references(records[:1]), tool_name='inspect_raw_scATAC')
    assert_code('LOCAL_RESOURCE_BINDING_INVALID', lambda: owner.resolve_collection(
        references(records[:1]) * 129, tool_name='inspect_raw_scATAC'))
    monkeypatch.setattr(resources, 'MAX_COLLECTION_MEMBERS', 2)
    assert_code('LOCAL_RESOURCE_BINDING_INVALID', lambda: owner.resolve_collection(
        references(records), tool_name='inspect_raw_scATAC'))
    monkeypatch.setattr(resources, 'MAX_COLLECTION_MEMBERS', 128)
    binding = owner.resolve_collection(references(records), tool_name='inspect_raw_scATAC')
    metadata_bytes = len(canonical(binding.attribution()))
    monkeypatch.setattr(resources, 'MAX_COLLECTION_BYTES', metadata_bytes)
    assert owner.resolve_collection(references(records), tool_name='inspect_raw_scATAC') == binding
    monkeypatch.setattr(resources, 'MAX_COLLECTION_BYTES', metadata_bytes - 1)
    assert_code('LOCAL_RESOURCE_BINDING_INVALID', lambda: owner.resolve_collection(
        references(records), tool_name='inspect_raw_scATAC'))


def test_maximum_collection_retains_every_exact_member_without_a_collection_store(fastqs):
    owner, initial = fastqs
    records = list(initial)
    root = Path(initial[0].source_path).parent
    for number in range(len(records), resources.MAX_COLLECTION_MEMBERS):
        path = root / f'explicit-source-{number}.fastq'
        path.write_bytes(f'Unqualified source {number}'.encode())
        records.append(register(owner, path))
    binding = owner.resolve_collection(references(records)[::-1], tool_name='inspect_raw_scATAC')
    assert len(binding.members) == len(binding.execution_inputs['raw_input_paths']) == 128
    assert len(canonical(binding.attribution())) <= resources.MAX_COLLECTION_BYTES
    assert set(binding.execution_inputs['raw_input_paths']) == {record.source_path for record in records}
    assert len(list((owner._workspace.root / 'local_resources').glob('*.json'))) == 128
    owner.validate_submission(submission(binding))


@pytest.mark.parametrize('extra', ['raw_input_paths', 'source_path', 'source_sha256',
    'output_dir', 'authority_payload', 'registered_input', 'unknown'])
def test_inspection_source_and_reserved_fields_cannot_override_members(fastqs, extra):
    owner, records = fastqs
    assert_code('LOCAL_RESOURCE_BINDING_INVALID', lambda: owner.resolve_collection(
        references(records), tool_name='inspect_raw_scATAC', scientific_inputs={extra: 'supplied'}))


@pytest.mark.parametrize('mutation', ['extra', 'digest', 'member_digest', 'mixed_scalar',
    'noncanonical', 'inputs', 'missing_inputs', 'wrong_tool'])
def test_submission_metadata_and_inputs_cannot_spoof_collection(fastqs, mutation):
    owner, records = fastqs
    binding = owner.resolve_collection(references(records), tool_name='inspect_raw_scATAC')
    captured = submission(binding)
    metadata = captured['registered_input']
    if mutation == 'extra':
        metadata['ready'] = True
    elif mutation == 'digest':
        metadata['collection_sha256'] = '0' * 64
    elif mutation == 'member_digest':
        metadata['members'][0]['record_sha256'] = '0' * 64
    elif mutation == 'mixed_scalar':
        metadata['resource_id'] = records[0].resource_id
    elif mutation == 'noncanonical':
        metadata['members'].reverse()
    elif mutation == 'inputs':
        captured['execution_inputs']['raw_input_paths'].append('/unregistered.fastq')
    elif mutation == 'missing_inputs':
        captured.pop('execution_inputs')
    else:
        metadata['tool_name'] = 'prepare_scATAC_fragments'
    assert_code('LOCAL_RESOURCE_BINDING_INVALID', lambda: owner.validate_submission(captured,
        verify_source=False))


@pytest.mark.parametrize('tool', ['inspect_scATAC', 'prepare_scATAC_bam_fragments',
    'import_scATAC_fragments', 'epizoo_embed_cells', 'prepare_neutral_bam_fragments'])
def test_collection_only_exposes_explicit_registered_operations(fastqs, tool):
    owner, records = fastqs
    assert_code('LOCAL_RESOURCE_OPERATION_UNSUPPORTED', lambda: owner.resolve_collection(
        references(records), tool_name=tool))


def test_collection_cannot_bypass_operator_registry_allowlist(fastqs):
    owner, records = fastqs
    restricted = LocalResourceAdmission(owner._workspace, registry=ToolRegistry(()))
    assert_code('LOCAL_RESOURCE_BINDING_INVALID', lambda: restricted.resolve_collection(
        references(records), tool_name='inspect_raw_scATAC'))


def test_fastq_requires_collection_even_for_single_file(fastqs):
    owner, records = fastqs
    assert_code('LOCAL_RESOURCE_BINDING_INVALID', lambda: owner.resolve(records[0].resource_id,
        tool_name='inspect_raw_scATAC'))
    binding = owner.resolve_collection(references(records[:1]), tool_name='inspect_raw_scATAC')
    assert owner.validate_binding(binding) == dict(raw_input_paths=[records[0].source_path])


def test_reopen_preserves_legacy_scalar_bindings_and_record_bytes(fastqs, tmp_path):
    owner, records = fastqs
    legacy = []
    for kind, tool, declarations in [
        ('h5ad', 'inspect_scATAC', {}),
        ('external_fragments', 'import_scATAC_fragments', dict(reference_bundle_path=str(tmp_path / 'ref.json'),
            reference_bundle_sha256='1' * 64, source_profile='10x-atac-fragments.v1', namespace='library')),
        ('bam', 'inspect_raw_scATAC', {}),
    ]:
        record = register(owner, Path(records[0].source_path), kind, input_type=kind)
        binding = owner.resolve(record.resource_id, tool_name=tool, scientific_inputs=declarations)
        legacy.append((record, binding, owner._path(record.resource_id).read_bytes()))
    owner.resolve_collection(references(records), tool_name='inspect_raw_scATAC')
    reopened = LocalResourceAdmission(owner._workspace.root)
    for record, binding, stored in legacy:
        assert type(binding) is RegisteredInput
        assert set(binding.attribution()) == {'resource_id', 'record_sha256', 'tool_name'}
        reopened.validate_submission(submission(binding))
        assert reopened._path(record.resource_id).read_bytes() == stored


@pytest.fixture
def producer_resources(tmp_path):
    approved = tmp_path / 'reads'
    approved.mkdir()
    owner = LocalResourceAdmission(tmp_path / 'workspace', approved_source_roots=(approved,))
    records = []
    for lane in ('001', '002'):
        for role in ('R1', 'R2', 'R3', 'I1'):
            sequence = b'ACGTACGTACGTACGT' if role == 'R2' else b'ACGTACGT'
            path = approved / f'sample_S1_L{lane}_{role}_001.fastq'
            path.write_bytes(b'@read\n' + sequence + b'\n+\n' + b'I' * len(sequence) + b'\n')
            records.append(register(owner, path))
    intake = inspect_raw_scATAC([record.source_path for record in records], tmp_path / 'intake',
                               species='human', raw_assay='TENX_ATAC')
    _, manifest, _ = raw_scatac_manifest.load_raw_intake_manifest(intake['manifest_path'],
        expected_sha256=intake['manifest_sha256'])
    whitelist_path = tmp_path / 'whitelist.txt'
    whitelist_path.write_bytes(b'ACGTACGTACGTACGT\n')
    whitelist = scatac_library_context.inspect_barcode_whitelist(whitelist_path)
    declarations = tuple(scatac_library_context.LibraryDeclaration(
        namespace=f'library_{index}', group_ids=(group.id,),
        membership_basis=scatac_library_context.MembershipBasis.SINGLE_GROUP,
        barcode_interpretation=scatac_library_context.BarcodeInterpretation.RAW_SEQUENCE,
        correction_policy=scatac_library_context.CorrectionPolicy.WHITELIST_REQUIRED,
        whitelist=whitelist) for index, group in enumerate(manifest.groups))
    def context(selected, mode, name):
        value = scatac_library_context.build_scatac_library_processing_context(
            intake_manifest_path=intake['manifest_path'], expected_intake_sha256=intake['manifest_sha256'],
            libraries=selected, selection_mode=mode)
        return scatac_library_context.publish_scatac_library_processing_context(value, tmp_path / name)
    complete = context(declarations, scatac_library_context.SelectionMode.ALL_GROUPS, 'all-context.json')
    subset = context(declarations[:1], scatac_library_context.SelectionMode.EXPLICIT_SUBSET, 'subset-context.json')
    scientific = dict(intake_manifest_path=intake['manifest_path'], intake_manifest_sha256=intake['manifest_sha256'],
        library_context_path=complete['manifest_path'], library_context_sha256=complete['manifest_sha256'],
        reference_bundle_path=str(tmp_path / 'explicit-reference.json'), reference_bundle_sha256='1' * 64)
    return owner, records, scientific, intake, manifest, subset


def test_exact_intake_and_explicit_library_subset_preserve_producer_contract(producer_resources):
    owner, records, declarations, intake, manifest, subset = producer_resources
    assert intake['readiness'] == 'READY'
    assert len(manifest.groups) == 2 and len(manifest.files) == 8
    inspect_binding = owner.resolve_collection(references(records), tool_name='inspect_raw_scATAC')
    complete = owner.resolve_collection(references(records), tool_name='prepare_scATAC_fragments',
                                       scientific_inputs=declarations)
    subset_declarations = declarations | dict(library_context_path=subset['manifest_path'],
                                              library_context_sha256=subset['manifest_sha256'])
    selected = owner.resolve_collection(references(records)[::-1], tool_name='prepare_scATAC_fragments',
                                       scientific_inputs=subset_declarations)
    assert type(selected) is RegisteredInputCollection
    assert owner.validate_binding(complete) == declarations
    assert owner.validate_binding(selected) == subset_declarations
    assert set(selected.execution_inputs) == set(build_default_tool_registry().get(
        'prepare_scATAC_fragments').required_arguments) - {'output_dir'}
    assert selected.collection_sha256 == complete.collection_sha256 == inspect_binding.collection_sha256
    assert _serialize(selected.members) == _serialize(complete.members)
    assert sorted(inspect_binding.execution_inputs['raw_input_paths']) == sorted(source.path for source in manifest.files)
    assert sum('_I1_' in source.path for source in manifest.files) == 2
    owner.validate_submission(submission(selected))


@pytest.mark.parametrize('missing', ['intake_manifest_path', 'intake_manifest_sha256',
    'library_context_path', 'library_context_sha256', 'reference_bundle_path', 'reference_bundle_sha256'])
def test_producer_required_declarations_are_not_manufactured(producer_resources, missing):
    owner, records, declarations, *_ = producer_resources
    supplied = dict(declarations)
    supplied.pop(missing)
    assert_code('LOCAL_RESOURCE_DECLARATION_REQUIRED', lambda: owner.resolve_collection(
        references(records), tool_name='prepare_scATAC_fragments', scientific_inputs=supplied))


@pytest.mark.parametrize('extra', ['raw_input_paths', 'source_path', 'source_sha256', 'output_dir',
    'source_profile', 'input_spec_path', 'species', 'assembly', 'authority_payload'])
def test_producer_admission_does_not_extend_unchanged_scientific_inputs(producer_resources, extra):
    owner, records, declarations, *_ = producer_resources
    assert_code('LOCAL_RESOURCE_BINDING_INVALID', lambda: owner.resolve_collection(
        references(records), tool_name='prepare_scATAC_fragments', scientific_inputs=declarations | {extra: 'supplied'}))


@pytest.mark.parametrize('mutation', ['other_source', 'directory', 'extra_source', 'subset_intake',
    'checksum', 'unpinned', 'corrupt', 'size'])
def test_producer_intake_must_match_complete_explicit_registered_collection(producer_resources, tmp_path, mutation):
    owner, records, declarations, intake, manifest, _ = producer_resources
    supplied = dict(declarations)
    if mutation in {'other_source', 'directory', 'extra_source', 'subset_intake'}:
        paths = [record.source_path for record in records]
        if mutation in {'other_source', 'extra_source'}:
            other = tmp_path / 'other_S1_L001_R1_001.fastq'
            other.write_bytes(Path(paths[0]).read_bytes())
            paths = [str(other), *paths[1:]] if mutation == 'other_source' else [*paths, str(other)]
        elif mutation == 'directory':
            paths = str(Path(paths[0]).parent)
        else:
            paths = [path for path in paths if '_L001_' in path]
        wrong = inspect_raw_scATAC(paths, tmp_path / 'unrelated-intake',
                                  species='human', raw_assay='TENX_ATAC')
        supplied.update(intake_manifest_path=wrong['manifest_path'], intake_manifest_sha256=wrong['manifest_sha256'])
    elif mutation == 'checksum':
        supplied['intake_manifest_sha256'] = '0' * 64
    elif mutation == 'unpinned':
        supplied['intake_manifest_sha256'] = None
    elif mutation == 'corrupt':
        Path(intake['manifest_path']).write_bytes(b'corrupt intake')
    else:
        changed = replace(manifest, files=(replace(manifest.files[0], size_bytes=manifest.files[0].size_bytes + 1),
                                           *manifest.files[1:]))
        pointer = raw_scatac_manifest.publish_raw_intake_manifest(changed, tmp_path / 'wrong-size.json')
        supplied.update(intake_manifest_path=pointer['manifest_path'], intake_manifest_sha256=pointer['manifest_sha256'])
    assert_code('LOCAL_RESOURCE_BINDING_INVALID', lambda: owner.resolve_collection(
        references(records), tool_name='prepare_scATAC_fragments', scientific_inputs=supplied))


def test_producer_historical_validation_reads_neither_raw_sources_nor_intake(producer_resources, monkeypatch):
    owner, records, declarations, *_ = producer_resources
    binding = owner.resolve_collection(references(records), tool_name='prepare_scATAC_fragments',
                                       scientific_inputs=declarations)
    def forbidden(*args, **kwargs):
        pytest.fail('Historical attribution accessed a scientific input.')
    monkeypatch.setattr(owner, '_source', forbidden)
    monkeypatch.setattr(raw_scatac_manifest, 'load_raw_intake_manifest', forbidden)
    monkeypatch.setattr(scatac_library_context, 'load_scatac_library_processing_context', forbidden)
    assert owner.validate_binding(binding, verify_source=False) == declarations
    owner.validate_submission(submission(binding), verify_source=False)


def test_producer_new_consumption_rechecks_intake_without_reconstructing_science(producer_resources, monkeypatch):
    owner, records, declarations, *_ = producer_resources
    binding = owner.resolve_collection(references(records), tool_name='prepare_scATAC_fragments',
                                       scientific_inputs=declarations)
    monkeypatch.setattr(_raw_fastq, 'inspect_fastq_inputs',
                        lambda *a, **k: pytest.fail('Admission reconstructed raw inspection.'))
    Path(declarations['intake_manifest_path']).write_bytes(b'intake changed after resolution')
    assert_code('LOCAL_RESOURCE_BINDING_INVALID', lambda: owner.validate_binding(binding))
    assert owner.validate_binding(binding, verify_source=False) == declarations


@pytest.mark.parametrize('consumption', ['resolve', 'validate_binding'])
def test_producer_collection_rechecks_source_changed_during_intake_validation(
        producer_resources, monkeypatch, consumption):
    owner, records, declarations, *_ = producer_resources
    binding = owner.resolve_collection(references(records), tool_name='prepare_scATAC_fragments',
                                       scientific_inputs=declarations)
    source = Path(records[0].source_path)
    before = source.stat()
    original_bytes = source.read_bytes()
    original_load = raw_scatac_manifest.load_raw_intake_manifest
    intake_reads = []
    def source_changed_after_intake_read(*args, **kwargs):
        value = original_load(*args, **kwargs)
        intake_reads.append(value)
        source.write_bytes(original_bytes.replace(b'ACGT', b'TGCA', 1))
        os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
        assert source.stat().st_size == before.st_size
        return value
    monkeypatch.setattr(raw_scatac_manifest, 'load_raw_intake_manifest', source_changed_after_intake_read)
    if consumption == 'resolve':
        consume = lambda: owner.resolve_collection(references(records),
            tool_name='prepare_scATAC_fragments', scientific_inputs=declarations)
    else:
        consume = lambda: owner.validate_binding(binding)
    assert_code('LOCAL_RESOURCE_INTEGRITY_INVALID', consume)
    assert len(intake_reads) == 1
