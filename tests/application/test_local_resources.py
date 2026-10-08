"""Registration and deterministic bindings, without scientific qualification."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
import os
from pathlib import Path
import sys

import pytest

from agent.application.local_resources import (
    LocalResourceAdmission, RegisteredInput, ResourceAdmissionError, MAX_RECORD_BYTES,
)
from agent.application.session_state import canonical, digest
from agent.orchestration.registry import build_default_tool_registry, ToolArgumentError, ToolRegistry
from agent.schemas.orchestration import _serialize

sys.path.insert(0, str(Path(__file__).parents[1]))
from bam_fragments.conftest import bam_factory


@pytest.fixture
def local(tmp_path):
    sources = tmp_path / 'approved'
    sources.mkdir()
    source = sources / 'sample.h5ad'
    source.write_bytes(b'deliberately unreadable scientific data')
    owner = LocalResourceAdmission(tmp_path / 'workspace', approved_source_roots=(sources,))
    return owner, source


def register(owner, source, **changes):
    return owner.register('operator-selection-1', source,
        **(dict(label='Sample 1', attribution='Declared by the operator; provenance unverified') | changes))


def companion_inputs(tmp_path):
    return dict(reference_manifest_path=str(tmp_path / 'reference.json'),
                reference_manifest_sha256='1' * 64, species='human', assembly='hg38',
                matrix_semantics='fragment_counts')


def fragments_inputs(tmp_path):
    return dict(reference_bundle_path=str(tmp_path / 'reference.json'),
                reference_bundle_sha256='1' * 64,
                source_profile='10x-atac-fragments.v1', namespace='explicit_library')


def assert_code(code, call):
    with pytest.raises(ResourceAdmissionError) as failed:
        call()
    assert failed.value.code == failed.value.error.code == code
    assert failed.value.message == failed.value.error.message
    assert '/' not in failed.value.message


@pytest.mark.parametrize('input_type', ['h5ad', 'external_fragments', 'bam'])
def test_registration_is_lazy_private_and_not_science(local, monkeypatch, input_type):
    owner, source = local
    workspace = owner._workspace.root
    assert not (workspace / 'local_resources').exists()
    assert not (workspace / 'sessions').exists()
    def forbidden(*args, **kwargs):
        pytest.fail('Registration entered planning or science.')
    import agent.tools.data.scatac as inspection
    import agent.tools.data.scatac_matrix_adoption as adoption
    import agent.tools.data.scatac_fragment_import as fragments
    import agent.tools.data.external_fragments as fragment_parser
    import agent.tools.data.external_fragments_verifier as fragment_verifier
    import agent.tools.data._raw_bam as raw_bam
    import agent.tools.data.bam_fragments as bam_producer
    import agent.orchestration.registry as registry
    monkeypatch.setattr(inspection, 'inspect_scATAC', forbidden)
    monkeypatch.setattr(adoption, 'adopt_scATAC_cell_by_ccre', forbidden)
    monkeypatch.setattr(fragments, 'import_scATAC_fragments', forbidden)
    monkeypatch.setattr(fragment_parser, 'scan_source', forbidden)
    monkeypatch.setattr(fragment_verifier, '_reconstruct_source', forbidden)
    monkeypatch.setattr(raw_bam, 'inspect_bam_inputs', forbidden)
    monkeypatch.setattr(bam_producer, 'prepare_in_stage', forbidden)
    monkeypatch.setattr(registry, 'build_default_tool_registry', forbidden)
    record = register(owner, source, input_type=input_type)
    assert record.source_sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    assert record.size_bytes == source.stat().st_size
    assert record.record_sha256 == digest(record.to_dict())
    public = record.public()
    assert public == dict(resource_id=record.resource_id, label='Sample 1', input_type=input_type, status='registered')
    assert str(source) not in json.dumps(public)
    assert not (workspace / 'sessions').exists()
    assert list((workspace / 'runs').iterdir()) == []
    assert list((workspace / 'run_state').iterdir()) == []
    with pytest.raises(FrozenInstanceError):
        record.label = 'changed'


def test_equivalent_retry_and_reopen_preserve_exact_record(local):
    owner, source = local
    record = register(owner, source)
    raw = owner._path(record.resource_id).read_bytes()
    assert register(owner, source) == record
    assert owner._path(record.resource_id).read_bytes() == raw
    reopened = LocalResourceAdmission(owner._workspace.root)
    assert reopened.load(record.resource_id) == record
    binding = reopened.resolve(record.resource_id, tool_name='inspect_scATAC')
    assert reopened.validate_binding(binding) == {'input_path': str(source)}
    assert_code('LOCAL_RESOURCE_ACCESS_INVALID', lambda: register(reopened, source))


@pytest.mark.parametrize('input_type', ['h5ad', 'external_fragments', 'bam'])
@pytest.mark.parametrize('change', ['label', 'attribution', 'source', 'bytes', 'type'])
def test_registration_key_cannot_change_meaning(local, change, input_type):
    owner, source = local
    original = register(owner, source, input_type=input_type)
    supplied = source
    options = dict(input_type=input_type)
    if change in {'label', 'attribution'}:
        options[change] = 'changed declaration'
    elif change == 'type':
        options['input_type'] = 'external_fragments' if input_type == 'h5ad' else 'h5ad'
    elif change == 'source':
        supplied = source.with_name('other.h5ad')
        supplied.write_bytes(source.read_bytes())
    else:
        source.write_bytes(b'changed source bytes')
    assert_code('LOCAL_RESOURCE_CONFLICT', lambda: register(owner, supplied, **options))
    assert owner.load(original.resource_id) == original


def test_new_key_can_register_changed_content_without_replacing_history(local):
    owner, source = local
    original = register(owner, source)
    source.write_bytes(b'new bytes')
    new = owner.register('operator-selection-2', source, label='Sample 2', attribution='Operator declaration')
    assert new.resource_id != original.resource_id and new.source_sha256 != original.source_sha256
    assert owner.load(original.resource_id) == original


@pytest.mark.parametrize('input_type', ['h5ad', 'external_fragments', 'bam'])
@pytest.mark.parametrize('kind', ['outside', 'directory', 'missing', 'url', 'symlink', 'ancestor_symlink'])
def test_only_operator_approved_canonical_regular_sources(local, tmp_path, kind, input_type):
    owner, source = local
    supplied = source
    if kind == 'outside':
        supplied = tmp_path / 'outside.h5ad'
        supplied.write_bytes(b'bytes')
    elif kind == 'directory':
        supplied = source.parent
    elif kind == 'missing':
        supplied = source.parent / 'missing.h5ad'
    elif kind == 'url':
        supplied = 'https://example.invalid/source.h5ad'
    elif kind == 'symlink':
        supplied = source.with_name('alias.h5ad')
        supplied.symlink_to(source)
    else:
        alias = tmp_path / 'alias'
        alias.symlink_to(source.parent, target_is_directory=True)
        supplied = alias / source.name
    assert_code('LOCAL_RESOURCE_ACCESS_INVALID', lambda: register(owner, supplied, input_type=input_type))
    assert not (owner._workspace.root / 'local_resources').exists()


@pytest.mark.parametrize('input_type', ['fastq', 'tbi', 'cram'])
def test_unsupported_type_and_unsafe_label_do_not_access_or_register(local, monkeypatch, input_type):
    owner, source = local
    monkeypatch.setattr(owner, '_source', lambda *args, **kwargs: pytest.fail('Source read occurred.'))
    assert_code('LOCAL_RESOURCE_TYPE_UNSUPPORTED', lambda: register(owner, source, input_type=input_type))
    assert_code('LOCAL_RESOURCE_RECORD_INVALID', lambda: register(owner, source, label=str(source)))
    assert not (owner._workspace.root / 'local_resources').exists()


@pytest.mark.parametrize('input_type', ['h5ad', 'external_fragments', 'bam'])
@pytest.mark.parametrize('mutation', ['truncated', 'checksum', 'duplicate', 'extra', 'bool_version', 'wrong_identity', 'oversize'])
def test_corrupt_or_incomplete_records_fail_closed(local, mutation, input_type):
    owner, source = local
    record = register(owner, source, input_type=input_type)
    path = owner._path(record.resource_id)
    envelope = json.loads(path.read_bytes())
    if mutation == 'truncated':
        payload = b'{"record":'
    elif mutation == 'duplicate':
        payload = path.read_bytes()[:-1] + b',"format":"agent.local-resource.v1"}'
    elif mutation == 'oversize':
        payload = b' ' * (MAX_RECORD_BYTES + 1)
    else:
        if mutation == 'checksum':
            envelope['record']['label'] = 'changed'
        elif mutation == 'extra':
            envelope['record']['scientifically_accepted'] = True
            envelope['sha256'] = digest(envelope['record'])
        elif mutation == 'bool_version':
            envelope['schema_version'] = True
        else:
            envelope['record']['resource_id'] = 'local-' + '0' * 64
            envelope['sha256'] = digest(envelope['record'])
        payload = canonical(envelope)
    path.write_bytes(payload)
    assert_code('LOCAL_RESOURCE_RECORD_INVALID', lambda: owner.load(record.resource_id))


def test_unknown_resource_and_read_does_not_create_store(local):
    owner, _ = local
    unknown = 'local-' + 'a' * 64
    assert_code('LOCAL_RESOURCE_UNAVAILABLE', lambda: owner.load(unknown))
    assert not (owner._workspace.root / 'local_resources').exists()


@pytest.mark.parametrize('kind', ['file', 'directory'])
def test_resource_store_symlinks_are_rejected(local, tmp_path, kind):
    owner, source = local
    record = register(owner, source)
    if kind == 'file':
        path = owner._path(record.resource_id)
        target = tmp_path / 'record-copy'
        path.rename(target)
        path.symlink_to(target)
    else:
        path = owner._workspace.root / 'local_resources'
        target = tmp_path / 'resource-copy'
        path.rename(target)
        path.symlink_to(target, target_is_directory=True)
    assert_code('LOCAL_RESOURCE_RECORD_INVALID', lambda: owner.load(record.resource_id))


@pytest.mark.parametrize('input_type', ['h5ad', 'external_fragments', 'bam'])
def test_interrupted_publication_exposes_no_partial_record(local, monkeypatch, input_type):
    owner, source = local
    def interrupted(*args, **kwargs):
        raise OSError('Synthetic interrupted publication.')
    monkeypatch.setattr(os, 'link', interrupted)
    assert_code('LOCAL_RESOURCE_RECORD_INVALID', lambda: register(owner, source, input_type=input_type))
    root = owner._workspace.root / 'local_resources'
    assert list(root.glob('*.json')) == [] and list(root.glob('*.tmp')) == []
    assert source.read_bytes() == b'deliberately unreadable scientific data'


@pytest.mark.parametrize('input_type', ['h5ad', 'external_fragments', 'bam'])
def test_source_changed_before_publication_is_not_registered(local, monkeypatch, input_type):
    owner, source = local
    original = owner._source
    def replaced(*args, **kwargs):
        identity = original(*args, **kwargs)
        source.write_bytes(b'changed after source hashing')
        return identity
    monkeypatch.setattr(owner, '_source', replaced)
    assert_code('LOCAL_RESOURCE_INTEGRITY_INVALID', lambda: register(owner, source, input_type=input_type))
    assert list((owner._workspace.root / 'local_resources').glob('*.json')) == []


@pytest.mark.parametrize('input_type', ['h5ad', 'external_fragments', 'bam'])
def test_source_mutation_during_hash_is_not_registered(local, monkeypatch, input_type):
    owner, source = local
    original = hashlib.sha256
    class MutatingHash:
        def __init__(self):
            self.hash = original()
        def update(self, chunk):
            self.hash.update(chunk)
            source.write_bytes(b'changed while hashing')
        def hexdigest(self):
            return self.hash.hexdigest()
    monkeypatch.setattr(hashlib, 'sha256', lambda value=b'': original(value) if value else MutatingHash())
    assert_code('LOCAL_RESOURCE_INTEGRITY_INVALID', lambda: register(owner, source, input_type=input_type))
    assert not (owner._workspace.root / 'local_resources').exists()


@pytest.mark.parametrize('input_type', ['h5ad', 'external_fragments', 'bam'])
def test_concurrent_equivalent_registration_is_idempotent(local, input_type):
    owner, source = local
    with ThreadPoolExecutor(max_workers=2) as executor:
        records = list(executor.map(lambda _: register(owner, source, input_type=input_type), range(2)))
    assert records[0] == records[1]
    assert len(list((owner._workspace.root / 'local_resources').glob('*.json'))) == 1


@pytest.mark.parametrize('input_type', ['h5ad', 'external_fragments', 'bam'])
@pytest.mark.parametrize('created', ['directory', 'symlink', 'file'])
def test_concurrent_resource_directory_creation_is_revalidated(local, tmp_path, monkeypatch, input_type, created):
    owner, source = local
    directory = owner._workspace.root / 'local_resources'
    original = Path.mkdir
    collided = False

    def collision(path, *args, **kwargs):
        nonlocal collided
        if path == directory and not collided:
            collided = True
            if created == 'directory':
                original(path, *args, **kwargs)
            elif created == 'symlink':
                path.symlink_to(tmp_path, target_is_directory=True)
            else:
                path.write_bytes(b'not a directory')
            raise FileExistsError('Another registration created the resource path.')
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'mkdir', collision)
    if created == 'directory':
        record = register(owner, source, input_type=input_type)
        assert owner.load(record.resource_id) == record
        assert register(owner, source, input_type=input_type) == record
    else:
        assert_code('LOCAL_RESOURCE_RECORD_INVALID', lambda: register(owner, source, input_type=input_type))
    assert collided


def test_resource_directory_access_error_is_not_retried(local, monkeypatch):
    owner, source = local
    calls = []

    def denied(path, *args, **kwargs):
        calls.append(path)
        raise PermissionError('Directory creation denied.')

    monkeypatch.setattr(Path, 'mkdir', denied)
    assert_code('LOCAL_RESOURCE_RECORD_INVALID', lambda: register(owner, source, input_type='external_fragments'))
    assert calls == [owner._workspace.root / 'local_resources']


def test_adoption_mapping_is_exact_without_scientific_qualification(local, tmp_path):
    owner, source = local
    record = register(owner, source)
    # A nonexistent reference and scientifically invalid semantics are not
    # validated by registration/resolution. Registry and the owner retain them.
    declarations = companion_inputs(tmp_path) | {'matrix_semantics': 'invented_counts'}
    binding = owner.resolve(record.resource_id, tool_name='adopt_scATAC_cell_by_ccre', scientific_inputs=declarations)
    expected = declarations | dict(source_path=str(source), source_sha256=record.source_sha256)
    assert owner.validate_binding(binding) == expected
    assert binding.attribution() == dict(resource_id=record.resource_id,
        record_sha256=record.record_sha256, tool_name='adopt_scATAC_cell_by_ccre')
    with pytest.raises(ToolArgumentError):
        build_default_tool_registry().validate_arguments(binding.tool_name, expected | {'output_dir': str(tmp_path / 'out')})


@pytest.mark.parametrize('missing', ['reference_manifest_path', 'reference_manifest_sha256', 'species', 'assembly', 'matrix_semantics'])
def test_required_adoption_declarations_are_not_guessed(local, tmp_path, missing):
    owner, source = local
    record = register(owner, source)
    declarations = companion_inputs(tmp_path)
    declarations.pop(missing)
    assert_code('LOCAL_RESOURCE_DECLARATION_REQUIRED', lambda: owner.resolve(record.resource_id,
        tool_name='adopt_scATAC_cell_by_ccre', scientific_inputs=declarations))


@pytest.mark.parametrize('extra', ['source_path', 'source_sha256', 'output_dir', 'unknown', 'authority_payload'])
def test_source_fields_and_reserved_declarations_cannot_be_supplied(local, tmp_path, extra):
    owner, source = local
    record = register(owner, source)
    assert_code('LOCAL_RESOURCE_BINDING_INVALID', lambda: owner.resolve(record.resource_id,
        tool_name='adopt_scATAC_cell_by_ccre', scientific_inputs=companion_inputs(tmp_path) | {extra: 'supplied'}))


def test_binding_is_frozen_and_preserves_exact_caller_declarations(local, tmp_path):
    owner, source = local
    record = register(owner, source)
    declarations = companion_inputs(tmp_path)
    binding = owner.resolve(record.resource_id, tool_name='adopt_scATAC_cell_by_ccre', scientific_inputs=declarations)
    declarations['assembly'] = 'mm10'
    assert binding.execution_inputs['assembly'] == 'hg38'
    with pytest.raises(TypeError):
        binding.execution_inputs['assembly'] = 'mm10'
    with pytest.raises(FrozenInstanceError):
        binding.resource_id = 'local-' + '0' * 64


@pytest.mark.parametrize('mutation', ['source_path', 'source_sha256', 'missing_path', 'record_digest', 'tool'])
def test_forged_bindings_do_not_change_registered_source(local, tmp_path, mutation):
    owner, source = local
    record = register(owner, source)
    binding = owner.resolve(record.resource_id, tool_name='adopt_scATAC_cell_by_ccre', scientific_inputs=companion_inputs(tmp_path))
    values = _serialize(binding.execution_inputs)
    if mutation == 'record_digest':
        forged = replace(binding, record_sha256='0' * 64)
    elif mutation == 'tool':
        forged = replace(binding, tool_name='inspect_scATAC')
    else:
        if mutation == 'missing_path':
            values.pop('source_path')
        else:
            values[mutation] = str(tmp_path / 'different.h5ad') if mutation == 'source_path' else '0' * 64
        forged = replace(binding, execution_inputs=values)
    assert_code('LOCAL_RESOURCE_BINDING_INVALID', lambda: owner.validate_binding(forged, verify_source=False))


@pytest.mark.parametrize('input_type', ['h5ad', 'external_fragments', 'bam'])
@pytest.mark.parametrize('mutation', ['replace', 'same_stat_size', 'delete', 'symlink'])
def test_new_consumption_fails_but_historical_binding_survives_source_change(local, tmp_path, mutation, input_type):
    owner, source = local
    record = register(owner, source, input_type=input_type)
    tool = {'h5ad': 'inspect_scATAC', 'external_fragments': 'import_scATAC_fragments',
            'bam': 'inspect_raw_scATAC'}[input_type]
    declarations = fragments_inputs(tmp_path) if input_type == 'external_fragments' else None
    binding = owner.resolve(record.resource_id, tool_name=tool, scientific_inputs=declarations)
    captured = dict(registered_input=binding.attribution(), execution_inputs=_serialize(binding.execution_inputs))
    original = source.stat()
    if mutation == 'replace':
        replacement = source.with_name('replacement')
        replacement.write_bytes(b'changed')
        replacement.replace(source)
    elif mutation == 'same_stat_size':
        source.write_bytes(b'x' * original.st_size)
        os.utime(source, ns=(original.st_atime_ns, original.st_mtime_ns))
    elif mutation == 'delete':
        source.unlink()
    else:
        target = source.with_name('elsewhere')
        source.rename(target)
        source.symlink_to(target)
    reopened = LocalResourceAdmission(owner._workspace.root)
    assert reopened.load(record.resource_id) == record
    assert reopened.validate_binding(binding, verify_source=False) == _serialize(binding.execution_inputs)
    reopened.validate_submission(captured, verify_source=False)
    assert_code('LOCAL_RESOURCE_INTEGRITY_INVALID', lambda: reopened.validate_binding(binding))
    assert_code('LOCAL_RESOURCE_INTEGRITY_INVALID', lambda: reopened.resolve(record.resource_id,
        tool_name=tool, scientific_inputs=declarations))


@pytest.mark.parametrize('change', ['metadata_extra', 'digest', 'inputs', 'missing'])
def test_submission_resource_attribution_is_exact(local, change):
    owner, source = local
    record = register(owner, source)
    binding = owner.resolve(record.resource_id, tool_name='inspect_scATAC')
    submission = dict(registered_input=binding.attribution(), execution_inputs=_serialize(binding.execution_inputs))
    owner.validate_submission(submission)
    if change == 'metadata_extra':
        submission['registered_input']['source_path'] = str(source)
    elif change == 'digest':
        submission['registered_input']['record_sha256'] = '0' * 64
    elif change == 'inputs':
        submission['execution_inputs']['input_path'] = str(source.with_name('different.h5ad'))
    else:
        submission.pop('registered_input')
    assert_code('LOCAL_RESOURCE_BINDING_INVALID', lambda: owner.validate_submission(submission, verify_source=False))


def test_unsupported_operation_is_not_selected_or_completed(local):
    owner, source = local
    record = register(owner, source)
    assert_code('LOCAL_RESOURCE_OPERATION_UNSUPPORTED', lambda: owner.resolve(record.resource_id, tool_name='epizoo_embed_cells'))
    assert_code('LOCAL_RESOURCE_OPERATION_UNSUPPORTED', lambda: RegisteredInput(record.resource_id,
        record.record_sha256, 'epizoo_embed_cells', {'input_path': str(source)}))


def test_registration_does_not_bypass_a_restricted_registry(local):
    owner, source = local
    record = register(owner, source)
    restricted = LocalResourceAdmission(owner._workspace, registry=ToolRegistry(()))
    assert_code('LOCAL_RESOURCE_BINDING_INVALID', lambda: restricted.resolve(record.resource_id, tool_name='inspect_scATAC'))


def test_record_only_binding_validation_never_opens_current_source(local, monkeypatch):
    owner, source = local
    record = register(owner, source)
    binding = owner.resolve(record.resource_id, tool_name='inspect_scATAC')
    monkeypatch.setattr(owner, '_source', lambda *args, **kwargs: pytest.fail('Historical read accessed source.'))
    assert owner.validate_binding(binding, verify_source=False) == {'input_path': str(source)}


def test_fragments_reopen_preserves_existing_h5ad_record(local, tmp_path):
    owner, source = local
    h5ad = register(owner, source)
    old_bytes = owner._path(h5ad.resource_id).read_bytes()
    fragments = owner.register('fragments-selection', source, input_type='external_fragments',
        label='Supplied fragments', attribution='Declared export, not qualified')
    raw = owner._path(fragments.resource_id).read_bytes()
    assert owner.register('fragments-selection', source, input_type='external_fragments',
        label='Supplied fragments', attribution='Declared export, not qualified') == fragments
    reopened = LocalResourceAdmission(owner._workspace.root)
    assert reopened.load(fragments.resource_id) == fragments
    assert reopened.load(h5ad.resource_id) == h5ad
    assert reopened.resolve(h5ad.resource_id, tool_name='inspect_scATAC').execution_inputs['input_path'] == str(source)
    binding = reopened.resolve(fragments.resource_id, tool_name='import_scATAC_fragments',
        scientific_inputs=fragments_inputs(tmp_path))
    assert binding.record_sha256 == fragments.record_sha256
    assert reopened._path(h5ad.resource_id).read_bytes() == old_bytes
    assert reopened._path(fragments.resource_id).read_bytes() == raw
    assert_code('LOCAL_RESOURCE_ACCESS_INVALID', lambda: reopened.register('new-fragments', source,
        input_type='external_fragments', label='New source', attribution='Operator declaration'))


@pytest.mark.parametrize('optional', [{}, {'source_selection': 'subset_export'},
    {'source_index_path': '/operator/index.tbi', 'source_index_sha256': '2' * 64},
    {'source_index_path': None, 'source_index_sha256': None}])
def test_fragments_mapping_preserves_explicit_declarations_without_science(local, tmp_path, optional):
    owner, source = local
    record = register(owner, source, input_type='external_fragments')
    declarations = fragments_inputs(tmp_path) | optional
    expected = declarations | dict(source_path=str(source), source_sha256=record.source_sha256)
    binding = owner.resolve(record.resource_id, tool_name='import_scATAC_fragments', scientific_inputs=declarations)
    declarations['namespace'] = 'later_change'
    assert owner.validate_binding(binding) == expected
    assert binding.attribution() == dict(resource_id=record.resource_id,
        record_sha256=record.record_sha256, tool_name='import_scATAC_fragments')
    owner.validate_submission(dict(registered_input=binding.attribution(), execution_inputs=expected))
    assert ('source_selection' in binding.execution_inputs) == ('source_selection' in optional)
    with pytest.raises(TypeError):
        binding.execution_inputs['namespace'] = 'changed'


@pytest.mark.parametrize('missing', ['reference_bundle_path', 'reference_bundle_sha256', 'source_profile', 'namespace'])
def test_fragments_mandatory_declarations_are_not_inferred(local, tmp_path, missing):
    owner, source = local
    record = register(owner, source, input_type='external_fragments')
    declarations = fragments_inputs(tmp_path)
    del declarations[missing]
    assert_code('LOCAL_RESOURCE_DECLARATION_REQUIRED', lambda: owner.resolve(record.resource_id,
        tool_name='import_scATAC_fragments', scientific_inputs=declarations))


@pytest.mark.parametrize('extra', ['source_path', 'source_sha256', 'output_dir', 'input_path',
    'raw_input_paths', 'primary_preparation', 'source_index_resource_id', 'authority_payload'])
def test_fragments_source_overrides_and_unsupported_inputs_fail_closed(local, tmp_path, extra):
    owner, source = local
    record = register(owner, source, input_type='external_fragments')
    assert_code('LOCAL_RESOURCE_BINDING_INVALID', lambda: owner.resolve(record.resource_id,
        tool_name='import_scATAC_fragments', scientific_inputs=fragments_inputs(tmp_path) | {extra: 'supplied'}))


@pytest.mark.parametrize('input_type,tool', [('h5ad', 'import_scATAC_fragments'),
    ('external_fragments', 'inspect_scATAC'), ('external_fragments', 'adopt_scATAC_cell_by_ccre'),
    ('h5ad', 'inspect_raw_scATAC'), ('h5ad', 'prepare_scATAC_bam_fragments'),
    ('external_fragments', 'inspect_raw_scATAC'), ('external_fragments', 'prepare_scATAC_bam_fragments'),
    ('bam', 'inspect_scATAC'), ('bam', 'adopt_scATAC_cell_by_ccre'), ('bam', 'import_scATAC_fragments')])
def test_cross_type_resolution_and_forged_bindings_fail_closed(local, input_type, tool):
    owner, source = local
    record = register(owner, source, input_type=input_type)
    assert_code('LOCAL_RESOURCE_BINDING_INVALID', lambda: owner.resolve(record.resource_id, tool_name=tool))
    forged = RegisteredInput(record.resource_id, record.record_sha256, tool, {})
    assert_code('LOCAL_RESOURCE_BINDING_INVALID', lambda: owner.validate_binding(forged, verify_source=False))
    assert_code('LOCAL_RESOURCE_BINDING_INVALID', lambda: owner.validate_submission(
        dict(registered_input=forged.attribution(), execution_inputs={}), verify_source=False))


@pytest.mark.parametrize('mutation', ['source_path', 'source_sha256', 'missing_path', 'record_digest'])
def test_fragments_forged_source_identity_is_rejected(local, tmp_path, mutation):
    owner, source = local
    record = register(owner, source, input_type='external_fragments')
    binding = owner.resolve(record.resource_id, tool_name='import_scATAC_fragments',
        scientific_inputs=fragments_inputs(tmp_path))
    values = _serialize(binding.execution_inputs)
    if mutation == 'record_digest':
        forged = replace(binding, record_sha256='0' * 64)
    else:
        if mutation == 'missing_path':
            del values['source_path']
        else:
            values[mutation] = str(source.with_name('different')) if mutation == 'source_path' else '0' * 64
        forged = replace(binding, execution_inputs=values)
    assert_code('LOCAL_RESOURCE_BINDING_INVALID', lambda: owner.validate_binding(forged, verify_source=False))


@pytest.mark.parametrize('resource_id,code', [('local-' + '0' * 64, 'LOCAL_RESOURCE_UNAVAILABLE'),
    ('invalid-id', 'LOCAL_RESOURCE_RECORD_INVALID')])
def test_fragments_resolution_requires_existing_exact_identity(local, tmp_path, resource_id, code):
    owner, _ = local
    assert_code(code, lambda: owner.resolve(resource_id, tool_name='import_scATAC_fragments',
        scientific_inputs=fragments_inputs(tmp_path)))


def test_fragments_scientific_values_remain_registry_owned(local, tmp_path):
    owner, source = local
    record = register(owner, source, input_type='external_fragments')
    binding = owner.resolve(record.resource_id, tool_name='import_scATAC_fragments',
        scientific_inputs=fragments_inputs(tmp_path) | dict(source_profile='unreviewed-profile'))
    assert binding.execution_inputs['source_profile'] == 'unreviewed-profile'
    with pytest.raises(ToolArgumentError):
        build_default_tool_registry().validate_arguments(binding.tool_name,
            _serialize(binding.execution_inputs) | dict(output_dir=str(tmp_path / 'out')))
    restricted = LocalResourceAdmission(owner._workspace, registry=ToolRegistry(()))
    assert_code('LOCAL_RESOURCE_BINDING_INVALID', lambda: restricted.resolve(record.resource_id,
        tool_name='import_scATAC_fragments', scientific_inputs=fragments_inputs(tmp_path)))


@pytest.fixture
def registered_bam(bam_factory, tmp_path):
    arguments = bam_factory()
    owner = LocalResourceAdmission(tmp_path / 'workspace', approved_source_roots=(tmp_path,))
    record = owner.register('bam-selection', arguments['source_path'], input_type='bam',
        label='Supplied BAM', attribution='Explicit caller declaration')
    declarations = {key: value for key, value in arguments.items()
                    if key not in {'source_path', 'source_sha256', 'output_dir'}}
    return owner, record, declarations


@pytest.mark.parametrize('tool', ['inspect_raw_scATAC', 'prepare_scATAC_bam_fragments'])
def test_bam_binding_preserves_registry_inputs_without_scientific_execution(registered_bam, monkeypatch, tool):
    owner, record, declarations = registered_bam
    import agent.tools.data._raw_bam as intake
    import agent.tools.data.bam_fragments as producer
    monkeypatch.setattr(intake, 'inspect_bam_inputs', lambda *a, **k: pytest.fail('Resolver inspected BAM.'))
    monkeypatch.setattr(producer, 'prepare_in_stage', lambda *a, **k: pytest.fail('Resolver produced fragments.'))
    if tool == 'inspect_raw_scATAC':
        declarations = dict(species='human', raw_assay='SCATAC', source_genome_assembly='hg38')
        source = dict(raw_input_paths=[record.source_path])
    else:
        source = dict(source_path=record.source_path, source_sha256=record.source_sha256)
    binding = owner.resolve(record.resource_id, tool_name=tool, scientific_inputs=declarations)
    assert owner.validate_binding(binding) == declarations | source
    assert binding.attribution() == dict(resource_id=record.resource_id,
        record_sha256=record.record_sha256, tool_name=tool)
    owner.validate_submission(dict(registered_input=binding.attribution(), execution_inputs=declarations | source))


@pytest.mark.parametrize('missing', ['intake_manifest_path', 'intake_manifest_sha256', 'library_context_path',
    'library_context_sha256', 'reference_bundle_path', 'reference_bundle_sha256', 'source_profile'])
def test_bam_producer_declarations_are_not_guessed(registered_bam, missing):
    owner, record, declarations = registered_bam
    del declarations[missing]
    assert_code('LOCAL_RESOURCE_DECLARATION_REQUIRED', lambda: owner.resolve(record.resource_id,
        tool_name='prepare_scATAC_bam_fragments', scientific_inputs=declarations))


@pytest.mark.parametrize('tool,extra', [('inspect_raw_scATAC', 'raw_input_paths'),
    ('inspect_raw_scATAC', 'source_path'), ('inspect_raw_scATAC', 'source_sha256'),
    ('inspect_raw_scATAC', 'output_dir'), ('prepare_scATAC_bam_fragments', 'source_path'),
    ('prepare_scATAC_bam_fragments', 'source_sha256'), ('prepare_scATAC_bam_fragments', 'raw_input_paths'),
    ('prepare_scATAC_bam_fragments', 'source_index_path'), ('prepare_scATAC_bam_fragments', 'input_spec_path'),
    ('prepare_scATAC_bam_fragments', 'output_dir')])
def test_bam_source_overrides_and_unsupported_inputs_fail_closed(registered_bam, tool, extra):
    owner, record, declarations = registered_bam
    if tool == 'inspect_raw_scATAC':
        declarations = {}
    assert_code('LOCAL_RESOURCE_BINDING_INVALID', lambda: owner.resolve(record.resource_id,
        tool_name=tool, scientific_inputs=declarations | {extra: 'caller-supplied'}))


@pytest.mark.parametrize('tool', ['inspect_raw_scATAC', 'prepare_scATAC_bam_fragments'])
@pytest.mark.parametrize('mutation', ['source', 'missing', 'record_digest'])
def test_bam_forged_source_identity_is_rejected(registered_bam, tool, mutation):
    owner, record, declarations = registered_bam
    binding = owner.resolve(record.resource_id, tool_name=tool,
        scientific_inputs={} if tool == 'inspect_raw_scATAC' else declarations)
    values = _serialize(binding.execution_inputs)
    field = 'raw_input_paths' if tool == 'inspect_raw_scATAC' else 'source_sha256'
    if mutation == 'record_digest':
        forged = replace(binding, record_sha256='0' * 64)
    else:
        if mutation == 'missing':
            del values[field]
        else:
            values[field] = [record.source_path, '/unregistered.bam'] if tool == 'inspect_raw_scATAC' else '0' * 64
        forged = replace(binding, execution_inputs=values)
    assert_code('LOCAL_RESOURCE_BINDING_INVALID', lambda: owner.validate_binding(forged, verify_source=False))


def test_bam_binding_historical_validation_opens_neither_source_nor_intake(registered_bam, monkeypatch):
    owner, record, declarations = registered_bam
    binding = owner.resolve(record.resource_id, tool_name='prepare_scATAC_bam_fragments', scientific_inputs=declarations)
    def forbidden(*args, **kwargs):
        pytest.fail('Historical attribution reopened an input.')
    monkeypatch.setattr(owner, '_source', forbidden)
    monkeypatch.setattr(owner, '_validate_bam_intake', forbidden)
    assert owner.validate_binding(binding, verify_source=False) == _serialize(binding.execution_inputs)
    owner.validate_submission(dict(registered_input=binding.attribution(),
        execution_inputs=_serialize(binding.execution_inputs)), verify_source=False)


def test_bam_registration_preserves_existing_resource_records(local, tmp_path):
    owner, source = local
    old = [owner.register(kind, source, input_type=kind, label=kind, attribution='Existing declaration')
           for kind in ('h5ad', 'external_fragments')]
    raw = {record.resource_id: owner._path(record.resource_id).read_bytes() for record in old}
    bam = owner.register('bam', source, input_type='bam', label='BAM', attribution='New declaration')
    reopened = LocalResourceAdmission(owner._workspace.root)
    assert reopened.load(bam.resource_id) == bam
    for record in old:
        assert reopened.load(record.resource_id) == record
        assert reopened._path(record.resource_id).read_bytes() == raw[record.resource_id]
    assert reopened.resolve(old[0].resource_id, tool_name='inspect_scATAC').execution_inputs['input_path'] == str(source)
    assert reopened.resolve(old[1].resource_id, tool_name='import_scATAC_fragments',
        scientific_inputs=fragments_inputs(tmp_path)).execution_inputs['source_path'] == str(source)


@pytest.mark.parametrize('mutation', ['other_source', 'directory', 'extra_source', 'checksum', 'unpinned', 'corrupt', 'size'])
def test_bam_intake_must_identify_exact_registered_file(registered_bam, bam_factory, tmp_path, mutation):
    owner, record, declarations = registered_bam
    from agent.tools.data import raw_scatac_manifest as intake_owner
    from agent.tools.data.raw_scatac import inspect_raw_scATAC
    declarations = dict(declarations)
    if mutation == 'other_source':
        other = bam_factory()
        declarations.update({key: other[key] for key in ('intake_manifest_path', 'intake_manifest_sha256')})
    elif mutation in {'directory', 'extra_source'}:
        paths = str(Path(record.source_path).parent) if mutation == 'directory' else [
            record.source_path, bam_factory()['source_path']]
        result = inspect_raw_scATAC(paths, tmp_path / 'other-intake',
            species='human', raw_assay='SCATAC', source_genome_assembly='hg38')
        declarations.update(intake_manifest_path=result['manifest_path'], intake_manifest_sha256=result['manifest_sha256'])
    elif mutation == 'checksum':
        declarations['intake_manifest_sha256'] = '0' * 64
    elif mutation == 'unpinned':
        declarations['intake_manifest_sha256'] = None
    elif mutation == 'corrupt':
        Path(declarations['intake_manifest_path']).write_bytes(b'Corrupt intake metadata')
    else:
        _, manifest, _ = intake_owner.load_raw_intake_manifest(declarations['intake_manifest_path'])
        changed = replace(manifest, files=(replace(manifest.files[0], size_bytes=record.size_bytes + 1),))
        result = intake_owner.publish_raw_intake_manifest(changed, tmp_path / 'wrong-size.json')
        declarations.update(intake_manifest_path=result['manifest_path'], intake_manifest_sha256=result['manifest_sha256'])
    assert_code('LOCAL_RESOURCE_BINDING_INVALID', lambda: owner.resolve(record.resource_id,
        tool_name='prepare_scATAC_bam_fragments', scientific_inputs=declarations))


def test_bam_producer_binding_rechecks_intake_before_new_consumption(registered_bam, bam_factory):
    owner, record, declarations = registered_bam
    binding = owner.resolve(record.resource_id, tool_name='prepare_scATAC_bam_fragments', scientific_inputs=declarations)
    other = bam_factory()
    forged = replace(binding, execution_inputs=_serialize(binding.execution_inputs) | {
        key: other[key] for key in ('intake_manifest_path', 'intake_manifest_sha256')})
    assert_code('LOCAL_RESOURCE_BINDING_INVALID', lambda: owner.validate_binding(forged))
    Path(declarations['intake_manifest_path']).write_bytes(b'Intake replaced after resolution')
    assert_code('LOCAL_RESOURCE_BINDING_INVALID', lambda: owner.validate_binding(binding))
