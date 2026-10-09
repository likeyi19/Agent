"""Typed one-library choices reuse existing inline FASTQ collection identities."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.application.local_resources import (
    LocalResourceAdmission, LocalResourceCollectionRecord, RegisteredInputCollection,
    ResourceAdmissionError,
)
from agent.schemas.orchestration import AgentPlan, PlanStep, StepOutputRef, _serialize
from agent.tools.data import _raw_fastq as owner


def declarations(role='R1', *, layout=owner.FastqLayout.A.value, lane='001', chunk='001', **changes):
    return dict(library_id='Research library', fastq_layout=layout, role=role,
                lane=lane, chunk=chunk, compression='plain') | changes


def register(admission, root, roles=('R1', 'R2', 'R3'), *, layout=owner.FastqLayout.A.value,
             lane='001', chunk='001', **changes):
    records = []
    for role in roles:
        source = root / f'Library_S1_L{lane}_{role}_{chunk}.fastq'
        source.write_bytes(b'@read\nACGT\n+\nIIII\n')
        records.append(admission.register(f'{lane}-{chunk}-{role}', source, input_type='fastq',
            label=f'Display {role}', attribution='Explicit synthetic source',
            fastq_member=declarations(role, layout=layout, lane=lane, chunk=chunk, **changes)))
    return records


@pytest.fixture
def supplied(tmp_path):
    admission = LocalResourceAdmission(tmp_path / 'workspace', approved_source_roots=(tmp_path,))
    records = register(admission, tmp_path)
    collection = admission.register_fastq_collection('library', [record.resource_id for record in records],
        label='Research library', attribution='Explicit typed FASTQ library')
    return admission, records, collection


def submit(binding):
    return dict(registered_input=binding.attribution(), execution_inputs=_serialize(binding.execution_inputs))


def plan(*steps):
    return AgentPlan('plan', 'request', 'scripted', steps)


def inspect(binding, **changes):
    keys = ('raw_input_paths', 'raw_assay', 'fastq_layout', 'species', 'source_genome_assembly')
    return PlanStep('inspect', 'inspect_raw_scATAC',
        {key: _serialize(binding.execution_inputs[key]) for key in keys if key in binding.execution_inputs} | changes)


def context(tmp_path):
    return dict(library_context_path=str(tmp_path / 'library.json'), library_context_sha256='a' * 64,
        reference_bundle_path=str(tmp_path / 'reference.json'), reference_bundle_sha256='b' * 64,
        species='human', source_genome_assembly='hg38')


def prepare(binding, **changes):
    keys = ('library_context_path', 'library_context_sha256',
            'reference_bundle_path', 'reference_bundle_sha256')
    arguments = {key: binding.execution_inputs[key] for key in keys if key in binding.execution_inputs}
    arguments.update(intake_manifest_path=StepOutputRef('inspect', 'manifest_path'),
                     intake_manifest_sha256=StepOutputRef('inspect', 'manifest_sha256'))
    return PlanStep('prepare', 'prepare_scATAC_fragments', arguments | changes, depends_on=('inspect',))


def test_collection_persists_in_same_catalog_and_preserves_inline_fingerprint(supplied):
    admission, records, collection = supplied
    assert isinstance(collection, LocalResourceCollectionRecord)
    assert admission.load(collection.resource_id) == collection
    assert admission.register_fastq_collection('library', [record.resource_id for record in records[::-1]],
        label=collection.label, attribution=collection.attribution) == collection
    legacy = admission.resolve_collection(collection.members, tool_name='inspect_raw_scATAC')
    composed = admission.compose_fastq_collection(collection.resource_id)
    assert legacy.collection_sha256 == composed.collection_sha256 == collection.collection_sha256
    assert composed.composition == 'fastq-science.v1'
    assert admission._submission_binding(submit(composed)) == composed
    assert 'composition' not in legacy.attribution()
    assert set(legacy.attribution()) == {'members', 'collection_sha256', 'tool_name'}
    assert composed.execution_inputs['raw_assay'] == 'TENX_ATAC'
    assert composed.execution_inputs['fastq_layout'] == owner.FastqLayout.A.value
    assert collection.public() == dict(resource_id=collection.resource_id, input_type='fastq',
        label='Research library', status='registered', fastq_collection=True)
    assert not (admission._workspace.root / 'collections').exists()
    assert len(list((admission._workspace.root / 'local_resources').glob('*.json'))) == 4
    admission.validate_plan(submit(composed), plan(inspect(composed)))
    admission.validate_result(submit(composed), [SimpleNamespace(tool_name='inspect_raw_scATAC',
        resolved_arguments=_serialize(inspect(composed).arguments))])


@pytest.mark.parametrize('layout,roles', [(owner.FastqLayout.A.value, ('R1', 'R2', 'R3', 'I1')),
    (owner.FastqLayout.B.value, ('R1', 'R2', 'I2', 'I1'))])
def test_complete_multiple_lane_and_chunk_groups_keep_optional_i1(tmp_path, layout, roles):
    admission = LocalResourceAdmission(tmp_path / 'workspace', approved_source_roots=(tmp_path,))
    records = (register(admission, tmp_path, roles, layout=layout)
        + register(admission, tmp_path, roles, layout=layout, lane='002')
        + register(admission, tmp_path, roles, layout=layout, lane='002', chunk='002'))
    collection = admission.register_fastq_collection('library', [record.resource_id for record in records],
        label='Library', attribution='Explicit synthetic collection')
    assert len(collection.members) == 12
    assert len(admission._fastq_groups(records)) == 3


@pytest.mark.parametrize('roles,layout,missing', [(('R1', 'R3'), owner.FastqLayout.A.value, 'R2'),
    (('R1', 'R2'), owner.FastqLayout.A.value, 'R3'),
    (('R1', 'R2'), owner.FastqLayout.B.value, 'I2')])
def test_owner_reports_exact_missing_read_role_without_partial_choice(tmp_path, roles, layout, missing):
    admission = LocalResourceAdmission(tmp_path / 'workspace', approved_source_roots=(tmp_path,))
    records = register(admission, tmp_path, roles, layout=layout)
    with pytest.raises(ResourceAdmissionError) as error:
        admission.register_fastq_collection('library', [record.resource_id for record in records],
            label='Library', attribution='Explicit')
    assert error.value.code == 'FASTQ_COLLECTION_INCOMPLETE'
    assert f'{missing} (' in error.value.message and 'Lane 001, chunk 001' in error.value.message
    assert len(list((admission._workspace.root / 'local_resources').glob('*.json'))) == len(records)


@pytest.mark.parametrize('changes', [dict(library_id='../private'), dict(role='R4'),
    dict(role='I2'), dict(lane='1'), dict(chunk='0010'), dict(compression='zip'),
    dict(fastq_layout='inferred'), dict(library_id='x' * 65), dict(role=None)])
def test_unsupported_or_untyped_member_declarations_fail_closed(changes):
    with pytest.raises(ResourceAdmissionError):
        LocalResourceAdmission.fastq_member_declaration(declarations(**changes))


@pytest.mark.parametrize('case', ['duplicate', 'foreign_library', 'mixed_layout', 'different_label'])
def test_conflicting_collection_completion_never_rewrites_accepted_catalog(supplied, tmp_path, case):
    admission, records, collection = supplied
    ids = [record.resource_id for record in records]
    before = admission._path(collection.resource_id).read_bytes()
    if case == 'duplicate':
        ids.append(ids[0])
    elif case in ('foreign_library', 'mixed_layout'):
        extra = register(admission, tmp_path, ('R1', 'R2', 'I2') if case == 'mixed_layout' else ('R1', 'R2', 'R3'),
            lane='002', layout=owner.FastqLayout.B.value if case == 'mixed_layout' else owner.FastqLayout.A.value,
            **({'library_id': 'Other library'} if case == 'foreign_library' else {}))
        ids.extend(record.resource_id for record in extra)
    with pytest.raises(ResourceAdmissionError):
        admission.register_fastq_collection('library', ids,
            label='Changed' if case == 'different_label' else collection.label, attribution=collection.attribution)
    assert admission._path(collection.resource_id).read_bytes() == before


@pytest.mark.parametrize('values', [dict(raw_input_paths=['/other.fastq']), dict(source_path='/other.fastq'),
    dict(raw_assay='TENX_ATAC'), dict(fastq_layout=owner.FastqLayout.B.value),
    dict(unknown='unsupported'), dict(min_qc_fragment_records=True), dict(min_tss_enrichment=1.5)])
def test_scientific_companion_cannot_replace_typed_source_or_invent_inputs(supplied, values):
    admission, _, collection = supplied
    with pytest.raises(ResourceAdmissionError) as error:
        admission.compose_fastq_collection(collection.resource_id, values)
    assert error.value.code == 'LOCAL_RESOURCE_BINDING_INVALID'


@pytest.mark.parametrize('keys,code', [(('library_context_path', 'library_context_sha256'),
    'FASTQ_LIBRARY_CONTEXT_REQUIRED'), (('reference_bundle_path', 'reference_bundle_sha256'),
    'FASTQ_REFERENCE_REQUIRED')])
def test_producer_prerequisites_do_not_block_ordinary_inspection(supplied, tmp_path, keys, code):
    admission, _, collection = supplied
    inputs = {key: value for key, value in context(tmp_path).items() if key not in keys}
    binding = admission.compose_fastq_collection(collection.resource_id, inputs)
    admission.validate_plan(submit(binding), plan(inspect(binding)))
    with pytest.raises(ResourceAdmissionError) as error:
        admission.validate_plan(submit(binding), plan(inspect(binding), prepare(binding)))
    assert error.value.code == code


@pytest.mark.parametrize('field', ['raw_input_paths', 'raw_assay', 'fastq_layout', 'species'])
def test_compiled_inspection_requires_exact_declared_members_and_context(supplied, tmp_path, field):
    admission, _, collection = supplied
    binding = admission.compose_fastq_collection(collection.resource_id, context(tmp_path))
    changes = {field: ['/other.fastq'] if field == 'raw_input_paths' else 'other'}
    with pytest.raises(ResourceAdmissionError) as error:
        admission.validate_plan(submit(binding), plan(inspect(binding, **changes)))
    assert error.value.code == 'FASTQ_SOURCE_MISMATCH'


@pytest.mark.parametrize('field', ['library_context_path', 'library_context_sha256',
    'reference_bundle_path', 'reference_bundle_sha256'])
def test_compiled_producer_requires_exact_selected_context(supplied, tmp_path, field):
    admission, _, collection = supplied
    binding = admission.compose_fastq_collection(collection.resource_id, context(tmp_path))
    admission.validate_plan(submit(binding), plan(inspect(binding), prepare(binding)))
    with pytest.raises(ResourceAdmissionError) as error:
        admission.validate_plan(submit(binding), plan(inspect(binding), prepare(binding, **{field: 'other'})))
    assert error.value.code == 'FASTQ_SOURCE_MISMATCH'


@pytest.mark.parametrize('changes', [dict(intake_manifest_sha256=StepOutputRef('other', 'manifest_sha256')),
    dict(intake_manifest_path=StepOutputRef('inspect', 'other')), dict(intake_manifest_sha256='a' * 64)])
def test_compiled_intake_pair_cannot_point_at_other_source(supplied, tmp_path, changes):
    admission, _, collection = supplied
    binding = admission.compose_fastq_collection(collection.resource_id, context(tmp_path))
    prepared = prepare(binding, **changes)
    steps = (inspect(binding), prepared)
    if changes.get('intake_manifest_sha256') == StepOutputRef('other', 'manifest_sha256'):
        steps = (steps[0], replace(inspect(binding), step_id='other'),
                 replace(prepared, depends_on=('inspect', 'other')))
    with pytest.raises(ResourceAdmissionError) as error:
        admission.validate_plan(submit(binding), plan(*steps))
    assert error.value.code == 'FASTQ_SOURCE_MISMATCH'


def test_ambiguous_defaults_remain_operation_specific(supplied):
    admission, _, collection = supplied
    binding = admission.compose_fastq_collection(collection.resource_id,
        resource_selection_error='FASTQ_RESOURCE_AMBIGUOUS')
    admission.validate_plan(submit(binding), plan(inspect(binding)))
    with pytest.raises(ResourceAdmissionError) as error:
        admission.validate_plan(submit(binding), plan(inspect(binding), prepare(binding)))
    assert error.value.code == 'FASTQ_RESOURCE_AMBIGUOUS'


def test_source_drift_blocks_completion_and_new_science_but_keeps_historical_metadata(supplied):
    admission, records, collection = supplied
    binding = admission.compose_fastq_collection(collection.resource_id)
    Path(records[0].source_path).unlink()
    assert admission.compose_fastq_collection(collection.resource_id, verify_source=False) == binding
    with pytest.raises(ResourceAdmissionError) as error:
        admission.validate_plan(submit(binding), plan(inspect(binding)))
    assert error.value.code == 'LOCAL_RESOURCE_INTEGRITY_INVALID'
    with pytest.raises(ResourceAdmissionError):
        admission.register_fastq_collection('other', [record.resource_id for record in records],
            label='Other', attribution='Explicit')
    assert not admission._path(admission.registration_id('other')).exists()


def test_legacy_unattributed_record_persistence_and_serialization_remain_exact(tmp_path):
    admission = LocalResourceAdmission(tmp_path / 'workspace', approved_source_roots=(tmp_path,))
    source = tmp_path / 'legacy.fastq'
    source.write_bytes(b'bytes')
    record = admission.register('legacy', source, input_type='fastq', label='Legacy', attribution='Explicit')
    assert 'fastq_member' not in record.to_dict() and 'source_index' not in record.to_dict()
    stored = json.loads(admission._path(record.resource_id).read_bytes())
    assert set(stored['record']) == {'resource_id', 'input_type', 'label', 'source_path',
                                   'source_sha256', 'size_bytes', 'attribution'}
    assert admission.load(record.resource_id) == record


def test_maximum_existing_collection_members_do_not_exhaust_separate_choice_listing(tmp_path):
    admission = LocalResourceAdmission(tmp_path / 'workspace', approved_source_roots=(tmp_path,))
    records = []
    for lane in range(1, 33):
        records.extend(register(admission, tmp_path, ('R1', 'R2', 'R3', 'I1'), lane=f'{lane:03d}'))
    assert len(records) == 128
    collection = admission.register_fastq_collection('library', [record.resource_id for record in records],
        label='Library', attribution='Explicit synthetic collection')
    assert len(collection.members) == 128
    assert admission.list_records(source_root=tmp_path, listing='choices') == (collection,)
    assert len(admission.list_records(source_root=tmp_path, listing='fastq_members')) == 128
    with pytest.raises(ResourceAdmissionError) as error:
        admission.list_records(source_root=tmp_path)
    assert error.value.code == 'LOCAL_RESOURCE_DISCOVERY_LIMIT'
    # Filtering never bypasses corruption in another catalog record.
    admission._path(records[0].resource_id).write_bytes(b'corrupt')
    with pytest.raises(ResourceAdmissionError):
        admission.list_records(source_root=tmp_path, listing='choices')
