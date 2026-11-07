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


def test_collection_persists_in_same_catalog_and_preserves_inline_fingerprint(supplied):
    admission, records, collection = supplied
    assert isinstance(collection, LocalResourceCollectionRecord)
    assert admission.load(collection.resource_id) == collection
    assert admission.register_fastq_collection('library', [record.resource_id for record in records[::-1]],
        label=collection.label, attribution=collection.attribution) == collection
    legacy = admission.resolve_collection(collection.members, tool_name='inspect_raw_scATAC')
    composed = admission.compose_native(collection.resource_id,{})
    assert legacy.collection_sha256 == composed.collection_sha256 == collection.collection_sha256
    assert composed.composition == 'native-science.v1'
    assert admission._submission_binding(submit(composed)) == composed
    assert 'composition' not in legacy.attribution()
    assert set(legacy.attribution()) == {'members', 'collection_sha256', 'tool_name'}
    assert len(composed.execution_inputs['read1_path']) == len(composed.execution_inputs['read2_path']) == len(composed.execution_inputs['barcode_path']) == 1
    assert collection.public() == dict(resource_id=collection.resource_id, input_type='fastq',
        label='Research library', status='registered', fastq_collection=True)
    assert not (admission._workspace.root / 'collections').exists()
    assert len(list((admission._workspace.root / 'local_resources').glob('*.json'))) == 4


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
        admission.compose_native(collection.resource_id, values)
    assert error.value.code == 'LOCAL_RESOURCE_BINDING_INVALID'


def test_source_drift_blocks_completion_and_new_science_but_keeps_historical_metadata(supplied):
    admission, records, collection = supplied
    binding = admission.compose_native(collection.resource_id,{})
    Path(records[0].source_path).unlink()
    assert admission.compose_native(collection.resource_id,{},verify_source=False) == binding
    with pytest.raises(ResourceAdmissionError) as error:
        admission.validate_binding(binding)
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
