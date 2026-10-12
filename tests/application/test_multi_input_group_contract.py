"""Closed registered membership and compiled consumer admission, without science."""
from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.application.local_resources import (
    LocalResourceAdmission, RegisteredInputGroup, ResourceAdmissionError,
)
from agent.schemas.orchestration import _serialize, StepOutputRef


@pytest.fixture
def selection(tmp_path):
    sources = tmp_path / 'sources'
    sources.mkdir()
    owner = LocalResourceAdmission(tmp_path / 'workspace', approved_source_roots=(sources,))
    records = []
    for index in range(2):
        source = sources / f'source-{index}.h5ad'
        source.write_bytes(f'Byte registration without scientific qualification {index}'.encode())
        records.append(owner.register(f'selected-{index}', source, label=f'Selected {index + 1}',
            attribution='Explicit independent synthetic source, not scientific proof.'))
    group = owner.compose_group([record.resource_id for record in records])
    return owner, records, group


def roles():
    return [dict(tool='cluster_cells', port='analysis', source='source.0'),
        dict(tool='evaluate_cell_clustering', port='ground_truth_dataset', source='source.1')]


def submission(group):
    return dict(registered_input=group.attribution(), execution_inputs=_serialize(group.execution_inputs))


def plan(group):
    values = _serialize(group.execution_inputs)
    return SimpleNamespace(steps=(
        SimpleNamespace(tool_name='cluster_cells', arguments=dict(analysis_path=values['analysis_path'])),
        SimpleNamespace(tool_name='evaluate_cell_clustering', arguments=dict(
            analysis_path=StepOutputRef('cluster', 'analysis_path'),
            reference_h5ad_path=values['reference_h5ad_path'])),
    ))


def test_order_membership_projection_and_aliases_have_distinct_ownership(selection):
    owner, records, group = selection
    assert not group.roles and not group.execution_inputs
    assert [member['resource_id'] for member in group.members] == [record.resource_id for record in records]
    projected = owner.group_choices(group)
    text = json.dumps(projected)
    for record in records:
        for private in (record.resource_id, record.record_sha256, record.source_sha256, record.source_path):
            assert private not in text
    assert [source['position'] for source in projected['sources']] == [1, 2]
    bound = owner.bind_group(group, roles())
    assert _serialize(bound.execution_inputs) == dict(analysis_path=records[0].source_path,
        reference_h5ad_path=records[1].source_path)
    owner.validate_plan(submission(bound), plan(bound))
    with pytest.raises(FrozenInstanceError):
        bound.members = ()
    with pytest.raises(TypeError):
        bound.members[0]['source_path'] = '/changed'


@pytest.mark.parametrize('alias', ['input_path', 'analysis_path', 'reference_h5ad_path',
                                  'reference_input_path', 'query_input_path', 'output_dir'])
def test_companion_path_cannot_masquerade_as_registered_selection(selection, alias):
    owner, records, _ = selection
    with pytest.raises(ResourceAdmissionError):
        owner.compose_group([record.resource_id for record in records], {alias: records[0].source_path})


def test_distinct_registrations_of_same_locator_cannot_collapse_roles(selection):
    owner, records, _ = selection
    duplicate = owner.register('second-registration', records[0].source_path, label='Another name',
        attribution='Duplicate locator is not an independent selected source.')
    with pytest.raises(ResourceAdmissionError):
        owner.compose_group([records[0].resource_id, duplicate.resource_id])


def test_unassigned_selected_source_and_conflicting_shared_alias_fail_closed(selection):
    owner, _, group = selection
    with pytest.raises(ResourceAdmissionError):
        owner.bind_group(group, roles()[:1])
    with pytest.raises(ResourceAdmissionError):
        owner.bind_group(group, [roles()[0],
            dict(tool='evaluate_cell_clustering', port='analysis', source='source.1')])


@pytest.mark.parametrize('field,value', [('source_path', '/forged/source.h5ad'),
    ('record_sha256', 'f' * 64), ('label', 'Changed identity')])
def test_captured_fields_require_exact_registration_not_path_allowance(selection, field, value):
    owner, _, group = selection
    members = _serialize(group.members)
    members[0][field] = value
    forged = replace(group, members=members)
    with pytest.raises(ResourceAdmissionError):
        owner.validate_binding(forged, verify_source=False)


@pytest.mark.parametrize('kind', ['bam', 'external_fragments', 'fastq'])
def test_independent_group_does_not_replace_source_specific_contract(selection, kind):
    owner, records, _ = selection
    raw = owner.register('raw-' + kind, records[0].source_path, input_type=kind, label='Raw input',
        attribution='Declared kind, with no independent scientific qualification.')
    with pytest.raises(ResourceAdmissionError):
        owner.compose_group([raw.resource_id, records[1].resource_id])


@pytest.mark.parametrize('variant', ['unused', 'substitute', 'foreign_consumer', 'literal', 'repeat'])
def test_exact_authorized_compiled_consumers_are_required(selection, variant):
    owner, records, group = selection
    bound = owner.bind_group(group, roles())
    compiled = plan(bound)
    first, second = compiled.steps
    if variant == 'unused':
        compiled.steps = (first,)
    elif variant == 'substitute':
        first.arguments['analysis_path'] = records[1].source_path
    elif variant == 'foreign_consumer':
        compiled.steps += (SimpleNamespace(tool_name='inspect_scATAC',
            arguments=dict(path=records[1].source_path)),)
    elif variant == 'literal':
        second.arguments['analysis_path'] = '/unregistered/clustered.h5ad'
    else:
        compiled.steps += (SimpleNamespace(tool_name=first.tool_name, arguments=dict(first.arguments)),)
    with pytest.raises(ResourceAdmissionError):
        owner.validate_plan(submission(bound), compiled)


def test_unbound_selection_cannot_authorize_a_plan(selection):
    owner, _, group = selection
    with pytest.raises(ResourceAdmissionError):
        owner.validate_plan(submission(group), SimpleNamespace(steps=()))


def test_completed_fastq_library_is_rejected_as_independent_h5ad_group(selection):
    from test_fastq_composition import register
    owner, records, _ = selection
    members = register(owner, Path(records[0].source_path).parent)
    collection = owner.register_fastq_collection('complete-library',
        [member.resource_id for member in members], label='Complete FASTQ library',
        attribution='Existing explicit source-specific library contract.')
    with pytest.raises(ResourceAdmissionError):
        owner.compose_group([collection.resource_id, records[1].resource_id])
    assert owner.compose_fastq_collection(collection.resource_id).collection_sha256 == collection.collection_sha256
