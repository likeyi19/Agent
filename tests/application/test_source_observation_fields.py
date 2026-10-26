"""Source-scoped metadata offers and pure role binding, without scientific work."""
from dataclasses import replace
import hashlib
import json

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from agent.application.local_resources import (
    RegisteredInputGroup, ResourceAdmissionError, source_field_choices, source_field_inputs,
)
from agent.application.turn_decisions import InputRole
from agent.orchestration import build_default_tool_registry
from agent.orchestration.registry import ArtifactSemanticKind, PlanningSourceEligibility


def member(path, position, *, input_type='h5ad'):
    return dict(resource_id='local-' + f'{position:064x}', record_sha256='a' * 64,
                source_path=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                input_type=input_type, label=f'Selected source {position}')


def dataset(path, names):
    obs = pd.DataFrame({name: pd.Categorical(['a', 'b']) for name in names}, index=['c0', 'c1'])
    obs['continuous'] = np.array([1.5, 2.5])
    ad.AnnData(sparse.eye(2, format='csr'), obs=obs).write_h5ad(path)
    return path


def test_shallow_fields_preserve_unicode_and_source_identity_without_array_reads(tmp_path, monkeypatch):
    first = dataset(tmp_path/'first.h5ad', ['Cell Type α', 'shared'])
    second = dataset(tmp_path/'second.h5ad', ['condition', 'shared'])
    binding = RegisteredInputGroup((member(first, 1), member(second, 2)), {})
    monkeypatch.setattr(h5py.Dataset, '__getitem__',
                        lambda *a, **k: pytest.fail('Field discovery read dataset values.'))
    choices = source_field_choices(binding)
    assert choices == [dict(source='source.0', resource_id=binding.members[0]['resource_id'],
                           source_sha256=binding.members[0]['source_sha256'],
                           fields=['Cell Type α', 'shared'], complete=True),
                       dict(source='source.1', resource_id=binding.members[1]['resource_id'],
                           source_sha256=binding.members[1]['source_sha256'],
                           fields=['condition', 'shared'], complete=True)]
    assert str(tmp_path) not in json.dumps(choices)


def test_selected_dataset_role_controls_field_names_and_replay_has_no_io(tmp_path, monkeypatch):
    first = dataset(tmp_path/'reference.h5ad', ['Cell Type α', 'shared'])
    second = dataset(tmp_path/'query.h5ad', ['condition', 'shared'])
    binding = RegisteredInputGroup((member(first, 1), member(second, 2)), {})
    choices = source_field_choices(binding)
    registry = build_default_tool_registry()
    first.unlink(); second.unlink()
    monkeypatch.setattr('agent.tools.analysis.categorical.fields',
                        lambda *a, **k: pytest.fail('Captured field replay reopened a source.'))
    roles = [InputRole('transfer_cell_labels', 'reference_dataset', 'source.1'),
             InputRole('transfer_cell_labels', 'query_dataset', 'source.0'),
             InputRole('evaluate_cell_clustering', 'ground_truth_dataset', 'source.0')]
    assert source_field_inputs(registry, binding, roles, choices) == {
        ('transfer_cell_labels', 'reference_label_key'): ('condition', 'shared'),
        ('evaluate_cell_clustering', 'label_key'): ('Cell Type α', 'shared')}
    assert source_field_inputs(registry, binding, roles[:1], choices) == {
        ('transfer_cell_labels', 'reference_label_key'): ('condition', 'shared')}


def test_private_long_and_unavailable_fields_have_explicit_incomplete_offers(tmp_path):
    source = dataset(tmp_path/'source.h5ad', ['cohort', 'api_key', 'private_path', 'x' * 81])
    broken = tmp_path/'broken.h5ad'; broken.write_bytes(b'not HDF5')
    groups = tmp_path/'groups.tsv'; groups.write_text('cell\tgroup\n')
    binding = RegisteredInputGroup((member(source, 1), member(broken, 2),
                                    member(groups, 3, input_type='groups')), {})
    choices = source_field_choices(binding)
    assert choices[0]['fields'] == ['cohort'] and choices[0]['complete'] is False
    assert all(row['fields'] == [] and row['complete'] is False for row in choices[1:])
    source.unlink()
    assert source_field_choices(binding)[0]['fields'] == []


def test_field_offers_are_bounded_and_do_not_follow_source_symlinks(tmp_path):
    source = dataset(tmp_path/'source.h5ad', [f'category_{i}' for i in range(66)])
    binding = RegisteredInputGroup((member(source, 1),), {})
    choice, = source_field_choices(binding)
    assert len(choice['fields']) == 64 and not choice['complete']
    renamed = source.rename(tmp_path/'relocated.h5ad'); source.symlink_to(renamed)
    choice, = source_field_choices(binding)
    assert choice['fields'] == [] and not choice['complete']


@pytest.mark.parametrize('mutation', ['source', 'resource_id', 'source_sha256', 'private', 'duplicate', 'extra'])
def test_captured_offers_reject_wrong_source_identity_or_untrusted_names(tmp_path, mutation):
    first = dataset(tmp_path/'first.h5ad', ['cohort'])
    second = dataset(tmp_path/'second.h5ad', ['condition'])
    binding = RegisteredInputGroup((member(first, 1), member(second, 2)), {})
    choices = source_field_choices(binding)
    if mutation == 'source': choices[0]['source'] = 'source.1'
    elif mutation == 'resource_id': choices[0]['resource_id'] = binding.members[1]['resource_id']
    elif mutation == 'source_sha256': choices[0]['source_sha256'] = 'b' * 64
    elif mutation == 'private': choices[0]['fields'] = ['credentials']
    elif mutation == 'duplicate': choices[0]['fields'] *= 2
    else: choices[0]['source_path'] = str(first)
    with pytest.raises(ResourceAdmissionError) as rejected:
        source_field_inputs(build_default_tool_registry(), binding,
            [dict(tool='evaluate_cell_clustering', port='ground_truth_dataset', source='source.0')], choices)
    assert rejected.value.code == 'MULTI_INPUT_ROLE_INVALID'


def test_registry_records_source_ownership_without_expanding_registered_ports():
    registry = build_default_tool_registry()
    for tool, argument, port in [
        ('evaluate_cell_clustering', 'label_key', 'ground_truth_dataset'),
        ('transfer_cell_labels', 'reference_label_key', 'reference_dataset'),
        ('evaluate_cell_annotation', 'ground_truth_label_key', 'ground_truth_dataset')]:
        assert registry.get(tool).required_arguments[argument].planning.observation_source == port


@pytest.mark.parametrize('mutation', ['unknown', 'scalar_port', 'numeric_parameter', 'missing_semantics',
                                    'compound_port', 'wrong_artifact', 'static_choices'])
def test_registry_rejects_observation_metadata_without_one_owned_dataset(mutation):
    spec = build_default_tool_registry().get('evaluate_cell_clustering')
    arguments = dict(spec.required_arguments)
    parameter = arguments['label_key']
    semantic = spec.semantic_planning
    if mutation == 'unknown': parameter = replace(parameter, planning=replace(parameter.planning, observation_source='unknown'))
    elif mutation == 'scalar_port': parameter = replace(parameter, planning=replace(parameter.planning, observation_source='output_dir'))
    elif mutation == 'numeric_parameter': parameter = replace(parameter, accepted_types=(int,))
    elif mutation == 'missing_semantics': semantic = None
    elif mutation == 'static_choices': parameter = replace(parameter, choices=('arbitrary',))
    elif mutation == 'wrong_artifact':
        dataset_arg = arguments['reference_h5ad_path']
        arguments['reference_h5ad_path'] = replace(dataset_arg,
            planning=replace(dataset_arg.planning, accepted_artifact_kinds=(ArtifactSemanticKind.CELL_ANNOTATION,)))
    else:
        port = next(p for p in semantic.consumer_ports if p.name == 'ground_truth_dataset')
        new_port = replace(port, members=(port.members[0], replace(port.members[0], name='other', field_name='label_key')))
        semantic = replace(semantic, consumer_ports=tuple(new_port if p is port else p for p in semantic.consumer_ports))
    arguments['label_key'] = parameter
    with pytest.raises(ValueError): replace(spec, required_arguments=arguments, semantic_planning=semantic)


@pytest.mark.parametrize('change', [dict(observation_source='not/a/port'), dict(observation_source=True),
    dict(observation_source='dataset', scientific_parameter=False),
    dict(observation_source='dataset', source_eligibility=PlanningSourceEligibility.UPSTREAM_RESULT_ONLY)])
def test_observation_source_requires_bounded_direct_scientific_metadata(change):
    planning = build_default_tool_registry().get('evaluate_cell_clustering').required_arguments['label_key'].planning
    with pytest.raises(ValueError): replace(planning, **change)
