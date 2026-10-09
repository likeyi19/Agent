"""Operation-specific BAM prerequisites and exact selected-source consumption."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.application.local_resources import LocalResourceAdmission, ResourceAdmissionError
from agent.schemas.orchestration import AgentPlan, PlanStep, StepOutputRef, _serialize


@pytest.fixture
def registered(tmp_path):
    path = tmp_path / 'source.bam'
    path.write_bytes(b'registration captures bytes without declaring BAM qualification')
    owner = LocalResourceAdmission(tmp_path / 'workspace', approved_source_roots=(tmp_path,))
    record = owner.register('source', path, input_type='bam', label='Supplied BAM', attribution='Explicit source')
    return owner, record


def context(tmp_path):
    return dict(library_context_path=str(tmp_path / 'library.json'), library_context_sha256='a' * 64,
        reference_bundle_path=str(tmp_path / 'reference.json'), reference_bundle_sha256='b' * 64,
        source_profile='agent-cb-paired-atac.v1', species='human', raw_assay='SCATAC',
        source_genome_assembly='hg38')


def submission(binding):
    return dict(registered_input=binding.attribution(), execution_inputs=_serialize(binding.execution_inputs))


def inspection(record, **changes):
    return PlanStep('inspect', 'inspect_raw_scATAC', dict(raw_input_paths=[record.source_path]) | changes)


def preparation(binding):
    values = _serialize(binding.execution_inputs)
    keys = ('source_path', 'source_sha256', 'source_profile', 'library_context_path',
            'library_context_sha256', 'reference_bundle_path', 'reference_bundle_sha256')
    arguments = {key: values[key] for key in keys if key in values}
    arguments.update(intake_manifest_path=StepOutputRef('inspect', 'manifest_path'),
                     intake_manifest_sha256=StepOutputRef('inspect', 'manifest_sha256'))
    return PlanStep('prepare', 'prepare_scATAC_bam_fragments', arguments, depends_on=('inspect',))


def selected_inspection(record, binding):
    values = binding.execution_inputs
    return inspection(record, **{key: values[key] for key in
        ('species', 'raw_assay', 'source_genome_assembly') if key in values})


def plan(*steps):
    return AgentPlan('plan', 'request', 'scripted', steps)


def test_bare_inspection_does_not_require_or_infer_producer_context(registered):
    owner, record = registered
    binding = owner.compose_bam(record.resource_id)
    assert binding.composition == 'bam-science.v1'
    assert owner.validate_binding(binding) == dict(raw_input_paths=[record.source_path],
        source_path=record.source_path, source_sha256=record.source_sha256)
    owner.validate_plan(submission(binding), plan(inspection(record)))
    accepted = SimpleNamespace(tool_name='inspect_raw_scATAC',
                               resolved_arguments=dict(raw_input_paths=record.source_path))
    owner.validate_result(submission(binding), [accepted])


def test_explicit_context_is_typed_without_selecting_or_completing_a_workflow(registered, tmp_path):
    owner, record = registered
    values = context(tmp_path) | dict(qc_reference_manifest_path=str(tmp_path / 'qc.json'),
        qc_reference_manifest_sha256='c' * 64, min_qc_fragment_records=1, min_tss_enrichment='3/2',
        reference_manifest_path=str(tmp_path / 'reference.json'), reference_manifest_sha256='b' * 64)
    binding = owner.compose_bam(record.resource_id, values)
    assert owner._submission_binding(submission(binding)) == binding
    assert set(binding.execution_inputs) == set(values) | {'raw_input_paths', 'source_path', 'source_sha256'}
    inspect = inspection(record, species='human', raw_assay='SCATAC', source_genome_assembly='hg38')
    owner.validate_plan(submission(binding), plan(inspect, preparation(binding)))


@pytest.mark.parametrize('values', [
    {'source_path': '/other.bam'}, {'source_sha256': 'a' * 64}, {'raw_input_paths': ['/other.bam']},
    {'source_index_path': '/source.bai'}, {'intake_manifest_path': None},
    {'fragments_manifest_path': '/old/fragments.json'}, {'unknown': 1},
    {'min_qc_fragment_records': True}, {'min_tss_enrichment': 1.5},
    {'source_profile': 'agent-cb-paired-atac.primary-neutral.v1'},
])
def test_overrides_unsupported_inputs_and_profiles_fail_closed(registered, values):
    owner, record = registered
    with pytest.raises(ResourceAdmissionError) as error:
        owner.compose_bam(record.resource_id, values)
    assert error.value.code == 'LOCAL_RESOURCE_BINDING_INVALID'


@pytest.mark.parametrize('fields,code', [
    (('library_context_path', 'library_context_sha256'), 'BAM_LIBRARY_CONTEXT_REQUIRED'),
    (('reference_bundle_path', 'reference_bundle_sha256'), 'BAM_REFERENCE_REQUIRED'),
    (('source_profile',), 'BAM_PROFILE_REQUIRED'),
])
def test_missing_context_is_required_only_for_actual_preparation(registered, tmp_path, fields, code):
    owner, record = registered
    values = {key: value for key, value in context(tmp_path).items() if key not in fields}
    binding = owner.compose_bam(record.resource_id, values)
    inspect = inspection(record, species='human', raw_assay='SCATAC', source_genome_assembly='hg38')
    owner.validate_plan(submission(binding), plan(inspect))
    with pytest.raises(ResourceAdmissionError) as error:
        owner.validate_plan(submission(binding), plan(inspect, preparation(binding)))
    assert error.value.code == code


@pytest.mark.parametrize('field', ['source_path', 'source_sha256', 'source_profile',
    'library_context_path', 'library_context_sha256', 'reference_bundle_path', 'reference_bundle_sha256'])
def test_compiled_preparation_consumes_exact_source_and_selected_context(registered, tmp_path, field):
    owner, record = registered
    binding = owner.compose_bam(record.resource_id, context(tmp_path))
    prepared = preparation(binding)
    wrong = replace(prepared, arguments=dict(prepared.arguments) | {field: 'changed'})
    with pytest.raises(ResourceAdmissionError) as error:
        owner.validate_plan(submission(binding), plan(selected_inspection(record, binding), wrong))
    assert error.value.code == 'BAM_SOURCE_MISMATCH'


@pytest.mark.parametrize('arguments', [dict(raw_input_paths=['/other.bam']),
    dict(raw_input_paths=['/other.bam', '/another.bam']), dict(raw_input_paths=[]),
    dict(raw_input_paths=StepOutputRef('other', 'path'))])
def test_inspection_cannot_substitute_or_add_other_sources(registered, arguments):
    owner, record = registered
    binding = owner.compose_bam(record.resource_id)
    steps = (PlanStep('inspect', 'inspect_raw_scATAC', arguments),)
    if isinstance(arguments['raw_input_paths'], StepOutputRef):
        steps = (PlanStep('other', 'inspect_scATAC', dict(path='/other.h5ad')),
                 PlanStep('inspect', 'inspect_raw_scATAC', arguments, depends_on=('other',)))
    with pytest.raises(ResourceAdmissionError) as error:
        owner.validate_plan(submission(binding), plan(*steps))
    assert error.value.code == 'BAM_SOURCE_MISMATCH'
    with pytest.raises(ResourceAdmissionError) as error:
        owner.validate_result(submission(binding), [SimpleNamespace(tool_name='inspect_raw_scATAC',
                                                                   resolved_arguments=arguments)])
    assert error.value.code == 'BAM_SOURCE_MISMATCH'


@pytest.mark.parametrize('changes', [
    dict(intake_manifest_sha256=StepOutputRef('other', 'manifest_sha256')),
    dict(intake_manifest_path=StepOutputRef('inspect', 'other')),
    dict(intake_manifest_sha256='a' * 64),
])
def test_intake_handoff_requires_one_exact_selected_inspection_pair(registered, tmp_path, changes):
    owner, record = registered
    binding = owner.compose_bam(record.resource_id, context(tmp_path))
    prepared = preparation(binding)
    wrong = replace(prepared, arguments=dict(prepared.arguments) | changes)
    steps = (selected_inspection(record, binding), wrong)
    if isinstance(changes.get('intake_manifest_sha256'), StepOutputRef) and changes['intake_manifest_sha256'].step_id == 'other':
        other = replace(selected_inspection(record, binding), step_id='other')
        steps = (steps[0], other, replace(wrong, depends_on=('inspect', 'other')))
    with pytest.raises(ResourceAdmissionError) as error:
        owner.validate_plan(submission(binding), plan(*steps))
    assert error.value.code == 'BAM_SOURCE_MISMATCH'


def test_no_intake_or_unrelated_plan_cannot_bypass_source_binding(registered, tmp_path):
    owner, record = registered
    binding = owner.compose_bam(record.resource_id, context(tmp_path))
    prepared = preparation(binding)
    arguments = dict(prepared.arguments) | dict(intake_manifest_path='/unselected.json', intake_manifest_sha256='a' * 64)
    with pytest.raises(ResourceAdmissionError) as error:
        owner.validate_plan(submission(binding), plan(replace(prepared, arguments=arguments, depends_on=())))
    assert error.value.code == 'BAM_INTAKE_REQUIRED'
    with pytest.raises(ResourceAdmissionError) as error:
        owner.validate_plan(submission(binding), plan(PlanStep('other', 'inspect_scATAC', dict(path='/other.h5ad'))))
    assert error.value.code == 'BAM_SOURCE_MISMATCH'


def test_ambiguous_default_does_not_prevent_inspection_or_invent_a_context(registered):
    owner, record = registered
    binding = owner.compose_bam(record.resource_id, resource_selection_error='BAM_RESOURCE_AMBIGUOUS')
    owner.validate_plan(submission(binding), plan(inspection(record)))
    with pytest.raises(ResourceAdmissionError) as error:
        owner.validate_plan(submission(binding), plan(inspection(record), preparation(binding)))
    assert error.value.code == 'BAM_RESOURCE_AMBIGUOUS'
    with pytest.raises(ResourceAdmissionError):
        owner.compose_bam(record.resource_id, resource_selection_error='EPIZOO_RESOURCE_REQUIRED')


def test_changed_source_blocks_new_science_but_metadata_reads_survive(registered):
    owner, record = registered
    binding = owner.compose_bam(record.resource_id)
    Path(record.source_path).write_bytes(b'changed')
    with pytest.raises(ResourceAdmissionError) as error:
        owner.validate_plan(submission(binding), plan(inspection(record)))
    assert error.value.code == 'LOCAL_RESOURCE_INTEGRITY_INVALID'
    assert owner.validate_binding(binding, verify_source=False) == _serialize(binding.execution_inputs)
