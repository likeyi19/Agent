"""Exact configured choice capture adds no scientific qualification or catalog."""
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pytest

from agent.application import InteractiveAgentApplication, InteractiveBoundaryError
from agent.application.local_resources import LocalResourceAdmission, ResourceAdmissionError
from agent.application.session_state import canonical, digest
from agent.schemas.orchestration import AgentPlan, PlanStep, StepOutputRef, _serialize
from agent.web.config import ScientificInputSet


def submission(binding):
    return dict(registered_input=binding.attribution(), execution_inputs=_serialize(binding.execution_inputs))


@pytest.fixture
def selected(tmp_path):
    owner = LocalResourceAdmission(tmp_path / 'workspace', approved_source_roots=(tmp_path,))
    records = {}
    for kind in ('h5ad', 'external_fragments', 'bam'):
        path = tmp_path / (kind + '.source')
        path.write_bytes(b'Byte registration is not scientific qualification: ' + kind.encode())
        records[kind] = owner.register(kind, path, input_type=kind, label=kind,
                                      attribution='Explicit unqualified byte fixture')
    extra = tmp_path / 'second.h5ad'
    extra.write_bytes(b'Another independent byte registration')
    records['second'] = owner.register('second', extra, label='Second',
                                     attribution='Explicit independent unqualified source')
    members = []
    for role in ('R1', 'R2', 'R3'):
        path = tmp_path / ('Library_S1_L001_' + role + '_001.fastq')
        path.write_bytes(b'@read\nACGT\n+\nIIII\n')
        members.append(owner.register(role.lower(), path, input_type='fastq', label=role,
            attribution='Explicit tiny FASTQ source', fastq_member=dict(library_id='Synthetic library',
                fastq_layout='tenx-atac-r1-r2-r3.v1', role=role,
                lane='001', chunk='001', compression='plain')))
    records['fastq'] = owner.register_fastq_collection('library', [m.resource_id for m in members],
        label='Completed synthetic library', attribution='Explicit collection, no producer qualification')
    return owner, records


def fragments_choice(tmp_path, identifier='approved', **changes):
    values = dict(reference_bundle_path=str(tmp_path / 'reference.json'), reference_bundle_sha256='a' * 64,
                  source_profile='10x-atac-fragments.v1', namespace='explicit-fixture')
    return ScientificInputSet(identifier, 'Approved fixture declarations', values,
                              fragments_companion=True, **changes)


def test_definition_identity_captures_every_operator_field_and_is_detached(tmp_path):
    choice = fragments_choice(tmp_path)
    # The absent optional extension must preserve persisted pre-extension digests.
    expected = {name: _serialize(getattr(choice, name)) for name in choice.__dataclass_fields__
                if name != "fastq_whitelist"}
    assert choice.fastq_whitelist is None
    token = choice.resource_choice()
    assert token == dict(input_set_id=choice.input_set_id, definition_sha256=digest(expected))
    assert choice.resource_choice() == token
    assert str(tmp_path) not in str(token) and 'execution_inputs' not in token
    for changed in (replace(choice, input_set_id='another'), replace(choice, label='Another label'),
                    replace(choice, fragments_default=True),
                    replace(choice, execution_inputs=choice.inputs() | {'reference_bundle_sha256': 'b' * 64})):
        assert changed.resource_choice() != token
    token['input_set_id'] = 'changed'
    assert choice.resource_choice()['input_set_id'] == 'approved'


def test_same_explicit_and_default_choice_captures_identical_values_and_identity(selected, tmp_path):
    owner, records = selected
    choice = fragments_choice(tmp_path, fragments_default=True)
    explicit = owner.compose_fragments(records['external_fragments'].resource_id, choice.inputs(),
                                     resource_choice=choice.resource_choice())
    default = owner.compose_fragments(records['external_fragments'].resource_id, choice.inputs(),
                                    resource_choice=choice.resource_choice())
    assert canonical(submission(explicit)) == canonical(submission(default))
    assert owner._submission_binding(submission(explicit)) == explicit
    assert 'resource_choice' not in explicit.execution_inputs
    with pytest.raises(TypeError):
        explicit.resource_choice['input_set_id'] = 'another'
    with pytest.raises(FrozenInstanceError):
        explicit.resource_choice = {}


def test_identical_values_do_not_erase_changed_choice_from_retry_identity(selected, tmp_path):
    owner, records = selected
    first = fragments_choice(tmp_path, 'first')
    second = fragments_choice(tmp_path, 'second')
    original = owner.compose_fragments(records['external_fragments'].resource_id, first.inputs(),
                                      resource_choice=first.resource_choice())
    changed = owner.compose_fragments(records['external_fragments'].resource_id, second.inputs(),
                                     resource_choice=second.resource_choice())
    assert original.execution_inputs == changed.execution_inputs
    assert submission(original) != submission(changed)
    state = SimpleNamespace(interactions=(SimpleNamespace(turn_id='turn', utterance='Prepare this source',
                                                         submission=submission(original)),), turns=())
    assert InteractiveAgentApplication._duplicate(state, 'turn', 'Prepare this source', submission(original))
    for altered in (submission(changed), submission(replace(original, resource_choice=None)),
                    submission(replace(original, resource_choice=first.resource_choice() | {
                        'definition_sha256': 'f' * 64}))):
        with pytest.raises(InteractiveBoundaryError) as rejected:
            InteractiveAgentApplication._duplicate(state, 'turn', 'Prepare this source', altered)
        assert rejected.value.error.code == 'INTERACTIVE_TURN_CONFLICT'


@pytest.mark.parametrize('kind', ['h5ad', 'external_fragments', 'bam', 'fastq', 'group'])
def test_shared_binding_reconstruction_and_role_binding_preserve_choice(selected, tmp_path, kind):
    owner, records = selected
    token = fragments_choice(tmp_path).resource_choice()
    if kind == 'h5ad':
        binding = owner.compose_h5ad(records[kind].resource_id, resource_choice=token)
    elif kind == 'external_fragments':
        binding = owner.compose_fragments(records[kind].resource_id, fragments_choice(tmp_path).inputs(),
                                         resource_choice=token)
    elif kind == 'bam':
        binding = owner.compose_bam(records[kind].resource_id, resource_choice=token)
    elif kind == 'fastq':
        binding = owner.compose_fastq_collection(records[kind].resource_id, resource_choice=token)
    else:
        binding = owner.compose_group([records['h5ad'].resource_id, records['second'].resource_id],
                                      resource_choice=token)
        binding = owner.bind_group(binding, [dict(tool='cluster_cells', port='analysis', source='source.0'),
            dict(tool='evaluate_cell_clustering', port='ground_truth_dataset', source='source.1')])
        assert 'definition_sha256' not in str(owner.group_choices(binding))
    assert _serialize(binding.resource_choice) == token
    assert owner._submission_binding(submission(binding)) == binding
    assert owner.validate_binding(binding) == _serialize(binding.execution_inputs)
    assert 'resource_choice' not in binding.execution_inputs


@pytest.mark.parametrize('malformed', [{}, {'input_set_id': 'approved'},
    {'definition_sha256': 'a' * 64}, {'input_set_id': '../private', 'definition_sha256': 'a' * 64},
    {'input_set_id': 'approved', 'definition_sha256': 'not-a-digest'},
    {'input_set_id': 'approved', 'definition_sha256': 'a' * 64, 'scientific_authority': True},
    {'input_set_id': 'approved', 'definition_sha256': None}, 'approved'])
def test_partial_or_forged_metadata_fails_closed_before_scientific_admission(selected, malformed):
    owner, records = selected
    with pytest.raises(ResourceAdmissionError) as rejected:
        owner.compose_bam(records['bam'].resource_id, resource_choice=malformed)
    assert rejected.value.code == 'LOCAL_RESOURCE_BINDING_INVALID'


def test_absent_metadata_preserves_legacy_attribution_and_explicit_null_is_noncanonical(selected):
    owner, records = selected
    legacy = owner.resolve(records['h5ad'].resource_id, tool_name='inspect_scATAC')
    assert legacy.attribution() == dict(resource_id=records['h5ad'].resource_id,
        record_sha256=records['h5ad'].record_sha256, tool_name='inspect_scATAC')
    assert owner._submission_binding(submission(legacy)) == legacy
    malformed = submission(legacy)
    malformed['registered_input']['resource_choice'] = None
    with pytest.raises(ResourceAdmissionError) as rejected:
        owner._submission_binding(malformed)
    assert rejected.value.code == 'LOCAL_RESOURCE_BINDING_INVALID'


def captured_pipeline(owner, records, tmp_path):
    choice = fragments_choice(tmp_path)
    values = choice.inputs() | dict(qc_reference_manifest_path=str(tmp_path / 'qc.json'),
        qc_reference_manifest_sha256='b' * 64, reference_manifest_path=str(tmp_path / 'reference.json'),
        reference_manifest_sha256='a' * 64, min_qc_fragment_records=1, min_tss_enrichment='1')
    choice = replace(choice, execution_inputs=values)
    binding = owner.compose_fragments(records['external_fragments'].resource_id, values,
                                     resource_choice=choice.resource_choice())
    inputs = _serialize(binding.execution_inputs)
    imported = PlanStep('import', 'import_scATAC_fragments', {key: inputs[key] for key in
        ('source_path', 'source_sha256', 'reference_bundle_path', 'reference_bundle_sha256',
         'source_profile', 'namespace')} | dict(output_dir=str(tmp_path / 'import')))
    qc = PlanStep('qc', 'compute_scATAC_qc', dict(
        fragments_manifest_path=StepOutputRef('import', 'manifest_path'),
        fragments_manifest_sha256=StepOutputRef('import', 'manifest_sha256'),
        qc_reference_manifest_path=inputs['qc_reference_manifest_path'],
        qc_reference_manifest_sha256=inputs['qc_reference_manifest_sha256'],
        output_dir=str(tmp_path / 'qc')), depends_on=('import',))
    selected = PlanStep('select', 'select_scATAC_cells', dict(
        barcode_qc_manifest_path=StepOutputRef('qc', 'manifest_path'),
        barcode_qc_manifest_sha256=StepOutputRef('qc', 'manifest_sha256'),
        min_qc_fragment_records=1, min_tss_enrichment='1', output_dir=str(tmp_path / 'select')),
        depends_on=('qc',))
    matrix = PlanStep('matrix', 'build_scATAC_cell_by_ccre', dict(
        fragments_manifest_path=StepOutputRef('import', 'manifest_path'),
        fragments_manifest_sha256=StepOutputRef('import', 'manifest_sha256'),
        selected_cells_manifest_path=StepOutputRef('select', 'manifest_path'),
        selected_cells_manifest_sha256=StepOutputRef('select', 'manifest_sha256'),
        reference_manifest_path=inputs['reference_manifest_path'],
        reference_manifest_sha256=inputs['reference_manifest_sha256'], output_dir=str(tmp_path / 'matrix')),
        depends_on=('import', 'select'))
    return binding, AgentPlan('plan', 'request', 'scripted', (imported, qc, selected, matrix))


@pytest.mark.parametrize('step_index,field', [(1, 'qc_reference_manifest_path'),
    (1, 'qc_reference_manifest_sha256'), (3, 'reference_manifest_path'), (3, 'reference_manifest_sha256')])
def test_downstream_compiled_and_resolved_consumers_pin_every_resource_group_member(
        selected, tmp_path, step_index, field):
    owner, records = selected
    binding, plan = captured_pipeline(owner, records, tmp_path)
    owner.validate_plan(submission(binding), plan)
    accepted = [SimpleNamespace(tool_name=step.tool_name, resolved_arguments=step.arguments, result={})
                for step in plan.steps]
    owner.validate_result(submission(binding), accepted)
    changed_value = 'f' * 64 if field.endswith('sha256') else str(tmp_path / 'another-approved-resource.json')
    steps = list(plan.steps)
    steps[step_index] = replace(steps[step_index], arguments=dict(steps[step_index].arguments) | {
        field: changed_value})
    with pytest.raises(ResourceAdmissionError) as rejected:
        owner.validate_plan(submission(binding), replace(plan, steps=tuple(steps)))
    assert rejected.value.code == 'LOCAL_RESOURCE_CONSUMPTION_MISMATCH'
    accepted[step_index].resolved_arguments = steps[step_index].arguments
    with pytest.raises(ResourceAdmissionError) as rejected:
        owner.validate_result(submission(binding), accepted)
    assert rejected.value.code == 'LOCAL_RESOURCE_CONSUMPTION_MISMATCH'


def test_resource_association_does_not_replace_upstream_payload_bindings_or_legacy_contract(selected, tmp_path):
    owner, records = selected
    binding, plan = captured_pipeline(owner, records, tmp_path)
    owner.validate_plan(submission(binding), plan)
    assert isinstance(plan.steps[1].arguments['fragments_manifest_path'], StepOutputRef)
    assert isinstance(plan.steps[3].arguments['selected_cells_manifest_path'], StepOutputRef)
    # Absence of the additive metadata retains the previous application guard;
    # the ordinary compiler and scientific owners still own its admission.
    legacy = replace(binding, resource_choice=None)
    steps = list(plan.steps)
    steps[1] = replace(steps[1], arguments=dict(steps[1].arguments) | {
        'qc_reference_manifest_sha256': 'f' * 64})
    owner.validate_plan(submission(legacy), replace(plan, steps=tuple(steps)))
