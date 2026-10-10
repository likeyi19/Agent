"""Exact registered-source admission over ordinary compiled scientific plans."""
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from agent.application.local_resources import (LocalResourceAdmission, ResourceAdmissionError,
    QualifiedEpiZooResource, ResourceConfigurationError, qualified_epizoo_resources, select_epizoo_resource)
from agent.orchestration import build_default_tool_registry
from agent.orchestration.llm_planner import LLMPlanner, PlanningWireMode
from agent.orchestration.semantic_compiler import (
    SemanticPlanCandidate, SemanticPlanStep, SemanticStepOutputSource,
    build_m92_semantic_compiler_contract, compile_semantic_plan,
)
from agent.schemas import AgentRequest
from agent.schemas.orchestration import AgentPlan, PlanStep, StepOutputRef, _serialize


@pytest.fixture
def registered(tmp_path):
    source = tmp_path / 'source.h5ad'
    source.write_bytes(b'registration establishes byte identity, not scientific compatibility')
    registry = build_default_tool_registry()
    owner = LocalResourceAdmission(tmp_path / 'workspace', approved_source_roots=(tmp_path,), registry=registry)
    record = owner.register('selected', source, label='Selected H5AD', attribution='Explicit selection')
    return owner, record, registry


def pins():
    return dict(resource_id='reviewed-model', checkpoint_sha256='a' * 64,
                frequencies_sha256='b' * 64, filter_indices_sha256='c' * 64)


def companions(tmp_path):
    return dict(species='mouse', checkpoint_path=str(tmp_path / 'checkpoint.pth'),
                expected_resource_identity=pins())


def submission(binding):
    return dict(registered_input=binding.attribution(), execution_inputs=_serialize(binding.execution_inputs))


def plan(*steps):
    return AgentPlan('plan', 'request', 'scripted', steps)


def test_composition_preserves_source_identity_and_scoped_registry_inputs(registered, tmp_path):
    owner, record, _ = registered
    declarations = companions(tmp_path) | dict(resolution=0.7, neighbors_random_seed=9,
                                               cluster_random_seed=7, umap_random_seed=6)
    binding = owner.compose_h5ad(record.resource_id, declarations)
    assert binding.resource_id == record.resource_id and binding.record_sha256 == record.record_sha256
    assert binding.execution_inputs['input_path'] == record.source_path
    assert owner.validate_binding(binding) == declarations | dict(input_path=record.source_path)
    declarations['expected_resource_identity']['checkpoint_sha256'] = 'd' * 64
    assert binding.execution_inputs['expected_resource_identity']['checkpoint_sha256'] == 'a' * 64
    assert owner._submission_binding(submission(binding)) == binding
    with pytest.raises(TypeError):
        binding.execution_inputs['species'] = 'human'


@pytest.mark.parametrize('field', ['input_path', 'path', 'source_path', 'source_sha256', 'output_dir',
                                  'reference_input_path', 'embedding_path', 'analysis_path', 'assembly', 'unknown'])
def test_companions_cannot_override_source_or_add_unsupported_inputs(registered, field):
    owner, record, _ = registered
    with pytest.raises(ResourceAdmissionError) as error:
        owner.compose_h5ad(record.resource_id, {field: 'untrusted'})
    assert error.value.code == 'LOCAL_RESOURCE_BINDING_INVALID'


@pytest.mark.parametrize('values', [{'species': 'zebrafish'}, {'resolution': True},
                                   {'neighbors_overwrite': True}, {'cluster_overwrite': True},
                                   {'umap_overwrite': True}, {'embedding_overwrite': True}])
def test_invalid_declarations_and_scoped_overwrite_bypass_are_rejected(registered, values):
    owner, record, _ = registered
    with pytest.raises(ResourceAdmissionError):
        owner.compose_h5ad(record.resource_id, values)


def test_source_bytes_and_pin_conflicts_fail_closed(registered, tmp_path):
    owner, record, _ = registered
    binding = owner.compose_h5ad(record.resource_id, companions(tmp_path))
    with pytest.raises(ResourceAdmissionError):
        owner.compose_h5ad(record.resource_id, companions(tmp_path), resource_selection_error='EPIZOO_RESOURCE_AMBIGUOUS')
    with pytest.raises(ResourceAdmissionError):
        owner.validate_binding(replace(binding, resource_selection_error='EPIZOO_RESOURCE_AMBIGUOUS'))
    with open(record.source_path, 'wb') as stream:
        stream.write(b'changed source')
    with pytest.raises(ResourceAdmissionError) as error:
        owner.validate_binding(binding)
    assert error.value.code == 'LOCAL_RESOURCE_INTEGRITY_INVALID'


@pytest.mark.parametrize('mode', ['direct', 'same_plan'])
def test_existing_compiler_supports_exact_direct_and_same_plan_consumption(registered, tmp_path, mode):
    owner, record, registry = registered
    binding = owner.compose_h5ad(record.resource_id, companions(tmp_path))
    steps = [SemanticPlanStep('embed', 'epizoo_embed_cells')]
    if mode == 'same_plan':
        steps = [SemanticPlanStep('source', 'inspect_scATAC'),
                 SemanticPlanStep('embed', 'epizoo_embed_cells',
                     sources=(SemanticStepOutputSource('dataset', 'source'),))]
    request = AgentRequest('request', 'Embed selected cells.',
                           _serialize(binding.execution_inputs) | dict(output_dir=str(tmp_path / 'new-output')))
    compiled = compile_semantic_plan(request, SemanticPlanCandidate(tuple(steps)), registry,
                                    build_m92_semantic_compiler_contract(registry))
    owner.validate_plan(submission(binding), compiled)


@pytest.mark.parametrize('mode', ['inspection', 'direct', 'wrong_output', 'different_upstream'])
def test_compiled_plan_substitution_is_rejected_before_consumption(registered, tmp_path, mode):
    owner, record, _ = registered
    binding = owner.compose_h5ad(record.resource_id, companions(tmp_path))
    embed = PlanStep('embed', 'epizoo_embed_cells', dict(input_path=record.source_path) | companions(tmp_path))
    if mode == 'inspection':
        candidate = plan(PlanStep('source', 'inspect_scATAC', dict(path=str(tmp_path / 'other.h5ad'))))
    elif mode == 'direct':
        candidate = plan(replace(embed, arguments=dict(embed.arguments) | dict(input_path=str(tmp_path / 'checkpoint.pth'))))
    else:
        upstream = PlanStep('arbitrary-id', 'inspect_scATAC',
                            dict(path=record.source_path if mode == 'wrong_output' else str(tmp_path / 'other.h5ad')))
        candidate = plan(upstream, replace(embed,
            arguments=dict(embed.arguments) | dict(input_path=StepOutputRef(upstream.step_id,
                            'n_cells' if mode == 'wrong_output' else 'input_path')), depends_on=(upstream.step_id,)))
    with pytest.raises(ResourceAdmissionError) as error:
        owner.validate_plan(submission(binding), candidate)
    assert error.value.code == 'H5AD_SOURCE_MISMATCH'


@pytest.mark.parametrize('field', ['species', 'checkpoint_path', 'expected_resource_identity', 'device'])
def test_plan_cannot_change_captured_resource_or_declarations(registered, tmp_path, field):
    owner, record, _ = registered
    values = companions(tmp_path) | dict(device='cuda:0')
    binding = owner.compose_h5ad(record.resource_id, values)
    arguments = values | dict(input_path=record.source_path)
    arguments.pop(field)
    with pytest.raises(ResourceAdmissionError) as error:
        owner.validate_plan(submission(binding), plan(PlanStep('embed', 'epizoo_embed_cells', arguments)))
    assert error.value.code == 'EPIZOO_RESOURCE_CONSUMPTION_MISMATCH'


@pytest.mark.parametrize('code', [None, 'EPIZOO_RESOURCE_REQUIRED', 'EPIZOO_RESOURCE_AMBIGUOUS'])
def test_resource_prerequisites_only_apply_to_actual_embedding(registered, code):
    owner, record, _ = registered
    binding = owner.compose_h5ad(record.resource_id, {'species': 'mouse'}, resource_selection_error=code)
    owner.validate_plan(submission(binding), plan(PlanStep('source', 'inspect_scATAC', dict(path=record.source_path))))
    with pytest.raises(ResourceAdmissionError) as error:
        owner.validate_plan(submission(binding), plan(PlanStep('embed', 'epizoo_embed_cells',
                            dict(input_path=record.source_path, species='mouse'))))
    assert error.value.code == (code or 'EPIZOO_RESOURCE_REQUIRED')


def test_unrelated_operations_are_not_assigned_the_selected_source(registered):
    owner, record, _ = registered
    binding = owner.compose_h5ad(record.resource_id)
    owner.validate_plan(submission(binding), plan(PlanStep('unrelated', 'cluster_cells',
                                                        dict(analysis_path='/explicit/graph.h5ad'))))


def test_legacy_source_only_binding_cannot_authorize_invented_embedding_context(registered):
    owner, record, _ = registered
    binding = owner.resolve(record.resource_id, tool_name='inspect_scATAC')
    with pytest.raises(ResourceAdmissionError) as error:
        owner.validate_plan(submission(binding), plan(PlanStep('embed', 'epizoo_embed_cells',
                            dict(input_path=record.source_path, species='mouse'))))
    assert error.value.code == 'H5AD_SPECIES_REQUIRED'


@pytest.mark.parametrize('selected_input', ['input_path', 'checkpoint_path'])
def test_v3_actual_request_binding_is_checked_without_trusting_path_type(registered, tmp_path, selected_input):
    owner, record, registry = registered
    binding = owner.compose_h5ad(record.resource_id, companions(tmp_path))
    payload = dict(schema_version=3, status='plan', reason=None, steps=[dict(
        step_id='provider-id', tool_name='inspect_scATAC', depends_on=[],
        description='Inspect the selected request input.',
        arguments=dict(path=dict(binding_type='input', input_name=selected_input)))])
    model = SimpleNamespace(model_id='scripted', complete=lambda **kwargs: json.dumps(payload))
    request = AgentRequest('request', 'Inspect selected data.', _serialize(binding.execution_inputs))
    compiled = LLMPlanner(model, wire_mode=PlanningWireMode.V3).plan(request, registry)
    if selected_input == 'input_path':
        owner.validate_plan(submission(binding), compiled)
    else:
        with pytest.raises(ResourceAdmissionError) as error:
            owner.validate_plan(submission(binding), compiled)
        assert error.value.code == 'H5AD_SOURCE_MISMATCH'


def test_first_acceptance_and_unfinished_recovery_check_actual_consumption(registered, tmp_path):
    owner, record, _ = registered
    binding = owner.compose_h5ad(record.resource_id, companions(tmp_path))
    step = SimpleNamespace(tool_name='epizoo_embed_cells',
        resolved_arguments=companions(tmp_path) | dict(input_path=record.source_path),
        result=dict(input_path=record.source_path))
    owner.validate_result(submission(binding), [step])
    step.resolved_arguments['input_path'] = str(tmp_path / 'different.h5ad')
    with pytest.raises(ResourceAdmissionError) as error:
        owner.validate_result(submission(binding), [step])
    assert error.value.code == 'H5AD_SOURCE_MISMATCH'


def reviewed_resource(**changes):
    values = dict(resource_id='reviewed-model', label='Reviewed mouse model', species='mouse',
        checkpoint_path='/operator/checkpoint.pth', checkpoint_sha256='a' * 64,
        frequencies_sha256='b' * 64, filter_indices_sha256='c' * 64,
        qualification='Previously qualified fixture resources.', default=True)
    return QualifiedEpiZooResource(**(values | changes))


def test_application_resource_policy_resolves_completed_declaration_and_preserves_companions(registered):
    owner, record, _ = registered
    resource = reviewed_resource()
    values, error = select_epizoo_resource({'species': 'mouse', 'resolution': 0.7}, None, (resource,))
    assert error is None
    binding = owner.compose_h5ad(record.resource_id, values)
    assert binding.resource_id == record.resource_id and binding.record_sha256 == record.record_sha256
    assert binding.execution_inputs['resolution'] == 0.7
    assert binding.execution_inputs['expected_resource_identity'] == pins()
    assert binding.execution_inputs['input_path'] == record.source_path


@pytest.mark.parametrize('species,resources,code', [
    ('human', (reviewed_resource(),), 'EPIZOO_RESOURCE_REQUIRED'),
    ('mouse', (), 'EPIZOO_RESOURCE_REQUIRED'),
    ('mouse', (reviewed_resource(), reviewed_resource(resource_id='second')), 'EPIZOO_RESOURCE_AMBIGUOUS'),
])
def test_application_resource_policy_never_falls_back_across_species_or_ambiguity(species, resources, code):
    original = {'species': species, 'resolution': 0.7}
    values, error = select_epizoo_resource(original, None, resources)
    assert values == original and error == code
    assert 'checkpoint_path' not in values


@pytest.mark.parametrize('values,selected,code', [
    ({'species': 'human'}, 'reviewed-model', 'EPIZOO_RESOURCE_SELECTION_INVALID'),
    ({'species': 'mouse'}, 'unavailable', 'EPIZOO_RESOURCE_SELECTION_INVALID'),
])
def test_application_resource_policy_rejects_invalid_explicit_selection(values, selected, code):
    with pytest.raises(ResourceAdmissionError) as error:
        select_epizoo_resource(values, selected, (reviewed_resource(),))
    assert error.value.code == code


def test_explicit_qualified_choice_preserves_pins_without_declaring_species(registered):
    owner, record, _ = registered
    selected = reviewed_resource(default=False)
    values, error = select_epizoo_resource({'resolution': 0.7}, selected.resource_id, (selected,))
    assert error is None and 'species' not in values
    assert values == {'resolution': 0.7} | selected.inputs()
    binding = owner.compose_h5ad(record.resource_id, values)
    owner.validate_plan(submission(binding), plan(PlanStep('inspect', 'inspect_scATAC',
        dict(path=record.source_path))))
    with pytest.raises(ResourceAdmissionError) as failed:
        owner.validate_plan(submission(binding), plan(PlanStep('embed', 'epizoo_embed_cells',
            dict(input_path=record.source_path, species='mouse') | selected.inputs())))
    assert failed.value.code == 'H5AD_SPECIES_REQUIRED'


@pytest.mark.parametrize('resources', [[], ('untrusted',), (reviewed_resource(), reviewed_resource())])
def test_application_resource_catalog_is_typed_bounded_and_unique(resources):
    with pytest.raises(ResourceConfigurationError):
        qualified_epizoo_resources(resources)
