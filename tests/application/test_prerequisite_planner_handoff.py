"""The UA3.5.4a decision boundary reaches existing compiler-owned clarification.

Semantic responses are scripted witnesses, not live-language qualification.
Original scientific artifacts need no execution to diagnose missing declarations.
Real continuation/owner validation remains covered by the existing UA3.5.3 suites.
"""
import hashlib
import json

import pytest

from agent.application import InteractiveAgentApplication
from agent.orchestration import PlanningModelProfile, PlanningRecoveryPolicy
from agent.orchestration.executor import PlanExecutor
from agent.orchestration.planning_recovery import SCOPED_PLANNING_RECOVERY_POLICY_VERSION
from agent.orchestration.registry import build_default_tool_registry
from agent.orchestration.semantic_compiler import (
    build_m92_semantic_compiler_contract, compile_semantic_plan,
)
from agent.orchestration.semantic_prompt import build_semantic_planning_catalog
from agent.orchestration.semantic_wire_v4 import parse_semantic_wire_v4
from agent.providers import PlanningModelFactoryRegistry
from agent.schemas import AgentRequest


PROFILE = PlanningModelProfile('handoff-test', 'scripted', 'model/handoff')
POLICY = PlanningRecoveryPolicy(
    policy_version=SCOPED_PLANNING_RECOVERY_POLICY_VERSION,
    max_transport_retries=0, max_repairs=0, max_profile_failovers=0,
    max_primary_local_recovery_actions=0, max_total_provider_calls=2,
    max_retry_delay_seconds=0,
)
CASES = (
    ('A1', 'Use this dataset to run the supported EpiZoo analysis.',
     'epizoo_embed_cells', 'embedding_analysis', ('species',),
     'The EpiZoo embedding step requires an explicit species choice (human or mouse), '
     'but the request does not specify one and no offered tool can derive it for the required species input.'),
    ('A3', 'Analyze this dataset with the available EpiZoo model.',
     'epizoo_embed_cells', 'embedding_analysis', ('species',),
     "EpiZoo embedding requires the dataset species ('human' or 'mouse'), but the request does not "
     'specify it and no available tool can determine or supply that required parameter.'),
    ('C1', 'Select cells from these barcode QC results.',
     'select_scATAC_cells', 'raw_preprocessing', ('min_qc_fragment_records', 'min_tss_enrichment'),
     'Cell selection requires explicit min_qc_fragment_records and min_tss_enrichment thresholds, '
     'but neither was provided and the catalog forbids inferring them.'),
)


def incomplete(tool, *, omit_artifact=False):
    port, input_name = (('dataset', 'input_path') if tool == 'epizoo_embed_cells'
                        else ('barcode_qc', 'barcode_qc_manifest_path'))
    sources = [] if omit_artifact else [dict(target=port, source=dict(kind='input', input=input_name))]
    if tool == 'epizoo_embed_cells':
        sources.append(dict(target='device', source=dict(kind='input', input='device')))
    sources.append(dict(target='output_dir', source=dict(kind='input', input='output_dir')))
    return dict(schema_version=4, decision=dict(kind='plan', steps=[
        dict(step_id='requested', tool=tool, sources=sources, control_dependencies=[])]))


class Witness:
    model_id = PROFILE.model_id

    def __init__(self, tool, family, detailed):
        self.tool, self.family, self.detailed, self.calls = tool, family, detailed, []

    def complete(self, *, prompt, response_schema):
        value = json.loads(prompt)
        self.calls.append(value)
        if 'turn_schema_version' in value:
            return json.dumps(dict(turn_schema_version=1,
                decision=dict(kind='execute_plan', target=self.tool)))
        if 'selection_schema_version' in value:
            return json.dumps(dict(selection_schema_version=1,
                decision=dict(kind='select', capability_ids=[self.family])))
        assert 'semantic_prompt_version' in value, 'No output selection or answer is admitted.'
        return json.dumps(self.detailed)

    def close(self):
        pass


class NoScience(PlanExecutor):
    def execute(self, *args, **kwargs):
        raise AssertionError('An incomplete or unsupported candidate must never execute.')


def setup(tmp_path, tool, family, detailed, *, provide_artifact=True):
    registry = build_default_tool_registry()
    model = Witness(tool, family, detailed)
    app = InteractiveAgentApplication(tmp_path / 'workspace',
        model_profiles=(PROFILE,), default_profile_id=PROFILE.profile_id,
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': lambda p: model}),
        registry=registry, executor=NoScience(registry), planning_recovery_policy=POLICY,
        approved_source_roots=(tmp_path,))
    app.create_session('session')
    if not provide_artifact:
        inputs = dict(device='cuda') if tool == 'epizoo_embed_cells' else {}
        return app, model, dict(execution_inputs=inputs), inputs
    if tool == 'epizoo_embed_cells':
        # Byte registration is provisional identity, not execution qualification.
        from anndata import AnnData
        from scipy.sparse import csr_matrix
        source = tmp_path / 'source.h5ad'
        AnnData(csr_matrix([[1, 0], [0, 1]])).write_h5ad(source)
        record = app.resources.register('source', source, label='Registered H5AD', attribution='Test selection')
        binding = app.resources.compose_h5ad(record.resource_id, dict(device='cuda'),
            resource_selection_error='EPIZOO_RESOURCE_REQUIRED')
        kwargs = dict(registered_input=binding)
        inputs = dict(binding.execution_inputs)
    else:
        # A structural input context only: no QC authority or scientific result is fabricated.
        source = tmp_path / 'qc-context.json'
        source.write_text('{"structural_context_only":true}\n')
        inputs = dict(barcode_qc_manifest_path=str(source),
            barcode_qc_manifest_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        kwargs = dict(execution_inputs=inputs)
    return app, model, kwargs, inputs


@pytest.mark.parametrize('case,text,tool,family,missing,reason', CASES)
@pytest.mark.parametrize('decision', ['recorded_unsupported', 'supported_incomplete'])
def test_recorded_failure_and_supported_candidate_have_distinct_authoritative_outcomes(
        tmp_path, case, text, tool, family, missing, reason, decision):
    detailed = (dict(schema_version=4, decision=dict(kind='unsupported', reason=reason))
                if decision == 'recorded_unsupported' else incomplete(tool))
    app, model, kwargs, inputs = setup(tmp_path, tool, family, detailed)
    view = app.submit_turn('session', 'initial', text, expected_generation=0, **kwargs)
    state = app._application.sessions.load('session')
    interaction = state.interactions[-1]
    run = app._application.run_store.load(view.run_id)
    assert len(model.calls) == 3
    assert not run.steps and not state.revisions and state.generation == 0
    assert not state.turns and run.plan is None
    assert dict(interaction.admitted['inputs']) == inputs
    assert interaction.utterance == text
    catalog = build_semantic_planning_catalog(AgentRequest(case, text, inputs), app._application.registry)
    for port in missing:
        specification = catalog['tools'][tool][1][port]
        assert specification[0] is True and specification[1] == 'none_available'
        assert not specification[2]
    instructions = model.calls[-1]['instructions']
    assert any('required explicit user declarations' in r for r in instructions)
    assert any('no execution permission' in r for r in instructions)
    if decision == 'recorded_unsupported':
        assert view.status == 'failed' and view.error.code == 'UNSUPPORTED_REQUEST'
        assert interaction.prerequisite is None
    else:
        assert view.status == 'clarification'
        error = run.errors[-1]
        assert error.code == 'MISSING_REQUIRED_SOURCE'
        assert error.details['tool_name'] == tool
        assert tuple(error.details['argument_names']) == missing
        if tool == 'epizoo_embed_cells':
            assert view.response.clarification['reason'] == 'missing_species'
            assert interaction.prerequisite['field'] == 'species'
        else:
            assert view.response.clarification['reason'] == 'missing_parameter_value'
            assert tuple(view.response.clarification['choices']) == missing
            assert tuple(interaction.prerequisite['missing']) == missing
        assert interaction.prerequisite['origin_turn_id'] == 'initial'


@pytest.mark.parametrize('case,text,tool,family,missing,reason', CASES)
def test_missing_required_artifact_is_not_converted_to_user_parameter_clarification(
        tmp_path, case, text, tool, family, missing, reason):
    app, model, kwargs, _ = setup(tmp_path, tool, family, incomplete(tool, omit_artifact=True),
                                 provide_artifact=False)
    view = app.submit_turn('session', 'initial', text, expected_generation=0, **kwargs)
    state = app._application.sessions.load('session')
    assert view.status == 'failed'
    assert state.interactions[-1].prerequisite is None
    assert not state.revisions
    assert not app._application.run_store.load(view.run_id).steps


@pytest.mark.parametrize('mutation', ['unknown_source', 'invalid_graph'])
def test_incomplete_candidate_permission_does_not_weaken_source_or_graph_validation(tmp_path, mutation):
    registry = build_default_tool_registry()
    request = AgentRequest('negative', CASES[0][1], dict(input_path='/explicit/source.h5ad',
        device='cuda', output_dir=str(tmp_path)))
    response = incomplete('epizoo_embed_cells')
    if mutation == 'unknown_source':
        response['decision']['steps'][0]['sources'][0]['source']['input'] = 'fabricated_input'
    else:
        response['decision']['steps'][0]['control_dependencies'] = ['requested']
    from agent.orchestration import PlannerError
    with pytest.raises(PlannerError) as caught:
        candidate = parse_semantic_wire_v4(json.dumps(response), request, registry)
        compile_semantic_plan(request, candidate, registry, build_m92_semantic_compiler_contract(registry))
    assert caught.value.code != 'MISSING_REQUIRED_SOURCE'


def test_required_reference_resource_absence_is_not_numeric_parameter_pending(tmp_path):
    tool = 'prepare_scATAC_fragments'
    response = dict(schema_version=4, decision=dict(kind='plan', steps=[
        dict(step_id='requested', tool=tool, sources=[], control_dependencies=[])]))
    app, _, _, _ = setup(tmp_path, tool, 'raw_preprocessing', response, provide_artifact=False)
    inputs = dict(intake_manifest_path='/explicit/intake.json', intake_manifest_sha256='a' * 64,
        library_context_path='/explicit/library.json', library_context_sha256='b' * 64)
    view = app.submit_turn('session', 'initial', 'Prepare fragments with the declared intake and library.',
        expected_generation=0, execution_inputs=inputs)
    state = app._application.sessions.load('session')
    run = app._application.run_store.load(view.run_id)
    assert view.status == 'failed' and state.interactions[-1].prerequisite is None
    assert run.errors[-1].code == 'MISSING_REQUIRED_SOURCE'
    assert set(run.errors[-1].details['argument_names']) == {'reference_bundle_path', 'reference_bundle_sha256'}
    assert not run.steps and not state.revisions


def test_named_backend_instruction_preserves_planner_rejection_even_for_wrong_interpreter_target(tmp_path):
    reason = ('The catalog does not offer EpiAgent; it only offers EpiZoo cell embedding. '
              'In addition, the required species (human or mouse) is not specified.')
    response = dict(schema_version=4, decision=dict(kind='unsupported', reason=reason))
    app, model, kwargs, _ = setup(tmp_path, 'epizoo_embed_cells', 'embedding_analysis', response)
    view = app.submit_turn('session', 'initial', 'Use EpiAgent to embed the cells in this dataset.',
        expected_generation=0, **kwargs)
    rules = model.calls[0]['instructions']
    identity_rule, = [r for r in rules if r.startswith('An explicitly named scientific backend')]
    assert 'if unavailable, use clarify/unsupported_intent' in identity_rule
    assert 'general operation without a named implementation' in identity_rule
    assert 'EpiAgent' not in identity_rule and 'EpiZoo' not in identity_rule
    assert view.status == 'failed' and view.error.code == 'UNSUPPORTED_REQUEST'
    state = app._application.sessions.load('session')
    assert not state.revisions and state.interactions[-1].prerequisite is None
