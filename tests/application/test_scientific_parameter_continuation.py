"""Conversational bindings run real, independently verified tiny selection.

Only the existing model interface is scripted. References, fragment adoption,
QC, scientific owners, compiler, persistence and application execution are real.
Synthetic qualification is explicit and makes no biological-quality claim.
"""
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from agent.application import InteractiveAgentApplication, InteractiveBoundaryError
from agent.orchestration import AgentRequest, LLMPlanner, PlanningModelProfile, build_default_tool_registry
from agent.orchestration.planner import PlannerError
from agent.orchestration.runtime import _planner_error_details
from agent.providers import PlanningModelFactoryRegistry
from agent.schemas.orchestration import _serialize
from agent.tools.data.scatac_barcode_qc import compute_scATAC_qc

sys.path.insert(0, str(Path(__file__).parents[1]))
from barcode_qc.conftest import runtime, fixture_factory, qc_case


TOOL = 'select_scATAC_cells'
REQUIRED = ('min_qc_fragment_records', 'min_tss_enrichment')
OBJECTIVE = 'Select candidate cells from the supplied verified barcode QC.'
BOTH = 'Require at least 1 QC fragment record and TSS enrichment of at least 0.5.'
DEPTH = 'Require at least 1 QC fragment record.'
TSS = 'Require TSS enrichment of at least 0.5.'
BAD = 'Require TSS enrichment of at least -1.'
QUESTION = 'Which selection thresholds were used, and which did I specify?'
QUESTION_FIELDS = ('effective_thresholds.min_qc_fragment_records',
    'effective_thresholds.min_tss_enrichment.numerator',
    'effective_thresholds.min_tss_enrichment.denominator',
    'selection_threshold_origins.min_tss_flank_evidence',
    'explicit_user_parameters.parameters.min_qc_fragment_records',
    'explicit_user_parameters.parameters.min_tss_enrichment')
PROFILE = PlanningModelProfile('selection-scripted', 'scripted', 'scripted/selection')


def argument(utterance, name, token, **changes):
    start = utterance.index(token)
    return dict(tool=TOOL, argument=name, literal=token,
                start=start, end=start + len(token)) | changes


def answer(utterance=BOTH, declarations=None):
    if declarations is None:
        declarations = [argument(utterance, REQUIRED[0], '1'),
                        argument(utterance, REQUIRED[1], '0.5')]
    return dict(kind='answer_parameters', pending='@parameters', arguments=declarations)


class SelectionModel:
    """Ordinary interpreter, scope selection and semantic-v4 planning interface."""

    def __init__(self, decisions, calls, *, plan_variant=None):
        self.decisions, self.calls, self.plan_variant = decisions, calls, plan_variant
        self.model_id = PROFILE.model_id

    def complete(self, *, prompt, response_schema):
        value = json.loads(prompt)
        self.calls.append(value)
        if 'turn_schema_version' in value:
            return json.dumps(dict(turn_schema_version=1,
                decision=self.decisions[value['utterance']]))
        if 'selection_schema_version' in response_schema.get('properties', {}):
            return json.dumps(dict(selection_schema_version=1,
                decision=dict(kind='select', capability_ids=['raw_preprocessing'])))
        if 'output_selection_schema_version' in value:
            return json.dumps(dict(outputs=[dict(name='selection', step_id='selection',
                output_key='manifest_path')]))
        if 'dialogue_schema_version' in value:
            claims = {claim['field']: claim['claim_id'] for claim in value['evidence']['claims']}
            return json.dumps(dict(support='supported', paragraphs=[dict(parts=[
                dict(kind='text', text='The accepted selection records:'),
                *(dict(kind='claim', id=claims[field]) for field in QUESTION_FIELDS)])]))
        steps = [dict(step_id='selection', tool=TOOL,
            sources=[dict(target='barcode_qc', source=dict(kind='input',
                input='barcode_qc_manifest_path'))], control_dependencies=[])]
        if self.plan_variant == 'repeated':
            steps.append(deepcopy(steps[0]) | dict(step_id='other_selection'))
        return json.dumps(dict(schema_version=4, decision=dict(kind='plan', steps=steps)))


@dataclass
class SelectionHarness:
    service: InteractiveAgentApplication
    decisions: dict
    calls: list
    constructions: list


def selection_harness(workspace, *, decisions=None, plan_variant=None, registry=None):
    choices = {OBJECTIVE: dict(kind='execute_plan', target=TOOL),
        BOTH: answer(), DEPTH: answer(DEPTH, [argument(DEPTH, REQUIRED[0], '1')]),
        TSS: answer(TSS, [argument(TSS, REQUIRED[1], '0.5')]),
        BAD: answer(BAD, [argument(BAD, REQUIRED[1], '-1')]),
        QUESTION: dict(kind='answer_scientific', target=dict(output='r0', subject=None),
            comparison=None, focus='question')} | (decisions or {})
    calls, constructions = [], []

    def factory(profile):
        model = SelectionModel(choices, calls, plan_variant=plan_variant)
        constructions.append(model)
        return model

    service = InteractiveAgentApplication(workspace, registry=registry,
        model_profiles=(PROFILE,), default_profile_id=PROFILE.profile_id,
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': factory}))
    return SelectionHarness(service, choices, calls, constructions)


def legacy_selection_registry():
    """Create pre-USI-3C pending records using the former required-port contract.

    Scientific implementation and all subsequent persisted-record validation use
    current code. This is a bounded test fixture, never a production registry.
    """
    from agent.orchestration import ToolRegistry
    registry = build_default_tool_registry()
    spec = registry.get(TOOL)
    optional = dict(spec.optional_arguments)
    required = dict(spec.required_arguments)
    for name in REQUIRED:
        required[name] = optional.pop(name)
    semantic = replace(spec.semantic_planning, consumer_ports=tuple(
        replace(port, required=True) if port.name in REQUIRED else port
        for port in spec.semantic_planning.consumer_ports))
    old = replace(spec, required_arguments=required, optional_arguments=optional,
                  semantic_planning=semantic)
    return ToolRegistry(tuple(old if name == TOOL else registry.get(name)
                              for name in registry.names()))


def legacy_selection_harness(workspace, **kwargs):
    return selection_harness(workspace, registry=legacy_selection_registry(), **kwargs)


@pytest.fixture
def selection_inputs(qc_case):
    verified = compute_scATAC_qc(**qc_case[0])
    assert verified['resource_qualification'] == 'synthetic_only'
    return dict(barcode_qc_manifest_path=verified['manifest_path'],
        barcode_qc_manifest_sha256=verified['manifest_sha256'])


@contextmanager
def selection_work():
    """Observe real production and the undecorated independent verifier body."""
    from agent.tools.data import _cell_selection_production as production
    from agent.tools.data import cell_selection_verifier as verification
    codes = {production.produce.__code__: 'production',
             verification.verify_cell_selection.__wrapped__.__code__: 'verification'}
    previous = sys.getprofile()
    calls = Counter()

    def observe(frame, event, arg):
        if event == 'call' and frame.f_code in codes:
            calls[codes[frame.f_code]] += 1

    sys.setprofile(observe)
    try:
        yield calls
    finally:
        sys.setprofile(previous)


def submit(harness, turn='first', utterance=OBJECTIVE, inputs=None, generation=0):
    return harness.service.submit_turn('session', turn, utterance,
        expected_generation=generation, execution_inputs=inputs).to_dict()


def assert_pending(view, *missing):
    assert view['response']['kind'] == 'clarify', view
    assert view['response']['clarification']['reason'] == 'missing_parameter_value', view
    assert set(view['response']['clarification']['choices']) == set(missing), view
    assert view['response']['clarification']['value_required']
    assert view['revision_id'] is None and not view['steps']


def accepted_step(harness, view, *, depth=1, tss='0.5'):
    assert view['status'] == 'succeeded' and view['response']['status'] == 'activated', view
    run = harness.service._application.run_store.load(view['run_id'])
    assert len(run.steps) == 1 and run.steps[0].tool_name == TOOL
    step = run.steps[0]
    assert step.verification.passed and step.verification.artifact_authority['schema_version'] == 2
    assert step.resolved_arguments['min_qc_fragment_records'] == depth
    assert step.resolved_arguments['min_tss_enrichment'] == tss
    assert 'min_tss_flank_evidence' not in step.resolved_arguments
    assert step.result['resource_qualification'] == 'synthetic_only'
    manifest = json.loads(Path(step.result['manifest_path']).read_bytes())
    assert manifest['arguments']['min_tss_flank_evidence'] is None
    assert manifest['thresholds']['min_tss_flank_evidence'] is None
    return step, manifest


def test_selection_requiredness_and_owner_defaults_are_registry_authority():
    spec = build_default_tool_registry().get(TOOL)
    parameters = {key: value for key, value in
        (*spec.required_arguments.items(), *spec.optional_arguments.items())
        if value.planning.scientific_parameter}
    assert not set(parameters).intersection(spec.required_arguments)
    assert {key: value.planning.default_when_omitted for key, value in
        spec.optional_arguments.items()} == dict(min_qc_fragment_records=1000, min_tss_enrichment=4, min_tss_flank_evidence=None,
            max_qc_fragment_records=None, max_nucleosome_signal=None)
    for name, value in [('min_qc_fragment_records', -1), ('min_tss_enrichment', '-1'),
                        ('min_tss_enrichment', .5), ('min_qc_fragment_records', True)]:
        with pytest.raises(ValueError):
            parameters[name].validate(name, value)


def test_missing_argument_identities_survive_existing_planner_diagnostic_transport():
    model = SelectionModel({}, [])
    request = AgentRequest('missing-thresholds', OBJECTIVE,
        dict(barcode_qc_manifest_path='/declared/qc.json', barcode_qc_manifest_sha256='1' * 64,
            output_dir='/declared/output'))
    with pytest.raises(PlannerError) as error:
        LLMPlanner(model).plan(request, legacy_selection_registry())
    assert error.value.code == 'MISSING_REQUIRED_SOURCE'
    details = _planner_error_details(error.value)
    assert details['tool_name'] == TOOL and details['step_id'] == 'selection'
    assert tuple(details['argument_names']) == REQUIRED
    assert error.value.diagnostics[-1].argument_names == REQUIRED
    # The metadata is a list of Registry identities, never parsed error prose or
    # a declaration of scientific input compatibility.
    assert '/declared' not in json.dumps(details)


def test_historical_selection_clarification_then_executes_true_science(
        tmp_path, selection_inputs):
    harness = legacy_selection_harness(tmp_path / 'application')
    harness.service.create_session('session')
    with selection_work() as calls:
        pending = submit(harness, inputs=selection_inputs)
    assert_pending(pending, *REQUIRED)
    assert not calls and not harness.service._application.sessions.load('session').revisions
    assert 'species' not in pending['response']['clarification']['choices']
    with selection_work() as calls:
        completed = submit(harness, 'answer', BOTH)
    assert calls == {'production': 1, 'verification': 1}
    step, manifest = accepted_step(harness, completed)
    assert step.result['n_selected'] >= 1
    assert manifest['thresholds']['min_tss_enrichment'] == dict(numerator=1, denominator=2)
    interaction = harness.service._application.sessions.load('session').interactions[-1]
    assert interaction.admitted['inputs']['barcode_qc_manifest_sha256'] == selection_inputs['barcode_qc_manifest_sha256']
    assert harness.service._application.run_store.load(completed['run_id']).request.prompt == OBJECTIVE
    parameter_prompt = next(call for call in harness.calls if call.get('utterance') == BOTH)
    assert str(tmp_path) not in json.dumps(parameter_prompt)
    assert parameter_prompt['dialogue']['pending_prerequisite']['objective'] == OBJECTIVE
    evidence = harness.service.evidence('session', completed['revision_id'], 'selection')
    facts = {fact.field: fact for fact in evidence.facts}
    effective = facts['effective_thresholds']
    assert effective.value['min_qc_fragment_records'] == 1
    assert effective.value['min_tss_enrichment']['numerator'] == 1
    assert facts['selection_threshold_origins'].value['min_tss_flank_evidence'] == 'existing_owner_default'
    assert facts['selection_threshold_origins'].value['min_tss_enrichment'] == 'explicit_execution_argument'
    assert _serialize(facts['explicit_user_parameters'].value) == dict(tool=TOOL,
        step_id='selection', parameters={name: 'explicit_user_instruction' for name in REQUIRED})
    assert facts['explicit_user_parameters'].artifact_name == 'accepted_session_interactions'
    with selection_work() as calls:
        response = submit(harness, 'parameters', QUESTION, generation=1)
    assert response['response']['status'] == 'answered', response
    assert not calls
    assert response['response']['scientific']['support'] == 'supported'
    assert response['response']['scientific']['targets'] == [dict(
        revision_id=completed['revision_id'], output_name='selection', subject=None)]
    prompt = next(call for call in harness.calls if 'dialogue_schema_version' in call)
    claims = {claim['field']: claim for claim in prompt['evidence']['claims']}
    assert claims[QUESTION_FIELDS[0]]['value'] == 1
    assert claims[QUESTION_FIELDS[1]]['value'] == 1 and claims[QUESTION_FIELDS[2]]['value'] == 2
    assert claims[QUESTION_FIELDS[3]]['value'] == 'existing_owner_default'
    assert all(claims[field]['value'] == 'explicit_user_instruction' for field in QUESTION_FIELDS[4:])


def test_initial_explicit_thresholds_use_the_same_binding_without_clarification(
        tmp_path, selection_inputs):
    request = OBJECTIVE + ' ' + BOTH
    declarations = [argument(request, REQUIRED[0], '1'), argument(request, REQUIRED[1], '0.5')]
    harness = selection_harness(tmp_path / 'application', decisions={request:
        dict(kind='execute_plan', target=TOOL, arguments=declarations)})
    harness.service.create_session('session')
    with selection_work() as calls:
        result = submit(harness, utterance=request, inputs=selection_inputs)
    assert calls == {'production': 1, 'verification': 1}
    accepted_step(harness, result)
    assert len(harness.service._application.sessions.load('session').interactions) == 1


def test_partial_answer_restart_invalid_attempt_and_retry_preserve_accepted_values(
        tmp_path, selection_inputs):
    workspace = tmp_path / 'application'
    harness = legacy_selection_harness(workspace)
    harness.service.create_session('session')
    initial = submit(harness, inputs=selection_inputs)
    assert_pending(initial, *REQUIRED)
    with selection_work() as calls:
        partial = submit(harness, 'partial', DEPTH)
    assert_pending(partial, REQUIRED[1])
    assert not calls
    original = harness.service._application.sessions.load('session').to_dict()
    restarted = selection_harness(workspace)
    assert restarted.service.turn('session', 'first').to_dict() == initial
    assert restarted.service.turn('session', 'partial').to_dict() == partial
    assert restarted.service._application.sessions.load('session').to_dict() == original
    assert not restarted.calls and not restarted.constructions
    with selection_work() as calls:
        invalid = submit(restarted, 'invalid', BAD)
    assert invalid['response']['kind'] == 'clarify', invalid
    assert invalid['response']['clarification']['reason'] == 'invalid_parameter_value', invalid
    assert not calls and not restarted.service._application.sessions.load('session').revisions
    with selection_work() as calls:
        done = submit(restarted, 'remaining', TSS)
    assert calls == {'production': 1, 'verification': 1}
    accepted_step(restarted, done)
    evidence = restarted.service.evidence('session', done['revision_id'], 'selection')
    origin = next(fact for fact in evidence.facts if fact.field == 'explicit_user_parameters')
    assert dict(origin.value['parameters']) == {name: 'explicit_user_instruction' for name in REQUIRED}
    calls_before = len(restarted.calls), len(restarted.constructions)
    snapshot = restarted.service._application.sessions.load('session').to_dict()
    with selection_work() as calls:
        duplicate = submit(restarted, 'remaining', TSS)
    assert duplicate == done and not calls
    assert calls_before == (len(restarted.calls), len(restarted.constructions))
    assert restarted.service._application.sessions.load('session').to_dict() == snapshot


def test_pending_correction_replaces_only_its_validated_scoped_parameter(
        tmp_path, selection_inputs):
    correction = 'Use at least 2 QC fragment records and TSS enrichment of at least 0.5.'
    decisions = {correction: answer(correction, [argument(correction, REQUIRED[0], '2'),
        argument(correction, REQUIRED[1], '0.5')])}
    harness = legacy_selection_harness(tmp_path / 'application', decisions=decisions)
    harness.service.create_session('session')
    submit(harness, inputs=selection_inputs)
    assert_pending(submit(harness, 'partial', DEPTH), REQUIRED[1])
    with selection_work() as calls:
        completed = submit(harness, 'corrected', correction)
    assert calls == {'production': 1, 'verification': 1}
    accepted_step(harness, completed, depth=2)


def test_invalid_multi_parameter_correction_is_atomic(tmp_path, selection_inputs):
    invalid = 'Use at least 2 QC fragment records and TSS enrichment of at least -1.'
    harness = legacy_selection_harness(tmp_path / 'application', decisions={invalid:
        answer(invalid, [argument(invalid, REQUIRED[0], '2'), argument(invalid, REQUIRED[1], '-1')])})
    harness.service.create_session('session')
    submit(harness, inputs=selection_inputs)
    assert_pending(submit(harness, 'partial', DEPTH), REQUIRED[1])
    with selection_work() as calls:
        rejected = submit(harness, 'invalid-correction', invalid)
    assert not calls
    assert rejected['response']['clarification']['reason'] == 'invalid_parameter_value', rejected
    with selection_work() as calls:
        completed = submit(harness, 'remaining', TSS)
    assert calls == {'production': 1, 'verification': 1}
    accepted_step(harness, completed, depth=1)


def test_partial_threshold_continuation_executes_after_actual_process_restart(
        tmp_path, selection_inputs):
    workspace = tmp_path / 'application'
    harness = legacy_selection_harness(workspace)
    harness.service.create_session('session')
    submit(harness, inputs=selection_inputs)
    assert_pending(submit(harness, 'partial', DEPTH), REQUIRED[1])
    script = '''
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'tests/application'))
from test_scientific_parameter_continuation import (
    selection_harness, selection_work, submit, assert_pending, accepted_step, REQUIRED, TSS)
harness = selection_harness(sys.argv[1])
assert_pending(harness.service.turn('session', 'partial').to_dict(), REQUIRED[1])
assert not harness.calls and not harness.constructions
with selection_work() as calls:
    result = submit(harness, 'fresh-process', TSS)
assert calls == {'production': 1, 'verification': 1}, calls
accepted_step(harness, result)
counts = len(harness.calls), len(harness.constructions)
with selection_work() as calls:
    assert submit(harness, 'fresh-process', TSS) == result
assert not calls and counts == (len(harness.calls), len(harness.constructions))
print(json.dumps(result))
'''
    child = subprocess.run([sys.executable, '-c', script, str(workspace)],
        capture_output=True, text=True, env=os.environ.copy(), timeout=90)
    assert child.returncode == 0, child.stdout + child.stderr
    completed = json.loads(child.stdout)
    accepted_step(harness, completed)


@pytest.mark.parametrize('changes', [dict(tool='unknown_tool'), dict(argument='unregistered_threshold'),
    dict(tool='cluster_cells', argument='resolution'), dict(literal='2'), dict(start=0, end=1)])
def test_invalid_or_unauthorized_parameter_answer_has_zero_science(
        tmp_path, selection_inputs, changes):
    declarations = [argument(DEPTH, REQUIRED[0], '1', **changes)]
    harness = legacy_selection_harness(tmp_path / 'application', decisions={DEPTH: answer(DEPTH, declarations)})
    harness.service.create_session('session')
    submit(harness, inputs=selection_inputs)
    with selection_work() as calls:
        invalid = submit(harness, 'invalid', DEPTH)
    assert invalid['response']['kind'] == 'clarify', invalid
    assert not calls and not harness.service._application.sessions.load('session').revisions
    with selection_work() as calls:
        completed = submit(harness, 'valid', BOTH)
    assert calls == {'production': 1, 'verification': 1}
    accepted_step(harness, completed)


def test_ambiguous_repeated_operation_never_binds_to_first_consumer(tmp_path, selection_inputs):
    request = OBJECTIVE + ' ' + BOTH
    harness = selection_harness(tmp_path / 'application', plan_variant='repeated', decisions={request:
        dict(kind='execute_plan', target=TOOL, arguments=[argument(request, REQUIRED[0], '1'),
            argument(request, REQUIRED[1], '0.5')])})
    harness.service.create_session('session')
    with selection_work() as calls:
        result = submit(harness, utterance=request, inputs=selection_inputs)
    assert result['revision_id'] is None and result['response']['status'] != 'activated'
    # Optional scalar inputs are rejected by the compiler before a candidate
    # reaches the application's declaration-scope guard. No first consumer wins.
    assert result['response']['kind'] == 'execute'
    assert result['response']['status'] == 'failed'
    assert result['response']['error']['code'] == 'AMBIGUOUS_OPTIONAL_INPUT_SCOPE'
    assert not calls and not harness.service._application.sessions.load('session').revisions
    counts = len(harness.calls), len(harness.constructions)
    with selection_work() as calls:
        duplicate = submit(harness, utterance=request, inputs=selection_inputs)
    assert duplicate == result and not calls
    assert counts == (len(harness.calls), len(harness.constructions))


def test_conflicting_typed_and_semantic_values_do_not_gain_authority(tmp_path, selection_inputs):
    request = OBJECTIVE + ' ' + BOTH
    harness = selection_harness(tmp_path / 'application', decisions={request:
        dict(kind='execute_plan', target=TOOL, arguments=[argument(request, REQUIRED[0], '1'),
            argument(request, REQUIRED[1], '0.5')])})
    harness.service.create_session('session')
    with selection_work() as calls:
        view = submit(harness, utterance=request, inputs=selection_inputs | dict(min_qc_fragment_records=2))
    assert view['response']['kind'] == 'clarify'
    assert view['response']['clarification']['reason'] == 'conflicting_scientific_parameter'
    assert not calls


def test_stale_generation_rejects_before_provider_and_preserves_pending(tmp_path, selection_inputs):
    harness = legacy_selection_harness(tmp_path / 'application')
    harness.service.create_session('session')
    submit(harness, inputs=selection_inputs)
    state = harness.service._application.sessions.load('session').to_dict()
    counts = len(harness.calls), len(harness.constructions)
    with selection_work() as calls, pytest.raises(InteractiveBoundaryError) as error:
        submit(harness, 'stale', BOTH, generation=1)
    assert error.value.error.code == 'INTERACTIVE_GENERATION_CONFLICT'
    assert not calls and counts == (len(harness.calls), len(harness.constructions))
    assert harness.service._application.sessions.load('session').to_dict() == state


def test_changed_input_cannot_redirect_pending_threshold_answer(tmp_path, selection_inputs):
    harness = legacy_selection_harness(tmp_path / 'application')
    harness.service.create_session('session')
    submit(harness, inputs=selection_inputs)
    with selection_work() as calls:
        changed = submit(harness, 'changed', BOTH,
            inputs=selection_inputs | dict(barcode_qc_manifest_sha256='f' * 64))
    assert changed['response']['kind'] == 'clarify', changed
    assert changed['response']['clarification']['reason'] == 'invalid_prerequisite'
    assert not calls and not harness.service._application.sessions.load('session').revisions


def test_independent_missing_qc_is_not_relabelled_as_missing_thresholds(tmp_path):
    harness = selection_harness(tmp_path / 'application')
    harness.service.create_session('session')
    with selection_work() as calls:
        result = submit(harness)
    assert not calls and result['revision_id'] is None
    assert result['response']['status'] == 'failed', result
    assert result['response']['clarification'] is None


def test_new_task_does_not_inherit_old_pending_parameter(tmp_path, selection_inputs):
    original = OBJECTIVE + ' ' + DEPTH
    fresh = 'Start a new selection with the supplied complete explicit thresholds.'
    harness = legacy_selection_harness(tmp_path / 'application', decisions={original:
        dict(kind='execute_plan', target=TOOL, arguments=[argument(original, REQUIRED[0], '1')]),
        fresh: dict(kind='execute_plan', target=TOOL)})
    harness.service.create_session('session')
    assert_pending(submit(harness, utterance=original, inputs=selection_inputs), REQUIRED[1])
    with selection_work() as calls:
        completed = submit(harness, 'fresh', fresh,
            inputs=selection_inputs | dict(min_qc_fragment_records=3, min_tss_enrichment='0.5'))
    accepted_step(harness, completed, depth=3)
    assert calls == {'production': 1, 'verification': 1}


# The string cases reuse the same pending mechanism and existing owner fixture;
# transfer/annotation placeholders below qualify semantic admission only.
from test_string_scientific_execution import (
    CLUSTER_TOOL, CLUSTER_KEY, STRING_ARGUMENTS, STRING_OBJECTIVE, STRING_LITERAL,
    STRING_ANSWER, accepted_evaluation, evaluation_work, semantic_string_inputs,
    string_answer, string_argument, string_harness, string_inputs, submit_string,
)
from test_dialogue_evidence import app


@pytest.mark.parametrize('tool', tuple(STRING_ARGUMENTS))
def test_missing_reviewed_string_uses_existing_compiler_grounded_pending_shape(tmp_path, string_inputs, tool):
    inputs = string_inputs if tool == CLUSTER_TOOL else semantic_string_inputs(tool)
    harness = string_harness(tmp_path / 'workspace', tool=tool)
    harness.service.create_session('session')
    with evaluation_work() as calls:
        pending = submit_string(harness, inputs=inputs)
    assert_pending(pending, STRING_ARGUMENTS[tool])
    assert not calls and not harness.service._application.sessions.load('session').revisions
    state = harness.service._application.sessions.load('session')
    interaction, = state.interactions
    assert interaction.prerequisite['field'] == 'parameters'
    assert interaction.prerequisite['tool'] == tool
    assert tuple(interaction.prerequisite['missing']) == (STRING_ARGUMENTS[tool],)
    assert interaction.admitted['inputs'] == inputs
    run = harness.service._application.run_store.load(interaction.admitted['request_id'] + ':run')
    assert not run.steps
    assert run.errors[-1].details['argument_names'] == (STRING_ARGUMENTS[tool],)


def test_exact_string_pending_restart_executes_original_objective_once_and_pins_provenance(tmp_path, string_inputs):
    workspace = tmp_path / 'workspace'
    harness = string_harness(workspace)
    harness.service.create_session('session')
    with evaluation_work() as calls:
        pending = submit_string(harness, inputs=string_inputs)
    assert_pending(pending, CLUSTER_KEY)
    assert not calls
    snapshot = harness.service._application.sessions.load('session').to_dict()
    restarted = string_harness(workspace)
    with evaluation_work() as calls:
        assert restarted.service.turn('session', 'first').to_dict() == pending
        assert restarted.service._application.sessions.load('session').to_dict() == snapshot
    assert not calls and not restarted.calls and not restarted.constructions
    with evaluation_work() as calls:
        done = submit_string(restarted, 'answer', STRING_ANSWER)
    assert calls['production'] == 1 and calls['verification'] >= 1
    assert calls['source_reinspection'] == calls['verification'] + 1
    accepted_evaluation(restarted, done, string_inputs)
    run = restarted.service._application.run_store.load(done['run_id'])
    assert run.request.prompt == STRING_OBJECTIVE
    state = restarted.service._application.sessions.load('session')
    first, resumed = state.interactions
    assert first.submission['execution_inputs'] == string_inputs
    assert not resumed.submission['execution_inputs']
    assert _serialize(resumed.admitted['arguments']) == [string_argument(STRING_ANSWER)]
    assert resumed.admitted['continuation'] == dict(origin_turn_id='first', binding_turn_id='first', field='parameters')
    evidence = restarted.service.evidence('session', done['revision_id'], 'evaluation')
    fact = next(item for item in evidence.facts if item.field == 'explicit_user_parameters')
    assert _serialize(fact.value) == dict(tool=CLUSTER_TOOL, step_id='evaluation',
        parameters={CLUSTER_KEY: 'explicit_user_instruction'})
    counts = len(restarted.calls), len(restarted.constructions)
    with evaluation_work() as calls:
        assert submit_string(restarted, 'answer', STRING_ANSWER) == done
    assert calls['production'] == 0 and counts == (len(restarted.calls), len(restarted.constructions))
    again = string_harness(workspace)
    with evaluation_work() as calls:
        assert again.service.turn('session', 'answer').to_dict() == done
        after = again.service.evidence('session', done['revision_id'], 'evaluation')
    assert calls['production'] == 0 and not again.calls and not again.constructions
    origin_after = next(item for item in after.facts if item.field == 'explicit_user_parameters')
    assert origin_after == fact


def test_exact_string_answer_survives_an_actual_fresh_process(tmp_path, string_inputs):
    workspace = tmp_path / 'workspace'
    harness = string_harness(workspace)
    harness.service.create_session('session')
    assert_pending(submit_string(harness, inputs=string_inputs), CLUSTER_KEY)
    script = '''
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'tests/application'))
from test_string_scientific_execution import (STRING_ANSWER, string_harness, submit_string,
    evaluation_work, accepted_evaluation, string_argument)
from agent.schemas.orchestration import _serialize
harness = string_harness(sys.argv[1])
assert harness.service.turn('session', 'first').to_dict()['response']['kind'] == 'clarify'
assert not harness.calls and not harness.constructions
with evaluation_work() as calls:
    result = submit_string(harness, 'fresh-process', STRING_ANSWER)
assert calls['production'] == 1 and calls['verification'] >= 1, calls
assert calls['source_reinspection'] == calls['verification'] + 1, calls
accepted_evaluation(harness, result, json.loads(sys.argv[2]))
interaction = harness.service._application.sessions.load('session').interactions[-1]
assert _serialize(interaction.admitted['arguments']) == [string_argument(STRING_ANSWER)]
counts = len(harness.calls), len(harness.constructions)
with evaluation_work() as calls:
    assert submit_string(harness, 'fresh-process', STRING_ANSWER) == result
assert calls['production'] == 0 and counts == (len(harness.calls), len(harness.constructions))
print(json.dumps(result))
'''
    child = subprocess.run([sys.executable, '-c', script, str(workspace), json.dumps(string_inputs)],
        capture_output=True, text=True, env=os.environ.copy(), timeout=90)
    assert child.returncode == 0, child.stdout + child.stderr
    accepted_evaluation(harness, json.loads(child.stdout), string_inputs)


def test_transfer_numeric_partial_and_mixed_invalid_string_correction_remain_atomic(tmp_path):
    tool = 'transfer_cell_labels'
    partial = 'Use 2 reference neighbors.'
    corrected = 'Correct this to 3 reference neighbors.'
    invalid = f'Use the reference label column "{STRING_LITERAL}" and 2.5 reference neighbors.'
    final = f'Use the reference label column "{STRING_LITERAL}".'
    numerical = lambda text, token: string_argument(text, token, tool=tool, argument='n_neighbors')
    decisions = {partial: string_answer(partial, [numerical(partial, '2')]),
        corrected: string_answer(corrected, [numerical(corrected, '3')]),
        invalid: string_answer(invalid, [string_argument(invalid, tool=tool), numerical(invalid, '2.5')]),
        final: string_answer(final, [string_argument(final, tool=tool)])}
    workspace = tmp_path / 'workspace'
    harness = string_harness(workspace, tool=tool, decisions=decisions)
    harness.service.create_session('session')
    assert_pending(submit_string(harness, inputs=semantic_string_inputs(tool)), 'reference_label_key')
    assert_pending(submit_string(harness, 'partial', partial), 'reference_label_key')
    with evaluation_work() as calls:
        rejected = submit_string(harness, 'invalid', invalid)
    assert not calls and rejected['response']['clarification']['reason'] == 'invalid_parameter_value', rejected
    state = harness.service._application.sessions.load('session')
    assert state.interactions[-1].prerequisite['binding_turn_id'] == 'partial'
    assert state.interactions[1].admitted['inputs']['n_neighbors'] == 2
    assert 'reference_label_key' not in state.interactions[1].admitted['inputs']
    assert_pending(submit_string(harness, 'corrected', corrected), 'reference_label_key')
    restarted = string_harness(workspace, tool=tool, decisions=decisions)
    assert restarted.service._application.sessions.load('session').interactions[-1].admitted['inputs']['n_neighbors'] == 3
    with evaluation_work() as calls:
        result = submit_string(restarted, 'final', final)
    assert not calls and result['revision_id'] is None
    # Semantic completion resumes the captured objective with both exact values.
    # The unqualified placeholder sources fail owner admission, as intended.
    state = restarted.service._application.sessions.load('session')
    admitted = state.interactions[-1].admitted
    assert admitted['kind'] == 'execute'
    assert admitted['inputs']['reference_label_key'] == STRING_LITERAL
    assert admitted['inputs']['n_neighbors'] == 3
    run = restarted.service._application.run_store.load(result['run_id'])
    assert run.request.prompt == STRING_OBJECTIVE
    assert len(run.plan.steps) == 1 and run.plan.steps[0].tool_name == tool
    assert run.plan.steps[0].arguments['reference_label_key'] == STRING_LITERAL
    assert run.plan.steps[0].arguments['n_neighbors'] == 3
    assert result['response']['status'] == 'failed'
    counts = len(restarted.calls), len(restarted.constructions)
    assert submit_string(restarted, 'final', final) == result
    assert counts == (len(restarted.calls), len(restarted.constructions))


@pytest.mark.parametrize('failure', ['source', 'stale_generation', 'wrong_operation', 'repeated'])
def test_exact_string_pending_cannot_change_source_generation_or_operation(tmp_path, string_inputs, failure):
    harness = string_harness(tmp_path / 'workspace', plan_variant='repeated' if failure == 'repeated' else None)
    harness.service.create_session('session')
    assert_pending(submit_string(harness, inputs=string_inputs), CLUSTER_KEY)
    if failure == 'source':
        changes = dict(inputs=string_inputs | dict(reference_h5ad_path='/other/truth.h5ad'))
    elif failure == 'wrong_operation':
        harness.decisions[STRING_ANSWER] = string_answer(STRING_ANSWER, [string_argument(
            STRING_ANSWER, tool='evaluate_cell_annotation')])
        changes = {}
    elif failure == 'stale_generation':
        changes = dict(generation=1)
    else:
        changes = {}
    counts = len(harness.calls), len(harness.constructions)
    with evaluation_work() as calls:
        if failure == 'stale_generation':
            with pytest.raises(InteractiveBoundaryError) as error:
                submit_string(harness, 'invalid', STRING_ANSWER, **changes)
            assert error.value.error.code == 'INTERACTIVE_GENERATION_CONFLICT'
            assert counts == (len(harness.calls), len(harness.constructions))
        else:
            rejected = submit_string(harness, 'invalid', STRING_ANSWER, **changes)
            assert rejected['response']['kind'] == 'clarify', rejected
            assert rejected['revision_id'] is None
            assert rejected['response']['clarification']['reason'] == (
                'invalid_prerequisite' if failure == 'source' else 'ambiguous_parameter')
    assert not calls and not harness.service._application.sessions.load('session').revisions


def test_missing_nonstring_scientific_source_does_not_become_a_column_question(tmp_path, string_inputs):
    harness = string_harness(tmp_path / 'workspace')
    harness.service.create_session('session')
    with evaluation_work() as calls:
        view = submit_string(harness, inputs={'analysis_path': string_inputs['analysis_path']})
    assert not calls and view['response']['status'] == 'failed', view
    assert view['response']['clarification'] is None
    assert harness.service._application.sessions.load('session').interactions[0].prerequisite is None


def test_selected_guidance_candidate_missing_string_retains_existing_failure_contract(app, string_inputs):
    from agent.application import ResearchAgentApplication
    from test_dialogue_evidence import accepted
    from test_guidance_selection import Selection, offer
    from test_scientific_guidance import ask
    from test_string_scientific_execution import StringModel

    accepted(app, 'inspect_scATAC')  # Existing accepted metadata fixture, not science.
    candidate = offer(app, tools=(CLUSTER_TOOL,))
    model = StringModel({}, [])
    continued = ResearchAgentApplication(app.workspace_root, planner=LLMPlanner(model))
    with evaluation_work() as calls:
        result = ask(continued, 'selected', 'Run option 1.', Selection(candidate, 'Run option 1.'),
            execution_inputs=string_inputs)
    assert not calls and result.status == 'failed'
    assert result.clarification is None
    state = continued.sessions.load('session')
    interaction = state.interactions[-1]
    assert interaction.admitted['selected_candidate']['capability'] == CLUSTER_TOOL
    assert interaction.prerequisite is None
    run = continued.run_store.load(interaction.admitted['request_id'] + ':run')
    assert not run.steps and run.errors[-1].code == 'MISSING_REQUIRED_SOURCE'
    assert run.errors[-1].details['argument_names'] == (CLUSTER_KEY,)


def test_new_scientific_request_starts_its_own_string_pending_without_inheriting_partial_values(tmp_path):
    tool = 'transfer_cell_labels'
    partial = 'Use 2 reference neighbors.'
    fresh = 'Start a separate label-transfer task with these supplied inputs.'
    harness = string_harness(tmp_path / 'workspace', tool=tool, decisions={
        partial: string_answer(partial, [string_argument(partial, '2', tool=tool, argument='n_neighbors')]),
        fresh: dict(kind='execute_plan', target=tool)})
    inputs = semantic_string_inputs(tool)
    harness.service.create_session('session')
    assert_pending(submit_string(harness, inputs=inputs), 'reference_label_key')
    assert_pending(submit_string(harness, 'partial', partial), 'reference_label_key')
    with evaluation_work() as calls:
        independent = submit_string(harness, 'fresh', fresh, inputs=inputs)
    assert not calls
    assert_pending(independent, 'reference_label_key')
    interaction = harness.service._application.sessions.load('session').interactions[-1]
    assert interaction.prerequisite['origin_turn_id'] == 'fresh'
    assert interaction.prerequisite['binding_turn_id'] == 'fresh'
    assert 'n_neighbors' not in interaction.admitted['inputs']
    assert 'continuation' not in interaction.admitted


def test_pending_string_answer_rejects_changed_captured_resource_catalog_without_loading_it(tmp_path, string_inputs):
    from agent.application.local_resources import QualifiedEpiZooResource

    harness = string_harness(tmp_path / 'workspace')
    harness.service.create_session('session')
    assert_pending(submit_string(harness, inputs=string_inputs), CLUSTER_KEY)
    # Pure catalog metadata simulates a policy change. No checkpoint is created,
    # hashed or consumed; clustering evaluation uses neither this model nor it.
    changed = QualifiedEpiZooResource('synthetic-context', 'Synthetic catalog context', 'mouse',
        '/unqualified/model.pth', '1' * 64, '2' * 64, '3' * 64,
        'Context-digest test only; no scientific resource qualification.')
    with evaluation_work() as calls:
        rejected = harness.service.submit_turn('session', 'changed-context', STRING_ANSWER,
            expected_generation=0, epizoo_resources=(changed,)).to_dict()
    assert not calls
    assert rejected['response']['clarification']['reason'] == 'invalid_prerequisite', rejected
    state = harness.service._application.sessions.load('session')
    assert not state.revisions
    assert state.interactions[-1].prerequisite['binding_turn_id'] == 'first'
    assert CLUSTER_KEY not in state.interactions[0].admitted['inputs']
