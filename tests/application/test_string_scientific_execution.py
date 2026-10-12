"""Exact column declarations exercise the existing registered evaluation owner.

The compact analysis and ground truth reuse existing synthetic owner fixtures.
They qualify software contracts only: no biological labels, upstream EpiZoo run,
checkpoint, model inference or multi-upload composition is claimed.
"""
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

import pytest

from agent.application import InteractiveAgentApplication
from agent.orchestration import PlanningModelProfile
from agent.providers import PlanningModelFactoryRegistry
from agent.schemas.orchestration import _serialize

sys.path.insert(0, str(Path(__file__).parents[1] / 'tools' / 'analysis'))
from test_clustering_evaluation import _write_analysis, _write_reference


CLUSTER_TOOL = 'evaluate_cell_clustering'
CLUSTER_KEY = 'label_key'
STRING_OBJECTIVE = 'Evaluate the fixed clustering against the supplied ground-truth dataset.'
STRING_LITERAL = 'Cell 类型 Δ'
STRING_ANSWER = f'请使用 ground-truth column "{STRING_LITERAL}".'
STRING_PROFILE = PlanningModelProfile('string-scripted', 'scripted', 'scripted/string')
STRING_ARGUMENTS = {
    CLUSTER_TOOL: CLUSTER_KEY,
    'transfer_cell_labels': 'reference_label_key',
    'evaluate_cell_annotation': 'ground_truth_label_key',
}
SOURCE_PORTS = {
    CLUSTER_TOOL: [('analysis', 'analysis_path'),
                   ('ground_truth_dataset', 'reference_h5ad_path')],
    'transfer_cell_labels': [('reference_dataset', 'reference_input_path'),
                             ('query_dataset', 'query_input_path'),
                             ('reference_embedding', 'reference_embedding_path'),
                             ('query_embedding', 'query_embedding_path')],
    'evaluate_cell_annotation': [('annotation', 'annotation_path'),
                                 ('ground_truth_dataset', 'ground_truth_h5ad_path')],
}


def string_argument(utterance, literal=STRING_LITERAL, *, tool=CLUSTER_TOOL, argument=None, **changes):
    start = utterance.index(literal)
    return dict(tool=tool, argument=argument or STRING_ARGUMENTS[tool],
        literal=literal, start=start, end=start + len(literal)) | changes


def string_answer(utterance=STRING_ANSWER, declarations=None):
    return dict(kind='answer_parameters', pending='@parameters', arguments=(
        [string_argument(utterance)] if declarations is None else declarations))


class StringModel:
    """Only the ordinary semantic model responses are scripted."""

    def __init__(self, decisions, calls, *, tool=CLUSTER_TOOL, plan_variant=None):
        self.decisions, self.calls, self.tool = decisions, calls, tool
        self.plan_variant = plan_variant
        self.model_id = STRING_PROFILE.model_id

    def complete(self, *, prompt, response_schema):
        value = json.loads(prompt)
        self.calls.append(value)
        if 'turn_schema_version' in value:
            return json.dumps(dict(turn_schema_version=1,
                decision=self.decisions[value['utterance']]))
        if 'selection_schema_version' in response_schema.get('properties', {}):
            capability = 'embedding_analysis' if self.tool == CLUSTER_TOOL else 'reference_annotation'
            return json.dumps(dict(selection_schema_version=1,
                decision=dict(kind='select', capability_ids=[capability])))
        if 'output_selection_schema_version' in value:
            output = 'annotation_path' if self.tool == 'transfer_cell_labels' else 'report_path'
            return json.dumps(dict(outputs=[dict(name='evaluation', step_id='evaluation', output_key=output)]))
        sources = [dict(target=port, source=dict(kind='input', input=input_name))
                   for port, input_name in SOURCE_PORTS[self.tool]]
        if self.plan_variant == 'wrong_ground_truth':
            sources[-1]['source']['input'] = 'analysis_path'
        if self.plan_variant == 'wrong_reference':
            sources[0]['source']['input'] = 'query_input_path'
        if self.plan_variant == 'wrong_annotation_truth':
            sources[-1]['source']['input'] = 'annotation_path'
        steps = [dict(step_id='evaluation', tool=self.tool, sources=sources, control_dependencies=[])]
        if self.plan_variant == 'repeated':
            steps.append(deepcopy(steps[0]) | dict(step_id='other_evaluation'))
        return json.dumps(dict(schema_version=4, decision=dict(kind='plan', steps=steps)))


@dataclass
class StringHarness:
    service: InteractiveAgentApplication
    decisions: dict
    calls: list
    constructions: list


def string_harness(workspace, *, decisions=None, tool=CLUSTER_TOOL, plan_variant=None, resources=()):
    choices = {STRING_OBJECTIVE: dict(kind='execute_plan', target=tool),
               STRING_ANSWER: string_answer()} | (decisions or {})
    calls, constructions = [], []

    def factory(profile):
        model = StringModel(choices, calls, tool=tool, plan_variant=plan_variant)
        constructions.append(model)
        return model

    service = InteractiveAgentApplication(workspace,
        model_profiles=(STRING_PROFILE,), default_profile_id=STRING_PROFILE.profile_id,
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': factory}),
        epizoo_resources=resources)
    return StringHarness(service, choices, calls, constructions)


@pytest.fixture
def string_inputs(tmp_path):
    return dict(analysis_path=str(_write_analysis(tmp_path / 'clustered.h5ad')),
        reference_h5ad_path=str(_write_reference(tmp_path / 'truth.h5ad', key=STRING_LITERAL)))


def semantic_string_inputs(tool):
    """Complete port declarations, deliberately unqualified for owner execution.

    These paths never resolve to files or load a checkpoint. They are used only
    to test compiler-owned missing-key admission and rejected partial answers.
    """
    if tool == 'evaluate_cell_annotation':
        return dict(annotation_path='/unqualified/annotation.h5ad',
            ground_truth_h5ad_path='/unqualified/truth.h5ad')
    if tool == 'transfer_cell_labels':
        return {f'{role}_{name}': value for role in ('reference', 'query')
                for name, value in [('input_path', f'/unqualified/{role}.h5ad'),
                    ('embedding_path', f'/unqualified/{role}.npy'),
                    ('cell_ids_path', f'/unqualified/{role}.txt'),
                    ('species', 'mouse'),
                    ('checkpoint_path', '/unqualified/model.pth')]}
    raise AssertionError(tool)


def submit_string(harness, turn='first', utterance=STRING_OBJECTIVE, inputs=None, generation=0):
    return harness.service.submit_turn('session', turn, utterance,
        expected_generation=generation, execution_inputs=inputs).to_dict()


@contextmanager
def evaluation_work():
    """Observe the actual owner, source reinspection and registered verifier."""
    from agent.tools.analysis import clustering_evaluation as owner
    from agent.orchestration import verifier
    codes = {owner.evaluate_cell_clustering.__code__: 'production',
             owner._evaluate_sources.__code__: 'source_reinspection',
             verifier._verify_clustering_evaluation.__code__: 'verification'}
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


def accepted_evaluation(harness, view, inputs, literal=STRING_LITERAL):
    assert view['status'] == 'succeeded' and view['response']['status'] == 'activated', view
    run = harness.service._application.run_store.load(view['run_id'])
    assert len(run.steps) == 1 and run.steps[0].tool_name == CLUSTER_TOOL
    step = run.steps[0]
    assert step.verification.passed
    assert step.resolved_arguments['label_key'] == step.result['label_key'] == literal
    assert step.resolved_arguments['analysis_path'] == inputs['analysis_path']
    assert step.resolved_arguments['reference_h5ad_path'] == inputs['reference_h5ad_path']
    assert step.result['n_cells'] == 8 and step.result['n_reference_classes'] == 4
    report = json.loads(Path(step.result['report_path']).read_text())
    assert report['inputs']['label_key'] == literal
    return step


def test_initial_exact_string_uses_registered_production_verification_and_user_provenance(tmp_path, string_inputs):
    utterance = STRING_OBJECTIVE + ' ' + STRING_ANSWER
    declaration = string_argument(utterance)
    harness = string_harness(tmp_path / 'workspace', decisions={utterance:
        dict(kind='execute_plan', target=CLUSTER_TOOL, arguments=[declaration])})
    harness.service.create_session('session')
    source_digests = {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                      for name, path in string_inputs.items()}
    with evaluation_work() as calls:
        view = submit_string(harness, utterance=utterance, inputs=string_inputs)
    assert calls['production'] == 1 and calls['verification'] >= 1
    assert calls['source_reinspection'] == calls['verification'] + 1
    accepted_evaluation(harness, view, string_inputs)
    state = harness.service._application.sessions.load('session')
    interaction, = state.interactions
    assert _serialize(interaction.admitted['arguments']) == [declaration]
    assert 'label_key' not in interaction.submission['execution_inputs']
    assert interaction.admitted['inputs']['label_key'] == STRING_LITERAL
    evidence = harness.service.evidence('session', view['revision_id'], 'evaluation')
    fact = next(item for item in evidence.facts if item.field == 'explicit_user_parameters')
    assert _serialize(fact.value) == dict(tool=CLUSTER_TOOL, step_id='evaluation',
        parameters={'label_key': 'explicit_user_instruction'})
    assert fact.artifact_name == 'accepted_session_interactions'
    assert source_digests == {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                              for name, path in string_inputs.items()}


@pytest.mark.parametrize('variant', ['missing_column', 'interchanged_datasets', 'different_cell_order'])
def test_existing_owner_rejects_missing_column_and_interchanged_data_roles(tmp_path, string_inputs, variant):
    literal = 'absent_column' if variant == 'missing_column' else STRING_LITERAL
    utterance = f'Evaluate the clustering using the ground-truth column "{literal}".'
    harness = string_harness(tmp_path / 'workspace', decisions={utterance:
        dict(kind='execute_plan', target=CLUSTER_TOOL, arguments=[string_argument(utterance, literal)])})
    harness.service.create_session('session')
    if variant == 'interchanged_datasets':
        string_inputs = dict(analysis_path=string_inputs['reference_h5ad_path'],
            reference_h5ad_path=string_inputs['analysis_path'])
    elif variant == 'different_cell_order':
        string_inputs = string_inputs | dict(reference_h5ad_path=str(_write_reference(
            tmp_path / 'reordered-truth.h5ad', ids=tuple(f'cell-{i}' for i in reversed(range(8))),
            key=STRING_LITERAL)))
    with evaluation_work() as calls:
        view = submit_string(harness, utterance=utterance, inputs=string_inputs)
    assert view['revision_id'] is None and view['response']['status'] == 'failed', view
    assert calls == {'production': 1, 'source_reinspection': 1}
    run = harness.service._application.run_store.load(view['run_id'])
    assert any(error.code == 'INVALID_ARGUMENT' and error.tool_name == CLUSTER_TOOL
               and error.step_id == 'evaluation' for error in run.errors)
    assert not harness.service._application.sessions.load('session').revisions


@pytest.mark.parametrize('tool,variant', [(CLUSTER_TOOL, 'wrong_ground_truth'),
    ('transfer_cell_labels', 'wrong_reference'),
    ('evaluate_cell_annotation', 'wrong_annotation_truth')])
def test_canonical_column_declaration_cannot_authorize_a_different_source_role(tmp_path, string_inputs, tool, variant):
    utterance = f'Use the declared column "{STRING_LITERAL}" for this scientific operation.'
    inputs = string_inputs if tool == CLUSTER_TOOL else semantic_string_inputs(tool)
    harness = string_harness(tmp_path / 'workspace', tool=tool, plan_variant=variant,
        decisions={utterance: dict(kind='execute_plan', target=tool,
            arguments=[string_argument(utterance, tool=tool)])})
    harness.service.create_session('session')
    with evaluation_work() as calls:
        result = submit_string(harness, utterance=utterance, inputs=inputs)
    assert not calls and result['revision_id'] is None
    assert result['response']['status'] == 'failed', result
    state = harness.service._application.sessions.load('session')
    assert state.interactions[-1].admitted['inputs'][STRING_ARGUMENTS[tool]] == STRING_LITERAL
    assert state.interactions[-1].prerequisite is None
    run = harness.service._application.run_store.load(result['run_id'])
    assert not run.steps and run.errors[-1].code == 'UNAUTHORIZED_REQUEST_INPUT'
