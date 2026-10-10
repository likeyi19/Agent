"""Generic actual-plan output admission, with tiny real inspection owners.

Model decisions are scripted contract witnesses, not language qualification.
The historical four-step selector response is replayed structurally; none of
its foundation-model or downstream science is executed.
"""
import json

import pytest

from agent.application import InteractiveAgentApplication
from agent.orchestration import PlanningModelProfile, build_default_tool_registry
from agent.orchestration.executor import PlanExecutor
from agent.providers import PlanningModelFactoryRegistry
from agent.schemas import ErrorCategory, RecoveryDisposition
from test_prerequisite_planner_handoff import POLICY
from test_service import _tiny_h5ad


PROFILE = PlanningModelProfile('output-selection-test', 'scripted', 'model/output-selection')


def selected(name='cells', step_id='inspect', output_key='n_cells'):
    return dict(name=name, step_id=step_id, output_key=output_key)


def inspection_plan(step_count=1):
    return dict(schema_version=4, decision=dict(kind='plan', steps=[
        dict(step_id='inspect' if index == 0 else 'inspect_again', tool='inspect_scATAC',
             sources=[dict(target='dataset', source=dict(kind='input', input='input_path'))],
             control_dependencies=[]) for index in range(step_count)]))


class OutputModel:
    model_id = PROFILE.model_id

    def __init__(self, outputs, detailed=None, target='inspect_scATAC', *, raw_selector=None):
        self.outputs = outputs
        self.detailed = detailed or inspection_plan()
        self.target = target
        self.raw_selector = raw_selector
        self.calls = []
        self.schemas = []

    def complete(self, *, prompt, response_schema):
        value = json.loads(prompt)
        self.calls.append(value)
        self.schemas.append(response_schema)
        if 'turn_schema_version' in value:
            response = dict(turn_schema_version=1,
                            decision=dict(kind='execute_plan', target=self.target))
        elif 'selection_schema_version' in value:
            capabilities = ['processed_inspection'] if self.target == 'inspect_scATAC' else ['embedding_analysis']
            response = dict(selection_schema_version=1,
                            decision=dict(kind='select', capability_ids=capabilities))
        elif 'semantic_prompt_version' in value:
            response = self.detailed
        else:
            assert 'output_selection_schema_version' in value
            if isinstance(self.outputs, Exception):
                raise self.outputs
            if self.raw_selector is not None:
                return self.raw_selector
            response = dict(outputs=self.outputs)
        return json.dumps(response)

    def close(self):
        pass


class TrackingExecutor(PlanExecutor):
    def __init__(self, registry):
        super().__init__(registry)
        self.plans = []

    def execute(self, plan, **kwargs):
        self.plans.append(plan)
        return super().execute(plan, **kwargs)


def submit(tmp_path, outputs, *, steps=1, utterance='Inspect this matrix and retain its cell count.',
           detailed=None, target='inspect_scATAC', companions=None, raw_selector=None):
    registry = build_default_tool_registry()
    model = OutputModel(outputs, detailed or inspection_plan(steps), target,
                        raw_selector=raw_selector)
    executor = TrackingExecutor(registry)
    service = InteractiveAgentApplication(tmp_path / 'workspace',
        model_profiles=(PROFILE,), default_profile_id=PROFILE.profile_id,
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': lambda _: model}),
        registry=registry, executor=executor, planning_recovery_policy=POLICY)
    service.create_session('session')
    source = _tiny_h5ad(tmp_path / 'source.h5ad')
    view = service.submit_turn('session', 'first', utterance, expected_generation=0,
        execution_inputs=dict(input_path=str(source)) | (companions or {}))
    return service, model, executor, view


def assert_retained(service, model, executor, view, outputs, steps=1):
    assert view.status == 'succeeded' and view.response.status == 'activated'
    assert len(executor.plans) == 1
    state = service._application.sessions.load('session')
    assert state.generation == 1 and len(state.revisions) == 1
    retained = state.revisions[0].outputs
    assert [dict(name=o.name, step_id=o.step_id, output_key=o.output_key) for o in retained] == outputs
    run = service._application.run_store.load(view.run_id)
    assert len(run.steps) == steps and all(step.verification.passed for step in run.steps)
    assert [step.tool_name for step in run.steps] == ['inspect_scATAC'] * steps
    assert sum('output_selection_schema_version' in call for call in model.calls) == 1
    assert len(model.calls) == 4


def assert_rejected(service, model, executor, view, error_code='PLANNER_OUTPUT_INVALID'):
    assert view.status == 'failed' and view.revision_id is None and view.error is not None
    assert view.error.code == error_code
    state = service._application.sessions.load('session')
    assert not executor.plans and not state.revisions and not state.turns and state.generation == 0
    run = service._application.run_store.load(view.run_id)
    assert run.plan is None and not run.steps
    if error_code == 'PLANNER_OUTPUT_INVALID':
        assert run.errors[0].category is ErrorCategory.INTERNAL_AGENT_ERROR
        assert run.errors[0].recovery_disposition is RecoveryDisposition.NO_AUTOMATIC_RECOVERY
        assert not run.errors[0].recoverable
    assert sum('output_selection_schema_version' in call for call in model.calls) == 1
    assert len(model.calls) == 4


def test_valid_single_step_selection_activates_exact_offered_output(tmp_path):
    outputs = [selected()]
    service, model, executor, view = submit(tmp_path, outputs)
    assert_retained(service, model, executor, view, outputs)


def test_valid_multi_step_selection_keeps_exact_step_output_pairs(tmp_path):
    outputs = [selected('first_cells'), selected('second_cells', 'inspect_again')]
    service, model, executor, view = submit(tmp_path, outputs, steps=2,
        utterance='Inspect this matrix twice and retain the cell count from each inspection.')
    assert_retained(service, model, executor, view, outputs, steps=2)


def test_unique_names_are_exact_nonblank_strings_without_extra_alias_policy(tmp_path):
    outputs = [selected('Result'), selected('result', 'inspect_again'),
               selected(' Result ', output_key='n_features'), selected('细胞数')]
    service, model, executor, view = submit(tmp_path, outputs, steps=2)
    assert_retained(service, model, executor, view, outputs, steps=2)


def test_duplicate_output_name_is_rejected_across_plan_steps(tmp_path):
    outputs = [selected('cells'), selected('cells', 'inspect_again')]
    assert_rejected(*submit(tmp_path, outputs, steps=2))


def test_output_key_absent_from_offered_step_is_rejected(tmp_path):
    assert_rejected(*submit(tmp_path, [selected(output_key='invented_embedding')]))


@pytest.mark.parametrize('step_id,output_key', [('fabricated_step', 'n_cells'), ('inspect_again', 'resolution')])
def test_fabricated_or_incompatible_step_output_pair_is_rejected(tmp_path, step_id, output_key):
    assert_rejected(*submit(tmp_path, [selected(step_id=step_id, output_key=output_key)], steps=2))


def test_bounded_subset_does_not_require_retaining_every_offered_field(tmp_path):
    outputs = [selected('requested_feature_count', output_key='n_features')]
    service, model, executor, view = submit(tmp_path, outputs, steps=2,
        utterance='Inspect this matrix twice and retain only the first inspection feature count.')
    assert_retained(service, model, executor, view, outputs, steps=2)
    offered = next(call['steps'] for call in model.calls if 'output_selection_schema_version' in call)
    assert len(outputs) < sum(len(step['outputs']) for step in offered)


def test_explicit_valid_multi_output_request_preserves_all_selected_results(tmp_path):
    outputs = [selected('cell_count'), selected('feature_count', output_key='n_features')]
    service, model, executor, view = submit(tmp_path, outputs,
        utterance='Inspect this matrix and retain both its cell count and feature count.')
    assert_retained(service, model, executor, view, outputs)


def test_same_offered_pair_with_distinct_names_remains_valid(tmp_path):
    outputs = [selected('count'), selected('requested_alias')]
    service, model, executor, view = submit(tmp_path, outputs)
    assert_retained(service, model, executor, view, outputs)


@pytest.mark.parametrize('outputs', [[], [selected(str(i)) for i in range(33)]])
def test_existing_one_to_32_selection_bounds_remain_authoritative(tmp_path, outputs):
    assert_rejected(*submit(tmp_path, outputs))


@pytest.mark.parametrize('name', ['', '   '])
def test_blank_output_name_remains_rejected(tmp_path, name):
    assert_rejected(*submit(tmp_path, [selected(name)]))


def test_provider_value_error_is_not_reclassified_as_invalid_model_selection(tmp_path):
    assert_rejected(*submit(tmp_path, ValueError('Provider adapter failure.')),
                    error_code='PLANNER_UNEXPECTED_ERROR')


@pytest.mark.parametrize('raw_selector', [
    'not JSON',
    json.dumps(dict(outputs=[selected()], unoffered_field=True)),
    json.dumps(dict(outputs=[dict(name='cells', step_id='inspect')])),
], ids=['malformed_json', 'extra_envelope_field', 'missing_member_field'])
def test_malformed_or_nonclosed_selector_response_is_classified_and_rejected(tmp_path, raw_selector):
    assert_rejected(*submit(tmp_path, [], raw_selector=raw_selector))


def test_output_selection_prompt_teaches_generic_semantics_separately_from_validation(tmp_path):
    outputs = [selected()]
    service, model, executor, view = submit(tmp_path, outputs, steps=2)
    assert_retained(service, model, executor, view, outputs, steps=2)
    index = next(i for i, call in enumerate(model.calls) if 'output_selection_schema_version' in call)
    prompt, schema = model.calls[index], model.schemas[index]
    assert prompt['question'] == view.utterance
    assert [step['step_id'] for step in prompt['steps']] == ['inspect', 'inspect_again']
    for offered in prompt['steps']:
        assert offered['outputs'] == list(service._application.registry.get(offered['tool']).result_contract.required_fields)
    bounds = schema['properties']['outputs']
    assert (bounds['minItems'], bounds['maxItems']) == (1, 32)
    item = bounds['items']
    assert set(item['required']) == {'name', 'step_id', 'output_key'}
    assert not item['additionalProperties']
    instructions = prompt['instructions'].lower()
    assert 'step_id' in instructions and 'output_key' in instructions
    assert 'unique' in instructions and 'revision' in instructions
    assert 'nonblank' in instructions and '1 to 32' in instructions
    assert 'subset' in instructions and 'explicit' in instructions
    assert 'invent' in instructions and 'append' in instructions


def test_recorded_ua354b_selector_failure_replays_without_foundation_execution(tmp_path):
    registry = build_default_tool_registry()
    # Exact four-step candidate and 32-entry selector decision from saved A4,
    # reconstructed from Registry field order so the test needs no eval files.
    tools = [('embed_cells', 'epizoo_embed_cells', 'dataset', 'input_path', None),
             ('build_neighbors', 'build_cell_neighbors', 'embedding', 'embedding', 'embed_cells'),
             ('cluster_cells', 'cluster_cells', 'analysis', 'neighbors', 'build_neighbors'),
             ('compute_umap', 'compute_cell_umap', 'analysis', 'clustered_analysis', 'cluster_cells')]
    steps = []
    for step_id, tool, target, source_port, upstream in tools:
        source = (dict(kind='input', input=source_port) if upstream is None else
                  dict(kind='step_port', step=upstream, source_port=source_port))
        sources = [dict(target=target, source=source)]
        if upstream is None:
            sources.append(dict(target='species', source=dict(kind='input', input='species')))
        sources.append(dict(target='output_dir', source=dict(kind='input', input='output_dir')))
        steps.append(dict(step_id=step_id, tool=tool, sources=sources, control_dependencies=[]))
    detailed = dict(schema_version=4, decision=dict(kind='plan', steps=steps))
    outputs = [selected(key, step_id, key) for step_id, tool, *_ in tools[:2]
               for key in registry.get(tool).result_contract.required_fields]
    outputs += [selected('status', 'cluster_cells', 'status'),
                selected('input_analysis_path', 'cluster_cells', 'input_analysis_path},{')]
    assert len(outputs) == 32 and len({output['name'] for output in outputs}) < len(outputs)
    service, model, executor, view = submit(tmp_path, outputs, detailed=detailed,
        target='epizoo_embed_cells', companions=dict(species='mouse'),
        utterance='Analyze this dataset with the available EpiZoo model.')
    assert_rejected(service, model, executor, view)
    offered = next(call['steps'] for call in model.calls if 'output_selection_schema_version' in call)
    assert [(step['step_id'], step['tool']) for step in offered] == [(row[0], row[1]) for row in tools]
    assert outputs[-1]['output_key'] not in next(step['outputs'] for step in offered if step['step_id'] == 'cluster_cells')
