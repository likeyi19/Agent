"""Stage 2 transport, exact binding and forbidden redundant model calls.

Plans/preflight are production. The executor explicitly withholds native science.
"""
from dataclasses import asdict
import json

import anndata as ad
import numpy as np
import pytest
from scipy import sparse

from agent.application import InteractiveAgentApplication
from agent.application.session_state import SessionError
from agent.application.structured_parameters import catalog
from agent.orchestration import PlanningModelProfile, build_default_tool_registry
from agent.providers import PlanningModelFactoryRegistry
from agent.schemas.orchestration import _serialize
from test_human_epizoo_mapping import WithholdScience

PROFILE = PlanningModelProfile('structured', 'scripted', 'structured-transport-test')


class Model:
    model_id = PROFILE.model_id

    def __init__(self, *, repeated=False, missing=(), declaration=None):
        self.calls = []
        self.interpret = True
        self.repeated, self.missing, self.declaration = repeated, missing, declaration

    def complete(self, *, prompt, response_schema):
        value = json.loads(prompt)
        self.calls.append(value)
        if 'turn_schema_version' in value:
            assert self.interpret, 'Unexpected Interpreter call on typed continuation'
            decision = dict(kind='execute_plan', target='run_scanpy', missing_parameters=list(self.missing))
            if self.declaration:
                decision['arguments'] = [self.declaration]
            return json.dumps(dict(turn_schema_version=1, decision=decision))
        if 'output_selection_schema_version' in value:
            return json.dumps(dict(outputs=[dict(name='scaled', step_id='scale', output_key='matrix_path')]))
        assert 'clarification' not in value, 'Unexpected clarification rewrite'
        fn = lambda name: dict(target='function_name', source=dict(kind='registered_choice', choice=name))
        steps = [dict(step_id='scale', tool='run_scanpy', sources=[fn('pp.scale'),
            dict(target='dataset', source=dict(kind='input', input='input_path'))], control_dependencies=[])]
        steps.append(dict(step_id='again', tool='run_scanpy', sources=[fn('pp.scale' if self.repeated else 'pp.log1p'),
            dict(target='dataset', source=dict(kind='step_port', step='scale', source_port='dataset'))], control_dependencies=[]))
        return json.dumps(dict(schema_version=4, decision=dict(kind='plan', steps=steps)))


def setup(tmp_path, **options):
    registry = build_default_tool_registry()
    executor = WithholdScience(registry)
    model = Model(**options)
    app = InteractiveAgentApplication(tmp_path / 'workspace', registry=registry, executor=executor,
        model_profiles=(PROFILE,), default_profile_id=PROFILE.profile_id,
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': lambda _: model}))
    app.create_session('session')
    path = tmp_path / 'input.h5ad'
    ad.AnnData(sparse.csr_matrix(np.eye(4))).write_h5ad(path)
    return app, executor, model, dict(input_path=str(path))


def setting(argument, value, function='pp.scale'):
    return dict(tool='run_scanpy', argument=argument, value=value, native_function=function)


def test_multiple_scoped_values_are_applied_without_model_reinterpretation(tmp_path):
    app, executor, model, inputs = setup(tmp_path)
    rows = [setting('kwargs', dict(zero_center=False, max_value=5)),
            setting('count_slot', 'X'), setting('kwargs', {'base': 2}, 'pp.log1p')]
    result = app.submit_turn('session', 'task', '标准化此数据并保留原始输入。', expected_generation=0,
        execution_inputs=inputs, parameter_overrides=rows)
    assert result.error.code == 'TEST_SCIENCE_WITHHELD', result.to_dict()
    scale, log = executor.plans[0].steps
    assert _serialize(scale.arguments['kwargs']) == dict(zero_center=False, max_value=5)
    assert _serialize(log.arguments['kwargs']) == {'base': 2}
    assert scale.arguments['count_slot'] == 'X' and 'count_slot' not in log.arguments
    assert sum('turn_schema_version' in call for call in model.calls) == 1
    state = app._application.sessions.load('session')
    declared = state.interactions[0].admitted['arguments']
    assert all(d['origin'] == 'structured' and 'literal' not in d and 'start' not in d for d in declared)
    assert state.interactions[0].submission['parameter_overrides'] == declared
    assert app._application.sessions.load('session') == state
    count = len(model.calls)
    assert app.submit_turn('session', 'task', '标准化此数据并保留原始输入。', expected_generation=0,
        execution_inputs=inputs, parameter_overrides=rows) == result
    assert len(model.calls) == count and len(executor.plans) == 1


def test_partial_native_override_preserves_grounded_other_members(tmp_path):
    text = 'Scale with zero_center=false and max_value=9.'
    literal = 'zero_center=false and max_value=9'
    start = text.index(literal)
    declaration = dict(tool='run_scanpy', argument='kwargs', native_function='pp.scale', members=[
        dict(name='zero_center', kind='json', literal='false', evidence='zero_center=false'),
        dict(name='max_value', kind='number', literal='9', evidence='max_value=9')])
    app, executor, _, inputs = setup(tmp_path, declaration=declaration)
    result = app.submit_turn('session', 'task', text, expected_generation=0, execution_inputs=inputs,
        parameter_overrides=[setting('kwargs', {'max_value': 4})])
    assert result.error.code == 'TEST_SCIENCE_WITHHELD', result.to_dict()
    assert _serialize(executor.plans[0].steps[0].arguments['kwargs']) == dict(zero_center=False, max_value=4)


def test_parameter_question_partial_values_skip_interpreter_and_rewrite(tmp_path):
    app, executor, model, inputs = setup(tmp_path, missing=('count_slot', 'matrix_semantics'))
    first = app.submit_turn('session', 'task', '分析输入。', expected_generation=0, execution_inputs=inputs)
    assert first.status == 'clarification', first.to_dict()
    assert not executor.plans
    model.interpret = False
    second = app.submit_turn('session', 'slot', '', expected_generation=0,
        parameter_overrides=[setting('count_slot', 'X')])
    assert second.status == 'clarification', second.to_dict()
    assert not executor.plans
    third = app.submit_turn('session', 'meaning', '', expected_generation=0,
        parameter_overrides=[setting('matrix_semantics', 'peak_counts')])
    assert third.error.code == 'TEST_SCIENCE_WITHHELD', third.to_dict()
    assert len(executor.plans) == 1
    assert sum('turn_schema_version' in call for call in model.calls) == 1
    assert all('clarification' not in call for call in model.calls)


def test_repeated_operation_offers_actual_steps_and_reuses_captured_plan(tmp_path):
    app, executor, model, inputs = setup(tmp_path, repeated=True)
    first = app.submit_turn('session', 'task', 'Scale the two planned representations.', expected_generation=0,
        execution_inputs=inputs, parameter_overrides=[setting('kwargs', {'max_value': 5})])
    assert first.status == 'clarification', first.to_dict()
    assert [r['handle'] for r in first.clarification_choices] == ['step:scale', 'step:again']
    assert not executor.plans
    model.interpret = False
    planned = len([c for c in model.calls if 'turn_schema_version' not in c and 'output_selection_schema_version' not in c])
    second = app.submit_turn('session', 'choose', 'again', expected_generation=0,
        clarification_choice=dict(question_turn_id='task', handle='step:again'))
    assert second.error.code == 'TEST_SCIENCE_WITHHELD', second.to_dict()
    scale, again = executor.plans[0].steps
    assert 'kwargs' not in scale.arguments
    assert _serialize(again.arguments['kwargs']) == {'max_value': 5}
    assert len([c for c in model.calls if 'turn_schema_version' not in c and 'output_selection_schema_version' not in c]) == planned
    assert app._application.sessions.load('session').interactions[-1].admitted['scope_selection'] == 'task'


def test_catalog_uses_registry_names_without_filling_defaults():
    registry = build_default_tool_registry()
    view, = catalog(registry, 'run_snapatac2', 'tl.leiden')
    assert [p['argument'] for p in view['kwargs']] == list(dict(registry.get('run_snapatac2').native_functions.functions)['tl.leiden'])
    assert 'resolution' in {p['argument'] for p in view['kwargs']}


def test_structured_selection_cannot_enter_through_language_parser():
    from agent.application.turn_decisions import parse_decision
    with pytest.raises((ValueError, TypeError)):
        parse_decision(json.dumps(dict(turn_schema_version=1, decision=dict(kind='execute_plan', target='run_scanpy',
            arguments=[setting('kwargs', {'max_value': 5}) | dict(origin='structured')]))))


def test_finite_role_click_uses_exact_source_and_no_interpretation(tmp_path):
    from test_role_declaration_continuation import transfer_continuation_harness, OBJECTIVE
    app, records, group, executor, calls, _ = transfer_continuation_harness(tmp_path)
    first = app.submit_turn('session', 'task', OBJECTIVE, expected_generation=0, registered_input=group)
    assert first.status == 'clarification'
    first_handle = next(c['handle'] for c in first.clarification_choices if c.get('port') == 'reference_dataset') + ':' + records[0].resource_id
    count = sum('turn_schema_version' in call['value'] for call in calls)
    second = app.submit_turn('session', 'ref', records[0].label, expected_generation=0,
        clarification_choice=dict(question_turn_id='task', handle=first_handle))
    assert second.status == 'clarification', second.to_dict()
    query = next(c['handle'] for c in second.clarification_choices if c.get('port') == 'query_dataset') + ':' + records[1].resource_id
    third = app.submit_turn('session', 'query', records[1].label, expected_generation=0,
        clarification_choice=dict(question_turn_id='ref', handle=query))
    assert third.error.code == 'TEST_SCIENCE_WITHHELD', third.to_dict()
    assert sum('turn_schema_version' in call['value'] for call in calls) == count
    step, = executor.plans[0].steps
    assert step.arguments['reference_h5ad_path'] == records[0].source_path
    assert step.arguments['query_h5ad_path'] == records[1].source_path


def test_candidate_click_bypasses_interpreter_but_keeps_planning(tmp_path):
    from agent.application import ResearchAgentApplication
    from test_dialogue_evidence import accepted
    from test_guidance_selection import offer, planner_app, Selection
    seed = ResearchAgentApplication(tmp_path / 'candidate')
    seed.sessions.create('session')
    accepted(seed, 'inspect_scATAC')
    cid = offer(seed)
    app, path, calls = planner_app(seed, tmp_path)
    class NoInterpret(Selection):
        def complete(self, *, prompt, response_schema):
            assert 'turn_schema_version' not in json.loads(prompt), 'Unexpected candidate interpretation'
            return super().complete(prompt=prompt, response_schema=response_schema)
    model = NoInterpret(cid, 'unused')
    submission = dict(execution_inputs={'path': str(path)},
        candidate_selection=dict(origin_turn_id='guide', candidate_id=cid))
    result = app.sessions.respond('session', 'choose', '选择此选项', interpreter=model,
        predecessor_turn_id='guide', execution_inputs=submission['execution_inputs'], submission=submission)
    assert result.status == 'activated', result
    assert len(calls) == 1
    admitted = app.sessions.load('session').interactions[-1].admitted
    assert admitted['selected_candidate']['candidate_id'] == cid


def test_same_function_source_scopes_keep_different_exact_values(tmp_path):
    from agent.application.structured_parameters import bind_plan, check_plan, declarations
    from agent.application.dialogue_execution import _bind_arguments
    from agent.schemas import AgentPlan, PlanStep
    registry = build_default_tool_registry()
    rows = [setting('kwargs', {'max_value': value}) | dict(port='dataset', source=f'source.{i}')
            for i, value in enumerate((4, 7))]
    declared = declarations(rows, registry)
    inputs = _bind_arguments(registry, '', {}, declared)
    rows = [asdict(d) for d in declared]
    paths = [str(tmp_path / f'{i}.h5ad') for i in range(2)]
    admitted = dict(roles=[dict(tool='run_scanpy', port='dataset', source=f'source.{i}') for i in range(2)],
        registered_input=dict(members=[dict(source_path=path) for path in paths]))
    plan = AgentPlan('p', 'r', 'Two explicitly scoped inputs.', tuple(
        PlanStep(f's{i}', 'run_scanpy', dict(input_path=path, output_dir=str(tmp_path / str(i)), function_name='pp.scale'))
        for i, path in enumerate(paths)))
    bound = bind_plan(plan, rows, inputs, registry, admitted)
    assert [_serialize(s.arguments['kwargs']) for s in bound.steps] == [{'max_value': 4}, {'max_value': 7}]
    check_plan(bound, rows, inputs, registry, admitted)


def test_two_ambiguous_parameters_resolve_in_the_same_saved_plan(tmp_path):
    app, executor, model, inputs = setup(tmp_path, repeated=True)
    first = app.submit_turn('session', 'task', 'Scale both representations.', expected_generation=0,
        execution_inputs=inputs, parameter_overrides=[setting('kwargs', {'max_value': 5}), setting('count_slot', 'X')])
    assert first.status == 'clarification', first.to_dict()
    model.interpret = False
    second = app.submit_turn('session', 'first-scope', 'scale', expected_generation=0,
        clarification_choice=dict(question_turn_id='task', handle='step:scale'))
    assert second.status == 'clarification', second.to_dict()
    third = app.submit_turn('session', 'second-scope', 'again', expected_generation=0,
        clarification_choice=dict(question_turn_id='first-scope', handle='step:again'))
    assert third.error.code == 'TEST_SCIENCE_WITHHELD', third.to_dict()
    scale, again = executor.plans[0].steps
    assert _serialize(scale.arguments['kwargs']) == {'max_value': 5} and 'kwargs' not in again.arguments
    assert 'count_slot' not in scale.arguments and again.arguments['count_slot'] == 'X'
    assert sum('semantic_prompt_version' in c for c in model.calls) == 1


@pytest.mark.parametrize('species', ['human', 'mouse'])
def test_finite_species_selection_keeps_original_input_without_interpreter(tmp_path, species):
    from test_human_epizoo_mapping import configuration, MappingWitness, OBJECTIVE, _tiny_h5ad
    from agent.web.config import build_interactive_application
    config = configuration(tmp_path)
    registry = build_default_tool_registry()
    executor = WithholdScience(registry)
    calls = []
    forbid = False
    class FiniteSpecies(MappingWitness):
        def complete(self, *, prompt, response_schema):
            if 'turn_schema_version' in json.loads(prompt):
                assert not forbid, 'Unexpected interpretation of a species click'
            return super().complete(prompt=prompt, response_schema=response_schema)
    app = build_interactive_application(config, registry=registry, executor=executor,
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted':lambda _: FiniteSpecies(species, calls)}))
    app.create_session('session')
    path = _tiny_h5ad(config.upload_root / 'species-fixture.h5ad')
    record = app.resources.register('species-input', path, label='Source', attribution='Simulated species binding fixture.')
    binding = app.resources.compose_h5ad(record.resource_id, config.input_sets[0].inputs(), resource_selection_error='EPIZOO_RESOURCE_REQUIRED')
    first = app.submit_turn('session', 'task', OBJECTIVE, expected_generation=0, registered_input=binding)
    assert first.status == 'clarification'
    assert {c['handle'] for c in first.clarification_choices} == {'species:human', 'species:mouse'}
    forbid = True
    result = app.submit_turn('session', 'species', species, expected_generation=0,
        clarification_choice=dict(question_turn_id='task', handle='species:' + species))
    assert result.error.code == 'TEST_SCIENCE_WITHHELD', result.to_dict()
    assert executor.plans[0].steps[0].arguments['input_path'] == record.source_path
    assert executor.plans[0].steps[0].arguments['species'] == species


def test_source_parameter_uses_file_identity_after_source_order_changes(tmp_path):
    from test_observed_parameter_binding import transfer_case, TRANSFER, LABEL
    app, records, group, executor, calls = transfer_case(tmp_path, reverse=True)
    result = app.submit_turn('session', 'field', TRANSFER, expected_generation=0, registered_input=group,
        parameter_overrides=[dict(tool='transfer_cell_labels', argument='reference_label_key', value=LABEL,
            port='reference_dataset', resource_id=records[0].resource_id)])
    assert result.error.code == 'TEST_SCIENCE_WITHHELD', result.to_dict()
    step, = executor.plans[0].steps
    assert step.arguments['reference_h5ad_path'] == records[0].source_path
    assert step.arguments['reference_label_key'] == LABEL
    state = app._application.sessions.load('session')
    declaration = next(row for row in state.interactions[-1].admitted['arguments'] if row.get('origin') == 'structured')
    assert declaration['source'] == 'source.1' and declaration['resource_id'] == records[0].resource_id
    assert state.interactions[-1].submission['parameter_overrides'][0]['source'] is None


def test_compiler_binds_settings_before_downstream_input_preparation(tmp_path):
    from agent.schemas import AgentRequest
    from agent.orchestration.active_context import ActivePlanningContext, planning_context, input_preparation
    from agent.orchestration.semantic_compiler import (compile_semantic_plan, build_semantic_compiler_contract,
        SemanticPlanCandidate, SemanticPlanStep, SemanticRegisteredChoiceSource as Choice,
        SemanticRequestInputSource as Input, SemanticStepOutputSource as Upstream)
    registry = build_default_tool_registry()
    inputs = dict(input_path=str(tmp_path/'counts.h5ad'), input_sha256='a'*64, output_dir=str(tmp_path/'output'),
        annotation_spec_path=str(tmp_path/'spec.json'), annotation_spec_sha256='b'*64,
        count_slot='layers/counts', run_snapatac2__pp_add_tile_matrix__kwargs={'counting_strategy': 'fragment'})
    rows = [dict(tool='run_snapatac2', argument='kwargs', native_function='pp.add_tile_matrix',
                 input_name='run_snapatac2__pp_add_tile_matrix__kwargs', value={'counting_strategy': 'fragment'}),
            dict(tool='annotate_scATAC_cell_types', argument='count_slot', input_name='count_slot', value='layers/counts')]
    context = ActivePlanningContext('s', None, 0, (), task=dict(parameter_overrides=rows))
    candidate = SemanticPlanCandidate((
        SemanticPlanStep('tile', 'run_snapatac2', (Input('dataset', 'input_path'), Choice('function_name', 'pp.add_tile_matrix'),
            Choice('counting_strategy', 'insertion'))),
        SemanticPlanStep('annotation', 'annotate_scATAC_cell_types', (Upstream('source', 'tile', 'dataset'),))))
    observed = []
    def prepare(tool, arguments, selected):
        assert tool.name == 'annotate_scATAC_cell_types'
        assert arguments['count_slot'] == 'layers/counts'
        assert selected['tile'][1]['kwargs']['counting_strategy'] == 'fragment'
        assert 'counting_strategy' not in selected['tile'][1]
        observed.append(arguments['count_slot'])
        return arguments
    with planning_context(context), input_preparation(prepare):
        plan = compile_semantic_plan(AgentRequest('r', 'Build selected counts and annotate.', inputs), candidate,
                                     registry, build_semantic_compiler_contract(registry))
    assert observed == ['layers/counts']
    assert plan.steps[1].arguments['count_slot'] == 'layers/counts'


def test_candidate_settings_keep_captured_objective_and_normal_planning(tmp_path):
    from agent.application import ResearchAgentApplication
    from test_dialogue_evidence import accepted
    from test_guidance_selection import offer
    from agent.orchestration.llm_planner import LLMPlanner
    seed = ResearchAgentApplication(tmp_path / 'candidate-settings')
    seed.sessions.create('session')
    accepted(seed, 'inspect_scATAC')
    cid = offer(seed, tools=('run_scanpy',))
    model = Model()
    model.interpret = False
    registry = seed.registry
    executor = WithholdScience(registry)
    app = ResearchAgentApplication(seed.workspace_root, registry=registry, executor=executor, planner=LLMPlanner(model))
    source = tmp_path / 'candidate-input.h5ad'
    ad.AnnData(sparse.csr_matrix(np.eye(4))).write_h5ad(source)
    from agent.application.structured_parameters import declarations
    rows = [asdict(d) for d in declarations([setting('kwargs', {'zero_center': False})], registry)]
    submission = dict(execution_inputs={'input_path': str(source)}, parameter_overrides=rows,
        candidate_selection=dict(origin_turn_id='guide', candidate_id=cid))
    result = app.sessions.respond('session', 'selected', 'run_scanpy', interpreter=model,
        predecessor_turn_id='guide', submission=submission, execution_inputs=submission['execution_inputs'])
    assert result.status == 'failed' and len(executor.plans) == 1, result
    assert executor.plans[0].steps[0].arguments['kwargs']['zero_center'] is False
    state = app.sessions.load('session')
    run = app.run_store.load(state.turns[-1].run_id)
    assert run.errors[0].code == 'TEST_SCIENCE_WITHHELD'
    intent = json.loads(run.request.prompt)
    assert intent['objective'] == 'What should I analyze next?' and intent['selection_intent'] == 'execute'
    assert not any('turn_schema_version' in call for call in model.calls)


def test_common_response_preserves_actual_owner_requirements():
    from test_response_contracts import OutcomeModel, outcome_session
    from agent.application.responses import summarize
    from agent.application.turns import TurnOutcome
    from agent.application.turn_decisions import Clarify
    sessions, interaction = outcome_session()
    question = 'Specify the annotation measurement.'
    model = OutcomeModel('请说明输入矩阵的测量含义。')
    result = summarize(sessions, 's', 'q', TurnOutcome('clarify', 'clarification',
        Clarify('missing_resources', (question,))), model=model)
    assert result.text == '请说明输入矩阵的测量含义。'
    assert model.calls[0]['evidence']['outcome']['clarification']['choices'] == [question]
    assert len(model.calls) == 2


def test_accepted_structured_parameters_remain_readable_without_native_replay(tmp_path, monkeypatch):
    from agent.tools import _native_process
    from agent.tools.analysis import native_matrix
    monkeypatch.setattr(_native_process, 'call', lambda operation, arguments, local: local())
    calls = []
    def simulated_native(self, args, stage, matrix):
        calls.append(args)
        matrix.write_h5ad(stage/'result.h5ad')
        return dict(completion='succeeded', versions={'scanpy':'SIMULATED'}, implementation=[],
            native_output_type='NoneType', result_slots=[], measurement='unchanged technical fixture')
    monkeypatch.setattr(native_matrix.MatrixPackage, 'native', simulated_native)
    seed, _, model, inputs = setup(tmp_path)
    def application():
        return InteractiveAgentApplication(seed._application.workspace_root, registry=seed._application.registry,
            model_profiles=(PROFILE,), default_profile_id=PROFILE.profile_id,
            planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': lambda _:model}))
    app = application()
    done = app.submit_turn('session', 'accepted', 'Scale and log-transform the input.', expected_generation=0,
        execution_inputs=inputs, parameter_overrides=[setting('kwargs', {'zero_center':False, 'max_value':4})])
    assert done.status == 'succeeded', done.to_dict()
    state = app._application.sessions.load('session')
    revision = state.revisions[0]
    presentation = _serialize(state.interactions[-1].presentation)
    count = len(model.calls)
    reopened = application()
    evidence = reopened._application.sessions.evidence('session', revision.revision_id, revision.outputs[0].name)
    assert evidence.status == 'available', evidence.to_dict()
    fact = next(f for f in evidence.facts if f.field == 'explicit_user_parameters')
    assert fact.reason == 'accepted_structured_session_declarations'
    assert fact.value['parameters']['kwargs'] == 'explicit_user_instruction'
    assert len(calls) == 2 and len(model.calls) == count
    assert _serialize(reopened._application.sessions.load('session').interactions[-1].presentation) == presentation


def test_unresolved_structured_scope_cannot_compile_as_executable_science(tmp_path):
    from agent.schemas import AgentRequest
    from agent.orchestration.active_context import ActivePlanningContext, planning_context
    from agent.orchestration.semantic_compiler import (compile_semantic_plan, build_semantic_compiler_contract,
        SemanticPlanCompileError, SemanticPlanCandidate, SemanticPlanStep, SemanticRegisteredChoiceSource as Choice,
        SemanticRequestInputSource as Input)
    registry = build_default_tool_registry()
    row = dict(tool='run_scanpy', argument='kwargs', native_function='pp.scale',
        input_name='run_scanpy__pp_scale__kwargs', value={'max_value':4})
    context = ActivePlanningContext('s', None, 0, (), task=dict(parameter_overrides=[row]))
    candidate = SemanticPlanCandidate(tuple(SemanticPlanStep(name,'run_scanpy',
        (Input('dataset','input_path'), Choice('function_name','pp.scale'))) for name in ('one','two')))
    request = AgentRequest('r','Scale inputs.',dict(input_path=str(tmp_path/'input.h5ad'), output_dir=str(tmp_path/'out'),
        run_scanpy__pp_scale__kwargs={'max_value':4}))
    with planning_context(context), pytest.raises(SemanticPlanCompileError, match='Select one actual operation'):
        compile_semantic_plan(request,candidate,registry,build_semantic_compiler_contract(registry))
