"""SPP compiler boundary: non-executable holes and authoritative graph rejection."""
import json
from dataclasses import replace

import pytest

from agent.schemas import AgentRequest, RunMode
from agent.orchestration import LLMPlanner, build_default_tool_registry
from agent.orchestration.semantic_compiler import (
    AnalysisDisposition as D, SemanticPlanCandidate as Candidate, SemanticPlanStep as Step,
    SemanticRequestInputSource as Input, SemanticStepOutputSource as Output,
    SemanticRegisteredChoiceSource as Choice, analyze_candidate, compile_semantic_plan,
    build_semantic_compiler_contract, SemanticPlanCompileError, candidate_record, candidate_from_record,
)


@pytest.fixture
def registry():
    return build_default_tool_registry()


def graph():
    return Candidate((Step('inspect', 'inspect_scATAC', (Input('dataset', 'input_path'),)),
        Step('peaks', 'call_scATAC_peaks', (Input('dataset', 'input_path'),)),
        Step('peaks_other', 'call_scATAC_peaks', (Input('dataset', 'input_path'),)),
        Step('pca', 'run_scanpy', (Input('dataset', 'input_path'), Choice('function_name', 'pp.pca')))))


def request(**values):
    return AgentRequest('analysis', 'Analyze my data', dict(input_path='/test/source.h5ad', output_dir='/test/output', **values))


@pytest.mark.parametrize('entrypoint',['acquire_candidate','plan'])
def test_explicit_unsupported_is_terminal_in_candidate_and_full_plan(registry,entrypoint):
    from agent.orchestration.planner import PlannerError
    from agent.orchestration.planning_diagnostics import PlanningDiagnosticStage
    class Model:
        model_id='scripted/terminal-unsupported'
        calls=0
        def complete(self,**kwargs):
            self.calls+=1
            return json.dumps(dict(schema_version=4,decision=dict(kind='unsupported',reason='No supported requested operation.')))
    model=Model()
    with pytest.raises(PlannerError) as caught:
        getattr(LLMPlanner(model),entrypoint)(request(),registry)
    assert caught.value.code=='UNSUPPORTED_REQUEST' and model.calls==1
    assert any(d.stage is PlanningDiagnosticStage.UNSUPPORTED for d in caught.value.diagnostics)
    assert not any(d.code=='PLAN_REPAIR_SCHEDULED' for d in caught.value.diagnostics)


def test_independent_requirements_exact_repeated_steps_and_reanalysis(registry):
    candidate = graph()
    outcome = analyze_candidate(request(), candidate, registry)
    assert outcome.disposition is D.NEEDS_INPUT
    assert [(r.step_id, r.argument) for r in outcome.requirements] == [
        ('peaks', 'effective_genome_size'), ('peaks', 'genome_size_basis'),
        ('peaks_other', 'effective_genome_size'), ('peaks_other', 'genome_size_basis')]
    assert len({r.requirement_id for r in outcome.requirements}) == 4
    assert not hasattr(outcome, 'plan')
    assert candidate_from_record(candidate_record(candidate)) == candidate
    bindings = {s: dict(effective_genome_size=100, genome_size_basis='explicit test applicability')
                for s in ('peaks', 'peaks_other')}
    complete = analyze_candidate(request(), candidate, registry, bindings=bindings)
    assert complete.disposition is D.COMPLETE
    assert complete.candidate_id == outcome.candidate_id
    strict = compile_semantic_plan(request(), candidate, registry, build_semantic_compiler_contract(registry), bindings=bindings)
    assert strict.steps[1].arguments['effective_genome_size'] == 100
    assert strict.steps[3].arguments['function_name'] == 'pp.pca'
    assert len({s.arguments['output_dir'] for s in strict.steps if 'output_dir' in s.arguments}) == 3


@pytest.mark.parametrize('later,code', [
    (Step('bad', 'run_scanpy', (Output('dataset','peaks','unknown'), Choice('function_name','pp.pca'))), 'WRONG_SOURCE_PORT'),
    (Step('bad', 'run_scanpy', (Input('dataset','absent'), Choice('function_name','pp.pca'))), 'UNKNOWN_REQUEST_INPUT'),
    (Step('bad', 'run_scanpy', (Input('nonexistent','input_path'), Choice('function_name','pp.pca'))), 'UNKNOWN_TARGET_PORT'),
])
def test_missing_scalar_never_hides_later_bad_source(registry, later, code):
    candidate = Candidate((*graph().steps[:2], later))
    outcome = analyze_candidate(request(), candidate, registry)
    assert outcome.disposition is D.INVALID and outcome.error.code == code


def test_cycle_is_invalid_before_missing_scalar(registry):
    candidate = graph()
    candidate = Candidate((replace(candidate.steps[0], control_dependencies=('pca',)),
        *candidate.steps[1:3], replace(candidate.steps[3], control_dependencies=('inspect',))))
    outcome = analyze_candidate(request(), candidate, registry)
    assert outcome.disposition is D.INVALID and outcome.error.code == 'INVALID_SEMANTIC_GRAPH'


def test_null_is_invalid_not_a_missing_value(registry):
    outcome = analyze_candidate(request(effective_genome_size=None), Candidate(graph().steps[1:2]), registry)
    assert outcome.disposition is D.INVALID
    assert analyze_candidate(request(), Candidate(graph().steps[1:2]), registry).disposition is D.NEEDS_INPUT


def test_no_incomplete_plan_even_in_plan_only(registry):
    with pytest.raises(SemanticPlanCompileError):
        compile_semantic_plan(replace(request(), mode=RunMode.PLAN_ONLY), graph(), registry,
                              build_semantic_compiler_contract(registry))


def test_missing_external_source_is_distinct_from_managed_output(registry):
    outcome = analyze_candidate(AgentRequest('empty', 'Inspect', {}), Candidate((Step('inspect', 'inspect_scATAC'),)), registry)
    assert outcome.disposition is D.NEEDS_INPUT
    assert outcome.requirements[0].kind == 'external_input'
    outcome = analyze_candidate(AgentRequest('peaks','Call peaks',{'input_path':'/test.h5ad'}),
                               Candidate(graph().steps[1:2]), registry)
    assert next(r for r in outcome.requirements if r.argument == 'output_dir').kind == 'application'


def test_native_function_cannot_borrow_another_functions_port(registry):
    candidate = Candidate((Step('extract', 'run_snapatac2', (Input('dataset','input_path'),
        Choice('function_name','pp.make_fragment_file'))),
        Step('peaks', 'call_scATAC_peaks', (Output('dataset','extract','dataset'),))))
    outcome = analyze_candidate(request(), candidate, registry)
    assert outcome.disposition is D.INVALID and outcome.error.code == 'WRONG_SOURCE_PORT'


def test_provider_returns_incomplete_candidate_without_repair(registry):
    class Model:
        model_id = 'test'
        calls = 0
        def complete(self, **kwargs):
            self.calls += 1
            return json.dumps(dict(schema_version=4,decision=dict(kind='plan',steps=[dict(
                step_id='peaks',tool='call_scATAC_peaks',sources=[dict(target='dataset',source=dict(kind='input',input='input_path'))],
                control_dependencies=[])])))
    model = Model()
    acquired = LLMPlanner(model).acquire_candidate(request(), registry)
    assert acquired.attempt.analysis.disposition is D.NEEDS_INPUT
    assert acquired.preflight is None
    assert acquired.provider_call_count == model.calls == 1
    assert not hasattr(acquired.attempt, 'plan')
