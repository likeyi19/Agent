"""Offline source binding and bounded repair for current native Registry tools."""
import json

import pytest

from agent.orchestration import AgentRuntime, LLMPlanner, PlanningModelProfile, build_default_tool_registry
from agent.orchestration.semantic_compiler import (
    SemanticPlanCompileError, build_semantic_compiler_contract, compile_semantic_plan,
)
from agent.orchestration.semantic_wire_v4 import parse_semantic_wire_v4
from agent.schemas import AgentRequest, RunMode, StepOutputRef


def _wire(*, inspect_input=True, matrix_port=True):
    def step(name, function, parent, port):
        source = {"kind": "input", "input": "input_path"} if parent is None else {
            "kind": "step_port", "step": parent, "source_port": port}
        if name == "measure" and not matrix_port:
            source = {"kind": "step", "step": parent}
        return dict(step_id=name, tool="run_snapatac2", control_dependencies=[], sources=[
            dict(target="dataset", source=source),
            dict(target="function_name", source=dict(kind="registered_choice", choice=function))])

    return json.dumps(dict(schema_version=4, decision=dict(kind="plan", steps=[
        dict(step_id="inspect", tool="inspect_raw_scATAC", control_dependencies=[],
             sources=[dict(target="raw_input", source=dict(kind="input", input="raw_input_paths"))]
             if inspect_input else []),
        step("extract", "pp.make_fragment_file", None, None),
        step("import", "pp.import_fragments", "extract", "fragments"),
        step("measure", "metrics.frag_size_distr", "import", "dataset"),
    ])))


@pytest.fixture
def agent_request(tmp_path):
    return AgentRequest("native-ports", "Inspect and measure the supplied BAM.", dict(
        input_path="/opaque/input.bam", raw_input_paths=["/opaque/input.bam"],
        output_dir=str(tmp_path / "outputs")), mode=RunMode.PLAN_ONLY)


def _compile(payload, agent_request, registry):
    return compile_semantic_plan(agent_request, parse_semantic_wire_v4(payload, agent_request, registry),
                                 registry, build_semantic_compiler_contract(registry))


def _bindings(plan):
    return [(s.step_id, dict(s.arguments), s.depends_on) for s in plan.steps]


def test_explicit_native_file_and_matrix_ports_keep_distinct_bindings(agent_request):
    plan = _compile(_wire(), agent_request, build_default_tool_registry())
    assert plan.steps[2].arguments["input_path"] == StepOutputRef("extract", "fragments_path")
    assert plan.steps[3].arguments["input_path"] == StepOutputRef("import", "matrix_path")
    assert plan.steps[3].depends_on == ("import",)


def test_native_bare_step_remains_ambiguous_even_with_selected_functions(agent_request):
    with pytest.raises(SemanticPlanCompileError) as caught:
        _compile(_wire(matrix_port=False), agent_request, build_default_tool_registry())
    assert caught.value.code == "AMBIGUOUS_SOURCE_PORT"
    assert caught.value.diagnostic_fields == dict(step_id="measure", tool_name="run_snapatac2",
        target_port="dataset", producer_step_id="import", source_port=None)


def test_unique_offered_raw_bundle_binds_without_a_model_source(agent_request):
    plan = _compile(_wire(inspect_input=False), agent_request, build_default_tool_registry())
    assert list(plan.steps[0].arguments["raw_input_paths"]) == ["/opaque/input.bam"]


def test_raw_inspection_without_an_offered_bundle_reports_the_missing_port(agent_request):
    from dataclasses import replace
    agent_request = replace(agent_request, inputs={k: v for k, v in agent_request.inputs.items() if k != "raw_input_paths"})
    with pytest.raises(SemanticPlanCompileError) as caught:
        _compile(_wire(inspect_input=False), agent_request, build_default_tool_registry())
    assert caught.value.code == "MISSING_REQUIRED_SOURCE"
    assert caught.value.diagnostic_fields["target_port"] == "raw_input"
    assert caught.value.diagnostic_fields["argument_names"] == ["raw_input_paths"]


class _Model:
    model_id = "offline-native-ports"

    def __init__(self, responses):
        self.responses = iter(responses)
        self.prompts = []

    def complete(self, *, prompt, response_schema):
        self.prompts.append(json.loads(prompt))
        return next(self.responses)


def _planner(responses):
    model = _Model(responses)
    return LLMPlanner(model, profile=PlanningModelProfile("offline", "scripted", model.model_id)), model


def test_complete_repair_preserves_other_explicit_native_bindings(agent_request):
    registry = build_default_tool_registry()
    planner, model = _planner([_wire(matrix_port=False), _wire()])
    result = AgentRuntime(planner=planner, registry=registry).run(agent_request)
    assert result.status.value == "PLANNED"
    actual = result.plan
    expected = _compile(_wire(), agent_request, registry)
    assert _bindings(actual) == _bindings(expected)
    assert len(model.prompts) == 2
    assert model.prompts[1]["repair"]["diagnostic"]["required_correction"] == "select_source_port_explicitly"


def test_new_repair_error_does_not_establish_that_the_first_error_was_fixed(agent_request):
    from dataclasses import replace
    agent_request = replace(agent_request, inputs={k: v for k, v in agent_request.inputs.items()
                                                   if k != "raw_input_paths"})
    # Both candidates lack the raw bundle. The replacement additionally loses
    # an explicit native port; channel selection now fails before the later
    # required-argument check. This reproduces the observed diagnostic order
    # without claiming to recover the actual rejected model responses.
    planner, model = _planner([_wire(inspect_input=False),
                              _wire(inspect_input=False, matrix_port=False)])
    result = AgentRuntime(planner=planner, registry=build_default_tool_registry()).run(agent_request)
    assert result.status.value == "FAILED"
    assert result.errors[-1].code == "AMBIGUOUS_SOURCE_PORT"
    assert len(model.prompts) == 2
    assert model.prompts[1]["repair"]["diagnostic"]["required_correction"] == "select_required_semantic_source"
