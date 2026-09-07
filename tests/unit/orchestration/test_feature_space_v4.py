"""Feature-space wire parity through PLAN_ONLY, without scientific execution."""

from dataclasses import replace
import json
from unittest.mock import Mock

import pytest

from agent.orchestration import (
    AgentRequest,
    AgentRuntime,
    LLMPlanner,
    PlanningWireMode,
    RunMode,
    RunStatus,
    StepOutputRef,
    ToolRegistry,
    build_default_tool_registry,
    build_semantic_wire_v4_schema,
)
from benchmarks.planner.benchmark import ScriptedPlanningModel


FEATURE_TOOL = "validate_scATAC_feature_space"
FEATURE_PARAMETERS = (
    "layer_key", "feature_chrom_key", "feature_start_key", "feature_end_key",
    "coordinate_system", "semantics_metadata_key",
)
COORDINATES = {
    "coordinate_source": "var_columns",
    "feature_chrom_key": "chromosome",
    "feature_start_key": "start",
    "feature_end_key": "end",
    "coordinate_system": "zero_based_half_open",
}
VARIANTS = (
    pytest.param({}, id="raw-X"),
    pytest.param({"matrix_source": "layer", "layer_key": "counts"}, id="layer"),
    pytest.param(COORDINATES, id="zero-based-coordinates"),
    pytest.param(
        {**COORDINATES, "coordinate_system": "one_based_closed"},
        id="one-based-coordinates",
    ),
    pytest.param({"semantics_metadata_key": "matrix_semantics"}, id="raw-uns"),
    pytest.param(
        {**COORDINATES, "matrix_source": "layer", "layer_key": "counts",
         "semantics_metadata_key": "matrix_semantics"},
        id="combined",
    ),
    pytest.param(dict.fromkeys(FEATURE_PARAMETERS), id="explicit-null"),
)


@pytest.fixture
def guarded_registry():
    registry = build_default_tool_registry()
    guard = Mock(side_effect=AssertionError("PLAN_ONLY executed science"))
    yield ToolRegistry(tuple(
        replace(registry.get(name), function=guard) for name in registry.names()
    ))
    guard.assert_not_called()


def _feature_inputs(extra):
    return {
        "input_path": "/synthetic/raw.h5ad", "output_dir": "/synthetic/output",
        "matrix_source": "X", "matrix_semantics": "fragment_counts",
        "species": "human", "genome_assembly": "hg38", "coordinate_source": "none",
        "overwrite": False, **extra,
    }


def _input_source(name):
    return {"target": name, "source": {"kind": "input", "input": name}}


def _step(step_id, tool, sources=()):
    return dict(step_id=step_id, tool=tool, sources=list(sources), control_dependencies=[])


def _v4(steps):
    return json.dumps(dict(schema_version=4, decision=dict(kind="plan", steps=steps)))


def _run(registry, inputs, payload, mode=PlanningWireMode.V4):
    return AgentRuntime(
        planner=LLMPlanner(ScriptedPlanningModel(payload), wire_mode=mode),
        registry=registry,
    ).run(AgentRequest("feature-parity", "Plan the explicit feature workflow.", inputs,
                       mode=RunMode.PLAN_ONLY))


@pytest.mark.parametrize("extra", VARIANTS)
@pytest.mark.parametrize("workflow", ("validation", "pseudobulk", "independent", "paired"))
def test_feature_variants_have_exact_v3_plan_only_parity(guarded_registry, extra, workflow):
    inputs = _feature_inputs(extra)
    expected = [("f", FEATURE_TOOL, dict(inputs), ())]
    # Exercise both explicit new sources and deterministic unique binding.
    sources = [_input_source(name) for name in (*FEATURE_PARAMETERS, "overwrite")
               if name in inputs]
    steps = [_step("f", FEATURE_TOOL, sources)]
    if workflow != "validation":
        bulk = dict(replicate_key="donor", group_key="cell_type", condition_key="condition",
                    group_source="raw_obs", output_dir=inputs["output_dir"])
        inputs.update(bulk)
        expected.append(("p", "build_replicate_pseudobulk",
                         {**bulk, "feature_space_path": StepOutputRef("f", "feature_space_path")},
                         ("f",)))
        steps.append(_step("p", "build_replicate_pseudobulk", (
            {"target": "feature_space", "source": {"kind": "step", "step": "f"}},
        )))
    if workflow in {"independent", "paired"}:
        da = dict(group_value="T", condition_key="condition", numerator_condition="treated",
                  denominator_condition="control", design_type=workflow,
                  output_dir=inputs["output_dir"])
        inputs.update(da)
        expected.append(("d", "run_replicate_differential_accessibility",
                         {**da, "pseudobulk_path": StepOutputRef("p", "pseudobulk_path")}, ("p",)))
        steps.append(_step("d", "run_replicate_differential_accessibility", (
            {"target": "pseudobulk", "source": {"kind": "step_port", "step": "p",
                                                "source_port": "pseudobulk"}},
        )))

    v3_steps = []
    for step_id, tool, arguments, dependencies in expected:
        bindings = dict.fromkeys(guarded_registry.get(tool).optional_arguments)
        for name, value in arguments.items():
            bindings[name] = (
                dict(binding_type="ref", ref_step_id=value.step_id, ref_output_key=value.output_key)
                if isinstance(value, StepOutputRef)
                else dict(binding_type="input", input_name=name)
            )
        v3_steps.append(dict(step_id=step_id, tool_name=tool, arguments=bindings,
                             depends_on=list(dependencies), description="Explicit workflow"))
    v3 = json.dumps(dict(schema_version=3, status="plan", steps=v3_steps, reason=None))
    for mode, payload in ((PlanningWireMode.V3, v3), (PlanningWireMode.V4, _v4(steps))):
        result = _run(guarded_registry, inputs, payload, mode)
        assert result.status is RunStatus.PLANNED, result.errors
        assert result.verification.passed
        assert tuple((s.step_id, s.tool_name, dict(s.arguments), s.depends_on)
                     for s in result.plan.steps) == tuple(expected)
        assert result.plan.steps[0].arguments["overwrite"] is False
        for name in FEATURE_PARAMETERS:
            if name not in extra:
                assert name not in result.plan.steps[0].arguments
            elif extra[name] is None:
                assert result.plan.steps[0].arguments[name] is None

    # The same reviewed unique inputs can be lowered without redundant sources.
    steps[0]["sources"] = [_input_source("overwrite")]
    omitted = _run(guarded_registry, inputs, _v4(steps))
    assert omitted.status is RunStatus.PLANNED, omitted.errors
    assert dict(omitted.plan.steps[0].arguments) == expected[0][2]


def test_feature_schema_exposes_only_reviewed_new_targets(guarded_registry):
    request = AgentRequest("schema", "Validate.", _feature_inputs(COORDINATES))
    schema = build_semantic_wire_v4_schema(guarded_registry, request)
    for step in schema["$defs"]["step"]["anyOf"]:
        tool = step["properties"]["tool"]["enum"][0]
        targets = step["properties"]["sources"]["items"]["properties"]["target"]["enum"]
        assert set(FEATURE_PARAMETERS).intersection(targets) == (
            set(FEATURE_PARAMETERS) if tool == FEATURE_TOOL else set()
        )


@pytest.mark.parametrize("name", FEATURE_PARAMETERS)
@pytest.mark.parametrize("invalid", ("wrong-input", "upstream", "duplicate", "wrong-type"))
def test_feature_sources_fail_closed(guarded_registry, name, invalid):
    inputs = _feature_inputs({name: "zero_based_half_open" if name == "coordinate_system" else "key",
                              "unreviewed_alias": "key"})
    steps = [_step("f", FEATURE_TOOL, [_input_source(name)])]
    if invalid == "wrong-input":
        steps[0]["sources"][0]["source"]["input"] = "unreviewed_alias"
        code = "UNAUTHORIZED_REQUEST_INPUT"
    elif invalid == "upstream":
        steps.insert(0, _step("i", "inspect_scATAC"))
        steps[1]["sources"][0]["source"] = {"kind": "step", "step": "i"}
        code = "ZERO_VALID_CHANNELS"
    elif invalid == "duplicate":
        steps[0]["sources"].append(_input_source(name))
        code = "DUPLICATE_TARGET_SOURCE"
    else:
        inputs[name] = {"invalid": True}
        code = "INVALID_REQUEST_BINDING"
    result = _run(guarded_registry, inputs, _v4(steps))
    assert result.status is RunStatus.FAILED
    assert result.errors[0].code == code
    assert result.plan is None


def test_invalid_coordinate_choice_fails_during_planning(guarded_registry):
    result = _run(guarded_registry, _feature_inputs({**COORDINATES, "coordinate_system": "guessed"}),
                  _v4([_step("f", FEATURE_TOOL)]))
    assert result.status is RunStatus.FAILED
    assert result.errors[0].code == "INVALID_REQUEST_BINDING"


def test_repeated_feature_steps_require_explicit_optional_scope(guarded_registry):
    inputs = _feature_inputs({"semantics_metadata_key": "matrix_semantics"})
    inputs.pop("overwrite")
    steps = [_step("a", FEATURE_TOOL), _step("b", FEATURE_TOOL)]
    ambiguous = _run(guarded_registry, inputs, _v4(steps))
    assert ambiguous.status is RunStatus.FAILED
    assert ambiguous.errors[0].code == "AMBIGUOUS_OPTIONAL_INPUT_SCOPE"
    steps[1]["sources"] = [_input_source("semantics_metadata_key")]
    scoped = _run(guarded_registry, inputs, _v4(steps))
    assert scoped.status is RunStatus.PLANNED
    assert "semantics_metadata_key" not in scoped.plan.steps[0].arguments
    assert scoped.plan.steps[1].arguments["semantics_metadata_key"] == "matrix_semantics"
