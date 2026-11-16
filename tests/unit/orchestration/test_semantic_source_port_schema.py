"""Response-schema channels use the same authority as semantic compilation."""
import json

import pytest

from agent.orchestration import AgentRequest, ToolRegistry, build_default_tool_registry
from agent.orchestration.active_context import ActiveContextItem, ActivePlanningContext, planning_context
from agent.orchestration.semantic_compiler import (
    SemanticPlanCompileError, build_semantic_compiler_contract, compile_semantic_plan,
)
from agent.orchestration.semantic_wire_v4 import build_semantic_wire_v4_schema, parse_semantic_wire_v4
from agent.schemas import StepOutputRef
from agent.schemas.prior_output import PriorOutputBinding
from schema_expansion import expand_schema


@pytest.fixture(scope="module")
def registry():
    return build_default_tool_registry()


def _scope(registry, *names):
    return ToolRegistry(tuple(registry.get(name) for name in names))


def _request():
    return AgentRequest("source-schema", "Plan native analysis.",
                        dict(input_path="/private/input.h5ad", output_dir="/outputs"))


def _sources(schema, tool, target):
    expanded = expand_schema(schema)
    step = next(s for s in expanded["properties"]["decision"]["anyOf"][0]["properties"]["steps"]["items"]["anyOf"]
                if s["properties"]["tool"]["enum"] == (tool,))
    return next(s["properties"]["source"]["anyOf"] for s in step["properties"]["sources"]["items"]["anyOf"]
                if s["properties"]["target"]["enum"] == (target,))


def _kinds(sources):
    return {s["properties"]["kind"]["enum"][0] for s in sources}


def _payload(producer, function, consumer, following, source):
    def step(name, tool, function, source):
        return dict(step_id=name, tool=tool, control_dependencies=[], sources=[
            dict(target="dataset", source=source),
            dict(target="function_name", source=dict(kind="registered_choice", choice=function))])
    return dict(schema_version=4, decision=dict(kind="plan", steps=[
        step("upstream", producer, function, dict(kind="input", input="input_path")),
        step("downstream", consumer, following, source)]))


def _validate(schema, payload):
    # Independent matcher for the provider's closed-object/anyOf schema subset.
    # Check the complete candidate, not only its source-kind discriminator.
    def matches(rule, value):
        if "anyOf" in rule:
            return any(matches(branch, value) for branch in rule["anyOf"])
        if "enum" in rule:
            return value in rule["enum"]
        kind = rule.get("type")
        if kind == "object":
            return (isinstance(value, dict) and set(rule["required"]) <= value.keys()
                    and value.keys() <= rule["properties"].keys()
                    and all(matches(rule["properties"][key], item) for key, item in value.items()))
        if kind == "array":
            return (isinstance(value, list) and len(value) <= rule.get("maxItems", len(value))
                    and all(matches(rule["items"], item) for item in value))
        return kind is None or isinstance(value, {"string": str, "integer": int}[kind])
    return matches(expand_schema(schema), payload)


def _compile(registry, payload):
    request = _request()
    return compile_semantic_plan(request, parse_semantic_wire_v4(json.dumps(payload), request, registry),
                                 registry, build_semantic_compiler_contract(registry))


@pytest.mark.parametrize("explicit", [False, True])
def test_single_port_keeps_both_upstream_forms(registry, explicit):
    registry = _scope(registry, "run_scanpy")
    schema = build_semantic_wire_v4_schema(registry, _request())
    assert _kinds(_sources(schema, "run_scanpy", "dataset")) == {"input", "step", "step_port"}
    source = dict(kind="step_port", step="upstream", source_port="dataset") if explicit else dict(kind="step", step="upstream")
    payload = _payload("run_scanpy", "pp.pca", "run_scanpy", "pp.neighbors", source)
    assert _validate(schema, payload)
    assert _compile(registry, payload).steps[-1].arguments["input_path"] == StepOutputRef("upstream", "matrix_path")


def test_actual_native_import_to_fragment_qc_requires_explicit_port(registry):
    registry = _scope(registry, "run_snapatac2")
    schema = build_semantic_wire_v4_schema(registry, _request())
    sources = _sources(schema, "run_snapatac2", "dataset")
    assert _kinds(sources) == {"input", "step_port"}
    assert next(s for s in sources if "source_port" in s["properties"])["properties"]["source_port"]["enum"] == ("dataset", "fragments")
    payload = _payload("run_snapatac2", "pp.import_fragments", "run_snapatac2", "metrics.frag_size_distr",
                       dict(kind="step", step="upstream"))
    assert not _validate(schema, payload)
    with pytest.raises(SemanticPlanCompileError) as error:
        _compile(registry, payload)
    assert error.value.code == "AMBIGUOUS_SOURCE_PORT"
    payload["decision"]["steps"][1]["sources"][0]["source"] = dict(kind="step_port", step="upstream", source_port="dataset")
    assert _validate(schema, payload)
    assert _compile(registry, payload).steps[-1].arguments["input_path"] == StepOutputRef("upstream", "matrix_path")


def test_mixed_scope_keeps_single_port_producer_expressible(registry):
    registry = _scope(registry, "run_snapatac2", "run_scanpy")
    schema = build_semantic_wire_v4_schema(registry, _request())
    assert _kinds(_sources(schema, "run_snapatac2", "dataset")) == {"input", "step_port"}
    assert "step" in _kinds(_sources(schema, "run_scanpy", "dataset"))
    payload = _payload("run_scanpy", "pp.pca", "run_snapatac2", "pp.select_features",
                       dict(kind="step_port", step="upstream", source_port="dataset"))
    assert _validate(schema, payload)
    assert _compile(registry, payload).steps[-1].arguments["input_path"] == StepOutputRef("upstream", "matrix_path")
    # Older portless candidates for this uniquely resolvable pair still compile;
    # only the new provider schema applies the conservative mixed-scope rule.
    payload["decision"]["steps"][1]["sources"][0]["source"] = dict(kind="step", step="upstream")
    assert not _validate(schema, payload)
    assert _compile(registry, payload).steps[-1].arguments["input_path"] == StepOutputRef("upstream", "matrix_path")


def test_explicit_native_fragments_channel_remains_available(registry):
    registry = _scope(registry, "run_snapatac2")
    schema = build_semantic_wire_v4_schema(registry, _request())
    payload = _payload("run_snapatac2", "pp.make_fragment_file", "run_snapatac2", "pp.import_fragments",
                       dict(kind="step_port", step="upstream", source_port="fragments"))
    assert _validate(schema, payload)
    assert _compile(registry, payload).steps[-1].arguments["input_path"] == StepOutputRef("upstream", "fragments_path")


@pytest.mark.parametrize("port", ["fragments", "matrix", "unregistered"])
def test_wrong_or_incompatible_explicit_port_still_rejected(registry, port):
    registry = _scope(registry, "run_snapatac2", "run_scanpy")
    schema = build_semantic_wire_v4_schema(registry, _request())
    payload = _payload("run_scanpy", "pp.pca", "run_snapatac2", "pp.select_features",
                       dict(kind="step_port", step="upstream", source_port=port))
    # Mixed schemas cannot correlate a producer step ID with its selected tool.
    assert _validate(schema, payload) == (port == "fragments")
    with pytest.raises(SemanticPlanCompileError) as error:
        _compile(registry, payload)
    assert error.value.code == "WRONG_SOURCE_PORT"


def test_hidden_multi_port_producer_does_not_restrict_visible_scope(registry):
    registry = _scope(registry, "run_scanpy", "run_snapatac2")
    schema = build_semantic_wire_v4_schema(registry, _request(), visible_tool_names=("run_scanpy",))
    assert _kinds(_sources(schema, "run_scanpy", "dataset")) == {"input", "step", "step_port"}


def test_no_compatible_producer_preserves_request_and_historical_sources(registry):
    registry = _scope(registry, "call_scATAC_peaks")
    binding = PriorOutputBinding("private:run", "private-step", "run_snapatac2", "dataset",
        "native_matrix_h5ad.v1", "1"*64, "2"*64, "3"*64)
    with planning_context(ActivePlanningContext("session", "revision", 0, (ActiveContextItem("ctx.0", binding),))):
        schema = build_semantic_wire_v4_schema(registry, _request())
    assert _kinds(_sources(schema, "call_scATAC_peaks", "dataset")) == {"input", "context"}
    assert "private:run" not in json.dumps(schema)


def test_registered_choices_and_all_source_domains_follow_the_compiler(registry):
    schema = build_semantic_wire_v4_schema(registry, _request())
    assert _kinds(_sources(schema, "run_snapatac2", "function_name")) == {"registered_choice"}
    contract = build_semantic_compiler_contract(registry)
    for name in registry.names():
        tool = registry.get(name)
        if tool.planning is None:
            continue
        for port in tool.semantic_planning.consumer_ports:
            matching = [c for c in contract.step_output_channels if c.consumer_tool_name == name
                        and c.target_port == port.name and registry.get(c.producer_tool_name).planning is not None]
            if not matching:
                continue
            sources = _sources(schema, name, port.name)
            choices = next(s for s in sources if "source_port" in s["properties"])
            assert set(choices["properties"]["source_port"]["enum"]) == {c.source_port for c in matching}
            per_producer = [{c.source_port for c in matching if c.producer_tool_name == producer}
                            for producer in {c.producer_tool_name for c in matching}]
            assert ("step" in _kinds(sources)) == all(len(ports) == 1 for ports in per_producer)


def test_planner_can_project_its_existing_contract_without_rebuilding_it(registry, monkeypatch):
    from agent.orchestration import semantic_compiler
    expected = build_semantic_wire_v4_schema(registry, _request())
    contract = build_semantic_compiler_contract(registry)
    monkeypatch.setattr(semantic_compiler, "build_semantic_compiler_contract",
        lambda *_: pytest.fail("An already constructed compiler contract was rebuilt"))
    assert build_semantic_wire_v4_schema(registry, _request(), compiler_contract=contract) == expected
