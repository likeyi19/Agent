"""Offline tests for the strict provider-neutral LLM planner."""

from __future__ import annotations

from dataclasses import replace
import json
from unittest.mock import Mock

import pytest

from agent.orchestration import (
    AgentRequest,
    AgentRuntime,
    DeterministicPlanner,
    ErrorCategory,
    LLMPlanner,
    Planner,
    PlannerError,
    PlanningModel,
    RunMode,
    RunStatus,
    StepOutputRef,
    ToolRegistry,
    build_default_tool_registry,
)
from agent.orchestration.llm_planner import (
    _build_prompt,
    _catalog_fingerprint,
    _response_schema,
)


class FakePlanningModel:
    def __init__(
        self,
        response: object,
        *,
        model_id: str = "fake-planner-v1",
        error: Exception | None = None,
    ) -> None:
        self.response = response
        self._model_id = model_id
        self.error = error
        self.calls: list[tuple[str, object]] = []

    @property
    def model_id(self) -> str:
        return self._model_id

    def complete(self, *, prompt: str, response_schema) -> str:
        self.calls.append((prompt, response_schema))
        if self.error is not None:
            raise self.error
        return self.response  # type: ignore[return-value]


@pytest.fixture
def registry() -> ToolRegistry:
    return build_default_tool_registry()


def _request(
    inputs: dict[str, object] | None = None,
    *,
    mode: RunMode = RunMode.EXECUTE,
) -> AgentRequest:
    return AgentRequest(
        "request-1",
        "Please inspect this scATAC dataset and compute embeddings when requested.",
        inputs or {"input_path": "/data/input.h5ad"},
        mode,
    )


def _input_binding(input_name: str) -> dict[str, object]:
    return {
        "binding_type": "input",
        "input_name": input_name,
    }


def _ref_binding(step_id: str, output_key: str) -> dict[str, object]:
    return {
        "binding_type": "ref",
        "ref_step_id": step_id,
        "ref_output_key": output_key,
    }


def _arguments(
    tool_name: str,
    **bindings: object,
) -> dict[str, object]:
    spec = build_default_tool_registry().get(tool_name)
    arguments: dict[str, object] = {
        name: None for name in spec.optional_arguments
    }
    arguments.update(bindings)
    return arguments


def _inspect_step(**changes) -> dict[str, object]:
    step: dict[str, object] = {
        "step_id": "inspect",
        "tool_name": "inspect_scATAC",
        "arguments": _arguments(
            "inspect_scATAC", path=_input_binding("input_path")
        ),
        "depends_on": [],
        "description": "Inspect the supplied dataset.",
    }
    step.update(changes)
    return step


def _embed_step(**changes) -> dict[str, object]:
    step: dict[str, object] = {
        "step_id": "embed",
        "tool_name": "epizoo_embed_cells",
        "arguments": _arguments(
            "epizoo_embed_cells",
            input_path=_ref_binding("inspect", "input_path"),
            output_dir=_input_binding("output_dir"),
            species=_input_binding("species"),
        ),
        "depends_on": ["inspect"],
        "description": "Persist EpiZoo cell embeddings.",
    }
    step.update(changes)
    return step


def _neighbors_step(**changes) -> dict[str, object]:
    step: dict[str, object] = {
        "step_id": "neighbors",
        "tool_name": "build_cell_neighbors",
        "arguments": _arguments(
            "build_cell_neighbors",
            embedding_path=_ref_binding("embed", "embedding_path"),
            cell_ids_path=_ref_binding("embed", "cell_ids_path"),
            output_dir=_input_binding("output_dir"),
        ),
        "depends_on": ["embed"],
        "description": "Build a sparse neighbor graph.",
    }
    step.update(changes)
    return step


def _cluster_step(**changes) -> dict[str, object]:
    step: dict[str, object] = {
        "step_id": "cluster",
        "tool_name": "cluster_cells",
        "arguments": _arguments(
            "cluster_cells",
            analysis_path=_ref_binding("neighbors", "analysis_path"),
            output_dir=_input_binding("output_dir"),
        ),
        "depends_on": ["neighbors"],
        "description": "Cluster cells with Leiden.",
    }
    step.update(changes)
    return step


def _umap_step(**changes) -> dict[str, object]:
    step: dict[str, object] = {
        "step_id": "umap",
        "tool_name": "compute_cell_umap",
        "arguments": _arguments(
            "compute_cell_umap",
            analysis_path=_ref_binding("cluster", "analysis_path"),
            output_dir=_input_binding("output_dir"),
        ),
        "depends_on": ["cluster"],
        "description": "Compute a two-dimensional UMAP.",
    }
    step.update(changes)
    return step


def _evaluation_step(**changes) -> dict[str, object]:
    step: dict[str, object] = {
        "step_id": "evaluate",
        "tool_name": "evaluate_cell_clustering",
        "arguments": _arguments(
            "evaluate_cell_clustering",
            analysis_path=_ref_binding("cluster", "analysis_path"),
            reference_h5ad_path=_ref_binding("inspect", "input_path"),
            label_key=_input_binding("label_key"),
            output_dir=_input_binding("output_dir"),
        ),
        "depends_on": ["cluster", "inspect"],
        "description": "Evaluate fixed clustering labels.",
    }
    step.update(changes)
    return step


def _transfer_step(**changes) -> dict[str, object]:
    step: dict[str, object] = {
        "step_id": "transfer",
        "tool_name": "transfer_cell_labels",
        "arguments": _arguments(
            "transfer_cell_labels",
            reference_embedding_path=_ref_binding(
                "embed_reference", "embedding_path"
            ),
            reference_cell_ids_path=_ref_binding(
                "embed_reference", "cell_ids_path"
            ),
            reference_h5ad_path=_ref_binding("inspect_reference", "input_path"),
            reference_label_key=_input_binding("reference_label_key"),
            reference_species=_ref_binding("embed_reference", "species"),
            reference_checkpoint_path=_ref_binding(
                "embed_reference", "checkpoint_path"
            ),
            query_embedding_path=_ref_binding("embed_query", "embedding_path"),
            query_cell_ids_path=_ref_binding("embed_query", "cell_ids_path"),
            query_h5ad_path=_ref_binding("inspect_query", "input_path"),
            query_species=_ref_binding("embed_query", "species"),
            query_checkpoint_path=_ref_binding("embed_query", "checkpoint_path"),
            output_dir=_input_binding("output_dir"),
        ),
        "depends_on": [
            "inspect_reference",
            "embed_reference",
            "inspect_query",
            "embed_query",
        ],
        "description": "Transfer biological labels.",
    }
    step.update(changes)
    return step


def _da_step(**changes) -> dict[str, object]:
    step: dict[str, object] = {
        "step_id": "da",
        "tool_name": "run_replicate_differential_accessibility",
        "arguments": _arguments(
            "run_replicate_differential_accessibility",
            pseudobulk_path=_input_binding("pseudobulk_path"),
            group_value=_input_binding("group_value"),
            condition_key=_input_binding("condition_key"),
            numerator_condition=_input_binding("numerator_condition"),
            denominator_condition=_input_binding("denominator_condition"),
            design_type=_input_binding("design_type"),
            output_dir=_input_binding("output_dir"),
        ),
        "depends_on": [],
        "description": "Run fixed replicate-aware differential accessibility.",
    }
    step.update(changes)
    return step


def _plan_response(*steps: dict[str, object], **extra) -> str:
    payload: dict[str, object] = {
        "schema_version": 3,
        "status": "plan",
        "steps": list(steps),
        "reason": None,
    }
    payload.update(extra)
    return json.dumps(payload)


def _planner(response: object, **kwargs) -> tuple[LLMPlanner, FakePlanningModel]:
    model = FakePlanningModel(response, **kwargs)
    return LLMPlanner(model), model


def test_llm_planner_and_fake_model_satisfy_protocols() -> None:
    planner, model = _planner(_plan_response(_inspect_step()))

    assert isinstance(model, PlanningModel)
    assert isinstance(planner, Planner)


def test_one_step_inspection_plan(registry) -> None:
    planner, _ = _planner(_plan_response(_inspect_step()))

    plan = planner.plan(_request(), registry)

    assert plan.request_id == "request-1"
    assert plan.planner_name == "llm:fake-planner-v1"
    assert len(plan.steps) == 1
    assert plan.steps[0].tool_name == "inspect_scATAC"
    assert dict(plan.steps[0].arguments) == {"path": "/data/input.h5ad"}
    assert plan.steps[0].description == "Inspect the supplied dataset."


def test_bounded_schema_v3_da_plan_uses_only_request_bindings(registry) -> None:
    planner, model = _planner(_plan_response(_da_step()))
    inputs = {
        "pseudobulk_path": "/data/pseudobulk.h5ad",
        "group_value": "T",
        "condition_key": "condition",
        "numerator_condition": "treated",
        "denominator_condition": "control",
        "design_type": "independent",
        "output_dir": "/output",
    }
    plan = planner.plan(_request(inputs), registry)
    assert plan.steps[0].tool_name == "run_replicate_differential_accessibility"
    assert dict(plan.steps[0].arguments) == inputs
    prompt, schema = model.calls[0]
    assert schema["properties"]["schema_version"]["enum"] == (3,)
    serialized = json.dumps(schema).casefold()
    for prohibited in ("rscript", "r command", "formula", "shell execution"):
        assert prohibited not in serialized
    assert "rscript" not in prompt.casefold()


def test_llm_da_plan_only_executes_zero_scientific_tools(registry) -> None:
    planner, _ = _planner(_plan_response(_da_step()))
    guard = Mock(side_effect=AssertionError("PLAN_ONLY invoked a scientific tool"))
    guarded = ToolRegistry(
        tuple(
            replace(registry.get(name), function=guard)
            for name in registry.names()
        )
    )
    result = AgentRuntime(planner=planner, registry=guarded).run(
        _request(
            {
                "pseudobulk_path": "/data/pseudobulk.h5ad",
                "group_value": "T",
                "condition_key": "condition",
                "numerator_condition": "treated",
                "denominator_condition": "control",
                "design_type": "independent",
                "output_dir": "/output",
            },
            mode=RunMode.PLAN_ONLY,
        )
    )
    assert result.status is RunStatus.PLANNED
    guard.assert_not_called()


def test_two_step_plan_binds_inputs_and_converts_reference(registry) -> None:
    planner, _ = _planner(_plan_response(_inspect_step(), _embed_step()))
    request = _request(
        {
            "input_path": "/data/input.h5ad",
            "output_dir": "/output",
            "species": "mouse",
        }
    )

    plan = planner.plan(request, registry)

    assert tuple(step.step_id for step in plan.steps) == ("inspect", "embed")
    assert plan.steps[1].arguments["input_path"] == StepOutputRef(
        "inspect", "input_path"
    )
    assert plan.steps[1].arguments["output_dir"] == "/output"
    assert plan.steps[1].arguments["species"] == "mouse"
    assert plan.steps[1].depends_on == ("inspect",)


def test_llm_label_transfer_plan_uses_only_inputs_and_upstream_references(
    registry,
) -> None:
    inspect_reference = _inspect_step(
        step_id="inspect_reference",
        arguments=_arguments(
            "inspect_scATAC", path=_input_binding("reference_input_path")
        ),
    )
    embed_reference = _embed_step(
        step_id="embed_reference",
        arguments=_arguments(
            "epizoo_embed_cells",
            input_path=_ref_binding("inspect_reference", "input_path"),
            output_dir=_input_binding("output_dir"),
            species=_input_binding("species"),
            checkpoint_path=_input_binding("checkpoint_path"),
        ),
        depends_on=["inspect_reference"],
    )
    inspect_query = _inspect_step(
        step_id="inspect_query",
        arguments=_arguments(
            "inspect_scATAC", path=_input_binding("query_input_path")
        ),
    )
    embed_query = _embed_step(
        step_id="embed_query",
        arguments=_arguments(
            "epizoo_embed_cells",
            input_path=_ref_binding("inspect_query", "input_path"),
            output_dir=_input_binding("output_dir"),
            species=_input_binding("species"),
            checkpoint_path=_input_binding("checkpoint_path"),
        ),
        depends_on=["inspect_query"],
    )
    planner, _ = _planner(
        _plan_response(
            inspect_reference,
            embed_reference,
            inspect_query,
            embed_query,
            _transfer_step(),
        )
    )
    request = _request(
        {
            "reference_input_path": "/data/reference.h5ad",
            "query_input_path": "/data/query.h5ad",
            "output_dir": "/output",
            "species": "mouse",
            "reference_label_key": "celltype",
            "checkpoint_path": "/models/epizoo.pth",
        }
    )
    plan = planner.plan(request, registry)
    transfer = plan.steps[-1]
    assert transfer.tool_name == "transfer_cell_labels"
    assert transfer.arguments["reference_species"] == StepOutputRef(
        "embed_reference", "species"
    )
    assert transfer.arguments["query_checkpoint_path"] == StepOutputRef(
        "embed_query", "checkpoint_path"
    )
    assert "n_neighbors" not in transfer.arguments
    assert "metric" not in transfer.arguments
    assert "min_confidence" not in transfer.arguments

def test_five_step_plan_uses_only_input_and_reference_bindings(registry) -> None:
    planner, _ = _planner(
        _plan_response(
            _inspect_step(),
            _embed_step(),
            _neighbors_step(),
            _cluster_step(),
            _umap_step(),
        )
    )
    request = _request(
        {
            "input_path": "/data/input.h5ad",
            "output_dir": "/output",
            "species": "mouse",
        }
    )
    plan = planner.plan(request, registry)
    assert tuple(step.tool_name for step in plan.steps) == registry.names()[:5]
    assert plan.steps[2].arguments["embedding_path"] == StepOutputRef(
        "embed", "embedding_path"
    )
    assert plan.steps[2].arguments["cell_ids_path"] == StepOutputRef(
        "embed", "cell_ids_path"
    )
    assert plan.steps[3].arguments["analysis_path"] == StepOutputRef(
        "neighbors", "analysis_path"
    )
    assert plan.steps[4].arguments["analysis_path"] == StepOutputRef(
        "cluster", "analysis_path"
    )
    assert "n_neighbors" not in plan.steps[2].arguments
    assert "resolution" not in plan.steps[3].arguments
    assert "min_dist" not in plan.steps[4].arguments


def test_llm_evaluation_plan_uses_only_inputs_and_references(registry) -> None:
    planner, _ = _planner(
        _plan_response(
            _inspect_step(),
            _embed_step(),
            _neighbors_step(),
            _cluster_step(),
            _evaluation_step(),
        )
    )
    plan = planner.plan(
        _request(
            {
                "input_path": "/data/input.h5ad",
                "output_dir": "/output",
                "species": "mouse",
                "label_key": "celltype",
            }
        ),
        registry,
    )
    evaluation = plan.steps[-1]
    assert evaluation.arguments["analysis_path"] == StepOutputRef(
        "cluster", "analysis_path"
    )
    assert evaluation.arguments["reference_h5ad_path"] == StepOutputRef(
        "inspect", "input_path"
    )
    assert evaluation.arguments["label_key"] == "celltype"
    assert "cluster_key" not in evaluation.arguments


def test_request_input_values_are_preserved_exactly(registry) -> None:
    arguments = dict(_embed_step()["arguments"])
    arguments.update(
        {
            "checkpoint_path": _input_binding("checkpoint_path"),
            "device": _input_binding("device"),
            "overwrite": _input_binding("overwrite"),
        }
    )
    planner, _ = _planner(
        _plan_response(_inspect_step(), _embed_step(arguments=arguments))
    )
    inputs = {
        "input_path": "/data/Input With Spaces.h5ad",
        "output_dir": "/output/Exact Case",
        "species": "mouse",
        "checkpoint_path": "/models/Exact Name.pth",
        "device": "cuda:0",
        "overwrite": False,
    }

    plan = planner.plan(_request(inputs), registry)

    embedding_arguments = plan.steps[1].arguments
    for name in ("output_dir", "species", "checkpoint_path", "device", "overwrite"):
        assert embedding_arguments[name] == inputs[name]
    assert plan.steps[0].arguments["path"] == inputs["input_path"]


@pytest.mark.parametrize(
    "literal",
    ["/invented/input.h5ad", "mouse", False, 4, None, ["value"]],
)
def test_model_cannot_invent_literal_executable_values(registry, literal) -> None:
    planner, _ = _planner(
        _plan_response(
            _inspect_step(arguments={"path": literal}),
        )
    )

    with pytest.raises(PlannerError) as raised:
        planner.plan(_request(), registry)

    assert raised.value.code == "PLANNER_BINDING_INVALID"
    assert raised.value.category is ErrorCategory.INTERNAL_AGENT_ERROR


def test_plan_id_is_deterministic_and_content_derived(registry) -> None:
    response = _plan_response(_inspect_step())
    first, _ = _planner(response)
    second, _ = _planner(response)

    first_plan = first.plan(_request(), registry)
    second_plan = second.plan(_request(), registry)
    changed_plan = second.plan(
        _request({"input_path": "/data/different.h5ad"}), registry
    )

    assert first_plan.plan_id == second_plan.plan_id
    assert first_plan.plan_id.startswith("request-1:llm:")
    assert first_plan.plan_id != changed_plan.plan_id


def test_planner_name_contains_only_sanitized_model_identity(registry) -> None:
    planner, _ = _planner(
        _plan_response(_inspect_step()), model_id=" Provider / Model v1 "
    )

    plan = planner.plan(_request(), registry)

    assert planner.name == "llm:Provider-Model-v1"
    assert plan.planner_name == planner.name


def test_model_is_called_exactly_once_with_deterministic_prompt(registry) -> None:
    planner, model = _planner(_plan_response(_inspect_step()))

    planner.plan(_request(), registry)

    assert len(model.calls) == 1
    prompt, response_schema = model.calls[0]
    assert json.loads(prompt)["request"]["prompt"].startswith("Please inspect")
    json.dumps(response_schema, allow_nan=False)


def test_response_schema_is_strict_v3_tool_discriminated_and_registry_derived(
    registry,
) -> None:
    planner, model = _planner(_plan_response(_inspect_step()))
    request = _request(
        {"input_path": "/data/input.h5ad", "output_dir": "/output"}
    )

    planner.plan(request, registry)

    schema = model.calls[0][1]
    assert schema["type"] == "object"
    assert "anyOf" not in schema
    assert set(schema["$defs"]) == {
        "input",
        "ref",
        "input_or_ref",
        "input_or_null",
        "input_or_ref_or_null",
        "ref_or_null",
    }
    assert schema["required"] == (
        "schema_version",
        "status",
        "steps",
        "reason",
    )
    assert schema["properties"]["schema_version"]["enum"] == (3,)
    assert schema["properties"]["reason"]["type"] == ("string", "null")
    step_union = schema["properties"]["steps"]["items"]
    assert set(step_union) == {"anyOf"}
    branches = step_union["anyOf"]
    assert len(branches) == len(registry.names())
    by_tool = {
        branch["properties"]["tool_name"]["enum"][0]: branch
        for branch in branches
    }
    assert tuple(by_tool) == registry.names()

    for tool_name in registry.names():
        branch = by_tool[tool_name]
        assert branch["type"] == "object"
        assert branch["additionalProperties"] is False
        assert set(branch["required"]) == set(branch["properties"])
        assert branch["properties"]["tool_name"]["enum"] == (tool_name,)
        argument_schema = branch["properties"]["arguments"]
        tool_spec = registry.get(tool_name)
        expected_arguments = set(tool_spec.required_arguments).union(
            tool_spec.optional_arguments
        )
        assert set(argument_schema["properties"]) == expected_arguments
        assert set(argument_schema["required"]) == expected_arguments
        assert argument_schema["additionalProperties"] is False

    inspect_arguments = by_tool["inspect_scATAC"]["properties"]["arguments"]
    assert set(inspect_arguments["properties"]) == {"path"}
    embed_arguments = by_tool["epizoo_embed_cells"]["properties"]["arguments"]
    assert "path" not in embed_arguments["properties"]
    assert set(embed_arguments["properties"]) == {
        "input_path",
        "output_dir",
        "species",
        "checkpoint_path",
        "device",
        "overwrite",
    }

    required_binding = schema["$defs"]["input_or_ref"]
    input_ref, output_ref = required_binding["anyOf"]
    assert input_ref == {"$ref": "#/$defs/input"}
    assert output_ref == {"$ref": "#/$defs/ref"}
    input_variant = schema["$defs"]["input"]
    ref_variant = schema["$defs"]["ref"]
    assert input_variant["properties"]["input_name"]["enum"] == (
        "input_path",
        "output_dir",
    )
    assert set(input_variant["properties"]) == {"binding_type", "input_name"}
    assert input_variant["properties"]["binding_type"]["enum"] == ("input",)
    assert set(ref_variant["properties"]) == {
        "binding_type",
        "ref_step_id",
        "ref_output_key",
    }
    assert ref_variant["properties"]["binding_type"]["enum"] == ("ref",)
    optional_schema = embed_arguments["properties"]["checkpoint_path"]
    assert optional_schema == {"$ref": "#/$defs/input_or_ref_or_null"}
    assert schema["$defs"]["input_or_ref_or_null"]["anyOf"] == (
        {"$ref": "#/$defs/input"},
        {"$ref": "#/$defs/ref"},
        {"type": "null"},
    )

    banned_keywords = {
        "oneOf",
        "allOf",
        "not",
        "if",
        "then",
        "else",
        "patternProperties",
    }

    def visit(node: object) -> None:
        if isinstance(node, dict):
            assert banned_keywords.isdisjoint(node)
            if set(node) == {"$ref"}:
                reference = node["$ref"]
                assert isinstance(reference, str)
                assert reference.startswith("#/$defs/")
                assert reference.removeprefix("#/$defs/") in schema["$defs"]
            if set(node) == {"anyOf"}:
                assert all(
                    not (isinstance(value, dict) and set(value) == {"anyOf"})
                    for value in node["anyOf"]
                )
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            if "additionalProperties" in node:
                assert node["additionalProperties"] is False
            for value in node.values():
                visit(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                visit(value)

    visit(schema)


def test_input_only_and_composable_binding_defs_preserve_requiredness() -> None:
    source = build_default_tool_registry()
    embed = source.get("epizoo_embed_cells")
    required = dict(embed.required_arguments)
    optional = dict(embed.optional_arguments)
    required["species"] = replace(
        required["species"], allow_step_output_ref=False
    )
    optional["device"] = replace(optional["device"], allow_step_output_ref=False)
    registry = ToolRegistry(
        tuple(
            replace(
                source.get(name),
                required_arguments=required,
                optional_arguments=optional,
            )
            if name == embed.name
            else source.get(name)
            for name in source.names()
        )
    )
    schema = _response_schema(
        registry,
        _request({"input_path": "/data/input.h5ad", "species": "mouse"}),
    )
    branch = next(
        item
        for item in schema["properties"]["steps"]["items"]["anyOf"]
        if item["properties"]["tool_name"]["enum"] == (embed.name,)
    )
    arguments = branch["properties"]["arguments"]["properties"]

    assert arguments["species"] == {"$ref": "#/$defs/input"}
    assert arguments["input_path"] == {"$ref": "#/$defs/input_or_ref"}
    assert arguments["device"] == {"$ref": "#/$defs/input_or_null"}
    assert arguments["checkpoint_path"] == {
        "$ref": "#/$defs/input_or_ref_or_null"
    }
    assert schema["$defs"]["input_or_null"]["anyOf"] == (
        {"$ref": "#/$defs/input"},
        {"type": "null"},
    )
    assert "literal" not in json.dumps(schema, sort_keys=True)


def test_full_catalog_schema_and_prompt_have_deterministic_size_headroom() -> None:
    registry = build_default_tool_registry()
    request = AgentRequest(
        "size-regression",
        "Inspect this scATAC-seq dataset and summarize its stored matrix metadata.",
        {"input_path": "/synthetic/inspect.h5ad"},
    )
    catalog_fingerprint = _catalog_fingerprint(registry)
    schema = _response_schema(registry, request)
    repeated_schema = _response_schema(registry, request)
    serialized_schema = json.dumps(
        schema,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    prompt = _build_prompt(request, registry)

    def count_key(value: object, key: str) -> int:
        if isinstance(value, dict):
            return int(key in value) + sum(
                count_key(item, key) for item in value.values()
            )
        if isinstance(value, (list, tuple)):
            return sum(count_key(item, key) for item in value)
        return 0

    def count_objects(value: object) -> int:
        if isinstance(value, dict):
            return int(value.get("type") == "object") + sum(
                count_objects(item) for item in value.values()
            )
        if isinstance(value, (list, tuple)):
            return sum(count_objects(item) for item in value)
        return 0

    assert schema == repeated_schema
    assert _catalog_fingerprint(registry) == catalog_fingerprint
    assert len(serialized_schema.encode("utf-8")) <= 13_000
    assert len(prompt.encode("utf-8")) <= 18_000
    assert len(serialized_schema.encode("utf-8")) + len(
        prompt.encode("utf-8")
    ) <= 31_000
    assert count_key(schema, "anyOf") <= 5
    assert count_objects(schema) <= 30
    assert count_key(schema, "$ref") >= 80


@pytest.mark.parametrize(
    "prompt",
    [
        "Inspect the supplied scATAC data.",
        "Embed the cells.",
        "Build neighbors, cluster cells, and compute UMAP.",
        "Transfer reference labels to the query.",
        "Build replicate pseudobulk counts.",
        "Run differential accessibility.",
        "Write an unsupported poem.",
    ],
)
def test_every_request_exposes_the_same_complete_tool_catalog(
    registry: ToolRegistry,
    prompt: str,
) -> None:
    request = AgentRequest(
        "full-catalog",
        prompt,
        {"input_path": "/synthetic/input.h5ad"},
    )
    prompt_payload = json.loads(_build_prompt(request, registry))
    schema = _response_schema(registry, request)
    schema_tools = tuple(
        branch["properties"]["tool_name"]["enum"][0]
        for branch in schema["properties"]["steps"]["items"]["anyOf"]
    )

    assert set(prompt_payload["tools"]) == set(registry.names())
    assert schema_tools == registry.names()


def test_prompt_catalog_is_sanitized_and_input_values_are_not_disclosed(
    registry,
) -> None:
    secret_path = "/private/not-for-provider/input.h5ad"
    planner, model = _planner(_plan_response(_inspect_step()))

    planner.plan(_request({"input_path": secret_path, "species": "mouse"}), registry)

    payload = json.loads(model.calls[0][0])
    assert secret_path not in model.calls[0][0]
    assert payload["request"]["available_input_names"] == [
        "input_path",
        "species",
    ]
    assert set(payload["tools"]) == set(registry.names())
    species = payload["tools"]["epizoo_embed_cells"][2]["species"]
    assert species[2]["c"] == [
        "human",
        "mouse",
    ]
    serialized_catalog = json.dumps(payload["tools"])
    assert "function" not in serialized_catalog
    assert "exception_classifier" not in serialized_catalog
    assert "module" not in serialized_catalog
    assert "EpiZooConfig" not in serialized_catalog


@pytest.mark.parametrize(
    "response",
    [
        "not json",
        "```json\n{}\n```",
        _plan_response(_inspect_step()) + " trailing prose",
        "[]",
        (
            '{"schema_version": NaN, "status": "plan", "steps": [],'
            ' "reason": null}'
        ),
    ],
)
def test_non_strict_or_malformed_json_is_rejected(registry, response) -> None:
    planner, _ = _planner(response)

    with pytest.raises(PlannerError) as raised:
        planner.plan(_request(), registry)

    assert raised.value.code == "PLANNER_OUTPUT_INVALID"


def test_duplicate_json_key_is_rejected(registry) -> None:
    response = (
        '{"schema_version":3,"status":"plan","status":"unsupported",'
        '"steps":[],"reason":null}'
    )
    planner, _ = _planner(response)

    with pytest.raises(PlannerError) as raised:
        planner.plan(_request(), registry)

    assert raised.value.code == "PLANNER_OUTPUT_INVALID"


def test_unknown_top_level_field_is_rejected(registry) -> None:
    planner, _ = _planner(_plan_response(_inspect_step(), extra="forbidden"))

    with pytest.raises(PlannerError, match="unknown fields") as raised:
        planner.plan(_request(), registry)

    assert raised.value.code == "PLANNER_OUTPUT_INVALID"


def test_unknown_step_field_is_rejected(registry) -> None:
    planner, _ = _planner(
        _plan_response(_inspect_step(python="print('unsafe')"))
    )

    with pytest.raises(PlannerError, match="unknown fields") as raised:
        planner.plan(_request(), registry)

    assert raised.value.code == "PLANNER_OUTPUT_INVALID"


@pytest.mark.parametrize("description", ["", "x" * 2049])
def test_invalid_step_description_is_rejected(registry, description) -> None:
    planner, _ = _planner(
        _plan_response(_inspect_step(description=description))
    )

    with pytest.raises(PlannerError) as raised:
        planner.plan(_request(), registry)

    assert raised.value.code == "PLANNER_OUTPUT_INVALID"


def test_nullable_step_description_is_accepted(registry) -> None:
    planner, _ = _planner(_plan_response(_inspect_step(description=None)))

    plan = planner.plan(_request(), registry)

    assert plan.steps[0].description is None


@pytest.mark.parametrize("version", [0, 1, 2, 4, "3", True])
def test_unsupported_schema_version_is_rejected(registry, version) -> None:
    payload = json.loads(_plan_response(_inspect_step()))
    payload["schema_version"] = version
    planner, _ = _planner(json.dumps(payload))

    with pytest.raises(PlannerError) as raised:
        planner.plan(_request(), registry)

    assert raised.value.code == "PLANNER_OUTPUT_INVALID"


@pytest.mark.parametrize(
    "binding",
    [
        {
            **_input_binding("input_path"),
            "extra": True,
        },
        {
            **_input_binding("input_path"),
            "ref_step_id": "inspect",
        },
        {
            **_input_binding("input_path"),
            "binding_type": "literal",
        },
        {},
    ],
)
def test_malformed_input_binding_is_rejected(registry, binding) -> None:
    planner, _ = _planner(
        _plan_response(_inspect_step(arguments={"path": binding}))
    )

    with pytest.raises(PlannerError) as raised:
        planner.plan(_request(), registry)

    assert raised.value.code == "PLANNER_BINDING_INVALID"


def test_missing_requested_input_binding_is_user_input_error(registry) -> None:
    planner, _ = _planner(_plan_response(_inspect_step()))

    with pytest.raises(PlannerError) as raised:
        planner.plan(_request({"different_input": "value"}), registry)

    assert raised.value.code == "MISSING_REQUIRED_INPUT"
    assert raised.value.category is ErrorCategory.USER_INPUT_ERROR


def test_cross_tool_argument_key_is_rejected(registry) -> None:
    arguments = dict(_inspect_step()["arguments"])
    arguments["output_dir"] = _input_binding("input_path")
    planner, _ = _planner(
        _plan_response(_inspect_step(arguments=arguments))
    )

    with pytest.raises(PlannerError) as raised:
        planner.plan(_request(), registry)

    assert raised.value.code == "INVALID_TOOL_ARGUMENTS"
    assert raised.value.diagnostics[-1].reason_code == "unknown_tool_argument"


def test_missing_required_tool_argument_is_rejected(registry) -> None:
    planner, _ = _planner(
        _plan_response(_inspect_step(arguments={}))
    )

    with pytest.raises(PlannerError) as raised:
        planner.plan(_request(), registry)

    assert raised.value.code == "INVALID_TOOL_ARGUMENTS"
    assert raised.value.diagnostics[-1].reason_code == (
        "missing_tool_argument"
    )


def test_missing_nullable_optional_argument_is_rejected(registry) -> None:
    arguments = dict(_embed_step()["arguments"])
    del arguments["overwrite"]
    planner, _ = _planner(
        _plan_response(_inspect_step(), _embed_step(arguments=arguments))
    )

    with pytest.raises(PlannerError) as raised:
        planner.plan(
            _request(
                {
                    "input_path": "/data/input.h5ad",
                    "output_dir": "/output",
                    "species": "mouse",
                }
            ),
            registry,
        )

    assert raised.value.code == "PLANNER_BINDING_INVALID"
    assert raised.value.diagnostics[-1].reason_code == (
        "missing_nullable_optional_argument"
    )


def test_v2_response_is_not_reinterpreted_as_v3(registry) -> None:
    payload = json.loads(_plan_response(_inspect_step()))
    payload["schema_version"] = 2
    planner, _ = _planner(json.dumps(payload))

    with pytest.raises(PlannerError) as raised:
        planner.plan(_request(), registry)

    assert raised.value.code == "PLANNER_OUTPUT_INVALID"


@pytest.mark.parametrize(
    "reference",
    [
        {**_ref_binding("inspect", "input_path"), "input_name": "x"},
        {**_ref_binding("inspect", "input_path"), "ref_step_id": None},
        {
            **_ref_binding("inspect", "input_path"),
            "ref_output_key": 1,
        },
        {**_ref_binding("inspect", "input_path"), "extra": True},
    ],
)
def test_malformed_reference_is_rejected(registry, reference) -> None:
    arguments = dict(_embed_step()["arguments"])
    arguments["input_path"] = reference
    planner, _ = _planner(
        _plan_response(_inspect_step(), _embed_step(arguments=arguments))
    )

    with pytest.raises(PlannerError) as raised:
        planner.plan(
            _request(
                {
                    "input_path": "/data/input.h5ad",
                    "output_dir": "/output",
                    "species": "mouse",
                }
            ),
            registry,
        )

    assert raised.value.code == "PLANNER_BINDING_INVALID"


def test_explicit_unsupported_response(registry) -> None:
    response = json.dumps(
        {
            "schema_version": 3,
            "status": "unsupported",
            "steps": [],
            "reason": "Clustering is outside the available tools.",
        }
    )
    planner, _ = _planner(response)

    with pytest.raises(PlannerError) as raised:
        planner.plan(_request(), registry)

    assert raised.value.code == "UNSUPPORTED_REQUEST"
    assert raised.value.category is ErrorCategory.USER_INPUT_ERROR


@pytest.mark.parametrize(
    "payload",
    [
        {
            "schema_version": 3,
            "status": "plan",
            "steps": [_inspect_step()],
            "reason": "must be null",
        },
        {
            "schema_version": 3,
            "status": "plan",
            "steps": [],
            "reason": None,
        },
        {
            "schema_version": 3,
            "status": "unsupported",
            "steps": [_inspect_step()],
            "reason": "not executable",
        },
        {
            "schema_version": 3,
            "status": "unsupported",
            "steps": [],
            "reason": None,
        },
        {
            "schema_version": 3,
            "status": "unsupported",
            "steps": [],
            "reason": "",
        },
    ],
)
def test_status_semantics_are_enforced_locally(registry, payload) -> None:
    planner, _ = _planner(json.dumps(payload))

    with pytest.raises(PlannerError) as raised:
        planner.plan(_request(), registry)

    assert raised.value.code == "PLANNER_OUTPUT_INVALID"


def test_duplicate_argument_binding_names_are_impossible_and_rejected(registry) -> None:
    response = (
        '{"schema_version":3,"status":"plan","steps":[{"step_id":"inspect",'
        '"tool_name":"inspect_scATAC","arguments":{'
        '"path":{"binding_type":"input","input_name":"input_path"},'
        '"path":{"binding_type":"input","input_name":"input_path"}},'
        '"depends_on":[],"description":null}],"reason":null}'
    )
    planner, _ = _planner(response)

    with pytest.raises(PlannerError) as raised:
        planner.plan(_request(), registry)

    assert raised.value.code == "PLANNER_OUTPUT_INVALID"


def test_incomplete_argument_binding_fields_are_rejected(registry) -> None:
    binding = _input_binding("input_path")
    del binding["input_name"]
    planner, _ = _planner(
        _plan_response(_inspect_step(arguments={"path": binding}))
    )

    with pytest.raises(PlannerError) as raised:
        planner.plan(_request(), registry)

    assert raised.value.code == "PLANNER_BINDING_INVALID"


def test_input_binding_rejects_non_null_reference_fields(registry) -> None:
    binding = _input_binding("input_path")
    binding["ref_step_id"] = "inspect"
    planner, _ = _planner(
        _plan_response(_inspect_step(arguments={"path": binding}))
    )

    with pytest.raises(PlannerError) as raised:
        planner.plan(_request(), registry)

    assert raised.value.code == "PLANNER_BINDING_INVALID"


def test_provider_exception_is_sanitized_and_classified(registry) -> None:
    planner, model = _planner(
        "",
        error=RuntimeError("Authorization: Bearer secret-token; raw response body"),
    )

    with pytest.raises(PlannerError) as raised:
        planner.plan(_request(), registry)

    assert len(model.calls) == 1
    assert raised.value.code == "PLANNING_PROVIDER_ERROR"
    assert raised.value.category is ErrorCategory.ENVIRONMENT_ERROR
    assert "secret-token" not in str(raised.value)
    assert "raw response" not in str(raised.value)


def test_oversized_response_is_rejected(registry) -> None:
    response = json.dumps(
        {
            "schema_version": 3,
            "status": "unsupported",
            "steps": [],
            "reason": "x" * 70_000,
        }
    )
    planner, _ = _planner(response)

    with pytest.raises(PlannerError, match="byte limit") as raised:
        planner.plan(_request(), registry)

    assert raised.value.code == "PLANNER_OUTPUT_INVALID"


def test_excessive_step_count_is_rejected(registry) -> None:
    steps = tuple(
        _inspect_step(step_id=f"inspect-{index}") for index in range(17)
    )
    planner, _ = _planner(_plan_response(*steps))

    with pytest.raises(PlannerError, match="step limit") as raised:
        planner.plan(_request(), registry)

    assert raised.value.code == "PLANNER_OUTPUT_INVALID"


def test_planner_never_invokes_registered_callables(registry) -> None:
    inspect_call = Mock(side_effect=AssertionError("must not execute"))
    embed_call = Mock(side_effect=AssertionError("must not execute"))
    guarded = ToolRegistry(
        (
            replace(registry.get("inspect_scATAC"), function=inspect_call),
            replace(registry.get("epizoo_embed_cells"), function=embed_call),
        )
    )
    planner, _ = _planner(_plan_response(_inspect_step()))

    planner.plan(_request(), guarded)

    inspect_call.assert_not_called()
    embed_call.assert_not_called()


def test_unknown_tool_response_is_rejected_before_candidate_construction(
    registry,
) -> None:
    inspect_call = Mock(side_effect=AssertionError("must not execute"))
    guarded = ToolRegistry(
        (replace(registry.get("inspect_scATAC"), function=inspect_call),)
    )
    planner, _ = _planner(
        _plan_response(
            _inspect_step(),
            {
                "step_id": "unsafe",
                "tool_name": "arbitrary_python",
                "arguments": {},
                "depends_on": [],
                "description": None,
            },
        )
    )

    with pytest.raises(PlannerError) as raised:
        planner.plan(_request(), guarded)

    assert raised.value.code == "UNKNOWN_TOOL"
    inspect_call.assert_not_called()


def test_invalid_reference_field_fails_existing_preflight(registry) -> None:
    arguments = dict(_embed_step()["arguments"])
    arguments["input_path"] = _ref_binding("inspect", "not_a_result_field")
    planner, _ = _planner(
        _plan_response(_inspect_step(), _embed_step(arguments=arguments))
    )
    request = _request(
        {
            "input_path": "/data/input.h5ad",
            "output_dir": "/output",
            "species": "mouse",
        }
    )

    result = AgentRuntime(planner=planner, registry=registry).run(request)

    assert result.status is RunStatus.FAILED
    assert result.errors[0].code == "INVALID_OUTPUT_REFERENCE"


def test_run_mode_does_not_change_llm_plan(registry) -> None:
    response = _plan_response(_inspect_step(), _embed_step())
    planner, _ = _planner(response)
    inputs = {
        "input_path": "/data/input.h5ad",
        "output_dir": "/output",
        "species": "mouse",
    }

    execute = planner.plan(_request(inputs, mode=RunMode.EXECUTE), registry)
    plan_only = planner.plan(_request(inputs, mode=RunMode.PLAN_ONLY), registry)

    assert execute == plan_only


def test_existing_deterministic_planner_behavior_is_unchanged(registry) -> None:
    plan = DeterministicPlanner().plan(
        AgentRequest(
            "request-1",
            "Inspect this scATAC dataset",
            {"input_path": "/data/input.h5ad"},
        ),
        registry,
    )

    assert plan.plan_id == "request-1:inspection"
    assert plan.planner_name == "deterministic"
    assert plan.steps[0].arguments["path"] == "/data/input.h5ad"


def test_default_runtime_remains_deterministic_offline() -> None:
    assert isinstance(AgentRuntime().planner, DeterministicPlanner)
