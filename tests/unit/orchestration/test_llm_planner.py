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


def _input_binding(name: str, input_name: str) -> dict[str, object]:
    return {
        "name": name,
        "binding_type": "input",
        "input_name": input_name,
        "ref_step_id": None,
        "ref_output_key": None,
    }


def _ref_binding(
    name: str, step_id: str, output_key: str
) -> dict[str, object]:
    return {
        "name": name,
        "binding_type": "ref",
        "input_name": None,
        "ref_step_id": step_id,
        "ref_output_key": output_key,
    }


def _inspect_step(**changes) -> dict[str, object]:
    step: dict[str, object] = {
        "step_id": "inspect",
        "tool_name": "inspect_scATAC",
        "arguments": [_input_binding("path", "input_path")],
        "depends_on": [],
        "description": "Inspect the supplied dataset.",
    }
    step.update(changes)
    return step


def _embed_step(**changes) -> dict[str, object]:
    step: dict[str, object] = {
        "step_id": "embed",
        "tool_name": "epizoo_embed_cells",
        "arguments": [
            _ref_binding("input_path", "inspect", "input_path"),
            _input_binding("output_dir", "output_dir"),
            _input_binding("species", "species"),
        ],
        "depends_on": ["inspect"],
        "description": "Persist EpiZoo cell embeddings.",
    }
    step.update(changes)
    return step


def _neighbors_step(**changes) -> dict[str, object]:
    step: dict[str, object] = {
        "step_id": "neighbors",
        "tool_name": "build_cell_neighbors",
        "arguments": [
            _ref_binding("embedding_path", "embed", "embedding_path"),
            _ref_binding("cell_ids_path", "embed", "cell_ids_path"),
            _input_binding("output_dir", "output_dir"),
        ],
        "depends_on": ["embed"],
        "description": "Build a sparse neighbor graph.",
    }
    step.update(changes)
    return step


def _cluster_step(**changes) -> dict[str, object]:
    step: dict[str, object] = {
        "step_id": "cluster",
        "tool_name": "cluster_cells",
        "arguments": [
            _ref_binding("analysis_path", "neighbors", "analysis_path"),
            _input_binding("output_dir", "output_dir"),
        ],
        "depends_on": ["neighbors"],
        "description": "Cluster cells with Leiden.",
    }
    step.update(changes)
    return step


def _umap_step(**changes) -> dict[str, object]:
    step: dict[str, object] = {
        "step_id": "umap",
        "tool_name": "compute_cell_umap",
        "arguments": [
            _ref_binding("analysis_path", "cluster", "analysis_path"),
            _input_binding("output_dir", "output_dir"),
        ],
        "depends_on": ["cluster"],
        "description": "Compute a two-dimensional UMAP.",
    }
    step.update(changes)
    return step


def _evaluation_step(**changes) -> dict[str, object]:
    step: dict[str, object] = {
        "step_id": "evaluate",
        "tool_name": "evaluate_cell_clustering",
        "arguments": [
            _ref_binding("analysis_path", "cluster", "analysis_path"),
            _ref_binding("reference_h5ad_path", "inspect", "input_path"),
            _input_binding("label_key", "label_key"),
            _input_binding("output_dir", "output_dir"),
        ],
        "depends_on": ["cluster", "inspect"],
        "description": "Evaluate fixed clustering labels.",
    }
    step.update(changes)
    return step


def _transfer_step(**changes) -> dict[str, object]:
    step: dict[str, object] = {
        "step_id": "transfer",
        "tool_name": "transfer_cell_labels",
        "arguments": [
            _ref_binding("reference_embedding_path", "embed_reference", "embedding_path"),
            _ref_binding("reference_cell_ids_path", "embed_reference", "cell_ids_path"),
            _ref_binding("reference_h5ad_path", "inspect_reference", "input_path"),
            _input_binding("reference_label_key", "reference_label_key"),
            _ref_binding("reference_species", "embed_reference", "species"),
            _ref_binding("reference_checkpoint_path", "embed_reference", "checkpoint_path"),
            _ref_binding("query_embedding_path", "embed_query", "embedding_path"),
            _ref_binding("query_cell_ids_path", "embed_query", "cell_ids_path"),
            _ref_binding("query_h5ad_path", "inspect_query", "input_path"),
            _ref_binding("query_species", "embed_query", "species"),
            _ref_binding("query_checkpoint_path", "embed_query", "checkpoint_path"),
            _input_binding("output_dir", "output_dir"),
        ],
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


def _plan_response(*steps: dict[str, object], **extra) -> str:
    payload: dict[str, object] = {
        "schema_version": 2,
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
        arguments=[_input_binding("path", "reference_input_path")],
    )
    embed_reference = _embed_step(
        step_id="embed_reference",
        arguments=[
            _ref_binding("input_path", "inspect_reference", "input_path"),
            _input_binding("output_dir", "output_dir"),
            _input_binding("species", "species"),
            _input_binding("checkpoint_path", "checkpoint_path"),
        ],
        depends_on=["inspect_reference"],
    )
    inspect_query = _inspect_step(
        step_id="inspect_query",
        arguments=[_input_binding("path", "query_input_path")],
    )
    embed_query = _embed_step(
        step_id="embed_query",
        arguments=[
            _ref_binding("input_path", "inspect_query", "input_path"),
            _input_binding("output_dir", "output_dir"),
            _input_binding("species", "species"),
            _input_binding("checkpoint_path", "checkpoint_path"),
        ],
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
    arguments = list(_embed_step()["arguments"])
    arguments.extend(
        [
            _input_binding("checkpoint_path", "checkpoint_path"),
            _input_binding("device", "device"),
            _input_binding("overwrite", "overwrite"),
        ]
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
            _inspect_step(arguments=[literal]),
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


def test_response_schema_is_strict_fixed_v2_and_registry_derived(registry) -> None:
    planner, model = _planner(_plan_response(_inspect_step()))

    planner.plan(_request(), registry)

    schema = model.calls[0][1]
    assert schema["type"] == "object"
    assert schema["required"] == (
        "schema_version",
        "status",
        "steps",
        "reason",
    )
    assert schema["properties"]["schema_version"]["enum"] == (2,)
    assert schema["properties"]["reason"]["type"] == ("string", "null")
    step_schema = schema["properties"]["steps"]["items"]
    assert step_schema["properties"]["tool_name"]["enum"] == registry.names()
    assert step_schema["properties"]["description"]["type"] == (
        "string",
        "null",
    )
    binding_schema = step_schema["properties"]["arguments"]["items"]
    assert binding_schema["properties"]["binding_type"]["enum"] == (
        "input",
        "ref",
    )
    assert set(binding_schema["properties"]["name"]["enum"]) == {
        "path",
        "input_path",
        "output_dir",
        "species",
        "checkpoint_path",
        "device",
        "overwrite",
        "embedding_path",
        "cell_ids_path",
        "analysis_path",
        "n_neighbors",
        "metric",
        "random_seed",
        "resolution",
        "min_dist",
        "spread",
        "reference_h5ad_path",
        "label_key",
        "cluster_key",
        "reference_embedding_path",
        "reference_cell_ids_path",
        "reference_label_key",
        "reference_species",
        "reference_checkpoint_path",
        "query_embedding_path",
        "query_cell_ids_path",
        "query_h5ad_path",
        "query_species",
        "query_checkpoint_path",
            "min_confidence",
            "annotation_path",
            "ground_truth_h5ad_path",
            "ground_truth_label_key",
        }
    for field_name in ("input_name", "ref_step_id", "ref_output_key"):
        assert binding_schema["properties"][field_name]["type"] == (
            "string",
            "null",
        )

    banned_keywords = {
        "oneOf",
        "anyOf",
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


def test_prompt_catalog_is_sanitized_and_input_values_are_not_disclosed(
    registry,
) -> None:
    secret_path = "/private/not-for-provider/input.h5ad"
    planner, model = _planner(_plan_response(_inspect_step()))

    planner.plan(_request({"input_path": secret_path, "species": "mouse"}), registry)

    payload = json.loads(model.calls[0][0])
    assert secret_path not in model.calls[0][0]
    assert payload["request"]["available_inputs"] == [
        {"json_type": "string", "name": "input_path"},
        {"json_type": "string", "name": "species"},
    ]
    assert [tool["name"] for tool in payload["tools"]] == list(registry.names())
    assert payload["tools"][1]["required_arguments"]["species"]["choices"] == [
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
        '{"schema_version":2,"status":"plan","status":"unsupported",'
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


@pytest.mark.parametrize("version", [0, 1, 3, "2", True])
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
            **_input_binding("path", "input_path"),
            "extra": True,
        },
        {
            **_input_binding("path", "input_path"),
            "name": 3,
        },
        {
            **_input_binding("path", "input_path"),
            "binding_type": "literal",
        },
        {},
    ],
)
def test_malformed_input_binding_is_rejected(registry, binding) -> None:
    planner, _ = _planner(
        _plan_response(_inspect_step(arguments=[binding]))
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


@pytest.mark.parametrize(
    "reference",
    [
        {**_ref_binding("input_path", "inspect", "input_path"), "input_name": "x"},
        {**_ref_binding("input_path", "inspect", "input_path"), "ref_step_id": None},
        {
            **_ref_binding("input_path", "inspect", "input_path"),
            "ref_output_key": 1,
        },
        {**_ref_binding("input_path", "inspect", "input_path"), "extra": True},
    ],
)
def test_malformed_reference_is_rejected(registry, reference) -> None:
    arguments = list(_embed_step()["arguments"])
    arguments[0] = reference
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
            "schema_version": 2,
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
            "schema_version": 2,
            "status": "plan",
            "steps": [_inspect_step()],
            "reason": "must be null",
        },
        {
            "schema_version": 2,
            "status": "plan",
            "steps": [],
            "reason": None,
        },
        {
            "schema_version": 2,
            "status": "unsupported",
            "steps": [_inspect_step()],
            "reason": "not executable",
        },
        {
            "schema_version": 2,
            "status": "unsupported",
            "steps": [],
            "reason": None,
        },
        {
            "schema_version": 2,
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


def test_duplicate_argument_binding_names_are_rejected(registry) -> None:
    planner, _ = _planner(
        _plan_response(
            _inspect_step(
                arguments=[
                    _input_binding("path", "input_path"),
                    _input_binding("path", "input_path"),
                ]
            )
        )
    )

    with pytest.raises(PlannerError, match="duplicate argument") as raised:
        planner.plan(_request(), registry)

    assert raised.value.code == "PLANNER_BINDING_INVALID"


def test_incomplete_argument_binding_fields_are_rejected(registry) -> None:
    binding = _input_binding("path", "input_path")
    del binding["ref_output_key"]
    planner, _ = _planner(
        _plan_response(_inspect_step(arguments=[binding]))
    )

    with pytest.raises(PlannerError) as raised:
        planner.plan(_request(), registry)

    assert raised.value.code == "PLANNER_BINDING_INVALID"


def test_input_binding_rejects_non_null_reference_fields(registry) -> None:
    binding = _input_binding("path", "input_path")
    binding["ref_step_id"] = "inspect"
    planner, _ = _planner(
        _plan_response(_inspect_step(arguments=[binding]))
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
            "schema_version": 2,
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


def test_registry_validation_remains_runtime_preflight_responsibility(registry) -> None:
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
                "arguments": [],
                "depends_on": [],
                "description": None,
            },
        )
    )

    direct_plan = planner.plan(_request(), guarded)
    result = AgentRuntime(planner=planner, registry=guarded).run(_request())

    assert direct_plan.steps[1].tool_name == "arbitrary_python"
    assert result.status is RunStatus.FAILED
    assert result.errors[0].code == "UNKNOWN_TOOL"
    inspect_call.assert_not_called()


def test_invalid_reference_field_fails_existing_preflight(registry) -> None:
    arguments = list(_embed_step()["arguments"])
    arguments[0] = _ref_binding(
        "input_path", "inspect", "not_a_result_field"
    )
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
