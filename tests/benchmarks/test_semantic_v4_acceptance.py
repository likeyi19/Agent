from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import pytest

from agent.orchestration import (
    AgentRequest,
    LLMPlanner,
    PlanExecutor,
    PlannerError,
    PlanningWireMode,
    StepOutputRef,
    build_default_tool_registry,
    build_semantic_compiler_contract,
    build_semantic_planning_prompt,
)
from agent.orchestration.llm_planner import (
    _MAX_RESPONSE_BYTES,
    _build_prompt,
)
from benchmarks.planner.benchmark import (
    BenchmarkCase,
    ScriptedPlanningModel,
    load_cases,
    oracle_response,
)


ROOT = Path(__file__).resolve().parents[2]
CASES_PATH = ROOT / "benchmarks" / "planner" / "cases.json"


@pytest.fixture(scope="module")
def cases() -> tuple[BenchmarkCase, ...]:
    return load_cases(CASES_PATH)


def _request(case: BenchmarkCase) -> AgentRequest:
    return AgentRequest(
        f"semantic-v4-{case.case_id}", case.prompt, case.inputs
    )


def _semantic_response(case: BenchmarkCase) -> str:
    """Project the accepted v3 oracle into registry-authorized semantic choices."""

    if case.expected_outcome != "plan":
        return json.dumps(
            {
                "schema_version": 4,
                "decision": {
                    "kind": "unsupported",
                    "reason": "The accepted benchmark intent has no executable plan.",
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    registry = build_default_tool_registry()
    contract = build_semantic_compiler_contract(registry)
    v3 = json.loads(oracle_response(case))
    raw_steps = v3["steps"]
    tools_by_step = {
        step["step_id"]: step["tool_name"] for step in raw_steps
    }
    semantic_steps: list[dict[str, object]] = []

    for raw_step in raw_steps:
        step_id = raw_step["step_id"]
        tool_name = raw_step["tool_name"]
        arguments = raw_step["arguments"]
        dependencies = tuple(raw_step["depends_on"])
        covered: set[str] = set()
        sources_by_producer: dict[str, list[dict[str, str]]] = defaultdict(list)
        input_sources: list[dict[str, str]] = []

        for channel in contract.step_output_channels:
            if channel.consumer_tool_name != tool_name:
                continue
            member_bindings = [
                arguments.get(member.argument_name) for member in channel.members
            ]
            if not member_bindings or any(
                not isinstance(binding, dict)
                or binding.get("binding_type") != "ref"
                or binding.get("ref_output_key") != member.output_key
                for binding, member in zip(member_bindings, channel.members)
            ):
                continue
            producer_ids = {
                binding["ref_step_id"] for binding in member_bindings
            }
            if len(producer_ids) != 1:
                continue
            producer_id = next(iter(producer_ids))
            if tools_by_step.get(producer_id) != channel.producer_tool_name:
                continue
            matching_ports = {
                candidate.source_port
                for candidate in contract.step_output_channels
                if candidate.producer_tool_name == channel.producer_tool_name
                and candidate.consumer_tool_name == tool_name
                and candidate.target_port == channel.target_port
            }
            source = {
                "kind": "step" if len(matching_ports) == 1 else "step_port",
                "target": channel.target_port,
                "step": producer_id,
            }
            if len(matching_ports) != 1:
                source["source_port"] = channel.source_port
            sources_by_producer[producer_id].append(source)
            covered.update(member.argument_name for member in channel.members)

        grouped_request_rules: dict[
            tuple[str, str], list[object]
        ] = defaultdict(list)
        for rule in contract.request_bindings:
            if rule.tool_name == tool_name:
                grouped_request_rules[(rule.target_port, rule.selector)].append(rule)
        for (target_port, selector), rules in grouped_request_rules.items():
            if all(
                isinstance(arguments.get(rule.argument_name), dict)
                and arguments[rule.argument_name].get("binding_type") == "input"
                and arguments[rule.argument_name].get("input_name")
                == rule.input_name
                for rule in rules
            ):
                input_sources.append(
                    {"kind": "input", "target": target_port, "input": selector}
                )
                covered.update(rule.argument_name for rule in rules)

        bound_arguments = {
            name for name, binding in arguments.items() if binding is not None
        }
        assert covered == bound_arguments, (
            case.case_id,
            step_id,
            sorted(bound_arguments - covered),
            sorted(covered - bound_arguments),
        )
        value_dependencies = set(sources_by_producer)
        sources = [
            source
            for dependency in dependencies
            for source in sources_by_producer[dependency]
        ]
        sources.extend(input_sources)
        semantic_steps.append(
            {
                "step_id": step_id,
                "tool": tool_name,
                "sources": sources,
                "control_dependencies": [
                    dependency
                    for dependency in dependencies
                    if dependency not in value_dependencies
                ],
            }
        )

    return json.dumps(
        {
            "schema_version": 4,
            "decision": {"kind": "plan", "steps": semantic_steps},
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _plan(case: BenchmarkCase, *, wire_mode: PlanningWireMode):
    response = (
        _semantic_response(case)
        if wire_mode is PlanningWireMode.V4
        else oracle_response(case)
    )
    return LLMPlanner(
        ScriptedPlanningModel(response), wire_mode=wire_mode
    ).plan(_request(case), build_default_tool_registry())


def _execution_signature(plan) -> tuple[object, ...]:
    return tuple(
        (
            step.step_id,
            step.tool_name,
            dict(step.arguments),
            frozenset(step.depends_on),
        )
        for step in plan.steps
    )


def test_complete_benchmark_corpus_is_classified_without_skips(
    cases: tuple[BenchmarkCase, ...],
) -> None:
    counts = defaultdict(int)
    for case in cases:
        if case.expected_outcome == "plan":
            v3_plan = _plan(case, wire_mode=PlanningWireMode.V3)
            v4_plan = _plan(case, wire_mode=PlanningWireMode.V4)
            assert _execution_signature(v4_plan) == _execution_signature(v3_plan)
            assert PlanExecutor(build_default_tool_registry()).preflight(
                v4_plan
            ).passed is case.expected_preflight_valid
            counts["accepted"] += 1
            continue

        with pytest.raises(PlannerError) as raised:
            _plan(case, wire_mode=PlanningWireMode.V4)
        assert raised.value.code == "UNSUPPORTED_REQUEST"
        counts[case.expected_outcome] += 1

    assert dict(counts) == {
        "accepted": 26,
        "unsupported": 7,
        "failure": 5,
    }


@pytest.mark.parametrize(
    "case_id",
    (
        "inspect_canonical",
        "embedding_verbose_optional",
        "downstream_explicit_steps",
        "clustering_evaluation_canonical",
        "label_transfer_canonical",
        "label_transfer_optional",
        "annotation_evaluation_standalone",
        "transfer_and_annotation_evaluation",
        "pseudobulk_canonical",
        "pseudobulk_verified_annotation",
        "differential_accessibility_fixed",
        "differential_accessibility_raw",
        "differential_accessibility_paired_covariates",
    ),
)
def test_required_workflow_classes_have_exact_v3_execution_semantics(
    cases: tuple[BenchmarkCase, ...], case_id: str
) -> None:
    case = next(case for case in cases if case.case_id == case_id)

    assert _execution_signature(
        _plan(case, wire_mode=PlanningWireMode.V4)
    ) == _execution_signature(_plan(case, wire_mode=PlanningWireMode.V3))


def test_semantic_projection_collapses_grouped_refs_and_direct_request_sources(
    cases: tuple[BenchmarkCase, ...],
) -> None:
    transfer = next(
        case for case in cases if case.case_id == "label_transfer_optional"
    )
    fixed_da = next(
        case
        for case in cases
        if case.case_id == "differential_accessibility_fixed"
    )
    transfer_payload = json.loads(_semantic_response(transfer))
    transfer_step = transfer_payload["decision"]["steps"][-1]
    fixed_da_payload = json.loads(_semantic_response(fixed_da))

    assert sum(
        source["kind"].startswith("step")
        for source in transfer_step["sources"]
    ) == 4
    assert sum(
        isinstance(value, StepOutputRef)
        for value in _plan(
            transfer, wire_mode=PlanningWireMode.V4
        ).steps[-1].arguments.values()
    ) == 10
    assert fixed_da_payload["decision"]["steps"][0]["sources"]
    assert all(
        source["kind"] == "input"
        for source in fixed_da_payload["decision"]["steps"][0]["sources"]
    )


def test_semantic_projection_keeps_control_only_dependencies_explicit(
    cases: tuple[BenchmarkCase, ...],
) -> None:
    case = next(
        case for case in cases if case.case_id == "label_transfer_canonical"
    )
    payload = json.loads(_semantic_response(case))
    query_inspect = payload["decision"]["steps"][2]
    query_inspect["control_dependencies"] = ["provider-step-101"]
    model = ScriptedPlanningModel(json.dumps(payload))

    plan = LLMPlanner(model, wire_mode=PlanningWireMode.V4).plan(
        _request(case), build_default_tool_registry()
    )

    assert plan.steps[2].depends_on == ("provider-step-101",)
    assert PlanExecutor(build_default_tool_registry()).preflight(plan).passed


def test_semantic_steps_may_arrive_in_noncanonical_branch_order(
    cases: tuple[BenchmarkCase, ...],
) -> None:
    case = next(
        case for case in cases if case.case_id == "label_transfer_canonical"
    )
    payload = json.loads(_semantic_response(case))
    steps = payload["decision"]["steps"]
    payload["decision"]["steps"] = [steps[2], steps[3], steps[0], steps[1], steps[4]]

    plan = LLMPlanner(
        ScriptedPlanningModel(json.dumps(payload)),
        wire_mode=PlanningWireMode.V4,
    ).plan(_request(case), build_default_tool_registry())

    assert PlanExecutor(build_default_tool_registry()).preflight(plan).passed
    assert plan.steps[-1].tool_name == "transfer_cell_labels"


def test_optional_sources_apply_only_to_selected_steps(
    cases: tuple[BenchmarkCase, ...],
) -> None:
    case = next(
        case for case in cases if case.case_id == "label_transfer_optional"
    )
    payload = json.loads(_semantic_response(case))
    embedding_steps = payload["decision"]["steps"]
    reference_sources = embedding_steps[1]["sources"]
    query_sources = embedding_steps[3]["sources"]

    assert any(source.get("input") == "checkpoint_path" for source in reference_sources)
    assert any(source.get("input") == "checkpoint_path" for source in query_sources)
    assert not any(
        source.get("input") == "transfer_overwrite"
        for source in reference_sources
    )
    assert not any(
        source.get("input") == "transfer_overwrite" for source in query_sources
    )

    plan = _plan(case, wire_mode=PlanningWireMode.V4)
    embedding_steps = tuple(
        step for step in plan.steps if step.tool_name == "epizoo_embed_cells"
    )
    transfer_step = plan.steps[-1]

    assert all("overwrite" not in step.arguments for step in embedding_steps)
    assert transfer_step.tool_name == "transfer_cell_labels"
    assert transfer_step.arguments["overwrite"] is False


def test_omitted_optional_and_explicit_none_remain_distinct_through_v4(
    cases: tuple[BenchmarkCase, ...],
) -> None:
    case = next(
        case for case in cases if case.case_id == "pseudobulk_canonical"
    )
    registry = build_default_tool_registry()
    omitted = _plan(case, wire_mode=PlanningWireMode.V4)
    payload = json.loads(_semantic_response(case))
    pseudobulk = payload["decision"]["steps"][-1]
    pseudobulk["sources"].append(
        {
            "kind": "input",
            "target": "group_annotation",
            "input": "group_annotation_path",
        }
    )
    explicit_request = AgentRequest(
        "semantic-v4-explicit-none",
        case.prompt,
        {**case.inputs, "group_annotation_path": None},
    )
    explicit = LLMPlanner(
        ScriptedPlanningModel(json.dumps(payload)),
        wire_mode=PlanningWireMode.V4,
    ).plan(explicit_request, registry)

    assert "group_annotation_path" not in omitted.steps[-1].arguments
    assert explicit.steps[-1].arguments["group_annotation_path"] is None
    assert PlanExecutor(registry).preflight(omitted).passed
    assert PlanExecutor(registry).preflight(explicit).passed


def _assert_v4_failure(
    case: BenchmarkCase, payload: dict[str, object], expected_code: str
) -> None:
    with pytest.raises(PlannerError) as raised:
        LLMPlanner(
            ScriptedPlanningModel(json.dumps(payload)),
            wire_mode=PlanningWireMode.V4,
        ).plan(_request(case), build_default_tool_registry())
    assert raised.value.code == expected_code


def test_ground_truth_cannot_be_sourced_from_query_lineage(
    cases: tuple[BenchmarkCase, ...],
) -> None:
    case = next(
        case
        for case in cases
        if case.case_id == "transfer_and_annotation_evaluation"
    )
    payload = json.loads(_semantic_response(case))
    evaluation = payload["decision"]["steps"][-1]
    evaluation["sources"] = [
        (
            {
                "kind": "step",
                "target": "ground_truth_dataset",
                "step": "provider-step-103",
            }
            if source["target"] == "ground_truth_dataset"
            else source
        )
        for source in evaluation["sources"]
    ]

    request = AgentRequest(
        "ground-truth-lineage-mismatch",
        case.prompt,
        {
            name: value
            for name, value in case.inputs.items()
            if name != "ground_truth_h5ad_path"
        },
    )
    with pytest.raises(PlannerError) as raised:
        LLMPlanner(
            ScriptedPlanningModel(json.dumps(payload)),
            wire_mode=PlanningWireMode.V4,
        ).plan(request, build_default_tool_registry())
    assert raised.value.code == "BROKEN_BRANCH_LINEAGE"


def test_wrong_semantic_artifact_type_fails_before_preflight(
    cases: tuple[BenchmarkCase, ...],
) -> None:
    case = next(
        case
        for case in cases
        if case.case_id == "transfer_and_annotation_evaluation"
    )
    payload = json.loads(_semantic_response(case))
    evaluation = payload["decision"]["steps"][-1]
    evaluation["sources"] = [
        (
            {
                "kind": "step",
                "target": "annotation",
                "step": "provider-step-103",
            }
            if source["target"] == "annotation"
            else source
        )
        for source in evaluation["sources"]
    ]

    _assert_v4_failure(case, payload, "ZERO_VALID_CHANNELS")


def test_multiple_compatible_producers_are_never_first_match_inferred(
    cases: tuple[BenchmarkCase, ...],
) -> None:
    case = next(
        case for case in cases if case.case_id == "downstream_canonical"
    )
    payload = json.loads(_semantic_response(case))
    steps = payload["decision"]["steps"]
    second_embedding = json.loads(json.dumps(steps[1]))
    second_embedding["step_id"] = "provider-step-extra-embedding"
    steps.insert(2, second_embedding)
    neighbors = next(step for step in steps if step["tool"] == "build_cell_neighbors")
    neighbors["sources"] = [
        source for source in neighbors["sources"] if source["target"] != "embedding"
    ]

    _assert_v4_failure(case, payload, "MISSING_REQUIRED_SOURCE")


def test_optional_source_scope_must_be_explicit_across_repeated_tools(
    cases: tuple[BenchmarkCase, ...],
) -> None:
    case = next(
        case for case in cases if case.case_id == "label_transfer_optional"
    )
    payload = json.loads(_semantic_response(case))
    for step in payload["decision"]["steps"]:
        if step["tool"] == "epizoo_embed_cells":
            step["sources"] = [
                source
                for source in step["sources"]
                if source.get("input") != "checkpoint_path"
            ]

    _assert_v4_failure(case, payload, "AMBIGUOUS_OPTIONAL_INPUT_SCOPE")


def test_incomplete_grouped_request_bundle_fails_closed() -> None:
    request = AgentRequest(
        "incomplete-grouped-request",
        "Transfer labels from direct artifacts.",
        {"reference_embedding_path": "/synthetic/reference.npy"},
    )
    payload = {
        "schema_version": 4,
        "decision": {
            "kind": "plan",
            "steps": [
                {
                    "step_id": "transfer",
                    "tool": "transfer_cell_labels",
                    "sources": [
                        {
                            "kind": "input",
                            "target": "reference_embedding",
                            "input": "reference_embedding_path",
                        }
                    ],
                    "control_dependencies": [],
                }
            ],
        },
    }

    with pytest.raises(PlannerError) as raised:
        LLMPlanner(
            ScriptedPlanningModel(json.dumps(payload)),
            wire_mode=PlanningWireMode.V4,
        ).plan(request, build_default_tool_registry())

    assert raised.value.code == "MISSING_REQUEST_SOURCE_MEMBER"


def test_unknown_explicit_source_port_fails_closed(
    cases: tuple[BenchmarkCase, ...],
) -> None:
    case = next(
        case for case in cases if case.case_id == "downstream_canonical"
    )
    payload = json.loads(_semantic_response(case))
    umap = payload["decision"]["steps"][-1]
    umap["sources"][0] = {
        "kind": "step_port",
        "target": "analysis",
        "step": "provider-step-104",
        "source_port": "unknown_analysis_port",
    }

    _assert_v4_failure(case, payload, "WRONG_SOURCE_PORT")


@pytest.mark.parametrize(
    "case_id",
    (
        "inspect_canonical",
        "downstream_canonical",
        "label_transfer_canonical",
        "label_transfer_optional",
        "differential_accessibility_paired_covariates",
    ),
)
def test_v4_prompt_and_response_retain_size_headroom(
    cases: tuple[BenchmarkCase, ...], case_id: str
) -> None:
    case = next(case for case in cases if case.case_id == case_id)
    request = _request(case)
    registry = build_default_tool_registry()
    def encode(value: object) -> bytes:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    v3_prompt_bytes = len(_build_prompt(request, registry).encode("utf-8"))
    v4_prompt_bytes = len(
        build_semantic_planning_prompt(request, registry).encode("utf-8")
    )
    v3_response_bytes = len(encode(json.loads(oracle_response(case))))
    v4_response_bytes = len(encode(json.loads(_semantic_response(case))))

    assert v4_prompt_bytes <= v3_prompt_bytes
    # Tool-correlated schema constraints intentionally increase schema size;
    # the old structural-only schema reduction is no longer a contract.
    assert v4_response_bytes < v3_response_bytes
    assert v4_response_bytes < _MAX_RESPONSE_BYTES
