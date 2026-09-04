"""Registry-attached semantic compiler metadata contracts."""

from __future__ import annotations

from dataclasses import replace

import pytest

from agent.orchestration import (
    AgentRequest,
    ArgumentSpec,
    ErrorCategory,
    ErrorClassification,
    ResultContract,
    SemanticConsumerPortSpec,
    SemanticLineage,
    SemanticPlanCandidate,
    SemanticPlanCompileError,
    SemanticPlanStep,
    SemanticPortMember,
    SemanticProducerPortSpec,
    SemanticRequestInputSource,
    SemanticRequestMember,
    SemanticRequestSourceSpec,
    SemanticToolSpec,
    ToolRegistry,
    ToolSpec,
    build_default_tool_registry,
    build_m92_semantic_compiler_contract,
    build_semantic_compiler_contract,
    compile_semantic_plan,
)


def _classify(_exception: Exception) -> ErrorClassification:
    return ErrorClassification(ErrorCategory.TOOL_EXECUTION_ERROR, "test")


def _tool_spec(
    name: str,
    *,
    required: dict[str, ArgumentSpec] | None = None,
    optional: dict[str, ArgumentSpec] | None = None,
    results: dict[str, tuple[type, ...]] | None = None,
    semantic: SemanticToolSpec | None = None,
) -> ToolSpec:
    return ToolSpec(
        name=name,
        function=lambda **_arguments: {},
        required_arguments=required or {},
        optional_arguments=optional or {},
        result_contract=ResultContract(f"{name}-result", results or {}),
        exception_classifier=_classify,
        semantic_planning=semantic,
    )


def _member(name: str, field_name: str | None = None) -> SemanticPortMember:
    return SemanticPortMember(name, field_name or name)


def _request_member(
    name: str, input_name: str | None = None
) -> SemanticRequestMember:
    return SemanticRequestMember(name, input_name or name)


def _source(
    selector: str,
    *members: SemanticRequestMember,
    lineage: SemanticLineage | None = None,
) -> SemanticRequestSourceSpec:
    return SemanticRequestSourceSpec(selector, tuple(members), lineage)


def _valid_registry() -> ToolRegistry:
    producer = _tool_spec(
        "producer",
        results={"artifact_path": (str,)},
        semantic=SemanticToolSpec(
            producer_ports=(
                SemanticProducerPortSpec(
                    "artifact",
                    "artifact.v1",
                    (_member("path", "artifact_path"),),
                ),
            )
        ),
    )
    consumer = _tool_spec(
        "consumer",
        required={"input_path": ArgumentSpec((str,))},
        semantic=SemanticToolSpec(
            consumer_ports=(
                SemanticConsumerPortSpec(
                    "artifact",
                    (_member("path", "input_path"),),
                    True,
                    accepted_upstream_types=("artifact.v1",),
                ),
            )
        ),
    )
    return ToolRegistry((producer, consumer))


def test_valid_semantic_metadata_is_accepted_and_generates_authority() -> None:
    contract = build_semantic_compiler_contract(_valid_registry())

    assert contract.request_bindings == ()
    assert len(contract.step_output_channels) == 1
    channel = contract.step_output_channels[0]
    assert channel.producer_tool_name == "producer"
    assert channel.consumer_tool_name == "consumer"
    assert channel.members[0].output_key == "artifact_path"
    assert channel.members[0].argument_name == "input_path"


def test_unknown_semantic_argument_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown argument"):
        _tool_spec(
            "invalid",
            required={"known": ArgumentSpec((str,))},
            semantic=SemanticToolSpec(
                consumer_ports=(
                    SemanticConsumerPortSpec(
                        "value",
                        (_member("value", "missing"),),
                        True,
                        request_sources=(
                            _source("known", _request_member("value", "known")),
                        ),
                    ),
                )
            ),
        )


def test_unknown_semantic_result_field_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown result field"):
        _tool_spec(
            "invalid",
            results={"known": (str,)},
            semantic=SemanticToolSpec(
                producer_ports=(
                    SemanticProducerPortSpec(
                        "value", "value.v1", (_member("value", "missing"),)
                    ),
                )
            ),
        )


def test_duplicate_semantic_port_fails_closed() -> None:
    port = SemanticProducerPortSpec(
        "value", "value.v1", (_member("value", "result"),)
    )
    with pytest.raises(ValueError, match="repeats a semantic producer port"):
        _tool_spec(
            "invalid",
            results={"result": (str,)},
            semantic=SemanticToolSpec(producer_ports=(port, port)),
        )


def test_duplicate_logical_member_fails_closed() -> None:
    with pytest.raises(ValueError, match="logical member"):
        _tool_spec(
            "invalid",
            required={
                "first": ArgumentSpec((str,)),
                "second": ArgumentSpec((str,)),
            },
            semantic=SemanticToolSpec(
                consumer_ports=(
                    SemanticConsumerPortSpec(
                        "value",
                        (
                            _member("same", "first"),
                            _member("same", "second"),
                        ),
                        True,
                        request_sources=(
                            _source(
                                "first",
                                _request_member("same", "first"),
                            ),
                        ),
                    ),
                )
            ),
        )


def test_overlapping_grouped_members_fail_closed() -> None:
    with pytest.raises(ValueError, match="overlapping execution fields"):
        _tool_spec(
            "invalid",
            required={"shared": ArgumentSpec((str,))},
            semantic=SemanticToolSpec(
                consumer_ports=(
                    SemanticConsumerPortSpec(
                        "value",
                        (
                            _member("first", "shared"),
                            _member("second", "shared"),
                        ),
                        True,
                        request_sources=(
                            _source(
                                "first",
                                _request_member("first"),
                                _request_member("second"),
                            ),
                        ),
                    ),
                )
            ),
        )


def test_overlapping_consumer_ports_fail_closed() -> None:
    source = _source("shared", _request_member("value", "shared"))
    with pytest.raises(ValueError, match="multiple semantic consumer ports"):
        _tool_spec(
            "invalid",
            required={"shared": ArgumentSpec((str,))},
            semantic=SemanticToolSpec(
                consumer_ports=(
                    SemanticConsumerPortSpec(
                        "first", (_member("value", "shared"),), True,
                        request_sources=(source,),
                    ),
                    SemanticConsumerPortSpec(
                        "second", (_member("value", "shared"),), True,
                        request_sources=(source,),
                    ),
                )
            ),
        )


def test_incompatible_semantic_result_and_argument_types_fail_closed() -> None:
    producer = _tool_spec(
        "producer",
        results={"value": (int,)},
        semantic=SemanticToolSpec(
            producer_ports=(
                SemanticProducerPortSpec(
                    "value", "value.v1", (_member("value"),)
                ),
            )
        ),
    )
    consumer = _tool_spec(
        "consumer",
        required={"value": ArgumentSpec((str,))},
        semantic=SemanticToolSpec(
            consumer_ports=(
                SemanticConsumerPortSpec(
                    "value",
                    (_member("value"),),
                    True,
                    accepted_upstream_types=("value.v1",),
                ),
            )
        ),
    )

    with pytest.raises(ValueError, match="incompatible"):
        ToolRegistry((producer, consumer))


def test_upstream_semantic_type_without_required_lineage_fails_closed() -> None:
    producer = _tool_spec(
        "producer",
        results={"value": (str,)},
        semantic=SemanticToolSpec(
            producer_ports=(
                SemanticProducerPortSpec(
                    "value", "value.v1", (_member("value"),)
                ),
            )
        ),
    )
    consumer = _tool_spec(
        "consumer",
        required={"value": ArgumentSpec((str,))},
        semantic=SemanticToolSpec(
            consumer_ports=(
                SemanticConsumerPortSpec(
                    "reference",
                    (_member("value"),),
                    True,
                    accepted_upstream_types=("value.v1",),
                    required_lineage=SemanticLineage.REFERENCE,
                ),
            )
        ),
    )

    with pytest.raises(ValueError, match="does not provide lineage"):
        ToolRegistry((producer, consumer))


def test_invalid_lineage_declaration_fails_closed() -> None:
    with pytest.raises(ValueError, match="invalid lineage"):
        _tool_spec(
            "invalid",
            required={"value": ArgumentSpec((str,))},
            semantic=SemanticToolSpec(
                consumer_ports=(
                    SemanticConsumerPortSpec(
                        "reference",
                        (_member("value"),),
                        True,
                        request_sources=(
                            _source(
                                "value",
                                _request_member("value"),
                                lineage=SemanticLineage.QUERY,
                            ),
                        ),
                        required_lineage=SemanticLineage.REFERENCE,
                    ),
                )
            ),
        )


def test_invalid_producer_lineage_port_fails_closed() -> None:
    with pytest.raises(ValueError, match="invalid lineage source port"):
        _tool_spec(
            "invalid",
            results={"value": (str,)},
            semantic=SemanticToolSpec(
                producer_ports=(
                    SemanticProducerPortSpec(
                        "value",
                        "value.v1",
                        (_member("value"),),
                        lineage_from_port="missing",
                    ),
                )
            ),
        )


def test_incomplete_request_source_group_fails_closed() -> None:
    with pytest.raises(ValueError, match="does not completely populate"):
        _tool_spec(
            "invalid",
            required={
                "data": ArgumentSpec((str,)),
                "ids": ArgumentSpec((str,)),
            },
            semantic=SemanticToolSpec(
                consumer_ports=(
                    SemanticConsumerPortSpec(
                        "bundle",
                        (_member("data"), _member("ids")),
                        True,
                        request_sources=(
                            _source("data", _request_member("data")),
                        ),
                    ),
                )
            ),
        )


def test_required_status_mismatch_fails_closed() -> None:
    with pytest.raises(ValueError, match="inconsistent required/optional"):
        _tool_spec(
            "invalid",
            required={"value": ArgumentSpec((str,))},
            semantic=SemanticToolSpec(
                consumer_ports=(
                    SemanticConsumerPortSpec(
                        "value",
                        (_member("value"),),
                        False,
                        request_sources=(
                            _source("value", _request_member("value")),
                        ),
                    ),
                )
            ),
        )


def test_missing_upstream_semantic_member_fails_closed() -> None:
    producer = _tool_spec(
        "producer",
        results={"first": (str,)},
        semantic=SemanticToolSpec(
            producer_ports=(
                SemanticProducerPortSpec(
                    "bundle", "bundle.v1", (_member("first"),)
                ),
            )
        ),
    )
    consumer = _tool_spec(
        "consumer",
        required={
            "first": ArgumentSpec((str,)),
            "second": ArgumentSpec((str,)),
        },
        semantic=SemanticToolSpec(
            consumer_ports=(
                SemanticConsumerPortSpec(
                    "bundle",
                    (_member("first"), _member("second")),
                    True,
                    accepted_upstream_types=("bundle.v1",),
                ),
            )
        ),
    )

    with pytest.raises(ValueError, match="does not provide members"):
        ToolRegistry((producer, consumer))


def test_duplicate_request_source_selector_fails_closed() -> None:
    source = _source("value", _request_member("value"))
    with pytest.raises(ValueError, match="repeats a request-source selector"):
        _tool_spec(
            "invalid",
            required={"value": ArgumentSpec((str,))},
            semantic=SemanticToolSpec(
                consumer_ports=(
                    SemanticConsumerPortSpec(
                        "value",
                        (_member("value"),),
                        True,
                        request_sources=(source, source),
                    ),
                )
            ),
        )


def test_unauthorized_upstream_binding_fails_closed() -> None:
    with pytest.raises(ValueError, match="non-reference argument"):
        _tool_spec(
            "invalid",
            required={
                "value": ArgumentSpec((str,), allow_step_output_ref=False)
            },
            semantic=SemanticToolSpec(
                consumer_ports=(
                    SemanticConsumerPortSpec(
                        "value",
                        (_member("value"),),
                        True,
                        accepted_upstream_types=("value.v1",),
                    ),
                )
            ),
        )


def test_every_planner_visible_tool_has_authoritative_semantic_metadata() -> None:
    registry = build_default_tool_registry()
    planner_visible = {
        name
        for name in registry.names()
        if registry.get(name).planning is not None
    }
    missing = sorted(
        name
        for name in planner_visible
        if registry.get(name).semantic_planning is None
    )

    assert planner_visible
    assert missing == []
    build_semantic_compiler_contract(registry)


def test_grouped_direct_embedding_source_expands_request_inputs() -> None:
    registry = build_default_tool_registry()
    request = AgentRequest(
        "direct-neighbors",
        "Build neighbors from these aligned artifacts.",
        {
            "embedding_path": "/inputs/cells.npy",
            "cell_ids_path": "/inputs/cells.ids.txt",
            "output_dir": "/outputs",
        },
    )
    candidate = SemanticPlanCandidate(
        (
            SemanticPlanStep(
                "neighbors",
                "build_cell_neighbors",
                sources=(
                    SemanticRequestInputSource("embedding", "embedding_path"),
                ),
            ),
        )
    )

    plan = compile_semantic_plan(
        request,
        candidate,
        registry,
        build_semantic_compiler_contract(registry),
    )

    assert dict(plan.steps[0].arguments) == {
        "embedding_path": "/inputs/cells.npy",
        "cell_ids_path": "/inputs/cells.ids.txt",
        "output_dir": "/outputs",
    }


def test_grouped_direct_transfer_sources_preserve_explicit_lineage() -> None:
    registry = build_default_tool_registry()
    inputs = {
        "reference_embedding_path": "/inputs/reference.npy",
        "reference_cell_ids_path": "/inputs/reference.ids.txt",
        "reference_input_path": "/inputs/reference.h5ad",
        "reference_label_key": "celltype",
        "reference_species": "mouse",
        "reference_checkpoint_path": "/models/epizoo.ckpt",
        "query_embedding_path": "/inputs/query.npy",
        "query_cell_ids_path": "/inputs/query.ids.txt",
        "query_input_path": "/inputs/query.h5ad",
        "query_species": "mouse",
        "query_checkpoint_path": "/models/epizoo.ckpt",
        "output_dir": "/outputs",
    }
    request = AgentRequest("direct-transfer", "Transfer labels.", inputs)
    candidate = SemanticPlanCandidate(
        (
            SemanticPlanStep(
                "transfer",
                "transfer_cell_labels",
                sources=(
                    SemanticRequestInputSource(
                        "reference_dataset", "reference_input_path"
                    ),
                    SemanticRequestInputSource(
                        "reference_embedding", "reference_embedding_path"
                    ),
                    SemanticRequestInputSource(
                        "query_dataset", "query_input_path"
                    ),
                    SemanticRequestInputSource(
                        "query_embedding", "query_embedding_path"
                    ),
                ),
            ),
        )
    )

    plan = compile_semantic_plan(
        request,
        candidate,
        registry,
        build_semantic_compiler_contract(registry),
    )

    expected = dict(inputs)
    expected["reference_h5ad_path"] = expected.pop("reference_input_path")
    expected["query_h5ad_path"] = expected.pop("query_input_path")
    assert dict(plan.steps[0].arguments) == expected
    assert plan.steps[0].depends_on == ()


def test_grouped_direct_source_missing_companion_fails_closed() -> None:
    registry = build_default_tool_registry()
    request = AgentRequest(
        "missing-companion",
        "Build neighbors.",
        {"embedding_path": "/inputs/cells.npy", "output_dir": "/outputs"},
    )
    candidate = SemanticPlanCandidate(
        (
            SemanticPlanStep(
                "neighbors",
                "build_cell_neighbors",
                sources=(
                    SemanticRequestInputSource("embedding", "embedding_path"),
                ),
            ),
        )
    )

    with pytest.raises(SemanticPlanCompileError) as caught:
        compile_semantic_plan(
            request,
            candidate,
            registry,
            build_semantic_compiler_contract(registry),
        )

    assert caught.value.code == "MISSING_REQUEST_SOURCE_MEMBER"


def test_multiple_complete_grouped_request_sources_are_ambiguous() -> None:
    semantic = SemanticToolSpec(
        consumer_ports=(
            SemanticConsumerPortSpec(
                "bundle",
                (_member("data"), _member("ids")),
                True,
                request_sources=(
                    _source(
                        "data",
                        _request_member("data"),
                        _request_member("ids"),
                    ),
                    _source(
                        "alternate_data",
                        _request_member("data", "alternate_data"),
                        _request_member("ids", "alternate_ids"),
                    ),
                ),
            ),
        )
    )
    tool = _tool_spec(
        "consumer",
        required={
            "data": ArgumentSpec((str,)),
            "ids": ArgumentSpec((str,)),
        },
        semantic=semantic,
    )
    registry = ToolRegistry((tool,))
    request = AgentRequest(
        "ambiguous-groups",
        "Use a bundle.",
        {
            "data": "data",
            "ids": "ids",
            "alternate_data": "alternate-data",
            "alternate_ids": "alternate-ids",
        },
    )
    candidate = SemanticPlanCandidate((SemanticPlanStep("step", "consumer"),))

    with pytest.raises(SemanticPlanCompileError) as caught:
        compile_semantic_plan(
            request,
            candidate,
            registry,
            build_semantic_compiler_contract(registry),
        )

    assert caught.value.code == "AMBIGUOUS_REQUEST_INPUT"


def test_grouped_request_source_with_wrong_lineage_fails_closed() -> None:
    registry = build_default_tool_registry()
    contract = build_m92_semantic_compiler_contract(registry)
    altered = tuple(
        replace(rule, lineage=SemanticLineage.QUERY)
        if rule.tool_name == "transfer_cell_labels"
        and rule.target_port == "reference_embedding"
        else rule
        for rule in contract.request_bindings
    )
    request = AgentRequest(
        "wrong-lineage",
        "Transfer labels.",
        {"reference_embedding_path": "/reference.npy"},
    )
    candidate = SemanticPlanCandidate(
        (
            SemanticPlanStep(
                "transfer",
                "transfer_cell_labels",
                sources=(
                    SemanticRequestInputSource(
                        "reference_embedding", "reference_embedding_path"
                    ),
                ),
            ),
        )
    )

    with pytest.raises(ValueError, match="invalid required lineage"):
        compile_semantic_plan(
            request,
            candidate,
            registry,
            replace(contract, request_bindings=altered),
        )
