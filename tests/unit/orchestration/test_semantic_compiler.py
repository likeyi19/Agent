"""Contracts for the opt-in Post-M9 semantic source compiler."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from agent.orchestration import (
    AgentRequest,
    ArgumentSpec,
    ChannelMember,
    ErrorCategory,
    ErrorClassification,
    LLMPlanner,
    PlanExecutor,
    PlanningCompilerContract,
    RequestInputBindingRule,
    ResultContract,
    SemanticLineage,
    SemanticPlanCandidate,
    SemanticPlanCompileError,
    SemanticPlanStep,
    SemanticRequestInputSource,
    SemanticStepOutputSource,
    StepOutputChannelRule,
    StepOutputRef,
    ToolRegistry,
    ToolSpec,
    build_default_tool_registry,
    build_m92_semantic_compiler_contract,
    compile_semantic_plan,
)
from benchmarks.planner.benchmark import (
    ScriptedPlanningModel,
    load_cases,
    oracle_response,
)


ROOT = Path(__file__).resolve().parents[3]
CASES_PATH = ROOT / "benchmarks" / "planner" / "cases.json"
PAIRED_CASE = "differential_accessibility_paired_covariates"
TRANSFER_CASE = "label_transfer_canonical"
OPTIONAL_TRANSFER_CASE = "label_transfer_optional"
DOWNSTREAM_CASE = "downstream_canonical"
CLUSTERING_EVALUATION_CASE = "clustering_evaluation_canonical"
ANNOTATION_EVALUATION_CASE = "annotation_evaluation_standalone"
TRANSFER_EVALUATION_CASE = "transfer_and_annotation_evaluation"
_MISSING = object()


@pytest.fixture
def registry() -> ToolRegistry:
    return build_default_tool_registry()


def _case(case_id: str):
    return next(case for case in load_cases(CASES_PATH) if case.case_id == case_id)


def _request(case_id: str, **changes: object) -> AgentRequest:
    case = _case(case_id)
    inputs = dict(case.inputs)
    for name, value in changes.items():
        if value is _MISSING:
            inputs.pop(name, None)
        else:
            inputs[name] = value
    return AgentRequest(f"{case_id}-request", case.prompt, inputs)


def _execution_fields(plan) -> tuple[object, ...]:
    return tuple(
        (
            step.step_id,
            step.tool_name,
            dict(step.arguments),
            step.depends_on,
        )
        for step in plan.steps
    )


def _v3_plan(case_id: str, request: AgentRequest, registry: ToolRegistry):
    case = _case(case_id)
    return LLMPlanner(ScriptedPlanningModel(oracle_response(case))).plan(
        request, registry
    )


def _input(target: str, input_name: str) -> SemanticRequestInputSource:
    return SemanticRequestInputSource(target, input_name)


def _step(
    target: str, step_id: str, source_port: str | None = None
) -> SemanticStepOutputSource:
    return SemanticStepOutputSource(target, step_id, source_port)


def _paired_candidate() -> SemanticPlanCandidate:
    return SemanticPlanCandidate(
        (
            SemanticPlanStep(
                "provider-step-101",
                "validate_scATAC_feature_space",
                sources=(_input("overwrite", "overwrite"),),
            ),
            SemanticPlanStep(
                "provider-step-102",
                "build_replicate_pseudobulk",
                sources=(
                    _step("feature_space", "provider-step-101"),
                    _input("overwrite", "overwrite"),
                ),
            ),
            SemanticPlanStep(
                "provider-step-103",
                "run_replicate_differential_accessibility",
                sources=(
                    _step("pseudobulk", "provider-step-102"),
                    _input("overwrite", "overwrite"),
                ),
            ),
        )
    )


def _transfer_candidate(*, optional: bool = False) -> SemanticPlanCandidate:
    reference_embedding_sources = [_step("dataset", "provider-step-101")]
    query_embedding_sources = [_step("dataset", "provider-step-103")]
    transfer_sources = [
        _step("reference_dataset", "provider-step-101"),
        _step("reference_embedding", "provider-step-102"),
        _step("query_dataset", "provider-step-103"),
        _step("query_embedding", "provider-step-104"),
    ]
    if optional:
        reference_embedding_sources.append(_input("checkpoint", "checkpoint_path"))
        query_embedding_sources.append(_input("checkpoint", "checkpoint_path"))
        transfer_sources.append(_input("overwrite", "overwrite"))
    return SemanticPlanCandidate(
        (
            SemanticPlanStep(
                "provider-step-101",
                "inspect_scATAC",
                sources=(_input("dataset", "reference_input_path"),),
            ),
            SemanticPlanStep(
                "provider-step-102",
                "epizoo_embed_cells",
                sources=tuple(reference_embedding_sources),
            ),
            SemanticPlanStep(
                "provider-step-103",
                "inspect_scATAC",
                sources=(_input("dataset", "query_input_path"),),
            ),
            SemanticPlanStep(
                "provider-step-104",
                "epizoo_embed_cells",
                sources=tuple(query_embedding_sources),
            ),
            SemanticPlanStep(
                "provider-step-105",
                "transfer_cell_labels",
                sources=tuple(transfer_sources),
            ),
        )
    )


def _downstream_candidate() -> SemanticPlanCandidate:
    return SemanticPlanCandidate(
        (
            SemanticPlanStep("provider-step-101", "inspect_scATAC"),
            SemanticPlanStep(
                "provider-step-102",
                "epizoo_embed_cells",
                sources=(_step("dataset", "provider-step-101"),),
            ),
            SemanticPlanStep(
                "provider-step-103",
                "build_cell_neighbors",
                sources=(_step("embedding", "provider-step-102"),),
            ),
            SemanticPlanStep(
                "provider-step-104",
                "cluster_cells",
                sources=(_step("analysis", "provider-step-103"),),
            ),
            SemanticPlanStep(
                "provider-step-105",
                "compute_cell_umap",
                sources=(_step("analysis", "provider-step-104"),),
            ),
        )
    )


def _clustering_evaluation_candidate() -> SemanticPlanCandidate:
    downstream = _downstream_candidate().steps[:4]
    return SemanticPlanCandidate(
        (
            *downstream,
            SemanticPlanStep(
                "provider-step-105",
                "evaluate_cell_clustering",
                sources=(
                    _step("analysis", "provider-step-104"),
                    _step("ground_truth_dataset", "provider-step-101"),
                ),
            ),
        )
    )


def _transfer_evaluation_candidate() -> SemanticPlanCandidate:
    transfer = _transfer_candidate()
    return SemanticPlanCandidate(
        (
            *transfer.steps,
            SemanticPlanStep(
                "provider-step-106",
                "evaluate_cell_annotation",
                sources=(
                    _step("annotation", "provider-step-105"),
                    _input("ground_truth_dataset", "ground_truth_h5ad_path"),
                ),
            ),
        )
    )


def _compile(
    request: AgentRequest,
    candidate: SemanticPlanCandidate,
    registry: ToolRegistry,
    contract: PlanningCompilerContract | None = None,
):
    return compile_semantic_plan(
        request,
        candidate,
        registry,
        contract or build_m92_semantic_compiler_contract(registry),
    )


def test_paired_da_migrates_to_sources_with_exact_v3_execution_semantics(
    registry: ToolRegistry,
) -> None:
    request = _request(PAIRED_CASE)
    compiled = _compile(request, _paired_candidate(), registry)

    assert _execution_fields(compiled) == _execution_fields(
        _v3_plan(PAIRED_CASE, request, registry)
    )
    assert PlanExecutor(registry).preflight(compiled).passed
    assert all(step.description is None for step in compiled.steps)


@pytest.mark.parametrize(
    ("case_id", "optional"),
    ((TRANSFER_CASE, False), (OPTIONAL_TRANSFER_CASE, True)),
)
def test_dual_branch_transfer_compiles_to_exact_v3_execution_semantics(
    registry: ToolRegistry, case_id: str, optional: bool
) -> None:
    request = _request(case_id)
    compiled = _compile(
        request, _transfer_candidate(optional=optional), registry
    )

    assert _execution_fields(compiled) == _execution_fields(
        _v3_plan(case_id, request, registry)
    )
    assert PlanExecutor(registry).preflight(compiled).passed


def test_transfer_embedding_ports_expand_to_complete_coordinated_bundles(
    registry: ToolRegistry,
) -> None:
    transfer = _compile(
        _request(TRANSFER_CASE), _transfer_candidate(), registry
    ).steps[-1]

    for branch, producer in (
        ("reference", "provider-step-102"),
        ("query", "provider-step-104"),
    ):
        assert transfer.arguments[f"{branch}_embedding_path"] == StepOutputRef(
            producer, "embedding_path"
        )
        assert transfer.arguments[f"{branch}_cell_ids_path"] == StepOutputRef(
            producer, "cell_ids_path"
        )
        assert transfer.arguments[f"{branch}_species"] == StepOutputRef(
            producer, "species"
        )
        assert transfer.arguments[f"{branch}_checkpoint_path"] == StepOutputRef(
            producer, "checkpoint_path"
        )


def test_optional_scope_is_explicit_for_checkpoint_and_overwrite(
    registry: ToolRegistry,
) -> None:
    plan = _compile(
        _request(OPTIONAL_TRANSFER_CASE),
        _transfer_candidate(optional=True),
        registry,
    )
    reference_embed, query_embed, transfer = plan.steps[1], plan.steps[3], plan.steps[4]

    assert reference_embed.arguments["checkpoint_path"] == "/synthetic/shared.ckpt"
    assert query_embed.arguments["checkpoint_path"] == "/synthetic/shared.ckpt"
    assert "overwrite" not in reference_embed.arguments
    assert "overwrite" not in query_embed.arguments
    assert transfer.arguments["overwrite"] is False
    assert transfer.arguments["n_neighbors"] == 25
    assert transfer.arguments["metric"] == "cosine"
    assert transfer.arguments["min_confidence"] == 0.7


def test_neighbors_uses_one_two_member_embedding_channel(
    registry: ToolRegistry,
) -> None:
    request = _request(DOWNSTREAM_CASE)
    candidate = SemanticPlanCandidate(
        (
            SemanticPlanStep("provider-step-101", "inspect_scATAC"),
            SemanticPlanStep(
                "provider-step-102",
                "epizoo_embed_cells",
                sources=(_step("dataset", "provider-step-101"),),
            ),
            SemanticPlanStep(
                "provider-step-103",
                "build_cell_neighbors",
                sources=(_step("embedding", "provider-step-102"),),
            ),
        )
    )
    plan = _compile(request, candidate, registry)
    neighbors = plan.steps[-1]

    assert _execution_fields(plan) == _execution_fields(
        _v3_plan(DOWNSTREAM_CASE, request, registry)
    )[:3]
    assert neighbors.arguments["embedding_path"] == StepOutputRef(
        "provider-step-102", "embedding_path"
    )
    assert neighbors.arguments["cell_ids_path"] == StepOutputRef(
        "provider-step-102", "cell_ids_path"
    )
    assert neighbors.depends_on == ("provider-step-102",)


def test_complete_downstream_workflow_matches_v3_execution_semantics(
    registry: ToolRegistry,
) -> None:
    request = _request(DOWNSTREAM_CASE)
    compiled = _compile(request, _downstream_candidate(), registry)

    assert _execution_fields(compiled) == _execution_fields(
        _v3_plan(DOWNSTREAM_CASE, request, registry)
    )
    assert PlanExecutor(registry).preflight(compiled).passed


def test_clustering_evaluation_matches_v3_execution_semantics(
    registry: ToolRegistry,
) -> None:
    request = _request(CLUSTERING_EVALUATION_CASE)
    compiled = _compile(
        request, _clustering_evaluation_candidate(), registry
    )

    assert _execution_fields(compiled) == _execution_fields(
        _v3_plan(CLUSTERING_EVALUATION_CASE, request, registry)
    )
    assert PlanExecutor(registry).preflight(compiled).passed


def test_standalone_annotation_evaluation_matches_v3_execution_semantics(
    registry: ToolRegistry,
) -> None:
    request = _request(ANNOTATION_EVALUATION_CASE)
    candidate = SemanticPlanCandidate(
        (
            SemanticPlanStep(
                "provider-step-101", "evaluate_cell_annotation"
            ),
        )
    )
    compiled = _compile(request, candidate, registry)

    assert _execution_fields(compiled) == _execution_fields(
        _v3_plan(ANNOTATION_EVALUATION_CASE, request, registry)
    )
    assert PlanExecutor(registry).preflight(compiled).passed


def test_transfer_and_annotation_evaluation_matches_v3_execution_semantics(
    registry: ToolRegistry,
) -> None:
    request = _request(TRANSFER_EVALUATION_CASE)
    compiled = _compile(
        request, _transfer_evaluation_candidate(), registry
    )

    assert _execution_fields(compiled) == _execution_fields(
        _v3_plan(TRANSFER_EVALUATION_CASE, request, registry)
    )
    assert PlanExecutor(registry).preflight(compiled).passed


def test_clustering_evaluation_rejects_embedding_as_analysis(
    registry: ToolRegistry,
) -> None:
    candidate = SemanticPlanCandidate(
        (
            SemanticPlanStep("inspect", "inspect_scATAC"),
            SemanticPlanStep(
                "embed",
                "epizoo_embed_cells",
                sources=(_step("dataset", "inspect"),),
            ),
            SemanticPlanStep(
                "evaluate",
                "evaluate_cell_clustering",
                sources=(
                    _step("analysis", "embed"),
                    _step("ground_truth_dataset", "inspect"),
                ),
            ),
        )
    )

    with pytest.raises(SemanticPlanCompileError) as caught:
        _compile(_request(CLUSTERING_EVALUATION_CASE), candidate, registry)

    assert caught.value.code == "ZERO_VALID_CHANNELS"


def test_multiple_analysis_producers_are_not_chosen_implicitly(
    registry: ToolRegistry,
) -> None:
    candidate = SemanticPlanCandidate(
        (
            *_downstream_candidate().steps,
            SemanticPlanStep(
                "provider-step-106",
                "evaluate_cell_clustering",
                sources=(
                    _step("ground_truth_dataset", "provider-step-101"),
                ),
            ),
        )
    )

    with pytest.raises(SemanticPlanCompileError) as caught:
        _compile(_request(CLUSTERING_EVALUATION_CASE), candidate, registry)

    assert caught.value.code == "MISSING_REQUIRED_SOURCE"


def test_clustering_evaluation_requires_ground_truth_source(
    registry: ToolRegistry,
) -> None:
    request = AgentRequest(
        "missing-ground-truth",
        "Evaluate fixed clusters.",
        {
            "analysis_path": "/synthetic/clustered.h5ad",
            "label_key": "celltype",
            "output_dir": "/synthetic/output",
        },
    )
    candidate = SemanticPlanCandidate(
        (
            SemanticPlanStep(
                "evaluate",
                "evaluate_cell_clustering",
                sources=(_input("analysis", "analysis_path"),),
            ),
        )
    )

    with pytest.raises(SemanticPlanCompileError) as caught:
        _compile(request, candidate, registry)

    assert caught.value.code == "MISSING_REQUIRED_SOURCE"


def test_annotation_evaluation_rejects_query_lineage_as_ground_truth(
    registry: ToolRegistry,
) -> None:
    request = AgentRequest(
        "wrong-ground-truth-lineage",
        "Evaluate this annotation.",
        {
            "annotation_path": "/synthetic/annotation.h5ad",
            "query_input_path": "/synthetic/query.h5ad",
            "ground_truth_label_key": "celltype",
            "output_dir": "/synthetic/output",
        },
    )
    candidate = SemanticPlanCandidate(
        (
            SemanticPlanStep(
                "inspect_query",
                "inspect_scATAC",
                sources=(_input("dataset", "query_input_path"),),
            ),
            SemanticPlanStep(
                "evaluate",
                "evaluate_cell_annotation",
                sources=(
                    _input("annotation", "annotation_path"),
                    _step("ground_truth_dataset", "inspect_query"),
                ),
            ),
        )
    )

    with pytest.raises(SemanticPlanCompileError) as caught:
        _compile(request, candidate, registry)

    assert caught.value.code == "BROKEN_BRANCH_LINEAGE"


def test_clustering_evaluation_requires_explicit_cluster_key_source(
    registry: ToolRegistry,
) -> None:
    request = _request(CLUSTERING_EVALUATION_CASE, cluster_key="clusters")

    with pytest.raises(SemanticPlanCompileError) as caught:
        _compile(request, _clustering_evaluation_candidate(), registry)

    assert caught.value.code == "AMBIGUOUS_SOURCE_SELECTION"


def test_annotation_evaluation_rejects_clustered_analysis_as_annotation(
    registry: ToolRegistry,
) -> None:
    candidate = SemanticPlanCandidate(
        (
            *_downstream_candidate().steps[:4],
            SemanticPlanStep(
                "provider-step-105",
                "evaluate_cell_annotation",
                sources=(
                    _step("annotation", "provider-step-104"),
                    _input(
                        "ground_truth_dataset", "ground_truth_h5ad_path"
                    ),
                ),
            ),
        )
    )
    request = _request(
        DOWNSTREAM_CASE,
        ground_truth_h5ad_path="/synthetic/ground-truth.h5ad",
        ground_truth_label_key="celltype",
    )

    with pytest.raises(SemanticPlanCompileError) as caught:
        _compile(request, candidate, registry)

    assert caught.value.code == "ZERO_VALID_CHANNELS"


def test_annotation_evaluation_rejects_unauthorized_request_source(
    registry: ToolRegistry,
) -> None:
    request = _request(
        ANNOTATION_EVALUATION_CASE,
        input_path="/synthetic/not-an-annotation.h5ad",
    )
    candidate = SemanticPlanCandidate(
        (
            SemanticPlanStep(
                "evaluate",
                "evaluate_cell_annotation",
                sources=(_input("annotation", "input_path"),),
            ),
        )
    )

    with pytest.raises(SemanticPlanCompileError) as caught:
        _compile(request, candidate, registry)

    assert caught.value.code == "UNAUTHORIZED_REQUEST_INPUT"


def test_control_only_dependencies_remain_separate(registry: ToolRegistry) -> None:
    candidate = _paired_candidate()
    da = replace(
        candidate.steps[-1],
        control_dependencies=("provider-step-101",),
    )
    plan = _compile(
        _request(PAIRED_CASE),
        SemanticPlanCandidate((*candidate.steps[:-1], da)),
        registry,
    )

    assert plan.steps[-1].depends_on == (
        "provider-step-102",
        "provider-step-101",
    )
    assert all(
        not (
            isinstance(value, StepOutputRef)
            and value.step_id == "provider-step-101"
        )
        for value in plan.steps[-1].arguments.values()
    )


def test_compilation_is_order_independent_and_does_not_use_step_names(
    registry: ToolRegistry,
) -> None:
    candidate = _transfer_candidate()
    opaque = SemanticPlanCandidate(
        (
            candidate.steps[2],
            candidate.steps[3],
            candidate.steps[0],
            candidate.steps[1],
            candidate.steps[4],
        )
    )

    plan = _compile(_request(TRANSFER_CASE), opaque, registry)

    assert PlanExecutor(registry).preflight(plan).passed
    assert plan.steps[-1].arguments["reference_embedding_path"] == StepOutputRef(
        "provider-step-102", "embedding_path"
    )


def test_direct_request_versus_upstream_routing_is_explicit(
    registry: ToolRegistry,
) -> None:
    candidate = _transfer_candidate()
    reference_embed = replace(
        candidate.steps[1],
        sources=(_input("dataset", "reference_input_path"),),
    )
    query_embed = replace(
        candidate.steps[3],
        sources=(_input("dataset", "query_input_path"),),
    )
    transfer = replace(
        candidate.steps[4],
        sources=(
            _input("reference_dataset", "reference_input_path"),
            _step("reference_embedding", "provider-step-102"),
            _input("query_dataset", "query_input_path"),
            _step("query_embedding", "provider-step-104"),
        ),
    )

    plan = _compile(
        _request(TRANSFER_CASE),
        SemanticPlanCandidate(
            (
                candidate.steps[0],
                reference_embed,
                candidate.steps[2],
                query_embed,
                transfer,
            )
        ),
        registry,
    )

    assert plan.steps[1].arguments["input_path"] == "/synthetic/reference.h5ad"
    assert plan.steps[3].arguments["input_path"] == "/synthetic/query.h5ad"
    assert plan.steps[4].arguments["reference_h5ad_path"] == (
        "/synthetic/reference.h5ad"
    )
    assert plan.steps[4].arguments["query_h5ad_path"] == "/synthetic/query.h5ad"
    assert plan.steps[4].depends_on == (
        "provider-step-102",
        "provider-step-104",
    )
    assert PlanExecutor(registry).preflight(plan).passed


def test_direct_request_versus_selected_producer_cannot_be_inferred(
    registry: ToolRegistry,
) -> None:
    candidate = SemanticPlanCandidate(
        (
            SemanticPlanStep("inspect", "inspect_scATAC"),
            SemanticPlanStep("embed", "epizoo_embed_cells"),
        )
    )

    with pytest.raises(SemanticPlanCompileError) as caught:
        _compile(_request(DOWNSTREAM_CASE), candidate, registry)

    assert caught.value.code == "AMBIGUOUS_SOURCE_SELECTION"


def test_every_argument_retains_request_or_step_reference_provenance(
    registry: ToolRegistry,
) -> None:
    request = _request(OPTIONAL_TRANSFER_CASE)
    before = request.to_dict()
    registry_names = registry.names()
    registry_specs = tuple(registry.get(name) for name in registry_names)
    plan = _compile(request, _transfer_candidate(optional=True), registry)

    for step in plan.steps:
        for value in step.arguments.values():
            assert isinstance(value, StepOutputRef) or any(
                value == request_value for request_value in request.inputs.values()
            )
    assert request.to_dict() == before
    assert registry.names() == registry_names
    assert all(
        registry.get(name) is spec
        for name, spec in zip(registry_names, registry_specs, strict=True)
    )


@pytest.mark.parametrize("raw_values", [["a", "b"], ("a", "b")])
def test_scalar_values_false_sequences_and_explicit_none_are_preserved(
    raw_values: object,
) -> None:
    registry = _scalar_registry()
    request = AgentRequest(
        "typed",
        "Use explicitly supplied values.",
        {
            "count": 7,
            "threshold": 0.25,
            "values": raw_values,
            "enabled": False,
            "nullable": None,
        },
    )
    contract = PlanningCompilerContract(
        request_bindings=tuple(
            RequestInputBindingRule("scalar", name, name, name)
            for name in request.inputs
        )
    )
    plan = _compile(
        request,
        SemanticPlanCandidate((SemanticPlanStep("scalar", "scalar"),)),
        registry,
        contract,
    )

    assert dict(plan.steps[0].arguments) == dict(request.inputs)
    assert plan.steps[0].arguments["enabled"] is False
    assert plan.steps[0].arguments["nullable"] is None
    assert plan.steps[0].arguments["values"] == ("a", "b")


def test_omitted_optional_and_explicit_none_remain_distinct() -> None:
    registry = _scalar_registry()
    contract = PlanningCompilerContract(
        request_bindings=(
            RequestInputBindingRule("scalar", "nullable", "nullable", "nullable"),
        )
    )
    candidate = SemanticPlanCandidate((SemanticPlanStep("scalar", "scalar"),))

    omitted = _compile(
        AgentRequest("omitted", "Omit it.", {}), candidate, registry, contract
    )
    explicit = _compile(
        AgentRequest("explicit", "Use null.", {"nullable": None}),
        candidate,
        registry,
        contract,
    )

    assert "nullable" not in omitted.steps[0].arguments
    assert explicit.steps[0].arguments["nullable"] is None


@pytest.mark.parametrize(
    "missing_name",
    (
        "species",
        "genome_assembly",
        "numerator_condition",
        "denominator_condition",
        "design_type",
    ),
)
def test_required_scientific_choices_are_not_guessed(
    registry: ToolRegistry, missing_name: str
) -> None:
    with pytest.raises(SemanticPlanCompileError) as caught:
        _compile(
            _request(PAIRED_CASE, **{missing_name: _MISSING}),
            _paired_candidate(),
            registry,
        )

    assert caught.value.code == "MISSING_REQUIRED_SOURCE"


def test_workflow_steps_are_not_inferred(registry: ToolRegistry) -> None:
    candidate = SemanticPlanCandidate(
        (
            SemanticPlanStep(
                "da", "run_replicate_differential_accessibility"
            ),
        )
    )

    with pytest.raises(SemanticPlanCompileError) as caught:
        _compile(_request(PAIRED_CASE), candidate, registry)

    assert caught.value.code == "MISSING_REQUIRED_SOURCE"


def test_reference_query_producer_swap_fails_lineage(
    registry: ToolRegistry,
) -> None:
    candidate = _transfer_candidate()
    transfer = candidate.steps[-1]
    swapped = tuple(
        _step(source.target_port, "provider-step-104")
        if isinstance(source, SemanticStepOutputSource)
        and source.target_port == "reference_embedding"
        else source
        for source in transfer.sources
    )

    with pytest.raises(SemanticPlanCompileError) as caught:
        _compile(
            _request(TRANSFER_CASE),
            SemanticPlanCandidate(
                (*candidate.steps[:-1], replace(transfer, sources=swapped))
            ),
            registry,
        )

    assert caught.value.code == "BROKEN_BRANCH_LINEAGE"


def test_direct_request_branch_swap_fails_lineage(
    registry: ToolRegistry,
) -> None:
    candidate = _transfer_candidate()
    reference_inspect = replace(
        candidate.steps[0],
        sources=(_input("dataset", "query_input_path"),),
    )

    with pytest.raises(SemanticPlanCompileError) as caught:
        _compile(
            _request(TRANSFER_CASE),
            SemanticPlanCandidate((reference_inspect, *candidate.steps[1:])),
            registry,
        )

    assert caught.value.code == "BROKEN_BRANCH_LINEAGE"


def test_missing_required_semantic_target_port_fails_closed(
    registry: ToolRegistry,
) -> None:
    candidate = _transfer_candidate()
    transfer = candidate.steps[-1]
    sources = tuple(
        source
        for source in transfer.sources
        if source.target_port != "query_embedding"
    )

    with pytest.raises(SemanticPlanCompileError) as caught:
        _compile(
            _request(TRANSFER_CASE),
            SemanticPlanCandidate(
                (*candidate.steps[:-1], replace(transfer, sources=sources))
            ),
            registry,
        )

    assert caught.value.code == "MISSING_REQUIRED_SOURCE"


def test_duplicate_target_port_source_selection_fails_closed(
    registry: ToolRegistry,
) -> None:
    candidate = _transfer_candidate()
    transfer = candidate.steps[-1]

    with pytest.raises(SemanticPlanCompileError) as caught:
        _compile(
            _request(TRANSFER_CASE),
            SemanticPlanCandidate(
                (
                    *candidate.steps[:-1],
                    replace(
                        transfer,
                        sources=(
                            *transfer.sources,
                            _step("reference_dataset", "provider-step-101"),
                        ),
                    ),
                )
            ),
            registry,
        )

    assert caught.value.code == "DUPLICATE_TARGET_SOURCE"


def test_unknown_target_port_fails_closed(registry: ToolRegistry) -> None:
    candidate = _transfer_candidate()
    transfer = replace(
        candidate.steps[-1],
        sources=(*candidate.steps[-1].sources, _step("invented", "provider-step-101")),
    )

    with pytest.raises(SemanticPlanCompileError) as caught:
        _compile(
            _request(TRANSFER_CASE),
            SemanticPlanCandidate((*candidate.steps[:-1], transfer)),
            registry,
        )

    assert caught.value.code == "UNKNOWN_TARGET_PORT"


def test_ambiguous_request_input_source_requires_explicit_selection(
    registry: ToolRegistry,
) -> None:
    candidate = _transfer_candidate()
    incomplete = replace(candidate.steps[0], sources=())

    with pytest.raises(SemanticPlanCompileError) as caught:
        _compile(
            _request(TRANSFER_CASE),
            SemanticPlanCandidate((incomplete, *candidate.steps[1:])),
            registry,
        )

    assert caught.value.code == "AMBIGUOUS_REQUEST_INPUT"


def test_repeated_tool_selection_is_rejected_when_branch_source_is_incomplete(
    registry: ToolRegistry,
) -> None:
    candidate = _transfer_candidate()
    incomplete = replace(candidate.steps[2], sources=())

    with pytest.raises(SemanticPlanCompileError) as caught:
        _compile(
            _request(TRANSFER_CASE),
            SemanticPlanCandidate(
                (*candidate.steps[:2], incomplete, *candidate.steps[3:])
            ),
            registry,
        )

    assert caught.value.code == "AMBIGUOUS_REQUEST_INPUT"


def test_optional_input_fanout_without_explicit_scope_fails_closed(
    registry: ToolRegistry,
) -> None:
    with pytest.raises(SemanticPlanCompileError) as caught:
        _compile(
            _request(OPTIONAL_TRANSFER_CASE),
            _transfer_candidate(optional=False),
            registry,
        )

    assert caught.value.code == "AMBIGUOUS_OPTIONAL_INPUT_SCOPE"


def test_unconfigured_scientific_input_is_not_silently_dropped(
    registry: ToolRegistry,
) -> None:
    with pytest.raises(SemanticPlanCompileError) as caught:
        _compile(
            _request(TRANSFER_CASE, device="cuda"),
            _transfer_candidate(),
            registry,
        )

    assert caught.value.code == "UNAUTHORIZED_REQUEST_INPUT"


def test_multiple_producer_ports_require_explicit_source_port() -> None:
    registry = _producer_consumer_registry()
    contract = PlanningCompilerContract(
        step_output_channels=(
            _channel("primary"),
            _channel("alternate", output="alternate"),
        )
    )

    with pytest.raises(SemanticPlanCompileError) as caught:
        _compile_producer_consumer(registry, contract)

    assert caught.value.code == "AMBIGUOUS_SOURCE_PORT"


def test_explicit_source_port_resolves_multiple_producer_ports() -> None:
    registry = _producer_consumer_registry()
    contract = PlanningCompilerContract(
        step_output_channels=(
            _channel("primary"),
            _channel("alternate", output="alternate"),
        )
    )

    plan = _compile_producer_consumer(
        registry, contract, source_port="alternate"
    )

    assert plan.steps[-1].arguments["artifact"] == StepOutputRef(
        "producer", "alternate"
    )


def test_wrong_producer_semantic_port_fails_closed() -> None:
    registry = _producer_consumer_registry()
    contract = PlanningCompilerContract(
        step_output_channels=(_channel("primary"),)
    )

    with pytest.raises(SemanticPlanCompileError) as caught:
        _compile_producer_consumer(registry, contract, source_port="wrong")

    assert caught.value.code == "WRONG_SOURCE_PORT"


def test_zero_valid_channel_mapping_fails_closed() -> None:
    registry = _producer_consumer_registry()
    contract = PlanningCompilerContract(
        step_output_channels=(_channel("primary"),)
    )

    with pytest.raises(SemanticPlanCompileError) as caught:
        _compile_producer_consumer(
            registry, contract, producer_step_id="producer-two"
        )

    assert caught.value.code == "ZERO_VALID_CHANNELS"


def test_multiple_valid_channel_mappings_fail_closed() -> None:
    registry = _producer_consumer_registry()
    contract = PlanningCompilerContract(
        step_output_channels=(
            _channel("shared", output="primary"),
            _channel("shared", output="alternate"),
        )
    )

    with pytest.raises(SemanticPlanCompileError) as caught:
        _compile_producer_consumer(registry, contract, source_port="shared")

    assert caught.value.code == "MULTIPLE_VALID_CHANNELS"


def test_incomplete_grouped_channel_authority_fails_before_plan(
    registry: ToolRegistry,
) -> None:
    contract = build_m92_semantic_compiler_contract()
    channels = tuple(
        replace(
            channel,
            members=tuple(
                member
                for member in channel.members
                if member.argument_name != "reference_checkpoint_path"
            ),
        )
        if channel.consumer_tool_name == "transfer_cell_labels"
        and channel.target_port == "reference_embedding"
        else channel
        for channel in contract.step_output_channels
    )

    with pytest.raises(SemanticPlanCompileError) as caught:
        _compile(
            _request(TRANSFER_CASE),
            _transfer_candidate(),
            registry,
            replace(contract, step_output_channels=channels),
        )

    assert caught.value.code == "MISSING_REQUIRED_BINDING"


def test_overlapping_grouped_channel_members_fail_before_plan() -> None:
    registry = _producer_consumer_registry()
    contract = PlanningCompilerContract(
        step_output_channels=(
            _channel("first", target="first"),
            _channel("second", output="alternate", target="second"),
        )
    )
    candidate = _producer_consumer_candidate(
        sources=(
            _step("first", "producer"),
            _step("second", "producer"),
        )
    )

    with pytest.raises(SemanticPlanCompileError) as caught:
        _compile(
            AgentRequest("overlap", "Reject overlap.", {}),
            candidate,
            registry,
            contract,
        )

    assert caught.value.code == "OVERLAPPING_SOURCE_MEMBERS"


@pytest.mark.parametrize(
    "bad_member",
    (
        ChannelMember("missing", "artifact"),
        ChannelMember("primary", "missing"),
    ),
)
def test_invalid_result_or_argument_authority_is_rejected(
    bad_member: ChannelMember,
) -> None:
    registry = _producer_consumer_registry()
    contract = PlanningCompilerContract(
        step_output_channels=(_channel("bad", members=(bad_member,)),)
    )

    with pytest.raises(ValueError, match="unknown"):
        _compile_producer_consumer(registry, contract)


def test_incompatible_producer_consumer_channel_is_rejected() -> None:
    registry = _producer_consumer_registry()
    contract = PlanningCompilerContract(
        step_output_channels=(
            _channel("count", output="count"),
        )
    )

    with pytest.raises(ValueError, match="incompatible"):
        _compile_producer_consumer(registry, contract)


def test_unknown_producer_lineage_port_is_rejected() -> None:
    registry = _producer_consumer_registry()
    contract = PlanningCompilerContract(
        step_output_channels=(
            replace(_channel("primary"), producer_lineage_port="missing"),
        )
    )

    with pytest.raises(ValueError, match="unknown producer lineage port"):
        _compile_producer_consumer(registry, contract)


def test_grouped_channel_definition_rejects_duplicate_members() -> None:
    with pytest.raises(ValueError, match="consumer argument"):
        StepOutputChannelRule(
            "producer",
            "invalid",
            "consumer",
            "artifact",
            (
                ChannelMember("primary", "artifact"),
                ChannelMember("alternate", "artifact"),
            ),
        )


def test_descriptive_registry_metadata_does_not_grant_authority(
    registry: ToolRegistry,
) -> None:
    candidate = SemanticPlanCandidate(
        (
            SemanticPlanStep("inspect", "inspect_scATAC"),
            SemanticPlanStep(
                "embed",
                "epizoo_embed_cells",
                sources=(_step("dataset", "inspect"),),
            ),
        )
    )

    with pytest.raises(SemanticPlanCompileError) as caught:
        _compile(
            AgentRequest(
                "no-authority",
                "Do not use descriptive metadata as authority.",
                {
                    "input_path": "/data/in.h5ad",
                    "output_dir": "/out",
                    "species": "mouse",
                },
            ),
            candidate,
            registry,
            PlanningCompilerContract(),
        )

    assert caught.value.code == "UNKNOWN_TARGET_PORT"


def _classification(_: Exception) -> ErrorClassification:
    return ErrorClassification(ErrorCategory.INTERNAL_AGENT_ERROR, "TEST_ERROR")


def _tool(
    name: str,
    *,
    required: dict[str, ArgumentSpec] | None = None,
    optional: dict[str, ArgumentSpec] | None = None,
    results: dict[str, tuple[type, ...]] | None = None,
) -> ToolSpec:
    return ToolSpec(
        name=name,
        function=lambda **_: {},
        required_arguments=required or {},
        optional_arguments=optional or {},
        result_contract=ResultContract(f"{name}-result", results or {}),
        exception_classifier=_classification,
    )


def _scalar_registry() -> ToolRegistry:
    return ToolRegistry(
        (
            _tool(
                "scalar",
                optional={
                    "count": ArgumentSpec((int,)),
                    "threshold": ArgumentSpec((float,)),
                    "values": ArgumentSpec((list, tuple)),
                    "enabled": ArgumentSpec((bool,)),
                    "nullable": ArgumentSpec((str, type(None))),
                },
            ),
        )
    )


def _producer_consumer_registry() -> ToolRegistry:
    producer_results = {
        "primary": (str,),
        "alternate": (str,),
        "count": (int,),
    }
    return ToolRegistry(
        (
            _tool("producer", results=producer_results),
            _tool("producer-two", results=producer_results),
            _tool(
                "consumer",
                required={"artifact": ArgumentSpec((str,))},
            ),
        )
    )


def _channel(
    source_port: str,
    *,
    output: str = "primary",
    target: str = "artifact",
    producer: str = "producer",
    members: tuple[ChannelMember, ...] | None = None,
) -> StepOutputChannelRule:
    return StepOutputChannelRule(
        producer,
        source_port,
        "consumer",
        target,
        members or (ChannelMember(output, "artifact"),),
    )


def _producer_consumer_candidate(
    *,
    producer_step_id: str = "producer",
    sources: tuple[SemanticStepOutputSource, ...] | None = None,
    source_port: str | None = None,
) -> SemanticPlanCandidate:
    return SemanticPlanCandidate(
        (
            SemanticPlanStep("producer", "producer"),
            SemanticPlanStep("producer-two", "producer-two"),
            SemanticPlanStep(
                "consumer",
                "consumer",
                sources=sources
                or (_step("artifact", producer_step_id, source_port),),
            ),
        )
    )


def _compile_producer_consumer(
    registry: ToolRegistry,
    contract: PlanningCompilerContract,
    *,
    producer_step_id: str = "producer",
    source_port: str | None = None,
):
    return _compile(
        AgentRequest("producer-consumer", "Connect selected ports.", {}),
        _producer_consumer_candidate(
            producer_step_id=producer_step_id,
            source_port=source_port,
        ),
        registry,
        contract,
    )
