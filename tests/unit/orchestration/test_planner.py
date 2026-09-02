"""Offline tests for the deliberately narrow deterministic planner."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import Mock

import pytest

from agent.orchestration import (
    DeterministicPlanner,
    ErrorCategory,
    Planner,
    PlannerError,
    RunMode,
    StepOutputRef,
    ToolRegistry,
    build_default_tool_registry,
)
from agent.schemas import AgentRequest


@pytest.fixture
def registry() -> ToolRegistry:
    return build_default_tool_registry()


def _request(
    prompt: str,
    inputs: dict[str, object],
    *,
    mode: RunMode = RunMode.EXECUTE,
) -> AgentRequest:
    return AgentRequest("request-1", prompt, inputs, mode)


def test_deterministic_planner_satisfies_protocol() -> None:
    assert isinstance(DeterministicPlanner(), Planner)


def test_pseudobulk_request_produces_exact_two_step_plan(registry) -> None:
    plan = DeterministicPlanner().plan(
        _request(
            "Build replicate-aware pseudobulk accessibility",
            {
                "input_path": "/data/raw.h5ad",
                "output_dir": "/output",
                "matrix_source": "layer",
                "layer_key": "counts",
                "matrix_semantics": "fragment_counts",
                "semantics_metadata_key": "matrix_semantics",
                "species": "human",
                "genome_assembly": "hg38",
                "coordinate_source": "none",
                "replicate_key": "donor",
                "group_key": "cell_type",
                "condition_key": "condition",
                "group_source": "raw_obs",
                "covariate_keys": ["sex", "age"],
            },
        ),
        registry,
    )

    assert plan.plan_id == "request-1:replicate-aware-pseudobulk"
    assert tuple(step.tool_name for step in plan.steps) == (
        "validate_scATAC_feature_space",
        "build_replicate_pseudobulk",
    )
    assert plan.steps[1].depends_on == ("validate_feature_space",)
    assert plan.steps[1].arguments["feature_space_path"] == StepOutputRef(
        "validate_feature_space", "feature_space_path"
    )
    assert plan.steps[0].arguments["layer_key"] == "counts"
    assert plan.steps[1].arguments["covariate_keys"] == ("sex", "age")


def test_inspection_request_produces_exact_one_step_plan(registry) -> None:
    plan = DeterministicPlanner().plan(
        _request("Inspect this scATAC-seq dataset", {"input_path": "/data/in.h5ad"}),
        registry,
    )

    assert plan.plan_id == "request-1:inspection"
    assert plan.request_id == "request-1"
    assert plan.planner_name == "deterministic"
    assert len(plan.steps) == 1
    assert plan.steps[0].step_id == "inspect"
    assert plan.steps[0].tool_name == "inspect_scATAC"
    assert dict(plan.steps[0].arguments) == {"path": "/data/in.h5ad"}
    assert plan.steps[0].depends_on == ()


def test_embedding_request_produces_exact_two_step_plan(registry) -> None:
    plan = DeterministicPlanner().plan(
        _request(
            "Compute EpiZoo embeddings",
            {
                "input_path": "/data/in.h5ad",
                "output_dir": "/output",
                "species": "mouse",
            },
        ),
        registry,
    )

    assert plan.plan_id == "request-1:epizoo-embedding"
    assert tuple(step.step_id for step in plan.steps) == ("inspect", "embed")
    assert tuple(step.tool_name for step in plan.steps) == (
        "inspect_scATAC",
        "epizoo_embed_cells",
    )
    assert plan.steps[1].depends_on == ("inspect",)
    assert plan.steps[1].arguments["input_path"] == StepOutputRef(
        "inspect", "input_path"
    )


def test_label_transfer_request_produces_exact_dual_input_plan(registry) -> None:
    plan = DeterministicPlanner().plan(
        _request(
            "Use the annotated reference to annotate the query cells",
            {
                "reference_input_path": "/data/reference.h5ad",
                "query_input_path": "/data/query.h5ad",
                "output_dir": "/output",
                "species": "mouse",
                "reference_label_key": "celltype",
                "checkpoint_path": "/models/epizoo.pth",
                "device": "cuda:0",
            },
        ),
        registry,
    )
    assert plan.plan_id == "request-1:epizoo-label-transfer"
    assert tuple(step.step_id for step in plan.steps) == (
        "inspect_reference",
        "embed_reference",
        "inspect_query",
        "embed_query",
        "transfer",
    )
    assert tuple(step.tool_name for step in plan.steps) == (
        "inspect_scATAC",
        "epizoo_embed_cells",
        "inspect_scATAC",
        "epizoo_embed_cells",
        "transfer_cell_labels",
    )
    reference_embed, query_embed, transfer = plan.steps[1], plan.steps[3], plan.steps[4]
    for embed in (reference_embed, query_embed):
        assert embed.arguments["species"] == "mouse"
        assert embed.arguments["checkpoint_path"] == "/models/epizoo.pth"
        assert embed.arguments["device"] == "cuda:0"
    assert transfer.depends_on == (
        "inspect_reference",
        "embed_reference",
        "inspect_query",
        "embed_query",
    )
    assert transfer.arguments == {
        "reference_embedding_path": StepOutputRef("embed_reference", "embedding_path"),
        "reference_cell_ids_path": StepOutputRef("embed_reference", "cell_ids_path"),
        "reference_h5ad_path": StepOutputRef("inspect_reference", "input_path"),
        "reference_label_key": "celltype",
        "reference_species": StepOutputRef("embed_reference", "species"),
        "reference_checkpoint_path": StepOutputRef("embed_reference", "checkpoint_path"),
        "query_embedding_path": StepOutputRef("embed_query", "embedding_path"),
        "query_cell_ids_path": StepOutputRef("embed_query", "cell_ids_path"),
        "query_h5ad_path": StepOutputRef("inspect_query", "input_path"),
        "query_species": StepOutputRef("embed_query", "species"),
        "query_checkpoint_path": StepOutputRef("embed_query", "checkpoint_path"),
        "output_dir": "/output",
    }


def test_label_transfer_forwards_only_explicit_optional_values(registry) -> None:
    inputs = {
        "reference_input_path": "/data/reference.h5ad",
        "query_input_path": "/data/query.h5ad",
        "output_dir": "/output",
        "species": "mouse",
        "reference_label_key": "celltype",
    }
    default_plan = DeterministicPlanner().plan(
        _request("Annotate query from reference", inputs), registry
    )
    transfer = default_plan.steps[-1]
    assert {"n_neighbors", "metric", "min_confidence", "overwrite"}.isdisjoint(
        transfer.arguments
    )
    explicit = DeterministicPlanner().plan(
        _request(
            "Annotate query from reference",
            inputs
            | {
                "n_neighbors": 7,
                "metric": "cosine",
                "min_confidence": 0.6,
                "overwrite": True,
            },
        ),
        registry,
    ).steps[-1]
    assert explicit.arguments["n_neighbors"] == 7
    assert explicit.arguments["metric"] == "cosine"
    assert explicit.arguments["min_confidence"] == 0.6
    assert explicit.arguments["overwrite"] is True


def test_standalone_annotation_evaluation_plan(registry) -> None:
    plan = DeterministicPlanner().plan(
        _request(
            "Evaluate this fixed cell annotation",
            {
                "annotation_path": "/data/query.label_transfer.h5ad",
                "ground_truth_h5ad_path": "/data/query.truth.h5ad",
                "ground_truth_label_key": "celltype",
                "output_dir": "/output",
            },
        ),
        registry,
    )
    assert plan.plan_id == "request-1:cell-annotation-evaluation"
    assert len(plan.steps) == 1
    step = plan.steps[0]
    assert step.step_id == "evaluate_annotation"
    assert step.tool_name == "evaluate_cell_annotation"
    assert step.depends_on == ()
    assert step.arguments == {
        "annotation_path": "/data/query.label_transfer.h5ad",
        "ground_truth_h5ad_path": "/data/query.truth.h5ad",
        "ground_truth_label_key": "celltype",
        "output_dir": "/output",
    }
    assert "overwrite" not in step.arguments


def test_chained_annotation_evaluation_plan_and_ground_truth_isolation(registry) -> None:
    plan = DeterministicPlanner().plan(
        _request(
            "Annotate the query and evaluate the annotation",
            {
                "reference_input_path": "/data/reference.h5ad",
                "query_input_path": "/data/query.h5ad",
                "ground_truth_h5ad_path": "/data/query.truth.h5ad",
                "output_dir": "/output",
                "species": "mouse",
                "reference_label_key": "reference_celltype",
                "ground_truth_label_key": "truth_celltype",
                "n_neighbors": 7,
                "metric": "cosine",
                "min_confidence": 0.6,
                "overwrite": True,
            },
        ),
        registry,
    )
    assert plan.plan_id == "request-1:epizoo-label-transfer-evaluation"
    assert tuple(step.step_id for step in plan.steps) == (
        "inspect_reference",
        "embed_reference",
        "inspect_query",
        "embed_query",
        "transfer",
        "evaluate_annotation",
    )
    evaluation = plan.steps[-1]
    assert evaluation.tool_name == "evaluate_cell_annotation"
    assert evaluation.depends_on == ("transfer",)
    assert evaluation.arguments == {
        "annotation_path": StepOutputRef("transfer", "annotation_path"),
        "ground_truth_h5ad_path": "/data/query.truth.h5ad",
        "ground_truth_label_key": "truth_celltype",
        "output_dir": "/output",
        "overwrite": True,
    }
    for step in plan.steps[:-1]:
        assert "ground_truth_h5ad_path" not in step.arguments
        assert "ground_truth_label_key" not in step.arguments
    transfer = plan.steps[-2]
    assert transfer.arguments["reference_label_key"] == "reference_celltype"
    assert transfer.arguments["n_neighbors"] == 7
    assert transfer.arguments["metric"] == "cosine"
    assert transfer.arguments["min_confidence"] == 0.6
    forbidden_metric_literals = {
        "macro_average",
        "zero_division",
        "confidence_diagnostics",
    }
    assert forbidden_metric_literals.isdisjoint(evaluation.arguments)


def test_ambiguous_annotation_evaluation_workflow_is_rejected(registry) -> None:
    with pytest.raises(PlannerError) as raised:
        DeterministicPlanner().plan(
            _request(
                "Evaluate this annotation",
                {
                    "annotation_path": "/data/fixed.h5ad",
                    "reference_input_path": "/data/reference.h5ad",
                    "query_input_path": "/data/query.h5ad",
                    "ground_truth_h5ad_path": "/data/truth.h5ad",
                    "ground_truth_label_key": "celltype",
                    "output_dir": "/output",
                },
            ),
            registry,
        )
    assert raised.value.code == "AMBIGUOUS_REQUEST"


def test_repeated_planning_is_deterministic(registry) -> None:
    request = _request(
        "embed this dataset",
        {
            "input_path": "/data/in.h5ad",
            "output_dir": "/output",
            "species": "human",
        },
    )
    planner = DeterministicPlanner()

    assert planner.plan(request, registry) == planner.plan(request, registry)
    assert planner.plan(request, registry).to_dict() == planner.plan(
        request, registry
    ).to_dict()


def test_embedding_intent_takes_precedence_over_inspection(registry) -> None:
    plan = DeterministicPlanner().plan(
        _request(
            "Inspect the dataset and compute EpiZoo embeddings",
            {
                "input_path": "/data/in.h5ad",
                "output_dir": "/output",
                "species": "mouse",
            },
        ),
        registry,
    )

    assert tuple(step.step_id for step in plan.steps) == ("inspect", "embed")


def test_downstream_request_produces_exact_five_step_reference_chain(registry) -> None:
    plan = DeterministicPlanner().plan(
        _request(
            "Cluster the cells and compute UMAP",
            {
                "input_path": "/data/in.h5ad",
                "output_dir": "/output",
                "species": "mouse",
            },
        ),
        registry,
    )
    assert plan.plan_id == "request-1:epizoo-downstream-analysis"
    assert tuple(step.step_id for step in plan.steps) == (
        "inspect",
        "embed",
        "neighbors",
        "cluster",
        "umap",
    )
    assert tuple(step.tool_name for step in plan.steps) == registry.names()[:5]
    assert plan.steps[1].arguments["input_path"] == StepOutputRef(
        "inspect", "input_path"
    )
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
    assert tuple(step.depends_on for step in plan.steps) == (
        (),
        ("inspect",),
        ("embed",),
        ("neighbors",),
        ("cluster",),
    )


def test_downstream_defaults_are_omitted_not_generated_as_literals(registry) -> None:
    plan = DeterministicPlanner().plan(
        _request(
            "Build neighbors, cluster, and UMAP",
            {
                "input_path": "/data/in.h5ad",
                "output_dir": "/output",
                "species": "mouse",
            },
        ),
        registry,
    )
    assert set(plan.steps[2].arguments) == {
        "embedding_path",
        "cell_ids_path",
        "output_dir",
    }
    assert set(plan.steps[3].arguments) == {"analysis_path", "output_dir"}
    assert set(plan.steps[4].arguments) == {"analysis_path", "output_dir"}


def test_clustering_evaluation_intent_produces_exact_five_step_plan(registry) -> None:
    plan = DeterministicPlanner().plan(
        _request(
            "Analyze this mouse scATAC dataset with EpiZoo and evaluate the clustering",
            {
                "input_path": "/data/in.h5ad",
                "output_dir": "/output",
                "species": "mouse",
                "label_key": "celltype",
            },
        ),
        registry,
    )
    assert plan.plan_id == "request-1:epizoo-clustering-evaluation"
    assert tuple(step.step_id for step in plan.steps) == (
        "inspect", "embed", "neighbors", "cluster", "evaluate"
    )
    evaluation = plan.steps[-1]
    assert evaluation.tool_name == "evaluate_cell_clustering"
    assert evaluation.depends_on == ("cluster", "inspect")
    assert evaluation.arguments == {
        "analysis_path": StepOutputRef("cluster", "analysis_path"),
        "reference_h5ad_path": StepOutputRef("inspect", "input_path"),
        "label_key": "celltype",
        "output_dir": "/output",
    }
    assert "cluster_key" not in evaluation.arguments


def test_clustering_evaluation_forwards_only_explicit_optional_values(registry) -> None:
    plan = DeterministicPlanner().plan(
        _request(
            "Evaluate clustering metrics",
            {
                "input_path": "/data/in.h5ad",
                "output_dir": "/output",
                "species": "mouse",
                "label_key": "celltype",
                "cluster_key": "clusters",
                "overwrite": True,
            },
        ),
        registry,
    )
    assert plan.steps[-1].arguments["cluster_key"] == "clusters"
    assert plan.steps[-1].arguments["overwrite"] is True


def test_clustering_evaluation_requires_structured_label_key(registry) -> None:
    with pytest.raises(PlannerError, match="label_key"):
        DeterministicPlanner().plan(
            _request(
                "Evaluate the clustering",
                {
                    "input_path": "/data/in.h5ad",
                    "output_dir": "/output",
                    "species": "mouse",
                },
            ),
            registry,
        )


def test_explicit_downstream_inputs_are_forwarded_to_correct_steps(registry) -> None:
    plan = DeterministicPlanner().plan(
        _request(
            "Run the full analysis with UMAP",
            {
                "input_path": "/data/in.h5ad",
                "output_dir": "/output",
                "species": "human",
                "n_neighbors": 20,
                "metric": "cosine",
                "random_seed": 7,
                "resolution": 0.8,
                "min_dist": 0.2,
                "spread": 1.2,
                "overwrite": True,
            },
        ),
        registry,
    )
    assert dict(plan.steps[2].arguments) == {
        "embedding_path": StepOutputRef("embed", "embedding_path"),
        "cell_ids_path": StepOutputRef("embed", "cell_ids_path"),
        "output_dir": "/output",
        "n_neighbors": 20,
        "metric": "cosine",
        "random_seed": 7,
        "overwrite": True,
    }
    assert plan.steps[3].arguments["resolution"] == 0.8
    assert plan.steps[3].arguments["random_seed"] == 7
    assert plan.steps[4].arguments["min_dist"] == 0.2
    assert plan.steps[4].arguments["spread"] == 1.2
    assert plan.steps[4].arguments["random_seed"] == 7


@pytest.mark.parametrize(
    ("prompt", "inputs", "missing_name"),
    [
        ("inspect dataset", {}, "input_path"),
        (
            "compute embeddings",
            {"output_dir": "/output", "species": "mouse"},
            "input_path",
        ),
        (
            "compute embeddings",
            {"input_path": "/data/in.h5ad", "species": "mouse"},
            "output_dir",
        ),
        (
            "compute embeddings",
            {"input_path": "/data/in.h5ad", "output_dir": "/output"},
            "species",
        ),
    ],
)
def test_workflows_require_structured_inputs(
    registry, prompt, inputs, missing_name
) -> None:
    with pytest.raises(PlannerError) as raised:
        DeterministicPlanner().plan(_request(prompt, inputs), registry)

    assert raised.value.code == "MISSING_REQUIRED_INPUT"
    assert raised.value.category is ErrorCategory.USER_INPUT_ERROR
    assert missing_name in str(raised.value)


@pytest.mark.parametrize("species", ["rat", "", 1, None])
def test_invalid_species_is_rejected(registry, species) -> None:
    with pytest.raises(PlannerError) as raised:
        DeterministicPlanner().plan(
            _request(
                "compute embedding",
                {
                    "input_path": "/data/in.h5ad",
                    "output_dir": "/output",
                    "species": species,
                },
            ),
            registry,
        )

    assert raised.value.code == "INVALID_REQUEST_INPUT"


def test_empty_path_input_is_rejected(registry) -> None:
    with pytest.raises(PlannerError, match="non-empty path"):
        DeterministicPlanner().plan(
            _request("inspect", {"input_path": "  "}), registry
        )


def test_optional_embedding_arguments_are_forwarded(registry) -> None:
    plan = DeterministicPlanner().plan(
        _request(
            "EpiZoo embedding",
            {
                "input_path": "/data/in.h5ad",
                "output_dir": "/output",
                "species": " Human ",
                "checkpoint_path": "/models/model.pth",
                "device": "cuda:0",
                "overwrite": True,
                "unregistered_option": "ignored",
            },
        ),
        registry,
    )

    arguments = plan.steps[1].arguments
    assert arguments["species"] == "human"
    assert arguments["checkpoint_path"] == "/models/model.pth"
    assert arguments["device"] == "cuda:0"
    assert arguments["overwrite"] is True
    assert "unregistered_option" not in arguments


def test_generated_plan_and_arguments_validate(registry) -> None:
    plan = DeterministicPlanner().plan(
        _request(
            "summarize this dataset and make embeddings",
            {
                "input_path": "/data/in.h5ad",
                "output_dir": "/output",
                "species": "mouse",
            },
        ),
        registry,
    )

    assert tuple(step.step_id for step in plan.stable_topological_steps()) == (
        "inspect",
        "embed",
    )
    for step in plan.steps:
        registry.validate_arguments(step.tool_name, step.arguments)


@pytest.mark.parametrize(
    "prompt",
    ["tell me about chromatin", "this dataset is inspectable"],
)
def test_unsupported_or_unrelated_prompt_is_rejected(registry, prompt) -> None:
    with pytest.raises(PlannerError) as raised:
        DeterministicPlanner().plan(
            _request(prompt, {"input_path": "/data/in.h5ad"}), registry
        )

    assert raised.value.code == "UNSUPPORTED_REQUEST"


def test_registry_contract_mismatch_is_an_internal_planner_error(registry) -> None:
    embedding_only = ToolRegistry((registry.get("epizoo_embed_cells"),))

    with pytest.raises(PlannerError) as raised:
        DeterministicPlanner().plan(
            _request("inspect", {"input_path": "/data/in.h5ad"}),
            embedding_only,
        )

    assert raised.value.code == "PLANNER_REGISTRY_CONTRACT_MISMATCH"
    assert raised.value.category is ErrorCategory.INTERNAL_AGENT_ERROR


def test_planner_never_executes_registered_tools(registry) -> None:
    inspection_callable = Mock(side_effect=AssertionError("must not execute"))
    embedding_callable = Mock(side_effect=AssertionError("must not execute"))
    guarded_registry = ToolRegistry(
        (
            replace(registry.get("inspect_scATAC"), function=inspection_callable),
            replace(
                registry.get("epizoo_embed_cells"), function=embedding_callable
            ),
        )
    )

    DeterministicPlanner().plan(
        _request(
            "inspect and embed",
            {
                "input_path": "/data/in.h5ad",
                "output_dir": "/output",
                "species": "mouse",
            },
        ),
        guarded_registry,
    )

    inspection_callable.assert_not_called()
    embedding_callable.assert_not_called()


def test_run_mode_does_not_change_scientific_plan(registry) -> None:
    inputs = {
        "input_path": "/data/in.h5ad",
        "output_dir": "/output",
        "species": "mouse",
    }
    planner = DeterministicPlanner()

    execute_plan = planner.plan(
        _request("embedding", inputs, mode=RunMode.EXECUTE), registry
    )
    plan_only = planner.plan(
        _request("embedding", inputs, mode=RunMode.PLAN_ONLY), registry
    )

    assert execute_plan == plan_only


def test_planner_requires_no_provider_or_network_configuration(registry) -> None:
    plan = DeterministicPlanner().plan(
        _request("inspection", {"input_path": "/data/in.h5ad"}), registry
    )

    assert plan.steps[0].tool_name == "inspect_scATAC"
