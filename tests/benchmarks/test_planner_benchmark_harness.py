from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.orchestration import (
    PlanningModelError,
    PlanningModelProfile,
    build_default_tool_registry,
)
from agent.providers import PlanningModelFactoryRegistry
from benchmarks.planner.benchmark import (
    BenchmarkCase,
    ScriptedPlanningModel,
    load_cases,
    oracle_response,
    run_benchmark,
)
from benchmarks.planner.run_benchmark import main


ROOT = Path(__file__).resolve().parents[2]
CASES_PATH = ROOT / "benchmarks" / "planner" / "cases.json"

REQUIRED_CATEGORY_TAGS = {
    "canonical",
    "paraphrase",
    "terse",
    "verbose",
    "multi_step",
    "dependency_heavy",
    "irrelevant_text",
    "missing_information",
    "ambiguous",
    "conflicting",
    "unsupported",
    "tool_name_mention",
    "hallucination_trap",
    "incorrect_input_key",
    "incorrect_result_reference",
    "prompt_injection",
    "mixed_supported_unsupported",
}

SUPPORTED_WORKFLOWS = {
    "inspection",
    "epizoo-embedding",
    "epizoo-downstream-analysis",
    "epizoo-clustering-evaluation",
    "epizoo-label-transfer",
    "cell-annotation-evaluation",
    "epizoo-label-transfer-evaluation",
    "replicate-aware-pseudobulk",
    "replicate-differential-accessibility",
    "raw-to-replicate-differential-accessibility",
}

REPRESENTATIVE_ACCEPTANCE_DAGS = {
    "inspect_canonical": ("inspect_scATAC",),
    "embedding_terse": ("inspect_scATAC", "epizoo_embed_cells"),
    "downstream_canonical": (
        "inspect_scATAC",
        "epizoo_embed_cells",
        "build_cell_neighbors",
        "cluster_cells",
        "compute_cell_umap",
    ),
    "clustering_evaluation_canonical": (
        "inspect_scATAC",
        "epizoo_embed_cells",
        "build_cell_neighbors",
        "cluster_cells",
        "evaluate_cell_clustering",
    ),
    "label_transfer_canonical": (
        "inspect_scATAC",
        "epizoo_embed_cells",
        "inspect_scATAC",
        "epizoo_embed_cells",
        "transfer_cell_labels",
    ),
    "transfer_and_annotation_evaluation": (
        "inspect_scATAC",
        "epizoo_embed_cells",
        "inspect_scATAC",
        "epizoo_embed_cells",
        "transfer_cell_labels",
        "evaluate_cell_annotation",
    ),
    "pseudobulk_canonical": (
        "validate_scATAC_feature_space",
        "build_replicate_pseudobulk",
    ),
    "differential_accessibility_fixed": (
        "run_replicate_differential_accessibility",
    ),
    "differential_accessibility_raw": (
        "validate_scATAC_feature_space",
        "build_replicate_pseudobulk",
        "run_replicate_differential_accessibility",
    ),
}


@pytest.fixture(scope="module")
def cases() -> tuple[BenchmarkCase, ...]:
    return load_cases(CASES_PATH)


def _case(cases: tuple[BenchmarkCase, ...], case_id: str) -> BenchmarkCase:
    return next(case for case in cases if case.case_id == case_id)


def _binding(step: dict[str, object], name: str) -> dict[str, object]:
    arguments = step["arguments"]
    assert isinstance(arguments, dict)
    binding = arguments[name]
    assert isinstance(binding, dict)
    return binding


def _bind_to_input(binding: dict[str, object], input_name: str) -> None:
    binding.clear()
    binding.update({"binding_type": "input", "input_name": input_name})


def _without_argument(step: dict[str, object], name: str) -> None:
    arguments = step["arguments"]
    assert isinstance(arguments, dict)
    arguments[name] = None


def test_corpus_covers_required_categories_workflows_and_tools(
    cases: tuple[BenchmarkCase, ...],
) -> None:
    assert 30 <= len(cases) <= 40
    assert len({case.case_id for case in cases}) == len(cases)

    tags = {tag for case in cases for tag in case.tags}
    assert REQUIRED_CATEGORY_TAGS <= tags

    plan_cases = tuple(case for case in cases if case.expected_outcome == "plan")
    assert {case.expected_workflow for case in plan_cases} == SUPPORTED_WORKFLOWS
    expected_tools = {
        str(step["tool"])
        for case in plan_cases
        for step in case.expected_steps
    }
    assert expected_tools == set(build_default_tool_registry().names())


def test_corpus_uses_only_synthetic_path_values(
    cases: tuple[BenchmarkCase, ...],
) -> None:
    def strings(value: object):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for item in value.values():
                yield from strings(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from strings(item)

    path_values = (
        value
        for case in cases
        for key, raw_value in case.inputs.items()
        if "path" in key or key == "output_dir"
        for value in strings(raw_value)
    )
    assert all(value.startswith("/synthetic/") for value in path_values)


def test_corpus_oracle_responses_match_every_structural_expectation(
    cases: tuple[BenchmarkCase, ...],
) -> None:
    report = run_benchmark(cases)

    assert all(score.semantically_correct for score in report.cases)
    assert report.metrics["scientific_call_count"] == 0


@pytest.mark.parametrize(
    ("case_id", "expected_tools"), REPRESENTATIVE_ACCEPTANCE_DAGS.items()
)
def test_final_supported_workflow_dags_are_semantic_preflight_valid_plans(
    cases: tuple[BenchmarkCase, ...],
    case_id: str,
    expected_tools: tuple[str, ...],
) -> None:
    """Exercise the complete representative DAG through the production Planner."""

    score = run_benchmark((_case(cases, case_id),)).cases[0]

    assert score.hard_semantic_correct
    assert score.syntactically_valid_plan
    assert score.preflight_valid_plan
    assert score.final_planning_success
    assert score.actual_tool_sequence == expected_tools
    assert score.binding_correct == score.binding_total
    assert (
        score.dependency_reference_correct
        == score.dependency_reference_total
    )
    assert score.provider_calls == 1
    assert score.scientific_calls == 0


def test_provider_step_ids_and_descriptions_are_normalized_away(
    cases: tuple[BenchmarkCase, ...],
) -> None:
    case = _case(cases, "embedding_canonical")
    baseline = json.loads(oracle_response(case))
    changed = json.loads(oracle_response(case))
    changed["steps"][0]["step_id"] = "opaque-alpha"
    changed["steps"][1]["step_id"] = "opaque-beta"
    changed["steps"][0]["description"] = None
    changed["steps"][1]["description"] = "Unrelated provider prose."
    changed["steps"][1]["depends_on"] = ["opaque-alpha"]
    changed["steps"][1]["arguments"]["input_path"]["ref_step_id"] = (
        "opaque-alpha"
    )

    baseline_report = run_benchmark(
        (case,), replay_overrides={case.case_id: {"response": baseline}}
    )
    changed_report = run_benchmark(
        (case,), replay_overrides={case.case_id: {"response": changed}}
    )

    baseline_score = baseline_report.cases[0]
    changed_score = changed_report.cases[0]
    assert baseline_score.semantically_correct
    assert changed_score.semantically_correct
    assert changed_score.binding_correct == changed_score.binding_total
    assert (
        changed_score.dependency_reference_correct
        == changed_score.dependency_reference_total
    )


def test_binding_origin_is_scored_separately_from_preflight(
    cases: tuple[BenchmarkCase, ...],
) -> None:
    case = _case(cases, "embedding_canonical")
    response = json.loads(oracle_response(case))
    output_binding = response["steps"][1]["arguments"]["output_dir"]
    output_binding["input_name"] = "input_path"

    score = run_benchmark(
        (case,),
        replay_overrides={case.case_id: {"response": response}},
    ).cases[0]

    assert score.syntactically_valid_plan
    assert score.preflight_valid_plan
    assert score.exact_tool_sequence
    assert score.binding_correct == score.binding_total - 1
    assert not score.semantically_correct
    assert score.semantic_wrong_but_preflight_valid


def test_embed_alone_is_semantic_when_inspection_was_not_requested(
    cases: tuple[BenchmarkCase, ...],
) -> None:
    case = _case(cases, "embedding_terse")
    response = json.loads(oracle_response(case))
    embed = response["steps"][1]
    _bind_to_input(_binding(embed, "input_path"), "input_path")
    embed["depends_on"] = []
    response["steps"] = [embed]

    score = run_benchmark(
        (case,), replay_overrides={case.case_id: {"response": response}}
    ).cases[0]

    assert score.preflight_valid_plan
    assert score.hard_semantic_correct
    assert not score.canonical_workflow_conformant
    assert not score.exact_tool_sequence


def test_explicitly_requested_inspection_remains_required(
    cases: tuple[BenchmarkCase, ...],
) -> None:
    case = _case(cases, "embedding_canonical")
    response = json.loads(oracle_response(case))
    embed = response["steps"][1]
    _bind_to_input(_binding(embed, "input_path"), "input_path")
    embed["depends_on"] = []
    response["steps"] = [embed]

    score = run_benchmark(
        (case,), replay_overrides={case.case_id: {"response": response}}
    ).cases[0]

    assert score.preflight_valid_plan
    assert not score.hard_semantic_correct
    assert "missing_required_role:inspect" in score.hard_semantic_failures


def test_audited_direct_path_is_equivalent_but_wrong_path_is_not(
    cases: tuple[BenchmarkCase, ...],
) -> None:
    case = _case(cases, "embedding_canonical")
    direct = json.loads(oracle_response(case))
    _bind_to_input(_binding(direct["steps"][1], "input_path"), "input_path")
    direct_score = run_benchmark(
        (case,), replay_overrides={case.case_id: {"response": direct}}
    ).cases[0]

    wrong = json.loads(json.dumps(direct))
    _bind_to_input(_binding(wrong["steps"][1], "input_path"), "output_dir")
    wrong_score = run_benchmark(
        (case,), replay_overrides={case.case_id: {"response": wrong}}
    ).cases[0]

    assert direct_score.preflight_valid_plan
    assert direct_score.hard_semantic_correct
    assert not direct_score.canonical_workflow_conformant
    assert wrong_score.preflight_valid_plan
    assert not wrong_score.hard_semantic_correct
    assert "binding_mismatch:embed.input_path" in wrong_score.hard_semantic_failures


def test_reference_query_branch_permutation_passes_but_swap_fails(
    cases: tuple[BenchmarkCase, ...],
) -> None:
    case = _case(cases, "label_transfer_canonical")
    permuted = json.loads(oracle_response(case))
    steps = permuted["steps"]
    permuted["steps"] = [steps[2], steps[3], steps[0], steps[1], steps[4]]
    permuted_score = run_benchmark(
        (case,), replay_overrides={case.case_id: {"response": permuted}}
    ).cases[0]

    swapped = json.loads(oracle_response(case))
    inspect_reference, embed_reference, inspect_query, embed_query, _ = swapped[
        "steps"
    ]
    reference_input = _binding(embed_reference, "input_path")
    reference_input["ref_step_id"] = inspect_query["step_id"]
    embed_reference["depends_on"] = [inspect_query["step_id"]]
    query_input = _binding(embed_query, "input_path")
    query_input["ref_step_id"] = inspect_reference["step_id"]
    embed_query["depends_on"] = [inspect_reference["step_id"]]
    swapped_score = run_benchmark(
        (case,), replay_overrides={case.case_id: {"response": swapped}}
    ).cases[0]

    assert permuted_score.preflight_valid_plan
    assert permuted_score.hard_semantic_correct
    assert not permuted_score.canonical_workflow_conformant
    assert swapped_score.preflight_valid_plan
    assert not swapped_score.hard_semantic_correct
    assert any(
        failure.startswith("binding_mismatch:")
        for failure in swapped_score.hard_semantic_failures
    )


def test_partial_order_equivalent_transfer_dag_passes(
    cases: tuple[BenchmarkCase, ...],
) -> None:
    case = _case(cases, "label_transfer_canonical")
    response = json.loads(oracle_response(case))
    _, embed_reference, _, embed_query, transfer = response["steps"]
    _bind_to_input(
        _binding(embed_reference, "input_path"), "reference_input_path"
    )
    _bind_to_input(_binding(embed_query, "input_path"), "query_input_path")
    _bind_to_input(
        _binding(transfer, "reference_h5ad_path"), "reference_input_path"
    )
    _bind_to_input(_binding(transfer, "query_h5ad_path"), "query_input_path")
    embed_reference["depends_on"] = []
    embed_query["depends_on"] = []
    transfer["depends_on"] = [embed_reference["step_id"], embed_query["step_id"]]
    response["steps"] = [embed_query, embed_reference, transfer]

    score = run_benchmark(
        (case,), replay_overrides={case.case_id: {"response": response}}
    ).cases[0]

    assert score.preflight_valid_plan
    assert score.hard_semantic_correct
    assert not score.canonical_workflow_conformant


def test_missing_required_artifact_flow_fails_even_when_preflight_valid(
    cases: tuple[BenchmarkCase, ...],
) -> None:
    case = _case(cases, "downstream_paraphrase")
    response = json.loads(oracle_response(case))
    neighbors = response["steps"][2]
    _bind_to_input(_binding(neighbors, "embedding_path"), "input_path")

    score = run_benchmark(
        (case,), replay_overrides={case.case_id: {"response": response}}
    ).cases[0]

    assert score.preflight_valid_plan
    assert not score.hard_semantic_correct
    assert "binding_mismatch:neighbors.embedding_path" in score.hard_semantic_failures


def test_default_equivalent_omission_passes_but_nondefault_omission_fails(
    cases: tuple[BenchmarkCase, ...],
) -> None:
    default_case = _case(cases, "annotation_evaluation_standalone")
    default_response = json.loads(oracle_response(default_case))
    _without_argument(default_response["steps"][0], "overwrite")
    default_score = run_benchmark(
        (default_case,),
        replay_overrides={default_case.case_id: {"response": default_response}},
    ).cases[0]

    nondefault_case = _case(cases, "downstream_explicit_steps")
    nondefault_response = json.loads(oracle_response(nondefault_case))
    _without_argument(nondefault_response["steps"][2], "n_neighbors")
    nondefault_score = run_benchmark(
        (nondefault_case,),
        replay_overrides={
            nondefault_case.case_id: {"response": nondefault_response}
        },
    ).cases[0]

    assert default_score.preflight_valid_plan
    assert default_score.hard_semantic_correct
    assert not default_score.canonical_workflow_conformant
    assert nondefault_score.preflight_valid_plan
    assert not nondefault_score.hard_semantic_correct
    assert (
        "binding_mismatch:neighbors.n_neighbors"
        in nondefault_score.hard_semantic_failures
    )


def test_normalized_report_retains_structure_without_provider_data_or_values(
    cases: tuple[BenchmarkCase, ...],
) -> None:
    case = _case(cases, "embedding_canonical")
    score = run_benchmark((case,)).cases[0]
    rendered = json.dumps(score.to_dict(), sort_keys=True)

    assert score.normalized_plan[0]["role"] == "inspect"
    assert score.normalized_plan[1]["bindings"]["input_path"] == {
        "kind": "ref",
        "producer_role": "inspect",
        "output_key": "input_path",
    }
    assert score.normalized_plan[1]["depends_on_roles"] == ["inspect"]
    assert "provider-step" not in rendered
    assert "Provider prose" not in rendered
    assert "/synthetic/" not in rendered


def test_dependency_and_result_reference_are_scored_structurally(
    cases: tuple[BenchmarkCase, ...],
) -> None:
    case = _case(cases, "embedding_canonical")
    response = json.loads(oracle_response(case))
    response["steps"][1]["arguments"]["input_path"]["ref_output_key"] = (
        "invented_result_path"
    )

    score = run_benchmark(
        (case,),
        replay_overrides={case.case_id: {"response": response}},
    ).cases[0]

    assert score.syntactically_valid_plan
    assert not score.preflight_valid_plan
    assert score.exact_tool_sequence
    assert score.dependency_reference_correct < score.dependency_reference_total
    assert not score.semantically_correct


def test_supplied_model_track_repeats_plan_only_without_scientific_calls(
    cases: tuple[BenchmarkCase, ...],
) -> None:
    case = _case(cases, "inspect_canonical")
    model = ScriptedPlanningModel(oracle_response(case))
    profile = PlanningModelProfile(
        profile_id="benchmark-test-model",
        provider_id="custom",
        model_id=model.model_id,
    )

    report = run_benchmark(
        (case,), model=model, model_profile=profile, repetitions=2
    )

    assert report.track == "live-provider"
    assert report.profile_id == "benchmark-test-model"
    assert report.provider_id == "custom"
    assert report.model_id == model.model_id
    assert report.to_dict()["schema_version"] == 4
    assert report.repetitions == 2
    assert model.calls == 2
    assert report.metrics["provider_calls_per_request"] == 1.0
    assert report.metrics["maximum_provider_calls"] == 1
    assert report.metrics["scientific_call_count"] == 0
    assert all(score.semantically_correct for score in report.cases)


def test_schema_v4_distinguishes_transport_recovered_success(
    cases: tuple[BenchmarkCase, ...],
) -> None:
    case = _case(cases, "inspect_canonical")

    class TransientThenValid:
        model_id = "transport-recovery-test"

        def __init__(self) -> None:
            self.calls = 0

        def complete(self, *, prompt, response_schema):
            self.calls += 1
            if self.calls == 1:
                raise PlanningModelError(
                    code="PROVIDER_TIMEOUT", retry_after_seconds=0
                )
            return oracle_response(case)

    model = TransientThenValid()
    profile = PlanningModelProfile(
        "transport-recovery-test", "custom", model.model_id
    )

    report = run_benchmark((case,), model=model, model_profile=profile)
    score = report.cases[0]

    assert report.to_dict()["schema_version"] == 4
    assert score.provider_calls == 2
    assert score.retry_used
    assert score.transport_recovered
    assert not score.first_attempt_semantic_correct
    assert score.final_provider_failure is None
    assert report.metrics["transport_recovery_success_rate"] == 1.0
    assert report.metrics["transport_retry_rate"] == 1.0
    assert report.metrics["first_attempt_plan_success_rate"] == 0.0
    assert report.metrics["final_plan_success_rate"] == 1.0


def test_schema_v4_distinguishes_repair_recovered_success(
    cases: tuple[BenchmarkCase, ...],
) -> None:
    case = _case(cases, "inspect_canonical")

    class InvalidThenValid:
        model_id = "plan-repair-test"

        def __init__(self) -> None:
            self.calls = 0

        def complete(self, *, prompt, response_schema):
            self.calls += 1
            return "not-json" if self.calls == 1 else oracle_response(case)

    model = InvalidThenValid()
    profile = PlanningModelProfile("plan-repair-test", "custom", model.model_id)

    report = run_benchmark((case,), model=model, model_profile=profile)
    score = report.cases[0]

    assert score.provider_calls == 2
    assert score.repair_attempted
    assert score.repair_success
    assert not score.retry_used
    assert score.recovery_path == "repair_recovered"
    assert score.final_failure_class is None
    assert report.metrics["repair_success_rate"] == 1.0
    assert report.metrics["final_plan_success_rate"] == 1.0


def test_schema_v4_records_configured_failover_success_and_profile_order(
    cases: tuple[BenchmarkCase, ...],
) -> None:
    case = _case(cases, "inspect_canonical")

    class ExhaustedPrimary:
        model_id = "exhausted-primary"

        def __init__(self) -> None:
            self.calls = 0

        def complete(self, *, prompt, response_schema):
            del prompt, response_schema
            self.calls += 1
            raise PlanningModelError(
                code="PROVIDER_TIMEOUT", retry_after_seconds=0
            )

    primary = ExhaustedPrimary()
    primary_profile = PlanningModelProfile(
        "benchmark-primary", "custom", primary.model_id
    )
    secondary_profile = PlanningModelProfile(
        "benchmark-secondary", "backup", "secondary-model"
    )
    factory_calls: list[PlanningModelProfile] = []

    def factory(profile: PlanningModelProfile):
        factory_calls.append(profile)
        return ScriptedPlanningModel(oracle_response(case))

    factories = PlanningModelFactoryRegistry({"backup": factory})

    report = run_benchmark(
        (case,),
        model=primary,
        model_profile=primary_profile,
        recovery_profiles=(secondary_profile,),
        model_factory_registry=factories,
    )
    score = report.cases[0]

    assert primary.calls == 2
    assert factory_calls == [secondary_profile]
    assert score.provider_calls == 3
    assert score.failover_attempted
    assert score.failover_success
    assert not score.transport_recovered
    assert score.final_planning_success
    assert score.recovery_path == "failover_recovered"
    assert score.final_recovery_source == "secondary_failover"
    assert score.ordered_profile_usage == (
        "benchmark-primary",
        "benchmark-primary",
        "benchmark-secondary",
    )
    assert report.metrics["failover_attempt_rate"] == 1.0
    assert report.metrics["failover_success_rate"] == 1.0
    assert report.metrics["final_plan_success_rate"] == 1.0
    assert report.metrics["fallback_rate"] is None


def test_schema_v3_uses_explicit_profile_provenance_without_model_id_parsing(
    cases: tuple[BenchmarkCase, ...],
) -> None:
    case = _case(cases, "inspect_canonical")
    model = ScriptedPlanningModel(oracle_response(case))
    first_profile = PlanningModelProfile(
        profile_id="groq-model-a",
        provider_id="groq",
        model_id="organization/model-a",
    )
    second_profile = PlanningModelProfile(
        profile_id="groq-model-b",
        provider_id="groq",
        model_id="organization/model-b",
    )

    first = run_benchmark(
        (case,), model=model, model_profile=first_profile
    ).to_dict()
    second = run_benchmark(
        (case,), model=model, model_profile=second_profile
    ).to_dict()

    assert first["schema_version"] == 4
    assert first["profile_id"] == "groq-model-a"
    assert first["provider_id"] == "groq"
    assert first["model_id"] == "organization/model-a"
    assert "provider" not in first
    assert "model" not in first
    assert first["profile_id"] != second["profile_id"]
    assert first["model_id"] != second["model_id"]


def test_provider_options_require_explicit_live_opt_in() -> None:
    with pytest.raises(
        ValueError,
        match="--provider and --model require the explicit --live flag",
    ):
        main(["--provider", "groq", "--model", "synthetic-model"])


def test_live_opt_in_requires_a_provider_and_model() -> None:
    with pytest.raises(
        ValueError,
        match="Live benchmark requires --provider and --model",
    ):
        main(["--live"])
