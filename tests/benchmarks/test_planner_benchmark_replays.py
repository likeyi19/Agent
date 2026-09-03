from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.planner.benchmark import (
    BenchmarkReport,
    load_cases,
    load_replay_overrides,
    run_benchmark,
)


ROOT = Path(__file__).resolve().parents[2]
CASES_PATH = ROOT / "benchmarks" / "planner" / "cases.json"
REPLAYS_PATH = ROOT / "tests" / "benchmarks" / "fixtures" / "offline_replays.json"


@pytest.fixture(scope="module")
def report() -> BenchmarkReport:
    return run_benchmark(
        load_cases(CASES_PATH),
        replay_overrides=load_replay_overrides(REPLAYS_PATH),
    )


def _score(report: BenchmarkReport, case_id: str):
    return next(score for score in report.cases if score.case_id == case_id)


def test_offline_replay_is_deterministic(report: BenchmarkReport) -> None:
    repeated = run_benchmark(
        load_cases(CASES_PATH),
        replay_overrides=load_replay_overrides(REPLAYS_PATH),
    )
    assert repeated.to_dict() == report.to_dict()


def test_offline_baseline_metrics_and_future_fields(report: BenchmarkReport) -> None:
    metrics = report.metrics
    assert report.track == "offline-replay"
    assert metrics["request_count"] == 38
    assert metrics["supported_request_count"] == 26
    assert metrics["unsupported_request_count"] == 7
    assert metrics["failure_request_count"] == 5
    assert metrics["planning_success_rate"] == 1.0
    assert metrics["executable_plan_rate"] == pytest.approx(25 / 26)
    assert metrics["exact_tool_sequence_accuracy"] == pytest.approx(25 / 26)
    assert metrics["hard_semantic_success_rate"] == pytest.approx(33 / 38)
    assert metrics["canonical_workflow_conformance_rate"] == pytest.approx(24 / 26)
    assert metrics["argument_binding_accuracy"] == pytest.approx(287 / 292)
    assert metrics["dependency_reference_accuracy"] == pytest.approx(153 / 155)
    assert metrics["hallucinated_tool_rate"] == pytest.approx(1 / 80)
    assert metrics["unsupported_request_rejection_accuracy"] == pytest.approx(5 / 7)
    assert metrics["false_unsupported_rate"] == 0.0
    assert metrics["unsupported_false_acceptance_rate"] == pytest.approx(2 / 7)
    assert metrics["semantic_wrong_but_preflight_valid_rate"] == pytest.approx(2 / 26)
    assert metrics["first_attempt_semantic_success_rate"] == pytest.approx(33 / 38)
    assert metrics["final_planning_success_rate"] == pytest.approx(33 / 38)
    assert metrics["repair_success_rate"] is None
    assert metrics["fallback_rate"] is None
    assert metrics["provider_calls_per_request"] == 1.0
    assert metrics["maximum_provider_calls"] == 1
    assert metrics["scientific_call_count"] == 0


def test_report_persists_hard_and_canonical_assessments_separately(
    report: BenchmarkReport,
) -> None:
    payload = report.to_dict()
    assert payload["schema_version"] == 3
    assert payload["profile_id"] == "offline-scripted-m9.1"
    assert payload["provider_id"] == "offline"
    assert payload["model_id"] == "offline-scripted-m9.1"
    assert all("hard_semantic_correct" in case for case in payload["cases"])
    assert all("canonical_workflow_conformant" in case for case in payload["cases"])
    assert all("normalized_plan" in case for case in payload["cases"])


def test_metrics_are_reported_for_every_corpus_tag(report: BenchmarkReport) -> None:
    expected_tags = {tag for score in report.cases for tag in score.tags}
    assert set(report.metrics_by_tag) == expected_tags
    assert all(
        metrics["request_count"] >= 1
        for metrics in report.metrics_by_tag.values()
    )
    assert all(
        metrics["scientific_call_count"] == 0
        for metrics in report.metrics_by_tag.values()
    )


def test_preflight_valid_semantic_error_is_detected(report: BenchmarkReport) -> None:
    score = _score(report, "inspect_verbose_context")
    assert score.syntactically_valid_plan
    assert score.preflight_valid_plan
    assert not score.semantically_correct
    assert score.semantic_wrong_but_preflight_valid


def test_unsupported_false_acceptance_is_detected(report: BenchmarkReport) -> None:
    score = _score(report, "unsupported_rna_analysis")
    assert score.syntactically_valid_plan
    assert score.preflight_valid_plan
    assert score.unsupported_false_acceptance
    assert not score.semantically_correct


def test_hallucinated_tool_is_measured_and_fails_preflight(
    report: BenchmarkReport,
) -> None:
    score = _score(report, "hallucinated_tool_trap")
    assert score.syntactically_valid_plan
    assert not score.preflight_valid_plan
    assert score.actual_error_code == "UNKNOWN_TOOL"
    assert score.hallucinated_tool_count == 1
    assert score.emitted_tool_count == 1


def test_parser_and_bad_reference_failures_are_distinct(
    report: BenchmarkReport,
) -> None:
    malformed = _score(report, "embedding_missing_output_dir")
    bad_reference = _score(report, "embedding_incorrect_result_reference_trap")

    assert not malformed.syntactically_valid_plan
    assert malformed.actual_error_code == "PLANNER_OUTPUT_INVALID"
    assert malformed.provider_calls == 1

    assert bad_reference.syntactically_valid_plan
    assert not bad_reference.preflight_valid_plan
    assert bad_reference.actual_error_code == "INVALID_OUTPUT_REFERENCE"
    assert bad_reference.exact_tool_sequence
    assert (
        bad_reference.dependency_reference_correct
        < bad_reference.dependency_reference_total
    )


def test_m91_has_one_attempt_and_no_repair_fallback_or_execution(
    report: BenchmarkReport,
) -> None:
    assert all(score.provider_calls == 1 for score in report.cases)
    assert all(score.scientific_calls == 0 for score in report.cases)
    assert (
        report.metrics["first_attempt_semantic_success_rate"]
        == report.metrics["final_planning_success_rate"]
    )
    assert report.metrics["repair_success_rate"] is None
    assert report.metrics["fallback_rate"] is None
