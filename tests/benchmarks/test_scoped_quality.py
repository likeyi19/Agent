"""Offline calibration of the measurement instrument, never live model quality."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from agent.orchestration import (
    PlanningModelError,
    PlanningModelProfile,
    build_default_tool_registry,
)
from benchmarks.planner import scoped as q
from benchmarks.planner.run_benchmark import main


@pytest.fixture(scope="module")
def cases():
    return {c.case.case_id: c for c in q.load_scoped_cases()}


@pytest.fixture(scope="module")
def registry():
    return build_default_tool_registry()


def selection(*families):
    return dict(
        selection_schema_version=1,
        decision=dict(kind="select", capability_ids=list(families)),
    )


def selected(c):
    return selection(
        *(c.scope["required_coverage"] + c.scope["plausible_optional_families"])
    )


def run(c, *outcomes):
    return q.run_scoped_benchmark(
        (c,), replay_outcomes={c.case.case_id: list(outcomes)}
    ).to_dict()


def detail(c, payload, registry):
    return q.inspect_candidate(
        c,
        json.dumps(payload),
        registry,
        q.PlanningScope(
            registry,
            tuple(
                c.scope["required_coverage"] + c.scope["plausible_optional_families"]
            ),
        ),
    )


def test_complete_corpus_and_no_science(cases, registry):
    assert len(cases) == 22
    assert len({c.intent_id for c in cases.values()}) == 18
    covered = {s["tool"] for c in cases.values() for s in c.case.expected_steps}
    assert covered == set(registry.names())
    report = q.run_scoped_benchmark(tuple(cases.values())).to_dict()
    assert report["schema_version"] == 5
    assert report["metrics"]["final_hard_semantic_success"] == dict(
        numerator=22, denominator=22, rate=1
    )
    assert report["metrics"]["hard_errors_by_category"] == {}
    assert report["metrics"]["wording_pair_consistency"]["numerator"] == 4
    for row in report["cases"]:
        assert row["scientific_calls"] == 0
        assert not row["excluded_reason"]
        assert row["assessment"]["root_cause"] == "unresolved"
        assert not row["retry_used"] and not row["repair_invoked"]
        assert row["provider_call_count"] == (1 if row["case_id"] == "C18" else 2)
    for intent in ("C02", "C04", "C07", "C13"):
        assert cases[intent].case.inputs == cases[intent + "-p1"].case.inputs
        assert (
            cases[intent].case.semantic_policy
            == cases[intent + "-p1"].case.semantic_policy
        )


def test_normal_two_calls_are_not_recovery(cases):
    r = q.run_scoped_benchmark((cases["C01"],)).to_dict()["cases"][0]
    assert [c["phase"] for c in r["calls"]] == ["scope_selection", "detailed_planning"]
    assert [c["attempt_kind"] for c in r["calls"]] == ["initial", "initial"]
    assert not r["retry_used"] and not r["repair_invoked"] and not r["failover_used"]
    assert r["initial_outcome"] is True and r["final_preflight_pass"]
    assert r["calls"][0]["scope_fingerprint"] == r["calls"][1]["scope_fingerprint"]


@pytest.mark.parametrize(
    "variation",
    ["ids", "branch_permutation", "control_order", "no_inspection", "default_omission"],
)
def test_valid_variations(cases, registry, variation):
    c = cases["C04"]
    p = q.witness(c, registry)
    steps = p["decision"]["steps"]
    if variation == "ids":
        ids = {s["step_id"]: "renamed" + str(i) for i, s in enumerate(steps)}
        for s in steps:
            s["step_id"] = ids[s["step_id"]]
            s["control_dependencies"] = [ids[d] for d in s["control_dependencies"]]
            for v in s["sources"]:
                if "step" in v["source"]:
                    v["source"]["step"] = ids[v["source"]["step"]]
    elif variation == "branch_permutation":
        p["decision"]["steps"] = [steps[2], steps[3], steps[0], steps[1], steps[4]]
    elif variation == "control_order":
        steps[2]["control_dependencies"] = ["inspect_reference"]
    elif variation == "no_inspection":
        p["decision"]["steps"] = [
            s for s in steps if not s["step_id"].startswith("inspect")
        ]
        for s in p["decision"]["steps"]:
            for v in s["sources"]:
                name = v["source"].get("step")
                if name in ("inspect_reference", "inspect_query"):
                    v["source"] = dict(
                        kind="input",
                        input=(
                            "reference_input_path"
                            if name == "inspect_reference"
                            else "query_input_path"
                        ),
                    )
    else:
        steps[-1]["sources"] = [
            v for v in steps[-1]["sources"] if v["target"] != "overwrite"
        ]
    row = detail(c, p, registry)
    assert row["hard_semantic_success"], row


@pytest.mark.parametrize(
    "mutation",
    [
        "source_port",
        "reference_query",
        "ground_truth",
        "lineage",
        "missing_tool",
        "parameter_loss",
        "invented_value",
        "substitution",
    ],
)
def test_semantic_mutations(cases, registry, mutation):
    cid = (
        "C04"
        if mutation in ("reference_query", "lineage", "parameter_loss")
        else "C06" if mutation == "ground_truth" else "C02"
    )
    c = cases[cid]
    p = q.witness(c, registry)
    steps = p["decision"]["steps"]
    if mutation == "source_port":
        s = next(s for s in steps if s["tool"] == "build_cell_neighbors")
        next(v for v in s["sources"] if v["target"] == "embedding")["source"][
            "source_port"
        ] = "dataset"
    elif mutation == "reference_query":
        for v in steps[-1]["sources"]:
            if v["target"] == "reference_embedding":
                v["source"]["step"] = "embed_query"
            elif v["target"] == "query_embedding":
                v["source"]["step"] = "embed_reference"
    elif mutation == "ground_truth":
        v = next(
            v for v in steps[-1]["sources"] if v["target"] == "ground_truth_dataset"
        )
        v["source"] = dict(kind="step_port", step="transfer", source_port="annotation")
    elif mutation == "lineage":
        s = next(s for s in steps if s["step_id"] == "embed_reference")
        next(v for v in s["sources"] if v["target"] == "dataset")["source"][
            "step"
        ] = "inspect_query"
    elif mutation == "missing_tool":
        p["decision"]["steps"] = steps[:-1]
    elif mutation == "parameter_loss":
        for s in steps:
            if s["tool"] == "epizoo_embed_cells":
                s["sources"] = [v for v in s["sources"] if v["target"] != "checkpoint"]
    elif mutation == "invented_value":
        steps[1]["sources"].append(
            dict(target="device", source=dict(kind="literal", value="PRIVATE-SECRET"))
        )
    elif mutation == "substitution":
        steps[-1]["tool"] = "invented_batch_corrector"
    row = detail(c, p, registry)
    assert row["hard_semantic_success"] is False, row


def test_compiled_parameter_loss_and_wrong_value(cases, registry):
    from agent.orchestration.semantic_wire_v4 import parse_semantic_wire_v4
    from agent.orchestration.semantic_compiler import (
        compile_semantic_plan,
        build_semantic_compiler_contract,
    )

    c = cases["C02"]
    candidate = parse_semantic_wire_v4(
        json.dumps(q.witness(c, registry)), c.request(), registry
    )
    plan = compile_semantic_plan(
        c.request(), candidate, registry, build_semantic_compiler_contract(registry)
    )
    for change in ("lost", "invented"):
        steps = list(plan.steps)
        idx = next(i for i, s in enumerate(steps) if s.tool_name == "cluster_cells")
        args = dict(steps[idx].arguments)
        if change == "lost":
            del args["resolution"]
        else:
            args["resolution"] = 99.5
        steps[idx] = replace(steps[idx], arguments=args)
        failures, _ = q.score_plan(
            c, candidate, replace(plan, steps=tuple(steps)), registry
        )
        assert "binding_mismatch:cluster.resolution" in failures


def test_scope_omission_and_extras(cases, registry):
    c = cases["C14"]
    bad = q.scope_score(c, ["processed_inspection"])
    assert bad["complete"] is False and bad["required_family_recall"] == 0.5
    report = run(
        c,
        selection("processed_inspection"),
        q.witness(c, registry),
        q.witness(c, registry),
    )
    assert report["cases"][0]["final_hard_semantic_success"] is False
    assert "scope_omission" in report["cases"][0]["evaluation_failure_kinds"]
    assert (
        report["metrics"]["conditional_stage_b"]["hard_semantic_success"]["denominator"]
        == 0
    )
    assert (
        q.scope_score(cases["C02"], ["embedding_analysis", "processed_inspection"])[
            "unrelated_family_count"
        ]
        == 0
    )
    assert (
        q.scope_score(cases["C02"], ["embedding_analysis", "raw_preprocessing"])[
            "unrelated_family_count"
        ]
        == 1
    )
    assert q.scope_score(cases["C02"], ["embedding_analysis", "raw_preprocessing"])[
        "complete"
    ]


def test_alternative_family_coverage(cases):
    c = cases["C02"]
    e = dict(
        c.scope,
        required_coverage=["embedding_analysis"],
        alternative_coverage_sets=[["reference_annotation"]],
        unrelated_families=[
            "raw_preprocessing",
            "marker_annotation",
            "exact_matrix_adoption",
            "differential_accessibility",
        ],
    )
    # A synthetic embedding-only coverage expectation tests overlap semantics.
    assert q.scope_score(replace(c, scope=e), ["reference_annotation"])["complete"]


def test_false_unsupported(cases):
    c = cases["C01"]
    r = run(c, dict(selection_schema_version=1, decision=dict(kind="unsupported")))
    assert r["cases"][0]["scope"]["false_unsupported"]
    assert r["cases"][0]["final_hard_semantic_success"] is False
    assert r["metrics"]["false_refusal"]["numerator"] == 1


@pytest.mark.parametrize("phase", ["scope", "detail", "repair"])
def test_rate_limit_is_unobserved(cases, registry, phase):
    c = cases["C02"]
    good = q.witness(c, registry)
    error = PlanningModelError(code="PROVIDER_RATE_LIMITED", retry_after_seconds=0)
    bad = deepcopy(good)
    next(
        v
        for s in bad["decision"]["steps"]
        if s["tool"] == "build_cell_neighbors"
        for v in s["sources"]
        if v["target"] == "embedding"
    )["source"]["source_port"] = "dataset"
    outcomes = (
        [error]
        if phase == "scope"
        else (
            [selected(c), error, error]
            if phase == "detail"
            else [selected(c), bad, error]
        )
    )
    report = run(c, *outcomes)
    r = report["cases"][0]
    assert r["final_hard_semantic_success"] is None
    assert report["metrics"]["rate_limited_sessions"] == 1
    assert report["metrics"]["unobserved_semantic_outcomes"] == 1
    if phase == "scope":
        assert r["scope"]["required_family_recall"] is None
    if phase == "repair":
        assert r["initial_outcome"] is False
        assert r["initial_error"]["code"] == "WRONG_SOURCE_PORT"
        assert r["repair_invoked"] and r["repair_success"] is None
        assert report["metrics"]["semantic_repair_success"]["denominator"] == 0
        assert report["metrics"]["operational_recovery_yield"]["denominator"] == 1


def test_repair_preserves_initial_error_and_scope(cases, registry):
    c = cases["C02"]
    good = q.witness(c, registry)
    bad = deepcopy(good)
    next(
        v
        for s in bad["decision"]["steps"]
        if s["tool"] == "build_cell_neighbors"
        for v in s["sources"]
        if v["target"] == "embedding"
    )["source"]["source_port"] = "dataset"
    report = run(c, selected(c), bad, good)
    r = report["cases"][0]
    assert r["initial_outcome"] is False and r["final_hard_semantic_success"] is True
    assert r["repair_invoked"] and r["repair_success"]
    assert r["provider_call_count"] == 3
    assert r["calls"][1]["code"] == "WRONG_SOURCE_PORT"
    assert r["calls"][1]["reason_code"] == "producer_channel_incompatible"
    assert r["calls"][1]["scope_fingerprint"] == r["calls"][2]["scope_fingerprint"]
    assert report["metrics"]["semantic_repair_success"]["rate"] == 1


def test_oracle_does_not_trigger_repair(cases, registry):
    c = cases["C02"]
    p = q.witness(c, registry)
    p["decision"]["steps"].pop()
    r = run(c, selected(c), p)["cases"][0]
    assert r["final_preflight_pass"] and r["final_hard_semantic_success"] is False
    assert r["provider_call_count"] == 2 and not r["repair_invoked"]


def test_s2_calibration_and_root_rubric():
    s = q.s2_calibration()
    assert s["candidate_summary"] is None
    assert s["assessment"]["root_cause"] == "llm_reasoning"
    assert q.assess_root_cause()["root_cause"] == "unresolved"
    assert (
        q.assess_root_cause(
            case_valid=True, interface_reconstructable=True, interface_defect=True
        )["root_cause"]
        == "agent_interface"
    )
    assert (
        q.assess_root_cause(case_valid=False)["excluded_reason"]
        == "fixture_or_harness_invalid"
    )
    with pytest.raises(ValueError):
        q.assess_root_cause(information_clear="secret")


def test_malformed_and_prose_do_not_leak(cases):
    c = cases["C01"]
    r = run(
        c, selected(c), "PRIVATE-SECRET invalid JSON", "PRIVATE-SECRET invalid JSON"
    )
    text = json.dumps(r)
    assert "PRIVATE-SECRET" not in text and "/synthetic/" not in text
    assert r["cases"][0]["initial_outcome"] is False
    assert r["cases"][0]["calls"][1]["response_bytes"] > 0


def test_stop_queue_after_rate_limit_without_live_calls(cases):
    profile = PlanningModelProfile("test", "test", "test")
    model = q.ReplayModel(
        [PlanningModelError(code="PROVIDER_RATE_LIMITED", retry_after_seconds=0)]
    )
    r = q.run_scoped_benchmark(
        (cases["C01"], cases["C02"]), model=model, model_profile=profile
    ).to_dict()
    assert (
        r["stopped_on_rate_limit"]
        and r["completed_sessions"] == 1
        and r["scheduled_requests"] == 2
    )


def test_cli_scoped_offline(capsys):
    assert main(["--track", "scoped-v4", "--case-id", "C18"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["track"] == "scoped-v4-replay" and result["schema_version"] == 5


def test_expected_missing_source_is_success_not_hard_error(cases):
    c = cases["C15"]
    p = dict(
        schema_version=4,
        decision=dict(
            kind="plan",
            steps=[
                dict(
                    step_id="select",
                    tool="select_scATAC_cells",
                    sources=[],
                    control_dependencies=[],
                )
            ],
        ),
    )
    r = run(c, selected(c), p, p)
    assert r["cases"][0]["initial_outcome"] is True
    assert r["cases"][0]["final_hard_semantic_success"] is True
    assert r["metrics"]["hard_errors_by_category"] == {}


@pytest.mark.parametrize("cid", ["C15", "C16", "C17", "C18"])
def test_negative_case_false_plan_rejected(cases, registry, cid):
    c = cases[cid]
    # A legal but unrelated inspection is never accepted as a refusal.
    inputs = dict(c.case.inputs, input_path="/synthetic/irrelevant.h5ad")
    c = replace(c, case=replace(c.case, inputs=inputs))
    scope = q.PlanningScope(registry, ("processed_inspection",))
    p = dict(
        schema_version=4,
        decision=dict(
            kind="plan",
            steps=[
                dict(
                    step_id="x",
                    tool="inspect_scATAC",
                    sources=[],
                    control_dependencies=[],
                )
            ],
        ),
    )
    assert (
        q.inspect_candidate(c, json.dumps(p), registry, scope)["hard_semantic_success"]
        is False
    )


def test_repair_new_error_is_separate(cases, registry):
    c = cases["C02"]
    p = q.witness(c, registry)
    next(
        v
        for s in p["decision"]["steps"]
        if s["tool"] == "build_cell_neighbors"
        for v in s["sources"]
        if v["target"] == "embedding"
    )["source"]["source_port"] = "dataset"
    r = run(
        c,
        selected(c),
        p,
        {"schema_version": 4, "decision": {"kind": "plan", "steps": []}},
    )["cases"][0]
    assert r["initial_error"]["code"] == "WRONG_SOURCE_PORT"
    assert r["repair_success"] is False
    assert r["repair_eligible"]
    assert r["correction_identity"]["previous_failure_code"] == "WRONG_SOURCE_PORT"
    assert r["calls"][2]["correction_sufficient"] is None


def test_explicit_failover_shares_four_call_budget(cases, registry):
    from agent.providers import PlanningModelFactoryRegistry

    c = cases["C01"]
    primary = PlanningModelProfile("primary", "custom", "primary-model")
    secondary = PlanningModelProfile("secondary", "backup", "secondary-model")
    model = q.ReplayModel([selected(c), "invalid", "invalid"])
    factories = PlanningModelFactoryRegistry(
        {"backup": lambda _: q.ReplayModel([q.witness(c, registry)])}
    )
    report = q.run_scoped_benchmark(
        (c,),
        model=model,
        model_profile=primary,
        recovery_profiles=(secondary,),
        model_factory_registry=factories,
    ).to_dict()
    r = report["cases"][0]
    assert r["provider_call_count"] == 4 and r["failover_used"]
    assert r["final_hard_semantic_success"] is True
    assert r["repair_success"] is False
    assert [x["attempt_kind"] for x in r["calls"]] == [
        "initial",
        "initial",
        "repair",
        "failover",
    ]
    assert [x["profile_id"] for x in r["calls"]] == [
        "primary",
        "primary",
        "primary",
        "secondary",
    ]
    assert len({x["scope_fingerprint"] for x in r["calls"]}) == 1


def test_fixture_invalid_excluded_before_provider(cases):
    from benchmarks.planner.benchmark import BenchmarkDefinitionError

    c = cases["C01"]
    bad = replace(c, case=replace(c.case, inputs={}))

    class Never:
        model_id = "never"

        def complete(self, **kwargs):
            pytest.fail("Invalid fixture reached provider")

    with pytest.raises(BenchmarkDefinitionError):
        q.run_scoped_benchmark(
            (c, bad),
            model=Never(),
            model_profile=PlanningModelProfile("test", "test", "test"),
        )


def test_writing_variants_both_wrong_is_not_consistency(cases, registry):
    pair = (cases["C02"], cases["C02-p1"])
    overrides = {
        c.case.case_id: [
            selected(c),
            dict(schema_version=4, decision=dict(kind="unsupported", reason="No")),
        ]
        for c in pair
    }
    r = q.run_scoped_benchmark(pair, replay_outcomes=overrides).to_dict()
    assert r["metrics"]["wording_pair_consistency"] == dict(
        numerator=0, denominator=1, rate=0
    )


def test_all_comparison_identities_and_no_values(cases):
    r = q.run_scoped_benchmark((cases["C07"],)).to_dict()
    manifest = r["comparison_manifest"]
    for key in (
        "dependency_fingerprint",
        "corpus_fingerprint",
        "registry_fingerprint",
        "capability_fingerprint",
        "production_source_fingerprint",
        "evaluator_source_fingerprint",
    ):
        assert len(manifest[key]) == 64
    assert manifest["sampling_controls"]["seed"] == "unspecified"
    assert manifest["deterministic_live_generation"] is False
    text = json.dumps(r)
    assert "/synthetic/" not in text
    assert "T_cell" not in json.dumps(r["cases"][0]["calls"])


def test_failover_failure_cannot_call_fifth_time(cases):
    from agent.providers import PlanningModelFactoryRegistry

    c = cases["C01"]
    primary = PlanningModelProfile("primary", "custom", "primary-model")
    secondary = PlanningModelProfile("secondary", "backup", "secondary-model")
    factories = PlanningModelFactoryRegistry(
        {"backup": lambda _: q.ReplayModel(["invalid"])}
    )
    r = q.run_scoped_benchmark(
        (c,),
        model=q.ReplayModel([selected(c), "invalid", "invalid"]),
        model_profile=primary,
        recovery_profiles=(secondary,),
        model_factory_registry=factories,
    ).to_dict()["cases"][0]
    assert r["provider_call_count"] == 4 and r["final_hard_semantic_success"] is False
    assert r["calls"][-1]["attempt_kind"] == "failover"


@pytest.mark.parametrize("entry", ["_run_execute", "_run_durable_execute"])
def test_execution_entry_attempt_cannot_be_reported_as_success(
    cases, monkeypatch, entry
):
    original = q.AgentRuntime.run

    def check(self, request):
        with pytest.raises(AssertionError):
            getattr(self, entry)()
        return original(self, request)

    monkeypatch.setattr(q.AgentRuntime, "run", check)
    with pytest.raises(AssertionError, match="attempted scientific execution"):
        q.run_scoped_benchmark((cases["C01"],))


def test_stage_a_malformed_is_failure_not_transport(cases):
    report = run(cases["C01"], "not json")
    row = report["cases"][0]
    assert row["final_hard_semantic_success"] is False
    assert row["evaluation_failure_kinds"] == ["scope_selection"]
    assert row["scope"]["required_family_recall"] is None
    assert row["calls"][0]["response_bytes"] == len("not json")
    assert report["metrics"]["transport_failures"] == 0


def test_expected_unsupported_scope_false_selection(cases):
    c = cases["C18"]
    report = run(
        c,
        selection("processed_inspection"),
        dict(
            schema_version=4,
            decision=dict(kind="unsupported", reason="RNA unsupported"),
        ),
    )
    assert report["cases"][0]["final_hard_semantic_success"] is False
    assert report["metrics"]["hard_errors_by_category"] == {"unsupported_handling": 1}


def test_unchanged_error_is_not_a_successful_repair(cases, registry):
    c = cases["C02"]
    p = q.witness(c, registry)
    next(
        v
        for s in p["decision"]["steps"]
        if s["tool"] == "build_cell_neighbors"
        for v in s["sources"]
        if v["target"] == "embedding"
    )["source"]["source_port"] = "dataset"
    row = run(c, selected(c), p, p)["cases"][0]
    assert row["original_error_corrected"] is False
    assert row["repair_new_error"] is False
    assert row["repair_success"] is False


def test_s2_stage_reuses_production_diagnostic_mapping():
    from agent.orchestration.llm_planner import _SEMANTIC_COMPILER_STAGES

    assert (
        q.s2_calibration()["diagnostic_stage"]
        == _SEMANTIC_COMPILER_STAGES["WRONG_SOURCE_PORT"].value
    )


def test_scoped_live_cli_defaults_to_only_the_smoke_without_calling_provider(
    monkeypatch, capsys
):
    import benchmarks.planner.run_benchmark as cli

    captured = {}

    def fake_run(cases, **kwargs):
        captured.update(kwargs)
        return q.ScopedReport({"schema_version": 5})

    monkeypatch.setattr(cli, "_live_model", lambda *args: ("profile", "model"))
    monkeypatch.setattr(q, "run_scoped_benchmark", fake_run)
    assert (
        cli.main(
            [
                "--track",
                "scoped-v4",
                "--live",
                "--provider",
                "groq",
                "--model",
                "fixture",
            ]
        )
        == 0
    )
    assert captured["selected_case_ids"] == frozenset(q.LIVE_SMOKE)
    assert captured["repetitions"] == 1
    capsys.readouterr()
