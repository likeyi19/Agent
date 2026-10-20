"""Offline calibration of the measurement instrument, never live model quality."""

from dataclasses import replace
import json

import pytest

from agent.orchestration import build_default_tool_registry
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


def test_s2_stage_reuses_production_diagnostic_mapping():
    from agent.orchestration.llm_planner import _SEMANTIC_COMPILER_STAGES

    assert (
        q.s2_calibration()["diagnostic_stage"]
        == _SEMANTIC_COMPILER_STAGES["WRONG_SOURCE_PORT"].value
    )


def test_preserved_corpus_witnesses_remain_readable_without_live_execution(cases,registry):
    for case in cases.values():
        if case.case.expected_outcome == 'plan':
            row = detail(case,q.witness(case,registry),registry)
            assert row['hard_semantic_success'] is True


def test_retired_two_stage_runner_is_explicit_and_creates_no_model(cases,monkeypatch):
    with pytest.raises(RuntimeError,match='retired'):
        q.run_scoped_benchmark(tuple(cases.values()))
    with pytest.raises(ValueError,match='retired'):
        main(['--track','scoped-v4','--live'])
