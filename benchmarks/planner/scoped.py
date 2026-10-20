"""Read-only scoring and corpus compatibility for the retired scoped-v4 experiment.

Nothing here routes production requests or supplies recovery feedback. Provider
responses are inspected transiently; only allowlisted structure leaves the observer.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

from agent.orchestration import AgentRequest, PlanExecutor, RunMode, StepOutputRef, build_default_tool_registry
from agent.orchestration.planner import PlannerError
from agent.orchestration.planning_scope import PlanningScope, capability_index, fingerprint
from agent.orchestration.semantic_compiler import (
    SemanticPlanCompileError,
    SemanticRequestInputSource,
    build_semantic_compiler_contract,
    compile_semantic_plan,
)
from agent.orchestration.semantic_wire_v4 import parse_semantic_wire_v4
from benchmarks.planner.benchmark import BenchmarkCase, BenchmarkDefinitionError, _load_json, _strict_keys, _match_semantic_roles, _normalized_plan

CORPUS_VERSION = 2
REPORT_VERSION = 5
SCORER_VERSION = "scoped-quality-v1"
DEFAULT_CASES = Path(__file__).with_name("scoped_cases.json")
LIVE_SMOKE = ("C02", "C04", "C13", "C14", "C15", "C18")
FAILURE_KINDS = frozenset(
    (
        "provider_transport",
        "scope_selection",
        "scope_omission",
        "scope_overselection",
        "wire_parse",
        "schema",
        "tool_selection",
        "argument_binding",
        "dependency_reference",
        "semantic_source",
        "scientific_intent",
        "missing_information_handling",
        "unsupported_handling",
        "compiler",
        "preflight",
        "repair_failure",
    )
)
ROOT_CAUSES = frozenset(("agent_interface", "llm_reasoning", "unresolved"))


@dataclass(frozen=True)
class ScopedCase:
    case: BenchmarkCase
    intent_id: str
    scientific_intent: str
    wording_variant: str
    input_profile_id: str
    scope: Mapping
    expected_rejection_targets: tuple[str, ...] = ()

    def request(self, repetition=1):
        return AgentRequest(
            f"scoped-{self.case.case_id}-{repetition}",
            self.case.prompt,
            self.case.inputs,
            RunMode.PLAN_ONLY,
        )


def load_scoped_cases(path=DEFAULT_CASES):
    payload = _load_json(path)
    _strict_keys(payload, {"schema_version", "cases"}, "Scoped corpus")
    if payload["schema_version"] != CORPUS_VERSION or not isinstance(
        payload["cases"], list
    ):
        raise BenchmarkDefinitionError("Invalid scoped corpus version/cases.")
    families = set(capability_index(build_default_tool_registry()))
    result = []
    for i, row in enumerate(payload["cases"]):
        _strict_keys(
            row,
            {
                "case",
                "intent_id",
                "scientific_intent",
                "wording_variant",
                "input_profile_id",
                "scope",
                "expected_rejection_targets",
            },
            "Scoped case",
        )
        for field in (
            "intent_id",
            "scientific_intent",
            "wording_variant",
            "input_profile_id",
        ):
            if not isinstance(row[field], str) or not row[field].strip():
                raise BenchmarkDefinitionError("Invalid scoped case identity.")
        case = BenchmarkCase.from_mapping(row["case"], i)
        rejection_targets = row["expected_rejection_targets"]
        if not isinstance(rejection_targets, list) or any(
            not isinstance(v, str) for v in rejection_targets
        ):
            raise BenchmarkDefinitionError("Invalid expected rejection targets.")
        scope = row["scope"]
        _strict_keys(
            scope,
            {
                "required_coverage",
                "alternative_coverage_sets",
                "plausible_optional_families",
                "unrelated_families",
                "expect_unsupported",
            },
            "Scope expectation",
        )
        if type(scope["expect_unsupported"]) is not bool:
            raise BenchmarkDefinitionError("Invalid scope unsupported expectation.")
        if not isinstance(scope["alternative_coverage_sets"], list):
            raise BenchmarkDefinitionError("Invalid alternative coverage sets.")
        groups = [
            scope[k]
            for k in (
                "required_coverage",
                "plausible_optional_families",
                "unrelated_families",
            )
        ] + scope["alternative_coverage_sets"]
        for group in groups:
            if (
                not isinstance(group, list)
                or any(not isinstance(v, str) for v in group)
                or len(set(group)) != len(group)
                or not set(group) <= families
            ):
                raise BenchmarkDefinitionError("Invalid capability coverage.")
        acceptable = set(scope["required_coverage"]) | set(
            scope["plausible_optional_families"]
        )
        for group in scope["alternative_coverage_sets"]:
            if not group:
                raise BenchmarkDefinitionError("Empty alternative coverage.")
            acceptable.update(group)
        unrelated = set(scope["unrelated_families"])
        if acceptable & unrelated or acceptable | unrelated != families:
            raise BenchmarkDefinitionError(
                "Scope families must be explicitly classified."
            )
        if scope["expect_unsupported"] != (not scope["required_coverage"]):
            raise BenchmarkDefinitionError("Unsupported scope cannot require coverage.")
        if case.semantic_policy is not None:
            required_tools = {
                s["tool"]
                for s in case.expected_steps
                if s["role"] in case.semantic_policy.required_roles
            }
            index = capability_index(build_default_tool_registry())
            for coverage in [scope["required_coverage"]] + scope[
                "alternative_coverage_sets"
            ]:
                if not required_tools <= {n for f in coverage for n in index[f]}:
                    raise BenchmarkDefinitionError(
                        "Required scope does not expose required tools."
                    )
        result.append(
            ScopedCase(
                case,
                **{
                    k: row[k]
                    for k in row
                    if k not in {"case", "expected_rejection_targets"}
                },
                expected_rejection_targets=tuple(rejection_targets),
            )
        )
    if len({c.case.case_id for c in result}) != len(result):
        raise BenchmarkDefinitionError("Duplicate scoped case ID.")
    for intent in {c.intent_id for c in result}:
        pairs = [c for c in result if c.intent_id == intent]
        if len({fingerprint(c.case.inputs) for c in pairs}) != 1:
            raise BenchmarkDefinitionError(
                "Wording variants must have identical inputs."
            )
        if len({c.wording_variant for c in pairs}) != len(pairs):
            raise BenchmarkDefinitionError("Duplicate wording variant.")
    return tuple(result)


def scope_score(case, selected=None, *, unsupported=False, observed=True):
    e = case.scope
    base = dict(
        observed=observed,
        selected_capability_families=selected,
        required_matched=None,
        required_total=None,
        required_family_recall=None,
        complete=None,
        multi_family=min(
            map(len, [e["required_coverage"]] + e["alternative_coverage_sets"])
        )
        > 1,
        unrelated_family_count=None,
        unsupported_correct=None,
        false_unsupported=False,
    )
    if not observed:
        return base
    base["unsupported_correct"] = unsupported == e["expect_unsupported"]
    base["false_unsupported"] = unsupported and not e["expect_unsupported"]
    if e["expect_unsupported"]:
        base["complete"] = unsupported
    else:
        selected_set = set(selected or ())
        alternatives = [set(e["required_coverage"])] + [
            set(g) for g in e["alternative_coverage_sets"]
        ]
        best = max(
            alternatives, key=lambda g: (len(g & selected_set) / len(g), -len(g))
        )
        base.update(
            required_matched=len(best & selected_set),
            required_total=len(best),
            required_family_recall=len(best & selected_set) / len(best),
            complete=any(g <= selected_set for g in alternatives),
        )
    base["unrelated_family_count"] = len(
        set(selected or ()) & set(e["unrelated_families"])
    )
    return base


def assess_root_cause(
    *,
    candidate_returned=False,
    case_valid=None,
    interface_reconstructable=None,
    correct_candidate_expressible=None,
    information_clear=None,
    interface_defect=None,
):
    """Explicit adjudication evidence; absence never implies model fault."""
    evidence = dict(
        candidate_returned=candidate_returned,
        case_valid=case_valid,
        interface_reconstructable=interface_reconstructable,
        correct_candidate_expressible=correct_candidate_expressible,
        information_clear=information_clear,
        interface_defect=interface_defect,
    )
    if any(
        value is not None and type(value) is not bool for value in evidence.values()
    ):
        raise ValueError("Root-cause evidence must be boolean or unassessed.")
    root = "unresolved"
    excluded = case_valid is False
    if case_valid is True and interface_reconstructable is True:
        if interface_defect is True:
            root = "agent_interface"
        elif (
            candidate_returned
            and correct_candidate_expressible is True
            and information_clear is True
            and interface_defect is False
        ):
            root = "llm_reasoning"
    return dict(
        root_cause=root,
        excluded_reason="fixture_or_harness_invalid" if excluded else None,
        evidence=evidence,
    )


def s2_calibration():
    """Only facts retained by accepted S2 audit; no reconstructed full DAG."""
    return dict(
        selected_capability_families=["processed_inspection", "embedding_analysis"],
        phase="detailed_planning",
        diagnostic_stage="dependency_reference",
        evaluation_failure_kind="semantic_source",
        code="WRONG_SOURCE_PORT",
        reason_code="producer_channel_incompatible",
        rejected_edge=dict(
            producer="embed",
            source_port="dataset",
            consumer="neighbors",
            target="embedding",
        ),
        candidate_summary=None,
        preflight=None,
        repair_semantic_success=None,
        repair_provider_code="PROVIDER_RATE_LIMITED",
        assessment=assess_root_cause(
            candidate_returned=True,
            case_valid=True,
            interface_reconstructable=True,
            correct_candidate_expressible=True,
            information_clear=True,
            interface_defect=False,
        ),
    )


def witness(case, registry):
    """One calibration witness projected from existing role/binding expectations.

    This is not the scoring oracle: alternatives remain in SemanticPolicy.
    """
    c = case.case
    if c.expected_outcome != "plan":
        return dict(
            schema_version=4,
            decision=dict(
                kind="unsupported",
                reason="Required information or capability unavailable.",
            ),
        )
    contract = build_semantic_compiler_contract(registry)
    tools = {s["role"]: s["tool"] for s in c.expected_steps}
    steps = []
    for s in c.expected_steps:
        bindings = s["bindings"]
        covered = set()
        sources = []
        for channel in contract.step_output_channels:
            if channel.consumer_tool_name != s["tool"]:
                continue
            bs = [bindings.get(m.argument_name) for m in channel.members]
            if not bs or any(
                not b or b["kind"] != "ref" or b["output_key"] != m.output_key
                for b, m in zip(bs, channel.members)
            ):
                continue
            producers = {b["producer_role"] for b in bs}
            if len(producers) != 1:
                continue
            producer = next(iter(producers))
            if tools[producer] != channel.producer_tool_name:
                continue
            sources.append(
                dict(
                    target=channel.target_port,
                    source=dict(
                        kind="step_port", step=producer, source_port=channel.source_port
                    ),
                )
            )
            covered.update(m.argument_name for m in channel.members)
        groups = {}
        for rule in contract.request_bindings:
            if rule.tool_name == s["tool"]:
                groups.setdefault((rule.target_port, rule.selector), []).append(rule)
        for (target, selector), rules in groups.items():
            if all(
                bindings.get(r.argument_name)
                == {"kind": "input", "input_name": r.input_name}
                for r in rules
            ):
                sources.append(
                    dict(target=target, source=dict(kind="input", input=selector))
                )
                covered.update(r.argument_name for r in rules)
        if covered != set(bindings):
            raise BenchmarkDefinitionError(
                f'Unexpressible witness: {c.case_id}/{s["role"]}'
            )
        producers = {
            v["source"]["step"] for v in sources if v["source"]["kind"] == "step_port"
        }
        steps.append(
            dict(
                step_id=s["role"],
                tool=s["tool"],
                sources=sources,
                control_dependencies=[d for d in s["depends_on"] if d not in producers],
            )
        )
    return dict(schema_version=4, decision=dict(kind="plan", steps=steps))


def _candidate_summary(candidate, registry, request):
    ids = {s.step_id: f"s{i}" for i, s in enumerate(candidate.steps)}
    known_ports = {
        p.name
        for n in registry.names()
        for p in registry.get(n).semantic_planning.producer_ports
    }
    rows = []
    for s in candidate.steps:
        targets = {
            p.name for p in registry.get(s.tool_name).semantic_planning.consumer_ports
        }
        sources = []
        for v in s.sources:
            row = {"target": v.target_port if v.target_port in targets else None}
            if isinstance(v, SemanticRequestInputSource):
                row.update(
                    kind="input",
                    input=v.input_name if v.input_name in request.inputs else None,
                )
            else:
                row.update(
                    kind="step",
                    producer=ids.get(v.step_id),
                    source_port=v.source_port if v.source_port in known_ports else None,
                )
            sources.append(row)
        rows.append(
            dict(
                step=ids[s.step_id],
                tool=s.tool_name,
                sources=sources,
                control_dependencies=[ids.get(d) for d in s.control_dependencies],
            )
        )
    return rows


def _plain(value):
    if isinstance(value, Mapping):
        return {
            json.dumps(k) if isinstance(k, tuple) else k: _plain(v)
            for k, v in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return value


def score_plan(case, candidate, plan, registry):
    """Recover authorized binding origins, then reuse the historical role matcher.

    Explicit selectors disambiguate equal values. Implicit origins must be unique;
    values themselves are never serialized. Compiler output remains authoritative.
    """
    contract = build_semantic_compiler_contract(registry)
    candidates = {s.step_id: s for s in candidate.steps}
    raw = []
    for s in plan.steps:
        explicit = {
            v.target_port: v.input_name
            for v in candidates[s.step_id].sources
            if isinstance(v, SemanticRequestInputSource)
        }
        arguments = {}
        for name, value in s.arguments.items():
            if isinstance(value, StepOutputRef):
                arguments[name] = dict(
                    binding_type="ref",
                    ref_step_id=value.step_id,
                    ref_output_key=value.output_key,
                )
                continue
            rules = [
                r
                for r in contract.request_bindings
                if r.tool_name == s.tool_name
                and r.argument_name == name
                and r.input_name in case.case.inputs
                and fingerprint(_plain(case.case.inputs[r.input_name]))
                == fingerprint(_plain(value))
                and (
                    r.target_port not in explicit
                    or r.selector == explicit[r.target_port]
                )
            ]
            origins = {r.input_name for r in rules}
            arguments[name] = (
                dict(binding_type="input", input_name=next(iter(origins)))
                if len(origins) == 1
                else dict(binding_type="unresolved")
            )
        raw.append(
            dict(
                step_id=s.step_id,
                tool_name=s.tool_name,
                arguments=arguments,
                depends_on=list(s.depends_on),
            )
        )
    indices = {s["step_id"]: i for i, s in enumerate(raw)}
    allowed = Counter(s["tool"] for s in case.case.expected_steps)
    emitted = Counter(s["tool_name"] for s in raw)
    if any(count > allowed[tool] for tool, count in emitted.items()):
        # Bound the legacy combinatorial matcher before considering role assignments.
        return ["unexpected_tool_cardinality"], list(
            _normalized_plan(
                raw, actual_id_to_index=indices, role_by_index=[None] * len(raw)
            )
        )
    dataflow = [
        {
            **s,
            "depends_on": sorted(
                {
                    b["ref_step_id"]
                    for b in s["arguments"].values()
                    if b.get("binding_type") == "ref"
                }
            ),
        }
        for s in raw
    ]
    roles, failures = _match_semantic_roles(
        case.case, dataflow, actual_id_to_index=indices
    )
    return list(failures), list(
        _normalized_plan(raw, actual_id_to_index=indices, role_by_index=roles)
    )


def _failure_kind(code, stage):
    if code.startswith("PROVIDER_") or code == "PLANNING_PROVIDER_ERROR":
        return "provider_transport"
    if code == "INVALID_PLANNING_SCOPE":
        return "scope_selection"
    if code in {
        "WRONG_SOURCE_PORT",
        "UNKNOWN_SOURCE_PORT",
        "AMBIGUOUS_SOURCE_PORT",
        "BROKEN_BRANCH_LINEAGE",
        "CONFLICTING_BRANCH_LINEAGE",
        "WRONG_SOURCE_LINEAGE",
    }:
        return "semantic_source"
    return {
        "parse": "wire_parse",
        "schema": "schema",
        "tool_selection": "tool_selection",
        "argument_binding": "argument_binding",
        "dependency_reference": "dependency_reference",
        "preflight": "preflight",
        "compiler": "compiler",
        "unsupported": "unsupported_handling",
    }.get(stage, "scientific_intent")


def inspect_candidate(case, response, registry, scope):
    row = dict(
        response_bytes=len(response.encode()) if isinstance(response, str) else None,
        candidate_returned=False,
        parse_schema_pass=None,
        compiler_pass=None,
        preflight_pass=None,
        hard_semantic_success=None,
        candidate_summary=None,
        compiled_plan_summary=None,
        failures=[],
        code=None,
        reason_code=None,
        diagnostic_stage=None,
        evaluation_failure_kind=None,
    )
    try:
        candidate = parse_semantic_wire_v4(
            response,
            case.request(),
            registry,
            visible_tool_names=scope.visible_tool_names,
        )
    except PlannerError as exc:
        code = exc.code
        stage = exc.diagnostic_stage.value if exc.diagnostic_stage else "schema"
        unsupported = code == "UNSUPPORTED_REQUEST"
        row.update(
            parse_schema_pass=unsupported,
            hard_semantic_success=(
                unsupported and case.case.expected_outcome != "plan"
            ),
            code=code,
            diagnostic_stage="unsupported" if unsupported else stage,
            evaluation_failure_kind=(
                None
                if unsupported and case.case.expected_outcome != "plan"
                else _failure_kind(code, stage)
            ),
        )
        return row
    row.update(
        candidate_returned=True,
        parse_schema_pass=True,
        candidate_summary=_candidate_summary(candidate, registry, case.request()),
    )
    try:
        plan = compile_semantic_plan(
            case.request(),
            candidate,
            registry,
            build_semantic_compiler_contract(registry),
        )
    except SemanticPlanCompileError as exc:
        row.update(
            compiler_pass=False,
            code=exc.code,
            reason_code=exc.diagnostic_fields.get("reason_code"),
            diagnostic_stage="compiler",
            hard_semantic_success=(
                exc.code in case.case.expected_error_codes
                and exc.diagnostic_fields.get("target_port")
                in case.expected_rejection_targets
            ),
            evaluation_failure_kind=_failure_kind(exc.code, "compiler"),
        )
        if row["hard_semantic_success"]:
            row["evaluation_failure_kind"] = None
        return row
    row["compiler_pass"] = True
    row["preflight_pass"] = PlanExecutor(registry).preflight(plan).passed
    failures, summary = score_plan(case, candidate, plan, registry)
    row["compiled_plan_summary"] = summary
    if case.case.expected_outcome != "plan":
        failures.append("false_executable_plan")
    if not row["preflight_pass"]:
        failures.append("preflight_rejected")
    row.update(
        failures=failures,
        hard_semantic_success=not failures,
        evaluation_failure_kind=(
            (
                "preflight"
                if not row["preflight_pass"]
                else (
                    "missing_information_handling"
                    if case.case.expected_outcome == "failure"
                    else (
                        "unsupported_handling"
                        if case.case.expected_outcome == "unsupported"
                        else "scientific_intent"
                    )
                )
            )
            if failures
            else None
        ),
    )
    if case.case.expected_outcome == "plan" and any(
        f.startswith(
            ("missing_required_role", "unexpected_tool_cardinality", "unexpected_step")
        )
        for f in failures
    ):
        row["evaluation_failure_kind"] = "tool_selection"
    return row


def run_scoped_benchmark(*args, **kwargs):
    raise RuntimeError("Historical two-stage scoped-v4 experiment retired by R5. "
                       "Read preserved reports or use the current direct planning harness.")
