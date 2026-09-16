"""Evaluation-only scoped-v4 extension of the historical Planner benchmark.

Nothing here routes production requests or supplies recovery feedback. Provider
responses are inspected transiently; only allowlisted structure leaves the observer.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
from typing import Mapping
from unittest.mock import patch

from agent.orchestration import (
    AgentRequest,
    AgentRuntime,
    LLMPlanner,
    PlanExecutor,
    PlanningWireMode,
    PlanningModelProfile,
    RunMode,
    StepOutputRef,
    build_default_tool_registry,
)
from agent.orchestration.planner import PlannerError
from agent.orchestration.planning_recovery import _REPAIRABLE_STAGES
from agent.orchestration.planning_scope import (
    CAPABILITY_DESCRIPTIONS,
    PlanningScope,
    capability_index,
    fingerprint,
    parse_selection,
)
from agent.orchestration.semantic_compiler import (
    SemanticPlanCompileError,
    SemanticRequestInputSource,
    build_semantic_compiler_contract,
    compile_semantic_plan,
)
from agent.orchestration.semantic_wire_v4 import parse_semantic_wire_v4
from benchmarks.planner.benchmark import (
    BenchmarkCase,
    BenchmarkDefinitionError,
    _load_json,
    _strict_keys,
    _match_semantic_roles,
    _normalized_plan,
    guarded_registry,
)

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


class ReplayModel:
    model_id = "scoped-offline"

    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)

    def complete(self, **_):
        value = next(self.outcomes)
        if isinstance(value, Exception):
            raise value
        return value if isinstance(value, str) else json.dumps(value)


class Observer:
    def __init__(self, model, case, registry):
        self.model = model
        self.model_id = getattr(model, "model_id", "observed")
        self.case = case
        self.registry = registry
        self.calls = []
        self.scope = None

    def complete(self, *, prompt, response_schema):
        selection = "selection_schema_version" in response_schema.get("properties", {})
        row = dict(
            phase="scope_selection" if selection else "detailed_planning",
            provider_call_index=len(self.calls) + 1,
            prompt_fingerprint=fingerprint(prompt),
            schema_fingerprint=fingerprint(response_schema),
            provider_outcome="returned",
            scope_fingerprint=(
                None if self.scope is None else self.scope.scope_fingerprint
            ),
            visible_tool_count=(
                0 if self.scope is None else len(self.scope.visible_tool_names)
            ),
        )
        context = json.loads(prompt)
        correction = context.get("repair", context.get("failover", {})).get(
            "diagnostic"
        )
        if correction is not None:
            # The prompt is Agent-generated; retain semantic identifiers only.
            row["correction_context"] = {
                k: correction[k]
                for k in (
                    "previous_failure_stage",
                    "previous_failure_code",
                    "reason_code",
                    "tool_name",
                    "target_port",
                    "source_port",
                    "input_name",
                    "candidate_constructed",
                    "candidate_preflight_passed",
                )
                if k in correction
            }
            row["correction_fingerprint"] = fingerprint(correction)
            row["correction_sufficient"] = None  # Requires explicit interface audit.
            known_ports = {
                p.name
                for n in self.registry.names()
                for p in (
                    *self.registry.get(n).semantic_planning.consumer_ports,
                    *self.registry.get(n).semantic_planning.producer_ports,
                )
            }
            for key, allowed in (
                ("target_port", known_ports),
                ("source_port", known_ports),
                ("input_name", self.case.case.inputs),
                ("tool_name", self.registry.names()),
            ):
                if (
                    key in row["correction_context"]
                    and row["correction_context"][key] not in allowed
                ):
                    row["correction_context"][key] = None
        self.calls.append(row)
        try:
            response = self.model.complete(
                prompt=prompt, response_schema=response_schema
            )
        except Exception:
            row["provider_outcome"] = "failed"
            raise
        row["response_bytes"] = (
            len(response.encode()) if isinstance(response, str) else None
        )
        # Evaluation faults cannot change the candidate or production recovery.
        try:
            if selection:
                try:
                    self.scope = parse_selection(response, self.registry)
                    row.update(
                        scope=scope_score(self.case, list(self.scope.capability_ids)),
                        scope_fingerprint=self.scope.scope_fingerprint,
                        visible_tool_count=len(self.scope.visible_tool_names),
                    )
                except PlannerError as exc:
                    row.update(
                        code=exc.code,
                        scope=(
                            scope_score(self.case, unsupported=True)
                            if exc.code == "UNSUPPORTED_REQUEST"
                            else scope_score(self.case, observed=False)
                        ),
                    )
            else:
                row.update(
                    inspect_candidate(self.case, response, self.registry, self.scope)
                )
        except Exception:
            row["excluded_reason"] = "fixture_or_harness_invalid"
        return response


class ObservedFactories:
    """Observe explicitly configured failover within the same session ledger."""

    def __init__(self, delegate, primary):
        self.delegate = delegate
        self.primary = primary

    @property
    def provider_ids(self):
        return self.delegate.provider_ids

    def create(self, profile):
        secondary = Observer(
            self.delegate.create(profile), self.primary.case, self.primary.registry
        )
        secondary.calls = self.primary.calls
        secondary.scope = self.primary.scope
        return secondary


def _ratio(values):
    observed = [v for v in values if v is not None]
    return dict(
        numerator=sum(bool(v) for v in observed),
        denominator=len(observed),
        rate=sum(bool(v) for v in observed) / len(observed) if observed else None,
    )


def _diagnostics(result):
    return [
        dict(e.details)
        for e in result.trace
        if e.details.get("diagnostic_schema_version") == 5
    ]


def _finish(case, observer, result, registry, guard, repetition, assessment=None):
    diagnostics = _diagnostics(result)
    for row in observer.calls:
        ds = [
            d
            for d in diagnostics
            if d.get("provider_call_index") == row["provider_call_index"]
            and d.get("phase") == row["phase"]
        ]
        start = next((d for d in ds if d["code"] == "PROVIDER_CALL_STARTED"), {})
        row["attempt_kind"] = start.get("attempt_kind", "initial")
        row["profile_id"] = start.get("profile_id")
        row["provider_id"] = start.get("provider_id")
        # Only stable diagnostic enums/codes are retained, never provider/step prose.
        row["diagnostics"] = [
            {k: d.get(k) for k in ("stage", "code", "reason_code", "outcome")}
            for d in ds
            if d.get("code") != "PLANNING_RECOVERY_SUMMARY"
        ]
        failed = [d for d in ds if d.get("outcome") in ("failed", "rejected")]
        if failed:
            d = failed[0]
            row.update(
                code=d["code"],
                reason_code=d.get("reason_code"),
                diagnostic_stage=d["stage"],
            )
            expected_scope_refusal = (
                row["phase"] == "scope_selection"
                and d["code"] == "UNSUPPORTED_REQUEST"
                and case.scope["expect_unsupported"]
            )
            if (
                row.get("hard_semantic_success") is not True
                and not expected_scope_refusal
            ):
                row["evaluation_failure_kind"] = _failure_kind(d["code"], d["stage"])
        if row["provider_outcome"] == "failed":
            row["hard_semantic_success"] = None
    scope = (
        observer.calls[0].get("scope", scope_score(case, observed=False))
        if observer.calls
        else scope_score(case, observed=False)
    )
    details = [r for r in observer.calls if r["phase"] == "detailed_planning"]
    initial = details[0] if details else None
    final = details[-1] if details else None
    repaired = [r for r in details if r["attempt_kind"] == "repair"]
    recovery = next(
        (
            d
            for d in reversed(diagnostics)
            if d.get("code") == "PLANNING_RECOVERY_SUMMARY"
        ),
        {},
    )
    semantic = (
        final.get("hard_semantic_success")
        if final
        else (
            scope["unsupported_correct"]
            if scope["observed"]
            and observer.calls[0].get("code") == "UNSUPPORTED_REQUEST"
            else None
        )
    )
    if scope["complete"] is False:
        semantic = False
    if observer.calls and observer.calls[0].get("code") == "INVALID_PLANNING_SCOPE":
        semantic = False
    kinds = []
    if scope["complete"] is False and not case.scope["expect_unsupported"]:
        kinds.append("scope_omission")
    if scope["complete"] is False and case.scope["expect_unsupported"]:
        kinds.append("unsupported_handling")
    if scope["unrelated_family_count"]:
        kinds.append("scope_overselection")
    kinds.extend(
        r["evaluation_failure_kind"]
        for r in observer.calls
        if r.get("evaluation_failure_kind")
    )
    if repaired and repaired[0].get("hard_semantic_success") is False:
        kinds.append("repair_failure")
    evidence = assessment or assess_root_cause(
        candidate_returned=any(r.get("candidate_returned", False) for r in details),
        case_valid=True,
        interface_reconstructable=True,
        correct_candidate_expressible=(
            scope["complete"] if case.case.expected_outcome == "plan" else None
        ),
    )
    repair = repaired[0] if repaired else None
    repair_returned = bool(repair and repair.get("candidate_returned"))
    repair_evaluable = bool(
        repair
        and repair.get("parse_schema_pass") is True
        and repair.get("hard_semantic_success") is not None
    )
    exclusions = [
        r.get("excluded_reason") for r in observer.calls if r.get("excluded_reason")
    ]
    return dict(
        case_id=case.case.case_id,
        intent_id=case.intent_id,
        scientific_intent=case.scientific_intent,
        wording_variant=case.wording_variant,
        input_profile_id=case.input_profile_id,
        repetition=repetition,
        expected_outcome=case.case.expected_outcome,
        expected_scope_unsupported=case.scope["expect_unsupported"],
        request_fingerprint=fingerprint(
            dict(
                prompt=case.case.prompt,
                inputs=case.case.inputs,
                mode="PLAN_ONLY",
                request_id=case.request(repetition).request_id,
            )
        ),
        scope=scope,
        calls=observer.calls,
        initial_outcome=(
            None if initial is None else initial.get("hard_semantic_success")
        ),
        final_hard_semantic_success=semantic,
        final_compiler_pass=None if final is None else final.get("compiler_pass"),
        final_preflight_pass=None if final is None else final.get("preflight_pass"),
        operational_plan_accepted=bool(
            result.plan and result.verification and result.verification.passed
        ),
        stage_b_quality_eligible=scope["complete"] is True,
        initial_error=(
            None
            if initial is None
            else {
                k: initial.get(k) for k in ("code", "reason_code", "diagnostic_stage")
            }
        ),
        repair_eligible=bool(
            initial
            and initial.get("diagnostic_stage") in {s.value for s in _REPAIRABLE_STAGES}
        ),
        repair_invoked=bool(repaired),
        correction_identity=(
            None if repair is None else repair.get("correction_context")
        ),
        repair_returned_candidate=repair_returned,
        repair_evaluable=repair_evaluable,
        repair_success=(
            None
            if not repair or repair.get("hard_semantic_success") is None
            else repair["hard_semantic_success"]
        ),
        original_error_corrected=(
            True
            if repair and repair.get("hard_semantic_success") is True
            else (
                False
                if repair_returned
                and repair.get("candidate_summary") == initial.get("candidate_summary")
                and repair.get("code") == initial.get("code")
                else None
            )
        ),
        repair_new_error=(
            None
            if not repair or repair.get("hard_semantic_success") is None
            else bool(
                repair.get("hard_semantic_success") is False
                and repair.get("code") != initial.get("code")
            )
        ),
        retry_used=bool(recovery.get("retry_used")),
        failover_used=bool(recovery.get("failover_used")),
        recovery_outcome=recovery.get("final_recovery_outcome"),
        provider_call_count=len(observer.calls),
        scientific_calls=guard.count,
        evaluation_failure_kinds=sorted(set(kinds)),
        assessment=evidence,
        excluded_reason=exclusions[0] if exclusions else evidence["excluded_reason"],
    )


def aggregate(rows):
    valid = [r for r in rows if not r.get("excluded_reason")]
    initial = [
        next((c for c in r["calls"] if c["phase"] == "detailed_planning"), {})
        for r in valid
    ]
    conditional = [c for r, c in zip(valid, initial) if r["scope"]["complete"] is True]
    positive_initial = [
        c for r, c in zip(valid, initial) if r["expected_outcome"] == "plan"
    ]
    repairs = [r for r in valid if r["repair_invoked"]]
    scopes = [r["scope"] for r in valid]
    matched = sum(s["required_matched"] or 0 for s in scopes)
    total = sum(s["required_total"] or 0 for s in scopes)
    metrics = dict(
        scope_required_family_recall=dict(
            numerator=matched,
            denominator=total,
            rate=matched / total if total else None,
        ),
        complete_scope=_ratio(
            [s["complete"] for s in scopes if s["required_total"] is not None]
        ),
        scope_unsupported_correctness=_ratio(
            [s["unsupported_correct"] for s in scopes]
        ),
        multi_family_completeness=_ratio(
            [s["complete"] for s in scopes if s["multi_family"]]
        ),
        unrelated_family_count=sum(s["unrelated_family_count"] or 0 for s in scopes),
        false_unsupported_count=sum(s["false_unsupported"] for s in scopes),
        initial_outcome_correctness=_ratio(
            [c.get("hard_semantic_success") for c in initial]
        ),
        first_candidate_hard_semantic_success=_ratio(
            [c.get("hard_semantic_success") for c in positive_initial]
        ),
        final_hard_semantic_success=_ratio(
            [r["final_hard_semantic_success"] for r in valid]
        ),
        final_positive_plan_semantic_success=_ratio(
            [
                r["final_hard_semantic_success"]
                for r in valid
                if r["expected_outcome"] == "plan"
            ]
        ),
        correct_non_plan_outcome=_ratio(
            [
                r["final_hard_semantic_success"]
                for r in valid
                if r["expected_outcome"] != "plan"
            ]
        ),
        safe_unsupported_accuracy=_ratio(
            [
                r["final_hard_semantic_success"]
                for r in valid
                if r["expected_outcome"] == "unsupported"
            ]
        ),
        false_refusal=_ratio(
            [
                (
                    any(c.get("code") == "UNSUPPORTED_REQUEST" for c in r["calls"])
                    if r["final_hard_semantic_success"] is not None
                    else None
                )
                for r in valid
                if r["expected_outcome"] == "plan"
            ]
        ),
        repair_invocations=len(repairs),
        operational_recovery_yield=_ratio(
            [r["repair_success"] is True for r in repairs]
        ),
        semantic_repair_success=_ratio(
            [r["repair_success"] for r in repairs if r["repair_evaluable"]]
        ),
        repair_parse_schema_failures=sum(
            any(
                c.get("parse_schema_pass") is False
                for c in r["calls"]
                if c["attempt_kind"] == "repair"
            )
            for r in repairs
        ),
        provider_transport_success=_ratio(
            [c["provider_outcome"] == "returned" for r in valid for c in r["calls"]]
        ),
        operational_plan_acceptance=_ratio(
            [
                r["operational_plan_accepted"]
                for r in valid
                if r["expected_outcome"] == "plan"
            ]
        ),
        transport_failures=sum(
            c["provider_outcome"] == "failed" for r in valid for c in r["calls"]
        ),
        rate_limited_sessions=sum(
            any(c.get("code") == "PROVIDER_RATE_LIMITED" for c in r["calls"])
            for r in valid
        ),
        unobserved_semantic_outcomes=sum(
            r["final_hard_semantic_success"] is None for r in valid
        ),
        excluded_cases=len(rows) - len(valid),
        hard_errors_by_category=dict(
            Counter(
                k
                for r in valid
                for k in r["evaluation_failure_kinds"]
                if k not in ("scope_overselection", "provider_transport")
                and (
                    r["stage_b_quality_eligible"]
                    or k in ("scope_omission", "scope_selection")
                    or (r["expected_scope_unsupported"] and k == "unsupported_handling")
                )
            )
        ),
        conditional_stage_b={
            k: _ratio([c.get(k) for c in conditional])
            for k in (
                "parse_schema_pass",
                "compiler_pass",
                "preflight_pass",
                "hard_semantic_success",
            )
        },
    )
    for key in ("parse_schema_pass", "compiler_pass", "preflight_pass"):
        metrics["initial_" + key] = _ratio([c.get(key) for c in initial])
    pairs = []
    for intent, repeat in sorted({(r["intent_id"], r["repetition"]) for r in valid}):
        group = [
            r for r in valid if (r["intent_id"], r["repetition"]) == (intent, repeat)
        ]
        if len(group) > 1:
            outcomes = [r["final_hard_semantic_success"] for r in group]
            pairs.append(None if None in outcomes else all(outcomes))
    metrics["wording_pair_consistency"] = _ratio(pairs)
    return metrics


def comparison_manifest(cases, registry, planner):
    root = Path(__file__).resolve().parents[2]

    def digest_files(paths):
        return fingerprint(
            {
                str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(paths)
            }
        )

    deps = sorted(
        (d.metadata["Name"], d.version)
        for d in importlib.metadata.distributions()
        if d.metadata["Name"]
    )
    return dict(
        repository_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        dependency_fingerprint=fingerprint(deps),
        corpus_fingerprint=fingerprint(_plain([asdict(c) for c in cases])),
        registry_fingerprint=PlanningScope(
            registry, tuple(capability_index(registry))
        ).scope_fingerprint,
        capability_fingerprint=fingerprint(dict(CAPABILITY_DESCRIPTIONS)),
        production_source_fingerprint=digest_files((root / "src/agent").rglob("*.py")),
        evaluator_source_fingerprint=digest_files(
            (root / "benchmarks/planner").glob("*.py")
        ),
        wire_mode=4,
        scorer_version=SCORER_VERSION,
        recovery_policy=planner.recovery_policy.to_dict(),
        sampling_controls=dict(
            temperature="unspecified", seed="unspecified", top_p="unspecified"
        ),
        built_in_sdk_retries=0,
        custom_hidden_provider_calls="not_observable",
        deterministic_live_generation=False,
    )


@dataclass(frozen=True)
class ScopedReport:
    data: Mapping

    def to_dict(self):
        return dict(self.data)


def run_scoped_benchmark(
    cases,
    *,
    model=None,
    model_profile=None,
    repetitions=1,
    selected_case_ids=None,
    replay_outcomes=None,
    stop_on_rate_limit=True,
    adjudications=None,
    recovery_profiles=(),
    model_factory_registry=None,
):
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("Invalid repetitions.")
    if (model is None) != (model_profile is None):
        raise ValueError("Model and profile must be supplied together.")
    if model is not None and replay_outcomes is not None:
        raise ValueError("Live and replay are exclusive.")
    if bool(recovery_profiles) != (model_factory_registry is not None):
        raise ValueError("Recovery profiles and factories must be supplied together.")
    selected = [
        c
        for c in cases
        if selected_case_ids is None or c.case.case_id in selected_case_ids
    ]
    if selected_case_ids and set(selected_case_ids) - {
        c.case.case_id for c in selected
    }:
        raise ValueError("Unknown scoped case IDs.")
    rows = []
    manifest = None
    stopped = False
    scripts = {}
    registry, guard = guarded_registry()
    # Calibrate every selected fixture before the first provider invocation.
    for case in selected:
        expected_selection = dict(
            selection_schema_version=1,
            decision=(
                dict(kind="unsupported")
                if case.scope["expect_unsupported"]
                else dict(
                    kind="select",
                    capability_ids=case.scope["required_coverage"]
                    + case.scope["plausible_optional_families"],
                )
            ),
        )
        scripted = [expected_selection, witness(case, registry)]
        # Validate positive witnesses separately, before any live call.
        if case.case.expected_outcome == "plan":
            calibration = inspect_candidate(
                case,
                json.dumps(scripted[1]),
                registry,
                PlanningScope(
                    registry, tuple(expected_selection["decision"]["capability_ids"])
                ),
            )
            if calibration["hard_semantic_success"] is not True:
                raise BenchmarkDefinitionError(
                    f'Invalid positive witness {case.case.case_id}: {calibration["code"]} {calibration["failures"]}'
                )
        scripts[case.case.case_id] = scripted
    for repeat in range(1, repetitions + 1):
        for case in selected:
            registry, guard = guarded_registry()
            scripted = scripts[case.case.case_id]
            actual_model = model or ReplayModel(
                (replay_outcomes or {}).get(case.case.case_id, scripted)
            )
            observed = Observer(actual_model, case, registry)
            planner = LLMPlanner(
                observed,
                profile=model_profile,
                wire_mode=PlanningWireMode.V4,
                recovery_profiles=recovery_profiles,
                model_factory_registry=(
                    None
                    if model_factory_registry is None
                    else ObservedFactories(model_factory_registry, observed)
                ),
            )
            if manifest is None:
                manifest = comparison_manifest(cases, registry, planner)
            with patch.object(
                AgentRuntime,
                "_run_execute",
                side_effect=AssertionError("Execution prohibited"),
            ) as execute_guard, patch.object(
                AgentRuntime,
                "_run_durable_execute",
                side_effect=AssertionError("Execution prohibited"),
            ) as durable_guard:
                result = AgentRuntime(planner=planner, registry=registry).run(
                    case.request(repeat)
                )
            evidence = (adjudications or {}).get(case.case.case_id)
            row = _finish(
                case,
                observed,
                result,
                registry,
                guard,
                repeat,
                assess_root_cause(**evidence) if evidence is not None else None,
            )
            if (
                guard.count
                or result.steps
                or execute_guard.called
                or durable_guard.called
            ):
                raise AssertionError("Benchmark attempted scientific execution.")
            rows.append(row)
            if (
                model is not None
                and stop_on_rate_limit
                and any(c.get("code") == "PROVIDER_RATE_LIMITED" for c in row["calls"])
            ):
                stopped = True
                break
        if stopped:
            break
    profile = (
        asdict(model_profile)
        if model_profile
        else dict(
            profile_id="offline", provider_id="offline", model_id="scoped-offline"
        )
    )
    for row in rows:
        row["model_profile"] = profile
    return ScopedReport(
        dict(
            schema_version=REPORT_VERSION,
            track="scoped-v4-live" if model else "scoped-v4-replay",
            model_profile=profile,
            comparison_manifest=manifest,
            cases=rows,
            metrics=aggregate(rows),
            recovery_profiles=[asdict(p) for p in recovery_profiles],
            pending_requests=[
                dict(case_id=c.case.case_id, repetition=i)
                for i in range(1, repetitions + 1)
                for c in selected
                if (c.case.case_id, i)
                not in {(r["case_id"], r["repetition"]) for r in rows}
            ],
            stopped_on_rate_limit=stopped,
            scheduled_requests=len(selected) * repetitions,
            completed_sessions=len(rows),
        )
    )
