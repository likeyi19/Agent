"""Deterministic benchmark harness for the production LLM planner.

The semantic oracle in this module is benchmark-only.  Production planning and
execution code must not import this package.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from agent.orchestration import (
    AgentRequest,
    AgentRuntime,
    LLMPlanner,
    PlanningModel,
    PlanningModelProfile,
    RunMode,
    ToolRegistry,
    build_default_tool_registry,
)


BENCHMARK_SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 4
_OUTCOMES = frozenset({"plan", "unsupported", "failure"})


class BenchmarkDefinitionError(ValueError):
    """Raised when the benchmark corpus or replay fixture is malformed."""


def _duplicate_rejecting_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise BenchmarkDefinitionError(f"Duplicate JSON key {key!r}.")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise BenchmarkDefinitionError(f"Non-standard JSON constant {value!r}.")


def _load_json(path: str | Path) -> object:
    source = Path(path)
    try:
        return json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=_reject_constant,
        )
    except BenchmarkDefinitionError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BenchmarkDefinitionError(
            f"Benchmark JSON could not be read from {source}."
        ) from exc


def _strict_keys(
    value: Mapping[str, object], required: set[str], context: str
) -> None:
    supplied = set(value)
    if supplied != required:
        raise BenchmarkDefinitionError(
            f"{context} fields must be exactly {sorted(required)}; "
            f"received {sorted(supplied)}."
        )


def _nonempty_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkDefinitionError(f"{context} must be a non-empty string.")
    return value


def _string_tuple(value: object, context: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise BenchmarkDefinitionError(f"{context} must be an array.")
    result = tuple(_nonempty_string(item, context) for item in value)
    if len(set(result)) != len(result):
        raise BenchmarkDefinitionError(f"{context} must not contain duplicates.")
    return result


def _plain_json(value: object, context: str) -> object:
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)
        return json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise BenchmarkDefinitionError(f"{context} must be strict JSON data.") from exc


@dataclass(frozen=True)
class SemanticPolicy:
    """Benchmark-only hard semantics layered over canonical plan expectations."""

    required_roles: tuple[str, ...]
    auxiliary_roles: tuple[str, ...]
    terminal_roles: tuple[str, ...]
    forbidden_tools: tuple[str, ...]
    required_order: tuple[tuple[str, str], ...]
    binding_alternatives: Mapping[
        tuple[str, str], tuple[Mapping[str, str], ...]
    ]
    optional_default_bindings: Mapping[tuple[str, str], object]


@dataclass(frozen=True)
class BenchmarkCase:
    """One provider-neutral prompt and its benchmark-only semantic oracle."""

    case_id: str
    tags: tuple[str, ...]
    prompt: str
    inputs: Mapping[str, object]
    expected_outcome: str
    expected_workflow: str
    expected_steps: tuple[Mapping[str, object], ...]
    expected_error_codes: tuple[str, ...]
    expected_preflight_valid: bool
    semantic_policy: SemanticPolicy | None

    @classmethod
    def from_mapping(cls, value: object, index: int) -> "BenchmarkCase":
        if not isinstance(value, Mapping):
            raise BenchmarkDefinitionError(f"Case {index} must be an object.")
        _strict_keys(value, {"id", "tags", "request", "expected"}, f"Case {index}")
        case_id = _nonempty_string(value["id"], f"Case {index}.id")
        tags = _string_tuple(value["tags"], f"Case {case_id}.tags")

        request = value["request"]
        if not isinstance(request, Mapping):
            raise BenchmarkDefinitionError(f"Case {case_id}.request must be an object.")
        _strict_keys(request, {"prompt", "inputs"}, f"Case {case_id}.request")
        prompt = _nonempty_string(request["prompt"], f"Case {case_id}.request.prompt")
        inputs = request["inputs"]
        if not isinstance(inputs, Mapping) or not all(
            isinstance(key, str) and key for key in inputs
        ):
            raise BenchmarkDefinitionError(
                f"Case {case_id}.request.inputs must be an object with string keys."
            )
        plain_inputs = _plain_json(inputs, f"Case {case_id}.request.inputs")
        if not isinstance(plain_inputs, Mapping):  # pragma: no cover - guarded above
            raise BenchmarkDefinitionError("Request inputs must remain a mapping.")

        expected = value["expected"]
        if not isinstance(expected, Mapping):
            raise BenchmarkDefinitionError(
                f"Case {case_id}.expected must be an object."
            )
        required_expected = {
            "outcome",
            "workflow",
            "steps",
            "error_codes",
            "preflight_valid",
        }
        allowed_expected_fields = {
            frozenset(required_expected),
            frozenset(required_expected | {"semantic"}),
        }
        if set(expected) not in allowed_expected_fields:
            raise BenchmarkDefinitionError(
                f"Case {case_id}.expected fields are invalid."
            )
        outcome = _nonempty_string(
            expected["outcome"], f"Case {case_id}.expected.outcome"
        )
        if outcome not in _OUTCOMES:
            raise BenchmarkDefinitionError(
                f"Case {case_id} uses unsupported expected outcome {outcome!r}."
            )
        workflow = _nonempty_string(
            expected["workflow"], f"Case {case_id}.expected.workflow"
        )
        error_codes = _string_tuple(
            expected["error_codes"], f"Case {case_id}.expected.error_codes"
        )
        preflight_valid = expected["preflight_valid"]
        if not isinstance(preflight_valid, bool):
            raise BenchmarkDefinitionError(
                f"Case {case_id}.expected.preflight_valid must be boolean."
            )
        raw_steps = expected["steps"]
        if not isinstance(raw_steps, list):
            raise BenchmarkDefinitionError(
                f"Case {case_id}.expected.steps must be an array."
            )
        steps = tuple(
            _validate_expected_step(step, case_id, step_index)
            for step_index, step in enumerate(raw_steps)
        )
        roles = tuple(str(step["role"]) for step in steps)
        if len(set(roles)) != len(roles):
            raise BenchmarkDefinitionError(
                f"Case {case_id} expected step roles must be unique."
            )
        role_set = set(roles)
        for step in steps:
            for dependency in step["depends_on"]:
                if dependency not in role_set:
                    raise BenchmarkDefinitionError(
                        f"Case {case_id} dependency {dependency!r} has no role."
                    )
            for binding in step["bindings"].values():
                if (
                    binding["kind"] == "ref"
                    and binding["producer_role"] not in role_set
                ):
                    raise BenchmarkDefinitionError(
                        f"Case {case_id} reference producer has no expected role."
                    )
        if outcome == "plan" and not steps:
            raise BenchmarkDefinitionError(f"Plan case {case_id} must have steps.")
        if outcome != "plan" and steps:
            raise BenchmarkDefinitionError(
                f"Non-plan case {case_id} must not define executable steps."
            )
        if outcome == "plan" and error_codes:
            raise BenchmarkDefinitionError(
                f"Plan case {case_id} must not define expected error codes."
            )
        if outcome != "plan" and not error_codes:
            raise BenchmarkDefinitionError(
                f"Non-plan case {case_id} requires expected error codes."
            )
        if outcome != "plan" and "semantic" in expected:
            raise BenchmarkDefinitionError(
                f"Non-plan case {case_id} cannot define plan semantics."
            )
        semantic_policy = (
            _default_semantic_policy(steps)
            if outcome == "plan" and "semantic" not in expected
            else _validate_semantic_policy(
                expected.get("semantic"),
                case_id,
                steps,
                inputs=plain_inputs,
            )
            if outcome == "plan"
            else None
        )
        return cls(
            case_id,
            tags,
            prompt,
            dict(plain_inputs),
            outcome,
            workflow,
            steps,
            error_codes,
            preflight_valid,
            semantic_policy,
        )


def _validate_expected_step(
    value: object, case_id: str, index: int
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise BenchmarkDefinitionError(
            f"Case {case_id} expected step {index} must be an object."
        )
    _strict_keys(
        value,
        {"role", "tool", "bindings", "depends_on"},
        f"Case {case_id} expected step {index}",
    )
    role = _nonempty_string(value["role"], f"Case {case_id} step role")
    tool = _nonempty_string(value["tool"], f"Case {case_id} step tool")
    depends_on = _string_tuple(
        value["depends_on"], f"Case {case_id} step {role}.depends_on"
    )
    raw_bindings = value["bindings"]
    if not isinstance(raw_bindings, Mapping) or not all(
        isinstance(key, str) and key for key in raw_bindings
    ):
        raise BenchmarkDefinitionError(
            f"Case {case_id} step {role}.bindings must be an object."
        )
    bindings: dict[str, Mapping[str, str]] = {}
    for argument_name, raw_binding in raw_bindings.items():
        bindings[argument_name] = _validate_expected_binding(
            raw_binding, case_id, role, argument_name
        )
    return {
        "role": role,
        "tool": tool,
        "bindings": bindings,
        "depends_on": depends_on,
    }


def _validate_expected_binding(
    value: object, case_id: str, role: str, argument_name: str
) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise BenchmarkDefinitionError(
            f"Case {case_id} {role}.{argument_name} binding must be an object."
        )
    kind = value.get("kind")
    if kind == "input":
        _strict_keys(
            value,
            {"kind", "input_name"},
            f"Case {case_id} {role}.{argument_name} binding",
        )
        return {
            "kind": "input",
            "input_name": _nonempty_string(
                value["input_name"],
                f"Case {case_id} {role}.{argument_name}.input_name",
            ),
        }
    if kind == "ref":
        _strict_keys(
            value,
            {"kind", "producer_role", "output_key"},
            f"Case {case_id} {role}.{argument_name} binding",
        )
        return {
            "kind": "ref",
            "producer_role": _nonempty_string(
                value["producer_role"],
                f"Case {case_id} {role}.{argument_name}.producer_role",
            ),
            "output_key": _nonempty_string(
                value["output_key"],
                f"Case {case_id} {role}.{argument_name}.output_key",
            ),
        }
    raise BenchmarkDefinitionError(
        f"Case {case_id} {role}.{argument_name} binding kind must be input or ref."
    )


def _role_cardinalities(
    value: object,
    *,
    context: str,
    canonical_roles: frozenset[str],
) -> tuple[str, ...]:
    if not isinstance(value, Mapping):
        raise BenchmarkDefinitionError(f"{context} must be an object.")
    roles: list[str] = []
    for raw_role, cardinality in value.items():
        role = _nonempty_string(raw_role, context)
        if role not in canonical_roles:
            raise BenchmarkDefinitionError(
                f"{context} names unknown canonical role {role!r}."
            )
        if type(cardinality) is not int or cardinality != 1:
            raise BenchmarkDefinitionError(
                f"{context}.{role} currently requires cardinality 1."
            )
        roles.append(role)
    return tuple(roles)


def _role_argument_key(
    value: object,
    *,
    context: str,
    canonical_steps: Mapping[str, Mapping[str, object]],
) -> tuple[str, str]:
    text = _nonempty_string(value, context)
    if text.count(".") != 1:
        raise BenchmarkDefinitionError(
            f"{context} must use the form 'role.argument'."
        )
    role, argument = text.split(".", 1)
    step = canonical_steps.get(role)
    if step is None or argument not in step["bindings"]:
        raise BenchmarkDefinitionError(
            f"{context} names unknown canonical binding {text!r}."
        )
    return role, argument


def _default_semantic_policy(
    steps: tuple[Mapping[str, object], ...],
) -> SemanticPolicy:
    roles = tuple(str(step["role"]) for step in steps)
    return SemanticPolicy(
        required_roles=roles,
        auxiliary_roles=(),
        terminal_roles=(roles[-1],),
        forbidden_tools=(),
        required_order=(),
        binding_alternatives={},
        optional_default_bindings={},
    )


def _validate_semantic_policy(
    value: object,
    case_id: str,
    steps: tuple[Mapping[str, object], ...],
    *,
    inputs: Mapping[str, object],
) -> SemanticPolicy:
    if not isinstance(value, Mapping):
        raise BenchmarkDefinitionError(
            f"Case {case_id}.expected.semantic must be an object."
        )
    required_fields = {
        "required_roles",
        "allowed_auxiliary_roles",
        "terminal_roles",
        "forbidden_tools",
        "required_order",
        "binding_alternatives",
        "optional_default_bindings",
    }
    _strict_keys(value, required_fields, f"Case {case_id}.expected.semantic")
    canonical_steps = {str(step["role"]): step for step in steps}
    canonical_roles = frozenset(canonical_steps)
    required_roles = _role_cardinalities(
        value["required_roles"],
        context=f"Case {case_id}.semantic.required_roles",
        canonical_roles=canonical_roles,
    )
    auxiliary_roles = _role_cardinalities(
        value["allowed_auxiliary_roles"],
        context=f"Case {case_id}.semantic.allowed_auxiliary_roles",
        canonical_roles=canonical_roles,
    )
    if set(required_roles).intersection(auxiliary_roles):
        raise BenchmarkDefinitionError(
            f"Case {case_id} semantic roles must be required or auxiliary, not both."
        )
    if set(required_roles).union(auxiliary_roles) != canonical_roles:
        raise BenchmarkDefinitionError(
            f"Case {case_id} semantic roles must cover every canonical role."
        )

    terminal_roles = _string_tuple(
        value["terminal_roles"], f"Case {case_id}.semantic.terminal_roles"
    )
    if not terminal_roles or not set(terminal_roles).issubset(required_roles):
        raise BenchmarkDefinitionError(
            f"Case {case_id} terminal roles must be required semantic roles."
        )
    forbidden_tools = _string_tuple(
        value["forbidden_tools"], f"Case {case_id}.semantic.forbidden_tools"
    )
    allowed_tools = {str(step["tool"]) for step in steps}
    if set(forbidden_tools).intersection(allowed_tools):
        raise BenchmarkDefinitionError(
            f"Case {case_id} forbids a tool used by an allowed semantic role."
        )

    raw_order = value["required_order"]
    if not isinstance(raw_order, list):
        raise BenchmarkDefinitionError(
            f"Case {case_id}.semantic.required_order must be an array."
        )
    required_order: list[tuple[str, str]] = []
    for index, edge in enumerate(raw_order):
        if (
            not isinstance(edge, list)
            or len(edge) != 2
            or not all(isinstance(role, str) and role for role in edge)
        ):
            raise BenchmarkDefinitionError(
                f"Case {case_id}.semantic.required_order[{index}] is invalid."
            )
        before, after = edge
        if (
            before == after
            or before not in canonical_roles
            or after not in canonical_roles
        ):
            raise BenchmarkDefinitionError(
                f"Case {case_id}.semantic.required_order[{index}] is invalid."
            )
        required_order.append((before, after))
    if len(set(required_order)) != len(required_order):
        raise BenchmarkDefinitionError(
            f"Case {case_id}.semantic.required_order contains duplicates."
        )

    raw_alternatives = value["binding_alternatives"]
    if not isinstance(raw_alternatives, Mapping):
        raise BenchmarkDefinitionError(
            f"Case {case_id}.semantic.binding_alternatives must be an object."
        )
    alternatives: dict[
        tuple[str, str], tuple[Mapping[str, str], ...]
    ] = {}
    for raw_key, raw_bindings in raw_alternatives.items():
        key = _role_argument_key(
            raw_key,
            context=f"Case {case_id}.semantic.binding_alternatives key",
            canonical_steps=canonical_steps,
        )
        if not isinstance(raw_bindings, list) or not raw_bindings:
            raise BenchmarkDefinitionError(
                f"Case {case_id} binding alternatives for {raw_key!r} must be nonempty."
            )
        bindings = tuple(
            _validate_expected_binding(
                binding,
                case_id,
                key[0],
                key[1],
            )
            for binding in raw_bindings
        )
        unique_bindings = {json.dumps(binding, sort_keys=True) for binding in bindings}
        if len(unique_bindings) != len(bindings):
            raise BenchmarkDefinitionError(
                f"Case {case_id} binding alternatives for {raw_key!r} duplicate."
            )
        alternatives[key] = bindings

    raw_optional_defaults = value["optional_default_bindings"]
    if not isinstance(raw_optional_defaults, Mapping):
        raise BenchmarkDefinitionError(
            f"Case {case_id}.semantic.optional_default_bindings must be an object."
        )
    optional_default_bindings: dict[tuple[str, str], object] = {}
    for raw_key, raw_default in raw_optional_defaults.items():
        key = _role_argument_key(
            raw_key,
            context=f"Case {case_id}.semantic.optional_default_bindings key",
            canonical_steps=canonical_steps,
        )
        expected_binding = canonical_steps[key[0]]["bindings"][key[1]]
        if expected_binding["kind"] != "input":
            raise BenchmarkDefinitionError(
                f"Case {case_id} default-equivalent binding {raw_key!r} "
                "must originate from request input."
            )
        input_name = expected_binding["input_name"]
        default = _plain_json(
            raw_default,
            f"Case {case_id}.semantic.optional_default_bindings.{raw_key}",
        )
        if input_name not in inputs or inputs[input_name] != default:
            raise BenchmarkDefinitionError(
                f"Case {case_id} default-equivalent binding {raw_key!r} "
                "must equal its structured request value."
            )
        optional_default_bindings[key] = default
    return SemanticPolicy(
        required_roles=required_roles,
        auxiliary_roles=auxiliary_roles,
        terminal_roles=terminal_roles,
        forbidden_tools=forbidden_tools,
        required_order=tuple(required_order),
        binding_alternatives=alternatives,
        optional_default_bindings=optional_default_bindings,
    )


def load_cases(path: str | Path) -> tuple[BenchmarkCase, ...]:
    """Load and strictly validate one benchmark corpus."""

    payload = _load_json(path)
    if not isinstance(payload, Mapping):
        raise BenchmarkDefinitionError("Benchmark corpus must be one JSON object.")
    _strict_keys(payload, {"schema_version", "cases"}, "Benchmark corpus")
    if payload["schema_version"] != BENCHMARK_SCHEMA_VERSION:
        raise BenchmarkDefinitionError("Unsupported benchmark corpus schema version.")
    raw_cases = payload["cases"]
    if not isinstance(raw_cases, list):
        raise BenchmarkDefinitionError("Benchmark corpus cases must be an array.")
    cases = tuple(
        BenchmarkCase.from_mapping(case, index)
        for index, case in enumerate(raw_cases)
    )
    case_ids = tuple(case.case_id for case in cases)
    if len(set(case_ids)) != len(case_ids):
        raise BenchmarkDefinitionError("Benchmark case IDs must be unique.")
    return cases


def load_replay_overrides(path: str | Path) -> Mapping[str, object]:
    """Load deterministic raw-response overrides for the offline track."""

    payload = _load_json(path)
    if not isinstance(payload, Mapping):
        raise BenchmarkDefinitionError("Replay fixture must be one JSON object.")
    _strict_keys(payload, {"schema_version", "overrides"}, "Replay fixture")
    if payload["schema_version"] != BENCHMARK_SCHEMA_VERSION:
        raise BenchmarkDefinitionError("Unsupported replay fixture schema version.")
    overrides = payload["overrides"]
    if not isinstance(overrides, Mapping):
        raise BenchmarkDefinitionError("Replay overrides must be an object.")
    return dict(overrides)


def _input_binding(input_name: str) -> dict[str, object]:
    return {
        "binding_type": "input",
        "input_name": input_name,
    }


def _ref_binding(producer_step_id: str, output_key: str) -> dict[str, object]:
    return {
        "binding_type": "ref",
        "ref_step_id": producer_step_id,
        "ref_output_key": output_key,
    }


def oracle_response(case: BenchmarkCase) -> str:
    """Create the deterministic scripted response for an unmodified case."""

    if case.expected_outcome != "plan":
        return json.dumps(
            {
                "schema_version": 3,
                "status": "unsupported",
                "steps": [],
                "reason": "Synthetic benchmark rejection text is not scored.",
            },
            sort_keys=True,
        )
    step_ids = {
        str(step["role"]): f"provider-step-{index + 101}"
        for index, step in enumerate(case.expected_steps)
    }
    registry = build_default_tool_registry()
    steps: list[dict[str, object]] = []
    for index, expected in enumerate(case.expected_steps):
        tool_name = str(expected["tool"])
        spec = registry.get(tool_name)
        arguments: dict[str, object] = {
            argument_name: None for argument_name in spec.optional_arguments
        }
        for argument_name, binding in expected["bindings"].items():
            if binding["kind"] == "input":
                arguments[argument_name] = _input_binding(binding["input_name"])
            else:
                arguments[argument_name] = _ref_binding(
                    step_ids[binding["producer_role"]],
                    binding["output_key"],
                )
        if set(spec.required_arguments).difference(arguments):
            raise BenchmarkDefinitionError(
                f"Case {case.case_id} omits a required wire argument."
            )
        steps.append(
            {
                "step_id": step_ids[str(expected["role"])],
                "tool_name": tool_name,
                "arguments": arguments,
                "depends_on": [step_ids[role] for role in expected["depends_on"]],
                "description": f"Provider prose {index} is deliberately ignored.",
            }
        )
    return json.dumps(
        {
            "schema_version": 3,
            "status": "plan",
            "steps": steps,
            "reason": None,
        },
        sort_keys=True,
    )


class ScriptedPlanningModel:
    """One-response offline PlanningModel with no provider dependency."""

    model_id = "offline-scripted-m9.1"

    def __init__(self, response: str) -> None:
        self._response = response
        self.calls = 0

    def complete(self, *, prompt: str, response_schema: Mapping[str, object]) -> str:
        self.calls += 1
        json.loads(prompt)
        json.dumps(response_schema, allow_nan=False)
        return self._response


class RecordingPlanningModel:
    """Benchmark-only observer preserving raw binding origins in memory."""

    def __init__(self, delegate: PlanningModel) -> None:
        self._delegate = delegate
        self.calls = 0
        self.responses: list[str] = []

    @property
    def model_id(self) -> str:
        return self._delegate.model_id

    def complete(self, *, prompt: str, response_schema: Mapping[str, object]) -> str:
        self.calls += 1
        response = self._delegate.complete(
            prompt=prompt,
            response_schema=response_schema,
        )
        if isinstance(response, str):
            self.responses.append(response)
        return response


class _RecordingModelFactoryResolver:
    """Benchmark-only observer around an explicitly supplied factory registry."""

    def __init__(self, delegate: object) -> None:
        self._delegate = delegate
        self.created: list[tuple[PlanningModelProfile, RecordingPlanningModel]] = []

    @property
    def provider_ids(self) -> tuple[str, ...]:
        values = getattr(self._delegate, "provider_ids")
        return tuple(values)

    def create(self, profile: PlanningModelProfile) -> PlanningModel:
        model = self._delegate.create(profile)
        observed = RecordingPlanningModel(model)
        self.created.append((profile, observed))
        return observed


class _ScientificCallGuard:
    def __init__(self) -> None:
        self.count = 0

    def function(self, tool_name: str):
        def guarded(**_: object) -> object:
            self.count += 1
            raise AssertionError(
                f"Planner benchmark invoked scientific tool {tool_name!r}."
            )

        return guarded


def guarded_registry() -> tuple[ToolRegistry, _ScientificCallGuard]:
    """Return the full registry with every scientific callable guarded."""

    source = build_default_tool_registry()
    guard = _ScientificCallGuard()
    guarded_specs = []
    for name in source.names():
        spec = source.get(name)
        guarded_specs.append(
            replace(spec, function=guard.function(spec.name))
        )
    return (
        ToolRegistry(tuple(guarded_specs)),
        guard,
    )


def _response_from_override(case: BenchmarkCase, override: object | None) -> str:
    if override is None:
        return oracle_response(case)
    if not isinstance(override, Mapping):
        raise BenchmarkDefinitionError(
            f"Replay override for {case.case_id} must be an object."
        )
    if set(override) == {"raw_response"}:
        return _nonempty_string(
            override["raw_response"], f"Replay override {case.case_id}.raw_response"
        )
    if set(override) == {"response"}:
        return json.dumps(
            _plain_json(
                override["response"],
                f"Replay override {case.case_id}.response",
            ),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
        )
    raise BenchmarkDefinitionError(
        f"Replay override for {case.case_id} must contain raw_response or response."
    )


def _parse_raw_response(response: str | None) -> Mapping[str, object] | None:
    if response is None:
        return None
    try:
        payload = json.loads(
            response,
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=_reject_constant,
        )
    except (BenchmarkDefinitionError, json.JSONDecodeError, ValueError):
        return None
    return payload if isinstance(payload, Mapping) else None


def _actual_binding_map(
    raw_step: object,
    *,
    actual_id_to_index: Mapping[str, int],
    expected_roles: Sequence[str],
) -> Mapping[str, Mapping[str, object]]:
    if not isinstance(raw_step, Mapping) or not isinstance(
        raw_step.get("arguments"), Mapping
    ):
        return {}
    result: dict[str, Mapping[str, object]] = {}
    for name, binding in raw_step["arguments"].items():
        if binding is None:
            continue
        if not isinstance(binding, Mapping):
            result[name] = {"kind": "invalid"}
            continue
        if binding.get("binding_type") == "input":
            result[name] = {
                "kind": "input",
                "input_name": binding.get("input_name"),
            }
        elif binding.get("binding_type") == "ref":
            producer_id = binding.get("ref_step_id")
            producer_index = (
                actual_id_to_index.get(producer_id)
                if isinstance(producer_id, str)
                else None
            )
            producer_role = (
                expected_roles[producer_index]
                if producer_index is not None and producer_index < len(expected_roles)
                else None
            )
            result[name] = {
                "kind": "ref",
                "producer_role": producer_role,
                "output_key": binding.get("ref_output_key"),
            }
        else:
            result[name] = {"kind": "invalid"}
    return result


def _score_items(
    expected: Mapping[object, object], actual: Mapping[object, object]
) -> tuple[int, int]:
    keys = set(expected).union(actual)
    return sum(expected.get(key) == actual.get(key) for key in keys), len(keys)


def _structural_binding_map(
    raw_step: object,
    *,
    actual_id_to_index: Mapping[str, int],
    role_by_index: Sequence[str | None],
) -> Mapping[str, Mapping[str, object]]:
    """Normalize bindings without retaining provider IDs or request values."""

    if not isinstance(raw_step, Mapping) or not isinstance(
        raw_step.get("arguments"), Mapping
    ):
        return {}
    result: dict[str, Mapping[str, object]] = {}
    for name, binding in raw_step["arguments"].items():
        if binding is None:
            continue
        if not isinstance(binding, Mapping):
            result[name] = {"kind": "invalid"}
            continue
        if binding.get("binding_type") == "input":
            result[name] = {
                "kind": "input",
                "input_name": binding.get("input_name"),
            }
        elif binding.get("binding_type") == "ref":
            producer_id = binding.get("ref_step_id")
            producer_index = (
                actual_id_to_index.get(producer_id)
                if isinstance(producer_id, str)
                else None
            )
            producer_role = (
                role_by_index[producer_index]
                if producer_index is not None
                and producer_index < len(role_by_index)
                else None
            )
            result[name] = {
                "kind": "ref",
                "producer_role": producer_role,
                "output_key": binding.get("ref_output_key"),
            }
        else:
            result[name] = {"kind": "invalid"}
    return result


def _normalized_plan(
    raw_steps: Sequence[object],
    *,
    actual_id_to_index: Mapping[str, int],
    role_by_index: Sequence[str | None],
) -> tuple[Mapping[str, object], ...]:
    """Return sanitized structural plan data suitable for persisted reports."""

    normalized: list[Mapping[str, object]] = []
    for index, raw_step in enumerate(raw_steps):
        step = raw_step if isinstance(raw_step, Mapping) else {}
        dependencies: list[str | None] = []
        raw_dependencies = step.get("depends_on")
        if isinstance(raw_dependencies, list):
            for producer_id in raw_dependencies:
                producer_index = (
                    actual_id_to_index.get(producer_id)
                    if isinstance(producer_id, str)
                    else None
                )
                dependencies.append(
                    role_by_index[producer_index]
                    if producer_index is not None
                    and producer_index < len(role_by_index)
                    else None
                )
        bindings = _structural_binding_map(
            raw_step,
            actual_id_to_index=actual_id_to_index,
            role_by_index=role_by_index,
        )
        normalized.append(
            {
                "role": role_by_index[index] if index < len(role_by_index) else None,
                "tool": step.get("tool_name")
                if isinstance(step.get("tool_name"), str)
                else None,
                "bindings": {
                    name: dict(binding) for name, binding in sorted(bindings.items())
                },
                "depends_on_roles": dependencies,
            }
        )
    return tuple(normalized)


def _candidate_semantic_failures(
    case: BenchmarkCase,
    raw_steps: Sequence[object],
    *,
    actual_id_to_index: Mapping[str, int],
    role_by_index: Sequence[str | None],
) -> tuple[str, ...]:
    policy = case.semantic_policy
    if policy is None:  # pragma: no cover - callers guard non-plan cases
        return ("semantic_policy_missing",)
    canonical = {str(step["role"]): step for step in case.expected_steps}
    assigned_index = {
        role: index for index, role in enumerate(role_by_index) if role is not None
    }
    failures: list[str] = []

    for role in policy.required_roles:
        if role not in assigned_index:
            failures.append(f"missing_required_role:{role}")
    for index, role in enumerate(role_by_index):
        if role is None:
            failures.append(f"unexpected_step:{index}")
    for index, raw_step in enumerate(raw_steps):
        tool = raw_step.get("tool_name") if isinstance(raw_step, Mapping) else None
        if tool in policy.forbidden_tools:
            failures.append(f"forbidden_tool:{tool}")

        role = role_by_index[index] if index < len(role_by_index) else None
        if role is None:
            continue
        actual_bindings = _structural_binding_map(
            raw_step,
            actual_id_to_index=actual_id_to_index,
            role_by_index=role_by_index,
        )
        expected_bindings = canonical[role]["bindings"]
        for argument_name, expected_binding in expected_bindings.items():
            key = (role, argument_name)
            actual_binding = actual_bindings.get(argument_name)
            if actual_binding is None and key in policy.optional_default_bindings:
                continue
            alternatives = policy.binding_alternatives.get(
                key, (expected_binding,)
            )
            if actual_binding not in alternatives:
                failures.append(f"binding_mismatch:{role}.{argument_name}")
        for argument_name in set(actual_bindings).difference(expected_bindings):
            failures.append(f"unexpected_binding:{role}.{argument_name}")

    for before, after in policy.required_order:
        before_index = assigned_index.get(before)
        after_index = assigned_index.get(after)
        if (
            before_index is not None
            and after_index is not None
            and before_index >= after_index
        ):
            failures.append(f"order_violation:{before}->{after}")

    consumers_by_role: dict[str, set[str | None]] = {}
    for consumer_index, raw_step in enumerate(raw_steps):
        if not isinstance(raw_step, Mapping) or not isinstance(
            raw_step.get("depends_on"), list
        ):
            continue
        consumer_role = (
            role_by_index[consumer_index]
            if consumer_index < len(role_by_index)
            else None
        )
        for producer_id in raw_step["depends_on"]:
            producer_index = (
                actual_id_to_index.get(producer_id)
                if isinstance(producer_id, str)
                else None
            )
            if producer_index is None or producer_index >= len(role_by_index):
                continue
            producer_role = role_by_index[producer_index]
            if producer_role is not None:
                consumers_by_role.setdefault(producer_role, set()).add(consumer_role)
    for role in policy.terminal_roles:
        if consumers_by_role.get(role):
            failures.append(f"terminal_role_has_consumer:{role}")
    return tuple(sorted(set(failures)))


def _match_semantic_roles(
    case: BenchmarkCase,
    raw_steps: Sequence[object],
    *,
    actual_id_to_index: Mapping[str, int],
) -> tuple[tuple[str | None, ...], tuple[str, ...]]:
    """Assign actual steps to semantic roles using tools and graph provenance."""

    policy = case.semantic_policy
    if policy is None:
        roles = tuple(None for _ in raw_steps)
        return roles, ()
    canonical = {str(step["role"]): step for step in case.expected_steps}
    allowed_roles = policy.required_roles + policy.auxiliary_roles
    choices: list[tuple[str | None, ...]] = []
    for raw_step in raw_steps:
        tool = raw_step.get("tool_name") if isinstance(raw_step, Mapping) else None
        matching_roles = tuple(
            role for role in allowed_roles if canonical[role]["tool"] == tool
        )
        choices.append(matching_roles + (None,))

    best_roles: tuple[str | None, ...] | None = None
    best_failures: tuple[str, ...] | None = None
    for candidate in product(*choices):
        assigned = tuple(role for role in candidate if role is not None)
        if len(set(assigned)) != len(assigned):
            continue
        failures = _candidate_semantic_failures(
            case,
            raw_steps,
            actual_id_to_index=actual_id_to_index,
            role_by_index=candidate,
        )
        rank = (len(failures), failures, tuple(role or "~" for role in candidate))
        if best_roles is None:
            best_roles = candidate
            best_failures = failures
            best_rank = rank
        elif rank < best_rank:
            best_roles = candidate
            best_failures = failures
            best_rank = rank
    if best_roles is None:  # pragma: no cover - every step always permits None
        return tuple(None for _ in raw_steps), ("role_assignment_failed",)
    return best_roles, best_failures or ()


@dataclass(frozen=True)
class CaseScore:
    case_id: str
    repetition: int
    tags: tuple[str, ...]
    workflow: str
    expected_outcome: str
    actual_error_code: str | None
    syntactically_valid_plan: bool
    preflight_valid_plan: bool
    hard_semantic_correct: bool
    semantically_correct: bool
    canonical_workflow_conformant: bool | None
    hard_semantic_failures: tuple[str, ...]
    exact_tool_sequence: bool | None
    binding_correct: int
    binding_total: int
    dependency_reference_correct: int
    dependency_reference_total: int
    hallucinated_tool_count: int
    emitted_tool_count: int
    unsupported_rejected: bool | None
    false_unsupported: bool
    unsupported_false_acceptance: bool
    semantic_wrong_but_preflight_valid: bool
    provider_calls: int
    first_attempt_semantic_correct: bool
    transport_recovered: bool
    retry_used: bool
    repair_attempted: bool
    repair_success: bool
    failover_attempted: bool
    failover_success: bool
    recovery_path: str
    final_recovery_source: str | None
    ordered_profile_usage: tuple[str, ...]
    final_planning_success: bool
    final_failure_class: str | None
    final_provider_failure: str | None
    scientific_calls: int
    actual_tool_sequence: tuple[str, ...]
    normalized_plan: tuple[Mapping[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "repetition": self.repetition,
            "tags": list(self.tags),
            "workflow": self.workflow,
            "expected_outcome": self.expected_outcome,
            "actual_error_code": self.actual_error_code,
            "syntactically_valid_plan": self.syntactically_valid_plan,
            "preflight_valid_plan": self.preflight_valid_plan,
            "hard_semantic_correct": self.hard_semantic_correct,
            "semantically_correct": self.semantically_correct,
            "canonical_workflow_conformant": self.canonical_workflow_conformant,
            "hard_semantic_failures": list(self.hard_semantic_failures),
            "exact_tool_sequence": self.exact_tool_sequence,
            "binding_correct": self.binding_correct,
            "binding_total": self.binding_total,
            "dependency_reference_correct": self.dependency_reference_correct,
            "dependency_reference_total": self.dependency_reference_total,
            "hallucinated_tool_count": self.hallucinated_tool_count,
            "emitted_tool_count": self.emitted_tool_count,
            "unsupported_rejected": self.unsupported_rejected,
            "false_unsupported": self.false_unsupported,
            "unsupported_false_acceptance": self.unsupported_false_acceptance,
            "semantic_wrong_but_preflight_valid": (
                self.semantic_wrong_but_preflight_valid
            ),
            "provider_calls": self.provider_calls,
            "first_attempt_semantic_correct": self.first_attempt_semantic_correct,
            "transport_recovered": self.transport_recovered,
            "retry_used": self.retry_used,
            "repair_attempted": self.repair_attempted,
            "repair_success": self.repair_success,
            "failover_attempted": self.failover_attempted,
            "failover_success": self.failover_success,
            "recovery_path": self.recovery_path,
            "final_recovery_source": self.final_recovery_source,
            "ordered_profile_usage": list(self.ordered_profile_usage),
            "final_planning_success": self.final_planning_success,
            "final_failure_class": self.final_failure_class,
            "final_provider_failure": self.final_provider_failure,
            "scientific_calls": self.scientific_calls,
            "actual_tool_sequence": list(self.actual_tool_sequence),
            "normalized_plan": [dict(step) for step in self.normalized_plan],
        }


def _score_case(
    case: BenchmarkCase,
    *,
    result: object,
    raw_response: str | None,
    provider_calls: int,
    scientific_calls: int,
    registry: ToolRegistry,
    repetition: int,
) -> CaseScore:
    plan = getattr(result, "plan", None)
    verification = getattr(result, "verification", None)
    errors = getattr(result, "errors", ())
    error_code = errors[0].code if errors else None
    trace = getattr(result, "trace", ())
    recovery_summary = next(
        (
            event.details
            for event in reversed(trace)
            if event.details.get("diagnostic_schema_version") == 3
            and event.details.get("code") == "PLANNING_RECOVERY_SUMMARY"
        ),
        {},
    )
    retry_used = bool(recovery_summary.get("retry_used", provider_calls > 1))
    repair_used = bool(recovery_summary.get("repair_used", False))
    failover_used = bool(recovery_summary.get("failover_used", False))
    total_calls = recovery_summary.get("total_provider_call_count")
    if isinstance(total_calls, int) and not isinstance(total_calls, bool):
        provider_calls = total_calls
    recovery_path = str(
        recovery_summary.get(
            "final_recovery_outcome",
            "initial_success" if error_code is None else "failed",
        )
    )
    ordered_profile_usage = tuple(
        str(event.details["profile_id"])
        for event in trace
        if event.details.get("diagnostic_schema_version") == 3
        and event.details.get("code") == "PROVIDER_CALL_STARTED"
        and isinstance(event.details.get("profile_id"), str)
    )
    payload = _parse_raw_response(raw_response)
    raw_steps = payload.get("steps", []) if payload is not None else []
    if not isinstance(raw_steps, list):
        raw_steps = []
    raw_plan_decision = bool(
        payload is not None
        and payload.get("status") == "plan"
        and raw_steps
    )
    emitted_tools = tuple(
        step["tool_name"]
        for step in raw_steps
        if isinstance(step, Mapping) and isinstance(step.get("tool_name"), str)
    )
    syntactically_valid = bool(
        plan is not None
        or (
            raw_plan_decision
            and error_code
            in {
                "INVALID_OUTPUT_REFERENCE",
                "INVALID_PLAN_STRUCTURE",
                "PLAN_PREFLIGHT_FAILED",
            }
        )
    )
    preflight_valid = bool(
        plan is not None
        and verification is not None
        and verification.passed
    )
    actual_tools = (
        emitted_tools
        if plan is None
        else tuple(step.tool_name for step in plan.steps)
    )
    expected_tools = tuple(str(step["tool"]) for step in case.expected_steps)
    exact_sequence = (
        actual_tools == expected_tools if case.expected_outcome == "plan" else None
    )
    actual_id_to_index = {
        step["step_id"]: index
        for index, step in enumerate(raw_steps)
        if isinstance(step, Mapping) and isinstance(step.get("step_id"), str)
    }
    expected_roles = tuple(str(step["role"]) for step in case.expected_steps)

    expected_bindings: dict[tuple[int, str], object] = {}
    actual_bindings: dict[tuple[int, str], object] = {}
    for index, expected_step in enumerate(case.expected_steps):
        for name, binding in expected_step["bindings"].items():
            expected_bindings[(index, name)] = binding
    for index, raw_step in enumerate(raw_steps):
        for name, binding in _actual_binding_map(
            raw_step,
            actual_id_to_index=actual_id_to_index,
            expected_roles=expected_roles,
        ).items():
            actual_bindings[(index, name)] = binding
    binding_correct, binding_total = _score_items(expected_bindings, actual_bindings)

    expected_relations: dict[tuple[object, ...], object] = {}
    actual_relations: dict[tuple[object, ...], object] = {}
    for consumer_index, expected_step in enumerate(case.expected_steps):
        consumer_role = expected_step["role"]
        for producer_role in expected_step["depends_on"]:
            key = ("dependency", producer_role, consumer_role)
            expected_relations[key] = True
        for argument_name, binding in expected_step["bindings"].items():
            if binding["kind"] == "ref":
                key = (
                    "reference",
                    consumer_role,
                    argument_name,
                    binding["producer_role"],
                    binding["output_key"],
                )
                expected_relations[key] = True
    for consumer_index, raw_step in enumerate(raw_steps):
        consumer_role = (
            expected_roles[consumer_index]
            if consumer_index < len(expected_roles)
            else f"__extra_{consumer_index}"
        )
        if isinstance(raw_step, Mapping) and isinstance(
            raw_step.get("depends_on"), list
        ):
            for producer_id in raw_step["depends_on"]:
                producer_index = actual_id_to_index.get(producer_id)
                producer_role = (
                    expected_roles[producer_index]
                    if producer_index is not None
                    and producer_index < len(expected_roles)
                    else None
                )
                actual_relations[("dependency", producer_role, consumer_role)] = True
        for argument_name, binding in _actual_binding_map(
            raw_step,
            actual_id_to_index=actual_id_to_index,
            expected_roles=expected_roles,
        ).items():
            if binding.get("kind") == "ref":
                actual_relations[
                    (
                        "reference",
                        consumer_role,
                        argument_name,
                        binding.get("producer_role"),
                        binding.get("output_key"),
                    )
                ] = True
    relation_correct, relation_total = _score_items(
        expected_relations, actual_relations
    )

    if case.expected_outcome == "plan":
        role_by_index, structural_failures = _match_semantic_roles(
            case,
            raw_steps,
            actual_id_to_index=actual_id_to_index,
        )
        hard_failures = list(structural_failures)
        if not syntactically_valid:
            hard_failures.append("plan_not_constructed")
        if not preflight_valid:
            hard_failures.append("plan_preflight_failed")
        hard_failures = sorted(set(hard_failures))
        hard_semantic = not hard_failures
        canonical_conformant = bool(
            syntactically_valid
            and preflight_valid == case.expected_preflight_valid
            and exact_sequence
            and binding_correct == binding_total
            and relation_correct == relation_total
        )
    else:
        role_by_index = tuple(None for _ in raw_steps)
        hard_semantic = bool(
            not syntactically_valid and error_code in case.expected_error_codes
        )
        hard_failures = [] if hard_semantic else ["expected_rejection_not_observed"]
        canonical_conformant = None
    normalized_plan = _normalized_plan(
        raw_steps,
        actual_id_to_index=actual_id_to_index,
        role_by_index=role_by_index,
    )
    hallucinated = sum(not registry.contains(name) for name in emitted_tools)
    unsupported_rejected = (
        error_code == "UNSUPPORTED_REQUEST"
        if case.expected_outcome == "unsupported"
        else None
    )
    false_unsupported = bool(
        case.expected_outcome == "plan" and error_code == "UNSUPPORTED_REQUEST"
    )
    unsupported_false_acceptance = bool(
        case.expected_outcome == "unsupported" and raw_plan_decision
    )
    return CaseScore(
        case_id=case.case_id,
        repetition=repetition,
        tags=case.tags,
        workflow=case.expected_workflow,
        expected_outcome=case.expected_outcome,
        actual_error_code=error_code,
        syntactically_valid_plan=syntactically_valid,
        preflight_valid_plan=preflight_valid,
        hard_semantic_correct=hard_semantic,
        semantically_correct=hard_semantic,
        canonical_workflow_conformant=canonical_conformant,
        hard_semantic_failures=tuple(hard_failures),
        exact_tool_sequence=exact_sequence,
        binding_correct=binding_correct,
        binding_total=binding_total,
        dependency_reference_correct=relation_correct,
        dependency_reference_total=relation_total,
        hallucinated_tool_count=hallucinated,
        emitted_tool_count=len(emitted_tools),
        unsupported_rejected=unsupported_rejected,
        false_unsupported=false_unsupported,
        unsupported_false_acceptance=unsupported_false_acceptance,
        semantic_wrong_but_preflight_valid=preflight_valid and not hard_semantic,
        provider_calls=provider_calls,
        first_attempt_semantic_correct=(
            hard_semantic and not retry_used and not repair_used and not failover_used
        ),
        transport_recovered=(
            hard_semantic and recovery_path == "transport_recovered"
        ),
        retry_used=retry_used,
        repair_attempted=repair_used,
        repair_success=hard_semantic and recovery_path == "repair_recovered",
        failover_attempted=failover_used,
        failover_success=hard_semantic and recovery_path == "failover_recovered",
        recovery_path=recovery_path,
        final_recovery_source={
            "initial_success": "primary_initial",
            "transport_recovered": "primary_transport_retry",
            "repair_recovered": "primary_repair",
            "failover_recovered": "secondary_failover",
        }.get(recovery_path),
        ordered_profile_usage=ordered_profile_usage,
        final_planning_success=(
            hard_semantic and case.expected_outcome == "plan" and plan is not None
        ),
        final_failure_class=(
            None
            if error_code is None
            else "unsupported"
            if error_code == "UNSUPPORTED_REQUEST"
            else "provider"
            if error_code.startswith("PROVIDER_")
            or error_code.startswith("PLANNING_PROVIDER_")
            else "candidate"
        ),
        final_provider_failure=(
            error_code
            if error_code is not None
            and (
                error_code.startswith("PROVIDER_")
                or error_code.startswith("PLANNING_PROVIDER_")
            )
            else None
        ),
        scientific_calls=scientific_calls,
        actual_tool_sequence=actual_tools,
        normalized_plan=normalized_plan,
    )


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def aggregate_metrics(scores: Sequence[CaseScore]) -> Mapping[str, object]:
    """Aggregate the required M9.1 metrics for any score subset."""

    supported = tuple(score for score in scores if score.expected_outcome == "plan")
    unsupported = tuple(
        score for score in scores if score.expected_outcome == "unsupported"
    )
    failures = tuple(score for score in scores if score.expected_outcome == "failure")
    preflight_plans = tuple(score for score in scores if score.preflight_valid_plan)
    emitted_tools = sum(score.emitted_tool_count for score in scores)
    provider_calls = sum(score.provider_calls for score in scores)
    return {
        "request_count": len(scores),
        "supported_request_count": len(supported),
        "unsupported_request_count": len(unsupported),
        "failure_request_count": len(failures),
        "planning_success_rate": _rate(
            sum(score.syntactically_valid_plan for score in supported), len(supported)
        ),
        "executable_plan_rate": _rate(
            sum(score.preflight_valid_plan for score in supported), len(supported)
        ),
        "exact_tool_sequence_accuracy": _rate(
            sum(score.exact_tool_sequence is True for score in supported),
            len(supported),
        ),
        "hard_semantic_success_rate": _rate(
            sum(score.hard_semantic_correct for score in scores), len(scores)
        ),
        "canonical_workflow_conformance_rate": _rate(
            sum(score.canonical_workflow_conformant is True for score in supported),
            len(supported),
        ),
        "argument_binding_accuracy": _rate(
            sum(score.binding_correct for score in supported),
            sum(score.binding_total for score in supported),
        ),
        "dependency_reference_accuracy": _rate(
            sum(score.dependency_reference_correct for score in supported),
            sum(score.dependency_reference_total for score in supported),
        ),
        "hallucinated_tool_rate": _rate(
            sum(score.hallucinated_tool_count for score in scores), emitted_tools
        ),
        "unsupported_request_rejection_accuracy": _rate(
            sum(score.unsupported_rejected is True for score in unsupported),
            len(unsupported),
        ),
        "false_unsupported_rate": _rate(
            sum(score.false_unsupported for score in supported), len(supported)
        ),
        "unsupported_false_acceptance_rate": _rate(
            sum(score.unsupported_false_acceptance for score in unsupported),
            len(unsupported),
        ),
        "semantic_wrong_but_preflight_valid_rate": _rate(
            sum(score.semantic_wrong_but_preflight_valid for score in preflight_plans),
            len(preflight_plans),
        ),
        "first_attempt_semantic_success_rate": _rate(
            sum(score.first_attempt_semantic_correct for score in scores), len(scores)
        ),
        "first_attempt_plan_success_rate": _rate(
            sum(score.first_attempt_semantic_correct for score in supported),
            len(supported),
        ),
        "transport_retry_rate": _rate(
            sum(score.retry_used for score in scores), len(scores)
        ),
        "transport_recovery_success_rate": _rate(
            sum(score.transport_recovered for score in scores),
            sum(score.retry_used for score in scores),
        ),
        "repair_attempt_rate": _rate(
            sum(score.repair_attempted for score in scores), len(scores)
        ),
        "repair_success_rate": _rate(
            sum(score.repair_success for score in scores),
            sum(score.repair_attempted for score in scores),
        ),
        "failover_attempt_rate": _rate(
            sum(score.failover_attempted for score in scores), len(scores)
        ),
        "failover_success_rate": _rate(
            sum(score.failover_success for score in scores),
            sum(score.failover_attempted for score in scores),
        ),
        "failover_rate": _rate(
            sum(score.failover_attempted for score in scores), len(scores)
        ),
        "fallback_rate": None,
        "final_planning_success_rate": _rate(
            sum(score.hard_semantic_correct for score in scores), len(scores)
        ),
        "final_plan_success_rate": _rate(
            sum(score.final_planning_success for score in supported),
            len(supported),
        ),
        "provider_calls_per_request": _rate(provider_calls, len(scores)),
        "maximum_provider_calls": max(
            (score.provider_calls for score in scores), default=0
        ),
        "scientific_call_count": sum(score.scientific_calls for score in scores),
    }


@dataclass(frozen=True)
class BenchmarkReport:
    track: str
    profile_id: str
    provider_id: str
    model_id: str
    repetitions: int
    cases: tuple[CaseScore, ...]
    metrics: Mapping[str, object]
    metrics_by_tag: Mapping[str, Mapping[str, object]]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": REPORT_SCHEMA_VERSION,
            "track": self.track,
            "profile_id": self.profile_id,
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "repetitions": self.repetitions,
            "cases": [case.to_dict() for case in self.cases],
            "metrics": dict(self.metrics),
            "metrics_by_tag": {
                tag: dict(metrics) for tag, metrics in self.metrics_by_tag.items()
            },
        }


def run_benchmark(
    cases: Sequence[BenchmarkCase],
    *,
    model: PlanningModel | None = None,
    model_profile: PlanningModelProfile | None = None,
    replay_overrides: Mapping[str, object] | None = None,
    repetitions: int = 1,
    selected_case_ids: frozenset[str] | None = None,
    recovery_profiles: tuple[PlanningModelProfile, ...] = (),
    model_factory_registry: object | None = None,
) -> BenchmarkReport:
    """Run offline scripted or explicitly supplied live planning in PLAN_ONLY."""

    if (
        isinstance(repetitions, bool)
        or not isinstance(repetitions, int)
        or repetitions < 1
    ):
        raise ValueError("`repetitions` must be a positive integer.")
    if model is not None and replay_overrides is not None:
        raise ValueError(
            "Live model and offline replay overrides are mutually exclusive."
        )
    if (model is None) != (model_profile is None):
        raise ValueError(
            "A live benchmark model and model profile must be supplied together."
        )
    if bool(recovery_profiles) != (model_factory_registry is not None):
        raise ValueError(
            "Recovery profiles and a model factory registry must be supplied together."
        )
    selected = tuple(
        case
        for case in cases
        if selected_case_ids is None or case.case_id in selected_case_ids
    )
    if selected_case_ids is not None:
        missing = selected_case_ids.difference(case.case_id for case in selected)
        if missing:
            raise ValueError(f"Unknown benchmark case IDs: {sorted(missing)}.")
    scores: list[CaseScore] = []
    for repetition in range(1, repetitions + 1):
        for case in selected:
            case_model: PlanningModel
            if model is None:
                override = (
                    None
                    if replay_overrides is None
                    else replay_overrides.get(case.case_id)
                )
                case_model = ScriptedPlanningModel(
                    _response_from_override(case, override)
                )
            else:
                case_model = model
            observed = RecordingPlanningModel(case_model)
            observed_factories = (
                None
                if model_factory_registry is None
                else _RecordingModelFactoryResolver(model_factory_registry)
            )
            registry, guard = guarded_registry()
            request = AgentRequest(
                request_id=f"benchmark-{case.case_id}-{repetition}",
                prompt=case.prompt,
                inputs=case.inputs,
                mode=RunMode.PLAN_ONLY,
            )
            result = AgentRuntime(
                planner=LLMPlanner(
                    observed,
                    profile=model_profile,
                    recovery_profiles=recovery_profiles,
                    model_factory_registry=observed_factories,
                ),
                registry=registry,
            ).run(request)
            secondary_responses = (
                []
                if observed_factories is None
                else [
                    response
                    for _, secondary in observed_factories.created
                    for response in secondary.responses
                ]
            )
            raw_response = (
                secondary_responses[-1]
                if secondary_responses
                else observed.responses[-1]
                if observed.responses
                else None
            )
            scores.append(
                _score_case(
                    case,
                    result=result,
                    raw_response=raw_response,
                    provider_calls=observed.calls,
                    scientific_calls=guard.count,
                    registry=registry,
                    repetition=repetition,
                )
            )
    score_tuple = tuple(scores)
    tags = sorted({tag for score in score_tuple for tag in score.tags})
    by_tag = {
        tag: aggregate_metrics(
            tuple(score for score in score_tuple if tag in score.tags)
        )
        for tag in tags
    }
    return BenchmarkReport(
        track="offline-replay" if model is None else "live-provider",
        profile_id=(
            ScriptedPlanningModel.model_id
            if model_profile is None
            else model_profile.profile_id
        ),
        provider_id="offline" if model_profile is None else model_profile.provider_id,
        model_id=(
            ScriptedPlanningModel.model_id
            if model_profile is None
            else model_profile.model_id
        ),
        repetitions=repetitions,
        cases=score_tuple,
        metrics=aggregate_metrics(score_tuple),
        metrics_by_tag=by_tag,
    )


__all__ = [
    "BENCHMARK_SCHEMA_VERSION",
    "BenchmarkCase",
    "BenchmarkDefinitionError",
    "BenchmarkReport",
    "CaseScore",
    "RecordingPlanningModel",
    "REPORT_SCHEMA_VERSION",
    "ScriptedPlanningModel",
    "aggregate_metrics",
    "guarded_registry",
    "load_cases",
    "load_replay_overrides",
    "oracle_response",
    "run_benchmark",
]
