"""Pure compilation from semantic source selections into an AgentPlan.

This experimental Post-M9 contract is intentionally not wired to LLMPlanner or
AgentRuntime. Its authoritative port metadata is separate from the registry's
descriptive planning metadata and describes tool interfaces, never workflows.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from typing import TypeAlias

from agent.schemas import AgentPlan, AgentRequest, PlanStep, StepOutputRef

from .registry import ArgumentSpec, ToolRegistry, UnknownToolError


class SemanticPlanCompileError(ValueError):
    """Fail-closed semantic compilation error with a stable reason code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        if not isinstance(code, str) or not code.strip():
            raise ValueError("Compiler error `code` must be a non-empty string.")
        self.code = code


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"`{name}` must be a non-empty string.")
    return value


def _unique_strings(values: object, name: str) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"`{name}` must be a tuple.")
    result = tuple(_nonempty(value, name) for value in values)
    if len(set(result)) != len(result):
        raise ValueError(f"`{name}` must not contain duplicates.")
    return result


class SemanticLineage(str, Enum):
    """Authoritatively declared source identity carried through a branch."""

    REFERENCE = "reference"
    QUERY = "query"
    GROUND_TRUTH = "ground_truth"


@dataclass(frozen=True)
class SemanticRequestInputSource:
    """Explicitly select one structured request input for a semantic port."""

    target_port: str
    input_name: str

    def __post_init__(self) -> None:
        _nonempty(self.target_port, "target_port")
        _nonempty(self.input_name, "input_name")


@dataclass(frozen=True)
class SemanticStepOutputSource:
    """Explicitly select a producer step and, when needed, its semantic port."""

    target_port: str
    step_id: str
    source_port: str | None = None

    def __post_init__(self) -> None:
        _nonempty(self.target_port, "target_port")
        _nonempty(self.step_id, "step_id")
        if self.source_port is not None:
            _nonempty(self.source_port, "source_port")


SemanticSourceSelection: TypeAlias = (
    SemanticRequestInputSource | SemanticStepOutputSource
)


@dataclass(frozen=True)
class SemanticPlanStep:
    """One selected tool with explicit semantic value sources."""

    step_id: str
    tool_name: str
    sources: tuple[SemanticSourceSelection, ...] = ()
    control_dependencies: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _nonempty(self.step_id, "step_id")
        _nonempty(self.tool_name, "tool_name")
        if not isinstance(self.sources, tuple) or not all(
            isinstance(
                source,
                (SemanticRequestInputSource, SemanticStepOutputSource),
            )
            for source in self.sources
        ):
            raise TypeError("`sources` must contain semantic source selections.")
        control = _unique_strings(
            self.control_dependencies, "control_dependencies"
        )
        if self.step_id in control:
            raise ValueError("A semantic step cannot depend on itself.")
        producer_ids = {
            source.step_id
            for source in self.sources
            if isinstance(source, SemanticStepOutputSource)
        }
        if self.step_id in producer_ids:
            raise ValueError("A semantic step cannot consume its own output.")
        if producer_ids.intersection(control):
            raise ValueError(
                "Value-producing and control-only dependencies must be distinct."
            )


@dataclass(frozen=True)
class SemanticPlanCandidate:
    """Provider-independent semantic DAG selected by a planner."""

    steps: tuple[SemanticPlanStep, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.steps, tuple):
            raise TypeError("`steps` must be a tuple.")
        if not self.steps or not all(
            isinstance(step, SemanticPlanStep) for step in self.steps
        ):
            raise ValueError("A semantic plan must contain typed steps.")
        step_ids = tuple(step.step_id for step in self.steps)
        if len(set(step_ids)) != len(step_ids):
            raise ValueError("Semantic plan step IDs must be unique.")


@dataclass(frozen=True)
class RequestInputBindingRule:
    """Authorize one exact request input for one consumer semantic port."""

    tool_name: str
    target_port: str
    argument_name: str
    input_name: str
    lineage: SemanticLineage | None = None

    def __post_init__(self) -> None:
        _nonempty(self.tool_name, "tool_name")
        _nonempty(self.target_port, "target_port")
        _nonempty(self.argument_name, "argument_name")
        _nonempty(self.input_name, "input_name")
        if self.lineage is not None and not isinstance(
            self.lineage, SemanticLineage
        ):
            raise TypeError("`lineage` must be a SemanticLineage or None.")


@dataclass(frozen=True)
class ChannelMember:
    """One mechanical result-field to argument mapping inside a channel."""

    output_key: str
    argument_name: str

    def __post_init__(self) -> None:
        _nonempty(self.output_key, "output_key")
        _nonempty(self.argument_name, "argument_name")


@dataclass(frozen=True)
class StepOutputChannelRule:
    """Authorize one semantic producer port for one consumer target port."""

    producer_tool_name: str
    source_port: str
    consumer_tool_name: str
    target_port: str
    members: tuple[ChannelMember, ...]
    producer_lineage_port: str | None = None
    required_lineage: SemanticLineage | None = None

    def __post_init__(self) -> None:
        _nonempty(self.producer_tool_name, "producer_tool_name")
        _nonempty(self.source_port, "source_port")
        _nonempty(self.consumer_tool_name, "consumer_tool_name")
        _nonempty(self.target_port, "target_port")
        if not isinstance(self.members, tuple) or not self.members or not all(
            isinstance(member, ChannelMember) for member in self.members
        ):
            raise TypeError("`members` must contain ChannelMember values.")
        output_keys = tuple(member.output_key for member in self.members)
        argument_names = tuple(member.argument_name for member in self.members)
        if len(set(output_keys)) != len(output_keys):
            raise ValueError("A grouped channel cannot repeat a result field.")
        if len(set(argument_names)) != len(argument_names):
            raise ValueError("A grouped channel cannot repeat a consumer argument.")
        if self.producer_lineage_port is not None:
            _nonempty(self.producer_lineage_port, "producer_lineage_port")
        if self.required_lineage is not None and not isinstance(
            self.required_lineage, SemanticLineage
        ):
            raise TypeError(
                "`required_lineage` must be a SemanticLineage or None."
            )
        if (
            self.required_lineage is not None
            and self.producer_lineage_port is None
        ):
            raise ValueError(
                "A lineage-constrained channel must name its producer lineage port."
            )


@dataclass(frozen=True)
class PlanningCompilerContract:
    """Explicit binding authority used only by semantic compilation."""

    request_bindings: tuple[RequestInputBindingRule, ...] = ()
    step_output_channels: tuple[StepOutputChannelRule, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.request_bindings, tuple) or not all(
            isinstance(rule, RequestInputBindingRule)
            for rule in self.request_bindings
        ):
            raise TypeError(
                "`request_bindings` must contain RequestInputBindingRule values."
            )
        if not isinstance(self.step_output_channels, tuple) or not all(
            isinstance(rule, StepOutputChannelRule)
            for rule in self.step_output_channels
        ):
            raise TypeError(
                "`step_output_channels` must contain StepOutputChannelRule values."
            )
        if len(set(self.request_bindings)) != len(self.request_bindings):
            raise ValueError("Compiler request-binding rules must be unique.")
        if len(set(self.step_output_channels)) != len(
            self.step_output_channels
        ):
            raise ValueError("Compiler step-output channels must be unique.")


def _argument_spec(
    registry: ToolRegistry, tool_name: str, argument_name: str
) -> ArgumentSpec:
    try:
        tool = registry.get(tool_name)
    except UnknownToolError as exc:
        raise ValueError(
            f"Compiler contract names unknown tool {tool_name!r}."
        ) from exc
    argument = tool.required_arguments.get(
        argument_name, tool.optional_arguments.get(argument_name)
    )
    if argument is None:
        raise ValueError(
            f"Compiler contract names unknown argument "
            f"{tool_name}.{argument_name}."
        )
    return argument


def _result_types_fit_argument(
    result_types: tuple[type, ...], argument_types: tuple[type, ...]
) -> bool:
    for result_type in result_types:
        if result_type is bool and bool not in argument_types:
            return False
        if not any(
            issubclass(result_type, accepted) for accepted in argument_types
        ):
            return False
    return True


def _validate_contract(
    registry: ToolRegistry, contract: PlanningCompilerContract
) -> None:
    target_shapes: dict[tuple[str, str], set[str]] = {}
    for rule in contract.request_bindings:
        _argument_spec(registry, rule.tool_name, rule.argument_name)
        key = (rule.tool_name, rule.target_port)
        shape = {rule.argument_name}
        existing = target_shapes.setdefault(key, shape)
        if existing != shape:
            raise ValueError(
                "Request alternatives for a semantic port must bind the same "
                "consumer argument."
            )

    for rule in contract.step_output_channels:
        try:
            producer = registry.get(rule.producer_tool_name)
        except UnknownToolError as exc:
            raise ValueError(
                f"Compiler contract names unknown producer tool "
                f"{rule.producer_tool_name!r}."
            ) from exc
        for member in rule.members:
            consumer = _argument_spec(
                registry, rule.consumer_tool_name, member.argument_name
            )
            if not consumer.allow_step_output_ref:
                raise ValueError(
                    f"Compiler channel targets non-reference argument "
                    f"{rule.consumer_tool_name}.{member.argument_name}."
                )
            result_types = producer.result_contract.required_fields.get(
                member.output_key
            )
            if result_types is None:
                raise ValueError(
                    f"Compiler contract names unknown result field "
                    f"{rule.producer_tool_name}.{member.output_key}."
                )
            if not _result_types_fit_argument(
                result_types, consumer.accepted_types
            ):
                raise ValueError(
                    "Compiler channel result types are incompatible with its "
                    "consumer argument."
                )
        key = (rule.consumer_tool_name, rule.target_port)
        shape = {member.argument_name for member in rule.members}
        existing = target_shapes.setdefault(key, shape)
        if existing != shape:
            raise ValueError(
                "Alternatives for a semantic target port must bind the same "
                "consumer arguments."
            )
    for rule in contract.step_output_channels:
        if rule.producer_lineage_port is None:
            continue
        if (rule.producer_tool_name, rule.producer_lineage_port) not in target_shapes:
            raise ValueError(
                "Compiler channel names an unknown producer lineage port "
                f"{rule.producer_tool_name}.{rule.producer_lineage_port}."
            )


def _plan_id(
    request: AgentRequest, planner_name: str, steps: tuple[PlanStep, ...]
) -> str:
    payload = {
        "request_id": request.request_id,
        "planner_name": planner_name,
        "steps": [step.to_dict() for step in steps],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"{request.request_id}:semantic:{hashlib.sha256(encoded).hexdigest()[:16]}"


def _raise_compile(code: str, message: str) -> None:
    raise SemanticPlanCompileError(code, message)


@dataclass
class _StepCompilation:
    arguments: dict[str, object] = field(default_factory=dict)
    target_ports: set[str] = field(default_factory=set)
    request_inputs: set[str] = field(default_factory=set)
    dependencies: list[str] = field(default_factory=list)
    lineages: dict[str, SemanticLineage] = field(default_factory=dict)
    channel_sources: list[tuple[str, StepOutputChannelRule]] = field(
        default_factory=list
    )


def _bind_request_value(
    *,
    request: AgentRequest,
    registry: ToolRegistry,
    step: SemanticPlanStep,
    state: _StepCompilation,
    rule: RequestInputBindingRule,
) -> None:
    if rule.argument_name in state.arguments:
        _raise_compile(
            "OVERLAPPING_SOURCE_MEMBERS",
            f"Multiple semantic sources target {step.tool_name}."
            f"{rule.argument_name}.",
        )
    argument = _argument_spec(registry, step.tool_name, rule.argument_name)
    value = request.inputs[rule.input_name]
    try:
        argument.validate(rule.argument_name, value)
    except ValueError as exc:
        raise SemanticPlanCompileError(
            "INVALID_REQUEST_BINDING",
            f"Authorized request input {rule.input_name!r} is invalid for "
            f"{step.tool_name}.{rule.argument_name}.",
        ) from exc
    state.arguments[rule.argument_name] = value
    state.target_ports.add(rule.target_port)
    state.request_inputs.add(rule.input_name)
    if rule.lineage is not None:
        state.lineages[rule.target_port] = rule.lineage


def _known_target_ports(
    contract: PlanningCompilerContract, tool_name: str
) -> set[str]:
    return {
        rule.target_port
        for rule in contract.request_bindings
        if rule.tool_name == tool_name
    }.union(
        rule.target_port
        for rule in contract.step_output_channels
        if rule.consumer_tool_name == tool_name
    )


def _select_channel(
    *,
    contract: PlanningCompilerContract,
    producer_tool_name: str,
    consumer_tool_name: str,
    source: SemanticStepOutputSource,
) -> StepOutputChannelRule:
    candidates = [
        rule
        for rule in contract.step_output_channels
        if rule.producer_tool_name == producer_tool_name
        and rule.consumer_tool_name == consumer_tool_name
        and rule.target_port == source.target_port
    ]
    if source.source_port is not None:
        selected = [
            rule for rule in candidates if rule.source_port == source.source_port
        ]
        if not selected:
            _raise_compile(
                "WRONG_SOURCE_PORT",
                f"Producer {producer_tool_name!r} does not expose authorized "
                f"source port {source.source_port!r} for target "
                f"{consumer_tool_name}.{source.target_port}.",
            )
    else:
        selected = candidates
        source_ports = {rule.source_port for rule in selected}
        if len(source_ports) > 1:
            _raise_compile(
                "AMBIGUOUS_SOURCE_PORT",
                f"Target {consumer_tool_name}.{source.target_port} accepts "
                "multiple producer semantic ports; source_port is required.",
            )
    if not selected:
        _raise_compile(
            "ZERO_VALID_CHANNELS",
            f"No authorized channel connects {producer_tool_name!r} to "
            f"{consumer_tool_name}.{source.target_port}.",
        )
    if len(selected) > 1:
        _raise_compile(
            "MULTIPLE_VALID_CHANNELS",
            f"Channel selection for {consumer_tool_name}."
            f"{source.target_port} is not unique.",
        )
    return selected[0]


def _semantic_order(
    candidate: SemanticPlanCandidate,
) -> tuple[SemanticPlanStep, ...]:
    emitted: set[str] = set()
    ordered: list[SemanticPlanStep] = []
    while len(ordered) < len(candidate.steps):
        ready = []
        for step in candidate.steps:
            if step.step_id in emitted:
                continue
            dependencies = {
                source.step_id
                for source in step.sources
                if isinstance(source, SemanticStepOutputSource)
            }.union(step.control_dependencies)
            if dependencies <= emitted:
                ready.append(step)
        if not ready:
            _raise_compile(
                "INVALID_SEMANTIC_GRAPH",
                "The semantic source graph contains a cycle.",
            )
        for step in ready:
            emitted.add(step.step_id)
            ordered.append(step)
    return tuple(ordered)


def compile_semantic_plan(
    request: AgentRequest,
    candidate: SemanticPlanCandidate,
    registry: ToolRegistry,
    contract: PlanningCompilerContract,
    *,
    planner_name: str = "semantic-compiler-contract",
) -> AgentPlan:
    """Compile explicit semantic choices without executing or inferring science."""

    if not isinstance(request, AgentRequest):
        raise TypeError("`request` must be an AgentRequest.")
    if not isinstance(candidate, SemanticPlanCandidate):
        raise TypeError("`candidate` must be a SemanticPlanCandidate.")
    if not isinstance(registry, ToolRegistry):
        raise TypeError("`registry` must be a ToolRegistry.")
    if not isinstance(contract, PlanningCompilerContract):
        raise TypeError("`contract` must be a PlanningCompilerContract.")
    _nonempty(planner_name, "planner_name")
    _validate_contract(registry, contract)

    semantic_steps = {step.step_id: step for step in candidate.steps}
    states = {step.step_id: _StepCompilation() for step in candidate.steps}
    explicit_optional_inputs: set[str] = set()

    for step in candidate.steps:
        try:
            tool = registry.get(step.tool_name)
        except UnknownToolError as exc:
            raise SemanticPlanCompileError(
                "UNKNOWN_TOOL",
                f"Semantic step selects unknown tool {step.tool_name!r}.",
            ) from exc
        state = states[step.step_id]
        known_ports = _known_target_ports(contract, step.tool_name)
        for source in step.sources:
            if source.target_port not in known_ports:
                _raise_compile(
                    "UNKNOWN_TARGET_PORT",
                    f"Tool {step.tool_name!r} has no authorized target port "
                    f"{source.target_port!r}.",
                )
            if source.target_port in state.target_ports:
                _raise_compile(
                    "DUPLICATE_TARGET_SOURCE",
                    f"Step {step.step_id!r} selects target port "
                    f"{source.target_port!r} more than once.",
                )
            if isinstance(source, SemanticRequestInputSource):
                if source.input_name not in request.inputs:
                    _raise_compile(
                        "UNKNOWN_REQUEST_INPUT",
                        f"Semantic source names unavailable request input "
                        f"{source.input_name!r}.",
                    )
                rules = [
                    rule
                    for rule in contract.request_bindings
                    if rule.tool_name == step.tool_name
                    and rule.target_port == source.target_port
                    and rule.input_name == source.input_name
                ]
                if not rules:
                    _raise_compile(
                        "UNAUTHORIZED_REQUEST_INPUT",
                        f"Request input {source.input_name!r} is not authorized "
                        f"for {step.tool_name}.{source.target_port}.",
                    )
                if len(rules) > 1:
                    _raise_compile(
                        "MULTIPLE_VALID_REQUEST_MAPPINGS",
                        f"Request source for {step.tool_name}."
                        f"{source.target_port} is not unique.",
                    )
                _bind_request_value(
                    request=request,
                    registry=registry,
                    step=step,
                    state=state,
                    rule=rules[0],
                )
                if rules[0].argument_name in tool.optional_arguments:
                    explicit_optional_inputs.add(source.input_name)
                continue

            producer = semantic_steps.get(source.step_id)
            if producer is None:
                _raise_compile(
                    "UNKNOWN_DEPENDENCY",
                    f"Semantic source names unknown producer step "
                    f"{source.step_id!r}.",
                )
            channel = _select_channel(
                contract=contract,
                producer_tool_name=producer.tool_name,
                consumer_tool_name=step.tool_name,
                source=source,
            )
            for member in channel.members:
                if member.argument_name in state.arguments:
                    _raise_compile(
                        "OVERLAPPING_SOURCE_MEMBERS",
                        f"Multiple semantic sources target {step.tool_name}."
                        f"{member.argument_name}.",
                    )
                state.arguments[member.argument_name] = StepOutputRef(
                    source.step_id, member.output_key
                )
            state.target_ports.add(source.target_port)
            state.dependencies.append(source.step_id)
            state.channel_sources.append((source.step_id, channel))

        for dependency in step.control_dependencies:
            if dependency not in semantic_steps:
                _raise_compile(
                    "UNKNOWN_DEPENDENCY",
                    f"Semantic step {step.step_id!r} names unknown control "
                    f"dependency {dependency!r}.",
                )

    request_rules_by_target: dict[
        tuple[str, str], list[RequestInputBindingRule]
    ] = defaultdict(list)
    for rule in contract.request_bindings:
        request_rules_by_target[(rule.tool_name, rule.target_port)].append(rule)

    optional_occurrences: dict[str, int] = defaultdict(int)
    for step in candidate.steps:
        tool = registry.get(step.tool_name)
        for (tool_name, _target_port), rules in request_rules_by_target.items():
            if tool_name != step.tool_name:
                continue
            available = [rule for rule in rules if rule.input_name in request.inputs]
            if len(available) != 1:
                continue
            if available[0].argument_name in tool.optional_arguments:
                optional_occurrences[available[0].input_name] += 1

    for step in candidate.steps:
        tool = registry.get(step.tool_name)
        state = states[step.step_id]
        for (tool_name, target_port), rules in request_rules_by_target.items():
            if tool_name != step.tool_name or target_port in state.target_ports:
                continue
            available = [rule for rule in rules if rule.input_name in request.inputs]
            if len(available) > 1:
                _raise_compile(
                    "AMBIGUOUS_REQUEST_INPUT",
                    f"Target {step.tool_name}.{target_port} accepts multiple "
                    "available request inputs; explicit selection is required.",
                )
            if not available:
                continue
            rule = available[0]
            selected_producer_is_available = any(
                producer.step_id != step.step_id
                and channel.producer_tool_name == producer.tool_name
                and channel.consumer_tool_name == step.tool_name
                and channel.target_port == target_port
                for producer in candidate.steps
                for channel in contract.step_output_channels
            )
            if selected_producer_is_available:
                _raise_compile(
                    "AMBIGUOUS_SOURCE_SELECTION",
                    f"Target {step.tool_name}.{target_port} can use either a "
                    "request input or a selected producer; source selection "
                    "must be explicit.",
                )
            if rule.argument_name in tool.optional_arguments:
                if optional_occurrences[rule.input_name] > 1:
                    if rule.input_name not in explicit_optional_inputs:
                        _raise_compile(
                            "AMBIGUOUS_OPTIONAL_INPUT_SCOPE",
                            f"Optional request input {rule.input_name!r} can apply "
                            "to multiple selected steps; scope must be explicit.",
                        )
                    continue
            _bind_request_value(
                request=request,
                registry=registry,
                step=step,
                state=state,
                rule=rule,
            )


    selected_argument_names = {
        argument_name
        for step in candidate.steps
        for argument_name in (
            *registry.get(step.tool_name).required_arguments,
            *registry.get(step.tool_name).optional_arguments,
        )
    }
    bound_request_inputs = {
        input_name
        for state in states.values()
        for input_name in state.request_inputs
    }
    unauthorized = sorted(
        input_name
        for input_name in request.inputs
        if input_name in selected_argument_names
        and input_name not in bound_request_inputs
    )
    if unauthorized:
        _raise_compile(
            "UNAUTHORIZED_REQUEST_INPUT",
            "Selected tools have no explicit compiler use for matching request "
            f"inputs {unauthorized}.",
        )

    for step in candidate.steps:
        tool = registry.get(step.tool_name)
        state = states[step.step_id]
        missing = sorted(set(tool.required_arguments).difference(state.arguments))
        if missing:
            port_arguments: dict[str, set[str]] = defaultdict(set)
            for rule in contract.request_bindings:
                if rule.tool_name == step.tool_name:
                    port_arguments[rule.target_port].add(rule.argument_name)
            for channel in contract.step_output_channels:
                if channel.consumer_tool_name == step.tool_name:
                    port_arguments[channel.target_port].update(
                        member.argument_name for member in channel.members
                    )
            missing_ports = sorted(
                port
                for port, arguments in port_arguments.items()
                if arguments.intersection(missing)
                and port not in state.target_ports
            )
            code = (
                "MISSING_REQUIRED_SOURCE"
                if missing_ports
                else "MISSING_REQUIRED_BINDING"
            )
            _raise_compile(
                code,
                f"Semantic step {step.step_id!r} has no authorized binding for "
                f"required arguments {missing}; missing source ports "
                f"{missing_ports}.",
            )
        try:
            registry.validate_arguments(step.tool_name, state.arguments)
        except ValueError as exc:
            raise SemanticPlanCompileError(
                "INVALID_COMPILED_ARGUMENTS",
                f"Compiled arguments are invalid for {step.tool_name!r}.",
            ) from exc

    for step in _semantic_order(candidate):
        state = states[step.step_id]
        for producer_step_id, channel in state.channel_sources:
            lineage = None
            if channel.producer_lineage_port is not None:
                lineage = states[producer_step_id].lineages.get(
                    channel.producer_lineage_port
                )
            if (
                channel.required_lineage is not None
                and lineage is not channel.required_lineage
            ):
                _raise_compile(
                    "BROKEN_BRANCH_LINEAGE",
                    f"Source for {step.tool_name}.{channel.target_port} does not "
                    f"carry required {channel.required_lineage.value!r} lineage.",
                )
            if lineage is not None:
                existing = state.lineages.get(channel.target_port)
                if existing is not None and existing is not lineage:
                    _raise_compile(
                        "CONFLICTING_BRANCH_LINEAGE",
                        f"Target {step.tool_name}.{channel.target_port} receives "
                        "conflicting branch lineage.",
                    )
                state.lineages[channel.target_port] = lineage

    compiled_steps = tuple(
        PlanStep(
            step_id=step.step_id,
            tool_name=step.tool_name,
            arguments=states[step.step_id].arguments,
            depends_on=tuple(
                dict.fromkeys(
                    (
                        *states[step.step_id].dependencies,
                        *step.control_dependencies,
                    )
                )
            ),
        )
        for step in candidate.steps
    )
    try:
        return AgentPlan(
            plan_id=_plan_id(request, planner_name, compiled_steps),
            request_id=request.request_id,
            planner_name=planner_name,
            steps=compiled_steps,
        )
    except ValueError as exc:
        raise SemanticPlanCompileError(
            "INVALID_SEMANTIC_GRAPH",
            "The semantic candidate does not form a valid executable DAG.",
        ) from exc


def _request_rule(
    tool: str,
    argument: str,
    *,
    target: str | None = None,
    input_name: str | None = None,
    lineage: SemanticLineage | None = None,
) -> RequestInputBindingRule:
    return RequestInputBindingRule(
        tool,
        target or argument,
        argument,
        input_name or argument,
        lineage,
    )


def _member(output_key: str, argument_name: str | None = None) -> ChannelMember:
    return ChannelMember(output_key, argument_name or output_key)


def build_m92_semantic_compiler_contract() -> PlanningCompilerContract:
    """Return only authority proven by the M9.1 and M9.2 fixtures."""

    inspect = "inspect_scATAC"
    embed = "epizoo_embed_cells"
    neighbors = "build_cell_neighbors"
    transfer = "transfer_cell_labels"
    feature = "validate_scATAC_feature_space"
    pseudobulk = "build_replicate_pseudobulk"
    da = "run_replicate_differential_accessibility"
    request_bindings = (
        _request_rule(inspect, "path", target="dataset", input_name="input_path"),
        _request_rule(
            inspect,
            "path",
            target="dataset",
            input_name="reference_input_path",
            lineage=SemanticLineage.REFERENCE,
        ),
        _request_rule(
            inspect,
            "path",
            target="dataset",
            input_name="query_input_path",
            lineage=SemanticLineage.QUERY,
        ),
        _request_rule(
            embed,
            "input_path",
            target="dataset",
            input_name="input_path",
        ),
        _request_rule(
            embed,
            "input_path",
            target="dataset",
            input_name="reference_input_path",
            lineage=SemanticLineage.REFERENCE,
        ),
        _request_rule(
            embed,
            "input_path",
            target="dataset",
            input_name="query_input_path",
            lineage=SemanticLineage.QUERY,
        ),
        _request_rule(embed, "output_dir"),
        _request_rule(embed, "species"),
        _request_rule(embed, "checkpoint_path", target="checkpoint"),
        _request_rule(embed, "overwrite"),
        _request_rule(neighbors, "output_dir"),
        _request_rule(
            transfer,
            "reference_h5ad_path",
            target="reference_dataset",
            input_name="reference_input_path",
            lineage=SemanticLineage.REFERENCE,
        ),
        _request_rule(
            transfer,
            "query_h5ad_path",
            target="query_dataset",
            input_name="query_input_path",
            lineage=SemanticLineage.QUERY,
        ),
        _request_rule(transfer, "reference_label_key"),
        _request_rule(transfer, "output_dir"),
        _request_rule(transfer, "n_neighbors"),
        _request_rule(transfer, "metric"),
        _request_rule(transfer, "min_confidence"),
        _request_rule(transfer, "overwrite"),
        *(
            _request_rule(feature, argument)
            for argument in (
                "input_path",
                "output_dir",
                "matrix_source",
                "matrix_semantics",
                "species",
                "genome_assembly",
                "coordinate_source",
                "overwrite",
            )
        ),
        *(
            _request_rule(pseudobulk, argument)
            for argument in (
                "replicate_key",
                "group_key",
                "condition_key",
                "output_dir",
                "group_source",
                "covariate_keys",
                "overwrite",
            )
        ),
        *(
            _request_rule(da, argument)
            for argument in (
                "group_value",
                "condition_key",
                "numerator_condition",
                "denominator_condition",
                "design_type",
                "output_dir",
                "covariates",
                "overwrite",
            )
        ),
    )
    channels = (
        StepOutputChannelRule(
            inspect,
            "dataset",
            embed,
            "dataset",
            (_member("input_path"),),
            producer_lineage_port="dataset",
        ),
        StepOutputChannelRule(
            embed,
            "embedding",
            neighbors,
            "embedding",
            (_member("embedding_path"), _member("cell_ids_path")),
            producer_lineage_port="dataset",
        ),
        StepOutputChannelRule(
            inspect,
            "dataset",
            transfer,
            "reference_dataset",
            (_member("input_path", "reference_h5ad_path"),),
            producer_lineage_port="dataset",
            required_lineage=SemanticLineage.REFERENCE,
        ),
        StepOutputChannelRule(
            inspect,
            "dataset",
            transfer,
            "query_dataset",
            (_member("input_path", "query_h5ad_path"),),
            producer_lineage_port="dataset",
            required_lineage=SemanticLineage.QUERY,
        ),
        StepOutputChannelRule(
            embed,
            "embedding",
            transfer,
            "reference_embedding",
            (
                _member("embedding_path", "reference_embedding_path"),
                _member("cell_ids_path", "reference_cell_ids_path"),
                _member("species", "reference_species"),
                _member("checkpoint_path", "reference_checkpoint_path"),
            ),
            producer_lineage_port="dataset",
            required_lineage=SemanticLineage.REFERENCE,
        ),
        StepOutputChannelRule(
            embed,
            "embedding",
            transfer,
            "query_embedding",
            (
                _member("embedding_path", "query_embedding_path"),
                _member("cell_ids_path", "query_cell_ids_path"),
                _member("species", "query_species"),
                _member("checkpoint_path", "query_checkpoint_path"),
            ),
            producer_lineage_port="dataset",
            required_lineage=SemanticLineage.QUERY,
        ),
        StepOutputChannelRule(
            feature,
            "feature_space",
            pseudobulk,
            "feature_space",
            (_member("feature_space_path"),),
        ),
        StepOutputChannelRule(
            pseudobulk,
            "pseudobulk",
            da,
            "pseudobulk",
            (_member("pseudobulk_path"),),
        ),
    )
    return PlanningCompilerContract(request_bindings, channels)


__all__ = [
    "ChannelMember",
    "PlanningCompilerContract",
    "RequestInputBindingRule",
    "SemanticLineage",
    "SemanticPlanCandidate",
    "SemanticPlanCompileError",
    "SemanticPlanStep",
    "SemanticRequestInputSource",
    "SemanticSourceSelection",
    "SemanticStepOutputSource",
    "StepOutputChannelRule",
    "build_m92_semantic_compiler_contract",
    "compile_semantic_plan",
]
