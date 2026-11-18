"""Pure compilation from semantic source selections into an AgentPlan.

LLMPlanner uses this compiler for semantic wire v4 before runtime preflight.
Authoritative registry port metadata is separate from descriptive planning
metadata and describes tool interfaces, never workflows.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
import hashlib
import json
from pathlib import Path
from typing import Mapping, TypeAlias
from enum import Enum
from agent.schemas.orchestration import freeze_json_mapping, _serialize, _freeze_plan_arguments

from agent.schemas import AgentPlan, AgentRequest, PlanStep, StepOutputRef

from .registry import (
    ArgumentSpec,
    SemanticLineage,
    SemanticProducerPortSpec,
    ToolRegistry,
    UnknownToolError,
    build_default_tool_registry,
    _semantic_result_fits_argument as _result_types_fit_argument,
)


class SemanticPlanCompileError(ValueError):
    """Fail-closed semantic compilation error with a stable reason code."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        diagnostic_fields: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        if not isinstance(code, str) or not code.strip():
            raise ValueError("Compiler error `code` must be a non-empty string.")
        self.code = code
        self.diagnostic_fields = dict(diagnostic_fields or {})


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


@dataclass(frozen=True)
class SemanticRegisteredChoiceSource:
    """Planner selection from an explicitly enabled Registry scalar vocabulary."""
    target_port: str
    choice: str

    def __post_init__(self):
        _nonempty(self.target_port, 'target_port')
        _nonempty(self.choice, 'choice')


@dataclass(frozen=True)
class SemanticContextSource:
    target_port: str
    handle: str

    def __post_init__(self):
        _nonempty(self.target_port, 'target_port')
        _nonempty(self.handle, 'handle')


SemanticSourceSelection: TypeAlias = (
    SemanticRequestInputSource | SemanticStepOutputSource | SemanticContextSource | SemanticRegisteredChoiceSource
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
                (SemanticRequestInputSource, SemanticStepOutputSource, SemanticContextSource, SemanticRegisteredChoiceSource),
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


class AnalysisDisposition(str, Enum):
    COMPLETE = 'COMPLETE'  # Compiler completeness, never application READY.
    NEEDS_INPUT = 'NEEDS_INPUT'
    INVALID = 'INVALID'


from .scientific_inputs import ScientificInputRequirement, ScientificInputRequired


@dataclass(frozen=True)
class ScientificRequirement:
    candidate_id: str
    step_id: str
    tool_name: str
    native_function: str | None
    port: str
    argument: str
    kind: str
    member: str | None = None
    dependencies: tuple[str, ...] = ()
    choices: tuple = ()
    source_dependencies: tuple[str, ...] = ()
    source_steps: tuple[str, ...] = ()

    @property
    def requirement_id(self):
        identity = [self.candidate_id, self.step_id, self.tool_name,
                    self.native_function, self.port, self.argument, self.member, self.kind]
        return hashlib.sha256(json.dumps(identity, separators=(',', ':')).encode()).hexdigest()

    def to_dict(self):
        from dataclasses import asdict
        return dict(asdict(self), requirement_id=self.requirement_id)


@dataclass(frozen=True)
class AnalyzedOperation:
    """Bound symbolic operands, without executable-plan authority."""
    step_id: str
    tool_name: str
    arguments: Mapping
    dependencies: tuple[str, ...]

    def __post_init__(self):
        object.__setattr__(self, 'arguments', _freeze_plan_arguments(self.arguments))


@dataclass(frozen=True)
class CandidateAnalysis:
    candidate: SemanticPlanCandidate
    disposition: AnalysisDisposition
    operations: tuple[AnalyzedOperation, ...] = ()
    requirements: tuple[ScientificRequirement, ...] = ()
    blocked_checks: tuple[tuple[str, tuple[str, ...]], ...] = ()
    error: SemanticPlanCompileError | None = None

    @property
    def candidate_id(self):
        return candidate_identity(self.candidate)


def candidate_record(candidate):
    """One normalized, provider-independent graph representation."""
    from dataclasses import asdict
    return dict(version=1, steps=[dict(step_id=step.step_id, tool_name=step.tool_name,
        sources=[dict(kind=type(source).__name__, **asdict(source)) for source in step.sources],
        control_dependencies=list(step.control_dependencies)) for step in candidate.steps])


def candidate_from_record(record):
    if set(record) != {'version', 'steps'} or type(record['version']) is not int or record['version'] != 1:
        raise ValueError('Unsupported semantic candidate record.')
    if any(set(row) != {'step_id','tool_name','sources','control_dependencies'} for row in record['steps']):
        raise ValueError('Invalid canonical candidate step.')
    kinds = {kind.__name__: kind for kind in (SemanticRequestInputSource,
        SemanticStepOutputSource, SemanticContextSource, SemanticRegisteredChoiceSource)}
    return SemanticPlanCandidate(tuple(SemanticPlanStep(row['step_id'], row['tool_name'],
        tuple(kinds[source['kind']](**{k: v for k, v in source.items() if k != 'kind'})
              for source in row['sources']), tuple(row['control_dependencies'])) for row in record['steps']))


def candidate_identity(candidate):
    return hashlib.sha256(json.dumps(candidate_record(candidate), sort_keys=True,
        separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


@dataclass(frozen=True)
class RequestInputBindingRule:
    """Authorize one exact request input for one consumer semantic port."""

    tool_name: str
    target_port: str
    argument_name: str
    input_name: str
    lineage: SemanticLineage | None = None
    source_name: str | None = None
    required_lineage: SemanticLineage | None = None

    def __post_init__(self) -> None:
        _nonempty(self.tool_name, "tool_name")
        _nonempty(self.target_port, "target_port")
        _nonempty(self.argument_name, "argument_name")
        _nonempty(self.input_name, "input_name")
        if self.source_name is not None:
            _nonempty(self.source_name, "source_name")
        if self.lineage is not None and not isinstance(
            self.lineage, SemanticLineage
        ):
            raise TypeError("`lineage` must be a SemanticLineage or None.")
        if self.required_lineage is not None and not isinstance(
            self.required_lineage, SemanticLineage
        ):
            raise TypeError(
                "`required_lineage` must be a SemanticLineage or None."
            )

    @property
    def selector(self) -> str:
        return self.input_name if self.source_name is None else self.source_name


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


def _validate_contract(
    registry: ToolRegistry, contract: PlanningCompilerContract
) -> None:
    target_shapes: dict[tuple[str, str], set[str]] = {}
    request_groups: dict[
        tuple[str, str, str], list[RequestInputBindingRule]
    ] = defaultdict(list)
    for rule in contract.request_bindings:
        _argument_spec(registry, rule.tool_name, rule.argument_name)
        request_groups[
            (rule.tool_name, rule.target_port, rule.selector)
        ].append(rule)

    for (tool_name, target_port, selector), rules in request_groups.items():
        argument_names = [rule.argument_name for rule in rules]
        input_names = [rule.input_name for rule in rules]
        if len(set(argument_names)) != len(argument_names):
            raise ValueError("A request-source group overlaps consumer arguments.")
        if len(set(input_names)) != len(input_names):
            raise ValueError("A request-source group repeats a request input.")
        if selector not in input_names:
            raise ValueError(
                "A request-source selector must name one grouped request input."
            )
        lineages = {rule.lineage for rule in rules}
        required_lineages = {rule.required_lineage for rule in rules}
        if len(lineages) != 1 or len(required_lineages) != 1:
            raise ValueError("A request-source group has conflicting lineage.")
        lineage = rules[0].lineage
        required_lineage = rules[0].required_lineage
        if required_lineage is not None and lineage is not required_lineage:
            raise ValueError("A request-source group has invalid required lineage.")
        key = (tool_name, target_port)
        shape = set(argument_names)
        existing = target_shapes.setdefault(key, shape)
        if existing != shape:
            raise SemanticPlanCompileError(
                "MISSING_REQUIRED_BINDING",
                "Request alternatives for a semantic port must bind the same "
                "consumer arguments."
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
            raise SemanticPlanCompileError(
                "MISSING_REQUIRED_BINDING",
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


def _raise_compile(
    code: str,
    message: str,
    *,
    diagnostic_fields: Mapping[str, object] | None = None,
) -> None:
    raise SemanticPlanCompileError(
        code, message, diagnostic_fields=diagnostic_fields
    )


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


def _bind_request_group(
    *,
    request: AgentRequest,
    registry: ToolRegistry,
    step: SemanticPlanStep,
    state: _StepCompilation,
    rules: tuple[RequestInputBindingRule, ...],
) -> None:
    missing = sorted(
        rule.input_name for rule in rules if rule.input_name not in request.inputs
    )
    if missing:
        _raise_compile(
            "MISSING_REQUEST_SOURCE_MEMBER",
            f"Semantic request source is missing companion inputs {missing}.",
            diagnostic_fields={
                "step_id": step.step_id,
                "tool_name": step.tool_name,
                "target_port": rules[0].target_port,
                "input_name": rules[0].selector,
            },
        )
    for rule in rules:
        if rule.argument_name in state.arguments:
            _raise_compile(
                "OVERLAPPING_SOURCE_MEMBERS",
                f"Multiple semantic sources target {step.tool_name}."
                f"{rule.argument_name}.",
                diagnostic_fields={
                    "step_id": step.step_id,
                    "tool_name": step.tool_name,
                    "target_port": rule.target_port,
                },
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
                diagnostic_fields={
                    "step_id": step.step_id,
                    "tool_name": step.tool_name,
                    "target_port": rule.target_port,
                    "input_name": rule.selector,
                },
            ) from exc
    first = rules[0]
    for rule in rules:
        state.arguments[rule.argument_name] = request.inputs[rule.input_name]
        state.request_inputs.add(rule.input_name)
    state.target_ports.add(first.target_port)
    if (
        first.required_lineage is not None
        and first.lineage is not first.required_lineage
    ):
        _raise_compile(
            "BROKEN_BRANCH_LINEAGE",
            f"Request source for {step.tool_name}.{first.target_port} does not "
            f"carry required {first.required_lineage.value!r} lineage.",
            diagnostic_fields={
                "step_id": step.step_id,
                "tool_name": step.tool_name,
                "target_port": first.target_port,
                "input_name": first.selector,
            },
        )
    if first.lineage is not None:
        state.lineages[first.target_port] = first.lineage


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
    consumer_step_id: str,
    source: SemanticStepOutputSource,
) -> StepOutputChannelRule:
    diagnostic_fields = {
        "step_id": consumer_step_id,
        "tool_name": consumer_tool_name,
        "target_port": source.target_port,
        "producer_step_id": source.step_id,
        "source_port": source.source_port,
    }
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
                diagnostic_fields=diagnostic_fields,
            )
    else:
        selected = candidates
        source_ports = {rule.source_port for rule in selected}
        if len(source_ports) > 1:
            _raise_compile(
                "AMBIGUOUS_SOURCE_PORT",
                f"Target {consumer_tool_name}.{source.target_port} accepts "
                "multiple producer semantic ports; source_port is required.",
                diagnostic_fields=diagnostic_fields,
            )
    if not selected:
        _raise_compile(
            "ZERO_VALID_CHANNELS",
            f"No authorized channel connects {producer_tool_name!r} to "
            f"{consumer_tool_name}.{source.target_port}.",
            diagnostic_fields=diagnostic_fields,
        )
    if len(selected) > 1:
        _raise_compile(
            "MULTIPLE_VALID_CHANNELS",
            f"Channel selection for {consumer_tool_name}."
            f"{source.target_port} is not unique.",
            diagnostic_fields=diagnostic_fields,
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


def _analyze_candidate(
    request: AgentRequest,
    candidate: SemanticPlanCandidate,
    registry: ToolRegistry,
    contract: PlanningCompilerContract,
    *,
    planner_name: str = "semantic-compiler-contract",
    bindings: Mapping[str, Mapping[str, object]] | None = None,
    prepare_inputs=None,
) -> CandidateAnalysis:
    """The shared source, graph, operand and requirement passes."""
    from .native_parameters import request_source_applies

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

    from .active_context import current_context
    context = current_context()
    settings = () if context is None else context.task.get('parameter_overrides', ())
    canonical_candidate = candidate
    original_inputs = request.inputs
    owned_inputs = {row['input_name'] for row in settings}
    if settings:
        # Structured values already have researcher authority. The model's
        # source spelling cannot remove, change or duplicate their binding.
        request = replace(request, inputs={k: v for k, v in request.inputs.items() if k not in owned_inputs})
        candidate = replace(candidate, steps=tuple(replace(step, sources=tuple(source for source in step.sources
            if not isinstance(source, SemanticRequestInputSource) or source.input_name not in owned_inputs))
            for step in candidate.steps))
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
                    diagnostic_fields={
                        "step_id": step.step_id,
                        "tool_name": step.tool_name,
                        "target_port": source.target_port,
                    },
                )
            if source.target_port in state.target_ports:
                _raise_compile(
                    "DUPLICATE_TARGET_SOURCE",
                    f"Step {step.step_id!r} selects target port "
                    f"{source.target_port!r} more than once.",
                    diagnostic_fields={
                        "step_id": step.step_id,
                        "tool_name": step.tool_name,
                        "target_port": source.target_port,
                    },
                )
            if isinstance(source, SemanticRegisteredChoiceSource):
                port = next(p for p in tool.semantic_planning.consumer_ports if p.name == source.target_port)
                argument = (dict(tool.required_arguments) | dict(tool.optional_arguments)).get(port.members[0].field_name)
                if (len(port.members) != 1 or argument is None or argument.planning is None
                        or not argument.planning.planner_choice or source.choice not in argument.choices):
                    raise SemanticPlanCompileError('UNAUTHORIZED_REGISTERED_CHOICE', 'Choice is not enabled by the Registry.')
                function = source.choice if port.members[0].field_name == 'function_name' else state.arguments.get('function_name')
                if any(m.input_name in request.inputs for r in port.request_sources
                       if request_source_applies(tool, r.selector, function) for m in r.members):
                    raise SemanticPlanCompileError('EXPLICIT_PARAMETER_CONFLICT', 'Use the supplied parameter source instead of choosing a value.')
                argument.validate(port.members[0].field_name, source.choice)
                state.arguments[port.members[0].field_name] = source.choice
                state.target_ports.add(source.target_port)
                continue
            if isinstance(source, SemanticContextSource):
                from .active_context import current_context
                from .prior_outputs import channel_for
                from agent.schemas.prior_output import PriorOutputRef
                context = current_context()
                try:
                    if context is None:
                        raise ValueError('No active context was offered.')
                    binding = context.binding(source.handle)
                    channel = channel_for(binding, step.tool_name, source.target_port, registry)
                except ValueError as exc:
                    raise SemanticPlanCompileError('PRIOR_OUTPUT_INVALID',
                        'Unknown or incompatible historical source.') from exc
                for member in channel.members:
                    if member.argument_name in state.arguments:
                        raise SemanticPlanCompileError('OVERLAPPING_SOURCE_MEMBERS', 'Overlapping historical source.')
                    state.arguments[member.argument_name] = PriorOutputRef(binding, member.output_key)
                state.target_ports.add(source.target_port)
                continue
            if isinstance(source, SemanticRequestInputSource):
                if source.input_name not in request.inputs:
                    _raise_compile(
                        "UNKNOWN_REQUEST_INPUT",
                        f"Semantic source names unavailable request input "
                        f"{source.input_name!r}.",
                        diagnostic_fields={
                            "step_id": step.step_id,
                            "tool_name": step.tool_name,
                            "target_port": source.target_port,
                            "input_name": source.input_name,
                        },
                    )
                rules = [
                    rule
                    for rule in contract.request_bindings
                    if rule.tool_name == step.tool_name
                    and rule.target_port == source.target_port
                    and rule.selector == source.input_name
                ]
                if not rules:
                    _raise_compile(
                        "UNAUTHORIZED_REQUEST_INPUT",
                        f"Request input {source.input_name!r} is not authorized "
                        f"for {step.tool_name}.{source.target_port}.",
                        diagnostic_fields={
                            "step_id": step.step_id,
                            "tool_name": step.tool_name,
                            "target_port": source.target_port,
                            "input_name": source.input_name,
                        },
                    )
                _bind_request_group(
                    request=request,
                    registry=registry,
                    step=step,
                    state=state,
                    rules=tuple(rules),
                )
                if all(rule.argument_name in tool.optional_arguments or (
                        tool.required_arguments[rule.argument_name].planning is not None
                        and tool.required_arguments[rule.argument_name].planning.scientific_parameter)
                        for rule in rules):
                    # Explicitly scoped required scientific declarations (e.g.
                    # EpiZoo species) also disambiguate optional consumers.
                    explicit_optional_inputs.add(source.input_name)
                continue

            producer = semantic_steps.get(source.step_id)
            if producer is None:
                _raise_compile(
                    "UNKNOWN_DEPENDENCY",
                    f"Semantic source names unknown producer step "
                    f"{source.step_id!r}.",
                    diagnostic_fields={
                        "step_id": step.step_id,
                        "tool_name": step.tool_name,
                        "target_port": source.target_port,
                        "producer_step_id": source.step_id,
                        "source_port": source.source_port,
                    },
                )
            channel = _select_channel(
                contract=contract,
                producer_tool_name=producer.tool_name,
                consumer_tool_name=step.tool_name,
                consumer_step_id=step.step_id,
                source=source,
            )
            for member in channel.members:
                if member.argument_name in state.arguments:
                    _raise_compile(
                        "OVERLAPPING_SOURCE_MEMBERS",
                        f"Multiple semantic sources target {step.tool_name}."
                        f"{member.argument_name}.",
                        diagnostic_fields={
                            "step_id": step.step_id,
                            "tool_name": step.tool_name,
                            "target_port": source.target_port,
                            "producer_step_id": source.step_id,
                            "source_port": source.source_port,
                        },
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
                    diagnostic_fields={
                        "step_id": step.step_id,
                        "tool_name": step.tool_name,
                        "producer_step_id": dependency,
                    },
                )

    request_rules_by_target: dict[
        tuple[str, str], dict[str, list[RequestInputBindingRule]]
    ] = defaultdict(lambda: defaultdict(list))
    for rule in contract.request_bindings:
        request_rules_by_target[(rule.tool_name, rule.target_port)][
            rule.selector
        ].append(rule)

    optional_occurrences: dict[str, int] = defaultdict(int)
    for step in candidate.steps:
        tool = registry.get(step.tool_name)
        for (tool_name, _target_port), grouped in request_rules_by_target.items():
            if tool_name != step.tool_name:
                continue
            available = [
                (selector, rules)
                for selector, rules in grouped.items()
                if all(rule.input_name in request.inputs for rule in rules)
                and request_source_applies(tool, selector, states[step.step_id].arguments.get('function_name'))
            ]
            if len(available) != 1:
                continue
            selector, rules = available[0]
            if all(rule.argument_name in tool.optional_arguments for rule in rules):
                optional_occurrences[selector] += 1

    for step in candidate.steps:
        tool = registry.get(step.tool_name)
        state = states[step.step_id]
        for (tool_name, target_port), grouped in request_rules_by_target.items():
            if tool_name != step.tool_name or target_port in state.target_ports:
                continue
            partial = [
                (selector, rules)
                for selector, rules in grouped.items()
                if selector in request.inputs
                and not all(rule.input_name in request.inputs for rule in rules)
                and request_source_applies(tool, selector, state.arguments.get('function_name'))
            ]
            if partial:
                missing = sorted(
                    rule.input_name
                    for _selector, rules in partial
                    for rule in rules
                    if rule.input_name not in request.inputs
                )
                _raise_compile(
                    "MISSING_REQUEST_SOURCE_MEMBER",
                    f"Semantic request source is missing companion inputs {missing}.",
                    diagnostic_fields={
                        "step_id": step.step_id,
                        "tool_name": step.tool_name,
                        "target_port": target_port,
                        "input_name": partial[0][0],
                    },
                )
            available = [
                (selector, rules)
                for selector, rules in grouped.items()
                if all(rule.input_name in request.inputs for rule in rules)
                and request_source_applies(tool, selector, state.arguments.get('function_name'))
            ]
            if len(available) > 1:
                _raise_compile(
                    "AMBIGUOUS_REQUEST_INPUT",
                    f"Target {step.tool_name}.{target_port} accepts multiple "
                    "available request inputs; explicit selection is required.",
                    diagnostic_fields={
                        "step_id": step.step_id,
                        "tool_name": step.tool_name,
                        "target_port": target_port,
                    },
                )
            if not available:
                continue
            selector, rules = available[0]
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
                    diagnostic_fields={
                        "step_id": step.step_id,
                        "tool_name": step.tool_name,
                        "target_port": target_port,
                    },
                )
            if all(rule.argument_name in tool.optional_arguments for rule in rules):
                if optional_occurrences[selector] > 1:
                    if selector not in explicit_optional_inputs:
                        _raise_compile(
                            "AMBIGUOUS_OPTIONAL_INPUT_SCOPE",
                            f"Optional request input {selector!r} can apply "
                            "to multiple selected steps; scope must be explicit.",
                            diagnostic_fields={
                                "step_id": step.step_id,
                                "tool_name": step.tool_name,
                                "target_port": target_port,
                                "input_name": selector,
                            },
                        )
                    continue
            _bind_request_group(
                request=request,
                registry=registry,
                step=step,
                state=state,
                rules=tuple(rules),
            )

        for name in state.request_inputs:
            if not request_source_applies(tool, name, state.arguments.get('function_name')):
                _raise_compile('UNAUTHORIZED_REQUEST_INPUT',
                    'Function-qualified request input belongs to another selected function.',
                    diagnostic_fields={'step_id': step.step_id, 'tool_name': step.tool_name,
                                       'input_name': name})

    blocked_lineages = []
    for step in _semantic_order(candidate):
        state = states[step.step_id]
        for producer_step_id, channel in state.channel_sources:
            lineage = None
            if channel.producer_lineage_port is not None:
                lineage = states[producer_step_id].lineages.get(
                    channel.producer_lineage_port
                )
            if lineage is None and channel.producer_lineage_port is not None:
                producer_state = states[producer_step_id]
                dependencies = [(origin, port) for consumer, target, origin, port in blocked_lineages
                                if consumer == producer_step_id and target == channel.producer_lineage_port]
                if dependencies or channel.producer_lineage_port not in producer_state.target_ports:
                    blocked_lineages.extend((step.step_id, channel.target_port, origin, port) for origin, port in
                        (dependencies or [(producer_step_id, channel.producer_lineage_port)]))
                    continue
            if (
                channel.required_lineage is not None
                and lineage is not channel.required_lineage
            ):
                _raise_compile(
                    "BROKEN_BRANCH_LINEAGE",
                    f"Source for {step.tool_name}.{channel.target_port} does not "
                    f"carry required {channel.required_lineage.value!r} lineage.",
                    diagnostic_fields={
                        "step_id": step.step_id,
                        "tool_name": step.tool_name,
                        "target_port": channel.target_port,
                        "producer_step_id": producer_step_id,
                        "source_port": channel.source_port,
                    },
                )
            if lineage is not None:
                existing = state.lineages.get(channel.target_port)
                if existing is not None and existing is not lineage:
                    _raise_compile(
                        "CONFLICTING_BRANCH_LINEAGE",
                        f"Target {step.tool_name}.{channel.target_port} receives "
                        "conflicting branch lineage.",
                        diagnostic_fields={
                            "step_id": step.step_id,
                            "tool_name": step.tool_name,
                            "target_port": channel.target_port,
                            "producer_step_id": producer_step_id,
                            "source_port": channel.source_port,
                        },
                    )
                state.lineages[channel.target_port] = lineage

    if bindings:
        for step_id, values in bindings.items():
            if step_id not in states:
                raise SemanticPlanCompileError('UNKNOWN_DEPENDENCY', 'Binding names an unknown step.')
            tool = registry.get(semantic_steps[step_id].tool_name)
            for name, value in values.items():
                argument = tool.required_arguments.get(name, tool.optional_arguments.get(name))
                if (argument is None or argument.planning is None or not argument.planning.scientific_parameter
                        or argument.planning.accepted_artifact_kinds or name in tool.application_inputs and name!='kwargs'
                        or name == 'function_name' and name in states[step_id].arguments):
                    raise SemanticPlanCompileError('INVALID_REQUEST_BINDING',
                        'Answer overlay cannot replace source identity or a selected method.')
                try:
                    argument.validate(name, value)
                except ValueError as exc:
                    raise SemanticPlanCompileError('INVALID_REQUEST_BINDING', 'Invalid typed requirement answer.',
                        diagnostic_fields=dict(step_id=step_id, tool_name=tool.name, argument_name=name)) from exc
                states[step_id].arguments[name] = value

    from .native_parameters import bind_structured_parameters
    try:
        selected, explicit_arguments, deferred_preparation = bind_structured_parameters(
            {step.step_id: (step.tool_name, states[step.step_id].arguments) for step in candidate.steps},
            settings, registry, original_inputs)
    except SemanticPlanCompileError:
        raise
    except ValueError as exc:
        raise SemanticPlanCompileError('INVALID_REQUEST_BINDING', 'Invalid explicit parameter binding.') from exc
    for step_id, (_, arguments) in selected.items():
        states[step_id].arguments = arguments
    # This policy belongs to AI semantic planning, after exact researcher binding.
    # Native Functions builds its own explicit plan and never enters this compiler.
    from agent.tools.analysis.native_clustering import ELIGIBLE
    for step in candidate.steps:
        arguments = states[step.step_id].arguments
        if (step.tool_name.removeprefix('run_'), arguments.get('function_name')) in ELIGIBLE:
            arguments.setdefault('parameter_selection', 'asw')
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
            "UNUSED_REQUEST_INPUT",
            "Selected tools have no explicit compiler use for matching request "
            f"inputs {unauthorized}.",
            diagnostic_fields={"input_name": unauthorized[0]},
        )

    requirements = []
    blocked_checks = []
    candidate_id = candidate_identity(canonical_candidate)
    # Supplied values and function output availability are independently checkable
    # even if another operation has absent scalars or owner-managed inputs.
    for step in candidate.steps:
        try:
            registry.validate_arguments(step.tool_name, states[step.step_id].arguments, require_complete=False)
        except ValueError as exc:
            raise SemanticPlanCompileError('INVALID_COMPILED_ARGUMENTS', 'Invalid supplied arguments.',
                diagnostic_fields=dict(step_id=step.step_id, tool_name=step.tool_name)) from exc
        tool = registry.get(step.tool_name)
        arguments = states[step.step_id].arguments
        if tool.native_functions is not None and arguments.get('function_name') is not None:
            functions = dict(tool.native_functions.functions)
            function = arguments['function_name']
            kwargs = arguments.get('kwargs', {})
            if function not in functions or not isinstance(kwargs, Mapping) or set(kwargs) - set(functions[function]):
                raise SemanticPlanCompileError('INVALID_COMPILED_ARGUMENTS', 'Unregistered native function or keyword.',
                    diagnostic_fields=dict(step_id=step.step_id, tool_name=step.tool_name))
        for producer_id, channel in states[step.step_id].channel_sources:
            producer = semantic_steps[producer_id]
            catalog = registry.get(producer.tool_name).native_functions
            arguments = states[producer_id].arguments
            if catalog and catalog.output_available and arguments.get('function_name') is not None:
                if any(not catalog.output_available(arguments, member.output_key) for member in channel.members):
                    raise SemanticPlanCompileError('WRONG_SOURCE_PORT', 'The selected native function cannot publish this source.',
                        diagnostic_fields=dict(step_id=step.step_id, tool_name=step.tool_name,
                            producer_step_id=producer_id, source_port=channel.source_port, target_port=channel.target_port))
    # Semantic types identify objects; scientific owners describe their contents.
    # This pass is pure even for future outputs and precedes all materialization.
    from .scientific_inputs import OperationPreparationContext, ScientificDiagnostic, validate_input_properties
    selected={step.step_id:(step.tool_name,dict(states[step.step_id].arguments)) for step in candidate.steps}
    for step in candidate.steps:
        tool=registry.get(step.tool_name)
        scientific_context=OperationPreparationContext(tool,step.step_id,registry,selected,{}, {},[],None,None,{},observe_existing=False)
        try:
            validate_input_properties(states[step.step_id].arguments,scientific_context)
        except ScientificDiagnostic as exc:
            raise SemanticPlanCompileError(exc.code,str(exc),diagnostic_fields=dict(step_id=step.step_id,tool_name=tool.name)) from exc
    for step in candidate.steps:
        tool = registry.get(step.tool_name)
        state = states[step.step_id]
        if prepare_inputs is not None and not deferred_preparation:
            prepared = prepare_inputs(tool, dict(state.arguments),
                {selected.step_id: (selected.tool_name, dict(states[selected.step_id].arguments))
                 for selected in candidate.steps}, step_id=step.step_id)
            changed = {key for key in set(state.arguments) | set(prepared)
                       if (key in state.arguments) != (key in prepared) or state.arguments.get(key) != prepared.get(key)}
            if not changed <= tool.application_inputs:
                raise SemanticPlanCompileError('INVALID_REQUEST_BINDING', 'Preparation changed a non-owned scientific operand.')
            for name in changed:
                value = prepared.get(name)
                if not isinstance(value, StepOutputRef):
                    continue
                producer = semantic_steps.get(value.step_id)
                channels = [ch for ch in contract.step_output_channels if producer is not None
                    and ch.producer_tool_name == producer.tool_name and ch.consumer_tool_name == step.tool_name
                    and any(m.argument_name == name and m.output_key == value.output_key for m in ch.members)]
                if not channels:
                    raise SemanticPlanCompileError('WRONG_SOURCE_PORT', 'Materialized dependency has no authoritative source channel.')
                catalog = registry.get(producer.tool_name).native_functions
                producer_args = states[producer.step_id].arguments
                if catalog and catalog.output_available and producer_args.get('function_name') is not None:
                    if not catalog.output_available(producer_args, value.output_key):
                        raise SemanticPlanCompileError('WRONG_SOURCE_PORT', 'Materialized dependency cannot be published by this function.')
            state.arguments = prepared
        if tool.native_functions is not None:
            # Check after request binding so source order cannot hide explicit
            # native function intent or kwargs. Omitted native defaults stay
            # omitted when the researcher selected the function itself.
            choices = {
                next(port.members[0].field_name for port in tool.semantic_planning.consumer_ports
                     if port.name == source.target_port): source.choice
                for source in step.sources if isinstance(source, SemanticRegisteredChoiceSource)
            }
            for name, value in choices.items():
                if name == 'function_name' or name in explicit_arguments.get(step.step_id, ()) or name not in state.arguments:
                    continue
                if 'function_name' not in choices:
                    _raise_compile('EXPLICIT_PARAMETER_CONFLICT',
                        'An explicit native function retains omitted native defaults; use supplied parameter sources.')
                kwargs = state.arguments.get('kwargs', {})
                if name in kwargs and kwargs[name] != value:
                    _raise_compile('EXPLICIT_PARAMETER_CONFLICT',
                        'A registered native choice conflicts with supplied native kwargs.')
        missing = sorted(set(tool.required_arguments).difference(state.arguments))
        for name in missing:
            argument = tool.required_arguments.get(name, tool.optional_arguments.get(name))
            port = next((p.name for p in tool.semantic_planning.consumer_ports
                         if any(m.field_name == name for m in p.members)), name) if tool.semantic_planning else name
            kind = ('application' if name in tool.application_inputs or name == 'output_dir' else
                    'scalar' if argument and argument.planning and argument.planning.scientific_parameter else 'external_input')
            requirements.append(ScientificRequirement(candidate_id, step.step_id, step.tool_name,
                state.arguments.get('function_name'), port, name, kind,
                choices=tuple(argument.choices) if argument else ()))
        try:
            registry.validate_arguments(step.tool_name, state.arguments, require_complete=False)
        except ValueError as exc:
            raise SemanticPlanCompileError('INVALID_COMPILED_ARGUMENTS',
                f'Compiled arguments are invalid for {step.tool_name!r}.',
                diagnostic_fields={'step_id': step.step_id, 'tool_name': step.tool_name}) from exc
        from .native_parameters import missing_clustering_parameters
        for condition in missing_clustering_parameters(tool, state.arguments):
            requirements.append(ScientificRequirement(candidate_id, step.step_id, step.tool_name,
                condition['native_function'], condition['argument'], condition['argument'], 'scalar', member=condition['member']))
        step_requirements = tuple(r.requirement_id for r in requirements if r.step_id == step.step_id)
        if step_requirements:
            blocked_checks.append((step.step_id + ':complete_arguments', step_requirements))
    if deferred_preparation:
        for index, row in enumerate(settings):
            from .native_parameters import parameter_matches
            matches = parameter_matches(tuple(AnalyzedOperation(step.step_id, step.tool_name,
                states[step.step_id].arguments, ()) for step in candidate.steps), row, registry,
                {} if context is None else context.task.get('parameter_sources', {}))
            if len(matches) > 1:
                requirements.append(ScientificRequirement(candidate_id, '@scope:' + str(index), row['tool'],
                    row.get('native_function'), row['argument'], row['argument'], 'scope',
                    choices=tuple(step.step_id for step in matches)))
    for consumer, target, producer, port in blocked_lineages:
        dependencies = tuple(r.requirement_id for r in requirements if r.step_id == producer and r.port == port)
        if not dependencies:
            raise SemanticPlanCompileError('BROKEN_BRANCH_LINEAGE', 'Required source lineage is unavailable.')
        blocked_checks.append((consumer + ':lineage:' + target, dependencies))
    operations = tuple(AnalyzedOperation(step.step_id, step.tool_name, states[step.step_id].arguments,
        tuple(dict.fromkeys((*states[step.step_id].dependencies, *step.control_dependencies,
            *(v.step_id for v in states[step.step_id].arguments.values() if isinstance(v, StepOutputRef))))))
        for step in candidate.steps)
    # Preparation hooks may add exact dependencies. Check these without making a plan.
    _semantic_order(SemanticPlanCandidate(tuple(SemanticPlanStep(op.step_id, op.tool_name,
        control_dependencies=op.dependencies) for op in operations)))
    from .preparation_dependencies import operation_sources, operation_dependencies
    requirements = [replace(r,
        source_dependencies=tuple(operation_sources(r.step_id, operations, registry)),
        source_steps=tuple(operation_dependencies(r.step_id, operations)))
        for r in requirements]
    return CandidateAnalysis(canonical_candidate, AnalysisDisposition.NEEDS_INPUT if requirements else AnalysisDisposition.COMPLETE,
        operations, tuple(requirements), tuple(blocked_checks))


def analyze_candidate(request, candidate, registry, contract=None, *, bindings=None, prepare_inputs=None):
    """Analyze a candidate without ever constructing an executable AgentPlan."""
    try:
        return _analyze_candidate(request, candidate, registry,
            contract or build_semantic_compiler_contract(registry), bindings=bindings, prepare_inputs=prepare_inputs)
    except SemanticPlanCompileError as exc:
        return CandidateAnalysis(candidate, AnalysisDisposition.INVALID, error=exc)


def compile_semantic_plan(request, candidate, registry, contract, *,
                          planner_name='semantic-compiler-contract', bindings=None, prepare_inputs=None):
    """Strict materialization uses exactly the same analysis as preparation."""
    if prepare_inputs is None:
        from .active_context import prepare_selected_inputs
        prepare_inputs = prepare_selected_inputs
    analysis = analyze_candidate(request, candidate, registry, contract,
                                 bindings=bindings, prepare_inputs=prepare_inputs)
    return compile_analysis(request, analysis, registry, planner_name=planner_name)


def compile_analysis(request, analysis, registry, *, planner_name='semantic-compiler-contract'):
    if analysis.error is not None:
        raise analysis.error
    if analysis.disposition is not AnalysisDisposition.COMPLETE:
        first = analysis.requirements[0]
        missing = [r.argument for r in analysis.requirements if r.step_id == first.step_id]
        raise SemanticPlanCompileError('MISSING_REQUIRED_SOURCE',
            f'Semantic step {first.step_id!r} has unresolved required arguments {missing}.',
            diagnostic_fields=dict(step_id=first.step_id, tool_name=first.tool_name,
                                   target_port=first.port, argument_names=missing))
    for operation in analysis.operations:
        registry.validate_arguments(operation.tool_name, operation.arguments)
    compiled_steps = isolate_output_directories(tuple(PlanStep(
        step_id=op.step_id, tool_name=op.tool_name, arguments=op.arguments,
        depends_on=op.dependencies) for op in analysis.operations))
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


def isolate_output_directories(steps: tuple[PlanStep, ...]) -> tuple[PlanStep, ...]:
    """Compile shared destination bindings into stable, disjoint step paths.

    Only new plans call this: persisted plans already own their exact paths.
    No directories are created and no publication/recovery policy is changed.
    A sole consumer retains its requested destination.
    """
    destinations: dict[Path, list[str]] = defaultdict(list)
    for step in steps:
        destination = step.arguments.get('output_dir')
        if isinstance(destination, (str, Path)):
            destinations[Path(destination)].append(step.step_id)
    shared = {path for path, members in destinations.items() if len(members) > 1}
    return tuple(replace(step, arguments=dict(step.arguments, output_dir=str(
        Path(step.arguments['output_dir']) / 'steps' /
        hashlib.sha256(step.step_id.encode('utf-8')).hexdigest())))
        if isinstance(step.arguments.get('output_dir'), (str, Path))
        and Path(step.arguments['output_dir']) in shared else step for step in steps)


def build_semantic_compiler_contract(
    registry: ToolRegistry,
) -> PlanningCompilerContract:
    """Derive compiler authority exclusively from registered semantic metadata."""

    if not isinstance(registry, ToolRegistry):
        raise TypeError("`registry` must be a ToolRegistry.")

    request_bindings: list[RequestInputBindingRule] = []
    channels: list[StepOutputChannelRule] = []
    producers: dict[
        str, list[tuple[str, SemanticProducerPortSpec]]
    ] = defaultdict(list)

    for tool_name in registry.names():
        tool = registry.get(tool_name)
        semantic = tool.semantic_planning
        if semantic is None:
            continue
        for producer in semantic.producer_ports:
            producers[producer.semantic_type].append((tool_name, producer))
        for consumer in semantic.consumer_ports:
            arguments = {
                member.name: member.field_name for member in consumer.members
            }
            for source in consumer.request_sources:
                for member in source.members:
                    request_bindings.append(
                        RequestInputBindingRule(
                            tool_name=tool_name,
                            target_port=consumer.name,
                            argument_name=arguments[member.name],
                            input_name=member.input_name,
                            lineage=source.lineage,
                            source_name=source.selector,
                            required_lineage=consumer.required_lineage,
                        )
                    )

    for consumer_tool_name in registry.names():
        tool = registry.get(consumer_tool_name)
        semantic = tool.semantic_planning
        if semantic is None:
            continue
        for consumer in semantic.consumer_ports:
            consumer_members = {
                member.name: member.field_name for member in consumer.members
            }
            for semantic_type in consumer.accepted_upstream_types:
                names = {local: upstream for kind, local, upstream in consumer.upstream_member_names if kind == semantic_type}
                for producer_tool_name, producer in producers[semantic_type]:
                    producer_members = {
                        member.name: member.field_name
                        for member in producer.members
                    }
                    channels.append(
                        StepOutputChannelRule(
                            producer_tool_name=producer_tool_name,
                            source_port=producer.name,
                            consumer_tool_name=consumer_tool_name,
                            target_port=consumer.name,
                            members=tuple(
                                ChannelMember(
                                    producer_members[names.get(logical_name, logical_name)],
                                    argument_name,
                                )
                                for logical_name, argument_name in (
                                    consumer_members.items()
                                )
                            ),
                            producer_lineage_port=producer.lineage_from_port,
                            required_lineage=consumer.required_lineage,
                        )
                    )

    contract = PlanningCompilerContract(tuple(request_bindings), tuple(channels))
    _validate_contract(registry, contract)
    return contract


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
    "build_semantic_compiler_contract",
    "compile_semantic_plan",
]
